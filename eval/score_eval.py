"""Checkpointed per-question RAGAS scoring for saved Phase 5 API results."""

from __future__ import annotations

import asyncio
import copy
import json
import math
import sys
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd
from langchain_core.rate_limiters import InMemoryRateLimiter
from ragas.dataset_schema import SingleTurnSample
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import context_precision, context_recall, faithfulness
from ragas.run_config import RunConfig

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.graph import GRADE_MODEL, _chat_model

RESULTS_PATH = ROOT / "eval" / "results.jsonl"
SCORES_PATH = ROOT / "eval" / "scores_detailed.jsonl"
PROGRESS_PATH = ROOT / "eval" / "scores_progress.json"
REPORT_PATH = ROOT / "eval" / "report.md"
METRICS = {
    "faithfulness": faithfulness,
    "context_precision": context_precision,
    "context_recall": context_recall,
}
RAGAS_COLUMNS = list(METRICS)
RUN_CONFIG = RunConfig(timeout=120, max_retries=0, max_wait=60, max_workers=1)


class QuotaExhausted(RuntimeError):
    """Raised after checkpointing when retries still hit provider quota."""


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write_json_atomic(path: Path, value: Any) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)


def _write_jsonl_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(
        "".join(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    temp.replace(path)


def _percentage(numerator: int, denominator: int) -> float | None:
    return 100 * numerator / denominator if denominator else None


def _mean(values: list[Any]) -> float | None:
    clean = []
    for value in values:
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(number):
            clean.append(number)
    return sum(clean) / len(clean) if clean else None


def _display(value: float | None, *, percent: bool = False) -> str:
    if value is None:
        return "N/A"
    return f"{value:.1f}%" if percent else f"{value:.3f}"


def _metric_value(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _quota_error(exc: BaseException) -> bool:
    status = getattr(exc, "status_code", None)
    message = str(exc).lower()
    return status == 429 or any(token in message for token in ("quota", "rate limit", "resource_exhausted", "429"))


def _ragas_input(result: dict[str, Any]) -> SingleTurnSample:
    contexts = [
        str(citation.get("snippet", ""))
        for citation in result.get("citations", [])
        if citation.get("snippet")
    ]
    return SingleTurnSample(
        user_input=result["question"],
        response=result["answer"],
        retrieved_contexts=contexts,
        reference=result["expected_answer"],
    )


def _load_progress(answerable: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    identities = [(int(row["id"]), row["category"]) for row in answerable]
    rows_by_id: dict[int, dict[str, Any]] = {}
    if SCORES_PATH.exists():
        for row in _read_jsonl(SCORES_PATH):
            rows_by_id[int(row["id"])] = row
        if set(rows_by_id) - {identifier for identifier, _ in identities}:
            raise ValueError("scores_detailed.jsonl contains IDs outside the answerable result set")

    scores = []
    for identifier, category in identities:
        saved = rows_by_id.get(identifier, {})
        if saved and saved.get("category") != category:
            raise ValueError(f"Saved score category mismatch for question {identifier}")
        scores.append({
            "id": identifier,
            "category": category,
            **{metric: _metric_value(saved.get(metric)) for metric in RAGAS_COLUMNS},
        })

    expected_jobs = {f"{identifier}:{metric}" for identifier, _ in identities for metric in RAGAS_COLUMNS}
    state: dict[str, Any] = {"completed": {}, "errors": {}, "expected_jobs": len(expected_jobs)}
    if PROGRESS_PATH.exists():
        state = json.loads(PROGRESS_PATH.read_text(encoding="utf-8"))
        if state.get("expected_jobs") != len(expected_jobs):
            raise ValueError("Progress file is for a different RAGAS job set")
        state.setdefault("completed", {})
        state.setdefault("errors", {})
    # Earlier checkpoints may have recorded transport failures as omissions.
    # Retry those now that all async model calls share one event loop.
    for key, error in list(state["errors"].items()):
        if str(error).startswith(("APIConnectionError:", "APITimeoutError:", "TimeoutError:")):
            state["completed"].pop(key, None)
    # A finite saved score is sufficient to recover from interruption between the
    # score checkpoint and progress-state checkpoint. Null requires state metadata.
    for row in scores:
        for metric in RAGAS_COLUMNS:
            key = f"{row['id']}:{metric}"
            if row[metric] is not None:
                state["completed"].setdefault(key, "scored")
    return scores, state


def _checkpoint(scores: list[dict[str, Any]], state: dict[str, Any]) -> None:
    _write_jsonl_atomic(SCORES_PATH, scores)
    _write_json_atomic(PROGRESS_PATH, state)


async def _score_detailed_async(
    answerable: list[dict[str, Any]], *, retry_cells: set[str] | None = None
) -> list[dict[str, Any]]:
    """Score/resume each question-metric job on one event loop."""
    scores, state = _load_progress(answerable)
    retry_targets: set[str] = set()
    if retry_cells:
        valid_keys = {f"{row['id']}:{metric}" for row in scores for metric in RAGAS_COLUMNS}
        unknown = retry_cells - valid_keys
        if unknown:
            raise ValueError(f"Retry selection contains unknown cells: {sorted(unknown)}")
        by_id = {int(row["id"]): row for row in scores}
        for key in retry_cells:
            identifier, metric = key.split(":", 1)
            row = by_id[int(identifier)]
            status = state["completed"].get(key)
            # Already-scored/frozen cells are skipped on a resumed invocation.
            if row[metric] is not None or status in {"scored", "judge unavailable"}:
                continue
            if status != "omitted":
                raise ValueError(f"Cell {key} is not an omitted score eligible for its one final retry")
            retry_targets.add(key)
        if not retry_targets:
            print("No requested cells remain eligible for a final retry.", flush=True)
        else:
            state.setdefault("retry_history", []).append({
                "keys": sorted(retry_targets), "reason": "one final retry requested"
            })
            print(f"Retrying exactly {len(retry_targets)} cells: {', '.join(sorted(retry_targets))}", flush=True)
    state.setdefault("retry_attempts", {})
    _checkpoint(scores, state)

    chat_llm = _chat_model(GRADE_MODEL).model_copy(update={
        "rate_limiter": InMemoryRateLimiter(
            requests_per_second=0.22,
            check_every_n_seconds=0.1,
            max_bucket_size=1,
        )
    })
    ragas_llm = LangchainLLMWrapper(chat_llm, run_config=RUN_CONFIG)
    metric_instances = {}
    for name, metric in METRICS.items():
        instance = copy.deepcopy(metric)
        instance.llm = ragas_llm
        instance.init(RUN_CONFIG)
        metric_instances[name] = instance

    by_id = {int(row["id"]): row for row in scores}
    total_jobs = len(answerable) * len(RAGAS_COLUMNS)
    job_ordinals = {
        f"{row['id']}:{metric}": index
        for index, (row, metric) in enumerate(
            ((row, metric) for row in answerable for metric in RAGAS_COLUMNS), start=1
        )
    }
    completed = state["completed"]
    for result in answerable:
        identifier = int(result["id"])
        for metric_name in RAGAS_COLUMNS:
            job_key = f"{identifier}:{metric_name}"
            if retry_cells and job_key in retry_cells:
                if by_id[identifier][metric_name] is not None:
                    continue
                attempts_used = int(state["retry_attempts"].get(job_key, 0))
                if state["completed"].get(job_key) == "judge unavailable":
                    continue
                if attempts_used >= 1:
                    by_id[identifier][metric_name] = "judge unavailable"
                    completed[job_key] = "judge unavailable"
                    state.setdefault("errors", {})[job_key] = "Unavailable after the one final retry"
                    _checkpoint(scores, state)
                    continue
                completed.pop(job_key, None)
                state["retry_attempts"][job_key] = attempts_used + 1
                _checkpoint(scores, state)
                completed.pop(job_key, None)
                state["retry_attempts"][job_key] = attempts_used + 1
                _checkpoint(scores, state)
            if job_key in completed:
                continue
            job_number = job_ordinals[job_key]
            transport_attempt = 0
            while True:
                try:
                    raw_score = await metric_instances[metric_name].single_turn_ascore(
                        _ragas_input(result), timeout=RUN_CONFIG.timeout
                    )
                    score = _metric_value(raw_score)
                    by_id[identifier][metric_name] = score
                    completed[job_key] = "scored" if score is not None else "omitted"
                    if score is None:
                        state.setdefault("errors", {})[job_key] = "RAGAS returned an empty/non-finite score"
                    else:
                        state.setdefault("errors", {}).pop(job_key, None)
                    _checkpoint(scores, state)
                    print(
                        f"{len(completed)}/{total_jobs} jobs done — id={identifier}, {metric_name}: "
                        f"{'OMITTED' if score is None else f'{score:.3f}'}",
                        flush=True,
                    )
                    break
                except Exception as exc:
                    if _quota_error(exc):
                        if retry_cells and job_key in retry_cells:
                            by_id[identifier][metric_name] = "judge unavailable"
                            completed[job_key] = "judge unavailable"
                            state.setdefault("retry_attempts", {}).setdefault(job_key, 1)
                            state.setdefault("errors", {})[job_key] = "Quota prevented a score on the one final retry"
                        _checkpoint(scores, state)
                        message = (
                            f"quota exhausted, stopped at job {job_number} of {total_jobs}; "
                            f"checkpointed {len(completed)}/{total_jobs} completed jobs in "
                            f"{SCORES_PATH.name} and {PROGRESS_PATH.name}."
                        )
                        print(message, file=sys.stderr, flush=True)
                        raise QuotaExhausted(message) from exc

                    if (not retry_cells or job_key not in retry_cells) and (
                        isinstance(exc, (TimeoutError, asyncio.TimeoutError)) or type(exc).__name__ in {
                        "APIConnectionError", "APITimeoutError"
                        }
                    ):
                        transport_attempt += 1
                        if transport_attempt < 3:
                            delay = 2 ** transport_attempt
                            print(
                                f"Transient transport error on job {job_number}/{total_jobs}; "
                                f"retrying in {delay}s ({transport_attempt}/2).",
                                file=sys.stderr,
                                flush=True,
                            )
                            await asyncio.sleep(delay)
                            continue

                    state.setdefault("errors", {})[job_key] = f"{type(exc).__name__}: {exc}"
                    by_id[identifier][metric_name] = None
                    completed[job_key] = "omitted"
                    _checkpoint(scores, state)
                    print(
                        f"{len(completed)}/{total_jobs} jobs done — id={identifier}, "
                        f"{metric_name}: OMITTED ({type(exc).__name__}: {exc})",
                        flush=True,
                    )
                    break

    if retry_targets:
        # Freeze unsuccessful final retries. Their string sentinel is terminal
        # and cannot be selected by later omitted-cell retries.
        for key in retry_targets:
            identifier, metric = key.split(":", 1)
            row = by_id[int(identifier)]
            if row[metric] is None:
                row[metric] = "judge unavailable"
                completed[key] = "judge unavailable"
                state.setdefault("errors", {})[key] = "Still unavailable after the one final retry"
        _checkpoint(scores, state)

    return scores


def score_detailed(
    answerable: list[dict[str, Any]], *, retry_cells: set[str] | None = None
) -> list[dict[str, Any]]:
    """Run the resumable scoring coroutine in one persistent event loop."""
    return asyncio.run(
        _score_detailed_async(answerable, retry_cells=retry_cells)
    )


def _load_detailed_scores(answerable: list[dict[str, Any]]) -> pd.DataFrame:
    if not SCORES_PATH.exists():
        raise FileNotFoundError(
            f"Missing {SCORES_PATH}; run score_eval.py to score and checkpoint detailed metrics"
        )
    rows = _read_jsonl(SCORES_PATH)
    expected_ids = [int(row["id"]) for row in answerable]
    if [int(row["id"]) for row in rows] != expected_ids:
        raise ValueError("scores_detailed.jsonl must contain exactly the 40 answerable questions in results order")
    required = {"id", "category", *RAGAS_COLUMNS}
    for row in rows:
        if set(row) != required:
            raise ValueError(f"Detailed score row {row.get('id')} has an unexpected schema")
        for metric in RAGAS_COLUMNS:
            value = row[metric]
            if value is not None and value != "judge unavailable" and _metric_value(value) is None:
                raise ValueError(f"Invalid {metric} score for question {row['id']}")
    return pd.DataFrame(rows)


def _category_rows(results: list[dict[str, Any]], category: str) -> list[dict[str, Any]]:
    return [row for row in results if row["category"] == category]


def build_report() -> str:
    results = _read_jsonl(RESULTS_PATH)
    if len(results) != 50:
        raise ValueError(f"Expected 50 saved API results; found {len(results)}")
    if any(row.get("error") or row.get("llm_error_count", 0) for row in results):
        raise RuntimeError("Saved API results contain failed requests; refusing to build the completed report")

    answerable = [row for row in results if row["category"] in {"easy", "hard"}]
    scored = _load_detailed_scores(answerable)
    abstention_rows = _category_rows(results, "abstention")
    answerable_not_abstained = sum(not bool(row["abstained"]) for row in answerable)
    abstention_correct = sum(bool(row["abstained"]) for row in abstention_rows)
    citation_correct = sum(
        row["expected_source_id"] in (row.get("cited_source_ids") or []) for row in answerable
    )

    def metric_means(category: str | None = None) -> dict[str, float | None]:
        subset = scored if category is None else scored[scored["category"] == category]
        return {
            metric: _mean(pd.to_numeric(subset[metric], errors="coerce").tolist())
            for metric in RAGAS_COLUMNS
        }

    overall_ragas = metric_means()
    easy_ragas = metric_means("easy")
    hard_ragas = metric_means("hard")
    grouped = {category: _category_rows(results, category) for category in ("easy", "hard", "abstention")}

    lines = [
        "# Phase 5 Evaluation Report",
        "",
        f"Run date: {date.today().isoformat()}  ",
        f"Results: `{RESULTS_PATH.relative_to(ROOT)}` (50 saved API answers)  ",
        f"Detailed RAGAS scores: `{SCORES_PATH.relative_to(ROOT)}` (40 answerable questions)  ",
        "RAGAS version: 0.2.3",
        "",
        "RAGAS faithfulness, context precision, and context recall use the saved API answers, expected answers, and cited snippets. Unavailable judgments are marked explicitly and excluded from means; each table reports scored denominators in Diagnostics.",
        "",
        "## Overall",
        "",
        "| Measure | Correct / n | Score |",
        "|---|---:|---:|",
        f"| Abstention accuracy (out-of-corpus) | {abstention_correct} / {len(abstention_rows)} | {_display(_percentage(abstention_correct, len(abstention_rows)), percent=True)} |",
        f"| Answerable non-abstention rate | {answerable_not_abstained} / {len(answerable)} | {_display(_percentage(answerable_not_abstained, len(answerable)), percent=True)} |",
        f"| Citation accuracy (exact expected chunk ID cited) | {citation_correct} / {len(answerable)} | {_display(_percentage(citation_correct, len(answerable)), percent=True)} |",
        f"| Faithfulness (answerable only) | — | {_display(overall_ragas['faithfulness'])} |",
        f"| Context precision (answerable only) | — | {_display(overall_ragas['context_precision'])} |",
        f"| Context recall (answerable only) | — | {_display(overall_ragas['context_recall'])} |",
        "",
        "## Category breakdown",
        "",
        "| Category | n | Correct abstention behavior | Citation accuracy | Faithfulness | Context precision | Context recall | Average retries |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for category in ("easy", "hard", "abstention"):
        rows = grouped[category]
        if category == "abstention":
            correct = sum(bool(row["abstained"]) for row in rows)
            citation_accuracy = None
            metric_values = {name: None for name in RAGAS_COLUMNS}
        else:
            correct = sum(not bool(row["abstained"]) for row in rows)
            citation_accuracy = _percentage(
                sum(row["expected_source_id"] in (row.get("cited_source_ids") or []) for row in rows),
                len(rows),
            )
            metric_values = easy_ragas if category == "easy" else hard_ragas
        avg_retries = _mean([row.get("retries", 0) for row in rows])
        lines.append(
            f"| {category.title()} | {len(rows)} | {_display(_percentage(correct, len(rows)), percent=True)} ({correct}/{len(rows)}) | "
            f"{_display(citation_accuracy, percent=True)} | {_display(metric_values['faithfulness'])} | "
            f"{_display(metric_values['context_precision'])} | {_display(metric_values['context_recall'])} | {_display(avg_retries)} |"
        )

    lines.extend([
        "",
        f"Average retries across all 50 questions: **{_display(_mean([row.get('retries', 0) for row in results]))}**.",
        "",
        "## Diagnostics",
        "",
        "### RAGAS job success/failure breakdown",
        "",
        "A finite numeric score is a successful job; `judge unavailable` marks a failed final retry. Aggregate means above use the finite scores in this same detailed file.",
        "",
        "| Category | Metric | Scored | Judge unavailable | Total |",
        "|---|---|---:|---:|---:|",
    ])
    for category in ("easy", "hard"):
        subset = scored[scored["category"] == category]
        for metric in RAGAS_COLUMNS:
            succeeded = sum(_metric_value(value) is not None for value in subset[metric].tolist())
            unavailable = sum(value == "judge unavailable" for value in subset[metric].tolist())
            total = len(subset)
            lines.append(f"| {category.title()} | {metric.replace('_', ' ').title()} | {succeeded} | {unavailable} | {total} |")

    answerable_by_id = {int(row["id"]): row for row in answerable}
    hard_scores = scored[scored["category"] == "hard"]
    lines.extend([
        "",
        "### Hard questions: retries and faithfulness",
        "",
        "| ID | Question | Retries | Faithfulness |",
        "|---:|---|---:|---:|",
    ])
    for _, score_row in hard_scores.iterrows():
        result = answerable_by_id[int(score_row["id"])]
        value = _metric_value(score_row["faithfulness"])
        lines.append(
            f"| {int(score_row['id'])} | {result['question']} | {result.get('retries', 0)} | "
            f"{_display(value)} |"
        )

    retry_summary = ", ".join(f"{row['id']}={row.get('retries', 0)}" for row in abstention_rows)
    all_two = len(abstention_rows) == 10 and all(int(row.get("retries", 0)) == 2 for row in abstention_rows)
    lines.extend([
        "",
        "### Abstention retry check",
        "",
        f"Retries by question ID: {retry_summary}.",
        f"All 10 abstention questions used exactly 2 retries: **{'yes' if all_two else 'no'}**.",
        "",
        "### Metric notes",
        "",
        "- Metric jobs with no returned judgment remain in `scores_detailed.jsonl` as `judge unavailable`; they are not dropped or imputed.",
        "- `scores_progress.json` records completed, unavailable, and retried jobs; unavailable cells are frozen and will not be retried again.",
        "- RAGAS uses the returned citation snippets as retrieved contexts. Missing citation contexts remain empty.",
        "",
    ])
    report = "\n".join(lines)
    REPORT_PATH.write_text(report, encoding="utf-8")
    return report


def run_scoring(*, retry_cells: set[str] | None = None) -> str:
    results = _read_jsonl(RESULTS_PATH)
    if len(results) != 50:
        raise ValueError(f"Expected 50 saved API results; found {len(results)}")
    if any(row.get("error") or row.get("llm_error_count", 0) for row in results):
        raise RuntimeError("Saved API results contain failed API queries; refusing to score")
    answerable = [row for row in results if row["category"] in {"easy", "hard"}]
    score_detailed(answerable, retry_cells=retry_cells)
    return build_report()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--retry-cells",
        default="",
        help="Comma-separated omitted cells to retry once, e.g. 2:faithfulness,15:context_precision.",
    )
    try:
        args = parser.parse_args()
        retry_cells = {cell.strip() for cell in args.retry_cells.split(",") if cell.strip()} or None
        print(run_scoring(retry_cells=retry_cells))
    except QuotaExhausted as exc:
        raise SystemExit(2)
