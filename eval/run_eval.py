"""Run every approved question through the live Evidence RAG API."""

from __future__ import annotations

import json
import os
import argparse
import time
from pathlib import Path
from typing import Any

import requests

ROOT = Path(__file__).resolve().parents[1]
TESTSET_PATH = ROOT / "testset.jsonl"
RESULTS_PATH = ROOT / "eval" / "results.jsonl"
API_URL = os.getenv("RAG_API_URL", "http://127.0.0.1:8000/query")
REQUEST_DELAY_SECONDS = float(os.getenv("RAG_EVAL_REQUEST_DELAY_SECONDS", "0"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _run_question(session: requests.Session, test: dict[str, Any]) -> dict[str, Any]:
    started = time.perf_counter()
    response = session.post(
        API_URL,
        json={"question": test["question"]},
        timeout=300,
    )
    response.raise_for_status()
    payload = response.json()
    citations = payload.get("citations", [])
    raw_errors = response.headers.get("X-RAG-LLM-Errors", "[]")
    try:
        llm_errors = json.loads(raw_errors)
    except json.JSONDecodeError:
        llm_errors = [raw_errors] if raw_errors not in {"", "0", "[]"} else []
    if isinstance(llm_errors, int):
        llm_error_count = llm_errors
        llm_errors = []
    elif isinstance(llm_errors, list):
        llm_error_count = len(llm_errors)
    else:
        llm_errors = [str(llm_errors)]
        llm_error_count = len(llm_errors)
    return {
        "id": test["id"],
        "category": test["category"],
        "question": test["question"],
        "answer": payload["text"],
        "abstained": response.headers.get("X-RAG-Abstained", "").lower() == "true",
        "confidence": float(payload["confidence"]),
        "retries": int(response.headers.get("X-RAG-Retries", "0")),
        "llm_error_count": llm_error_count,
        "llm_errors": llm_errors,
        "cited_source_ids": [
            citation["source_id"]
            for citation in citations
            if citation.get("source_id")
        ],
        "citations": citations,
        "expected_answer": test["expected_answer"],
        "expected_source_id": test["expected_source_id"],
        "duration_seconds": round(time.perf_counter() - started, 3),
    }


def run_eval(*, retry_errors: bool = False) -> list[dict[str, Any]]:
    tests = _read_jsonl(TESTSET_PATH)
    if len(tests) != 50:
        raise ValueError(f"Expected the approved 50-row testset, found {len(tests)} rows")

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    existing: list[dict[str, Any]] = []
    if retry_errors and RESULTS_PATH.exists():
        existing = _read_jsonl(RESULTS_PATH)
        if len(existing) != len(tests) or [row["id"] for row in existing] != [row["id"] for row in tests]:
            raise ValueError("Cannot retry errors: existing results do not match the complete testset")
    results = existing or [None] * len(tests)
    pending = [
        index for index, row in enumerate(results)
        if row is None or (retry_errors and row.get("llm_error_count", 0) > 0)
    ]
    with requests.Session() as session:
        for done, index in enumerate(pending, start=1):
            test = tests[index]
            result = _run_question(session, test)
            results[index] = result
            with RESULTS_PATH.open("w", encoding="utf-8") as output:
                output.writelines(json.dumps(row, ensure_ascii=False) + "\n" for row in results if row is not None)
            print(f"{done}/{len(pending)} processed — id={test['id']} ({test['category']}); overall row {index + 1}/{len(tests)}", flush=True)
            if done < len(pending) and REQUEST_DELAY_SECONDS > 0:
                time.sleep(REQUEST_DELAY_SECONDS)
    print(f"Saved {len(results)} results to {RESULTS_PATH}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--retry-errors", action="store_true", help="Retry only rows with recorded LLM errors in the existing full results file")
    run_eval(retry_errors=parser.parse_args().retry_errors)
