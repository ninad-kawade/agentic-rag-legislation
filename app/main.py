"""FastAPI entry point for the evidence-grounded query service."""

from __future__ import annotations

import logging
import json
import threading
import time

from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel, Field, field_validator

from app.graph import LLMQuotaError, ask
from app.schemas import Answer

logger = logging.getLogger(__name__)
app = FastAPI(title="Evidence RAG", version="0.4.0")
_metrics_lock = threading.Lock()
_request_count = 0
_total_latency_seconds = 0.0
_abstention_count = 0


class QueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)

    @field_validator("question")
    @classmethod
    def strip_and_validate_question(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("question must not be empty")
        return value


@app.post("/query", response_model=Answer)
def query(request: QueryRequest, response: Response) -> Answer:
    """Run the Phase 3 graph and return its validated answer object."""
    global _request_count, _total_latency_seconds, _abstention_count
    started = time.perf_counter()
    try:
        result = ask(request.question)
        answer = result["answer"]
        response.headers["X-RAG-Abstained"] = str(bool(result.get("abstained", False))).lower()
        response.headers["X-RAG-Retries"] = str(int(result.get("retries", 0)))
        response.headers["X-RAG-LLM-Errors"] = json.dumps(result.get("llm_errors", []))
        with _metrics_lock:
            _request_count += 1
            _abstention_count += int(result.get("abstained", False))
            _total_latency_seconds += time.perf_counter() - started
        return answer
    except Exception as exc:
        with _metrics_lock:
            _request_count += 1
            _total_latency_seconds += time.perf_counter() - started
        logger.exception("Query agent failed")
        if isinstance(exc, LLMQuotaError):
            raise HTTPException(
                status_code=429,
                detail="service is busy, please try again later",
            ) from exc
        raise HTTPException(
            status_code=503,
            detail="The answer service is temporarily unavailable.",
        ) from exc


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/metrics")
def metrics() -> dict[str, float | int]:
    with _metrics_lock:
        return {
            "request_count": _request_count,
            "avg_latency_seconds": (
                _total_latency_seconds / _request_count if _request_count else 0.0
            ),
            "abstention_rate": (
                _abstention_count / _request_count if _request_count else 0.0
            ),
        }
