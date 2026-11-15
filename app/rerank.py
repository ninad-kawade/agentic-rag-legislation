"""Cross-encoder reranking for hybrid retrieval candidates."""

from __future__ import annotations

from functools import lru_cache
from typing import Any

from sentence_transformers import CrossEncoder

RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


@lru_cache(maxsize=2)
def _cross_encoder(model_name: str) -> CrossEncoder:
    return CrossEncoder(model_name, device="cpu")


def rerank(
    question: str,
    candidates: list[dict[str, Any]],
    *,
    top_k: int = 5,
    model_name: str = RERANK_MODEL,
    model: CrossEncoder | None = None,
) -> list[dict[str, Any]]:
    """Score query/candidate pairs and return the highest-scoring candidates."""
    if not candidates:
        return []
    cross_encoder = model or _cross_encoder(model_name)
    pairs = [(question, candidate["text"]) for candidate in candidates]
    scores = cross_encoder.predict(pairs, show_progress_bar=False)
    reranked = [
        {**candidate, "rerank_score": float(score)}
        for candidate, score in zip(candidates, scores)
    ]
    return sorted(reranked, key=lambda result: result["rerank_score"], reverse=True)[:top_k]


def retrieve_and_rerank(
    question: str,
    *,
    index_dir: str = "data/index",
    candidate_k: int = 50,
    fused_k: int = 20,
    top_k: int = 5,
) -> list[dict[str, Any]]:
    """Complete Phase 2 pipeline: parallel retrieval, RRF fusion, reranking."""
    from pathlib import Path

    from app.retrieval import retrieve_hybrid

    candidates = retrieve_hybrid(
        question,
        Path(index_dir),
        candidate_k=candidate_k,
        top_k=fused_k,
    )
    return rerank(question, candidates, top_k=top_k)
