"""Hybrid BM25 and vector retrieval with Reciprocal Rank Fusion."""

from __future__ import annotations

import pickle
import re
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from pathlib import Path
from typing import Any

import chromadb
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from ingest.build_index import EMBEDDING_MODEL


def _tokens(text: str) -> list[str]:
    return re.findall(r"\b\w+\b", text.lower())


@lru_cache(maxsize=1)
def _embedding_model() -> SentenceTransformer:
    return SentenceTransformer(EMBEDDING_MODEL, device="cpu")


def query_bm25(question: str, index_dir: Path, top_k: int) -> list[dict[str, Any]]:
    with (index_dir / "bm25.pkl").open("rb") as handle:
        payload = pickle.load(handle)
    bm25 = BM25Okapi(payload["tokenized_corpus"])
    scores = bm25.get_scores(_tokens(question))
    ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
    return [{**payload["chunks"][i], "score": float(scores[i])} for i in ranked]


def query_vector(question: str, index_dir: Path, top_k: int) -> list[dict[str, Any]]:
    client = chromadb.PersistentClient(path=str(index_dir / "chroma"))
    collection = client.get_collection("legislation_chunks")
    embedding = _embedding_model().encode(
        [question], batch_size=32, normalize_embeddings=True
    ).tolist()
    result = collection.query(
        query_embeddings=embedding,
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )
    matches: list[dict[str, Any]] = []
    for chunk_id, text, metadata, distance in zip(
        result["ids"][0], result["documents"][0], result["metadatas"][0], result["distances"][0]
    ):
        matches.append({**metadata, "chunk_id": chunk_id, "text": text, "score": float(1 - distance)})
    return matches


def reciprocal_rank_fusion(
    ranked_lists: list[list[dict[str, Any]]], *, rrf_k: int = 60
) -> list[dict[str, Any]]:
    """Fuse ranked candidates, deduplicating by stable chunk ID (or content)."""
    fused: dict[str, dict[str, Any]] = {}
    names = ("bm25", "vector")
    for list_index, results in enumerate(ranked_lists):
        source_name = names[list_index] if list_index < len(names) else f"retriever_{list_index + 1}"
        for rank, result in enumerate(results, start=1):
            key = f"{result.get('source')}|{result.get('text')}"
            if key not in fused:
                fused[key] = {**result, "rrf_score": 0.0, "retrieval_ranks": {}}
            fused[key]["rrf_score"] += 1.0 / (rrf_k + rank)
            fused[key]["retrieval_ranks"][source_name] = rank
    return sorted(fused.values(), key=lambda result: result["rrf_score"], reverse=True)


def retrieve_hybrid(
    question: str,
    index_dir: Path = Path("data/index"),
    *,
    candidate_k: int = 50,
    top_k: int = 20,
    rrf_k: int = 60,
) -> list[dict[str, Any]]:
    """Run BM25 and dense retrieval concurrently, then return fused candidates."""
    with ThreadPoolExecutor(max_workers=2, thread_name_prefix="hybrid-retrieval") as pool:
        bm25_future = pool.submit(query_bm25, question, index_dir, candidate_k)
        vector_future = pool.submit(query_vector, question, index_dir, candidate_k)
        bm25_results = bm25_future.result()
        vector_results = vector_future.result()

    # Shared source + chunk text identity deduplicates records from both indexes;
    # the underlying chunk IDs are not stored in Chroma metadata.
    return reciprocal_rank_fusion([bm25_results, vector_results], rrf_k=rrf_k)[:top_k]
