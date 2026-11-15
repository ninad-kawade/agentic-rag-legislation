"""Query the Phase 1 vector and BM25 indexes independently."""

from __future__ import annotations

import os

# Query-time embeddings must use the same locally cached model as index builds.
# Set these before importing Transformers/SentenceTransformers so a cache miss
# fails locally instead of attempting a Hub request.
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import argparse
import pickle
import re
from pathlib import Path
from typing import Any

import chromadb
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from ingest.build_index import EMBEDDING_MODEL


def _tokens(text: str) -> list[str]:
    return re.findall(r"\b\w+\b", text.lower())


def query_bm25(question: str, index_dir: Path, top_k: int) -> list[dict[str, Any]]:
    with (index_dir / "bm25.pkl").open("rb") as handle:
        payload = pickle.load(handle)
    bm25 = BM25Okapi(payload["tokenized_corpus"])
    scores = bm25.get_scores(_tokens(question))
    ranked = sorted(range(len(scores)), key=lambda index: scores[index], reverse=True)[:top_k]
    return [{**payload["chunks"][index], "score": float(scores[index])} for index in ranked]


def query_vector(question: str, index_dir: Path, top_k: int) -> list[dict[str, Any]]:
    client = chromadb.PersistentClient(path=str(index_dir / "chroma"))
    collection = client.get_collection("legislation_chunks")
    model = SentenceTransformer(EMBEDDING_MODEL, device="cpu")
    embedding = model.encode([question], batch_size=32, normalize_embeddings=True).tolist()
    result = collection.query(query_embeddings=embedding, n_results=top_k, include=["documents", "metadatas", "distances"])
    matches: list[dict[str, Any]] = []
    for text, metadata, distance in zip(result["documents"][0], result["metadatas"][0], result["distances"][0]):
        matches.append({**metadata, "text": text, "score": float(1 - distance)})
    return matches


def _print_results(label: str, results: list[dict[str, Any]]) -> None:
    print(f"\n=== {label} ===")
    for rank, result in enumerate(results, start=1):
        print(f"\n[{rank}] score={result['score']:.6f}")
        print(f"source: {result['source']}")
        print(f"title: {result['title']}")
        print(f"text: {result['text']}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question")
    parser.add_argument("--index-dir", type=Path, default=Path("data/index"))
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()
    _print_results("BM25", query_bm25(args.question, args.index_dir, args.top_k))
    _print_results("VECTOR", query_vector(args.question, args.index_dir, args.top_k))


if __name__ == "__main__":
    main()
