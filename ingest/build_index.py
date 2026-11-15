"""Build the Phase 1 ChromaDB and BM25 indexes from shared chunks."""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import argparse
import json
import pickle
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import chromadb
import torch
from sentence_transformers import SentenceTransformer

from ingest.chunk import chunk_documents, write_chunks
from ingest.load import load_documents, write_documents

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
ENCODE_BATCH_SIZE = 32
CHECKPOINT_CHUNK_SIZE = 500
CHECKPOINT_PATH = Path("data/.embed_progress.json")


def _tokens(text: str) -> list[str]:
    return re.findall(r"\b\w+\b", text.lower())


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _read_checkpoint(path: Path, total_chunks: int) -> set[int]:
    if not path.exists():
        return set()
    checkpoint = json.loads(path.read_text(encoding="utf-8"))
    if checkpoint.get("total_chunks") != total_chunks:
        return set()
    return {int(batch) for batch in checkpoint.get("completed_batches", [])}


def _write_checkpoint(path: Path, total_chunks: int, completed_batches: set[int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(".tmp")
    temporary_path.write_text(
        json.dumps(
            {
                "total_chunks": total_chunks,
                "batch_size": CHECKPOINT_CHUNK_SIZE,
                "completed_batches": sorted(completed_batches),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    temporary_path.replace(path)


def _encode_with_timestamps(
    model: SentenceTransformer,
    texts: list[str],
    run_started_at: float,
) -> list[list[float]]:
    embeddings: list[list[float]] = []
    model_batch_count = 0
    call_size = ENCODE_BATCH_SIZE * 10
    for start in range(0, len(texts), call_size):
        end = min(start + call_size, len(texts))
        encoded = model.encode(
            texts[start:end],
            batch_size=ENCODE_BATCH_SIZE,
            show_progress_bar=True,
            convert_to_numpy=True,
        )
        embeddings.extend(encoded.tolist())
        model_batch_count += (end - start + ENCODE_BATCH_SIZE - 1) // ENCODE_BATCH_SIZE
        if model_batch_count % 10 == 0 or end == len(texts):
            elapsed = time.perf_counter() - run_started_at
            print(
                f"[{datetime.now().isoformat(timespec='seconds')}] "
                f"encoded {end}/{len(texts)} chunks in this checkpoint batch; "
                f"model_batches={model_batch_count}; elapsed={elapsed:.1f}s"
            )
    return embeddings


def build_indexes(
    raw_dir: Path = Path("data/raw"),
    processed_dir: Path = Path("data/processed"),
    index_dir: Path = Path("data/index"),
    model_name: str = EMBEDDING_MODEL,
    reset: bool = False,
) -> tuple[int, int]:
    documents = load_documents(raw_dir)
    write_documents(documents, processed_dir / "documents.jsonl")
    chunks = chunk_documents(documents, size=800, overlap=150)
    write_chunks(chunks, processed_dir / "chunks.jsonl")
    chunk_dicts = [chunk.__dict__ for chunk in chunks]
    texts = [chunk["text"] for chunk in chunk_dicts]
    ids = [chunk["chunk_id"] for chunk in chunk_dicts]
    metadatas = [
        {
            "document_id": chunk["document_id"],
            "source": chunk["source"],
            "title": chunk["title"],
            "section": chunk["section"],
            "chunk_index": str(chunk["chunk_index"]),
        }
        for chunk in chunk_dicts
    ]

    index_dir.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(index_dir / "chroma"))
    if reset:
        try:
            client.delete_collection("legislation_chunks")
        except ValueError:
            pass
        CHECKPOINT_PATH.unlink(missing_ok=True)
    collection = client.get_or_create_collection(
        name="legislation_chunks",
        metadata={"embedding_model": model_name, "distance": "cosine"},
    )

    run_started_at = time.perf_counter()
    torch.set_num_threads(os.cpu_count() or 1)
    model = SentenceTransformer(model_name, device="cpu")
    completed_batches = _read_checkpoint(CHECKPOINT_PATH, len(texts))
    total_batches = (len(texts) + CHECKPOINT_CHUNK_SIZE - 1) // CHECKPOINT_CHUNK_SIZE
    print(
        f"Embedding {len(texts)} chunks in {total_batches} batches; "
        f"completed batches: {len(completed_batches)}"
    )
    for batch_number, start in enumerate(range(0, len(texts), CHECKPOINT_CHUNK_SIZE)):
        if batch_number in completed_batches:
            print(f"Skipping completed batch {batch_number + 1}/{total_batches}")
            continue
        end = min(start + CHECKPOINT_CHUNK_SIZE, len(texts))
        batch_texts = texts[start:end]
        print(f"Encoding chunks {start + 1}-{end} locally with {model_name}")
        embeddings = _encode_with_timestamps(model, batch_texts, run_started_at)
        collection.upsert(
            ids=ids[start:end],
            documents=batch_texts,
            metadatas=metadatas[start:end],
            embeddings=embeddings,
        )
        completed_batches.add(batch_number)
        _write_checkpoint(CHECKPOINT_PATH, len(texts), completed_batches)
        print(
            f"[{datetime.now().isoformat(timespec='seconds')}] "
            f"Persisted chunks {start + 1}-{end} of {len(texts)} "
            f"({len(completed_batches)}/{total_batches} batches)"
        )

    tokenized = [_tokens(text) for text in texts]
    bm25_payload = {
        "chunks": chunk_dicts,
        "tokenized_corpus": tokenized,
        "model_name": model_name,
    }
    with (index_dir / "bm25.pkl").open("wb") as handle:
        pickle.dump(bm25_payload, handle)
    elapsed = time.perf_counter() - run_started_at
    print(f"Full embedding run elapsed={elapsed:.1f}s ({elapsed / 60:.2f}m)")
    return len(documents), len(chunks)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--index-dir", type=Path, default=Path("data/index"))
    parser.add_argument("--reset", action="store_true", help="Rebuild vector embeddings from scratch")
    args = parser.parse_args()
    document_count, chunk_count = build_indexes(args.raw_dir, args.processed_dir, args.index_dir, reset=args.reset)
    print(f"Indexed {document_count} documents and {chunk_count} chunks")


if __name__ == "__main__":
    main()
