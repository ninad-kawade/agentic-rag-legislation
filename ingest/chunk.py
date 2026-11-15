"""Chunk parsed documents while retaining their source metadata."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

from ingest.load import DocumentRecord, load_documents


@dataclass(frozen=True)
class ChunkRecord:
    chunk_id: str
    document_id: str
    source: str
    title: str
    section: str
    chunk_index: int
    text: str


def chunk_text(text: str, size: int = 800, overlap: int = 150) -> list[str]:
    if size <= overlap:
        raise ValueError("Chunk size must be larger than overlap")
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            boundary = text.rfind(" ", start + size // 2, end)
            if boundary > start:
                end = boundary
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return chunks


def chunk_documents(documents: list[DocumentRecord], size: int = 800, overlap: int = 150) -> list[ChunkRecord]:
    chunks: list[ChunkRecord] = []
    for document in documents:
        for index, text in enumerate(chunk_text(document.text, size=size, overlap=overlap)):
            digest = hashlib.sha1(f"{document.document_id}:{index}".encode()).hexdigest()[:16]
            chunks.append(
                ChunkRecord(
                    chunk_id=f"{document.document_id}-{digest}",
                    document_id=document.document_id,
                    source=document.source,
                    title=document.title,
                    section=document.section,
                    chunk_index=index,
                    text=text,
                )
            )
    return chunks


def write_chunks(chunks: list[ChunkRecord], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for chunk in chunks:
            handle.write(json.dumps(asdict(chunk), ensure_ascii=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/chunks.jsonl"))
    parser.add_argument("--size", type=int, default=800)
    parser.add_argument("--overlap", type=int, default=150)
    args = parser.parse_args()
    documents = load_documents(args.raw_dir)
    chunks = chunk_documents(documents, size=args.size, overlap=args.overlap)
    write_chunks(chunks, args.output)
    print(f"Created {len(chunks)} chunks from {len(documents)} documents into {args.output}")


if __name__ == "__main__":
    main()
