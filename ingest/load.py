"""Parse legislation.gov.uk XML documents into plain text records."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
import xml.etree.ElementTree as ET


@dataclass(frozen=True)
class DocumentRecord:
    document_id: str
    source: str
    title: str
    section: str
    text: str


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _normalise(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _find_metadata(root: ET.Element, local_name: str, namespace: str) -> str:
    return _normalise(root.findtext(f".//{{{namespace}}}{local_name}", default=""))


def _body_text(root: ET.Element) -> str:
    pieces: list[str] = []

    def visit(element: ET.Element, in_metadata: bool = False) -> None:
        metadata = in_metadata or _local_name(element.tag).lower() == "metadata"
        if not metadata and element.text:
            pieces.append(element.text)
        for child in element:
            visit(child, metadata)
            if not metadata and child.tail:
                pieces.append(child.tail)

    visit(root)
    return _normalise(" ".join(pieces))


def parse_document(path: Path) -> DocumentRecord:
    root = ET.parse(path).getroot()
    dc_namespace = "http://purl.org/dc/elements/1.1/"
    identifier = _find_metadata(root, "identifier", dc_namespace)
    title = _find_metadata(root, "title", dc_namespace) or path.stem
    source = identifier or root.attrib.get("DocumentURI", "") or path.stem
    source = source.replace("http://", "https://", 1)
    text = _body_text(root)
    if not text:
        raise ValueError(f"No body text found in {path}")
    return DocumentRecord(
        document_id=path.stem,
        source=source,
        title=title,
        section="full document",
        text=text,
    )


def load_documents(raw_dir: Path) -> list[DocumentRecord]:
    documents: list[DocumentRecord] = []
    paths = sorted(raw_dir.glob("*.xml"))
    if not paths:
        raise FileNotFoundError(f"No XML documents found in {raw_dir}")
    for path in paths:
        documents.append(parse_document(path))
    return documents


def write_documents(documents: list[DocumentRecord], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for document in documents:
            handle.write(json.dumps(asdict(document), ensure_ascii=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/documents.jsonl"))
    args = parser.parse_args()
    documents = load_documents(args.raw_dir)
    write_documents(documents, args.output)
    print(f"Parsed {len(documents)} documents into {args.output}")


if __name__ == "__main__":
    main()
