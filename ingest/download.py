"""Download a reproducible slice of UK legislation from legislation.gov.uk."""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from urllib.parse import urljoin, urlparse
import xml.etree.ElementTree as ET

import requests

FEED_URL = "https://www.legislation.gov.uk/all/data.feed"
ATOM_NS = "http://www.w3.org/2005/Atom"
USER_AGENT = "evidence-rag/0.1 (research project; contact via repository)"


def _feed_document_urls(session: requests.Session, count: int) -> list[str]:
    urls: list[str] = []
    seen: set[str] = set()
    feed_url = FEED_URL

    while feed_url and len(urls) < count:
        response = session.get(feed_url, timeout=30)
        response.raise_for_status()
        root = ET.fromstring(response.content)

        for entry in root.findall(f"{{{ATOM_NS}}}entry"):
            link = entry.find(f"{{{ATOM_NS}}}link[@rel='alternate']")
            if link is None:
                link = entry.find(f"{{{ATOM_NS}}}link")
            href = link.get("href") if link is not None else None
            if not href:
                continue
            document_url = href.replace("http://", "https://", 1).rstrip("/")
            if document_url.endswith("/data.xml"):
                source_url = document_url[:-9]
            else:
                source_url = document_url
            xml_url = f"{source_url}/data.xml"
            if xml_url not in seen:
                seen.add(xml_url)
                urls.append(xml_url)
                if len(urls) >= count:
                    break

        next_link = root.find(f"{{{ATOM_NS}}}link[@rel='next']")
        feed_url = next_link.get("href") if next_link is not None else ""
        if feed_url:
            feed_url = urljoin(FEED_URL, feed_url).replace("http://", "https://", 1)

    return urls


def _filename(url: str) -> str:
    parts = [part for part in urlparse(url).path.split("/") if part]
    stem = "-".join(parts[-3:]) if len(parts) >= 3 else parts[-1]
    stem = re.sub(r"[^A-Za-z0-9._-]+", "-", stem).strip("-")
    return f"{stem}.xml"


def download_corpus(count: int, output_dir: Path) -> list[dict[str, str]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT, "Accept": "application/xml"})
    urls = _feed_document_urls(session, count)
    if len(urls) < count:
        raise RuntimeError(f"Feed returned only {len(urls)} documents; wanted {count}.")

    manifest: list[dict[str, str]] = []
    for number, url in enumerate(urls, start=1):
        path = output_dir / _filename(url)
        if not path.exists():
            response = session.get(url, timeout=60)
            response.raise_for_status()
            path.write_bytes(response.content)
        manifest.append({"source": url.removesuffix("/data.xml"), "path": str(path)})
        if number % 25 == 0 or number == len(urls):
            print(f"Downloaded {number}/{len(urls)} documents")
        time.sleep(0.05)

    manifest_path = output_dir / "manifest.jsonl"
    with manifest_path.open("w", encoding="utf-8") as handle:
        for item in manifest:
            handle.write(json.dumps(item, ensure_ascii=True) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=200)
    parser.add_argument("--output-dir", type=Path, default=Path("data/raw"))
    args = parser.parse_args()
    manifest = download_corpus(args.count, args.output_dir)
    print(f"Saved {len(manifest)} documents to {args.output_dir}")


if __name__ == "__main__":
    main()
