from ingest.chunk import chunk_text


def test_chunk_overlap_and_size() -> None:
    text = "word " * 500
    chunks = chunk_text(text, size=800, overlap=150)
    assert len(chunks) > 1
    assert all(len(chunk) <= 800 for chunk in chunks)
    assert chunks[0][-20:].strip() in chunks[1]
