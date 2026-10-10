from pathlib import Path
from app.rag import RAGService

def test_chunk_text_splits_long_text_with_overlap():
    text = "A" * 2200
    chunks = RAGService._chunk_text(text)
    assert len(chunks) >= 3
    assert all(chunks)
    assert chunks[0][-100:] == chunks[1][:100]

def test_chunk_text_empty_input():
    assert RAGService._chunk_text("   ") == []
