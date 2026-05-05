"""Builds and persists a FAISS index over the legal corpus."""

import os
import json
import pickle
import numpy as np
import faiss
from pathlib import Path
from sentence_transformers import SentenceTransformer

CORPUS_DIR = Path(__file__).parent.parent / "corpus"
INDEX_DIR = Path(__file__).parent.parent / "data" / "index"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
MAX_CHUNK_SIZE = 900  # chars; sections bigger than this get split further
CHAR_OVERLAP = 100


def _split_large_section(section: str, doc_title: str) -> list[dict]:
    """Fall back to character chunking for unusually large sections."""
    chunks = []
    start = 0
    while start < len(section):
        end = min(start + MAX_CHUNK_SIZE, len(section))
        chunk = section[start:end].strip()
        if len(chunk) > 80:
            chunks.append({"text": chunk, "doc": doc_title})
        start += MAX_CHUNK_SIZE - CHAR_OVERLAP
    return chunks


def _chunk_text(text: str, doc_title: str) -> list[dict]:
    """
    Split on '---' section separators first (preserves semantic boundaries
    for cases and amendments), then fall back to character chunking only for
    unusually large sections like Constitution articles.
    """
    sections = [s.strip() for s in text.split("---") if s.strip()]
    chunks = []
    for section in sections:
        if len(section) <= MAX_CHUNK_SIZE:
            chunks.append({"text": section, "doc": doc_title})
        else:
            chunks.extend(_split_large_section(section, doc_title))
    return chunks


def build_index() -> None:
    """Read corpus, embed chunks, build FAISS index, save to disk."""
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    model = SentenceTransformer(MODEL_NAME)

    all_chunks = []
    for corpus_file in sorted(CORPUS_DIR.glob("*.txt")):
        doc_title = corpus_file.stem.replace("_", " ").title()
        text = corpus_file.read_text(encoding="utf-8")
        all_chunks.extend(_chunk_text(text, doc_title))

    print(f"Indexing {len(all_chunks)} chunks from {CORPUS_DIR}...")
    texts = [c["text"] for c in all_chunks]
    embeddings = model.encode(texts, batch_size=32, show_progress_bar=True, normalize_embeddings=True)
    embeddings = embeddings.astype("float32")

    # Inner product on normalized vectors == cosine similarity
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    faiss.write_index(index, str(INDEX_DIR / "lawbot.index"))
    with open(INDEX_DIR / "chunks.pkl", "wb") as f:
        pickle.dump(all_chunks, f)

    print(f"Index saved: {len(all_chunks)} chunks, dim={dim}")


def load_index():
    """Return (faiss_index, chunks_list). Builds on first run if missing."""
    index_path = INDEX_DIR / "lawbot.index"
    chunks_path = INDEX_DIR / "chunks.pkl"

    if not index_path.exists() or not chunks_path.exists():
        build_index()

    index = faiss.read_index(str(index_path))
    with open(chunks_path, "rb") as f:
        chunks = pickle.load(f)
    return index, chunks


if __name__ == "__main__":
    build_index()
