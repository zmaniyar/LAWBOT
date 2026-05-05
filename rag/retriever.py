"""Semantic search over the FAISS index with similarity-threshold filtering."""

import numpy as np
from functools import lru_cache
from sentence_transformers import SentenceTransformer
from rag.indexer import load_index

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_THRESHOLD = 0.35  # cosine similarity on normalized vecs; below this → no result
DEFAULT_TOP_K = 5


@lru_cache(maxsize=1)
def _load_resources():
    model = SentenceTransformer(MODEL_NAME)
    index, chunks = load_index()
    return model, index, chunks


def search(query: str, top_k: int = DEFAULT_TOP_K, threshold: float = DEFAULT_THRESHOLD) -> list[dict]:
    """
    Returns up to top_k chunks whose cosine similarity to query exceeds threshold.
    Each result: {"text": str, "doc": str, "score": float, "rank": int}
    Returns [] if nothing passes the threshold.
    """
    model, index, chunks = _load_resources()
    query_emb = model.encode([query], normalize_embeddings=True).astype("float32")
    scores, indices = index.search(query_emb, k=min(top_k * 2, len(chunks)))

    results = []
    for rank, (idx, score) in enumerate(zip(indices[0], scores[0])):
        if score < threshold:
            break
        results.append({
            "text": chunks[idx]["text"],
            "doc": chunks[idx]["doc"],
            "score": float(score),
            "rank": rank + 1,
        })
        if len(results) == top_k:
            break
    return results
