"""One-time script to build (or rebuild) the FAISS index from the corpus."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from rag.indexer import build_index

if __name__ == "__main__":
    build_index()
