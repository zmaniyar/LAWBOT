"""
Eval runner: measures retrieval recall@5 and citation accuracy.

Usage:
    python -m evals.run_evals [--save]

Metrics produced:
    recall@5          — was the expected_doc in the top-5 retrieved chunks?
    citation_accuracy — did the LLM answer cite the expected_doc?
    no_answer_rate    — fraction of queries that returned no chunks (below threshold)

Saves results to evals/results/latest.json and prints a markdown table.
"""

import json
import re
import sys
import os
import argparse
from datetime import datetime
from pathlib import Path

# Allow running from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from rag.retriever import search
from rag.llm import answer

TEST_SET_PATH = Path(__file__).parent / "test_set.json"
RESULTS_DIR = Path(__file__).parent / "results"


def check_recall(chunks: list[dict], expected_doc: str) -> bool:
    """True if expected_doc appears in any of the top-k retrieved chunks."""
    expected_lower = expected_doc.lower()
    for c in chunks:
        if expected_lower in c["doc"].lower():
            return True
    return False


def check_citation(response: str, expected_doc: str) -> bool:
    """True if LLM response contains a citation referencing expected_doc."""
    expected_lower = expected_doc.lower()
    # Match patterns like [Doc N — Title] or [Doc N - Title]
    citations = re.findall(r'\[Doc\s*\d+\s*[—\-]\s*([^\]]+)\]', response, re.IGNORECASE)
    for citation in citations:
        if expected_lower in citation.lower():
            return True
    return False


def check_keywords(response: str, keywords: list[str]) -> float:
    """Returns fraction of expected_keywords found (case-insensitive) in the response."""
    response_lower = response.lower()
    hits = sum(1 for kw in keywords if kw.lower() in response_lower)
    return hits / len(keywords) if keywords else 0.0


def run_evals(use_llm: bool = True) -> dict:
    with open(TEST_SET_PATH) as f:
        test_set = json.load(f)

    results = []
    recall_hits = 0
    citation_hits = 0
    no_answer_count = 0
    keyword_scores = []

    print(f"\nRunning {len(test_set)} queries...\n")
    for i, item in enumerate(test_set):
        query = item["query"]
        expected_doc = item["expected_doc"]
        expected_keywords = item.get("expected_keywords", [])

        chunks = search(query)
        recalled = check_recall(chunks, expected_doc)
        recall_hits += int(recalled)

        if not chunks:
            no_answer_count += 1
            cited = False
            kw_score = 0.0
            response_text = "[NO CHUNKS RETRIEVED]"
        elif use_llm:
            response_text = answer(query, chunks, stream=False)
            cited = check_citation(response_text, expected_doc)
            kw_score = check_keywords(response_text, expected_keywords)
            citation_hits += int(cited)
        else:
            response_text = "[LLM SKIPPED]"
            cited = None
            kw_score = None

        keyword_scores.append(kw_score)

        status = "✓" if recalled else "✗"
        print(f"[{i+1:02d}/{len(test_set)}] {status} {query[:60]}")

        results.append({
            "id": item["id"],
            "category": item.get("category", ""),
            "query": query,
            "expected_doc": expected_doc,
            "chunks_retrieved": len(chunks),
            "recalled": recalled,
            "cited": cited,
            "keyword_score": kw_score,
            "response": response_text,
            "top_chunks": [{"doc": c["doc"], "score": c["score"]} for c in chunks],
        })

    n = len(test_set)
    recall_pct = recall_hits / n * 100
    citation_pct = citation_hits / (n - no_answer_count) * 100 if use_llm and (n - no_answer_count) > 0 else None
    no_answer_pct = no_answer_count / n * 100
    avg_kw = sum(s for s in keyword_scores if s is not None) / max(sum(1 for s in keyword_scores if s is not None), 1) * 100

    summary = {
        "timestamp": datetime.now().isoformat(),
        "n_queries": n,
        "recall_at_5": round(recall_pct, 1),
        "citation_accuracy": round(citation_pct, 1) if citation_pct is not None else None,
        "no_answer_rate": round(no_answer_pct, 1),
        "avg_keyword_coverage": round(avg_kw, 1),
        "results": results,
    }

    print(f"""
╔══════════════════════════════════════╗
║          EVAL RESULTS                ║
╠══════════════════════════════════════╣
║  Recall@5:              {recall_pct:5.1f}%          ║
║  Citation accuracy:     {(citation_pct or 0):5.1f}%          ║
║  No-answer rate:        {no_answer_pct:5.1f}%          ║
║  Avg keyword coverage:  {avg_kw:5.1f}%          ║
║  Queries run:           {n:3d}            ║
╚══════════════════════════════════════╝
""")

    return summary


def save_results(summary: dict) -> Path:
    RESULTS_DIR.mkdir(exist_ok=True)
    out_path = RESULTS_DIR / "latest.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Results saved to {out_path}")
    return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save", action="store_true", help="Save results to evals/results/latest.json")
    parser.add_argument("--no-llm", action="store_true", help="Skip LLM calls (retrieval metrics only)")
    args = parser.parse_args()

    summary = run_evals(use_llm=not args.no_llm)
    if args.save:
        save_results(summary)
