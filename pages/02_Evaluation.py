"""Eval dashboard — shows retrieval and citation metrics with per-query breakdown."""

import json
import streamlit as st
from pathlib import Path

RESULTS_PATH = Path(__file__).parent.parent / "evals" / "results" / "latest.json"

st.set_page_config(page_title="LawBot — Evaluation", page_icon="📊", layout="wide")
st.title("📊 LawBot Evaluation Dashboard")
st.caption("Retrieval recall@5, citation accuracy, and keyword coverage across 40 hand-written legal queries.")

if not RESULTS_PATH.exists():
    st.warning(
        "No eval results found. Run the eval harness first:\n\n"
        "```bash\npython -m evals.run_evals --save\n```"
    )
    st.stop()

with open(RESULTS_PATH) as f:
    data = json.load(f)

summary = {k: v for k, v in data.items() if k != "results"}
results = data["results"]

# ── Summary metrics ──────────────────────────────────────────────────────────
st.subheader("Summary Metrics")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Recall@5", f"{summary['recall_at_5']}%", help="Was the expected document in the top-5 retrieved chunks?")
c2.metric("Citation Accuracy", f"{summary.get('citation_accuracy', 'N/A')}%", help="Did the LLM cite the correct source document?")
c3.metric("No-Answer Rate", f"{summary['no_answer_rate']}%", help="Fraction of queries where no chunk passed the similarity threshold.")
c4.metric("Avg Keyword Coverage", f"{summary['avg_keyword_coverage']}%", help="Average fraction of expected keywords present in the LLM answer.")

st.caption(f"Eval run: {summary.get('timestamp', 'unknown')} · {summary['n_queries']} queries")
st.divider()

# ── Per-category breakdown ────────────────────────────────────────────────────
st.subheader("Recall@5 by Category")
categories = {}
for r in results:
    cat = r.get("category", "other")
    categories.setdefault(cat, {"total": 0, "recalled": 0})
    categories[cat]["total"] += 1
    categories[cat]["recalled"] += int(r["recalled"])

cat_data = {
    cat: f"{v['recalled']}/{v['total']} ({v['recalled']/v['total']*100:.0f}%)"
    for cat, v in sorted(categories.items())
}
cols = st.columns(len(cat_data))
for col, (cat, val) in zip(cols, cat_data.items()):
    col.metric(cat.replace("_", " ").title(), val)

st.divider()

# ── Per-query table ───────────────────────────────────────────────────────────
st.subheader("Per-Query Results")

category_filter = st.selectbox(
    "Filter by category",
    ["all"] + sorted(set(r.get("category", "other") for r in results)),
)
show_failures_only = st.checkbox("Show failed retrievals only")

filtered = [
    r for r in results
    if (category_filter == "all" or r.get("category") == category_filter)
    and (not show_failures_only or not r["recalled"])
]

for r in filtered:
    recall_icon = "✅" if r["recalled"] else "❌"
    cite_icon = "✅" if r.get("cited") else ("➖" if r.get("cited") is None else "❌")
    with st.expander(f"{recall_icon} [{r['id']}] {r['query']}"):
        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Recalled", "Yes" if r["recalled"] else "No")
        col_b.metric("Cited", "Yes" if r.get("cited") else ("N/A" if r.get("cited") is None else "No"))
        col_c.metric("Keyword coverage", f"{r['keyword_score']*100:.0f}%" if r.get("keyword_score") is not None else "N/A")

        st.markdown(f"**Expected doc:** `{r['expected_doc']}`")

        if r.get("top_chunks"):
            st.markdown("**Top retrieved chunks:**")
            for chunk in r["top_chunks"]:
                st.markdown(f"- `{chunk['doc']}` — score: `{chunk['score']:.3f}`")

        if r.get("response") and r["response"] not in ("[NO CHUNKS RETRIEVED]", "[LLM SKIPPED]"):
            st.markdown("**LLM Response:**")
            st.markdown(r["response"])
