"""LawBot — Citation-Grounded RAG for US Legal Documents."""

import streamlit as st
from rag.retriever import search
from rag.llm import answer

st.set_page_config(
    page_title="LawBot",
    page_icon="⚖️",
    layout="centered",
    initial_sidebar_state="collapsed",
)

st.title("⚖️ LawBot")
st.caption(
    "Citation-grounded Q&A over the US Constitution, Amendments, and landmark Supreme Court cases. "
    "Answers include inline citations. Responses are limited to the indexed corpus."
)

SUGGESTED_QUERIES = [
    "What does the 4th Amendment protect?",
    "How is the President elected?",
    "What are the powers of Congress?",
    "What did Miranda v. Arizona establish?",
    "What is the Supremacy Clause?",
    "Can the government restrict freedom of speech?",
]

if "messages" not in st.session_state:
    st.session_state.messages = []

if not st.session_state.messages:
    st.markdown("**Try asking:**")
    cols = st.columns(3)
    for i, q in enumerate(SUGGESTED_QUERIES):
        if cols[i % 3].button(q, key=f"suggest_{i}", use_container_width=True):
            st.session_state.messages.append({"role": "user", "content": q})
            st.rerun()

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("chunks"):
            with st.expander(f"Retrieved {len(msg['chunks'])} source chunks"):
                for c in msg["chunks"]:
                    st.markdown(f"**[{c['doc']}]** — similarity: `{c['score']:.3f}`")
                    st.markdown(f"> {c['text'][:400]}{'...' if len(c['text']) > 400 else ''}")
                    st.divider()

if prompt := st.chat_input("Ask a legal question…"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        chunks = search(prompt)

        if not chunks:
            response_text = (
                "I don't have enough information in my corpus to answer this question. "
                "My knowledge is limited to the US Constitution, its Amendments, and "
                "the landmark Supreme Court cases in my index."
            )
            st.markdown(response_text)
        else:
            response_text = st.write_stream(answer(prompt, chunks, stream=True))

            with st.expander(f"Retrieved {len(chunks)} source chunks"):
                for c in chunks:
                    st.markdown(f"**[{c['doc']}]** — similarity: `{c['score']:.3f}`")
                    st.markdown(f"> {c['text'][:400]}{'...' if len(c['text']) > 400 else ''}")
                    st.divider()

    st.session_state.messages.append({
        "role": "assistant",
        "content": response_text,
        "chunks": chunks,
    })

with st.sidebar:
    st.markdown("### About")
    st.markdown(
        "LawBot uses **sentence-transformers/all-MiniLM-L6-v2** for dense retrieval "
        "and **GPT-4o-mini** with citation-grounded prompting. "
        "Similarity threshold: 0.35 (queries below this return no result)."
    )
    st.markdown("**Corpus**")
    st.markdown("- US Constitution (Articles I–VII)")
    st.markdown("- Amendments I–XXVII")
    st.markdown("- 18 landmark Supreme Court cases")
    st.divider()
    st.markdown("[View eval results →](Evaluation)", unsafe_allow_html=False)
    if st.button("Clear chat history"):
        st.session_state.messages = []
        st.rerun()
