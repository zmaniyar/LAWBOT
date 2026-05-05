"""Citation-grounded answer generation using GPT-4o-mini."""

import os
from openai import OpenAI

_client = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        api_key = os.environ.get("OPENAI_API_KEY") or _get_streamlit_secret()
        _client = OpenAI(api_key=api_key)
    return _client


def _get_streamlit_secret() -> str:
    try:
        import streamlit as st
        return st.secrets["OPENAI_API_KEY"]
    except Exception:
        raise ValueError(
            "OPENAI_API_KEY not found. Set it as an environment variable or in .streamlit/secrets.toml"
        )


SYSTEM_PROMPT = """\
You are LawBot, a legal research assistant that answers questions about the US Constitution, \
its Amendments, and landmark Supreme Court cases.

Rules you must follow:
1. Answer ONLY using the provided context chunks. Do not use outside knowledge.
2. After each factual claim, cite the source using [Doc N — Title] notation.
3. If the context does not contain enough information to answer, respond with exactly:
   "I don't have enough information in my corpus to answer this question."
4. Do not speculate, infer, or fill gaps with general legal knowledge.
5. Keep answers concise and accurate.
"""


def answer(question: str, chunks: list[dict], stream: bool = False):
    """
    Generate a citation-grounded answer from retrieved chunks.

    Args:
        question: User's question.
        chunks: Retrieved chunks from retriever.search().
        stream: If True, returns a streaming generator; otherwise returns full string.
    """
    if not chunks:
        no_info = "I don't have enough information in my corpus to answer this question."
        if stream:
            def _no_info_gen():
                yield no_info
            return _no_info_gen()
        return no_info

    context_blocks = "\n\n".join(
        f"[Doc {i+1} — {c['doc']}]\n{c['text']}"
        for i, c in enumerate(chunks)
    )

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"CONTEXT:\n{context_blocks}\n\nQUESTION: {question}\n\nANSWER (with citations):",
        },
    ]

    client = _get_client()
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        max_tokens=600,
        temperature=0.1,
        stream=stream,
    )

    if stream:
        def _stream_gen():
            for chunk in response:
                delta = chunk.choices[0].delta.content
                if delta:
                    yield delta
        return _stream_gen()

    return response.choices[0].message.content.strip()
