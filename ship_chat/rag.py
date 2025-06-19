from __future__ import annotations

import ollama
from ship_chat.vector import retrieve


def _build_messages(query: str, context: str) -> list[dict]:
    return [
        {"role": "system",
         "content": "You are a helpful assistant. Use the context below and show your reasoning step-by-step."},
        {"role": "system", "content": context},
        {"role": "user",   "content": query},
    ]


def stream_answer(query: str, col, cfg: dict):
    """Yield response chunks in real time."""
    context = retrieve(col, query, cfg["retriever"]["top_k"])
    stream = ollama.chat(
        model   = cfg["llm"]["model"],
        messages= _build_messages(query, context),
        stream  = True,
        options = {"temperature": cfg["llm"].get("temperature", 0.7)},
    )
    for chunk in stream:
        yield chunk["message"]["content"]


# Non-streaming (fallback / tests)

def answer(query: str, col, cfg: dict) -> str:
    """RAG response using the local Ollama model."""
    # 1) context
    context = retrieve(col, query, cfg["retriever"]["top_k"])

    # 2) prompt
    msgs = [
        {"role": "system",
         "content": "You are a helpful assistant. Use the context below and show your reasoning step-by-step."},
        {"role": "system", "content": context},
        {"role": "user",   "content": query},
    ]

    # 3) chat
    resp = ollama.chat(
        model   = cfg["llm"]["model"],
        messages= msgs,
        options = {"temperature": cfg["llm"].get("temperature", 0.7)},
    )

    return resp["message"]["content"]