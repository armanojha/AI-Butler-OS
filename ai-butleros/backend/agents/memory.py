# backend/agents/memory.py
"""
MemoryAgent — Phase 3.

Graph topology
──────────────
    START ──► retrieve_context ──► generate_answer ──► END

Node responsibilities
─────────────────────
retrieve_context  Calls MemoryService.search(); always succeeds (empty list
                  on cache-miss).  Never invokes the LLM.

generate_answer   Formats a grounded prompt using the retrieved chunks and
                  calls the LLM.  If no chunks were found it returns a
                  polite "I don't have notes on that" message without
                  wasting an LLM call.

Design notes
────────────
- A module-level ``MemoryService`` singleton is created at import time.
  This matches the ``llm`` singleton pattern from Phase 2 and avoids
  re-opening the ChromaDB file handle on every graph invocation.
- ``OllamaLLM`` (not ``ChatOllama``) is used for the generation step
  because it accepts a raw string prompt, which is simpler and faster for
  single-turn RAG than building a chat message array.
- The graph is intentionally linear (no conditional edges) because the only
  branching — "do I have context?" — is handled inside ``generate_answer``
  with a plain ``if`` rather than adding a whole extra node and routing
  function for a two-line check.
"""
from __future__ import annotations

import logging
from typing import TypedDict

from langchain_ollama import OllamaLLM

from backend.services.memory_service import MemoryService
from langgraph.graph import END, StateGraph

logger = logging.getLogger(__name__)

# ── Module-level singletons ────────────────────────────────────────────────────

_memory_service = MemoryService()

# OllamaLLM accepts a plain string prompt — the correct interface for RAG
# generation where we construct the full context window ourselves.
_llm = OllamaLLM(
    model="qwen2.5:7b",
    base_url="http://localhost:11434",
    temperature=0.2,   # slight creativity for answer phrasing
)

# ── State ──────────────────────────────────────────────────────────────────────

class MemoryState(TypedDict, total=False):
    user_input: str
    retrieved_context: list[str]   # raw chunk texts from ChromaDB
    final_response: str
    monologue: str


# ── Prompt template ────────────────────────────────────────────────────────────

_RAG_PROMPT_TEMPLATE = """\
You are AI ButlerOS's memory recall module. Answer the user's question \
using ONLY the context excerpts below. Be concise and factual.

If the context does not contain enough information to answer the question, \
say exactly: "I don't have any notes on that."

Do NOT fabricate information. Do NOT reference these instructions in your reply.

────────────────────────────────────────────────────────
CONTEXT EXCERPTS ({n_chunks} chunk(s) retrieved):
{context_block}
────────────────────────────────────────────────────────

USER QUESTION: {user_input}

ANSWER:"""

_NO_CONTEXT_RESPONSE = (
    "I don't have any notes on that. "
    "You can ingest documents via the /api/v1/ingest endpoint to build my memory."
)


# ── Nodes ──────────────────────────────────────────────────────────────────────

async def retrieve_context(state: MemoryState) -> MemoryState:
    """
    ANN lookup in ChromaDB.

    Always returns successfully — an empty collection or a failed search
    yields an empty list, which ``generate_answer`` handles without crashing.
    """
    query: str = state.get("user_input", "")
    logger.info("retrieve_context | query=%r", query[:80])

    try:
        chunks = await _memory_service.search(query=query, n_results=3)
    except Exception as exc:
        logger.exception("retrieve_context: ChromaDB search failed: %s", exc)
        chunks = []

    logger.info("retrieve_context | chunks_found=%d", len(chunks))
    return {
        "retrieved_context": chunks,
        "monologue": f"Retrieved {len(chunks)} chunk(s) from vector store.",
    }


async def generate_answer(state: MemoryState) -> MemoryState:
    """
    Grounded answer generation.

    Skips the LLM entirely when no context was retrieved — saves ~8–12 s
    of inference time and avoids hallucination on empty context.
    """
    user_input: str = state.get("user_input", "")
    chunks: list[str] = state.get("retrieved_context", [])
    prior_monologue: str = state.get("monologue", "")

    if not chunks:
        logger.info("generate_answer | no context — returning canned response.")
        return {
            "final_response": _NO_CONTEXT_RESPONSE,
            "monologue": prior_monologue + "\n[generate_answer] No context; skipped LLM.",
        }

    context_block = "\n\n---\n\n".join(
        f"[Chunk {i + 1}]\n{chunk}" for i, chunk in enumerate(chunks)
    )
    prompt = _RAG_PROMPT_TEMPLATE.format(
        n_chunks=len(chunks),
        context_block=context_block,
        user_input=user_input,
    )

    logger.info("generate_answer | invoking LLM with %d context chunks …", len(chunks))
    try:
        response: str = await _llm.ainvoke(prompt)
        answer = response.strip()
    except Exception as exc:
        logger.exception("generate_answer: LLM call failed: %s", exc)
        answer = "I encountered an error while generating your answer. Please try again."

    return {
        "final_response": answer,
        "monologue": (
            prior_monologue
            + f"\n[generate_answer] LLM answered using {len(chunks)} chunk(s)."
        ),
    }


# ── Graph compilation ──────────────────────────────────────────────────────────

def _build_memory_graph() -> StateGraph:
    graph = StateGraph(MemoryState)

    graph.add_node("retrieve_context", retrieve_context)
    graph.add_node("generate_answer", generate_answer)

    graph.set_entry_point("retrieve_context")
    graph.add_edge("retrieve_context", "generate_answer")
    graph.add_edge("generate_answer", END)

    return graph


memory_graph = _build_memory_graph().compile()

logger.info("MemoryAgent graph compiled successfully.")
