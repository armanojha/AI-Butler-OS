# backend/agents/llm.py
"""
Central LLM factory.

A single `ChatOllama` instance is constructed at import time and reused
across all agents.  `temperature=0.1` keeps routing decisions close to
deterministic while leaving a tiny window for the model to recover from
awkward phrasing rather than hard-failing.

If the Ollama daemon is unreachable at startup the import will still
succeed; the error surfaces at first invocation, which is the correct
behaviour for a lifespan-managed service (don't block startup for an
optional local daemon).
"""
from __future__ import annotations

import logging

from langchain_ollama import ChatOllama

logger = logging.getLogger(__name__)

_OLLAMA_BASE_URL = "http://localhost:11434"
_MODEL_ID = "qwen2.5:7b"

logger.info("Initialising ChatOllama | model=%s base_url=%s", _MODEL_ID, _OLLAMA_BASE_URL)

llm: ChatOllama = ChatOllama(
    model=_MODEL_ID,
    base_url=_OLLAMA_BASE_URL,
    temperature=0.1,
    # Keep the context window wide enough for multi-turn traces later.
    num_ctx=8192,
    # Disable streaming for structured-output nodes; we need the full
    # response before we can parse the JSON.
    streaming=False,
)
