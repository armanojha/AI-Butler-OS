# backend/agents/llm.py
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
    num_ctx=2048,
    streaming=False,
)
