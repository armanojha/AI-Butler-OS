# backend/agents/state.py
"""
Shared state contract for every node in the PlannerAgent graph.

Rules
-----
- All fields have explicit defaults so any node can be the entry point
  without needing to pre-populate the whole dict.
- `retry_count` is the single source of truth for loop-break logic; no
  node should implement its own counter.
- `final_response` is written exclusively by terminal nodes
  (`handle_chat`, `stub_execution`) and is treated as immutable once set.
"""
from __future__ import annotations

from typing import Any, TypedDict


class AgentState(TypedDict, total=False):
    # ── Input ──────────────────────────────────────────────────────────────
    user_input: str

    # ── Routing ────────────────────────────────────────────────────────────
    # One of: SCHEDULE | MEMORY_SEARCH | CHAT | UNKNOWN
    intent: str

    # ── Structured parameters extracted by the LLM ─────────────────────────
    # e.g. {"datetime_raw": "tomorrow at 9am", "task": "review PR"}
    parameters: dict[str, Any]

    # ── LLM chain-of-thought text (persisted to ExecutionTrace.monologue) ──
    monologue: str

    # ── Loop guard ─────────────────────────────────────────────────────────
    retry_count: int

    # Scratch space for passing raw LLM text from analyze_intent to validate_schema.
    _raw_llm_output: str | None

    # ── Terminal output ────────────────────────────────────────────────────
    final_response: str
