# backend/agents/planner.py
"""
PlannerAgent — Phase 2.

Graph topology
──────────────
                    ┌──────────────────────┐
         START ───► │   analyze_intent     │ ◄─────────────────┐
                    └──────────┬───────────┘                   │
                               │ route_intent()                │
                    ┌──────────▼────────────────────────────┐  │
                    │                                       │  │
              ┌─────▼──────┐  ┌───────────────┐  ┌────────▼──┤
              │ handle_chat│  │stub_execution │  │VALIDATION │
              │  (CHAT)    │  │(SCHEDULE /    │  │  RETRY    │
              └─────┬──────┘  │MEMORY_SEARCH/ │  └───────────┘
                    │         │  UNKNOWN)     │
                    └────┬────┘               │
                         └────────► END ◄─────┘

Retry policy
────────────
If `validate_schema` cannot parse the LLM response, it increments
`retry_count` and routes back to `analyze_intent`.  After
`_MAX_RETRIES` consecutive failures it short-circuits to
`stub_execution` with intent=UNKNOWN so the user always gets a reply.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any, Literal

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph
from pydantic import BaseModel, ValidationError, field_validator

from backend.agents.llm import llm
from backend.agents.state import AgentState

logger = logging.getLogger(__name__)

_MAX_RETRIES: int = 2

# ── Pydantic contract for LLM output ──────────────────────────────────────────

_VALID_INTENTS = {"SCHEDULE", "MEMORY_SEARCH", "CHAT", "UNKNOWN"}


class PlannerOutput(BaseModel):
    """
    Strict schema the LLM must conform to.

    The validator normalises intent to uppercase so minor model drift
    (e.g. "schedule" vs "SCHEDULE") doesn't trigger an unnecessary retry.
    """

    intent: str
    parameters: dict[str, Any] = {}
    reasoning: str = ""

    @field_validator("intent", mode="before")
    @classmethod
    def normalise_intent(cls, v: str) -> str:
        upper = str(v).strip().upper()
        if upper not in _VALID_INTENTS:
            raise ValueError(
                f"intent must be one of {_VALID_INTENTS}, got {v!r}"
            )
        return upper


# ── System prompt ──────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """\
You are AI ButlerOS's routing core. Your sole job is to classify the \
user's message and extract structured parameters from it.

Respond with ONLY a single valid JSON object — no markdown fences, no \
preamble, no trailing text. The object must have exactly these keys:

{
  "intent":     "<SCHEDULE | MEMORY_SEARCH | CHAT | UNKNOWN>",
  "parameters": { /* extracted key/value pairs relevant to the intent */ },
  "reasoning":  "<one sentence explaining your classification>"
}

Intent definitions
──────────────────
SCHEDULE      – The user wants to create, update, delete, or query a \
calendar event or reminder.  Extract: datetime_raw, task, recurrence.
MEMORY_SEARCH – The user wants to retrieve or search stored knowledge, \
notes, or past conversations.  Extract: query, filters.
CHAT          – General conversation, questions, or anything that \
doesn't fit the above.  parameters may be empty.
UNKNOWN       – You genuinely cannot classify the message.

Rules
─────
- Never output anything except the JSON object.
- If you are uncertain between two intents, pick the most specific one.
- parameters values must be strings or simple scalar types — no nested objects.
"""


# ── Helpers ────────────────────────────────────────────────────────────────────

def _extract_json(raw: str) -> dict[str, Any]:
    """
    Attempt to extract a JSON object from `raw`.

    Strategy (in order):
    1. Direct parse — model behaved.
    2. Strip markdown fences (```json … ```) — common Qwen habit.
    3. Regex to find the first ``{…}`` block — last resort.
    """
    text = raw.strip()

    # 1. Direct
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # 2. Strip fences
    fenced = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.DOTALL).strip()
    try:
        return json.loads(fenced)
    except json.JSONDecodeError:
        pass

    # 3. First brace block
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    raise ValueError(f"No valid JSON found in LLM response: {text[:200]!r}")


# ── Node: analyze_intent ───────────────────────────────────────────────────────

async def analyze_intent(state: AgentState) -> AgentState:
    """
    Call the local LLM and populate intent / parameters / monologue.

    Returns a *partial* state update; LangGraph merges it with the
    existing state so we only need to set the keys we touch.
    """
    user_input: str = state.get("user_input", "")
    retry: int = state.get("retry_count", 0)

    logger.info("analyze_intent | retry=%d | input=%r", retry, user_input[:80])

    messages = [
        SystemMessage(content=_SYSTEM_PROMPT),
        HumanMessage(content=user_input),
    ]

    try:
        response = await llm.ainvoke(messages)
        raw_content: str = response.content  # type: ignore[attr-defined]
    except Exception as exc:
        logger.exception("Ollama invocation failed: %s", exc)
        # Synthesise a safe fallback payload so validate_schema can proceed
        # and route to UNKNOWN rather than crashing the graph.
        raw_content = json.dumps(
            {
                "intent": "UNKNOWN",
                "parameters": {},
                "reasoning": f"LLM call failed: {exc}",
            }
        )

    return {
        # Stash raw text so validate_schema can parse it.
        "_raw_llm_output": raw_content,  # type: ignore[typeddict-unknown-key]
        "retry_count": retry,
    }


# ── Node: validate_schema ──────────────────────────────────────────────────────

async def validate_schema(state: AgentState) -> AgentState:
    """
    Pure-Python guard node.

    Parses the raw LLM text into `PlannerOutput`.  On failure, increments
    `retry_count`.  On success, promotes intent / parameters / monologue
    into the canonical state fields and clears the scratch key.
    """
    raw: str = state.get("_raw_llm_output", "")  # type: ignore[typeddict-item]
    retry: int = state.get("retry_count", 0)

    try:
        payload = _extract_json(raw)
        parsed = PlannerOutput.model_validate(payload)
    except (ValueError, ValidationError) as exc:
        logger.warning(
            "validate_schema failed (retry=%d): %s | raw=%r",
            retry,
            exc,
            raw[:200],
        )
        return {
            "retry_count": retry + 1,
            "monologue": f"[validation_error retry={retry}] {exc}",
        }

    logger.info(
        "validate_schema OK | intent=%s params=%s",
        parsed.intent,
        list(parsed.parameters.keys()),
    )
    return {
        "intent": parsed.intent,
        "parameters": parsed.parameters,
        "monologue": parsed.reasoning,
        "retry_count": retry,
        "_raw_llm_output": None,  # type: ignore[typeddict-unknown-key]  # clear scratch
    }


# ── Node: handle_chat ──────────────────────────────────────────────────────────

async def handle_chat(state: AgentState) -> AgentState:
    """
    Terminal node for CHAT intent.

    In Phase 2 this is a lightweight conversational echo.  Phase 3 will
    wire in a full dialogue chain here.
    """
    user_input: str = state.get("user_input", "")
    monologue: str = state.get("monologue", "")

    logger.info("handle_chat | input=%r", user_input[:80])

    messages = [
        SystemMessage(
            content=(
                "You are AI ButlerOS, a helpful personal assistant. "
                "Reply concisely and conversationally."
            )
        ),
        HumanMessage(content=user_input),
    ]

    try:
        response = await llm.ainvoke(messages)
        reply: str = response.content.strip()  # type: ignore[attr-defined]
    except Exception as exc:
        logger.exception("handle_chat LLM call failed: %s", exc)
        reply = "I'm sorry, I couldn't generate a response right now."

    return {
        "final_response": reply,
        "monologue": monologue + f"\n[handle_chat] Generated conversational reply.",
    }


# ── Node: stub_execution ───────────────────────────────────────────────────────

async def stub_execution(state: AgentState) -> AgentState:
    """
    Temporary terminal node for SCHEDULE / MEMORY_SEARCH / UNKNOWN.

    Phase 3 will replace this with real hand-offs to SchedulerAgent and
    MemoryAgent respectively.
    """
    intent: str = state.get("intent", "UNKNOWN")
    params: dict[str, Any] = state.get("parameters", {})
    monologue: str = state.get("monologue", "")

    logger.info("stub_execution | intent=%s params=%s", intent, list(params.keys()))

    stub_messages = {
        "SCHEDULE": (
            f"[STUB] SchedulerAgent will handle: {params}. "
            "Calendar integration coming in Phase 3."
        ),
        "MEMORY_SEARCH": (
            f"[STUB] MemoryAgent will search for: {params.get('query', '...')}. "
            "RAG pipeline coming in Phase 3."
        ),
        "UNKNOWN": (
            "I wasn't able to classify your request. "
            "Could you rephrase it?"
        ),
    }

    return {
        "final_response": stub_messages.get(intent, stub_messages["UNKNOWN"]),
        "monologue": monologue + f"\n[stub_execution] Routed intent={intent}.",
    }


# ── Edge: route_intent ─────────────────────────────────────────────────────────

def route_intent(
    state: AgentState,
) -> Literal["handle_chat", "stub_execution", "analyze_intent"]:
    """
    Conditional edge function — called after `validate_schema`.

    Routing table
    ─────────────
    CHAT            → handle_chat
    SCHEDULE        → stub_execution   (Phase 3: SchedulerAgent)
    MEMORY_SEARCH   → stub_execution   (Phase 3: MemoryAgent)
    UNKNOWN         → stub_execution
    validation fail → analyze_intent   (retry loop, capped by _MAX_RETRIES)
    """
    intent: str = state.get("intent", "")
    retry: int = state.get("retry_count", 0)

    # Intent not yet set → validation failed on this pass.
    if not intent:
        if retry >= _MAX_RETRIES:
            logger.warning(
                "Max retries (%d) reached without valid intent; forcing UNKNOWN.",
                _MAX_RETRIES,
            )
            # Mutate state in-place so stub_execution has something to read.
            # (LangGraph allows partial mutations inside edge functions.)
            state["intent"] = "UNKNOWN"
            state["parameters"] = {}
            return "stub_execution"
        logger.info("Routing back to analyze_intent (retry %d/%d).", retry, _MAX_RETRIES)
        return "analyze_intent"

    if intent == "CHAT":
        return "handle_chat"

    return "stub_execution"


# ── Graph compilation ──────────────────────────────────────────────────────────

def _build_planner_graph() -> StateGraph:
    graph = StateGraph(AgentState)

    graph.add_node("analyze_intent", analyze_intent)
    graph.add_node("validate_schema", validate_schema)
    graph.add_node("handle_chat", handle_chat)
    graph.add_node("stub_execution", stub_execution)

    # Entry point
    graph.set_entry_point("analyze_intent")

    # analyze_intent always flows into validate_schema
    graph.add_edge("analyze_intent", "validate_schema")

    # validate_schema fans out conditionally
    graph.add_conditional_edges(
        "validate_schema",
        route_intent,
        {
            "analyze_intent": "analyze_intent",
            "handle_chat": "handle_chat",
            "stub_execution": "stub_execution",
        },
    )

    # Both terminal nodes connect to END
    graph.add_edge("handle_chat", END)
    graph.add_edge("stub_execution", END)

    return graph


_graph = _build_planner_graph()
planner_graph = _graph.compile()

logger.info("PlannerAgent graph compiled successfully.")
