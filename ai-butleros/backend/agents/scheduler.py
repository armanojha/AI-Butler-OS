# backend/agents/scheduler.py
"""
SchedulerAgent — Phase 4.

Graph topology
──────────────
    START ──► extract_details ──► temporal_parse ──► generate_confirmation ──► END

Node responsibilities
─────────────────────
extract_details       LLM pulls {task, time_string} from free-form user input.
                      Returns structured JSON; markdown-fence-safe extraction
                      mirrors the PlannerAgent's _extract_json pattern.

temporal_parse        PURE PYTHON — no LLM.  Uses ``dateparser`` to resolve
                      relative expressions ("next Tuesday at 5pm") against the
                      real system clock.  Emits an ISO-8601 string or the
                      sentinel "INVALID_DATE" so downstream nodes never have
                      to None-check.

generate_confirmation LLM formats a user-facing confirmation (or clarification
                      request on INVALID_DATE) using the strict timestamp.

Design rationale
────────────────
The Neuro-Symbolic split is the key insight of Phase 4:

  LLM  ──► "the user means next Tuesday at 17:00"   (semantic understanding)
  Python ──► datetime(2025, 7, 1, 17, 0, tzinfo=…)  (mathematical correctness)
  LLM  ──► "Got it! I'll remind you on Tuesday…"    (natural language output)

Letting the LLM calculate dates is a known failure mode — it hallucinates
offsets because it has no access to `datetime.now()`.  ``dateparser`` has
multilingual, timezone-aware relative-date resolution built in.
"""
from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone
from typing import Any, TypedDict

import dateparser
from langchain_ollama import OllamaLLM

logger = logging.getLogger(__name__)

# Deferred import to keep the langgraph import at module level while
# avoiding a circular dependency with the llm singleton.
from langgraph.graph import END, StateGraph  # noqa: E402

# ── LLM singleton (raw string interface) ──────────────────────────────────────

_llm = OllamaLLM(
    model="qwen2.5:7b",
    base_url="http://localhost:11434",
    temperature=0.1,   # near-deterministic for extraction
)

# ── State ──────────────────────────────────────────────────────────────────────

class SchedulerState(TypedDict, total=False):
    user_input: str
    raw_time_string: str          # LLM-extracted, e.g. "next Tuesday at 5pm"
    task_description: str         # LLM-extracted, e.g. "Study Group"
    parsed_timestamp: str         # ISO-8601 or "INVALID_DATE"
    final_response: str
    monologue: str


# ── JSON extraction helper (mirrors planner.py) ────────────────────────────────

def _extract_json(raw: str) -> dict[str, Any]:
    text = raw.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    fenced = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.DOTALL).strip()
    try:
        return json.loads(fenced)
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    raise ValueError(f"No valid JSON found in LLM output: {text[:200]!r}")


# ── Prompts ────────────────────────────────────────────────────────────────────

_EXTRACT_PROMPT = """\
You are AI ButlerOS's scheduling parser. Extract the scheduling details \
from the user's message.

Output ONLY a single valid JSON object with exactly these two keys:

{{
  "task": "<a short description of what needs to be scheduled>",
  "time_string": "<the exact time expression from the user's message, \
verbatim if possible>"
}}

Rules:
- No markdown fences, no preamble, no trailing text — JSON only.
- If no time expression is present, set "time_string" to "unspecified".
- Preserve the user's exact phrasing for time_string (e.g. "next Tuesday at 5pm").

USER MESSAGE: {user_input}
"""

_CONFIRMATION_PROMPT_SUCCESS = """\
You are AI ButlerOS, a helpful personal assistant. \
Compose a short, friendly confirmation message for the user.

Task:      {task}
Scheduled: {timestamp}   (this is a precise, verified ISO-8601 datetime)

Rules:
- Convert the ISO timestamp to a natural, readable format in your reply \
  (e.g. "Tuesday, 1 July 2025 at 5:00 PM").
- Do NOT mention ISO-8601 or raw timestamps to the user.
- Keep it to 1–2 sentences.
- End with "Is there anything else you'd like to add?"
"""

_CONFIRMATION_PROMPT_INVALID = """\
You are AI ButlerOS, a helpful personal assistant.
The user asked to schedule: "{task}"
Unfortunately, the time expression "{time_string}" could not be parsed \
into a concrete date.

Write a short, friendly message asking the user to clarify the date and time. \
Suggest they use a format like "Monday 14 July at 3pm" or \
"tomorrow at 10am". Keep it to 2 sentences.
"""


# ── Nodes ──────────────────────────────────────────────────────────────────────

async def extract_details(state: SchedulerState) -> SchedulerState:
    """
    LLM node — semantic extraction only, no date arithmetic.

    On parse failure, synthesises safe defaults so ``temporal_parse``
    always receives strings rather than None.
    """
    user_input: str = state.get("user_input", "")
    logger.info("extract_details | input=%r", user_input[:80])

    prompt = f"""You are a temporal data extractor.
Analyze this user input: "{user_input}"

Your job is to extract the task and the time string.
CRITICAL INSTRUCTION: You must fix any typos or slang in the time string before outputting it. 
For example, if the user says 'tommorow', output 'tomorrow'. 
If they use weird formatting, normalize it into clean English (e.g., 'tomorrow at 16:00').

Output strictly valid JSON with NO markdown formatting:
{{
    "task": "...",
    "time_string": "..."
}}"""

    try:
        raw: str = await _llm.ainvoke(prompt)
        payload = _extract_json(raw)
        task: str = str(payload.get("task", "Unnamed task")).strip()
        time_string: str = str(payload.get("time_string", "unspecified")).strip()
    except Exception as exc:
        logger.exception("extract_details: extraction failed: %s", exc)
        task = "Unnamed task"
        time_string = "unspecified"

    logger.info(
        "extract_details | task=%r time_string=%r", task, time_string
    )
    return {
        "task_description": task,
        "raw_time_string": time_string,
        "monologue": f"[extract_details] task={task!r} time_string={time_string!r}",
    }


async def temporal_parse(state: SchedulerState) -> SchedulerState:
    """
    Pure Python node — the LLM is never called here.

    ``dateparser.parse`` understands hundreds of natural-language patterns
    in multiple languages and resolves relative expressions against
    ``datetime.now()`` with correct timezone handling.

    RETURN_TIME_AS_PERIOD=False forces a concrete time even when only a
    date is given (defaults to midnight).
    """
    time_string: str = state.get("raw_time_string", "unspecified")
    prior_monologue: str = state.get("monologue", "")

    if time_string == "unspecified":
        logger.info("temporal_parse | time_string=unspecified → INVALID_DATE")
        return {
            "parsed_timestamp": "INVALID_DATE",
            "monologue": prior_monologue + "\n[temporal_parse] No time expression found.",
        }

    settings: dict[str, Any] = {
        "PREFER_DATES_FROM": "future",
        "RETURN_TIME_AS_PERIOD": False,
        "TIMEZONE": "UTC",
        "RETURN_AS_TIMEZONE_AWARE": True,
    }

    parsed: datetime | None = dateparser.parse(time_string, settings=settings)

    if parsed is None:
        logger.warning("temporal_parse | dateparser returned None for %r", time_string)
        return {
            "parsed_timestamp": "INVALID_DATE",
            "monologue": (
                prior_monologue
                + f"\n[temporal_parse] dateparser could not resolve {time_string!r}."
            ),
        }

    # Normalise to UTC ISO-8601 with explicit Z suffix.
    utc_dt = parsed.astimezone(timezone.utc)
    iso_ts = utc_dt.strftime("%Y-%m-%dT%H:%M:%SZ")

    logger.info("temporal_parse | %r → %s", time_string, iso_ts)
    return {
        "parsed_timestamp": iso_ts,
        "monologue": (
            prior_monologue
            + f"\n[temporal_parse] {time_string!r} → {iso_ts} (UTC)."
        ),
    }


async def generate_confirmation(state: SchedulerState) -> SchedulerState:
    """
    LLM node — natural-language output only, no logic.

    Receives a mathematically correct ISO-8601 timestamp (or INVALID_DATE)
    and turns it into a user-facing sentence.
    """
    task: str = state.get("task_description", "Unnamed task")
    timestamp: str = state.get("parsed_timestamp", "INVALID_DATE")
    time_string: str = state.get("raw_time_string", "")
    prior_monologue: str = state.get("monologue", "")

    if timestamp == "INVALID_DATE":
        prompt = _CONFIRMATION_PROMPT_INVALID.format(
            task=task, time_string=time_string
        )
        log_tag = "INVALID_DATE path"
    else:
        prompt = _CONFIRMATION_PROMPT_SUCCESS.format(
            task=task, timestamp=timestamp
        )
        log_tag = f"success path ts={timestamp}"

    logger.info("generate_confirmation | %s", log_tag)

    try:
        response: str = await _llm.ainvoke(prompt)
        reply = response.strip()
    except Exception as exc:
        logger.exception("generate_confirmation: LLM call failed: %s", exc)
        reply = (
            f"I've noted your task '{task}'"
            + (f" for {timestamp}." if timestamp != "INVALID_DATE" else ".")
        )

    return {
        "final_response": reply,
        "monologue": (
            prior_monologue
            + f"\n[generate_confirmation] {log_tag}. Reply generated."
        ),
    }


# ── Graph compilation ──────────────────────────────────────────────────────────

def _build_scheduler_graph() -> StateGraph:
    graph = StateGraph(SchedulerState)

    graph.add_node("extract_details", extract_details)
    graph.add_node("temporal_parse", temporal_parse)
    graph.add_node("generate_confirmation", generate_confirmation)

    graph.set_entry_point("extract_details")
    graph.add_edge("extract_details", "temporal_parse")
    graph.add_edge("temporal_parse", "generate_confirmation")
    graph.add_edge("generate_confirmation", END)

    return graph


scheduler_graph = _build_scheduler_graph().compile()

logger.info("SchedulerAgent graph compiled successfully.")
