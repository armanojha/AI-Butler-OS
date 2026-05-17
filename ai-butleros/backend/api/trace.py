# backend/api/trace.py
"""
Trace polling API.

Clients POST to /api/v1/interact and receive a ``trace_id`` immediately.
They then poll GET /api/v1/trace/{trace_id} every ~2 seconds until they see
a row with ``step_name == "processing_finished"``, at which point they can
read ``output_data.final_response`` and stop polling.

Why not WebSockets / SSE?
──────────────────────────
Polling keeps the frontend dead-simple (a plain ``setInterval`` + ``fetch``)
and is perfectly adequate at 2 s resolution for LLM inference that takes
8–15 s.  We can upgrade to SSE in a later phase without changing the backend
data model — the trace rows are already timestamped and ordered.
"""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from backend.core.database import get_db
from backend.services.log_service import TraceService

router = APIRouter(prefix="/trace", tags=["trace"])


# ── Response schema ────────────────────────────────────────────────────────────

class TraceRow(BaseModel):
    id: str
    timestamp: str
    agent_name: str
    step_name: str
    input_data: dict[str, Any] | list[Any] | None
    output_data: dict[str, Any] | list[Any] | None
    monologue: str | None


class TraceResponse(BaseModel):
    trace_id: str
    # Convenience flag so clients can stop polling without parsing step_name.
    completed: bool
    rows: list[TraceRow]


# ── Endpoint ───────────────────────────────────────────────────────────────────

@router.get(
    "/{trace_id}",
    response_model=TraceResponse,
    summary="Poll the execution trace for a queued interaction",
)
async def get_trace(
    trace_id: str,
    db: AsyncSession = Depends(get_db),
) -> TraceResponse:
    """
    Fetch all trace rows whose ``id`` starts with ``trace_id``.

    Returns 404 while the event is still sitting in the queue (no rows yet).
    Once the worker picks it up, a ``processing_started`` row appears.
    When processing is done, a ``processing_finished`` row appears and
    ``completed`` flips to ``true``.
    """
    svc = TraceService(db=db)
    rows = await svc.get_traces_by_id(trace_id)

    if not rows:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=(
                f"No trace found for id={trace_id!r}. "
                "The event may still be queued — retry in 1–2 seconds."
            ),
        )

    completed = any(r["step_name"] == "processing_finished" for r in rows)

    return TraceResponse(
        trace_id=trace_id,
        completed=completed,
        rows=[TraceRow(**r) for r in rows],
    )
