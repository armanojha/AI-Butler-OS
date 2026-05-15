# backend/main.py
from __future__ import annotations

import asyncio
import logging
import uuid
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from backend.core.database import AsyncSessionLocal, close_db, init_db
from backend.services.log_service import TraceService

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger("butleros")

# ── Internal event envelope ────────────────────────────────────────────────────

class _QueueEvent(BaseModel):
    """Internal-only envelope placed on the asyncio.Queue."""

    trace_id: str
    user_input: str
    metadata: dict[str, Any] = Field(default_factory=dict)


# ── Global event queue (initialised in lifespan) ──────────────────────────────
_event_queue: asyncio.Queue[_QueueEvent | None]   # None = poison-pill


# ── Background worker ──────────────────────────────────────────────────────────

async def _process_events(queue: asyncio.Queue[_QueueEvent | None]) -> None:
    """
    Drain the queue indefinitely.

    Each iteration opens its own DB session so a failure in one event
    never poisons the session state for the next.  The worker absorbs
    all exceptions to keep the task alive; individual failures are
    logged at ERROR level so they surface in observability tooling.
    """
    logger.info("Background worker started.")

    while True:
        event = await queue.get()

        # Poison-pill — clean shutdown requested by lifespan.
        if event is None:
            logger.info("Background worker received shutdown signal.")
            queue.task_done()
            break

        logger.info(
            "Worker picked up event | trace_id=%s input=%r",
            event.trace_id,
            event.user_input[:80],
        )

        try:
            async with AsyncSessionLocal() as session:
                svc = TraceService(db=session)

                # ── Trace: processing started ─────────────────────────────
                await svc.create_trace(
                    trace_id=event.trace_id + "-start",
                    agent_name="Orchestrator",
                    step_name="processing_started",
                    input_data={"user_input": event.user_input, **event.metadata},
                    monologue="Event dequeued; beginning orchestration pipeline.",
                )
                logger.info("Trace START written | trace_id=%s", event.trace_id)

                # ── TODO: dispatch to PlannerAgent (Phase 2) ─────────────

                # ── Trace: processing finished ────────────────────────────
                await svc.create_trace(
                    trace_id=event.trace_id + "-end",
                    agent_name="Orchestrator",
                    step_name="processing_finished",
                    input_data={"user_input": event.user_input},
                    output_data={"status": "stub_complete"},
                    monologue="Pipeline stub completed successfully.",
                )
                logger.info("Trace END written   | trace_id=%s", event.trace_id)

        except Exception as exc:  # noqa: BLE001
            # Log and continue — never let a bad event crash the worker.
            logger.exception(
                "Worker failed to process event | trace_id=%s | error=%s",
                event.trace_id,
                exc,
            )
        finally:
            queue.task_done()


# ── Lifespan ───────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    global _event_queue

    logger.info("ButlerOS startup sequence initiated.")
    await init_db()

    _event_queue = asyncio.Queue()
    worker_task = asyncio.create_task(
        _process_events(_event_queue),
        name="butler_event_worker",
    )

    logger.info("ButlerOS ready.")
    yield  # ── application runs ──────────────────────────────────────────

    logger.info("ButlerOS shutdown sequence initiated.")
    await _event_queue.put(None)           # send poison-pill
    await asyncio.wait_for(worker_task, timeout=15.0)
    await close_db()
    logger.info("ButlerOS shutdown complete.")


# ── App factory ────────────────────────────────────────────────────────────────

app = FastAPI(
    title="AI ButlerOS",
    version="0.1.0",
    description="Local-first personal orchestration assistant.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Next.js dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Pydantic schemas ───────────────────────────────────────────────────────────

class InteractRequest(BaseModel):
    input: str = Field(
        ...,
        min_length=1,
        max_length=8_192,
        description="Raw user message to process.",
    )


class InteractResponse(BaseModel):
    status: str
    trace_id: str


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.post(
    "/api/v1/interact",
    response_model=InteractResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Submit a message for async processing",
)
async def interact(body: InteractRequest) -> InteractResponse:
    """
    Enqueue a user message for the background worker.

    Returns immediately with a ``trace_id`` the client can use to poll
    for results once the SchedulerAgent and MemoryAgent are wired up.
    """
    trace_id = str(uuid.uuid4())

    event = _QueueEvent(trace_id=trace_id, user_input=body.input)

    try:
        _event_queue.put_nowait(event)
    except asyncio.QueueFull:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Event queue is at capacity. Retry shortly.",
        )

    logger.info("Event enqueued | trace_id=%s", trace_id)
    return InteractResponse(status="queued", trace_id=trace_id)


@app.get("/health", include_in_schema=False)
async def health() -> dict[str, str]:
    return {"status": "ok"}