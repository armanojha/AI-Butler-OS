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

from backend.agents.planner import planner_graph
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

    Each event gets its own DB session and a self-contained try/except so
    a bad event never poisons the next.  The worker task itself never raises
    — all exceptions are absorbed and logged at ERROR level.
    """
    logger.info("Background worker started.")

    while True:
        event = await queue.get()

        # Poison-pill — lifespan requests a clean shutdown.
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

                # ── Trace 1: processing started ───────────────────────────
                await svc.create_trace(
                    trace_id=f"{event.trace_id}-start",
                    agent_name="Orchestrator",
                    step_name="processing_started",
                    input_data={"user_input": event.user_input, **event.metadata},
                    monologue="Event dequeued; dispatching to PlannerAgent.",
                )
                logger.info("Trace START written | trace_id=%s", event.trace_id)

                # ── Phase 2: invoke the PlannerAgent graph ────────────────
                graph_result: dict[str, Any] = await planner_graph.ainvoke(
                    {
                        "user_input": event.user_input,
                        "retry_count": 0,
                    }
                )

                intent: str = graph_result.get("intent", "UNKNOWN")
                parameters: dict[str, Any] = graph_result.get("parameters", {})
                monologue: str = graph_result.get("monologue", "")
                final_response: str = graph_result.get("final_response", "")
                retry_count: int = graph_result.get("retry_count", 0)

                logger.info(
                    "PlannerAgent finished | intent=%s retries=%d trace_id=%s",
                    intent,
                    retry_count,
                    event.trace_id,
                )

                # ── Trace 2: processing finished ──────────────────────────
                await svc.create_trace(
                    trace_id=f"{event.trace_id}-end",
                    agent_name="PlannerAgent",
                    step_name="processing_finished",
                    input_data={"user_input": event.user_input},
                    output_data={
                        "intent": intent,
                        "parameters": parameters,
                        "final_response": final_response,
                        "retry_count": retry_count,
                    },
                    monologue=monologue,
                )
                logger.info("Trace END written | trace_id=%s", event.trace_id)
                await session.commit()

        except Exception as exc:  # noqa: BLE001
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
    await asyncio.wait_for(worker_task, timeout=30.0)  # increased for LLM latency
    await close_db()
    logger.info("ButlerOS shutdown complete.")


# ── App factory ────────────────────────────────────────────────────────────────

app = FastAPI(
    title="AI ButlerOS",
    version="0.2.0",
    description="Local-first personal orchestration assistant — Phase 2.",
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

    Returns immediately with a ``trace_id``.  The PlannerAgent processes
    the event asynchronously; the final state is persisted to SQLite and
    will be queryable via the trace endpoint in Phase 3.
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
