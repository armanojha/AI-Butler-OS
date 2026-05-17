from __future__ import annotations

import asyncio
import logging
import uuid
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from backend.agents.memory import memory_graph
from backend.agents.planner import planner_graph
from backend.agents.scheduler import scheduler_graph
from backend.api.trace import router as trace_router
from backend.api.ingest import router as ingest_router
from backend.core.database import AsyncSessionLocal, close_db, init_db
from backend.services.log_service import TraceService

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger("butleros")

# ── Internal event envelope ───────────────────────────────────────────────────
class _QueueEvent(BaseModel):
    trace_id: str
    user_input: str
    metadata: dict[str, Any] = Field(default_factory=dict)

# ── Global event queue ───────────────────────────────────────────────────────
_event_queue: asyncio.Queue[_QueueEvent | None]

# ── Background worker ─────────────────────────────────────────────────────────
async def _process_events(queue: asyncio.Queue[_QueueEvent | None]) -> None:
    logger.info("Background worker started.")

    while True:
        event = await queue.get()

        if event is None:
            logger.info("Background worker received shutdown signal.")
            queue.task_done()
            break

        logger.info("Worker picked up event | trace_id=%s", event.trace_id)

        try:
            async with AsyncSessionLocal() as session:
                svc = TraceService(db=session)

                # ── Trace 1: processing started ─────────────────────────
                await svc.create_trace(
                    trace_id=f"{event.trace_id}-start",
                    agent_name="Orchestrator",
                    step_name="processing_started",
                    input_data={"user_input": event.user_input, **event.metadata},
                    monologue="Event dequeued; dispatching to PlannerAgent.",
                )

                # ── Phase 2: PlannerAgent ─────────────────────────────
                planner_result: dict[str, Any] = await planner_graph.ainvoke(
                    {"user_input": event.user_input, "retry_count": 0}
                )

                intent: str = planner_result.get("intent", "UNKNOWN")
                parameters: dict[str, Any] = planner_result.get("parameters", {})
                planner_monologue: str = planner_result.get("monologue", "")
                final_response: str = planner_result.get("final_response", "")
                retry_count: int = planner_result.get("retry_count", 0)
                
                agent_name = "PlannerAgent"
                combined_monologue = planner_monologue
                extra_output: dict[str, Any] = {}

                # ── Phase 3: MemoryAgent ─────────────────────────────
                if intent == "MEMORY_SEARCH":
                    agent_name = "MemoryAgent"
                    memory_result: dict[str, Any] = await memory_graph.ainvoke(
                        {"user_input": event.user_input}
                    )
                    final_response = memory_result.get("final_response", "")
                    combined_monologue = f"[PlannerAgent]\n{planner_monologue}\n\n[MemoryAgent]\n{memory_result.get('monologue', '')}"
                    extra_output = {"retrieved_context_count": len(memory_result.get("retrieved_context", []))}

                # ── Phase 4: SchedulerAgent ───────────────────────────
                elif intent == "SCHEDULE":
                    agent_name = "SchedulerAgent"
                    scheduler_result: dict[str, Any] = await scheduler_graph.ainvoke(
                        {"user_input": event.user_input}
                    )
                    final_response = scheduler_result.get("final_response", "")
                    combined_monologue = f"[PlannerAgent]\n{planner_monologue}\n\n[SchedulerAgent]\n{scheduler_result.get('monologue', '')}"
                    extra_output = {
                        "task_description": scheduler_result.get("task_description", ""),
                        "raw_time_string": scheduler_result.get("raw_time_string", ""),
                        "parsed_timestamp": scheduler_result.get("parsed_timestamp", "INVALID_DATE"),
                    }

                # ── Trace 2: processing finished ──────────────────────
                await svc.create_trace(
                    trace_id=f"{event.trace_id}-end",
                    agent_name=agent_name,
                    step_name="processing_finished",
                    input_data={"user_input": event.user_input},
                    output_data={
                        "intent": intent,
                        "parameters": parameters,
                        "final_response": final_response,
                        "retry_count": retry_count,
                        **extra_output,
                    },
                    monologue=combined_monologue,
                )
                await session.commit()
                logger.info("Trace END written | trace_id=%s", event.trace_id)

        except Exception as exc:  # noqa: BLE001
            logger.exception("Worker failed to process event | trace_id=%s", event.trace_id)
        finally:
            queue.task_done()

# ── Lifespan ─────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    global _event_queue
    logger.info("ButlerOS startup sequence initiated.")
    await init_db()
    _event_queue = asyncio.Queue()
    worker_task = asyncio.create_task(_process_events(_event_queue), name="butler_event_worker")
    yield
    await _event_queue.put(None)
    await asyncio.wait_for(worker_task, timeout=30.0)
    await close_db()

# ── App factory ─────────────────────────────────────────────────────────────
app = FastAPI(title="AI ButlerOS", version="0.4.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routers ─────────────────────────────────────────────────────────────────
app.include_router(trace_router, prefix="/api/v1")
app.include_router(ingest_router, prefix="/api/v1")

class InteractRequest(BaseModel):
    input: str = Field(..., min_length=1, max_length=8_192)

class InteractResponse(BaseModel):
    status: str
    trace_id: str

@app.post("/api/v1/interact", response_model=InteractResponse, status_code=status.HTTP_202_ACCEPTED)
async def interact(body: InteractRequest) -> InteractResponse:
    trace_id = str(uuid.uuid4())
    event = _QueueEvent(trace_id=trace_id, user_input=body.input)
    try:
        _event_queue.put_nowait(event)
    except asyncio.QueueFull:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Queue full.")
    return InteractResponse(status="queued", trace_id=trace_id)

@app.get("/health", include_in_schema=False)
async def health() -> dict[str, str]:
    return {"status": "ok"}
