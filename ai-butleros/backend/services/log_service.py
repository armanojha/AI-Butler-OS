# backend/services/log_service.py
from __future__ import annotations

import logging
import uuid
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from backend.domain.models import ExecutionTrace

logger = logging.getLogger(__name__)


class TraceService:
    """
    Persistence layer for ExecutionTrace records.

    Deliberately session-injected (not session-owning) so the caller
    controls transaction boundaries.  In the background worker we open
    a dedicated session per event; in request handlers the FastAPI
    `get_db` dependency provides the session.
    """

    def __init__(self, db: AsyncSession) -> None:
        self._db = db

    async def create_trace(
        self,
        *,
        agent_name: str,
        step_name: str,
        input_data: dict[str, Any] | list[Any] | None = None,
        output_data: dict[str, Any] | list[Any] | None = None,
        monologue: str | None = None,
        trace_id: str | None = None,
    ) -> ExecutionTrace:
        """
        Persist a single trace row and return the committed ORM object.

        Parameters
        ----------
        agent_name:  Logical agent identifier (e.g. ``"PlannerAgent"``).
        step_name:   Step label within the agent graph.
        input_data:  Arbitrary JSON-serialisable input payload.
        output_data: Arbitrary JSON-serialisable output payload.
        monologue:   Free-form LLM chain-of-thought text.
        trace_id:    Caller-supplied UUID string; generated if omitted.
                     Passing the same ID for "start" and "end" traces
                     lets us correlate a full processing run.
        """
        record = ExecutionTrace(
            id=trace_id or str(uuid.uuid4()),
            agent_name=agent_name,
            step_name=step_name,
            input_data=input_data,
            output_data=output_data,
            monologue=monologue,
        )

        self._db.add(record)
        await self._db.flush()   # assigns DB defaults; session still open
        await self._db.commit()

        logger.debug(
            "Trace persisted | id=%s agent=%s step=%s",
            record.id,
            record.agent_name,
            record.step_name,
        )
        return record