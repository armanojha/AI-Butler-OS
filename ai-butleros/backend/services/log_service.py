# backend/services/log_service.py
from __future__ import annotations

import logging
import uuid
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.domain.models import ExecutionTrace

logger = logging.getLogger(__name__)


class TraceService:
    """
    Persistence layer for ExecutionTrace records.

    Session-injected — the caller owns transaction boundaries.
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
        record = ExecutionTrace(
            id=trace_id or str(uuid.uuid4()),
            agent_name=agent_name,
            step_name=step_name,
            input_data=input_data,
            output_data=output_data,
            monologue=monologue,
        )

        self._db.add(record)
        await self._db.flush()
        await self._db.commit()

        logger.debug(
            "Trace persisted | id=%s agent=%s step=%s",
            record.id,
            record.agent_name,
            record.step_name,
        )
        return record

    async def get_traces_by_id(self, trace_id: str) -> list[dict[str, Any]]:
        """
        Return all ExecutionTrace rows whose ``id`` starts with ``trace_id``.

        Rows are ordered chronologically by ``timestamp``.  Each row is
        returned as a plain dict so the route layer never touches ORM objects
        directly — preventing implicit lazy-load errors after the session
        is closed.

        An empty list is returned (not raised) when nothing matches; the
        route layer is responsible for converting that to a 404.
        """
        stmt = (
            select(ExecutionTrace)
            .where(ExecutionTrace.id.like(f"{trace_id}%"))
            .order_by(ExecutionTrace.timestamp)
        )
        result = await self._db.execute(stmt)
        rows: list[ExecutionTrace] = list(result.scalars().all())

        return [
            {
                "id": row.id,
                "timestamp": row.timestamp.isoformat(),
                "agent_name": row.agent_name,
                "step_name": row.step_name,
                "input_data": row.input_data,
                "output_data": row.output_data,
                "monologue": row.monologue,
            }
            for row in rows
        ]
