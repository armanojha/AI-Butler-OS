# backend/domain/models.py
from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy import DateTime, String, Text
from sqlalchemy.dialects.sqlite import JSON
from sqlalchemy.orm import Mapped, mapped_column

from backend.core.database import Base


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _new_uuid() -> str:
    return str(uuid.uuid4())


class ExecutionTrace(Base):
    """
    Immutable audit log for every agent step.

    Each row captures one discrete unit of work: what came in, what
    went out, and any chain-of-thought the agent produced.  The
    `monologue` column is intentionally TEXT so long LLM reasoning
    strings are never truncated by a VARCHAR limit.
    """

    __tablename__ = "execution_traces"

    # ── Identity ────────────────────────────────────────────────────────────
    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=_new_uuid,
        index=True,
    )

    # ── Provenance ──────────────────────────────────────────────────────────
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=_utcnow,
        nullable=False,
        index=True,
    )
    agent_name: Mapped[str] = mapped_column(
        String(128),
        nullable=False,
        index=True,
        comment="Logical agent identifier, e.g. 'PlannerAgent'",
    )
    step_name: Mapped[str] = mapped_column(
        String(128),
        nullable=False,
        comment="Step label within the agent graph, e.g. 'intent_extraction'",
    )

    # ── Payload ─────────────────────────────────────────────────────────────
    input_data: Mapped[dict | list | None] = mapped_column(
        JSON,
        nullable=True,
        comment="Raw input passed to this step (arbitrary JSON)",
    )
    output_data: Mapped[dict | list | None] = mapped_column(
        JSON,
        nullable=True,
        comment="Structured output produced by this step",
    )

    # ── Reasoning ───────────────────────────────────────────────────────────
    monologue: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
        comment="Free-form internal reasoning / chain-of-thought from the LLM",
    )

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"<ExecutionTrace id={self.id!r} "
            f"agent={self.agent_name!r} "
            f"step={self.step_name!r} "
            f"ts={self.timestamp.isoformat()}>"
        )