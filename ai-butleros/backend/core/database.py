# backend/core/database.py
from __future__ import annotations

import logging
from collections.abc import AsyncGenerator
from pathlib import Path

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase

logger = logging.getLogger(__name__)

# ── Path resolution ────────────────────────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parents[2]  # repo root
_DATA_DIR = _PROJECT_ROOT / "data"
_DATA_DIR.mkdir(parents=True, exist_ok=True)

DATABASE_URL: str = f"sqlite+aiosqlite:///{_DATA_DIR / 'butler.db'}"

# ── Engine & session factory ───────────────────────────────────────────────────
engine: AsyncEngine = create_async_engine(
    DATABASE_URL,
    echo=False,          # flip to True for SQL debug output
    future=True,
    connect_args={"check_same_thread": False},
)

AsyncSessionLocal: async_sessionmaker[AsyncSession] = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autoflush=False,
    autocommit=False,
)


# ── Declarative base ───────────────────────────────────────────────────────────
class Base(DeclarativeBase):
    """Shared metadata base for all domain models."""
    pass


# ── Lifecycle helpers ──────────────────────────────────────────────────────────
async def init_db() -> None:
    """Create all tables that don't yet exist. Called once at startup."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logger.info("Database initialised at %s", DATABASE_URL)


async def close_db() -> None:
    """Dispose the engine connection pool. Called at shutdown."""
    await engine.dispose()
    logger.info("Database engine disposed.")


# ── FastAPI dependency ─────────────────────────────────────────────────────────
async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """
    Yields a scoped AsyncSession per request.
    Always rolls back on unhandled exceptions so the connection is clean.
    """
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise