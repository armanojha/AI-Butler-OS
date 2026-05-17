# backend/api/ingest.py
from __future__ import annotations

import logging
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

# Import the class instead of the missing function
from backend.services.memory_service import MemoryService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/ingest", tags=["ingest"])

# Initialize the service directly
_memory_service = MemoryService()

class IngestTextRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Raw text to ingest.")
    source_name: str = Field(..., min_length=1, description="Label for this source.")

class IngestResponse(BaseModel):
    status: str
    source_name: str
    chunks_stored: int

@router.post("/text", response_model=IngestResponse, status_code=status.HTTP_201_CREATED)
async def ingest_text(body: IngestTextRequest) -> IngestResponse:
    try:
        chunks_stored = await _memory_service.ingest_text(
            text=body.text,
            source_name=body.source_name,
        )
    except Exception as exc:
        logger.error("ingest_text route error: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ingestion failed: {exc}",
        )

    return IngestResponse(
        status="ingested",
        source_name=body.source_name,
        chunks_stored=chunks_stored,
    )