# backend/services/memory_service.py
"""
MemoryService — local RAG persistence layer.

Responsibilities
────────────────
1. Maintain a persistent ChromaDB collection (``butler_memory``).
2. Embed text chunks via a local Ollama ``nomic-embed-text`` model.
3. Expose two async-safe operations:
   - ``ingest_text``  → chunk → embed → upsert into Chroma.
   - ``search``       → embed query → ANN lookup → return text chunks.

Threading note
──────────────
ChromaDB's Python client is synchronous.  We run all blocking Chroma and
embedding calls inside ``asyncio.get_event_loop().run_in_executor(None, ...)``
so we never stall the FastAPI event loop.  The executor defaults to the
process-level ``ThreadPoolExecutor``, which is safe here because ChromaDB's
``PersistentClient`` is thread-safe for concurrent reads and serialised writes.
"""
from __future__ import annotations

import asyncio
import hashlib
import logging
from functools import partial
from pathlib import Path
from typing import Any

import chromadb
from chromadb import Collection
from chromadb.config import Settings
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)

# ── Paths & constants ──────────────────────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_CHROMA_PATH = str(_PROJECT_ROOT / "data" / "chroma_db")

_COLLECTION_NAME = "butler_memory"
_EMBED_MODEL = "nomic-embed-text"
_OLLAMA_BASE_URL = "http://localhost:11434"

_CHUNK_SIZE = 1_000
_CHUNK_OVERLAP = 200


def _chunk_id(source: str, chunk_index: int, text: str) -> str:
    """
    Deterministic, content-addressed ID for each chunk.

    Using a hash means re-ingesting the same document is idempotent —
    Chroma's ``upsert`` will overwrite the identical record rather than
    creating duplicates.
    """
    digest = hashlib.sha256(f"{source}:{chunk_index}:{text}".encode()).hexdigest()[:16]
    return f"{source}-{chunk_index}-{digest}"


class MemoryService:
    """
    Singleton-friendly RAG service.

    Instantiate once at app startup (e.g. in ``lifespan``) and share the
    instance via FastAPI dependency injection or direct import.  All public
    methods are ``async`` and executor-offloaded internally.
    """

    def __init__(self) -> None:
        Path(_CHROMA_PATH).mkdir(parents=True, exist_ok=True)

        # Synchronous Chroma client — safe to construct in the main thread.
        self._client: chromadb.PersistentClient = chromadb.PersistentClient(
            path=_CHROMA_PATH,
            settings=Settings(anonymized_telemetry=False),
        )
        self._collection: Collection = self._client.get_or_create_collection(
            name=_COLLECTION_NAME,
            # cosine distance plays nicer with normalised sentence embeddings
            # than the default l2 distance.
            metadata={"hnsw:space": "cosine"},
        )

        self._embedder = OllamaEmbeddings(
            model=_EMBED_MODEL,
            base_url=_OLLAMA_BASE_URL,
        )

        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=_CHUNK_SIZE,
            chunk_overlap=_CHUNK_OVERLAP,
            length_function=len,
            add_start_index=True,
        )

        logger.info(
            "MemoryService ready | chroma_path=%s collection=%s",
            _CHROMA_PATH,
            _COLLECTION_NAME,
        )

    # ── Private helpers ────────────────────────────────────────────────────────

    async def _embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts without blocking the event loop."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            partial(self._embedder.embed_documents, texts),
        )

    async def _embed_query(self, text: str) -> list[float]:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            partial(self._embedder.embed_query, text),
        )

    def _collection_count(self) -> int:
        return self._collection.count()

    # ── Public API ─────────────────────────────────────────────────────────────

    async def ingest_text(self, text: str, source_name: str) -> int:
        """
        Chunk, embed, and upsert ``text`` into the vector store.

        Parameters
        ----------
        text:        Raw document text (plain or pre-extracted from PDF).
        source_name: Human-readable label stored in Chroma metadata
                     (e.g. a filename, URL, or note title).

        Returns
        -------
        Number of chunks upserted.
        """
        if not text.strip():
            logger.warning("ingest_text called with empty text for source=%r", source_name)
            return 0

        chunks: list[str] = self._splitter.split_text(text)
        if not chunks:
            return 0

        ids = [_chunk_id(source_name, i, chunk) for i, chunk in enumerate(chunks)]
        metadatas: list[dict[str, Any]] = [
            {"source": source_name, "chunk_index": i} for i in range(len(chunks))
        ]

        logger.info(
            "Embedding %d chunks for source=%r …", len(chunks), source_name
        )
        embeddings = await self._embed(chunks)

        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None,
            partial(
                self._collection.upsert,
                ids=ids,
                embeddings=embeddings,
                documents=chunks,
                metadatas=metadatas,
            ),
        )

        logger.info(
            "Upserted %d chunks | source=%r | collection_total=%d",
            len(chunks),
            source_name,
            self._collection.count(),
        )
        return len(chunks)

    async def search(self, query: str, n_results: int = 3) -> list[str]:
        """
        Embed ``query`` and return the top-``n_results`` matching text chunks.

        Returns an empty list — never raises — when the collection is empty
        or the query finds nothing above threshold.  Callers must handle the
        empty-list case gracefully (the MemoryAgent does this).
        """
        loop = asyncio.get_event_loop()

        count: int = await loop.run_in_executor(None, self._collection_count)
        if count == 0:
            logger.info("search called on empty collection; returning [].")
            return []

        # Clamp n_results so Chroma doesn't error when count < n_results.
        effective_n = min(n_results, count)

        query_embedding = await self._embed_query(query)

        results = await loop.run_in_executor(
            None,
            partial(
                self._collection.query,
                query_embeddings=[query_embedding],
                n_results=effective_n,
                include=["documents", "metadatas", "distances"],
            ),
        )

        # results["documents"] is a list-of-lists (one per query embedding).
        docs: list[str] = results.get("documents", [[]])[0]

        logger.info(
            "search | query=%r | hits=%d / requested=%d",
            query[:60],
            len(docs),
            effective_n,
        )
        return docs
