"""Vector store using `pgvector`.

This module provides a basic wrapper around a PostgreSQL table with a `vector`
column powered by the `pgvector` extension.  Each row stores an embedding and
an associated payload.  The API allows inserting new vectors and performing
nearest‑neighbour search.  For this MVP, the search method returns an empty
list; implementing similarity search can be added later.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional, Tuple

from sqlalchemy import Column, Integer, String, JSON, Table, MetaData
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, create_async_engine
from sqlalchemy.ext.asyncio import async_sessionmaker
from sqlalchemy.sql import text

from ..config import get_settings


logger = logging.getLogger(__name__)


class VectorStore:
    """Simple vector store backed by PostgreSQL with pgvector."""

    def __init__(self) -> None:
        settings = get_settings()
        self.engine: AsyncEngine = create_async_engine(
            settings.database_url, future=True, echo=False
        )
        self.async_session: async_sessionmaker[AsyncSession] = async_sessionmaker(
            self.engine, expire_on_commit=False
        )
        metadata = MetaData()
        # The `vector` column type is created via pgvector; here we declare it
        # generically as `String` for SQLAlchemy type checking.  In a real
        # implementation, use `sqlalchemy_pgvector.Vector`.
        self.table = Table(
            "vectors",
            metadata,
            Column("id", String, primary_key=True),
            Column("vector", String, nullable=False),
            Column("payload", JSON, nullable=False),
        )
        self._create_table_task = self._create_table(metadata)

    async def _create_table(self, metadata: MetaData) -> None:
        async with self.engine.begin() as conn:
            await conn.run_sync(metadata.create_all)

    async def add(self, id_: str, vector: List[float], payload: Dict[str, Any]) -> None:
        """Insert or update a vector and its payload."""
        vector_str = json.dumps(vector)
        async with self.async_session() as session:
            async with session.begin():
                stmt = pg_insert(self.table).values(id=id_, vector=vector_str, payload=payload)
                stmt = stmt.on_conflict_do_update(
                    index_elements=[self.table.c.id], set_={"vector": vector_str, "payload": payload}
                )
                await session.execute(stmt)

    async def search(self, vector: List[float], top_k: int = 5) -> List[Tuple[str, float, Dict[str, Any]]]:
        """Return the top_k closest vectors to the query.

        Placeholder implementation returns an empty list.  To implement
        similarity search, you can use cosine distance via `pgvector`'s
        `<->` operator in a raw SQL query.
        """
        # TODO: implement real similarity search using pgvector
        logger.info("VectorStore.search called but not implemented.")
        return []