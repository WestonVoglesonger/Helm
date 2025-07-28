"""Preference store implementation.

The preference store manages user/task preferences persisted in a JSONB column
within PostgreSQL.  A Redis cache is used for hot reads.  For this MVP, the
store includes an in‑memory fallback when running without a database.

Methods are asynchronous to allow integration with async frameworks.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict

from sqlalchemy import Column, String, JSON, Table, MetaData, select
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, create_async_engine
from sqlalchemy.ext.asyncio import async_sessionmaker

# Use the asyncio support provided by redis-py (redis>=4).  If unavailable,
# set aioredis to None so that the code falls back to in-memory caching.
try:
    import redis.asyncio as aioredis  # type: ignore[attr-defined]
except Exception:
    aioredis = None  # type: ignore

from ..config import get_settings

logger = logging.getLogger(__name__)


class PrefStore:
    """Abstraction over the preferences table and Redis cache."""

    _cache_prefix = "prefs:"

    def __init__(self) -> None:
        settings = get_settings()
        # Create async DB engine.  In development, SQLite may be used.
        self.engine: AsyncEngine = create_async_engine(
            settings.database_url, future=True, echo=False
        )
        self.async_session: async_sessionmaker[AsyncSession] = async_sessionmaker(
            self.engine, expire_on_commit=False
        )
        # Redis client: attempt to connect if redis.asyncio is available
        self.redis = None
        if aioredis is not None:
            try:
                self.redis = aioredis.from_url(
                    settings.redis_url,
                    encoding="utf-8",
                    decode_responses=True,
                )
            except Exception as exc:
                logger.warning("Could not connect to Redis: %s", exc)
        # Define table metadata
        metadata = MetaData()
        self.table = Table(
            "preferences",
            metadata,
            Column("id", String, primary_key=True),
            Column("data", JSON, nullable=False),
        )
        # Store metadata for table creation
        self._metadata = metadata

    async def _create_table(self, metadata: MetaData) -> None:
        async with self.engine.begin() as conn:
            await conn.run_sync(metadata.create_all)

    async def get(self, user_id: str) -> Dict[str, Any]:
        """Retrieve preferences for a user.

        If a cached value exists in Redis, it is returned.  Otherwise the
        database is queried.  On cache miss, the result is cached.
        """
        # Ensure table exists
        await self._create_table(self._metadata)
        
        # Attempt cache
        if self.redis is not None:
            cached = await self.redis.get(self._cache_prefix + user_id)
            if cached:
                try:
                    return json.loads(cached)
                except Exception:
                    logger.warning("Failed to decode cached prefs for %s", user_id)

        # Query database
        async with self.async_session() as session:
            stmt = select(self.table.c.data).where(self.table.c.id == user_id)
            result = await session.execute(stmt)
            row = result.fetchone()
            prefs = row.data if row else {}
        # Cache
        if self.redis is not None:
            try:
                await self.redis.set(self._cache_prefix + user_id, json.dumps(prefs))
            except Exception as exc:
                logger.warning("Could not cache prefs for %s: %s", user_id, exc)
        return prefs

    async def set(self, user_id: str, prefs: Dict[str, Any]) -> None:
        """Persist preferences for a user.

        Updates the database and invalidates the cache.
        """
        # Ensure table exists
        await self._create_table(self._metadata)
        
        from sqlalchemy.dialects.postgresql import insert as pg_insert

        async with self.async_session() as session:
            async with session.begin():
                stmt = pg_insert(self.table).values(id=user_id, data=prefs)
                # On conflict, update data
                stmt = stmt.on_conflict_do_update(
                    index_elements=[self.table.c.id], set_={"data": prefs}
                )
                await session.execute(stmt)
        # Invalidate cache
        if self.redis is not None:
            try:
                await self.redis.delete(self._cache_prefix + user_id)
            except Exception as exc:
                logger.warning("Could not invalidate cache for %s: %s", user_id, exc)
