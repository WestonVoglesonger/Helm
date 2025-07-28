"""Simple Redis key‑value wrapper.

This wrapper encapsulates Redis operations to store arbitrary data as strings.
It gracefully handles the absence of a Redis server, logging warnings.
"""

from __future__ import annotations

import logging
from typing import Optional

import aioredis

from ..config import get_settings


logger = logging.getLogger(__name__)


class KVStore:
    def __init__(self) -> None:
        settings = get_settings()
        try:
            self.redis = aioredis.from_url(
                settings.redis_url, encoding="utf-8", decode_responses=True
            )
        except Exception as exc:
            logger.warning("Could not connect to Redis: %s", exc)
            self.redis = None

    async def get(self, key: str) -> Optional[str]:
        if self.redis is None:
            return None
        try:
            return await self.redis.get(key)
        except Exception as exc:
            logger.warning("KVStore.get failed: %s", exc)
            return None

    async def set(self, key: str, value: str, expire: Optional[int] = None) -> None:
        if self.redis is None:
            return
        try:
            await self.redis.set(key, value, ex=expire)
        except Exception as exc:
            logger.warning("KVStore.set failed: %s", exc)