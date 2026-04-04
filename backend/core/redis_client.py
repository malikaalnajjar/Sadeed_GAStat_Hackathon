"""
Redis client factory and helper utilities.

Provides a singleton async Redis connection used for caching detection
results and storing recent anomaly events for the dashboard.

The singleton is created lazily on the first call to :func:`get_redis_client`
and reused for the lifetime of the process.  Both helper functions
(:func:`cache_result` and :func:`get_cached_result`) are intentionally
fail-safe: any Redis error is logged as a warning and the caller continues
uninterrupted.
"""

from __future__ import annotations

import json
import logging
from typing import Any

import redis.asyncio as aioredis

from backend.core.config import settings

logger = logging.getLogger(__name__)

_redis_client: aioredis.Redis | None = None


def get_redis_client() -> aioredis.Redis:
    """Return a configured async Redis client instance (singleton).

    The client is created once using connection parameters from
    :data:`~backend.core.config.settings` and reused on subsequent calls.

    Returns:
        A :class:`redis.asyncio.Redis` client with ``decode_responses=True``.
    """
    global _redis_client
    if _redis_client is None:
        _redis_client = aioredis.Redis(
            host=settings.redis_host,
            port=settings.redis_port,
            db=settings.redis_db,
            decode_responses=True,
        )
    return _redis_client


async def cache_result(
    key: str,
    value: dict[str, Any],
    ttl_seconds: int = 300,
) -> None:
    """Serialize *value* to JSON and store it in Redis with the given TTL.

    Silently logs a warning and returns on any Redis error so that a
    cache write failure never crashes the detection pipeline.

    Args:
        key: Redis key under which the value is stored.
        value: JSON-serialisable dictionary (typically a
            :class:`~backend.models.schemas.DetectionResponse` dumped via
            ``model_dump()``).
        ttl_seconds: Time-to-live in seconds.  Defaults to 300.
    """
    client = get_redis_client()
    try:
        await client.setex(key, ttl_seconds, json.dumps(value))
        logger.debug("Cached result at key '%s' (TTL %ds).", key, ttl_seconds)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Redis cache_result failed for key '%s': %s", key, exc)


async def get_cached_result(key: str) -> dict[str, Any] | None:
    """Retrieve and deserialise a cached detection result.

    Returns ``None`` on a cache miss or any Redis error so that callers can
    treat both cases uniformly.

    Args:
        key: Redis key to look up.

    Returns:
        The deserialised dictionary if the key exists, otherwise ``None``.
    """
    client = get_redis_client()
    try:
        raw: str | None = await client.get(key)
        if raw is None:
            return None
        return json.loads(raw)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Redis get_cached_result failed for key '%s': %s", key, exc)
        return None
