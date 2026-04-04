"""
Health-check routes.

GET /health        — liveness probe (always 200 if the process is up)
GET /health/ready  — readiness probe (checks Redis and Ollama connectivity)
"""

from typing import Any

import redis.asyncio as aioredis
from fastapi import APIRouter, Request

from backend.core.config import settings

router = APIRouter(prefix="/health", tags=["health"])


@router.get("")
async def liveness() -> dict[str, str]:
    """Return 200 OK to signal the process is alive."""
    return {"status": "ok"}


@router.get("/ready")
async def readiness(request: Request) -> dict[str, Any]:
    """Check downstream dependencies and return their status."""
    redis_ok = False
    redis_client: aioredis.Redis | None = getattr(request.app.state, "redis_client", None)
    if redis_client is not None:
        try:
            await redis_client.ping()
            redis_ok = True
        except Exception:
            pass

    ollama_ok = False
    try:
        import httpx
        async with httpx.AsyncClient(timeout=3.0) as client:
            resp = await client.get(f"{settings.ollama_base_url}/api/tags")
            ollama_ok = resp.is_success
    except Exception:
        pass

    ready = redis_ok and ollama_ok
    return {
        "ready": ready,
        "redis": redis_ok,
        "ollama": ollama_ok,
    }
