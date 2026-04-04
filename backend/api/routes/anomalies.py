"""
API routes for anomaly detection.

POST /anomalies/detect              — submit a data record for analysis
GET  /anomalies/detect/recent       — list the most recent anomaly events
GET  /anomalies/detect/{record_id}  — retrieve a previously computed result from cache
GET  /anomalies/health              — report health of all detectors and Redis
"""

from __future__ import annotations

import json
import logging
import time
from typing import Annotated, Any

import redis.asyncio as aioredis
from fastapi import APIRouter, Depends, HTTPException, Request, status

from backend.models.schemas import DataPayload, DetectionResponse
from backend.services.detection_service import DetectionService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/anomalies", tags=["anomalies"])

_RECENT_KEY = "recent_detections"


# ---------------------------------------------------------------------------
# Dependency helpers
# ---------------------------------------------------------------------------


def get_detection_service(request: Request) -> DetectionService:
    return request.app.state.detection_service


def get_redis(request: Request) -> aioredis.Redis | None:
    return request.app.state.redis_client


ServiceDep = Annotated[DetectionService, Depends(get_detection_service)]
RedisDep = Annotated[aioredis.Redis | None, Depends(get_redis)]


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@router.post("/detect", response_model=DetectionResponse, status_code=status.HTTP_200_OK)
async def detect_anomaly(
    payload: DataPayload,
    service: ServiceDep,
    redis: RedisDep,
) -> DetectionResponse:
    """Submit a data record and receive anomaly detection results."""
    response = await service.run(payload.record_id, payload.data)

    # Record in the recent_detections sorted set (score = Unix timestamp).
    if redis is not None:
        try:
            member = json.dumps(response.model_dump())
            await redis.zadd(_RECENT_KEY, {member: time.time()})
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to update %s sorted set: %s", _RECENT_KEY, exc)

    return response


@router.get("/detect/recent", response_model=list[DetectionResponse])
async def list_recent(redis: RedisDep, limit: int = 20) -> list[DetectionResponse]:
    """Return the most recent anomaly detection results stored in Redis."""
    if redis is None:
        return []

    try:
        # Highest scores (most recent timestamps) first.
        members: list[str] = await redis.zrevrange(_RECENT_KEY, 0, limit - 1)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to fetch %s: %s", _RECENT_KEY, exc)
        return []

    results: list[DetectionResponse] = []
    for member in members:
        try:
            results.append(DetectionResponse.model_validate(json.loads(member)))
        except Exception as exc:  # noqa: BLE001
            logger.warning("Skipping malformed recent_detections entry: %s", exc)

    return results


@router.get("/detect/{record_id}", response_model=DetectionResponse)
async def get_result(record_id: str, redis: RedisDep) -> DetectionResponse:
    """Fetch a cached detection result by record ID."""
    if redis is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No cached result for record_id '{record_id}'",
        )

    try:
        raw: str | None = await redis.get(f"detection:{record_id}")
    except Exception as exc:  # noqa: BLE001
        logger.warning("Redis error fetching record_id '%s': %s", record_id, exc)
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No cached result for record_id '{record_id}'",
        )

    if raw is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No cached result for record_id '{record_id}'",
        )

    return DetectionResponse.model_validate(json.loads(raw))


@router.get("/health")
async def detector_health(request: Request, redis: RedisDep) -> dict[str, Any]:
    """Return health status of all three detectors and Redis connectivity."""
    state = request.app.state

    async def _check(attr: str) -> bool | None:
        detector = getattr(state, attr, None)
        if detector is None:
            return None
        try:
            return await detector.health_check()
        except Exception as exc:  # noqa: BLE001
            logger.warning("health_check failed for %s: %s", attr, exc)
            return False

    redis_ok: bool = False
    if redis is not None:
        try:
            await redis.ping()
            redis_ok = True
        except Exception as exc:  # noqa: BLE001
            logger.warning("Redis ping failed: %s", exc)

    return {
        "great_expectations": await _check("ge_detector"),
        "svm": await _check("svm_detector"),
        "llm": await _check("llm_detector"),
        "redis": redis_ok,
    }
