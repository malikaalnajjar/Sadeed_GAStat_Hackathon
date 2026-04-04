"""
FastAPI application entry point for the Sadeed anomaly detection service.

Registers all API routers and configures middleware (CORS, lifespan events).
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from typing import AsyncIterator

import redis.asyncio as aioredis
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.api.routes import anomalies, health
from backend.core.config import settings
from backend.detection.great_expectations_strategy import GreatExpectationsDetector
from backend.detection.lfs_preprocessing import add_derived_columns
from backend.detection.llm_strategy import LLMDetector
from backend.detection.svm_strategy import SVMDetector
from backend.services.detection_service import DetectionService

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Initialise and tear down application-level resources."""

    # --- Redis -----------------------------------------------------------
    redis_client: aioredis.Redis | None = None
    try:
        rc = aioredis.Redis(
            host=settings.redis_host,
            port=settings.redis_port,
            db=settings.redis_db,
            decode_responses=True,
            socket_connect_timeout=2,
            socket_timeout=2,
        )
        await rc.ping()
        redis_client = rc
        logger.info("Redis connected.")
    except Exception:  # noqa: BLE001
        logger.warning("Redis unavailable — running without cache.")
    app.state.redis_client = redis_client

    # --- Great Expectations detector -------------------------------------
    ge_detector: GreatExpectationsDetector | None = None
    if settings.enable_ge_strategy:
        if os.path.exists(settings.ge_expectation_suite_path):
            ge_detector = GreatExpectationsDetector(
                settings.ge_expectation_suite_path,
                preprocessor=add_derived_columns,
            )
        else:
            logger.warning(
                "GE suite file not found at '%s'; Great Expectations strategy disabled.",
                settings.ge_expectation_suite_path,
            )
    app.state.ge_detector = ge_detector

    # --- OC-SVM detector -------------------------------------------------
    svm_detector: SVMDetector | None = None
    if settings.enable_svm_strategy:
        if os.path.exists(settings.oc_svm_training_data_path):
            svm_detector = SVMDetector(
                training_data_path=settings.oc_svm_training_data_path,
                model_path=settings.oc_svm_model_path,
                kernel=settings.oc_svm_kernel,
                nu=settings.oc_svm_nu,
                gamma=settings.oc_svm_gamma,
                feature_columns=["act_1_total", "age", "cut_5_total", "edu_ordinal", "q_602_val"],
            )
        else:
            logger.warning(
                "SVM training data not found at '%s'; SVM strategy disabled.",
                settings.oc_svm_training_data_path,
            )
    app.state.svm_detector = svm_detector

    # --- LLM detector ----------------------------------------------------
    llm_detector: LLMDetector | None = None
    if settings.enable_llm_strategy:
        llm_detector = LLMDetector(
            base_url=settings.ollama_base_url,
            model=settings.ollama_model,
        )
    app.state.llm_detector = llm_detector

    # --- DetectionService ------------------------------------------------
    app.state.detection_service = DetectionService(
        ge_detector=ge_detector,
        svm_detector=svm_detector,
        llm_detector=llm_detector,
        redis_client=redis_client,
        cache_ttl_seconds=settings.detection_cache_ttl_seconds,
        svm_llm_threshold=settings.svm_llm_threshold,
    )

    logger.info(
        "Sadeed started — GE=%s  SVM=%s  LLM=%s",
        ge_detector is not None,
        svm_detector is not None,
        llm_detector is not None,
    )

    # --- Warm up LLM (pre-load model into Ollama memory) -----------------
    if llm_detector is not None:
        logger.info("Warming up LLM model (this may take a minute on first load)…")
        try:
            import httpx

            async with httpx.AsyncClient() as client:
                await client.post(
                    f"{settings.ollama_base_url.rstrip('/')}/api/generate",
                    json={"model": settings.ollama_model, "prompt": "ping", "stream": False},
                    timeout=300.0,
                )
            logger.info("LLM model warmed up successfully.")
        except Exception as exc:  # noqa: BLE001
            logger.warning("LLM warmup failed (non-fatal): %s", exc)

    yield

    # --- Shutdown --------------------------------------------------------
    if redis_client is not None:
        await redis_client.aclose()
        logger.info("Redis connection closed.")


def create_app() -> FastAPI:
    """Construct and configure the FastAPI application instance."""
    app = FastAPI(
        title="Sadeed Anomaly Detection",
        version="0.1.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(anomalies.router)
    app.include_router(health.router)

    return app


app = create_app()
