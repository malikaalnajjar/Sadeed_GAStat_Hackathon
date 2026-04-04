"""
Detection orchestration service — cascading pipeline.

Pipeline
--------
1. Run GE and SVM in parallel via :func:`asyncio.gather`.
2. If GE flags or SVM flags → call LLM to **confirm + explain**.
3. Final verdict: ``is_anomaly = (ge_anomaly OR svm_anomaly) AND llm_anomaly``
   GE flags are ground truth — the LLM cannot override them.
   SVM-only flags can be overridden by the LLM (false-positive filter).

Cache key format: ``"detection:{record_id}"``
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from backend.detection.base import BaseDetector
from backend.detection.llm_strategy import LLMDetector
from backend.models.schemas import DetectionResponse, StrategyName, StrategyResult

logger = logging.getLogger(__name__)


class DetectionService:
    """Coordinates cascading multi-strategy anomaly detection.

    GE and SVM detect; LLM confirms and explains.  The LLM acts as a
    false-positive filter — it can clear records that GE/SVM incorrectly
    flagged, but it cannot introduce new anomalies on its own.
    """

    def __init__(
        self,
        ge_detector: BaseDetector | None,
        svm_detector: BaseDetector | None,
        llm_detector: LLMDetector | None,
        redis_client: Any | None = None,
        *,
        cache_ttl_seconds: int = 300,
        svm_llm_threshold: float = 0.5,
    ) -> None:
        self._ge_detector = ge_detector
        self._svm_detector = svm_detector
        self._llm_detector = llm_detector
        self._redis: Any | None = redis_client
        self._cache_ttl: int = cache_ttl_seconds
        self._svm_llm_threshold: float = svm_llm_threshold

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    async def run(
        self,
        record_id: str,
        data: dict[str, Any],
    ) -> DetectionResponse:
        """Execute the cascading detection pipeline and return results."""
        cache_key = self._cache_key(record_id)

        # --- Cache hit path ---
        if self._redis is not None:
            try:
                cached_raw: str | None = await self._redis.get(cache_key)
                if cached_raw is not None:
                    logger.debug("Cache hit for record_id '%s'.", record_id)
                    payload = json.loads(cached_raw) if isinstance(cached_raw, str) else cached_raw
                    return DetectionResponse.model_validate(payload)
            except Exception:  # noqa: BLE001
                logger.warning("Redis unavailable, skipping cache read.")

        # --- Stage 1: Run GE and SVM concurrently ---
        stage1_tasks: list[asyncio.Task] = []
        if self._ge_detector is not None:
            stage1_tasks.append(
                asyncio.ensure_future(
                    self._run_strategy(StrategyName.great_expectations, self._ge_detector, data)
                )
            )
        if self._svm_detector is not None:
            stage1_tasks.append(
                asyncio.ensure_future(
                    self._run_strategy(StrategyName.svm, self._svm_detector, data)
                )
            )

        stage1_results: list[StrategyResult] = list(await asyncio.gather(*stage1_tasks))

        ge_result = next((r for r in stage1_results if r.strategy == StrategyName.great_expectations), None)
        svm_result = next((r for r in stage1_results if r.strategy == StrategyName.svm), None)

        # --- Preliminary verdict: GE OR SVM ---
        ge_anomaly = ge_result.is_anomaly if ge_result else False
        svm_anomaly = svm_result.is_anomaly if svm_result else False
        svm_score = svm_result.score if svm_result else 0.0
        # Two-pass: also send borderline SVM records to LLM
        svm_borderline = (
            not svm_anomaly
            and svm_score >= self._svm_llm_threshold
        )
        preliminary_verdict = ge_anomaly or svm_anomaly or svm_borderline

        # --- Stage 2: If anomaly detected, call LLM to confirm + explain ---
        llm_triggered = False
        llm_skip_reason: str | None = None
        llm_result: StrategyResult | None = None
        explanation: str | None = None

        if not preliminary_verdict:
            llm_skip_reason = "No anomaly detected; LLM not needed."
        elif self._llm_detector is None:
            llm_skip_reason = "LLM detector not configured."
        else:
            llm_triggered = True

            # Run LLM detect + explain in parallel
            ge_score = ge_result.score if ge_result else None
            ge_failed = 0
            if ge_result and ge_result.raw:
                ge_failed = len(ge_result.raw.get("failed_expectations", []))
            svm_score = svm_result.score if svm_result else None

            # Build findings list from GE details for the LLM
            findings: list[str] = []
            if ge_result and ge_result.is_anomaly and ge_result.raw:
                for detail in ge_result.raw.get("failed_details", []):
                    label = detail.get("label", "")
                    if label:
                        findings.append(label)
                    elif "min_value" in detail:
                        col = detail.get("column", "")
                        actual = data.get(col, "?")
                        findings.append(
                            f"{col} is {actual}, allowed range is "
                            f"{detail['min_value']}-{detail['max_value']}"
                        )

            llm_result = await self._llm_detector.detect(
                data, svm_score=svm_score, ge_score=ge_score,
                findings=findings or None,
            )
            explanation = llm_result.explanation or None

        # --- Final verdict: (GE OR SVM) AND LLM ---
        # LLM acts as a false-positive filter. If it wasn't triggered
        # (no LLM configured or no preliminary anomaly), fall back to
        # the preliminary verdict.
        if llm_triggered and llm_result is not None:
            # GE flags are ground truth — LLM cannot override them.
            # LLM only filters SVM-only flags (false positive reduction).
            ge_flagged = ge_result is not None and ge_result.is_anomaly
            if ge_flagged:
                verdict = True
                # Prefer LLM explanation (natural language), fall back to GE
                if llm_result and llm_result.explanation:
                    explanation = llm_result.explanation
                else:
                    explanation = ge_result.explanation
            else:
                verdict = preliminary_verdict and llm_result.is_anomaly
            if not verdict:
                explanation = None
        else:
            verdict = preliminary_verdict

        # --- Assemble results ---
        results: list[StrategyResult] = []
        if ge_result is not None:
            results.append(ge_result)
        if svm_result is not None:
            results.append(svm_result)
        if llm_result is not None:
            results.append(llm_result)

        # Determine severity: GE severity takes priority, then SVM defaults to warning
        severity: str | None = None
        if verdict:
            if ge_result is not None and ge_result.raw:
                severity = ge_result.raw.get("severity")
            if severity is None and (svm_result is not None and svm_result.is_anomaly):
                severity = "warning"  # SVM flags are always warnings (statistical)

        response = DetectionResponse(
            record_id=record_id,
            results=results,
            is_anomaly=verdict,
            severity=severity,
            explanation=explanation,
            llm_triggered=llm_triggered,
            llm_skip_reason=llm_skip_reason,
        )

        # --- Store in cache ---
        if self._redis is not None:
            try:
                serialised = json.dumps(response.model_dump())
                await self._redis.setex(cache_key, self._cache_ttl, serialised)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Failed to cache result for record_id '%s': %s", record_id, exc
                )

        return response

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    async def _run_strategy(
        self,
        name: StrategyName,
        detector: BaseDetector,
        data: dict[str, Any],
    ) -> StrategyResult:
        """Dispatch a single strategy with fail-safe error handling."""
        try:
            return await detector.detect(data)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Strategy '%s' raised an unexpected error: %s: %s",
                name.value,
                type(exc).__name__,
                exc,
            )
            return StrategyResult(
                strategy=name,
                is_anomaly=False,
                score=0.0,
                explanation=f"Strategy error: {type(exc).__name__}: {exc}",
            )

    def _cache_key(self, record_id: str) -> str:
        return f"detection:{record_id}"
