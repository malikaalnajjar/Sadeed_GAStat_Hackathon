"""
Ensemble LLM detector — runs two models and requires agreement.

Uses an aggressive primary model (high recall) paired with a conservative
secondary model (high precision).  A record is flagged as anomalous only
when **both** models agree, cutting false positives while preserving recall
on clear anomalies.

Both models run concurrently via asyncio.gather for minimal latency overhead.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from backend.detection.llm_strategy import LLMDetector
from backend.models.schemas import StrategyName, StrategyResult

logger = logging.getLogger(__name__)


class EnsembleLLMDetector(LLMDetector):
    """Two-model LLM ensemble that requires agreement to flag anomalies.

    Inherits from :class:`LLMDetector` so it can be used as a drop-in
    replacement in the detection service.  The primary model (inherited)
    handles the main classification; the secondary model provides a
    second opinion.

    Verdict logic:
        - **is_anomaly**: both models must agree (AND)
        - **score**: average of both scores
        - **explanation**: taken from whichever model gave the higher score
    """

    def __init__(
        self,
        base_url: str,
        primary_model: str = "gemma2:9b",
        secondary_model: str = "mistral:latest",
    ) -> None:
        super().__init__(base_url=base_url, model=primary_model)
        self._secondary = LLMDetector(base_url=base_url, model=secondary_model)

    async def detect(
        self,
        data: dict[str, Any],
        *,
        svm_score: float | None = None,
        ge_score: float | None = None,
        findings: list[str] | None = None,
    ) -> StrategyResult:
        """Run both models concurrently and combine verdicts."""
        primary_task = super().detect(
            data, svm_score=svm_score, ge_score=ge_score, findings=findings,
        )
        secondary_task = self._secondary.detect(
            data, svm_score=svm_score, ge_score=ge_score, findings=findings,
        )

        primary_result, secondary_result = await asyncio.gather(
            primary_task, secondary_task,
        )

        # Both must agree to flag anomaly
        is_anomaly = primary_result.is_anomaly and secondary_result.is_anomaly

        # Average scores
        p_score = primary_result.score or 0.0
        s_score = secondary_result.score or 0.0
        avg_score = round((p_score + s_score) / 2, 4)

        # Use explanation from the higher-confidence model
        if p_score >= s_score:
            explanation = primary_result.explanation
        else:
            explanation = secondary_result.explanation

        logger.debug(
            "Ensemble: primary=%s (%.2f), secondary=%s (%.2f) → %s (%.2f)",
            primary_result.is_anomaly, p_score,
            secondary_result.is_anomaly, s_score,
            is_anomaly, avg_score,
        )

        return StrategyResult(
            strategy=StrategyName.llm,
            is_anomaly=is_anomaly,
            score=avg_score,
            explanation=explanation,
            raw={
                "ensemble": True,
                "primary": {
                    "model": self._model,
                    "is_anomaly": primary_result.is_anomaly,
                    "score": p_score,
                },
                "secondary": {
                    "model": self._secondary._model,
                    "is_anomaly": secondary_result.is_anomaly,
                    "score": s_score,
                },
            },
        )

    async def explain(
        self,
        data: dict[str, Any],
        ge_score: float | None = None,
        ge_failed_rules: int = 0,
        svm_score: float | None = None,
    ) -> str:
        """Delegate explanation to the primary model."""
        return await super().explain(
            data, ge_score=ge_score, ge_failed_rules=ge_failed_rules,
            svm_score=svm_score,
        )

    async def health_check(self) -> bool:
        """Both models must be reachable."""
        primary_ok, secondary_ok = await asyncio.gather(
            super().health_check(),
            self._secondary.health_check(),
        )
        return primary_ok and secondary_ok
