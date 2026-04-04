"""
Pydantic schemas for API request and response payloads.

Covers data submission, per-strategy results, and the aggregated
anomaly detection response returned to callers.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel


class StrategyName(str, Enum):
    """Identifiers for the three supported detection strategies."""

    great_expectations = "great_expectations"
    svm = "svm"
    llm = "llm"


class DataPayload(BaseModel):
    """Incoming data record submitted for anomaly analysis."""

    record_id: str
    data: dict[str, Any]
    strategies: list[StrategyName] | None = None  # None means run all


class StrategyResult(BaseModel):
    """Result produced by a single detection strategy."""

    strategy: StrategyName
    is_anomaly: bool
    score: float | None = None
    explanation: str | None = None
    raw: dict[str, Any] | None = None


class DetectionResponse(BaseModel):
    """Aggregated response containing results from all requested strategies."""

    record_id: str
    results: list[StrategyResult]
    is_anomaly: bool  # Final verdict: (GE OR SVM) AND LLM
    severity: str | None = None  # "hard_error" or "warning" (from business rules)
    explanation: str | None = None  # LLM-generated explanation (when anomaly found)
    llm_triggered: bool  # True if GE or SVM flagged anomaly and LLM was called
    llm_skip_reason: str | None = None  # Explanation when LLM was not called
