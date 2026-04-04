"""
Tests for the cascading detection pipeline.

Covers:
    - Cache hit/miss
    - LLM called when GE flags, SVM flags, or SVM score >= borderline threshold
    - LLM NOT called when SVM score below threshold and GE normal
    - Verdict: GE flags are ground truth (LLM cannot override); SVM flags
      use (SVM OR borderline) AND LLM
    - LLM can override SVM false positives (but not GE)
    - Explanation field populated from LLM detect result
    - Error handling / fail-safe
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock

import pytest

from backend.detection.base import BaseDetector
from backend.detection.llm_strategy import LLMDetector
from backend.models.schemas import DetectionResponse, StrategyName, StrategyResult
from backend.services.detection_service import DetectionService


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_result(strategy: StrategyName, *, is_anomaly: bool, score: float = 0.5) -> StrategyResult:
    return StrategyResult(
        strategy=strategy,
        is_anomaly=is_anomaly,
        score=score,
        raw={"failed_expectations": ["rule_a"]} if is_anomaly and strategy == StrategyName.great_expectations else None,
    )


def _mock_detector(strategy: StrategyName, *, is_anomaly: bool, score: float = 0.5) -> AsyncMock:
    det = AsyncMock(spec=BaseDetector)
    det.detect.return_value = _make_result(strategy, is_anomaly=is_anomaly, score=score)
    return det


def _mock_llm(
    *,
    detect_anomaly: bool = True,
    detect_score: float = 0.9,
    explain_text: str = "تفسير تجريبي",
) -> AsyncMock:
    det = AsyncMock(spec=LLMDetector)
    det.detect.return_value = StrategyResult(
        strategy=StrategyName.llm,
        is_anomaly=detect_anomaly,
        score=detect_score,
        explanation=explain_text if detect_anomaly else None,
    )
    det.explain.return_value = explain_text
    return det


def _mock_redis(*, cached: dict | None = None) -> AsyncMock:
    redis = AsyncMock()
    redis.get.return_value = json.dumps(cached) if cached is not None else None
    redis.setex.return_value = True
    return redis


def _make_service(
    *,
    ge: AsyncMock | None = None,
    svm: AsyncMock | None = None,
    llm: AsyncMock | None = None,
    redis: AsyncMock | None = None,
    cache_ttl: int = 60,
) -> DetectionService:
    return DetectionService(
        ge_detector=ge,
        svm_detector=svm,
        llm_detector=llm,
        redis_client=redis,
        cache_ttl_seconds=cache_ttl,
    )


# ---------------------------------------------------------------------------
# Cache tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_cache_hit_returns_cached_response() -> None:
    cached_response = DetectionResponse(
        record_id="rec-001",
        results=[_make_result(StrategyName.svm, is_anomaly=True, score=0.9)],
        is_anomaly=True,
        explanation="cached explanation",
        llm_triggered=True,
    )
    redis = _mock_redis(cached=cached_response.model_dump())

    ge = _mock_detector(StrategyName.great_expectations, is_anomaly=False)
    svm = _mock_detector(StrategyName.svm, is_anomaly=True)
    llm = _mock_llm()

    svc = _make_service(ge=ge, svm=svm, llm=llm, redis=redis)
    result = await svc.run("rec-001", {"x": 1})

    assert result.is_anomaly is True
    assert result.explanation == "cached explanation"
    ge.detect.assert_not_called()
    svm.detect.assert_not_called()
    llm.detect.assert_not_called()
    llm.explain.assert_not_called()


@pytest.mark.asyncio
async def test_cache_miss_stores_result() -> None:
    redis = _mock_redis(cached=None)
    ge = _mock_detector(StrategyName.great_expectations, is_anomaly=False, score=0.0)
    svm = _mock_detector(StrategyName.svm, is_anomaly=False, score=0.1)

    svc = _make_service(ge=ge, svm=svm, redis=redis)
    result = await svc.run("rec-002", {"x": 1})

    redis.setex.assert_awaited_once()
    stored = json.loads(redis.setex.call_args.args[2])
    assert stored["record_id"] == "rec-002"


@pytest.mark.asyncio
async def test_no_redis_skips_caching() -> None:
    ge = _mock_detector(StrategyName.great_expectations, is_anomaly=False, score=0.0)
    svc = _make_service(ge=ge, redis=None)
    result = await svc.run("rec-003", {"a": 1})
    ge.detect.assert_awaited_once()
    assert result.record_id == "rec-003"


@pytest.mark.asyncio
async def test_cache_key_format() -> None:
    svc = _make_service()
    assert svc._cache_key("abc-123") == "detection:abc-123"


# ---------------------------------------------------------------------------
# LLM called to detect + explain when anomaly detected
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_llm_called_when_ge_flags_anomaly() -> None:
    """GE flags anomaly → LLM.detect() called."""
    ge = _mock_detector(StrategyName.great_expectations, is_anomaly=True, score=0.3)
    svm = _mock_detector(StrategyName.svm, is_anomaly=False, score=0.1)
    llm = _mock_llm(detect_anomaly=True, explain_text="الشخص عمره 5 سنوات ومتزوج")

    svc = _make_service(ge=ge, svm=svm, llm=llm)
    result = await svc.run("rec-ge", {"age": 5})

    llm.detect.assert_awaited_once()
    assert result.llm_triggered is True
    assert result.explanation == "الشخص عمره 5 سنوات ومتزوج"
    assert result.is_anomaly is True


@pytest.mark.asyncio
async def test_llm_called_when_svm_flags_anomaly() -> None:
    """SVM flags anomaly → LLM.detect() called."""
    ge = _mock_detector(StrategyName.great_expectations, is_anomaly=False, score=0.0)
    svm = _mock_detector(StrategyName.svm, is_anomaly=True, score=0.9)
    llm = _mock_llm(detect_anomaly=True, explain_text="الراتب مرتفع جداً")

    svc = _make_service(ge=ge, svm=svm, llm=llm)
    result = await svc.run("rec-svm", {"q_602_val": 650000})

    llm.detect.assert_awaited_once()
    assert result.llm_triggered is True
    assert result.explanation == "الراتب مرتفع جداً"


@pytest.mark.asyncio
async def test_llm_called_when_both_flag_anomaly() -> None:
    """Both GE and SVM flag anomaly → LLM.detect() called once."""
    ge = _mock_detector(StrategyName.great_expectations, is_anomaly=True, score=0.5)
    svm = _mock_detector(StrategyName.svm, is_anomaly=True, score=0.9)
    llm = _mock_llm(detect_anomaly=True, explain_text="عدة مخالفات")

    svc = _make_service(ge=ge, svm=svm, llm=llm)
    result = await svc.run("rec-both", {"age": 5})

    llm.detect.assert_awaited_once()
    assert result.explanation == "عدة مخالفات"


# ---------------------------------------------------------------------------
# LLM as false-positive filter — overrides GE/SVM
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_llm_overrides_svm_false_positive() -> None:
    """SVM flags anomaly but LLM says normal → verdict is False."""
    ge = _mock_detector(StrategyName.great_expectations, is_anomaly=False, score=0.0)
    svm = _mock_detector(StrategyName.svm, is_anomaly=True, score=0.7)
    llm = _mock_llm(detect_anomaly=False, detect_score=0.1)

    svc = _make_service(ge=ge, svm=svm, llm=llm)
    result = await svc.run("fp-override", {"age": 35})

    assert result.is_anomaly is False
    assert result.llm_triggered is True
    assert result.explanation is None  # cleared on override


@pytest.mark.asyncio
async def test_ge_ground_truth_not_overridden_by_llm() -> None:
    """GE flags anomaly, LLM says normal → verdict is still True (GE is ground truth)."""
    ge = _mock_detector(StrategyName.great_expectations, is_anomaly=True, score=0.5)
    svm = _mock_detector(StrategyName.svm, is_anomaly=True, score=0.7)
    llm = _mock_llm(detect_anomaly=False, detect_score=0.2)

    svc = _make_service(ge=ge, svm=svm, llm=llm)
    result = await svc.run("ge-ground-truth", {"age": 35})

    assert result.is_anomaly is True
    assert result.llm_triggered is True


@pytest.mark.asyncio
async def test_llm_confirms_anomaly() -> None:
    """GE flags anomaly, LLM confirms → verdict is True with explanation."""
    ge = _mock_detector(StrategyName.great_expectations, is_anomaly=True, score=0.5)
    svm = _mock_detector(StrategyName.svm, is_anomaly=False, score=0.1)
    llm = _mock_llm(detect_anomaly=True, detect_score=0.9, explain_text="عمر غير منطقي")

    svc = _make_service(ge=ge, svm=svm, llm=llm)
    result = await svc.run("confirm", {"age": 5})

    assert result.is_anomaly is True
    assert result.llm_triggered is True
    assert result.explanation == "عمر غير منطقي"


# ---------------------------------------------------------------------------
# LLM NOT called when no anomaly
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_llm_not_called_when_no_anomaly() -> None:
    """Neither GE nor SVM flags anomaly and SVM below threshold → LLM NOT called."""
    ge = _mock_detector(StrategyName.great_expectations, is_anomaly=False, score=0.0)
    svm = _mock_detector(StrategyName.svm, is_anomaly=False, score=0.1)
    llm = _mock_llm()

    svc = _make_service(ge=ge, svm=svm, llm=llm)
    result = await svc.run("rec-clean", {"age": 35})

    llm.detect.assert_not_called()
    assert result.llm_triggered is False
    assert result.explanation is None
    assert result.llm_skip_reason is not None
    assert result.is_anomaly is False


@pytest.mark.asyncio
async def test_llm_called_on_svm_borderline() -> None:
    """SVM score >= custom threshold but below 0.5 → LLM called (borderline path)."""
    ge = _mock_detector(StrategyName.great_expectations, is_anomaly=False, score=0.0)
    svm = _mock_detector(StrategyName.svm, is_anomaly=False, score=0.3)
    llm = _mock_llm(detect_anomaly=True, explain_text="شذوذ دقيق")

    svc = DetectionService(
        ge_detector=ge, svm_detector=svm, llm_detector=llm,
        redis_client=None, svm_llm_threshold=0.2,
    )
    result = await svc.run("borderline", {"q_602_val": 800})

    llm.detect.assert_awaited_once()
    assert result.llm_triggered is True
    assert result.is_anomaly is True


@pytest.mark.asyncio
async def test_llm_skip_reason_when_no_llm_detector() -> None:
    """LLM detector is None → skip reason explains it, verdict falls back to GE OR SVM."""
    ge = _mock_detector(StrategyName.great_expectations, is_anomaly=True, score=0.5)
    svm = _mock_detector(StrategyName.svm, is_anomaly=True, score=0.9)

    svc = _make_service(ge=ge, svm=svm, llm=None)
    result = await svc.run("rec-no-llm", {"x": 1})

    assert result.llm_triggered is False
    assert result.llm_skip_reason == "LLM detector not configured."
    assert result.explanation is None
    assert result.is_anomaly is True  # falls back to GE OR SVM


# ---------------------------------------------------------------------------
# Verdict tests — (GE OR SVM) AND LLM
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_verdict_both_normal() -> None:
    ge = _mock_detector(StrategyName.great_expectations, is_anomaly=False, score=0.0)
    svm = _mock_detector(StrategyName.svm, is_anomaly=False, score=0.1)
    svc = _make_service(ge=ge, svm=svm)
    result = await svc.run("v-1", {})
    assert result.is_anomaly is False


@pytest.mark.asyncio
async def test_verdict_ge_anomaly_llm_confirms() -> None:
    ge = _mock_detector(StrategyName.great_expectations, is_anomaly=True, score=0.3)
    svm = _mock_detector(StrategyName.svm, is_anomaly=False, score=0.1)
    llm = _mock_llm(detect_anomaly=True)
    svc = _make_service(ge=ge, svm=svm, llm=llm)
    result = await svc.run("v-2", {})
    assert result.is_anomaly is True


@pytest.mark.asyncio
async def test_verdict_ge_anomaly_llm_cannot_override() -> None:
    """GE is ground truth — LLM denial does not flip the verdict."""
    ge = _mock_detector(StrategyName.great_expectations, is_anomaly=True, score=0.3)
    svm = _mock_detector(StrategyName.svm, is_anomaly=False, score=0.1)
    llm = _mock_llm(detect_anomaly=False)
    svc = _make_service(ge=ge, svm=svm, llm=llm)
    result = await svc.run("v-2b", {})
    assert result.is_anomaly is True


@pytest.mark.asyncio
async def test_verdict_svm_anomaly_llm_confirms() -> None:
    ge = _mock_detector(StrategyName.great_expectations, is_anomaly=False, score=0.0)
    svm = _mock_detector(StrategyName.svm, is_anomaly=True, score=0.9)
    llm = _mock_llm(detect_anomaly=True)
    svc = _make_service(ge=ge, svm=svm, llm=llm)
    result = await svc.run("v-3", {})
    assert result.is_anomaly is True


@pytest.mark.asyncio
async def test_verdict_svm_anomaly_llm_overrides() -> None:
    ge = _mock_detector(StrategyName.great_expectations, is_anomaly=False, score=0.0)
    svm = _mock_detector(StrategyName.svm, is_anomaly=True, score=0.9)
    llm = _mock_llm(detect_anomaly=False)
    svc = _make_service(ge=ge, svm=svm, llm=llm)
    result = await svc.run("v-3b", {})
    assert result.is_anomaly is False


@pytest.mark.asyncio
async def test_verdict_both_anomaly_llm_confirms() -> None:
    ge = _mock_detector(StrategyName.great_expectations, is_anomaly=True, score=0.5)
    svm = _mock_detector(StrategyName.svm, is_anomaly=True, score=0.9)
    llm = _mock_llm(detect_anomaly=True)
    svc = _make_service(ge=ge, svm=svm, llm=llm)
    result = await svc.run("v-4", {})
    assert result.is_anomaly is True


@pytest.mark.asyncio
async def test_verdict_both_anomaly_llm_cannot_override_ge() -> None:
    """GE + SVM flag, LLM denies — GE ground truth keeps verdict True."""
    ge = _mock_detector(StrategyName.great_expectations, is_anomaly=True, score=0.5)
    svm = _mock_detector(StrategyName.svm, is_anomaly=True, score=0.9)
    llm = _mock_llm(detect_anomaly=False)
    svc = _make_service(ge=ge, svm=svm, llm=llm)
    result = await svc.run("v-4b", {})
    assert result.is_anomaly is True


@pytest.mark.asyncio
async def test_no_detectors_returns_clean() -> None:
    svc = _make_service()
    result = await svc.run("empty", {})
    assert result.results == []
    assert result.is_anomaly is False
    assert result.explanation is None


# ---------------------------------------------------------------------------
# Response shape
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_response_has_all_fields() -> None:
    ge = _mock_detector(StrategyName.great_expectations, is_anomaly=False, score=0.0)
    svm = _mock_detector(StrategyName.svm, is_anomaly=False, score=0.1)
    svc = _make_service(ge=ge, svm=svm)
    result = await svc.run("shape", {})

    d = result.model_dump()
    assert "is_anomaly" in d
    assert "explanation" in d
    assert "llm_triggered" in d
    assert "llm_skip_reason" in d
    assert "results" in d


@pytest.mark.asyncio
async def test_results_contain_llm_when_triggered() -> None:
    """When LLM is triggered, results list includes GE + SVM + LLM."""
    ge = _mock_detector(StrategyName.great_expectations, is_anomaly=True, score=0.3)
    svm = _mock_detector(StrategyName.svm, is_anomaly=True, score=0.9)
    llm = _mock_llm(detect_anomaly=True, explain_text="explanation text")

    svc = _make_service(ge=ge, svm=svm, llm=llm)
    result = await svc.run("shape-2", {})

    assert len(result.results) == 3
    strategies = {r.strategy for r in result.results}
    assert strategies == {StrategyName.great_expectations, StrategyName.svm, StrategyName.llm}
    assert result.explanation == "explanation text"


@pytest.mark.asyncio
async def test_results_exclude_llm_when_not_triggered() -> None:
    """When SVM score below threshold and GE normal, results has only GE + SVM."""
    ge = _mock_detector(StrategyName.great_expectations, is_anomaly=False, score=0.0)
    svm = _mock_detector(StrategyName.svm, is_anomaly=False, score=0.1)
    llm = _mock_llm()

    svc = _make_service(ge=ge, svm=svm, llm=llm)
    result = await svc.run("shape-3", {})

    assert len(result.results) == 2
    strategies = {r.strategy for r in result.results}
    assert strategies == {StrategyName.great_expectations, StrategyName.svm}


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_detector_exception_returns_failsafe() -> None:
    ge = AsyncMock(spec=BaseDetector)
    ge.detect.side_effect = RuntimeError("Model not loaded")
    svm = _mock_detector(StrategyName.svm, is_anomaly=False, score=0.1)

    svc = _make_service(ge=ge, svm=svm)
    result = await svc.run("err", {"x": 1})

    ge_result = next(r for r in result.results if r.strategy == StrategyName.great_expectations)
    assert ge_result.is_anomaly is False
    assert "RuntimeError" in (ge_result.explanation or "")


@pytest.mark.asyncio
async def test_llm_failsafe_ge_ground_truth_preserved() -> None:
    """LLM detect fails but GE flagged → GE ground truth keeps verdict True."""
    ge = _mock_detector(StrategyName.great_expectations, is_anomaly=True, score=0.5)
    svm = _mock_detector(StrategyName.svm, is_anomaly=False, score=0.1)
    llm = _mock_llm(detect_anomaly=False, detect_score=0.0)

    svc = _make_service(ge=ge, svm=svm, llm=llm)
    result = await svc.run("llm-fail-ge", {})

    # GE is ground truth — LLM failure doesn't override
    assert result.is_anomaly is True
    assert result.llm_triggered is True


@pytest.mark.asyncio
async def test_llm_failsafe_svm_only_overrides_to_false() -> None:
    """LLM detect fails, only SVM flagged → verdict overridden to False."""
    ge = _mock_detector(StrategyName.great_expectations, is_anomaly=False, score=0.0)
    svm = _mock_detector(StrategyName.svm, is_anomaly=True, score=0.8)
    llm = _mock_llm(detect_anomaly=False, detect_score=0.0)

    svc = _make_service(ge=ge, svm=svm, llm=llm)
    result = await svc.run("llm-fail-svm", {})

    # SVM-only flag, LLM denies → False
    assert result.is_anomaly is False
    assert result.llm_triggered is True


@pytest.mark.asyncio
async def test_redis_failure_does_not_crash() -> None:
    redis = AsyncMock()
    redis.get.return_value = None
    redis.setex.side_effect = ConnectionError("Redis unavailable")

    ge = _mock_detector(StrategyName.great_expectations, is_anomaly=True, score=0.5)
    llm = _mock_llm(detect_anomaly=True, explain_text="explanation")
    svc = _make_service(ge=ge, llm=llm, redis=redis)

    result = await svc.run("redis-err", {})
    assert result.record_id == "redis-err"
    assert result.is_anomaly is True


@pytest.mark.asyncio
async def test_record_id_matches() -> None:
    ge = _mock_detector(StrategyName.great_expectations, is_anomaly=False, score=0.0)
    svc = _make_service(ge=ge)
    result = await svc.run("my-id-99", {})
    assert result.record_id == "my-id-99"
