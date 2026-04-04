"""
Edge-case tests across all detection layers and the preprocessing pipeline.

50 tests covering:
    - LFS preprocessing: coercion, derived columns, boundary conditions
    - SVM: feature filtering, extreme scores, string coercion
    - LLM: response parsing variants, explain fallback, missing keys
    - Detection service: partial detector configs, cache behaviour, LLM override combos
"""

from __future__ import annotations

import json
import math
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import numpy as np
import pytest

from backend.detection.great_expectations_strategy import GreatExpectationsDetector
from backend.detection.lfs_preprocessing import add_derived_columns, _safe_int
from backend.detection.llm_strategy import LLMDetector
from backend.detection.svm_strategy import SVMDetector
from backend.models.schemas import DetectionResponse, StrategyName, StrategyResult
from backend.services.detection_service import DetectionService


# ===================================================================
# Helpers
# ===================================================================


def _make_result(
    strategy: StrategyName, *, is_anomaly: bool, score: float = 0.5
) -> StrategyResult:
    return StrategyResult(
        strategy=strategy,
        is_anomaly=is_anomaly,
        score=score,
        raw={"failed_expectations": ["rule_a"]}
        if is_anomaly and strategy == StrategyName.great_expectations
        else None,
    )


def _mock_detector(
    strategy: StrategyName, *, is_anomaly: bool, score: float = 0.5
) -> AsyncMock:
    det = AsyncMock()
    det.detect.return_value = _make_result(strategy, is_anomaly=is_anomaly, score=score)
    return det


def _mock_llm(
    *, detect_anomaly: bool = True, detect_score: float = 0.9, explain_text: str = "test"
) -> AsyncMock:
    det = AsyncMock(spec=LLMDetector)
    det.detect.return_value = _make_result(
        StrategyName.llm, is_anomaly=detect_anomaly, score=detect_score
    )
    det.explain.return_value = explain_text
    return det


def _mock_redis(*, cached: dict | None = None) -> AsyncMock:
    redis = AsyncMock()
    redis.get.return_value = json.dumps(cached) if cached is not None else None
    redis.setex.return_value = True
    return redis


def _ollama_transport(response_text: str, *, status: int = 200) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/tags":
            return httpx.Response(200, json={"models": []})
        return httpx.Response(status, json={"response": response_text, "done": True})

    return httpx.MockTransport(handler)


# ===================================================================
# LFS Preprocessing — 25 tests
# ===================================================================


class TestSafeInt:
    def test_int_passthrough(self) -> None:
        assert _safe_int(42) == 42

    def test_string_int(self) -> None:
        assert _safe_int("1600001") == 1600001

    def test_float_string_returns_int(self) -> None:
        """_safe_int handles decimal strings like '35.0' → 35."""
        assert _safe_int("35.0") == 35

    def test_none_returns_none(self) -> None:
        assert _safe_int(None) is None

    def test_garbage_string_returns_none(self) -> None:
        assert _safe_int("not-a-number") is None

    def test_empty_string_returns_none(self) -> None:
        assert _safe_int("") is None


class TestDerivedColumns:
    def test_string_values_coerced_to_int(self) -> None:
        data = {"age": "35", "gender": "1600001", "q_602_val": "15000"}
        result = add_derived_columns(data)
        assert result["age"] == 35
        assert result["gender"] == 1600001
        assert result["q_602_val"] == 15000

    def test_non_numeric_fields_untouched(self) -> None:
        data = {"name": "Ahmed", "age": 30}
        result = add_derived_columns(data)
        assert result["name"] == "Ahmed"

    def test_head_of_household_sets_min_age_15(self) -> None:
        data = {"family_relation": 1700001, "age": 20}
        result = add_derived_columns(data)
        assert result["_rule_2001_min_head_age"] == 15

    def test_non_head_sets_min_age_0(self) -> None:
        data = {"family_relation": 1700022, "age": 20}
        result = add_derived_columns(data)
        assert result["_rule_2001_min_head_age"] == 0

    def test_secondary_education_sets_min_age_17(self) -> None:
        data = {"q_301": 10500019, "age": 18}
        result = add_derived_columns(data)
        assert result["_rule_2011_min_secondary_age"] == 17

    def test_bachelor_education_sets_min_age_21(self) -> None:
        data = {"q_301": 10500025, "age": 22}
        result = add_derived_columns(data)
        assert result["_rule_2013_min_bachelor_age"] == 21

    def test_phd_education_sets_min_age_25(self) -> None:
        data = {"q_301": 10500010, "age": 30}
        result = add_derived_columns(data)
        assert result["_rule_2016_min_phd_age"] == 25

    def test_spouse_not_married_violation(self) -> None:
        data = {"family_relation": 1700021, "marage_status": 10600001}
        result = add_derived_columns(data)
        assert result["_rule_2031_spouse_married"] == -1

    def test_spouse_married_passes(self) -> None:
        data = {"family_relation": 1700021, "marage_status": 10600002}
        result = add_derived_columns(data)
        assert result["_rule_2031_spouse_married"] == 1

    def test_grandparent_under_30_violation(self) -> None:
        data = {"family_relation": 1700007, "age": 25}
        result = add_derived_columns(data)
        assert result["_rule_2075_grandparent_age"] == -1

    def test_grandparent_over_30_passes(self) -> None:
        data = {"family_relation": 1700007, "age": 55}
        result = add_derived_columns(data)
        assert result["_rule_2075_grandparent_age"] == 1

    def test_public_sector_under_17_violation(self) -> None:
        data = {"q_534": 99400001, "age": 15}
        result = add_derived_columns(data)
        assert result["_rule_2078_public_sector_age"] == -1

    def test_domestic_worker_under_15_violation(self) -> None:
        data = {"family_relation": 1700010, "age": 12}
        result = add_derived_columns(data)
        assert result["_rule_2141_domestic_worker_age"] == -1

    def test_child_married_violation(self) -> None:
        data = {"age": 10, "marage_status": 10600002}
        result = add_derived_columns(data)
        assert result["_rule_4069_child_not_married"] == -1

    def test_spouse_under_18_violation(self) -> None:
        data = {"family_relation": 1700021, "age": 16, "marage_status": 10600002}
        result = add_derived_columns(data)
        assert result["_rule_4030_spouse_age"] == -1

    def test_uni_salary_under_3000_violation(self) -> None:
        data = {"q_301": 10500025, "q_602_val": 2000}
        result = add_derived_columns(data)
        assert result["_rule_3047_uni_salary"] == -1

    def test_uni_salary_above_3000_passes(self) -> None:
        data = {"q_301": 10500025, "q_602_val": 5000}
        result = add_derived_columns(data)
        assert result["_rule_3047_uni_salary"] == 1

    def test_public_sector_low_hours_violation(self) -> None:
        data = {"q_534": 99400001, "cut_5_total": 20, "act_1_total": 20}
        result = add_derived_columns(data)
        assert result["_rule_3031_public_usual_hours"] == -1
        assert result["_rule_3032_public_actual_hours"] == -1

    def test_private_sector_low_hours_violation(self) -> None:
        data = {"q_534": 99400003, "cut_5_total": 30, "act_1_total": 30}
        result = add_derived_columns(data)
        assert result["_rule_3035_private_usual_hours"] == -1
        assert result["_rule_3036_private_actual_hours"] == -1

    def test_empty_data_produces_all_derived_columns(self) -> None:
        result = add_derived_columns({})
        assert "_rule_zero" in result
        assert "_rule_2001_min_head_age" in result
        assert "_rule_2031_spouse_married" in result
        assert "_rule_3047_uni_salary" in result


# ===================================================================
# SVM Edge Cases — 6 tests
# ===================================================================


class TestSVMEdgeCases:
    @pytest.fixture
    def detector(self) -> SVMDetector:
        with (
            patch.object(SVMDetector, "_load_training_data", return_value=np.zeros((10, 5))),
            patch.object(SVMDetector, "train"),
        ):
            det = SVMDetector(
                training_data_path="fake.npy",
                feature_columns=["act_1_total", "age", "cut_5_total", "edu_ordinal", "q_602_val"],
            )
        mock_model = MagicMock()
        det._model = mock_model
        det._n_features = 5
        return det

    @pytest.mark.asyncio
    async def test_feature_columns_filters_extra_keys(self, detector: SVMDetector) -> None:
        """Extra keys in data dict should be ignored when feature_columns is set."""
        detector._model.predict.return_value = np.array([1])
        detector._model.decision_function.return_value = np.array([0.5])

        result = await detector.detect({
            "age": 35, "act_1_total": 40, "cut_5_total": 40, "q_602_val": 15000,
            "q_301": 10500025,
            "gender": 1600001, "name": "Ahmed",  # extra keys
        })
        assert result.is_anomaly is False

    @pytest.mark.asyncio
    async def test_string_numeric_values_coerced(self, detector: SVMDetector) -> None:
        """String values that are valid floats should be coerced."""
        detector._model.predict.return_value = np.array([1])
        detector._model.decision_function.return_value = np.array([0.3])

        result = await detector.detect({
            "age": "35", "act_1_total": "40", "cut_5_total": "40",
            "q_602_val": "15000", "q_301": "10500025",
        })
        assert result.is_anomaly is False

    @pytest.mark.asyncio
    async def test_extreme_inlier_score_near_zero(self, detector: SVMDetector) -> None:
        """Very large positive decision function should give score near 0."""
        detector._model.predict.return_value = np.array([1])
        detector._model.decision_function.return_value = np.array([50.0])

        result = await detector.detect({
            "age": 35, "act_1_total": 40, "cut_5_total": 40,
            "q_602_val": 15000, "q_301": 10500025,
        })
        assert result.score < 0.01

    @pytest.mark.asyncio
    async def test_extreme_outlier_score_near_one(self, detector: SVMDetector) -> None:
        """Very large negative decision function should give score near 1."""
        detector._model.predict.return_value = np.array([-1])
        detector._model.decision_function.return_value = np.array([-50.0])

        result = await detector.detect({
            "age": 35, "act_1_total": 40, "cut_5_total": 40,
            "q_602_val": 15000, "q_301": 10500025,
        })
        assert result.score > 0.99

    @pytest.mark.asyncio
    async def test_missing_feature_column_raises(self, detector: SVMDetector) -> None:
        """Missing a required feature column should raise KeyError."""
        with pytest.raises(KeyError):
            await detector.detect({"age": 35, "act_1_total": 40})  # missing features

    @pytest.mark.asyncio
    async def test_feature_columns_sorted_order(self, detector: SVMDetector) -> None:
        """Features should be extracted in sorted order regardless of dict order."""
        detector._model.predict.return_value = np.array([1])
        detector._model.decision_function.return_value = np.array([0.5])

        # Pass in reverse order — should still work
        result = await detector.detect({
            "q_602_val": 15000, "cut_5_total": 40, "age": 35,
            "act_1_total": 40, "q_301": 10500025,
        })
        assert result.strategy == StrategyName.svm


# ===================================================================
# LLM Edge Cases — 11 tests
# ===================================================================


class TestLLMEdgeCases:
    def test_parse_response_with_think_block(self) -> None:
        """Qwen's <think>...</think> blocks should be stripped before parsing."""
        det = LLMDetector(base_url="http://localhost:11434")
        raw = '<think>Let me analyze this...</think>{"is_anomaly": true, "score": 0.8, "explanation": "high salary"}'
        result = det._parse_response(raw)
        assert result["is_anomaly"] is True
        assert result["score"] == 0.8

    def test_parse_response_with_markdown_fence(self) -> None:
        """JSON inside markdown code fences should be extracted."""
        det = LLMDetector(base_url="http://localhost:11434")
        raw = '```json\n{"is_anomaly": false, "score": 0.1, "explanation": "normal"}\n```'
        result = det._parse_response(raw)
        assert result["is_anomaly"] is False

    def test_parse_response_with_think_and_fence(self) -> None:
        """Combined think block + markdown fence should both be handled."""
        det = LLMDetector(base_url="http://localhost:11434")
        raw = (
            "<think>Reasoning about the record...</think>\n"
            "```json\n"
            '{"is_anomaly": true, "score": 0.95, "explanation": "anomalous"}\n'
            "```"
        )
        result = det._parse_response(raw)
        assert result["is_anomaly"] is True
        assert result["score"] == 0.95

    @pytest.mark.asyncio
    async def test_empty_response_string_failsafe(self) -> None:
        """Empty response from Ollama should trigger fail-safe."""
        transport = _ollama_transport("")
        det = LLMDetector(base_url="http://localhost:11434", model="test", transport=transport)
        result = await det.detect({"age": 30})
        assert result.is_anomaly is False
        assert result.score == 0.0

    @pytest.mark.asyncio
    async def test_whitespace_only_response_failsafe(self) -> None:
        """Whitespace-only response should trigger fail-safe."""
        transport = _ollama_transport("   \n\t  ")
        det = LLMDetector(base_url="http://localhost:11434", model="test", transport=transport)
        result = await det.detect({"age": 30})
        assert result.is_anomaly is False
        assert result.score == 0.0

    @pytest.mark.asyncio
    async def test_missing_is_anomaly_defaults_false(self) -> None:
        """Missing 'is_anomaly' in LLM response should default to False."""
        transport = _ollama_transport(json.dumps({"score": 0.7, "explanation": "maybe"}))
        det = LLMDetector(base_url="http://localhost:11434", model="test", transport=transport)
        result = await det.detect({"age": 30})
        assert result.is_anomaly is False

    @pytest.mark.asyncio
    async def test_missing_score_defaults_zero(self) -> None:
        """Missing 'score' in LLM response should default to 0.0."""
        transport = _ollama_transport(json.dumps({"is_anomaly": False, "explanation": "ok"}))
        det = LLMDetector(base_url="http://localhost:11434", model="test", transport=transport)
        result = await det.detect({"age": 30})
        assert result.score == 0.0

    @pytest.mark.asyncio
    async def test_explain_success(self) -> None:
        """explain() should return the explanation text from Ollama."""
        response = json.dumps({"explanation": "الراتب مرتفع جداً"})
        transport = _ollama_transport(response)
        det = LLMDetector(base_url="http://localhost:11434", model="test", transport=transport)
        text = await det.explain({"q_602_val": 650000}, ge_score=0.5, svm_score=0.9)
        assert "الراتب" in text

    @pytest.mark.asyncio
    async def test_explain_failure_returns_fallback(self) -> None:
        """explain() should return Arabic fallback when Ollama fails."""

        def failing_handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("Connection refused", request=request)

        transport = httpx.MockTransport(failing_handler)
        det = LLMDetector(base_url="http://localhost:11434", model="test", transport=transport)
        text = await det.explain({"age": 30})
        assert "تعذر" in text

    @pytest.mark.asyncio
    async def test_detect_score_zero_from_model(self) -> None:
        """Score of exactly 0.0 from model should be preserved."""
        transport = _ollama_transport(
            json.dumps({"is_anomaly": False, "score": 0.0, "explanation": "normal"})
        )
        det = LLMDetector(base_url="http://localhost:11434", model="test", transport=transport)
        result = await det.detect({"age": 30})
        assert result.score == 0.0

    @pytest.mark.asyncio
    async def test_detect_score_one_from_model(self) -> None:
        """Score of exactly 1.0 from model should be preserved."""
        transport = _ollama_transport(
            json.dumps({"is_anomaly": True, "score": 1.0, "explanation": "certain anomaly"})
        )
        det = LLMDetector(base_url="http://localhost:11434", model="test", transport=transport)
        result = await det.detect({"age": 5})
        assert result.score == 1.0


# ===================================================================
# Detection Service Edge Cases — 8 tests
# ===================================================================


class TestDetectionServiceEdgeCases:
    @pytest.mark.asyncio
    async def test_only_ge_configured_anomaly(self) -> None:
        """Only GE configured, flags anomaly, no LLM → verdict from GE alone."""
        ge = _mock_detector(StrategyName.great_expectations, is_anomaly=True, score=0.3)
        svc = DetectionService(ge_detector=ge, svm_detector=None, llm_detector=None)
        result = await svc.run("ge-only", {"age": 5})
        assert result.is_anomaly is True
        assert result.llm_skip_reason == "LLM detector not configured."

    @pytest.mark.asyncio
    async def test_only_svm_configured_anomaly(self) -> None:
        """Only SVM configured, flags anomaly, no LLM → verdict from SVM alone."""
        svm = _mock_detector(StrategyName.svm, is_anomaly=True, score=0.9)
        svc = DetectionService(ge_detector=None, svm_detector=svm, llm_detector=None)
        result = await svc.run("svm-only", {"q_602_val": 650000})
        assert result.is_anomaly is True

    @pytest.mark.asyncio
    async def test_only_llm_configured_no_preliminary(self) -> None:
        """Only LLM configured (no GE/SVM) → no preliminary anomaly → LLM not called."""
        llm = _mock_llm(detect_anomaly=True)
        svc = DetectionService(ge_detector=None, svm_detector=None, llm_detector=llm)
        result = await svc.run("llm-only", {"age": 5})
        assert result.is_anomaly is False
        assert result.llm_triggered is False
        llm.detect.assert_not_called()

    @pytest.mark.asyncio
    async def test_ge_and_llm_no_svm_confirmed(self) -> None:
        """GE + LLM (no SVM): GE flags, LLM confirms → anomaly."""
        ge = _mock_detector(StrategyName.great_expectations, is_anomaly=True, score=0.5)
        llm = _mock_llm(detect_anomaly=True)
        svc = DetectionService(ge_detector=ge, svm_detector=None, llm_detector=llm)
        result = await svc.run("ge-llm-confirm", {"age": 5})
        assert result.is_anomaly is True

    @pytest.mark.asyncio
    async def test_ge_and_llm_no_svm_ge_ground_truth(self) -> None:
        """GE + LLM (no SVM): GE flags, LLM disagrees → still anomaly (GE is ground truth)."""
        ge = _mock_detector(StrategyName.great_expectations, is_anomaly=True, score=0.5)
        llm = _mock_llm(detect_anomaly=False)
        svc = DetectionService(ge_detector=ge, svm_detector=None, llm_detector=llm)
        result = await svc.run("ge-llm-override", {"age": 35})
        assert result.is_anomaly is True  # GE ground truth cannot be overridden

    @pytest.mark.asyncio
    async def test_cache_stores_correct_record_id(self) -> None:
        """Cached response should have the correct record_id."""
        redis = _mock_redis()
        ge = _mock_detector(StrategyName.great_expectations, is_anomaly=False, score=0.0)
        svc = DetectionService(
            ge_detector=ge, svm_detector=None, llm_detector=None, redis_client=redis
        )
        await svc.run("unique-id-42", {})
        stored = json.loads(redis.setex.call_args.args[2])
        assert stored["record_id"] == "unique-id-42"

    @pytest.mark.asyncio
    async def test_second_run_uses_cache(self) -> None:
        """Second call with same record_id should use cache, not re-run detectors."""
        cached = DetectionResponse(
            record_id="dup",
            results=[],
            is_anomaly=False,
            llm_triggered=False,
        )
        redis = _mock_redis(cached=cached.model_dump())
        ge = _mock_detector(StrategyName.great_expectations, is_anomaly=True)
        svc = DetectionService(
            ge_detector=ge, svm_detector=None, llm_detector=None, redis_client=redis
        )
        result = await svc.run("dup", {})
        assert result.is_anomaly is False  # from cache, not from GE
        ge.detect.assert_not_called()

    @pytest.mark.asyncio
    async def test_llm_result_in_results_array_when_ge_flags(self) -> None:
        """LLM result should be in results[] even when GE ground truth holds."""
        ge = _mock_detector(StrategyName.great_expectations, is_anomaly=True, score=0.5)
        svm = _mock_detector(StrategyName.svm, is_anomaly=True, score=0.8)
        llm = _mock_llm(detect_anomaly=False, detect_score=0.1)
        svc = DetectionService(ge_detector=ge, svm_detector=svm, llm_detector=llm)
        result = await svc.run("override-results", {})
        strategies = {r.strategy for r in result.results}
        assert StrategyName.llm in strategies
        assert result.is_anomaly is True  # GE ground truth cannot be overridden
