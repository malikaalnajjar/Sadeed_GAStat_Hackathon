"""
Tests for the LLM few-shot detection strategy.

Uses ``httpx.MockTransport`` to avoid requiring a live Ollama server.
Each test constructs an :class:`~backend.detection.llm_strategy.LLMDetector`
whose ``transport`` parameter is injected with a mock that returns a
pre-defined Ollama API response.

Covers:
    - Successful anomaly detection (model returns ``is_anomaly=true``)
    - Successful normal classification (model returns ``is_anomaly=false``)
    - Score is clamped to ``[0.0, 1.0]`` and the correct strategy name is set
    - Malformed / non-JSON model response → graceful degradation (fail-safe)
    - Ollama unreachable → ``health_check`` returns ``False``
    - Ollama reachable → ``health_check`` returns ``True``
    - ``_build_prompt`` includes all few-shot examples and the candidate record
    - ``_parse_response`` handles JSON wrapped in prose
"""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from backend.detection.llm_strategy import FEW_SHOT_EXAMPLES, LLMDetector
from backend.models.schemas import StrategyName


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ollama_transport(
    response_text: str,
    *,
    generate_status: int = 200,
    tags_status: int = 200,
) -> httpx.MockTransport:
    """Build an :class:`httpx.MockTransport` that mimics Ollama's REST API.

    Args:
        response_text: Value placed in the ``"response"`` field of the
            ``/api/generate`` reply.
        generate_status: HTTP status for ``POST /api/generate``.
        tags_status: HTTP status for ``GET /api/tags``.

    Returns:
        A synchronous :class:`httpx.MockTransport` compatible with
        :class:`httpx.AsyncClient`.
    """

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == "/api/tags":
            return httpx.Response(tags_status, json={"models": []})
        # /api/generate
        body: dict[str, Any] = {"response": response_text, "done": True}
        return httpx.Response(generate_status, json=body)

    return httpx.MockTransport(handler)


def _make_detector(
    response_text: str,
    *,
    generate_status: int = 200,
    tags_status: int = 200,
) -> LLMDetector:
    """Return an :class:`LLMDetector` backed by a mock transport.

    Args:
        response_text: The text the mock Ollama server will return as the
            model's response.
        generate_status: HTTP status for ``/api/generate``.
        tags_status: HTTP status for ``/api/tags``.
    """
    transport = _ollama_transport(
        response_text,
        generate_status=generate_status,
        tags_status=tags_status,
    )
    return LLMDetector(
        base_url="http://localhost:11434",
        model="test-model",
        transport=transport,
    )


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def detector() -> LLMDetector:
    """Provide an :class:`LLMDetector` whose mock transport returns an anomaly verdict."""
    return _make_detector(
        json.dumps(
            {"is_anomaly": True, "score": 0.91, "explanation": "high cpu and error rate"}
        )
    )


# ---------------------------------------------------------------------------
# Inference tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_anomaly_detected(detector: LLMDetector) -> None:
    """When the model returns ``is_anomaly=true`` the result must reflect it.

    Asserts:
        - ``is_anomaly`` is ``True``
        - ``score`` matches the model's value
        - ``strategy`` is ``StrategyName.llm``
        - ``explanation`` is forwarded from the model response
    """
    result = await detector.detect({"cpu_usage_pct": 98.0, "error_rate": 0.95})

    assert result.strategy == StrategyName.llm
    assert result.is_anomaly is True
    assert result.score == pytest.approx(0.91, abs=1e-4)
    assert "cpu" in result.explanation.lower() or result.explanation != ""


@pytest.mark.asyncio
async def test_normal_classified() -> None:
    """When the model returns ``is_anomaly=false`` no anomaly should be reported.

    Asserts:
        - ``is_anomaly`` is ``False``
        - ``score`` matches the model's value (below 0.5)
        - ``strategy`` is ``StrategyName.llm``
    """
    det = _make_detector(
        json.dumps(
            {"is_anomaly": False, "score": 0.08, "explanation": "all metrics nominal"}
        )
    )
    result = await det.detect({"cpu_usage_pct": 12.0, "error_rate": 0.01})

    assert result.strategy == StrategyName.llm
    assert result.is_anomaly is False
    assert result.score == pytest.approx(0.08, abs=1e-4)
    assert result.score < 0.5


@pytest.mark.asyncio
async def test_score_clamped_to_unit_interval() -> None:
    """Score values outside ``[0, 1]`` returned by the model must be clamped.

    Asserts:
        - ``score`` is clamped to ``[0.0, 1.0]`` even when the model returns
          an out-of-range value.
    """
    det = _make_detector(
        json.dumps({"is_anomaly": True, "score": 1.5, "explanation": "very anomalous"})
    )
    result = await det.detect({"x": 1.0})
    assert result.score <= 1.0

    det2 = _make_detector(
        json.dumps({"is_anomaly": False, "score": -0.3, "explanation": "very normal"})
    )
    result2 = await det2.detect({"x": 1.0})
    assert result2.score >= 0.0


@pytest.mark.asyncio
async def test_strategy_name_is_llm(detector: LLMDetector) -> None:
    """Every result produced by :class:`LLMDetector` must carry ``StrategyName.llm``."""
    result = await detector.detect({"a": 1})
    assert result.strategy == StrategyName.llm


# ---------------------------------------------------------------------------
# Fail-safe / error-path tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_malformed_response_handled() -> None:
    """A non-JSON model response must not raise; fail-safe result expected.

    Asserts:
        - No exception is raised
        - ``is_anomaly`` is ``False``
        - ``score`` is ``0.0``
        - ``strategy`` is ``StrategyName.llm``
    """
    det = _make_detector("I am not JSON at all, just rambling text from the model.")
    result = await det.detect({"cpu_usage_pct": 50.0})

    assert result.strategy == StrategyName.llm
    assert result.is_anomaly is False
    assert result.score == pytest.approx(0.0)


@pytest.mark.asyncio
async def test_http_error_handled() -> None:
    """A non-2xx response from Ollama must trigger the fail-safe path.

    Asserts:
        - No exception propagates
        - ``is_anomaly`` is ``False``
        - ``score`` is ``0.0``
    """
    det = _make_detector("", generate_status=503)
    result = await det.detect({"a": 1})

    assert result.is_anomaly is False
    assert result.score == pytest.approx(0.0)


@pytest.mark.asyncio
async def test_connection_error_handled() -> None:
    """A network connection error must trigger the fail-safe path.

    Uses a bare :class:`LLMDetector` with no transport override so that the
    ``httpx.AsyncClient`` attempts to connect to a port that refuses connections.
    """

    def refusing_handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("Connection refused", request=request)

    transport = httpx.MockTransport(refusing_handler)
    det = LLMDetector(
        base_url="http://localhost:11434",
        model="test-model",
        transport=transport,
    )
    result = await det.detect({"a": 1})

    assert result.is_anomaly is False
    assert result.score == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Health check tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_health_check_reachable() -> None:
    """``health_check`` returns ``True`` when Ollama responds with 200."""
    det = _make_detector("", tags_status=200)
    assert await det.health_check() is True


@pytest.mark.asyncio
async def test_health_check_unreachable() -> None:
    """``health_check`` returns ``False`` when Ollama is not reachable.

    Simulated by raising :class:`httpx.ConnectError` from the mock transport.
    """

    def refusing_handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("Connection refused", request=request)

    transport = httpx.MockTransport(refusing_handler)
    det = LLMDetector(
        base_url="http://localhost:11434",
        model="test-model",
        transport=transport,
    )
    assert await det.health_check() is False


@pytest.mark.asyncio
async def test_health_check_server_error() -> None:
    """``health_check`` returns ``False`` on a non-2xx status from /api/tags."""
    det = _make_detector("", tags_status=500)
    assert await det.health_check() is False


# ---------------------------------------------------------------------------
# Prompt construction tests
# ---------------------------------------------------------------------------


def test_build_prompt_contains_all_examples() -> None:
    """The built prompt must include every entry from ``FEW_SHOT_EXAMPLES``."""
    det = LLMDetector(base_url="http://localhost:11434")
    prompt = det._build_prompt({"test_feature": 42.0})

    for example in FEW_SHOT_EXAMPLES:
        assert example["label"] in prompt
        for key in example["input"]:
            assert key in prompt


def test_build_prompt_contains_candidate() -> None:
    """The built prompt must embed the candidate record's feature values."""
    det = LLMDetector(base_url="http://localhost:11434")
    candidate = {"special_metric": 99.9, "another": 0.1}
    prompt = det._build_prompt(candidate)

    assert "special_metric" in prompt
    assert "99.9" in prompt


# ---------------------------------------------------------------------------
# Response parsing tests
# ---------------------------------------------------------------------------


def test_parse_response_direct_json() -> None:
    """``_parse_response`` must parse a clean JSON string directly."""
    det = LLMDetector(base_url="http://localhost:11434")
    raw = '{"is_anomaly": true, "score": 0.8, "explanation": "test"}'
    result = det._parse_response(raw)

    assert result["is_anomaly"] is True
    assert result["score"] == pytest.approx(0.8)


def test_parse_response_json_embedded_in_prose() -> None:
    """``_parse_response`` must extract JSON even when surrounded by prose."""
    det = LLMDetector(base_url="http://localhost:11434")
    raw = (
        'Sure! Here is my assessment:\n'
        '{"is_anomaly": false, "score": 0.05, "explanation": "looks normal"}\n'
        'Hope that helps!'
    )
    result = det._parse_response(raw)

    assert result["is_anomaly"] is False
    assert result["score"] == pytest.approx(0.05)


def test_parse_response_raises_on_garbage() -> None:
    """``_parse_response`` must raise ``ValueError`` when no JSON can be found."""
    det = LLMDetector(base_url="http://localhost:11434")
    with pytest.raises(ValueError, match="Cannot extract"):
        det._parse_response("this is just plain text with no JSON at all")
