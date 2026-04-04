"""
Tests for the Great Expectations rule-based detection strategy.

All GE internals are mocked so the tests run without a DataContext, a real
expectation suite file, or a live GE installation beyond the import check.

The ``_load_suite`` method is patched on every ``detector`` fixture so that
construction succeeds without touching the filesystem.  Individual test cases
patch ``_to_dataframe`` to inject a controlled
``ExpectationSuiteValidationResult`` without running real GE validation.

Covers:
    - Valid records that satisfy all expectations → no anomaly, score 0.0
    - Records missing a required field → anomaly, failed name in raw
    - Records with an out-of-range value → anomaly, correct score
    - Suite file not found → FileNotFoundError on construction
    - Suite file contains invalid JSON → ValueError on construction
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from backend.detection.great_expectations_strategy import GreatExpectationsDetector
from backend.models.schemas import StrategyName


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_exp_result(expectation_type: str, *, success: bool) -> MagicMock:
    """Build a mock ``ExpectationValidationResult`` for one expectation."""
    r = MagicMock()
    r.success = success
    r.expectation_config.expectation_type = expectation_type
    return r


def _make_validation_result(
    passed: list[str],
    failed: list[str],
) -> MagicMock:
    """Build a mock ``ExpectationSuiteValidationResult``.

    Args:
        passed: ``expectation_type`` strings for expectations that passed.
        failed: ``expectation_type`` strings for expectations that failed.

    Returns:
        A :class:`~unittest.mock.MagicMock` whose attributes mirror the
        real GE ``ExpectationSuiteValidationResult`` API consumed by
        :meth:`GreatExpectationsDetector._map_result`.
    """
    total = len(passed) + len(failed)
    vr = MagicMock()
    vr.success = len(failed) == 0
    vr.results = [_make_exp_result(e, success=True) for e in passed] + [
        _make_exp_result(e, success=False) for e in failed
    ]
    vr.statistics = {
        "evaluated_expectations": total,
        "successful_expectations": len(passed),
        "unsuccessful_expectations": len(failed),
        "success_percent": (len(passed) / total * 100) if total else None,
    }
    return vr


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def detector() -> GreatExpectationsDetector:
    """Provide a ``GreatExpectationsDetector`` with ``_load_suite`` mocked.

    The suite itself is a ``MagicMock``; no JSON file is accessed.
    """
    with patch.object(
        GreatExpectationsDetector,
        "_load_suite",
        return_value=MagicMock(expectation_suite_name="test_suite"),
    ):
        return GreatExpectationsDetector("/fake/path/suite.json")


# ---------------------------------------------------------------------------
# Tests — normal behaviour
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_valid_record_not_flagged(detector: GreatExpectationsDetector) -> None:
    """A record that satisfies all expectations must not be an anomaly.

    Asserts:
        - ``is_anomaly`` is ``False``
        - ``score`` is ``0.0``
        - ``raw["failed_expectations"]`` is empty
        - ``strategy`` is ``StrategyName.great_expectations``
    """
    vr = _make_validation_result(
        passed=["expect_column_to_exist", "expect_column_values_to_be_between"],
        failed=[],
    )
    with patch.object(detector, "_to_dataframe") as mock_to_df:
        mock_to_df.return_value.validate.return_value = vr
        result = await detector.detect({"age": 30, "name": "Alice"})

    assert result.strategy == StrategyName.great_expectations
    assert result.is_anomaly is False
    assert result.score == 0.0
    assert result.raw["failed_expectations"] == []
    assert result.raw["total_expectations"] == 2
    assert "passed" in result.explanation.lower()


@pytest.mark.asyncio
async def test_missing_field_flagged(detector: GreatExpectationsDetector) -> None:
    """A record missing a required field must be flagged as an anomaly.

    Asserts:
        - ``is_anomaly`` is ``True``
        - The missing-column expectation appears in ``raw["failed_expectations"]``
        - ``score`` equals the fraction of failed expectations
    """
    failed_name = "expect_column_to_exist"
    vr = _make_validation_result(
        passed=["expect_column_values_to_be_between"],
        failed=[failed_name],
    )
    with patch.object(detector, "_to_dataframe") as mock_to_df:
        mock_to_df.return_value.validate.return_value = vr
        result = await detector.detect({})  # record missing the required column

    assert result.is_anomaly is True
    assert failed_name in result.raw["failed_expectations"]
    assert result.raw["total_expectations"] == 2
    # 1 of 2 expectations failed → score 0.5
    assert result.score == pytest.approx(0.5, abs=1e-4)
    assert failed_name in result.explanation


@pytest.mark.asyncio
async def test_out_of_range_value_flagged(detector: GreatExpectationsDetector) -> None:
    """A record with a value outside the expected range must be an anomaly.

    Asserts:
        - ``is_anomaly`` is ``True``
        - The range expectation appears in ``raw["failed_expectations"]``
        - When all expectations fail, ``score`` is ``1.0``
    """
    failed_name = "expect_column_values_to_be_between"
    vr = _make_validation_result(
        passed=[],
        failed=[failed_name],
    )
    with patch.object(detector, "_to_dataframe") as mock_to_df:
        mock_to_df.return_value.validate.return_value = vr
        result = await detector.detect({"age": 999})

    assert result.is_anomaly is True
    assert failed_name in result.raw["failed_expectations"]
    assert result.score == pytest.approx(1.0, abs=1e-4)


# ---------------------------------------------------------------------------
# Tests — error paths
# ---------------------------------------------------------------------------


def test_suite_file_not_found_raises() -> None:
    """Constructor must raise ``FileNotFoundError`` for a missing suite file."""
    with pytest.raises(FileNotFoundError, match="not found"):
        GreatExpectationsDetector("/nonexistent/path/suite.json")


def test_suite_file_invalid_json_raises(tmp_path) -> None:
    """Constructor must raise ``ValueError`` when the file is not valid JSON."""
    bad_file = tmp_path / "suite.json"
    bad_file.write_text("{ this is not json }", encoding="utf-8")

    with pytest.raises(ValueError, match="not valid JSON"):
        GreatExpectationsDetector(str(bad_file))


# ---------------------------------------------------------------------------
# Tests — edge cases
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_empty_suite_no_anomaly(detector: GreatExpectationsDetector) -> None:
    """A suite with zero expectations must not flag an anomaly (score 0.0)."""
    vr = _make_validation_result(passed=[], failed=[])
    with patch.object(detector, "_to_dataframe") as mock_to_df:
        mock_to_df.return_value.validate.return_value = vr
        result = await detector.detect({"x": 1})

    assert result.is_anomaly is False
    assert result.score == 0.0


@pytest.mark.asyncio
async def test_health_check_returns_true(detector: GreatExpectationsDetector) -> None:
    """``health_check`` returns ``True`` when the suite loaded successfully."""
    assert await detector.health_check() is True
