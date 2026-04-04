"""
Tests for the One-Class SVM statistical detection strategy.

sklearn calls are mocked throughout — no live model training occurs.
The ``trained_detector`` fixture bypasses ``__init__`` training logic and
injects a :class:`unittest.mock.MagicMock` in place of the fitted
:class:`~sklearn.svm.OneClassSVM`.

The sole exception is ``test_train_save_load_roundtrip``, which uses a real
but minimal sklearn model (10 samples, 2 features) because the point of that
test is to verify joblib serialisation fidelity — mocking the model object
would make the test vacuous.

Covers:
    - Inlier record (predict=+1) → is_anomaly=False, score < 0.5
    - Outlier record (predict=-1) → is_anomaly=True, score > 0.5
    - Anomaly score is always in [0, 1]
    - Correct strategy name on every result
    - RuntimeError when no model is loaded
    - ValueError when feature count mismatches training dimensionality
    - FileNotFoundError when training data path is absent
    - save / load roundtrip produces identical predictions
"""

from __future__ import annotations

import math
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from sklearn.svm import OneClassSVM

from backend.detection.svm_strategy import SVMDetector
from backend.models.schemas import StrategyName


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def trained_detector() -> SVMDetector:
    """Provide an ``SVMDetector`` with a mocked :class:`~sklearn.svm.OneClassSVM`.

    ``__init__`` training / file-loading is bypassed entirely.  The injected
    mock model's ``predict`` and ``decision_function`` are configured per
    test via ``mock_model.predict.return_value = ...``.
    """
    with (
        patch.object(SVMDetector, "_load_training_data", return_value=np.zeros((10, 3))),
        patch.object(SVMDetector, "train"),
    ):
        detector = SVMDetector(
            training_data_path="fake/path/data.npy",
            kernel="rbf",
            nu=0.1,
            gamma="scale",
        )

    mock_model = MagicMock(spec=OneClassSVM)
    detector._model = mock_model
    detector._n_features = 3
    return detector


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sigmoid_neg(df: float) -> float:
    """Return ``sigmoid(−df) = 1 / (1 + exp(df))`` — the expected score formula."""
    return 1.0 / (1.0 + math.exp(df))


# ---------------------------------------------------------------------------
# Inference tests (all use mocked model)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_inlier_not_flagged(trained_detector: SVMDetector) -> None:
    """A record inside the learned boundary must not be an anomaly.

    Asserts:
        - ``is_anomaly`` is ``False``
        - ``score`` matches ``sigmoid(−df)`` and is below ``0.5``
        - ``strategy`` is ``StrategyName.svm``
        - ``raw["svm_prediction"]`` is ``1``
    """
    df_value = 0.8
    trained_detector._model.predict.return_value = np.array([1])
    trained_detector._model.decision_function.return_value = np.array([df_value])

    result = await trained_detector.detect({"a": 1.0, "b": 2.0, "c": 3.0})

    assert result.strategy == StrategyName.svm
    assert result.is_anomaly is False
    assert result.score == pytest.approx(_sigmoid_neg(df_value), abs=1e-4)
    assert result.score < 0.5
    assert result.raw["svm_prediction"] == 1
    assert result.raw["decision_function_score"] == pytest.approx(df_value)


@pytest.mark.asyncio
async def test_outlier_flagged(trained_detector: SVMDetector) -> None:
    """A record outside the learned boundary must be flagged as an anomaly.

    Asserts:
        - ``is_anomaly`` is ``True``
        - ``score`` matches ``sigmoid(−df)`` and is above ``0.5``
        - ``raw["svm_prediction"]`` is ``-1``
    """
    df_value = -1.5
    trained_detector._model.predict.return_value = np.array([-1])
    trained_detector._model.decision_function.return_value = np.array([df_value])

    result = await trained_detector.detect({"a": 99.0, "b": -50.0, "c": 1000.0})

    assert result.strategy == StrategyName.svm
    assert result.is_anomaly is True
    assert result.score == pytest.approx(_sigmoid_neg(df_value), abs=1e-4)
    assert result.score > 0.5
    assert result.raw["svm_prediction"] == -1
    assert result.raw["decision_function_score"] == pytest.approx(df_value)


@pytest.mark.asyncio
async def test_score_always_in_unit_interval(trained_detector: SVMDetector) -> None:
    """Anomaly score must be in ``[0, 1]`` and match ``sigmoid(−df)`` after rounding.

    Note: for extreme ``|df|`` values, rounding to 4 decimal places can produce
    exactly ``0.0`` or ``1.0`` — both are valid boundary outputs.
    """
    cases = [(-5.0, -1), (-0.01, -1), (0.0, 1), (0.5, 1), (10.0, 1)]
    for df_value, prediction in cases:
        trained_detector._model.predict.return_value = np.array([prediction])
        trained_detector._model.decision_function.return_value = np.array([df_value])

        result = await trained_detector.detect({"a": 1.0, "b": 1.0, "c": 1.0})

        assert 0.0 <= result.score <= 1.0, (
            f"score={result.score} out of [0,1] for df={df_value}"
        )
        assert result.score == pytest.approx(_sigmoid_neg(df_value), abs=5e-4), (
            f"score mismatch for df={df_value}"
        )


@pytest.mark.asyncio
async def test_boundary_score_is_half(trained_detector: SVMDetector) -> None:
    """A record exactly on the decision boundary (df=0) must score 0.5."""
    trained_detector._model.predict.return_value = np.array([1])
    trained_detector._model.decision_function.return_value = np.array([0.0])

    result = await trained_detector.detect({"a": 0.0, "b": 0.0, "c": 0.0})

    assert result.score == pytest.approx(0.5, abs=1e-4)


# ---------------------------------------------------------------------------
# Error-path tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_detect_without_model_raises() -> None:
    """``detect`` must raise ``RuntimeError`` when no model is available."""
    with (
        patch.object(SVMDetector, "_load_training_data", return_value=np.zeros((10, 3))),
        patch.object(SVMDetector, "train"),
    ):
        detector = SVMDetector("fake.npy")

    # model was not injected → should raise
    with pytest.raises(RuntimeError, match="no fitted model"):
        await detector.detect({"x": 1.0, "y": 2.0, "z": 3.0})


@pytest.mark.asyncio
async def test_feature_count_mismatch_raises(trained_detector: SVMDetector) -> None:
    """``detect`` must raise ``ValueError`` when feature count differs from training."""
    # trained_detector._n_features == 3; pass 2 features
    with pytest.raises(ValueError, match="Feature count mismatch"):
        await trained_detector.detect({"a": 1.0, "b": 2.0})


def test_training_data_not_found_raises() -> None:
    """``__init__`` must raise ``FileNotFoundError`` for a missing ``.npy`` file."""
    with pytest.raises(FileNotFoundError, match="Training data file not found"):
        SVMDetector(training_data_path="/nonexistent/data.npy")


def test_train_raises_on_1d_array() -> None:
    """``train`` must raise ``ValueError`` when passed a 1-D array."""
    with (
        patch.object(SVMDetector, "_load_training_data", return_value=np.zeros((10, 2))),
        patch.object(SVMDetector, "train"),
    ):
        detector = SVMDetector("fake.npy")

    with pytest.raises(ValueError, match="2-D array"):
        detector.train(np.array([1.0, 2.0, 3.0]))


def test_save_without_model_raises() -> None:
    """``save`` must raise ``RuntimeError`` when called before training."""
    with (
        patch.object(SVMDetector, "_load_training_data", return_value=np.zeros((10, 2))),
        patch.object(SVMDetector, "train"),
    ):
        detector = SVMDetector("fake.npy")

    with pytest.raises(RuntimeError, match="no fitted model"):
        detector.save("/tmp/model.joblib")


# ---------------------------------------------------------------------------
# Persistence test (real sklearn — minimal, deterministic)
# ---------------------------------------------------------------------------


def test_train_save_load_roundtrip(tmp_path: Path) -> None:
    """A saved model must produce identical predictions after a joblib round-trip.

    Uses a real :class:`~sklearn.svm.OneClassSVM` on a tiny synthetic dataset
    (20 samples, 2 features) because the point of this test is to verify
    joblib serialisation fidelity — mocking the model object would be vacuous.
    """
    rng = np.random.default_rng(0)
    X_train = rng.standard_normal((20, 2))

    # Train and save
    model_path = str(tmp_path / "svm.joblib")
    with (
        patch.object(SVMDetector, "_load_training_data", return_value=X_train),
    ):
        d1 = SVMDetector(
            training_data_path="fake.npy",
            model_path=model_path,
            kernel="rbf",
            nu=0.1,
            gamma="scale",
        )

    # Load into a fresh detector
    with (
        patch.object(SVMDetector, "_load_training_data", return_value=X_train),
    ):
        d2 = SVMDetector(
            training_data_path="fake.npy",
            model_path=model_path,
            kernel="rbf",
            nu=0.1,
            gamma="scale",
        )

    assert d2._model is not None
    assert d2._n_features == 2

    # Both detectors must agree on every test sample
    test_samples = rng.standard_normal((5, 2))
    orig_preds = d1._model.predict(test_samples)
    loaded_preds = d2._model.predict(test_samples)
    assert np.array_equal(orig_preds, loaded_preds), (
        f"Predictions diverged after round-trip: {orig_preds} vs {loaded_preds}"
    )


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_health_check_with_model(trained_detector: SVMDetector) -> None:
    """``health_check`` returns ``True`` when a model is loaded."""
    assert await trained_detector.health_check() is True


@pytest.mark.asyncio
async def test_health_check_without_model() -> None:
    """``health_check`` returns ``False`` when no model has been fitted."""
    with (
        patch.object(SVMDetector, "_load_training_data", return_value=np.zeros((10, 2))),
        patch.object(SVMDetector, "train"),
    ):
        detector = SVMDetector("fake.npy")

    assert await detector.health_check() is False
