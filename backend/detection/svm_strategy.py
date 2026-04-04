"""
Statistical anomaly detection using a One-Class SVM (scikit-learn).

Training data is loaded from a ``.npy`` file whose path is configured via
``OC_SVM_TRAINING_DATA_PATH``.  The array must be 2-D with shape
``(n_samples, n_features)``.  Feature ordering at inference time must match
column ordering in the training array; ``_preprocess`` enforces this by
extracting dict values in **sorted key order** — training data must be
prepared with the same convention.

Startup flow
------------
1. If a cached joblib model exists at ``OC_SVM_MODEL_PATH``, load it
   (fast path — no retraining).
2. Otherwise train on ``OC_SVM_TRAINING_DATA_PATH`` and, if
   ``OC_SVM_MODEL_PATH`` is provided, persist the fitted model for the
   next startup.

Anomaly score convention (shared across all Sadeed detectors)
--------------------------------------------------------------
``score = sigmoid(−df) = 1 / (1 + exp(df))``

where *df* is the signed distance returned by
:meth:`~sklearn.svm.OneClassSVM.decision_function`.

* ``df > 0`` → inside the learned boundary → score < 0.5 (normal)
* ``df < 0`` → outside the boundary → score > 0.5 (anomaly)
* Boundary (``df = 0``) → score = 0.5

Dependencies:
    scikit-learn, numpy, joblib (bundled with scikit-learn)
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

from backend.detection.base import BaseDetector
from backend.models.schemas import StrategyName, StrategyResult

logger = logging.getLogger(__name__)

# Ordinal encoding for education level (q_301 code → 0-11 scale).
# Must match the mapping used in scripts/prepare_data.py during training.
EDU_ORDINAL_MAP: dict[int, int] = {
    10500031: 0,   # No formal education
    10500011: 1,   # Primary incomplete
    10500003: 2,   # Primary education
    10500017: 3,   # Intermediate
    10500019: 4,   # Secondary (general)
    10500020: 5,   # Secondary (vocational)
    10500021: 6,   # Associate diploma
    10500023: 7,   # Diploma
    10500025: 8,   # Bachelor's (3-4 yr)
    10500026: 9,   # Bachelor's (5 yr)
    10500009: 10,  # Master's
    10500010: 11,  # PhD
}


class SVMDetector(BaseDetector):
    """One-Class SVM wrapper for unsupervised anomaly detection.

    Attributes:
        _model: A fitted :class:`~sklearn.pipeline.Pipeline` containing a
            :class:`~sklearn.preprocessing.StandardScaler` followed by a
            :class:`~sklearn.svm.OneClassSVM`, or ``None`` before training.
        _n_features: Number of features the model was trained on.  Used to
            validate incoming records at inference time.
        _kernel: SVM kernel identifier (e.g. ``"rbf"``).
        _nu: Upper bound on the fraction of training outliers and lower bound
            on the fraction of support vectors.
        _gamma: Kernel coefficient — ``"scale"``, ``"auto"``, or a float.
    """

    def __init__(
        self,
        training_data_path: str,
        model_path: str | None = None,
        *,
        kernel: str = "rbf",
        nu: float = 0.1,
        gamma: str | float = "scale",
        feature_columns: list[str] | None = None,
    ) -> None:
        """Initialise the detector and ensure a fitted model is ready.

        Attempts to load a cached model from *model_path* first.  Falls back
        to training from *training_data_path* when the cache is absent.  If
        *model_path* is provided and the cache was absent, the freshly trained
        model is written there for the next startup.

        Args:
            training_data_path: Path to a ``.npy`` file containing the
                normal-class training matrix, shape ``(n_samples, n_features)``.
                Used only when no cached model is found.
            model_path: Optional path to a joblib-serialised
                :class:`~sklearn.svm.OneClassSVM`.  If the file exists it is
                loaded directly; if it does not exist the trained model is
                saved here after fitting.
            kernel: SVM kernel type passed to
                :class:`~sklearn.svm.OneClassSVM`.  Default ``"rbf"``.
            nu: Regularisation parameter in ``(0, 1]``.  Default ``0.1``.
            gamma: Kernel coefficient.  Accepts ``"scale"``, ``"auto"``, or a
                positive float.  Default ``"scale"``.
            feature_columns: Optional list of column names (in sorted order)
                that the model was trained on.  When provided,
                :meth:`_preprocess` extracts only these columns from the
                incoming data dict, ignoring all other keys.  When ``None``,
                all keys are used (legacy behaviour).

        Raises:
            FileNotFoundError: If *training_data_path* does not exist and no
                valid *model_path* cache is available.
            ValueError: If the training array is not 2-D.
        """
        self._model: Pipeline | None = None
        self._n_features: int | None = None
        self._kernel = kernel
        self._nu = nu
        self._gamma = gamma
        self._feature_columns: list[str] | None = (
            sorted(feature_columns) if feature_columns else None
        )

        if model_path and Path(model_path).exists():
            logger.info("Loading cached SVM model from '%s'.", model_path)
            self._load(model_path)
        else:
            logger.info(
                "No cached model found. Training from '%s'.", training_data_path
            )
            X = self._load_training_data(training_data_path)
            self.train(X)
            if model_path:
                logger.info("Saving trained model to '%s'.", model_path)
                self.save(model_path)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    async def detect(self, data: dict[str, Any]) -> StrategyResult:
        """Score *data* using the fitted One-Class SVM.

        Converts *data* to a feature vector, calls
        :meth:`~sklearn.svm.OneClassSVM.predict` and
        :meth:`~sklearn.svm.OneClassSVM.decision_function`, then maps the
        results to a :class:`~backend.models.schemas.StrategyResult`.

        Args:
            data: Flat dictionary mapping feature names to numeric values.
                Keys must be sorted identically to the training array columns
                (``_preprocess`` enforces alphabetical order on both sides).

        Returns:
            A :class:`~backend.models.schemas.StrategyResult` where:

            * ``is_anomaly`` is ``True`` when the SVM prediction is ``-1``.
            * ``score`` is ``sigmoid(−df)`` ∈ ``(0, 1)``.  Values above
              ``0.5`` indicate anomaly territory.
            * ``explanation`` describes the verdict and raw distance.
            * ``raw["decision_function_score"]`` is the unscaled *df* value.
            * ``raw["svm_prediction"]`` is the raw ``+1`` / ``-1`` integer.

        Raises:
            RuntimeError: If the model has not been trained or loaded yet.
            ValueError: If the number of features in *data* does not match
                the training dimensionality.
        """
        if self._model is None:
            raise RuntimeError(
                "SVMDetector has no fitted model. "
                "Provide a valid training_data_path or model_path."
            )

        feature_vector: np.ndarray = self._preprocess(data)
        prediction: int = int(self._model.predict(feature_vector)[0])
        df_score: float = float(self._model.decision_function(feature_vector)[0])

        return self._map_result(prediction, df_score)

    def train(self, X: np.ndarray) -> None:
        """Fit the One-Class SVM on *X*.

        The pipeline scales features to zero mean / unit variance before
        fitting the SVM.  This prevents high-range features (e.g. salary)
        from dominating the decision boundary.

        Args:
            X: 2-D array of shape ``(n_samples, n_features)`` containing
                normal-class observations.

        Raises:
            ValueError: If *X* is not a 2-D array.
        """
        if X.ndim != 2:
            raise ValueError(
                f"Training data must be a 2-D array; got shape {X.shape}."
            )
        self._model = Pipeline([
            ("scaler", StandardScaler()),
            ("svm", OneClassSVM(
                kernel=self._kernel,
                nu=self._nu,
                gamma=self._gamma,
            )),
        ])
        self._model.fit(X)
        self._n_features = X.shape[1]
        logger.info(
            "SVM trained on %d samples, %d features.", X.shape[0], X.shape[1]
        )

    def save(self, path: str) -> None:
        """Serialise the fitted model to *path* via joblib.

        Args:
            path: Destination file path.  Parent directories must exist.

        Raises:
            RuntimeError: If no model has been trained yet.
        """
        if self._model is None:
            raise RuntimeError("Cannot save: no fitted model available.")
        joblib.dump(
            {
                "model": self._model,
                "n_features": self._n_features,
                "feature_columns": self._feature_columns,
            },
            path,
        )
        logger.info("SVM model saved to '%s'.", path)

    async def health_check(self) -> bool:
        """Return ``True`` if a fitted model is loaded and ready for inference.

        Returns:
            ``True`` when ``self._model`` is not ``None``.
        """
        return self._model is not None

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load(self, path: str) -> None:
        """Deserialise a joblib model bundle from *path*.

        Args:
            path: Path to the joblib file written by :meth:`save`.

        Raises:
            FileNotFoundError: If *path* does not exist.
            ValueError: If the file does not contain a valid model bundle.
        """
        if not Path(path).exists():
            raise FileNotFoundError(f"Model file not found: '{path}'")
        try:
            bundle: dict = joblib.load(path)
            self._model = bundle["model"]
            self._n_features = bundle["n_features"]
            if "feature_columns" in bundle and self._feature_columns is None:
                self._feature_columns = bundle["feature_columns"]
        except (KeyError, Exception) as exc:
            raise ValueError(
                f"Failed to load SVM model from '{path}': {exc}"
            ) from exc
        logger.info(
            "SVM model loaded from '%s' (%d features).", path, self._n_features
        )

    def _load_training_data(self, path: str) -> np.ndarray:
        """Load and validate the training array from a ``.npy`` file.

        Args:
            path: Path to a NumPy ``.npy`` file containing a 2-D float array.

        Returns:
            The loaded array with shape ``(n_samples, n_features)``.

        Raises:
            FileNotFoundError: If *path* does not exist.
            ValueError: If the loaded array is not 2-D.
        """
        if not Path(path).exists():
            raise FileNotFoundError(
                f"Training data file not found: '{path}'"
            )
        X: np.ndarray = np.load(path)
        if X.ndim != 2:
            raise ValueError(
                f"Training data must be a 2-D array; got shape {X.shape}."
            )
        logger.info(
            "Loaded training data from '%s': shape %s.", path, X.shape
        )
        return X

    def _preprocess(self, data: dict[str, Any]) -> np.ndarray:
        """Convert *data* to a ``(1, n_features)`` float array.

        When ``feature_columns`` was provided at construction time, only those
        columns are extracted (in sorted order).  Otherwise all keys are used
        in alphabetically sorted order — callers must ensure the dict contains
        exactly the training features.

        Derived features (like ``edu_ordinal``) are computed on the fly from
        their source columns (``q_301``).

        Args:
            data: Flat dictionary of feature names to numeric values.

        Returns:
            A 2-D NumPy array of shape ``(1, n_features)``.

        Raises:
            ValueError: If the number of keys in *data* does not match the
                training dimensionality.
        """
        # Build a working copy with derived features
        working = dict(data)

        # Derive edu_ordinal from q_301 if needed
        if self._feature_columns and "edu_ordinal" in self._feature_columns:
            q301_raw = data.get("q_301")
            if q301_raw is not None:
                q301_int = int(float(str(q301_raw)))
                working["edu_ordinal"] = EDU_ORDINAL_MAP.get(q301_int, 4)
            else:
                working["edu_ordinal"] = 4  # default mid-range (secondary)

        keys = self._feature_columns if self._feature_columns else sorted(working.keys())
        values: list[float] = [float(working[k]) for k in keys]
        if self._n_features is not None and len(values) != self._n_features:
            raise ValueError(
                f"Feature count mismatch: model expects {self._n_features} "
                f"feature(s) but received {len(values)}."
            )
        return np.array(values, dtype=np.float64).reshape(1, -1)

    def _map_result(self, prediction: int, df_score: float) -> StrategyResult:
        """Build a :class:`~backend.models.schemas.StrategyResult` from raw SVM outputs.

        Args:
            prediction: Raw SVM label — ``+1`` for inlier, ``-1`` for outlier.
            df_score: Signed distance from the decision boundary as returned
                by :meth:`~sklearn.svm.OneClassSVM.decision_function`.

        Returns:
            A fully populated :class:`~backend.models.schemas.StrategyResult`.
        """
        is_anomaly = prediction == -1

        # sigmoid(−df): maps (−∞, +∞) → (0, 1) with anomaly → >0.5
        anomaly_score = round(1.0 / (1.0 + math.exp(df_score)), 4)

        if is_anomaly:
            explanation = (
                f"SVM classified record as anomalous (prediction=-1). "
                f"Decision boundary distance: {df_score:.4f}."
            )
        else:
            explanation = (
                f"SVM classified record as normal (prediction=+1). "
                f"Decision boundary distance: {df_score:.4f}."
            )

        return StrategyResult(
            strategy=StrategyName.svm,
            is_anomaly=is_anomaly,
            score=anomaly_score,
            explanation=explanation,
            raw={
                "svm_prediction": prediction,
                "decision_function_score": df_score,
            },
        )
