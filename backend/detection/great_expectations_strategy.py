"""
Rule-based anomaly detection using Great Expectations expectation suites.

Loads a pre-defined expectation suite from a JSON file whose path is
configured via the ``GE_EXPECTATION_SUITE_PATH`` environment variable,
converts each incoming data record into a single-row pandas DataFrame, and
validates it against the suite.  Any failed expectations are reported as an
anomaly.

The anomaly *score* follows the convention used by all Sadeed detectors:
    0.0 → all expectations passed (no anomaly)
    1.0 → all expectations failed (maximum anomaly)

Dependencies:
    great-expectations>=0.18, pandas
"""

import json
import logging
from pathlib import Path
from typing import Any, Callable

import great_expectations as ge
import pandas as pd
from great_expectations.core import ExpectationSuite
from great_expectations.core.expectation_validation_result import (
    ExpectationSuiteValidationResult,
)
from great_expectations.dataset import PandasDataset

from backend.detection.base import BaseDetector
from backend.models.schemas import StrategyName, StrategyResult

# Rules from LFS_Business_Rules.xlsx: 2xxx series are hard errors,
# 3xxx/4030 are warnings. Used to determine severity of GE findings.
_HARD_ERROR_RULES: set[str] = {
    "2001", "2011", "2012", "2013", "2015", "2016", "2031", "2053",
    "2075", "2078", "2088", "2090", "2141", "4069",
    "required_field", "range_check", "valid_values",
}
_WARNING_RULES: set[str] = {
    "3027", "3027+3028", "3028", "3030", "2088+3030", "3031", "3032",
    "3035", "3036", "3039", "3039+3068", "3042", "3043", "3044",
    "3047", "3068", "3071", "4030",
}

logger = logging.getLogger(__name__)


class GreatExpectationsDetector(BaseDetector):
    """Validates data records against a Great Expectations expectation suite.

    Attributes:
        _suite: The loaded :class:`~great_expectations.core.ExpectationSuite`
            used for every ``detect`` call.
        _preprocessor: Optional callable that transforms a raw data dict
            before validation (e.g. to add derived columns for conditional rules).
    """

    def __init__(
        self,
        expectation_suite_path: str,
        *,
        preprocessor: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
    ) -> None:
        """Load the expectation suite from *expectation_suite_path*.

        Args:
            expectation_suite_path: Filesystem path to a JSON expectation
                suite exported from Great Expectations
                (``suite.to_json_dict()``).
            preprocessor: Optional callable that receives the raw data dict
                and returns a new dict with any derived columns added. Used
                to compute conditional rule columns (e.g. ``_rule_*`` fields).

        Raises:
            FileNotFoundError: If *expectation_suite_path* does not exist.
            ValueError: If the file cannot be parsed as a valid expectation
                suite JSON.
        """
        self._suite: ExpectationSuite = self._load_suite(expectation_suite_path)
        self._preprocessor = preprocessor
        logger.info(
            "Loaded expectation suite '%s' from '%s'",
            self._suite.expectation_suite_name,
            expectation_suite_path,
        )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    async def detect(self, data: dict[str, Any]) -> StrategyResult:
        """Validate *data* against the loaded expectation suite.

        Converts *data* to a single-row pandas DataFrame, runs GE
        validation synchronously (GE has no async API), then maps the
        :class:`~great_expectations.core.ExpectationSuiteValidationResult`
        to a :class:`~backend.models.schemas.StrategyResult`.

        Args:
            data: Flat dictionary of feature names to values.  Nested
                values are accepted by pandas but expectation results may
                be less meaningful.

        Returns:
            A :class:`~backend.models.schemas.StrategyResult` where:

            * ``is_anomaly`` is ``True`` if *any* expectation failed.
            * ``score`` is the fraction of expectations that *failed*
              (range ``[0.0, 1.0]``).
            * ``explanation`` is a human-readable summary.
            * ``raw["failed_expectations"]`` lists the
              ``expectation_type`` strings of every failing rule.
            * ``raw["total_expectations"]`` is the total count evaluated.
        """
        if self._preprocessor is not None:
            data = self._preprocessor(data)
        dataset: PandasDataset = self._to_dataframe(data)
        validation_result: ExpectationSuiteValidationResult = dataset.validate(
            expectation_suite=self._suite,
            result_format="SUMMARY",
        )
        return self._map_result(validation_result, data)

    async def health_check(self) -> bool:
        """Return ``True`` if the expectation suite was loaded successfully.

        Returns:
            ``True`` when the suite is non-``None`` and contains at least
            the suite name; ``False`` otherwise.
        """
        return self._suite is not None

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_suite(self, path: str) -> ExpectationSuite:
        """Read and deserialise the expectation suite JSON file.

        Args:
            path: Filesystem path to the expectation suite JSON file.

        Returns:
            A :class:`~great_expectations.core.ExpectationSuite` instance
            ready for validation.

        Raises:
            FileNotFoundError: If *path* does not exist.
            ValueError: If the file content is not valid JSON or cannot be
                interpreted as an :class:`~great_expectations.core.ExpectationSuite`.
        """
        suite_path = Path(path)
        if not suite_path.exists():
            raise FileNotFoundError(
                f"Expectation suite file not found: '{path}'"
            )

        try:
            suite_dict: dict[str, Any] = json.loads(suite_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Expectation suite file is not valid JSON: '{path}'"
            ) from exc

        try:
            return ExpectationSuite(**suite_dict)
        except Exception as exc:
            raise ValueError(
                f"Could not construct ExpectationSuite from '{path}': {exc}"
            ) from exc

    def _to_dataframe(self, data: dict[str, Any]) -> PandasDataset:
        """Wrap *data* in a single-row :class:`~great_expectations.dataset.PandasDataset`.

        Args:
            data: Feature dictionary for one data record.

        Returns:
            A GE-instrumented pandas DataFrame with one row.
        """
        return ge.from_pandas(pd.DataFrame([data]))

    def _map_result(
        self,
        validation_result: ExpectationSuiteValidationResult,
        data: dict[str, Any] | None = None,
    ) -> StrategyResult:
        """Convert a GE validation result into a :class:`~backend.models.schemas.StrategyResult`.

        Args:
            validation_result: The raw result returned by
                :meth:`~great_expectations.dataset.PandasDataset.validate`.

        Returns:
            A :class:`~backend.models.schemas.StrategyResult` populated from
            the validation result.
        """
        stats: dict[str, Any] = validation_result.statistics or {}
        total: int = int(stats.get("evaluated_expectations", 0))
        n_failed: int = int(stats.get("unsuccessful_expectations", 0))

        failed_types: list[str] = [
            r.expectation_config.expectation_type
            for r in validation_result.results
            if not r.success
        ]

        # Human-readable names for derived rule columns
        _RULE_LABELS: dict[str, str] = {
            "_rule_2001_min_head_age": "Head of household must be >= 15",
            "_rule_2011_min_secondary_age": "Secondary education requires age >= 17",
            "_rule_2012_min_diploma_age": "Diploma requires age >= 19",
            "_rule_2013_min_bachelor_age": "Bachelor's requires age >= 21",
            "_rule_2015_min_masters_age": "Master's requires age >= 23",
            "_rule_2016_min_phd_age": "PhD requires age >= 25",
            "_rule_2031_spouse_married": "Spouse must be married",
            "_rule_2053_grandparent_married": "Grandparent cannot be never married",
            "_rule_2075_grandparent_age": "Grandparent must be >= 30",
            "_rule_2078_public_sector_age": "Public sector worker must be >= 17",
            "_rule_2090_never_married_spouse": "Never married cannot be spouse",
            "_rule_2141_domestic_worker_age": "Domestic worker must be >= 15",
            "_rule_3027_low_usual_hours": "Usual working hours under 20/week",
            "_rule_3028_high_usual_hours": "Usual working hours over 48/week",
            "_rule_3030_high_actual_hours": "Actual working hours over 48/week",
            "_rule_3031_public_usual_hours": "Public sector: usual hours under 35/week",
            "_rule_3032_public_actual_hours": "Public sector: actual hours under 35/week",
            "_rule_3035_private_usual_hours": "Private sector: usual hours under 40/week",
            "_rule_3036_private_actual_hours": "Private sector: actual hours under 40/week",
            "_rule_3042_age_salary": "Age over 30 with salary under 1000 SAR",
            "_rule_3043_head_salary": "Head of household with salary under 2000 SAR",
            "_rule_3044_saudi_salary": "Saudi national with salary under 3000 SAR",
            "_rule_3047_uni_salary": "University degree+ with salary under 3000 SAR",
            "_rule_3071_young_high_salary": "Under 25 with salary over 10000 SAR",
            "_rule_4030_spouse_age": "Spouse must be >= 18",
            "_rule_4069_child_not_married": "Under 15 must be never married",
        }

        # Collect detailed failure info for explanations
        failed_details: list[dict[str, Any]] = []
        seen_rules: set[str] = set()
        for r in validation_result.results:
            if not r.success:
                kwargs = r.expectation_config.kwargs or {}
                col = kwargs.get("column", kwargs.get("column_A", ""))
                col_b = kwargs.get("column_B", "")
                # Deduplicate pair rules by the derived column name
                dedup_key = f"{col}|{col_b}" if col_b else col
                if dedup_key in seen_rules:
                    continue
                seen_rules.add(dedup_key)
                detail: dict[str, Any] = {
                    "rule": r.expectation_config.expectation_type,
                    "column": col,
                }
                if col_b:
                    detail["column_b"] = col_b
                    # Check both columns for a rule label
                    rule_col = col if col.startswith("_rule_") else col_b if col_b.startswith("_rule_") else ""
                    detail["label"] = _RULE_LABELS.get(rule_col, "")
                if "min_value" in kwargs:
                    detail["min_value"] = kwargs["min_value"]
                if "max_value" in kwargs:
                    detail["max_value"] = kwargs["max_value"]
                if "value_set" in kwargs:
                    detail["allowed_values"] = kwargs["value_set"]
                result_detail = r.result or {}
                if "unexpected_list" in result_detail:
                    detail["actual_values"] = result_detail["unexpected_list"]
                if "observed_value" in result_detail:
                    detail["observed_value"] = result_detail["observed_value"]
                failed_details.append(detail)

        # Determine severity from failed rule IDs (hard_error > warning)
        failed_rule_ids: set[str] = set()
        for r in validation_result.results:
            if not r.success:
                meta = r.expectation_config.meta or {}
                rule_id = meta.get("rule", "")
                if rule_id:
                    failed_rule_ids.add(rule_id)

        if failed_rule_ids & _HARD_ERROR_RULES:
            severity = "hard_error"
        elif failed_rule_ids & _WARNING_RULES:
            severity = "warning"
        elif failed_rule_ids:
            severity = "warning"
        else:
            severity = None

        # Anomaly score: fraction of expectations that failed.
        score: float = round(n_failed / total, 4) if total > 0 else 0.0

        if failed_details:
            parts: list[str] = []
            for d in failed_details:
                col = d.get("column", "unknown")
                label = d.get("label", "")
                if label:
                    parts.append(label)
                elif "min_value" in d and "max_value" in d:
                    actual = d.get("actual_values", d.get("observed_value"))
                    if actual is None and data is not None:
                        actual = data.get(col, "?")
                    parts.append(
                        f"{col} value {actual} is outside allowed range "
                        f"({d['min_value']}-{d['max_value']})"
                    )
                elif "allowed_values" in d:
                    actual = d.get("actual_values", d.get("observed_value"))
                    if actual is None and data is not None:
                        actual = data.get(col, "?")
                    parts.append(f"{col} value {actual} is not in the allowed set")
                else:
                    parts.append(f"{col}: rule {d['rule']} failed")
            explanation = "; ".join(parts) + "."
        else:
            explanation = f"All {total} expectation(s) passed."

        return StrategyResult(
            strategy=StrategyName.great_expectations,
            is_anomaly=not validation_result.success,
            score=score,
            explanation=explanation,
            raw={
                "failed_expectations": failed_types,
                "failed_details": failed_details,
                "total_expectations": total,
                "severity": severity,
            },
        )
