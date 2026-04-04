"""
End-to-end metrics evaluation across all detection layers.

Evaluates each layer independently and in combination on:
  1. Real LFS records (70 records, GE rules as ground truth)
  2. Synthetic anomalies (10 GE-passing edge cases, all labeled anomalous)

Run with:
    pytest tests/test_metrics.py -v -s              # GE + SVM only
    SADEED_LLM=1 pytest tests/test_metrics.py -v -s # include LLM (requires Ollama)
"""

from __future__ import annotations

import asyncio
import copy
import os
import sys
from pathlib import Path

import pandas as pd
import pytest
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.detection.great_expectations_strategy import GreatExpectationsDetector
from backend.detection.lfs_preprocessing import add_derived_columns
from backend.detection.llm_strategy import LLMDetector
from backend.detection.svm_strategy import SVMDetector
from backend.models.schemas import StrategyResult

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

SUITE_PATH = "expectations/suite.json"
TRAINING_DATA_PATH = "data/normal_samples.npy"
MODEL_PATH = "models/svm.joblib"
DATASET_PATH = "data/LFS_Training_Dataset.xlsx"
SVM_FEATURES = ["act_1_total", "age", "cut_5_total", "edu_ordinal", "q_602_val"]

REQUIRED_COLS = [
    "age", "gender", "family_relation", "marage_status", "nationality",
    "q_301", "q_602_val", "cut_5_total", "act_1_total",
]

# ---------------------------------------------------------------------------
# Synthetic anomalies — pass all GE rules but are semantically anomalous
# ---------------------------------------------------------------------------

SYNTHETIC_ANOMALIES: list[dict] = [
    # --- Salary–education mismatches (not seen in few-shot) ---
    {"age": 28, "gender": 1600002, "family_relation": 1700022,
     "marage_status": 10600001, "nationality": 1800001,
     "q_301": 10500009, "q_602_val": 800, "cut_5_total": 35, "act_1_total": 33,
     "_label": "28yo female MSc, 800 SAR, normal hrs"},
    {"age": 33, "gender": 1600001, "family_relation": 1700001,
     "marage_status": 10600002, "nationality": 1800001,
     "q_301": 10500003, "q_602_val": 28000, "cut_5_total": 45, "act_1_total": 44,
     "_label": "Primary edu earning 28k SAR"},
    # --- Hours contradictions (different shapes than few-shot) ---
    {"age": 42, "gender": 1600001, "family_relation": 1700001,
     "marage_status": 10600002, "nationality": 1800001,
     "q_301": 10500023, "q_602_val": 8000, "cut_5_total": 60, "act_1_total": 2,
     "_label": "60 usual hrs, 2 actual"},
    {"age": 37, "gender": 1600002, "family_relation": 1700021,
     "marage_status": 10600002, "nationality": 1800001,
     "q_301": 10500025, "q_602_val": 12000, "cut_5_total": 3, "act_1_total": 70,
     "_label": "3 usual hrs, 70 actual"},
    # --- Age–context implausibility ---
    {"age": 19, "gender": 1600001, "family_relation": 1700022,
     "marage_status": 10600001, "nationality": 1800001,
     "q_301": 10500031, "q_602_val": 700, "cut_5_total": 75, "act_1_total": 75,
     "_label": "19yo no edu, 75 hrs/wk, 700 SAR"},
    {"age": 68, "gender": 1600002, "family_relation": 1700001,
     "marage_status": 10600003, "nationality": 1800001,
     "q_301": 10500017, "q_602_val": 35000, "cut_5_total": 50, "act_1_total": 48,
     "_label": "68yo female, intermediate edu, 35k SAR"},
    # --- Extreme salary–hours ratio ---
    {"age": 26, "gender": 1600001, "family_relation": 1700022,
     "marage_status": 10600001, "nationality": 1800001,
     "q_301": 10500019, "q_602_val": 48000, "cut_5_total": 5, "act_1_total": 4,
     "_label": "Secondary edu, 48k SAR, 5 hrs/wk"},
    {"age": 50, "gender": 1600001, "family_relation": 1700001,
     "marage_status": 10600002, "nationality": 1800001,
     "q_301": 10500025, "q_602_val": 600, "cut_5_total": 84, "act_1_total": 80,
     "_label": "Bachelor's, 600 SAR, 84 hrs/wk"},
    # --- Multi-field compound anomalies ---
    {"age": 21, "gender": 1600001, "family_relation": 1700001,
     "marage_status": 10600002, "nationality": 1800001,
     "q_301": 10500025, "q_602_val": 45000, "cut_5_total": 2, "act_1_total": 2,
     "_label": "21yo head, married, bachelor, 45k, 2hrs"},
    {"age": 60, "gender": 1600001, "family_relation": 1700001,
     "marage_status": 10600004, "nationality": 1800001,
     "q_301": 10500031, "q_602_val": 1000, "cut_5_total": 10, "act_1_total": 78,
     "_label": "60yo widower, no edu, 10 usual, 78 actual"},
]


# ---------------------------------------------------------------------------
# Metrics helpers
# ---------------------------------------------------------------------------


def compute_metrics(y_true: list[bool], y_pred: list[bool]) -> dict:
    tp = sum(t and p for t, p in zip(y_true, y_pred))
    tn = sum(not t and not p for t, p in zip(y_true, y_pred))
    fp = sum(not t and p for t, p in zip(y_true, y_pred))
    fn = sum(t and not p for t, p in zip(y_true, y_pred))
    total = len(y_true)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / total if total > 0 else 0.0

    return {
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "precision": precision, "recall": recall, "f1": f1, "accuracy": accuracy,
    }


def print_matrix(name: str, m: dict, n_total: int, n_anomaly: int) -> None:
    n_normal = n_total - n_anomaly
    print(f"\n{'=' * 60}")
    print(f"  {name}")
    print(f"{'=' * 60}")
    print(f"  Dataset: {n_total} records ({n_anomaly} anomalous, {n_normal} normal)")
    print()
    print(f"                    Predicted")
    print(f"                    Normal  Anomaly")
    print(f"  Actual Normal     {m['tn']:>5}    {m['fp']:>5}")
    print(f"  Actual Anomaly    {m['fn']:>5}    {m['tp']:>5}")
    print()
    print(f"  Accuracy:   {m['accuracy']:.3f}")
    print(f"  Precision:  {m['precision']:.3f}")
    print(f"  Recall:     {m['recall']:.3f}")
    print(f"  F1 Score:   {m['f1']:.3f}")
    print(f"{'=' * 60}")


def print_synth_table(
    name: str,
    labels: list[str],
    predictions: list[bool],
    scores: list[float | None],
) -> None:
    caught = sum(predictions)
    total = len(predictions)
    print(f"\n{'=' * 60}")
    print(f"  {name} — Synthetic Anomalies")
    print(f"{'=' * 60}")
    for label, pred, score in zip(labels, predictions, scores):
        flag = "CAUGHT" if pred else "missed"
        sc = f"{score:.4f}" if score is not None else "n/a"
        print(f"  {label:42s}  {flag:6s}  (score={sc})")
    print(f"\n  Caught: {caught}/{total}  "
          f"({caught / total * 100:.0f}%)")
    print(f"{'=' * 60}")


def print_summary(results: dict) -> None:
    print(f"\n{'=' * 70}")
    print(f"  SUMMARY")
    print(f"{'=' * 70}")
    print(f"  {'Configuration':<30s}  {'Acc':>6s}  {'Prec':>6s}  {'Rec':>6s}  {'F1':>6s}  {'FP':>4s}  {'Synth':>7s}")
    print(f"  {'-' * 30}  {'-' * 6}  {'-' * 6}  {'-' * 6}  {'-' * 6}  {'-' * 4}  {'-' * 7}")
    for name, data in results.items():
        m = data["metrics"]
        sc = data.get("synth_caught")
        sc_str = f"{sc[0]}/{sc[1]}" if sc else "n/a"
        print(f"  {name:<30s}  {m['accuracy']:>6.3f}  {m['precision']:>6.3f}  "
              f"{m['recall']:>6.3f}  {m['f1']:>6.3f}  {m['fp']:>4d}  {sc_str:>7s}")
    print(f"{'=' * 70}")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def real_records() -> list[dict]:
    """Load real LFS records with complete SVM features."""
    df = pd.read_excel(DATASET_PATH)
    records = []
    for _, row in df.iterrows():
        record = {}
        for col in REQUIRED_COLS:
            val = row.get(col)
            if pd.notna(val):
                record[col] = val
        # edu_ordinal is derived from q_301 inside SVMDetector._preprocess,
        # so check only the raw features that must be present in the data.
        raw_svm_features = [f for f in SVM_FEATURES if f != "edu_ordinal"]
        if all(k in record for k in raw_svm_features):
            records.append(record)
    return records


@pytest.fixture(scope="module")
def ge_detector() -> GreatExpectationsDetector:
    return GreatExpectationsDetector(SUITE_PATH, preprocessor=add_derived_columns)


@pytest.fixture(scope="module")
def svm_detector() -> SVMDetector:
    return SVMDetector(
        training_data_path=TRAINING_DATA_PATH,
        model_path=MODEL_PATH,
        feature_columns=SVM_FEATURES,
        nu=0.1,
        gamma=0.001,
    )


@pytest.fixture(scope="module")
def llm_detector() -> LLMDetector | None:
    if not os.environ.get("SADEED_LLM"):
        return None
    try:
        import httpx
        resp = httpx.get("http://localhost:11434/api/tags", timeout=3.0)
        if resp.is_success:
            return LLMDetector(base_url="http://localhost:11434", model="gemma2:9b")
    except Exception:
        pass
    return None


@pytest.fixture(scope="module")
def synth_anomalies() -> list[dict]:
    return copy.deepcopy(SYNTHETIC_ANOMALIES)


# ---------------------------------------------------------------------------
# Async detection helpers
# ---------------------------------------------------------------------------


async def run_ge(det: GreatExpectationsDetector, records: list[dict]) -> list[StrategyResult]:
    results = []
    for r in tqdm(records, desc="GE", unit="rec"):
        results.append(await det.detect(r))
    return results


async def run_svm(det: SVMDetector, records: list[dict]) -> list[StrategyResult]:
    results = []
    for r in tqdm(records, desc="SVM", unit="rec"):
        results.append(await det.detect(r))
    return results


async def run_llm(
    det: LLMDetector,
    records: list[dict],
    svm_results: list[StrategyResult] | None = None,
    ge_results: list[StrategyResult] | None = None,
) -> list[StrategyResult]:
    results = []
    for i, r in enumerate(tqdm(records, desc="LLM", unit="rec")):
        svm_score = svm_results[i].score if svm_results else None
        ge_score = ge_results[i].score if ge_results else None
        results.append(await det.detect(r, svm_score=svm_score, ge_score=ge_score))
    return results


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_all_metrics(
    real_records: list[dict],
    synth_anomalies: list[dict],
    ge_detector: GreatExpectationsDetector,
    svm_detector: SVMDetector,
    llm_detector: LLMDetector | None,
) -> None:
    """Evaluate all detection layers and print comprehensive metrics."""

    n_total = len(real_records)
    synth_labels = [r.pop("_label") for r in synth_anomalies]
    n_synth = len(synth_anomalies)

    # ---------------------------------------------------------------
    # Run detectors on real records
    # ---------------------------------------------------------------
    ge_real = await run_ge(ge_detector, real_records)
    svm_real = await run_svm(svm_detector, real_records)

    # Ground truth from GE
    y_true = [r.is_anomaly for r in ge_real]
    n_anomaly = sum(y_true)

    y_ge = [r.is_anomaly for r in ge_real]
    y_svm = [r.is_anomaly for r in svm_real]
    y_ge_or_svm = [g or s for g, s in zip(y_ge, y_svm)]

    # ---------------------------------------------------------------
    # Run detectors on synthetic anomalies (all labeled True)
    # ---------------------------------------------------------------
    ge_synth = await run_ge(ge_detector, synth_anomalies)
    svm_synth = await run_svm(svm_detector, synth_anomalies)
    y_synth_true = [True] * n_synth

    synth_ge = [r.is_anomaly for r in ge_synth]
    synth_svm = [r.is_anomaly for r in svm_synth]
    synth_ge_or_svm = [g or s for g, s in zip(synth_ge, synth_svm)]

    # ---------------------------------------------------------------
    # LLM (optional)
    # ---------------------------------------------------------------
    y_llm: list[bool] | None = None
    synth_llm: list[bool] | None = None
    y_ge_or_svm_and_llm: list[bool] | None = None
    synth_full: list[bool] | None = None
    llm_real_results: list[StrategyResult] | None = None
    llm_synth_results: list[StrategyResult] | None = None

    if llm_detector is not None:
        llm_real_results = await run_llm(
            llm_detector, real_records,
            svm_results=svm_real, ge_results=ge_real,
        )
        y_llm = [r.is_anomaly for r in llm_real_results]

        # Full pipeline: (GE OR SVM) AND LLM
        # LLM only called when GE OR SVM flags, otherwise verdict = False
        y_ge_or_svm_and_llm = [
            (prelim and llm_v) if prelim else False
            for prelim, llm_v in zip(y_ge_or_svm, y_llm)
        ]

        llm_synth_results = await run_llm(
            llm_detector, synth_anomalies,
            svm_results=svm_synth, ge_results=ge_synth,
        )
        synth_llm = [r.is_anomaly for r in llm_synth_results]

        synth_full = [
            (prelim and llm_v) if prelim else False
            for prelim, llm_v in zip(synth_ge_or_svm, synth_llm)
        ]
    else:
        print("\n  LLM skipped (pass --llm to pytest to enable)")

    # ---------------------------------------------------------------
    # Print — Real records
    # ---------------------------------------------------------------
    print(f"\n\n{'#' * 60}")
    print(f"  PART 1: Real LFS Records ({n_total} records)")
    print(f"{'#' * 60}")

    m_ge = compute_metrics(y_true, y_ge)
    print_matrix("GE Alone", m_ge, n_total, n_anomaly)

    m_svm = compute_metrics(y_true, y_svm)
    print_matrix("SVM Alone", m_svm, n_total, n_anomaly)

    m_ge_svm = compute_metrics(y_true, y_ge_or_svm)
    print_matrix("GE OR SVM (preliminary verdict)", m_ge_svm, n_total, n_anomaly)

    if y_llm is not None:
        m_llm = compute_metrics(y_true, y_llm)
        print_matrix("LLM Alone", m_llm, n_total, n_anomaly)

    if y_ge_or_svm_and_llm is not None:
        m_full = compute_metrics(y_true, y_ge_or_svm_and_llm)
        print_matrix("(GE OR SVM) AND LLM (full pipeline)", m_full, n_total, n_anomaly)

    # ---------------------------------------------------------------
    # Print — Synthetic anomalies
    # ---------------------------------------------------------------
    print(f"\n\n{'#' * 60}")
    print(f"  PART 2: Synthetic Anomalies ({n_synth} records, all anomalous)")
    print(f"{'#' * 60}")

    print_synth_table("GE Alone", synth_labels, synth_ge,
                      [r.score for r in ge_synth])

    print_synth_table("SVM Alone", synth_labels, synth_svm,
                      [r.score for r in svm_synth])

    print_synth_table("GE OR SVM", synth_labels, synth_ge_or_svm,
                      [max(g.score or 0, s.score or 0) for g, s in zip(ge_synth, svm_synth)])

    if synth_llm is not None and llm_synth_results is not None:
        print_synth_table("LLM Alone", synth_labels, synth_llm,
                          [r.score for r in llm_synth_results])

    if synth_full is not None and llm_synth_results is not None:
        print_synth_table("(GE OR SVM) AND LLM", synth_labels, synth_full,
                          [r.score for r in llm_synth_results])

    # ---------------------------------------------------------------
    # Summary table
    # ---------------------------------------------------------------
    summary: dict = {
        "GE Alone": {
            "metrics": m_ge,
            "synth_caught": (sum(synth_ge), n_synth),
        },
        "SVM Alone": {
            "metrics": m_svm,
            "synth_caught": (sum(synth_svm), n_synth),
        },
        "GE OR SVM": {
            "metrics": m_ge_svm,
            "synth_caught": (sum(synth_ge_or_svm), n_synth),
        },
    }

    if y_llm is not None:
        summary["LLM Alone"] = {
            "metrics": compute_metrics(y_true, y_llm),
            "synth_caught": (sum(synth_llm), n_synth) if synth_llm else None,
        }

    if y_ge_or_svm_and_llm is not None:
        summary["(GE OR SVM) AND LLM"] = {
            "metrics": compute_metrics(y_true, y_ge_or_svm_and_llm),
            "synth_caught": (sum(synth_full), n_synth) if synth_full else None,
        }

    print_summary(summary)

    # ---------------------------------------------------------------
    # Assertions — sanity checks
    # ---------------------------------------------------------------
    # GE must be perfect against itself
    assert m_ge["f1"] == 1.0, "GE should have perfect F1 against its own ground truth"

    # Combined pipeline should have perfect recall on real records
    assert m_ge_svm["recall"] == 1.0, "GE OR SVM should catch all GE-flagged anomalies"

    # SVM should catch at least some synthetic anomalies
    assert sum(synth_svm) >= 4, f"SVM should catch at least 4/10 synthetic anomalies, got {sum(synth_svm)}"

    # GE rules should catch at least some synthetics
    assert sum(synth_ge) >= 1, f"GE should catch at least 1/10 synthetic anomalies, got {sum(synth_ge)}"
