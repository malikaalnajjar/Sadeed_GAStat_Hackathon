"""
Large-scale metrics evaluation (400 synthetic records).

Generates 200 normal + 200 anomalous records with known ground truth.
All records pass GE rules — anomalies are semantic (salary/hours/age mismatches).

Run with:
    pytest tests/test_metrics_large.py -v -s              # GE + SVM only
    SADEED_LLM=1 pytest tests/test_metrics_large.py -v -s # include LLM
"""

from __future__ import annotations

import asyncio
import os
import random
import sys
from pathlib import Path

import pytest
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.detection.great_expectations_strategy import GreatExpectationsDetector
from backend.detection.lfs_preprocessing import add_derived_columns
from backend.detection.llm_strategy import LLMDetector
from backend.detection.svm_strategy import SVMDetector
from backend.models.schemas import StrategyResult

# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------

SUITE_PATH = "expectations/suite.json"
TRAINING_DATA_PATH = "data/normal_samples.npy"
MODEL_PATH = "models/svm.joblib"
SVM_FEATURES = ["act_1_total", "age", "cut_5_total", "edu_ordinal", "q_602_val"]

# Education code -> (label, min_age, salary_low, salary_high)
EDU_PROFILES: dict[int, tuple[str, int, int, int]] = {
    10500031: ("No formal edu", 15, 1500, 5000),
    10500003: ("Primary", 15, 1500, 5000),
    10500017: ("Intermediate", 15, 2000, 8000),
    10500019: ("Secondary", 17, 3000, 12000),
    10500023: ("Diploma", 19, 4000, 18000),
    10500025: ("Bachelor's", 21, 5000, 25000),
    10500009: ("Master's", 23, 8000, 35000),
    10500010: ("PhD", 25, 12000, 45000),
}

GENDERS = [1600001, 1600002]
FAMILY_RELATIONS = [1700001, 1700021, 1700022]
MARITAL_STATUSES = [10600001, 10600002, 10600003, 10600004]


# ---------------------------------------------------------------------------
# Record generators
# ---------------------------------------------------------------------------


def _base_record(rng: random.Random, edu_code: int) -> dict:
    """Build a valid base record for the given education level."""
    _, min_age, _, _ = EDU_PROFILES[edu_code]
    gender = rng.choice(GENDERS)
    family = rng.choice(FAMILY_RELATIONS)

    # Age: respect education minimum + family constraints
    age_low = max(min_age, 15)
    if family == 1700001:  # Head
        age_low = max(age_low, 15)
    if family == 1700021:  # Spouse
        age_low = max(age_low, 18)
    age = rng.randint(age_low, min(age_low + 35, 70))

    # Spouse must be married
    if family == 1700021:
        marital = 10600002
    else:
        marital = rng.choice(MARITAL_STATUSES)

    return {
        "age": age,
        "gender": gender,
        "family_relation": family,
        "marage_status": marital,
        "nationality": 1800001,
        "q_301": edu_code,
    }


def generate_normal(rng: random.Random) -> dict:
    """Generate a plausible normal record."""
    edu_code = rng.choice(list(EDU_PROFILES.keys()))
    _, _, sal_low, sal_high = EDU_PROFILES[edu_code]

    rec = _base_record(rng, edu_code)
    rec["q_602_val"] = rng.randint(sal_low, sal_high)

    usual = rng.randint(25, 55)
    actual = usual + rng.randint(-8, 8)
    actual = max(1, min(84, actual))
    rec["cut_5_total"] = usual
    rec["act_1_total"] = actual
    return rec


def generate_anomaly(rng: random.Random) -> tuple[dict, str]:
    """Generate an anomalous record that passes GE rules. Returns (record, label)."""
    anomaly_type = rng.choice([
        "salary_too_high",
        "salary_too_low",
        "hours_mismatch_high_low",
        "hours_mismatch_low_high",
        "few_hours_high_salary",
        "extreme_hours_low_salary",
        "elderly_extreme_hours",
        "young_extreme_hours",
        "salary_hours_combo",
        "education_salary_age_combo",
    ])

    edu_code = rng.choice(list(EDU_PROFILES.keys()))
    _, _, sal_low, sal_high = EDU_PROFILES[edu_code]
    rec = _base_record(rng, edu_code)

    if anomaly_type == "salary_too_high":
        # Salary 3-10x above the education range
        multiplier = rng.uniform(3, 10)
        salary = min(int(sal_high * multiplier), 49999)
        rec["q_602_val"] = salary
        rec["cut_5_total"] = rng.randint(30, 50)
        rec["act_1_total"] = rec["cut_5_total"] + rng.randint(-5, 5)
        label = f"Salary {salary} too high for {EDU_PROFILES[edu_code][0]} (max {sal_high})"

    elif anomaly_type == "salary_too_low":
        # Pick a high-education code, give it a poverty salary
        edu_code = rng.choice([10500025, 10500009, 10500010])
        _, _, sal_low, sal_high = EDU_PROFILES[edu_code]
        rec = _base_record(rng, edu_code)
        salary = rng.randint(500, max(501, sal_low // 5))
        rec["q_602_val"] = salary
        rec["cut_5_total"] = rng.randint(30, 50)
        rec["act_1_total"] = rec["cut_5_total"] + rng.randint(-5, 5)
        label = f"Salary {salary} too low for {EDU_PROFILES[edu_code][0]} (min {sal_low})"

    elif anomaly_type == "hours_mismatch_high_low":
        # High usual, very low actual (gap > 40)
        rec["q_602_val"] = rng.randint(sal_low, sal_high)
        usual = rng.randint(60, 84)
        actual = rng.randint(0, 10)
        rec["cut_5_total"] = usual
        rec["act_1_total"] = actual
        label = f"Hours mismatch: {usual} usual, {actual} actual (gap {usual - actual})"

    elif anomaly_type == "hours_mismatch_low_high":
        # Low usual, very high actual (gap > 40)
        rec["q_602_val"] = rng.randint(sal_low, sal_high)
        usual = rng.randint(1, 10)
        actual = rng.randint(60, 84)
        rec["cut_5_total"] = usual
        rec["act_1_total"] = actual
        label = f"Hours mismatch: {usual} usual, {actual} actual (gap {actual - usual})"

    elif anomaly_type == "few_hours_high_salary":
        # <=5 hours but salary > 20k
        rec["q_602_val"] = rng.randint(20000, 49999)
        hours = rng.randint(1, 5)
        rec["cut_5_total"] = hours
        rec["act_1_total"] = hours
        label = f"{hours} hrs/wk at {rec['q_602_val']} SAR"

    elif anomaly_type == "extreme_hours_low_salary":
        # 70+ hours but salary < 1000
        rec["q_602_val"] = rng.randint(500, 999)
        hours = rng.randint(70, 84)
        rec["cut_5_total"] = hours
        rec["act_1_total"] = hours + rng.randint(-3, 3)
        rec["act_1_total"] = max(0, min(84, rec["act_1_total"]))
        label = f"{hours} hrs/wk at {rec['q_602_val']} SAR"

    elif anomaly_type == "elderly_extreme_hours":
        # Age 65+ with 70+ hours
        rec["age"] = rng.randint(65, 80)
        rec["q_602_val"] = rng.randint(sal_low, sal_high)
        hours = rng.randint(70, 84)
        rec["cut_5_total"] = hours
        rec["act_1_total"] = hours
        label = f"{rec['age']}yo working {hours} hrs/wk"

    elif anomaly_type == "young_extreme_hours":
        # Age 15-18, 70+ hours
        edu_code = rng.choice([10500031, 10500003, 10500017])
        rec = _base_record(rng, edu_code)
        rec["age"] = rng.randint(15, 18)
        rec["family_relation"] = 1700022  # Son/daughter
        rec["marage_status"] = 10600001  # Never married
        rec["q_602_val"] = rng.randint(500, 3000)
        hours = rng.randint(70, 84)
        rec["cut_5_total"] = hours
        rec["act_1_total"] = hours
        label = f"{rec['age']}yo working {hours} hrs/wk"

    elif anomaly_type == "salary_hours_combo":
        # High salary + extreme hours + wrong education
        edu_code = rng.choice([10500031, 10500003, 10500017, 10500019])
        _, _, sal_low, sal_high = EDU_PROFILES[edu_code]
        rec = _base_record(rng, edu_code)
        rec["q_602_val"] = rng.randint(30000, 49999)
        rec["cut_5_total"] = rng.randint(1, 5)
        rec["act_1_total"] = rng.randint(1, 5)
        label = f"{EDU_PROFILES[edu_code][0]}, {rec['q_602_val']} SAR, {rec['cut_5_total']} hrs"

    else:  # education_salary_age_combo
        # Young person with high education and mismatched salary
        edu_code = rng.choice([10500009, 10500010])
        _, min_age, sal_low, sal_high = EDU_PROFILES[edu_code]
        rec = _base_record(rng, edu_code)
        rec["age"] = min_age  # Youngest possible for this education
        rec["q_602_val"] = rng.randint(500, max(501, sal_low // 4))
        rec["cut_5_total"] = rng.randint(30, 50)
        rec["act_1_total"] = rec["cut_5_total"] + rng.randint(-5, 5)
        label = f"{rec['age']}yo {EDU_PROFILES[edu_code][0]}, {rec['q_602_val']} SAR"

    return rec, label


# ---------------------------------------------------------------------------
# Generate dataset
# ---------------------------------------------------------------------------

def generate_dataset(
    n_normal: int = 200,
    n_anomaly: int = 200,
    seed: int = 42,
) -> tuple[list[dict], list[bool], list[str]]:
    """Generate a labelled dataset of normal + anomalous records.

    Returns:
        (records, labels, descriptions) where labels[i] is True if anomalous.
    """
    rng = random.Random(seed)
    records: list[dict] = []
    labels: list[bool] = []
    descriptions: list[str] = []

    for _ in range(n_normal):
        rec = generate_normal(rng)
        records.append(rec)
        labels.append(False)
        descriptions.append("normal")

    for _ in range(n_anomaly):
        rec, desc = generate_anomaly(rng)
        records.append(rec)
        labels.append(True)
        descriptions.append(desc)

    # Shuffle deterministically
    combined = list(zip(records, labels, descriptions))
    rng.shuffle(combined)
    records, labels, descriptions = zip(*combined)  # type: ignore[assignment]
    return list(records), list(labels), list(descriptions)


# ---------------------------------------------------------------------------
# Metrics helpers (same as test_metrics.py)
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


def print_summary(results: dict) -> None:
    print(f"\n{'=' * 70}")
    print(f"  SUMMARY — 400 Synthetic Records (200 normal + 200 anomalous)")
    print(f"{'=' * 70}")
    print(f"  {'Configuration':<30s}  {'Acc':>6s}  {'Prec':>6s}  {'Rec':>6s}  {'F1':>6s}  {'TP':>4s}  {'FP':>4s}  {'FN':>4s}")
    print(f"  {'-' * 30}  {'-' * 6}  {'-' * 6}  {'-' * 6}  {'-' * 6}  {'-' * 4}  {'-' * 4}  {'-' * 4}")
    for name, m in results.items():
        print(f"  {name:<30s}  {m['accuracy']:>6.3f}  {m['precision']:>6.3f}  "
              f"{m['recall']:>6.3f}  {m['f1']:>6.3f}  {m['tp']:>4d}  {m['fp']:>4d}  {m['fn']:>4d}")
    print(f"{'=' * 70}")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def dataset() -> tuple[list[dict], list[bool], list[str]]:
    return generate_dataset(n_normal=200, n_anomaly=200, seed=42)


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


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_large_scale_metrics(
    dataset: tuple[list[dict], list[bool], list[str]],
    ge_detector: GreatExpectationsDetector,
    svm_detector: SVMDetector,
    llm_detector: LLMDetector | None,
) -> None:
    """Evaluate all detection layers on 400 synthetic records."""

    records, y_true, descriptions = dataset
    n_total = len(records)
    n_anomaly = sum(y_true)

    print(f"\n\n  Generated {n_total} records: {n_anomaly} anomalous, "
          f"{n_total - n_anomaly} normal\n")

    # --- Run GE + SVM ---
    ge_results: list[StrategyResult] = []
    for r in tqdm(records, desc="GE", unit="rec"):
        ge_results.append(await ge_detector.detect(r))

    svm_results: list[StrategyResult] = []
    for r in tqdm(records, desc="SVM", unit="rec"):
        svm_results.append(await svm_detector.detect(r))

    y_ge = [r.is_anomaly for r in ge_results]
    y_svm = [r.is_anomaly for r in svm_results]
    y_ge_or_svm = [g or s for g, s in zip(y_ge, y_svm)]

    # --- Run LLM (confirmation mode with SVM context) ---
    y_llm: list[bool] | None = None
    y_full: list[bool] | None = None

    if llm_detector is not None:
        llm_results: list[StrategyResult] = []
        for i, r in enumerate(tqdm(records, desc="LLM", unit="rec")):
            result = await llm_detector.detect(
                r,
                svm_score=svm_results[i].score,
                ge_score=ge_results[i].score,
            )
            llm_results.append(result)

        y_llm = [r.is_anomaly for r in llm_results]
        y_full = [
            (prelim and llm_v) if prelim else False
            for prelim, llm_v in zip(y_ge_or_svm, y_llm)
        ]
    else:
        print("\n  LLM skipped (set SADEED_LLM=1 to enable)")

    # --- Print results ---
    m_ge = compute_metrics(y_true, y_ge)
    print_matrix("GE Alone", m_ge, n_total, n_anomaly)

    m_svm = compute_metrics(y_true, y_svm)
    print_matrix("SVM Alone", m_svm, n_total, n_anomaly)

    m_ge_svm = compute_metrics(y_true, y_ge_or_svm)
    print_matrix("GE OR SVM", m_ge_svm, n_total, n_anomaly)

    if y_llm is not None:
        m_llm = compute_metrics(y_true, y_llm)
        print_matrix("LLM Alone", m_llm, n_total, n_anomaly)

    if y_full is not None:
        m_full = compute_metrics(y_true, y_full)
        print_matrix("(GE OR SVM) AND LLM", m_full, n_total, n_anomaly)

    # --- Print missed anomalies ---
    if y_full is not None:
        missed = [(desc, y_true[i], y_ge_or_svm[i], y_llm[i])
                  for i, desc in enumerate(descriptions)
                  if y_true[i] and not y_full[i]]
        if missed:
            print(f"\n  Missed anomalies ({len(missed)}):")
            for desc, _, prelim, llm_v in missed[:20]:
                reason = "SVM missed" if not prelim else "LLM denied"
                print(f"    [{reason}] {desc}")
            if len(missed) > 20:
                print(f"    ... and {len(missed) - 20} more")

    # --- Print false positives ---
    if y_full is not None:
        fps = [(desc, svm_results[i].score, ge_results[i].score)
               for i, desc in enumerate(descriptions)
               if not y_true[i] and y_full[i]]
        if fps:
            print(f"\n  False positives ({len(fps)}):")
            for desc, svm_s, ge_s in fps[:20]:
                print(f"    [SVM={svm_s:.2f} GE={ge_s:.2f}] {desc}")

    # --- Summary table ---
    summary: dict = {
        "GE Alone": m_ge,
        "SVM Alone": m_svm,
        "GE OR SVM": m_ge_svm,
    }
    if y_llm is not None:
        summary["LLM Alone"] = compute_metrics(y_true, y_llm)
    if y_full is not None:
        summary["(GE OR SVM) AND LLM"] = compute_metrics(y_true, y_full)

    print_summary(summary)

    # --- Assertions ---
    # GE catches rule violations (not statistical outliers — that's the SVM's job)
    assert m_ge["recall"] > 0.1, f"GE recall too low: {m_ge['recall']:.2f}"

    # SVM should catch a meaningful chunk
    assert m_svm["recall"] > 0.3, f"SVM recall too low: {m_svm['recall']:.2f}"

    # Full pipeline should beat SVM alone on precision
    if y_full is not None:
        assert m_full["precision"] >= m_svm["precision"] - 0.05, \
            "Full pipeline precision should be at least close to SVM"
