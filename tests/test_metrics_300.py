"""
300-sample metrics evaluation with LLM explanations.

Tests multiple pipeline configurations:
  1. Current: (GE OR SVM>0.5) AND LLM
  2. Lower threshold: (GE OR SVM>0.3) AND LLM
  3. Two-pass: (GE OR SVM>0.5 OR SVM-borderline>0.2) AND LLM

Run with:
    SADEED_LLM=1 pytest tests/test_metrics_300.py -v -s
"""

from __future__ import annotations

import asyncio
import os
import random
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
# Paths & constants
# ---------------------------------------------------------------------------

SUITE_PATH = "expectations/suite.json"
TRAINING_DATA_PATH = "data/normal_samples.npy"
MODEL_PATH = "models/svm.joblib"
SVM_FEATURES = ["act_1_total", "age", "cut_5_total", "edu_ordinal", "q_602_val"]
OUTPUT_XLSX = "data/test_300_samples.xlsx"

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
# Record generators (same as test_metrics_large.py)
# ---------------------------------------------------------------------------

def _base_record(rng: random.Random, edu_code: int) -> dict:
    _, min_age, _, _ = EDU_PROFILES[edu_code]
    gender = rng.choice(GENDERS)
    family = rng.choice(FAMILY_RELATIONS)

    age_low = max(min_age, 15)
    if family == 1700001:
        age_low = max(age_low, 15)
    if family == 1700021:
        age_low = max(age_low, 18)
    age = rng.randint(age_low, min(age_low + 35, 70))

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
        multiplier = rng.uniform(3, 10)
        salary = min(int(sal_high * multiplier), 49999)
        rec["q_602_val"] = salary
        rec["cut_5_total"] = rng.randint(30, 50)
        rec["act_1_total"] = rec["cut_5_total"] + rng.randint(-5, 5)
        label = f"Salary {salary} too high for {EDU_PROFILES[edu_code][0]} (max {sal_high})"

    elif anomaly_type == "salary_too_low":
        edu_code = rng.choice([10500025, 10500009, 10500010])
        _, _, sal_low, sal_high = EDU_PROFILES[edu_code]
        rec = _base_record(rng, edu_code)
        salary = rng.randint(500, max(501, sal_low // 5))
        rec["q_602_val"] = salary
        rec["cut_5_total"] = rng.randint(30, 50)
        rec["act_1_total"] = rec["cut_5_total"] + rng.randint(-5, 5)
        label = f"Salary {salary} too low for {EDU_PROFILES[edu_code][0]} (min {sal_low})"

    elif anomaly_type == "hours_mismatch_high_low":
        rec["q_602_val"] = rng.randint(sal_low, sal_high)
        usual = rng.randint(60, 84)
        actual = rng.randint(0, 10)
        rec["cut_5_total"] = usual
        rec["act_1_total"] = actual
        label = f"Hours mismatch: {usual} usual, {actual} actual (gap {usual - actual})"

    elif anomaly_type == "hours_mismatch_low_high":
        rec["q_602_val"] = rng.randint(sal_low, sal_high)
        usual = rng.randint(1, 10)
        actual = rng.randint(60, 84)
        rec["cut_5_total"] = usual
        rec["act_1_total"] = actual
        label = f"Hours mismatch: {usual} usual, {actual} actual (gap {actual - usual})"

    elif anomaly_type == "few_hours_high_salary":
        rec["q_602_val"] = rng.randint(20000, 49999)
        hours = rng.randint(1, 5)
        rec["cut_5_total"] = hours
        rec["act_1_total"] = hours
        label = f"{hours} hrs/wk at {rec['q_602_val']} SAR"

    elif anomaly_type == "extreme_hours_low_salary":
        rec["q_602_val"] = rng.randint(500, 999)
        hours = rng.randint(70, 84)
        rec["cut_5_total"] = hours
        rec["act_1_total"] = hours + rng.randint(-3, 3)
        rec["act_1_total"] = max(0, min(84, rec["act_1_total"]))
        label = f"{hours} hrs/wk at {rec['q_602_val']} SAR"

    elif anomaly_type == "elderly_extreme_hours":
        rec["age"] = rng.randint(65, 80)
        rec["q_602_val"] = rng.randint(sal_low, sal_high)
        hours = rng.randint(70, 84)
        rec["cut_5_total"] = hours
        rec["act_1_total"] = hours
        label = f"{rec['age']}yo working {hours} hrs/wk"

    elif anomaly_type == "young_extreme_hours":
        edu_code = rng.choice([10500031, 10500003, 10500017])
        rec = _base_record(rng, edu_code)
        rec["age"] = rng.randint(15, 18)
        rec["family_relation"] = 1700022
        rec["marage_status"] = 10600001
        rec["q_602_val"] = rng.randint(500, 3000)
        hours = rng.randint(70, 84)
        rec["cut_5_total"] = hours
        rec["act_1_total"] = hours
        label = f"{rec['age']}yo working {hours} hrs/wk"

    elif anomaly_type == "salary_hours_combo":
        edu_code = rng.choice([10500031, 10500003, 10500017, 10500019])
        _, _, sal_low, sal_high = EDU_PROFILES[edu_code]
        rec = _base_record(rng, edu_code)
        rec["q_602_val"] = rng.randint(30000, 49999)
        rec["cut_5_total"] = rng.randint(1, 5)
        rec["act_1_total"] = rng.randint(1, 5)
        label = f"{EDU_PROFILES[edu_code][0]}, {rec['q_602_val']} SAR, {rec['cut_5_total']} hrs"

    else:  # education_salary_age_combo
        edu_code = rng.choice([10500009, 10500010])
        _, min_age, sal_low, sal_high = EDU_PROFILES[edu_code]
        rec = _base_record(rng, edu_code)
        rec["age"] = min_age
        rec["q_602_val"] = rng.randint(500, max(501, sal_low // 4))
        rec["cut_5_total"] = rng.randint(30, 50)
        rec["act_1_total"] = rec["cut_5_total"] + rng.randint(-5, 5)
        label = f"{rec['age']}yo {EDU_PROFILES[edu_code][0]}, {rec['q_602_val']} SAR"

    return rec, label


# ---------------------------------------------------------------------------
# Dataset generation
# ---------------------------------------------------------------------------

def generate_dataset(
    n_normal: int = 150,
    n_anomaly: int = 150,
    seed: int = 99,
) -> tuple[list[dict], list[bool], list[str]]:
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

    combined = list(zip(records, labels, descriptions))
    rng.shuffle(combined)
    records, labels, descriptions = zip(*combined)  # type: ignore[assignment]
    return list(records), list(labels), list(descriptions)


def save_to_excel(
    records: list[dict],
    labels: list[bool],
    descriptions: list[str],
    path: str,
) -> None:
    rows = []
    for rec, lbl, desc in zip(records, labels, descriptions):
        row = dict(rec)
        row["_ground_truth"] = "anomaly" if lbl else "normal"
        row["_description"] = desc
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_excel(path, index=False)
    print(f"\n  Saved {len(rows)} records to {path}")


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


def print_summary(title: str, results: dict) -> None:
    print(f"\n{'=' * 90}")
    print(f"  {title}")
    print(f"{'=' * 90}")
    print(f"  {'Configuration':<40s}  {'Acc':>6s}  {'Prec':>6s}  {'Rec':>6s}"
          f"  {'F1':>6s}  {'TP':>4s}  {'FP':>4s}  {'FN':>4s}  {'TN':>4s}  {'LLM calls':>9s}")
    print(f"  {'-' * 40}  {'-' * 6}  {'-' * 6}  {'-' * 6}"
          f"  {'-' * 6}  {'-' * 4}  {'-' * 4}  {'-' * 4}  {'-' * 4}  {'-' * 9}")
    for name, (m, llm_calls) in results.items():
        print(f"  {name:<40s}  {m['accuracy']:>6.3f}  {m['precision']:>6.3f}  "
              f"{m['recall']:>6.3f}  {m['f1']:>6.3f}  {m['tp']:>4d}  "
              f"{m['fp']:>4d}  {m['fn']:>4d}  {m['tn']:>4d}  {llm_calls:>9d}")
    print(f"{'=' * 90}")


# ---------------------------------------------------------------------------
# Pipeline simulation helpers
# ---------------------------------------------------------------------------

def simulate_pipeline(
    y_true: list[bool],
    y_ge: list[bool],
    svm_scores: list[float],
    y_llm: list[bool],
    svm_threshold: float = 0.5,
    borderline_threshold: float | None = None,
) -> tuple[list[bool], int]:
    """Simulate a pipeline with configurable SVM thresholds.

    Args:
        svm_threshold: SVM score above which a record is flagged for LLM.
        borderline_threshold: If set, records with SVM score between this and
            svm_threshold are also sent to LLM (two-pass).

    Returns:
        (predictions, llm_call_count)
    """
    n = len(y_true)
    predictions: list[bool] = []
    llm_calls = 0

    for i in range(n):
        ge_flag = y_ge[i]
        svm_score = svm_scores[i]
        svm_flag = svm_score >= svm_threshold

        # Determine if LLM should be called
        send_to_llm = ge_flag or svm_flag
        if borderline_threshold is not None and not send_to_llm:
            send_to_llm = svm_score >= borderline_threshold

        if send_to_llm:
            llm_calls += 1
            # Final verdict: preliminary AND LLM confirms
            predictions.append(y_llm[i])
        else:
            predictions.append(False)

    return predictions, llm_calls


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def dataset() -> tuple[list[dict], list[bool], list[str]]:
    records, labels, descriptions = generate_dataset(
        n_normal=150, n_anomaly=150, seed=99,
    )
    save_to_excel(records, labels, descriptions, OUTPUT_XLSX)
    return records, labels, descriptions


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
async def test_300_threshold_comparison(
    dataset: tuple[list[dict], list[bool], list[str]],
    ge_detector: GreatExpectationsDetector,
    svm_detector: SVMDetector,
    llm_detector: LLMDetector | None,
) -> None:
    """Compare pipeline configurations with different SVM thresholds."""

    records, y_true, descriptions = dataset
    n_total = len(records)
    n_anomaly = sum(y_true)
    n_normal = n_total - n_anomaly

    print(f"\n\n  Generated {n_total} records: {n_anomaly} anomalous, {n_normal} normal\n")

    if llm_detector is None:
        pytest.skip("LLM not available (set SADEED_LLM=1 and ensure Ollama is running)")

    # --- Run GE + SVM ---
    ge_results: list[StrategyResult] = []
    for r in tqdm(records, desc="GE", unit="rec"):
        ge_results.append(await ge_detector.detect(r))

    svm_results: list[StrategyResult] = []
    for r in tqdm(records, desc="SVM", unit="rec"):
        svm_results.append(await svm_detector.detect(r))

    y_ge = [r.is_anomaly for r in ge_results]
    svm_scores = [r.score for r in svm_results]

    # --- Analyze SVM score distribution for anomalies ---
    anom_svm_scores = sorted([s for s, t in zip(svm_scores, y_true) if t], reverse=True)
    norm_svm_scores = sorted([s for s, t in zip(svm_scores, y_true) if not t], reverse=True)

    print(f"\n  SVM score distribution:")
    for thresh in [0.5, 0.4, 0.3, 0.2, 0.1]:
        anom_above = sum(1 for s in anom_svm_scores if s >= thresh)
        norm_above = sum(1 for s in norm_svm_scores if s >= thresh)
        print(f"    >= {thresh}: {anom_above}/{n_anomaly} anomalies, "
              f"{norm_above}/{n_normal} normals would be sent to LLM")

    # --- Run LLM on ALL records (we need LLM verdicts for all threshold sims) ---
    print(f"\n  Running LLM on all {n_total} records to enable threshold comparison...\n")

    llm_results: list[StrategyResult] = []
    for i, r in enumerate(tqdm(records, desc="LLM (all)", unit="rec")):
        result = await llm_detector.detect(
            r,
            svm_score=svm_results[i].score,
            ge_score=ge_results[i].score,
        )
        llm_results.append(result)

    y_llm = [r.is_anomaly for r in llm_results]

    # --- Simulate different pipeline configurations ---
    configs: dict[str, tuple[dict, int]] = {}

    # Baselines
    m_ge = compute_metrics(y_true, y_ge)
    configs["GE Alone"] = (m_ge, 0)

    y_svm_binary = [s >= 0.5 for s in svm_scores]
    m_svm = compute_metrics(y_true, y_svm_binary)
    configs["SVM Alone (>0.5)"] = (m_svm, 0)

    m_llm = compute_metrics(y_true, y_llm)
    configs["LLM Alone (all records)"] = (m_llm, n_total)

    # Current pipeline: SVM>0.5 AND LLM
    preds, calls = simulate_pipeline(y_true, y_ge, svm_scores, y_llm,
                                     svm_threshold=0.5)
    configs["CURRENT: (GE|SVM>0.5) AND LLM"] = (compute_metrics(y_true, preds), calls)

    # Lower SVM thresholds
    for thresh in [0.4, 0.3, 0.2, 0.1]:
        preds, calls = simulate_pipeline(y_true, y_ge, svm_scores, y_llm,
                                         svm_threshold=thresh)
        configs[f"Lower: (GE|SVM>{thresh}) AND LLM"] = (
            compute_metrics(y_true, preds), calls,
        )

    # Two-pass: SVM>0.5 + borderline
    for border in [0.3, 0.2, 0.1]:
        preds, calls = simulate_pipeline(y_true, y_ge, svm_scores, y_llm,
                                         svm_threshold=0.5,
                                         borderline_threshold=border)
        configs[f"Two-pass: SVM>0.5 + border>{border}"] = (
            compute_metrics(y_true, preds), calls,
        )

    # --- Print results ---
    print_summary(
        f"THRESHOLD COMPARISON — 300 Records ({n_normal} normal + {n_anomaly} anomalous)",
        configs,
    )

    # --- Detailed analysis of best config ---
    # Find config with best F1 (excluding LLM-alone and baselines)
    pipeline_configs = {k: v for k, v in configs.items()
                        if "AND LLM" in k or "Two-pass" in k}
    best_name = max(pipeline_configs, key=lambda k: pipeline_configs[k][0]["f1"])
    best_m, best_calls = pipeline_configs[best_name]

    print(f"\n  BEST PIPELINE: {best_name}")
    print(f"    F1={best_m['f1']:.3f}  Prec={best_m['precision']:.3f}  "
          f"Rec={best_m['recall']:.3f}  LLM calls={best_calls}")

    current_m = configs["CURRENT: (GE|SVM>0.5) AND LLM"][0]
    print(f"\n  vs CURRENT pipeline:")
    print(f"    F1:     {current_m['f1']:.3f} -> {best_m['f1']:.3f}  "
          f"({best_m['f1'] - current_m['f1']:+.3f})")
    print(f"    Recall: {current_m['recall']:.3f} -> {best_m['recall']:.3f}  "
          f"({best_m['recall'] - current_m['recall']:+.3f})")
    print(f"    Prec:   {current_m['precision']:.3f} -> {best_m['precision']:.3f}  "
          f"({best_m['precision'] - current_m['precision']:+.3f})")

    # --- Show what the best config catches that current misses ---
    best_preds, _ = simulate_pipeline(y_true, y_ge, svm_scores, y_llm,
                                      svm_threshold=0.5,
                                      borderline_threshold=0.2)
    current_preds, _ = simulate_pipeline(y_true, y_ge, svm_scores, y_llm,
                                         svm_threshold=0.5)

    newly_caught = [(i, descriptions[i]) for i in range(n_total)
                    if y_true[i] and best_preds[i] and not current_preds[i]]
    new_fps = [(i, descriptions[i]) for i in range(n_total)
               if not y_true[i] and best_preds[i] and not current_preds[i]]

    if newly_caught:
        print(f"\n  NEWLY CAUGHT by two-pass (SVM>0.5 + border>0.2): {len(newly_caught)}")
        for idx, desc in newly_caught[:20]:
            svm_s = svm_scores[idx]
            llm_s = llm_results[idx].score
            print(f"    SVM={svm_s:.2f} LLM={llm_s:.2f}  {desc}")
        if len(newly_caught) > 20:
            print(f"    ... and {len(newly_caught) - 20} more")

    if new_fps:
        print(f"\n  NEW FALSE POSITIVES from two-pass: {len(new_fps)}")
        for idx, desc in new_fps[:10]:
            svm_s = svm_scores[idx]
            llm_s = llm_results[idx].score
            llm_exp = llm_results[idx].explanation or ""
            print(f"    SVM={svm_s:.2f} LLM={llm_s:.2f}  {desc}")
            if llm_exp:
                print(f"      LLM: {llm_exp[:120]}")

    # --- LLM explanations for a sample of true positives ---
    tp_indices = [i for i in range(n_total) if y_true[i] and best_preds[i]][:15]
    if tp_indices:
        print(f"\n\n{'#' * 70}")
        print(f"  LLM EXPLANATIONS (sample of {len(tp_indices)} true positives)")
        print(f"{'#' * 70}")

        explanations: dict[int, str] = {}
        for i in tqdm(tp_indices, desc="Explain", unit="rec"):
            exp = await llm_detector.explain(
                records[i],
                ge_score=ge_results[i].score,
                ge_failed_rules=0,
                svm_score=svm_results[i].score,
            )
            explanations[i] = exp

        for idx, exp in explanations.items():
            desc = descriptions[idx]
            svm_s = svm_scores[idx]
            llm_s = llm_results[idx].score
            print(f"\n  [{idx}] {desc}")
            print(f"       SVM={svm_s:.2f}  LLM={llm_s:.2f}")
            print(f"       {exp}")

    # --- Save full results ---
    result_rows = []
    for i in range(n_total):
        row = dict(records[i])
        row["_ground_truth"] = "anomaly" if y_true[i] else "normal"
        row["_description"] = descriptions[i]
        row["ge_anomaly"] = y_ge[i]
        row["ge_score"] = ge_results[i].score
        row["svm_anomaly"] = svm_scores[i] >= 0.5
        row["svm_score"] = svm_scores[i]
        row["llm_anomaly"] = y_llm[i]
        row["llm_score"] = llm_results[i].score
        row["llm_explanation"] = llm_results[i].explanation or ""
        row["current_verdict"] = current_preds[i]
        row["twopass_verdict"] = best_preds[i]
        result_rows.append(row)

    results_path = "data/test_300_results.xlsx"
    pd.DataFrame(result_rows).to_excel(results_path, index=False)
    print(f"\n  Full results saved to {results_path}")
