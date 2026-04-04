"""
Ensemble LLM metrics — gemma2:9b + mistral agreement vs single model.

Reuses the same 300 synthetic records (seed=99) for fair comparison.

Run with:
    SADEED_LLM=1 pytest tests/test_metrics_ensemble.py -v -s
"""

from __future__ import annotations

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
from backend.detection.ensemble_llm_strategy import EnsembleLLMDetector
from backend.detection.svm_strategy import SVMDetector
from backend.models.schemas import StrategyResult

# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------

SUITE_PATH = "expectations/suite.json"
TRAINING_DATA_PATH = "data/normal_samples.npy"
MODEL_PATH = "models/svm.joblib"
SVM_FEATURES = ["act_1_total", "age", "cut_5_total", "edu_ordinal", "q_602_val"]

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
# Generators (identical to test_metrics_300.py, same seed for consistency)
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
        "age": age, "gender": gender, "family_relation": family,
        "marage_status": marital, "nationality": 1800001, "q_301": edu_code,
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
        "salary_too_high", "salary_too_low",
        "hours_mismatch_high_low", "hours_mismatch_low_high",
        "few_hours_high_salary", "extreme_hours_low_salary",
        "elderly_extreme_hours", "young_extreme_hours",
        "salary_hours_combo", "education_salary_age_combo",
    ])
    edu_code = rng.choice(list(EDU_PROFILES.keys()))
    _, _, sal_low, sal_high = EDU_PROFILES[edu_code]
    rec = _base_record(rng, edu_code)

    if anomaly_type == "salary_too_high":
        salary = min(int(sal_high * rng.uniform(3, 10)), 49999)
        rec["q_602_val"] = salary
        rec["cut_5_total"] = rng.randint(30, 50)
        rec["act_1_total"] = rec["cut_5_total"] + rng.randint(-5, 5)
        label = f"Salary {salary} too high for {EDU_PROFILES[edu_code][0]}"
    elif anomaly_type == "salary_too_low":
        edu_code = rng.choice([10500025, 10500009, 10500010])
        _, _, sal_low, sal_high = EDU_PROFILES[edu_code]
        rec = _base_record(rng, edu_code)
        salary = rng.randint(500, max(501, sal_low // 5))
        rec["q_602_val"] = salary
        rec["cut_5_total"] = rng.randint(30, 50)
        rec["act_1_total"] = rec["cut_5_total"] + rng.randint(-5, 5)
        label = f"Salary {salary} too low for {EDU_PROFILES[edu_code][0]}"
    elif anomaly_type == "hours_mismatch_high_low":
        rec["q_602_val"] = rng.randint(sal_low, sal_high)
        usual = rng.randint(60, 84)
        actual = rng.randint(0, 10)
        rec["cut_5_total"] = usual
        rec["act_1_total"] = actual
        label = f"Hours: {usual} usual, {actual} actual"
    elif anomaly_type == "hours_mismatch_low_high":
        rec["q_602_val"] = rng.randint(sal_low, sal_high)
        usual = rng.randint(1, 10)
        actual = rng.randint(60, 84)
        rec["cut_5_total"] = usual
        rec["act_1_total"] = actual
        label = f"Hours: {usual} usual, {actual} actual"
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
    else:
        edu_code = rng.choice([10500009, 10500010])
        _, min_age, sal_low, sal_high = EDU_PROFILES[edu_code]
        rec = _base_record(rng, edu_code)
        rec["age"] = min_age
        rec["q_602_val"] = rng.randint(500, max(501, sal_low // 4))
        rec["cut_5_total"] = rng.randint(30, 50)
        rec["act_1_total"] = rec["cut_5_total"] + rng.randint(-5, 5)
        label = f"{rec['age']}yo {EDU_PROFILES[edu_code][0]}, {rec['q_602_val']} SAR"
    return rec, label


def generate_dataset(
    n_normal: int = 150, n_anomaly: int = 150, seed: int = 99,
) -> tuple[list[dict], list[bool], list[str]]:
    rng = random.Random(seed)
    records, labels, descriptions = [], [], []
    for _ in range(n_normal):
        records.append(generate_normal(rng))
        labels.append(False)
        descriptions.append("normal")
    for _ in range(n_anomaly):
        rec, desc = generate_anomaly(rng)
        records.append(rec)
        labels.append(True)
        descriptions.append(desc)
    combined = list(zip(records, labels, descriptions))
    rng.shuffle(combined)
    records, labels, descriptions = zip(*combined)
    return list(records), list(labels), list(descriptions)


# ---------------------------------------------------------------------------
# Metrics
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
    return {"tp": tp, "tn": tn, "fp": fp, "fn": fn,
            "precision": precision, "recall": recall, "f1": f1, "accuracy": accuracy}


def print_matrix(name: str, m: dict, n_total: int, n_anomaly: int) -> None:
    n_normal = n_total - n_anomaly
    print(f"\n{'=' * 60}")
    print(f"  {name}")
    print(f"{'=' * 60}")
    print(f"  Dataset: {n_total} records ({n_anomaly} anomalous, {n_normal} normal)\n")
    print(f"                    Predicted")
    print(f"                    Normal  Anomaly")
    print(f"  Actual Normal     {m['tn']:>5}    {m['fp']:>5}")
    print(f"  Actual Anomaly    {m['fn']:>5}    {m['tp']:>5}\n")
    print(f"  Accuracy:   {m['accuracy']:.3f}")
    print(f"  Precision:  {m['precision']:.3f}")
    print(f"  Recall:     {m['recall']:.3f}")
    print(f"  F1 Score:   {m['f1']:.3f}")
    print(f"{'=' * 60}")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def dataset() -> tuple[list[dict], list[bool], list[str]]:
    return generate_dataset(n_normal=150, n_anomaly=150, seed=99)


@pytest.fixture(scope="module")
def ge_detector() -> GreatExpectationsDetector:
    return GreatExpectationsDetector(SUITE_PATH, preprocessor=add_derived_columns)


@pytest.fixture(scope="module")
def svm_detector() -> SVMDetector:
    return SVMDetector(
        training_data_path=TRAINING_DATA_PATH, model_path=MODEL_PATH,
        feature_columns=SVM_FEATURES, nu=0.1, gamma=0.001,
    )


@pytest.fixture(scope="module")
def gemma_detector() -> LLMDetector | None:
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
def mistral_detector() -> LLMDetector | None:
    if not os.environ.get("SADEED_LLM"):
        return None
    try:
        import httpx
        resp = httpx.get("http://localhost:11434/api/tags", timeout=3.0)
        if resp.is_success:
            return LLMDetector(base_url="http://localhost:11434", model="mistral:latest")
    except Exception:
        pass
    return None


@pytest.fixture(scope="module")
def ensemble_detector() -> EnsembleLLMDetector | None:
    if not os.environ.get("SADEED_LLM"):
        return None
    try:
        import httpx
        resp = httpx.get("http://localhost:11434/api/tags", timeout=3.0)
        if resp.is_success:
            return EnsembleLLMDetector(
                base_url="http://localhost:11434",
                primary_model="gemma2:9b",
                secondary_model="mistral:latest",
            )
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_ensemble_vs_single(
    dataset: tuple[list[dict], list[bool], list[str]],
    ge_detector: GreatExpectationsDetector,
    svm_detector: SVMDetector,
    gemma_detector: LLMDetector | None,
    mistral_detector: LLMDetector | None,
    ensemble_detector: EnsembleLLMDetector | None,
) -> None:
    """Compare single-model vs ensemble on 300 records."""

    if gemma_detector is None or mistral_detector is None or ensemble_detector is None:
        pytest.skip("LLM not available (set SADEED_LLM=1)")

    records, y_true, descriptions = dataset
    n_total = len(records)
    n_anomaly = sum(y_true)
    n_normal = n_total - n_anomaly

    print(f"\n\n  Dataset: {n_total} records ({n_anomaly} anomalous, {n_normal} normal)\n")

    # --- GE + SVM ---
    ge_results: list[StrategyResult] = []
    for r in tqdm(records, desc="GE", unit="rec"):
        ge_results.append(await ge_detector.detect(r))

    svm_results: list[StrategyResult] = []
    for r in tqdm(records, desc="SVM", unit="rec"):
        svm_results.append(await svm_detector.detect(r))

    y_ge = [r.is_anomaly for r in ge_results]
    svm_scores = [r.score for r in svm_results]

    # --- Gemma alone ---
    gemma_results: list[StrategyResult] = []
    for i, r in enumerate(tqdm(records, desc="Gemma2:9b", unit="rec")):
        gemma_results.append(await gemma_detector.detect(
            r, svm_score=svm_results[i].score, ge_score=ge_results[i].score,
        ))
    y_gemma = [r.is_anomaly for r in gemma_results]

    # --- Mistral alone ---
    mistral_results: list[StrategyResult] = []
    for i, r in enumerate(tqdm(records, desc="Mistral", unit="rec")):
        mistral_results.append(await mistral_detector.detect(
            r, svm_score=svm_results[i].score, ge_score=ge_results[i].score,
        ))
    y_mistral = [r.is_anomaly for r in mistral_results]

    # --- Ensemble (both must agree) ---
    ensemble_results: list[StrategyResult] = []
    for i, r in enumerate(tqdm(records, desc="Ensemble", unit="rec")):
        ensemble_results.append(await ensemble_detector.detect(
            r, svm_score=svm_results[i].score, ge_score=ge_results[i].score,
        ))
    y_ensemble = [r.is_anomaly for r in ensemble_results]

    # --- Build pipeline verdicts with SVM threshold 0.2 ---
    def pipeline(y_llm: list[bool]) -> list[bool]:
        return [
            y_llm[i] if (y_ge[i] or svm_scores[i] >= 0.2) else False
            for i in range(n_total)
        ]

    y_pipe_gemma = pipeline(y_gemma)
    y_pipe_mistral = pipeline(y_mistral)
    y_pipe_ensemble = pipeline(y_ensemble)

    # --- Also compute LLM-alone metrics ---
    print(f"\n\n{'#' * 80}")
    print(f"  LLM-ALONE COMPARISON (no SVM gate)")
    print(f"{'#' * 80}")

    m_gemma_alone = compute_metrics(y_true, y_gemma)
    print_matrix("Gemma2:9b Alone", m_gemma_alone, n_total, n_anomaly)

    m_mistral_alone = compute_metrics(y_true, y_mistral)
    print_matrix("Mistral Alone", m_mistral_alone, n_total, n_anomaly)

    m_ensemble_alone = compute_metrics(y_true, y_ensemble)
    print_matrix("Ensemble Alone (both agree)", m_ensemble_alone, n_total, n_anomaly)

    # --- Full pipeline comparison ---
    print(f"\n\n{'#' * 80}")
    print(f"  FULL PIPELINE: (GE OR SVM>=0.2) AND LLM")
    print(f"{'#' * 80}")

    m_pipe_gemma = compute_metrics(y_true, y_pipe_gemma)
    print_matrix("Pipeline + Gemma2:9b", m_pipe_gemma, n_total, n_anomaly)

    m_pipe_mistral = compute_metrics(y_true, y_pipe_mistral)
    print_matrix("Pipeline + Mistral", m_pipe_mistral, n_total, n_anomaly)

    m_pipe_ensemble = compute_metrics(y_true, y_pipe_ensemble)
    print_matrix("Pipeline + Ensemble", m_pipe_ensemble, n_total, n_anomaly)

    # --- Summary table ---
    print(f"\n{'=' * 90}")
    print(f"  SUMMARY")
    print(f"{'=' * 90}")
    print(f"  {'Configuration':<35s}  {'Acc':>6s}  {'Prec':>6s}  {'Rec':>6s}"
          f"  {'F1':>6s}  {'TP':>4s}  {'FP':>4s}  {'FN':>4s}  {'TN':>4s}")
    print(f"  {'-' * 35}  {'-' * 6}  {'-' * 6}  {'-' * 6}"
          f"  {'-' * 6}  {'-' * 4}  {'-' * 4}  {'-' * 4}  {'-' * 4}")
    rows = [
        ("Gemma2:9b alone", m_gemma_alone),
        ("Mistral alone", m_mistral_alone),
        ("Ensemble alone (AND)", m_ensemble_alone),
        ("Pipeline + Gemma2:9b", m_pipe_gemma),
        ("Pipeline + Mistral", m_pipe_mistral),
        ("Pipeline + Ensemble", m_pipe_ensemble),
    ]
    for name, m in rows:
        print(f"  {name:<35s}  {m['accuracy']:>6.3f}  {m['precision']:>6.3f}  "
              f"{m['recall']:>6.3f}  {m['f1']:>6.3f}  {m['tp']:>4d}  "
              f"{m['fp']:>4d}  {m['fn']:>4d}  {m['tn']:>4d}")
    print(f"{'=' * 90}")

    # --- Disagreements: where ensemble helps ---
    print(f"\n  MODEL AGREEMENT ANALYSIS:")
    both_flag = sum(g and m for g, m in zip(y_gemma, y_mistral))
    only_gemma = sum(g and not m for g, m in zip(y_gemma, y_mistral))
    only_mistral = sum(not g and m for g, m in zip(y_gemma, y_mistral))
    neither = sum(not g and not m for g, m in zip(y_gemma, y_mistral))
    print(f"    Both flag:     {both_flag}")
    print(f"    Only Gemma:    {only_gemma}")
    print(f"    Only Mistral:  {only_mistral}")
    print(f"    Neither flags: {neither}")

    # FPs that ensemble eliminates
    gemma_fps = [(i, descriptions[i]) for i in range(n_total)
                 if not y_true[i] and y_pipe_gemma[i]]
    ensemble_fps = [(i, descriptions[i]) for i in range(n_total)
                    if not y_true[i] and y_pipe_ensemble[i]]
    eliminated_fps = [(i, d) for i, d in gemma_fps if not y_pipe_ensemble[i]]

    print(f"\n  FALSE POSITIVE REDUCTION:")
    print(f"    Gemma pipeline FPs:    {len(gemma_fps)}")
    print(f"    Ensemble pipeline FPs: {len(ensemble_fps)}")
    print(f"    Eliminated by ensemble: {len(eliminated_fps)}")

    if eliminated_fps:
        print(f"\n  FPs eliminated by ensemble:")
        for idx, desc in eliminated_fps[:10]:
            g_s = gemma_results[idx].score or 0
            m_s = mistral_results[idx].score or 0
            print(f"    Gemma={g_s:.2f} Mistral={m_s:.2f}  {desc}")

    # TPs lost by ensemble
    lost_tps = [(i, descriptions[i]) for i in range(n_total)
                if y_true[i] and y_pipe_gemma[i] and not y_pipe_ensemble[i]]
    if lost_tps:
        print(f"\n  TRUE POSITIVES LOST by ensemble: {len(lost_tps)}")
        for idx, desc in lost_tps[:10]:
            g_s = gemma_results[idx].score or 0
            m_s = mistral_results[idx].score or 0
            g_flag = gemma_results[idx].is_anomaly
            m_flag = mistral_results[idx].is_anomaly
            print(f"    Gemma={'Y' if g_flag else 'N'}({g_s:.2f}) "
                  f"Mistral={'Y' if m_flag else 'N'}({m_s:.2f})  {desc}")
        if len(lost_tps) > 10:
            print(f"    ... and {len(lost_tps) - 10} more")

    # --- Save results ---
    result_rows = []
    for i in range(n_total):
        row = dict(records[i])
        row["_ground_truth"] = "anomaly" if y_true[i] else "normal"
        row["_description"] = descriptions[i]
        row["svm_score"] = svm_scores[i]
        row["gemma_anomaly"] = y_gemma[i]
        row["gemma_score"] = gemma_results[i].score
        row["mistral_anomaly"] = y_mistral[i]
        row["mistral_score"] = mistral_results[i].score
        row["ensemble_anomaly"] = y_ensemble[i]
        row["ensemble_score"] = ensemble_results[i].score
        row["pipe_gemma"] = y_pipe_gemma[i]
        row["pipe_ensemble"] = y_pipe_ensemble[i]
        result_rows.append(row)

    results_path = "data/test_ensemble_results.xlsx"
    pd.DataFrame(result_rows).to_excel(results_path, index=False)
    print(f"\n  Full results saved to {results_path}")
