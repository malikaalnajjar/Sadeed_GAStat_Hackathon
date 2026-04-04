"""
Evaluate detector performance on the LFS training dataset.

Uses Great Expectations business rules as ground-truth labels, then
evaluates the SVM detector and the combined pipeline against those labels.

Usage:
    python scripts/evaluate.py
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.detection.great_expectations_strategy import GreatExpectationsDetector
from backend.detection.lfs_preprocessing import add_derived_columns
from backend.detection.llm_strategy import LLMDetector
from backend.detection.svm_strategy import SVMDetector


SUITE_PATH = "expectations/suite.json"
TRAINING_DATA_PATH = "data/normal_samples.npy"
MODEL_PATH = "models/svm.joblib"
DATASET_PATH = "data/LFS_Training_Dataset.xlsx"

# The 5 features the SVM uses (sorted alphabetically)
SVM_FEATURES = ["act_1_total", "age", "cut_5_total", "edu_ordinal", "q_602_val"]

# Columns we need from the Excel file
REQUIRED_COLS = [
    "age", "gender", "family_relation", "marage_status", "nationality",
    "q_301", "q_602_val", "cut_5_total", "act_1_total",
]


# Records that pass all 21 GE rules but are statistically anomalous.
# The SVM should catch these — GE cannot.
SYNTHETIC_ANOMALIES: list[dict] = [
    # 1. PhD holder earning minimum wage (500 SAR)
    {"age": 55, "gender": 1600001, "family_relation": 1700001,
     "marage_status": 10600002, "nationality": 1800001,
     "q_301": 10500010, "q_602_val": 500, "cut_5_total": 40, "act_1_total": 40,
     "_label": "PhD earning 500 SAR"},
    # 2. Works 1 hour/week with 49k salary
    {"age": 40, "gender": 1600001, "family_relation": 1700001,
     "marage_status": 10600002, "nationality": 1800001,
     "q_301": 10500025, "q_602_val": 49000, "cut_5_total": 1, "act_1_total": 1,
     "_label": "1 hr/wk at 49k salary"},
    # 3. Works 84 hours usual, 0 actual
    {"age": 35, "gender": 1600001, "family_relation": 1700001,
     "marage_status": 10600002, "nationality": 1800001,
     "q_301": 10500025, "q_602_val": 15000, "cut_5_total": 84, "act_1_total": 0,
     "_label": "84 usual hrs, 0 actual"},
    # 4. 15-year-old head of household, no education, max allowed salary
    {"age": 15, "gender": 1600001, "family_relation": 1700001,
     "marage_status": 10600001, "nationality": 1800001,
     "q_301": 10500031, "q_602_val": 4999, "cut_5_total": 80, "act_1_total": 80,
     "_label": "15yo head, no edu, 5k salary, 80hrs"},
    # 5. Master's degree holder, 23 years old, earning 500 SAR
    {"age": 23, "gender": 1600002, "family_relation": 1700022,
     "marage_status": 10600001, "nationality": 1800001,
     "q_301": 10500009, "q_602_val": 500, "cut_5_total": 5, "act_1_total": 3,
     "_label": "23yo MSc, 500 SAR, 5 hrs"},
    # 6. 70-year-old working 84 hours at minimum wage
    {"age": 70, "gender": 1600001, "family_relation": 1700001,
     "marage_status": 10600004, "nationality": 1800001,
     "q_301": 10500031, "q_602_val": 500, "cut_5_total": 84, "act_1_total": 84,
     "_label": "70yo, 84hrs, 500 SAR"},
    # 7. Extreme salary with extreme hours
    {"age": 30, "gender": 1600001, "family_relation": 1700001,
     "marage_status": 10600002, "nationality": 1800001,
     "q_301": 10500025, "q_602_val": 49500, "cut_5_total": 84, "act_1_total": 84,
     "_label": "49.5k SAR at 84 hrs"},
    # 8. Secondary education, very young, high salary for tier
    {"age": 17, "gender": 1600001, "family_relation": 1700022,
     "marage_status": 10600001, "nationality": 1800001,
     "q_301": 10500019, "q_602_val": 14900, "cut_5_total": 60, "act_1_total": 60,
     "_label": "17yo secondary, 14.9k salary"},
    # 9. Usual hours 1, actual hours 84 (inverse)
    {"age": 45, "gender": 1600002, "family_relation": 1700021,
     "marage_status": 10600002, "nationality": 1800001,
     "q_301": 10500023, "q_602_val": 10000, "cut_5_total": 1, "act_1_total": 84,
     "_label": "1 usual hr, 84 actual"},
    # 10. All values at extreme boundaries simultaneously
    {"age": 74, "gender": 1600001, "family_relation": 1700001,
     "marage_status": 10600002, "nationality": 1800001,
     "q_301": 10500010, "q_602_val": 49999, "cut_5_total": 84, "act_1_total": 1,
     "_label": "All values at boundary extremes"},
]


def print_confusion_matrix(
    y_true: list[bool], y_pred: list[bool], name: str
) -> None:
    """Print a confusion matrix and metrics."""
    tp = sum(1 for t, p in zip(y_true, y_pred) if t and p)
    tn = sum(1 for t, p in zip(y_true, y_pred) if not t and not p)
    fp = sum(1 for t, p in zip(y_true, y_pred) if not t and p)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t and not p)

    total = len(y_true)
    n_anomaly = sum(y_true)
    n_normal = total - n_anomaly

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / total if total > 0 else 0.0

    print(f"\n{'=' * 50}")
    print(f"  {name}")
    print(f"{'=' * 50}")
    print(f"  Dataset: {total} records ({n_anomaly} anomalous, {n_normal} normal)")
    print()
    print(f"                  Predicted")
    print(f"                  Normal  Anomaly")
    print(f"  Actual Normal   {tn:>5}    {fp:>5}")
    print(f"  Actual Anomaly  {fn:>5}    {tp:>5}")
    print()
    print(f"  Accuracy:   {accuracy:.3f}")
    print(f"  Precision:  {precision:.3f}")
    print(f"  Recall:     {recall:.3f}")
    print(f"  F1 Score:   {f1:.3f}")
    print(f"{'=' * 50}")


async def main() -> None:
    # --- Load dataset ---
    if not Path(DATASET_PATH).exists():
        print(f"ERROR: Dataset not found at {DATASET_PATH}")
        sys.exit(1)

    df = pd.read_excel(DATASET_PATH)
    print(f"Loaded {len(df)} records from {DATASET_PATH}")

    # Filter to rows that have the required columns
    available = [c for c in REQUIRED_COLS if c in df.columns]
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        print(f"WARNING: Missing columns: {missing}")

    # Build record dicts
    records: list[dict] = []
    for _, row in df.iterrows():
        record = {}
        for col in available:
            val = row.get(col)
            if pd.notna(val):
                record[col] = val
        # Skip rows missing critical SVM features
        if all(k in record for k in SVM_FEATURES):
            records.append(record)

    print(f"Using {len(records)} records with complete SVM features")

    # --- Initialize detectors ---
    ge_detector = None
    if Path(SUITE_PATH).exists():
        ge_detector = GreatExpectationsDetector(
            SUITE_PATH, preprocessor=add_derived_columns
        )
        print(f"GE detector loaded ({SUITE_PATH})")
    else:
        print(f"WARNING: GE suite not found at {SUITE_PATH}")

    svm_detector = None
    if Path(TRAINING_DATA_PATH).exists():
        svm_detector = SVMDetector(
            training_data_path=TRAINING_DATA_PATH,
            model_path=MODEL_PATH,
            feature_columns=SVM_FEATURES,
            nu=0.1,
            gamma=0.001,
        )
        print(f"SVM detector loaded ({MODEL_PATH})")
    else:
        print(f"WARNING: SVM training data not found at {TRAINING_DATA_PATH}")

    if ge_detector is None:
        print("ERROR: Cannot evaluate without GE detector (used as ground truth)")
        sys.exit(1)

    # --- Run GE on all records (ground truth) ---
    ge_results = []
    for record in records:
        result = await ge_detector.detect(record)
        ge_results.append(result)

    y_true = [r.is_anomaly for r in ge_results]
    ge_scores = [r.score for r in ge_results]

    n_anomaly = sum(y_true)
    n_normal = len(y_true) - n_anomaly
    print(f"\nGround truth (GE rules): {n_anomaly} anomalous, {n_normal} normal")

    # --- GE self-evaluation (sanity check — should be 100%) ---
    print_confusion_matrix(y_true, y_true, "GE vs GE (sanity check)")

    # --- SVM evaluation ---
    if svm_detector is not None:
        svm_results = []
        for record in records:
            result = await svm_detector.detect(record)
            svm_results.append(result)

        y_pred_svm = [r.is_anomaly for r in svm_results]
        svm_scores = [r.score for r in svm_results]

        print_confusion_matrix(y_true, y_pred_svm, "SVM vs GE ground truth")

        # --- Combined pipeline: (GE OR SVM) ---
        y_pred_combined = [
            ge_r.is_anomaly or svm_r.is_anomaly
            for ge_r, svm_r in zip(ge_results, svm_results)
        ]
        print_confusion_matrix(y_true, y_pred_combined, "GE OR SVM (preliminary verdict)")

        # --- Score distribution ---
        print(f"\n{'=' * 50}")
        print(f"  Score Distribution")
        print(f"{'=' * 50}")

        anomaly_svm_scores = [s for s, t in zip(svm_scores, y_true) if t]
        normal_svm_scores = [s for s, t in zip(svm_scores, y_true) if not t]

        if anomaly_svm_scores:
            print(f"  SVM scores (GE-anomalous records):")
            print(f"    min={min(anomaly_svm_scores):.4f}  "
                  f"max={max(anomaly_svm_scores):.4f}  "
                  f"mean={sum(anomaly_svm_scores)/len(anomaly_svm_scores):.4f}")
        if normal_svm_scores:
            print(f"  SVM scores (GE-normal records):")
            print(f"    min={min(normal_svm_scores):.4f}  "
                  f"max={max(normal_svm_scores):.4f}  "
                  f"mean={sum(normal_svm_scores)/len(normal_svm_scores):.4f}")

        # --- Per-record details for mismatches ---
        mismatches = [
            (i, ge_r, svm_r)
            for i, (ge_r, svm_r) in enumerate(zip(ge_results, svm_results))
            if ge_r.is_anomaly != svm_r.is_anomaly
        ]
        if mismatches:
            print(f"\n{'=' * 50}")
            print(f"  GE/SVM Disagreements ({len(mismatches)} records)")
            print(f"{'=' * 50}")
            for i, ge_r, svm_r in mismatches[:10]:
                ge_label = "ANOMALY" if ge_r.is_anomaly else "normal"
                svm_label = "ANOMALY" if svm_r.is_anomaly else "normal"
                print(f"  Record {i}: GE={ge_label} (score={ge_r.score:.3f})  "
                      f"SVM={svm_label} (score={svm_r.score:.4f})")
            if len(mismatches) > 10:
                print(f"  ... and {len(mismatches) - 10} more")

    # --- Synthetic anomalies (pass GE, should fail SVM) ---
    await evaluate_synthetic_anomalies(ge_detector, svm_detector)


async def evaluate_synthetic_anomalies(
    ge_detector: GreatExpectationsDetector,
    svm_detector: SVMDetector | None,
) -> None:
    """Evaluate GE-passing anomalies — the SVM and LLM's unique value."""
    print(f"\n{'=' * 80}")
    print(f"  Synthetic Anomalies (pass GE, should fail SVM/LLM)")
    print(f"{'=' * 80}")

    # LLM evaluation is optional — pass --llm flag to enable
    llm_detector: LLMDetector | None = None
    import sys as _sys
    if "--llm" in _sys.argv:
        try:
            import httpx
            async with httpx.AsyncClient() as client:
                resp = await client.get("http://localhost:11434/api/tags", timeout=3.0)
                if resp.is_success:
                    llm_detector = LLMDetector(
                        base_url="http://localhost:11434", model="mistral:latest"
                    )
                    print("  LLM detector: online")
        except Exception:
            print("  LLM detector: offline (skipping)")
    else:
        print("  LLM detector: skipped (pass --llm to enable)")

    ge_caught = 0
    svm_caught = 0
    llm_caught = 0

    # Deep copy so we can re-run
    import copy
    anomalies = copy.deepcopy(SYNTHETIC_ANOMALIES)

    for record in anomalies:
        label = record.pop("_label")
        ge_r = await ge_detector.detect(record)
        svm_r = await svm_detector.detect(record) if svm_detector else None
        llm_r = await llm_detector.detect(record) if llm_detector else None

        ge_flag = "CAUGHT" if ge_r.is_anomaly else "missed"
        svm_flag = "CAUGHT" if (svm_r and svm_r.is_anomaly) else "missed"
        llm_flag = "CAUGHT" if (llm_r and llm_r.is_anomaly) else "missed"
        svm_score = f"{svm_r.score:.4f}" if svm_r else "n/a"
        llm_score = f"{llm_r.score:.4f}" if llm_r else "n/a"

        if ge_r.is_anomaly:
            ge_caught += 1
        if svm_r and svm_r.is_anomaly:
            svm_caught += 1
        if llm_r and llm_r.is_anomaly:
            llm_caught += 1

        print(f"  {label:40s}  GE={ge_flag:6s}  SVM={svm_flag:6s}({svm_score})  LLM={llm_flag:6s}({llm_score})")

    total = len(anomalies)
    print(f"\n  GE caught:  {ge_caught}/{total}")
    print(f"  SVM caught: {svm_caught}/{total}")
    if llm_detector:
        print(f"  LLM caught: {llm_caught}/{total}")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    asyncio.run(main())
