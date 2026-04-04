"""
Run the 1000-record validation set through the full pipeline and verify behavior.

Usage:
    SADEED_LLM=1 python scripts/validate_1000.py
"""

from __future__ import annotations

import asyncio
import csv
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tqdm import tqdm

from backend.detection.great_expectations_strategy import GreatExpectationsDetector
from backend.detection.lfs_preprocessing import add_derived_columns
from backend.detection.llm_strategy import LLMDetector
from backend.detection.svm_strategy import SVMDetector
from backend.models.schemas import StrategyResult

SUITE_PATH = "expectations/suite.json"
TRAINING_DATA_PATH = "data/normal_samples.npy"
MODEL_PATH = "models/svm.joblib"
SVM_FEATURES = ["act_1_total", "age", "cut_5_total", "edu_ordinal", "q_602_val"]
CSV_PATH = "data/validation_1000.csv"

RECORD_FIELDS = [
    "age", "gender", "family_relation", "marage_status", "nationality",
    "q_301", "q_602_val", "cut_5_total", "act_1_total",
]


def load_csv() -> list[dict]:
    with open(CSV_PATH, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = []
        for row in reader:
            rec = {}
            for field in RECORD_FIELDS:
                val = row.get(field)
                if val is not None and val != "":
                    try:
                        rec[field] = int(val)
                    except ValueError:
                        rec[field] = val
            rec["_expected_verdict"] = row["_expected_verdict"]
            rec["_category"] = row["_category"]
            rec["_description"] = row["_description"]
            rows.append(rec)
    return rows


def compute_metrics(y_true: list[bool], y_pred: list[bool]) -> dict:
    tp = sum(t and p for t, p in zip(y_true, y_pred))
    tn = sum(not t and not p for t, p in zip(y_true, y_pred))
    fp = sum(not t and p for t, p in zip(y_true, y_pred))
    fn = sum(t and not p for t, p in zip(y_true, y_pred))
    total = len(y_true)
    prec = tp / (tp + fp) if (tp + fp) else 0
    rec = tp / (tp + fn) if (tp + fn) else 0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0
    acc = (tp + tn) / total if total else 0
    return {"tp": tp, "tn": tn, "fp": fp, "fn": fn,
            "precision": prec, "recall": rec, "f1": f1, "accuracy": acc}


async def main() -> None:
    rows = load_csv()
    print(f"Loaded {len(rows)} records from {CSV_PATH}\n")

    # Count categories
    cats = {}
    for r in rows:
        cats[r["_category"]] = cats.get(r["_category"], 0) + 1
    for cat, count in sorted(cats.items()):
        print(f"  {cat}: {count}")
    print()

    # Init detectors
    ge = GreatExpectationsDetector(SUITE_PATH, preprocessor=add_derived_columns)
    svm = SVMDetector(
        training_data_path=TRAINING_DATA_PATH, model_path=MODEL_PATH,
        feature_columns=SVM_FEATURES, nu=0.1, gamma=0.001,
    )
    llm = None
    if os.environ.get("SADEED_LLM"):
        try:
            import httpx
            resp = httpx.get("http://localhost:11434/api/tags", timeout=3.0)
            if resp.is_success:
                llm = LLMDetector(base_url="http://localhost:11434", model="mistral:latest")
                print("  LLM: Mistral loaded\n")
        except Exception:
            pass
    if llm is None:
        print("  LLM: not available (set SADEED_LLM=1)\n")

    # Run pipeline
    results = []
    for r in tqdm(rows, desc="Pipeline", unit="rec"):
        data = {k: r[k] for k in RECORD_FIELDS if k in r}

        ge_res = await ge.detect(data)
        svm_res = await svm.detect(data)

        # LLM called only when GE or SVM flags
        llm_res = None
        ge_flag = ge_res.is_anomaly
        svm_flag = svm_res.is_anomaly

        if llm and (ge_flag or svm_flag):
            llm_res = await llm.detect(
                data, svm_score=svm_res.score, ge_score=ge_res.score,
            )

        # Final verdict: (GE OR SVM) AND LLM
        # GE is ground truth (cannot be overridden)
        if llm_res is not None:
            if ge_flag:
                verdict = True
            else:
                verdict = svm_flag and llm_res.is_anomaly
        else:
            verdict = ge_flag or svm_flag

        results.append({
            "category": r["_category"],
            "expected": r["_expected_verdict"],
            "description": r["_description"],
            "ge_flag": ge_flag,
            "ge_score": ge_res.score,
            "svm_flag": svm_flag,
            "svm_score": svm_res.score,
            "llm_flag": llm_res.is_anomaly if llm_res else None,
            "llm_score": llm_res.score if llm_res else None,
            "llm_explanation": llm_res.explanation if llm_res else None,
            "verdict": verdict,
        })

    # Analysis by category
    print(f"\n{'=' * 80}")
    print(f"  RESULTS BY CATEGORY")
    print(f"{'=' * 80}\n")

    for cat in ["normal", "ge_anomaly", "svm_anomaly", "svm_false_positive"]:
        cat_results = [r for r in results if r["category"] == cat]
        n = len(cat_results)
        if n == 0:
            continue

        if cat == "normal":
            expected_anomaly = False
        elif cat == "svm_false_positive":
            expected_anomaly = False  # They're actually normal
        else:
            expected_anomaly = True

        correct = sum(1 for r in cat_results if r["verdict"] == expected_anomaly)
        ge_caught = sum(1 for r in cat_results if r["ge_flag"])
        svm_caught = sum(1 for r in cat_results if r["svm_flag"])
        llm_called = sum(1 for r in cat_results if r["llm_flag"] is not None)
        llm_confirmed = sum(1 for r in cat_results if r["llm_flag"] is True)
        llm_denied = sum(1 for r in cat_results if r["llm_flag"] is False)

        print(f"  {cat} ({n} records):")
        print(f"    Correct verdicts: {correct}/{n} ({correct/n*100:.1f}%)")
        print(f"    GE flagged:       {ge_caught}/{n}")
        print(f"    SVM flagged:      {svm_caught}/{n}")
        print(f"    LLM called:       {llm_called}/{n}")
        print(f"    LLM confirmed:    {llm_confirmed}/{n}")
        print(f"    LLM denied (FP filter): {llm_denied}/{n}")

        # Show failures
        failures = [r for r in cat_results if r["verdict"] != expected_anomaly]
        if failures:
            print(f"    FAILURES ({len(failures)}):")
            for r in failures[:10]:
                print(f"      [{r['description']}] GE={'Y' if r['ge_flag'] else 'N'} "
                      f"SVM={r['svm_score']:.2f} "
                      f"LLM={'Y' if r['llm_flag'] else 'N' if r['llm_flag'] is not None else '-'} "
                      f"verdict={'anomaly' if r['verdict'] else 'normal'}")
            if len(failures) > 10:
                print(f"      ... and {len(failures) - 10} more")
        print()

    # Overall metrics
    y_true = [r["expected"] == "anomaly" for r in results]
    y_pred = [r["verdict"] for r in results]
    m = compute_metrics(y_true, y_pred)

    print(f"{'=' * 80}")
    print(f"  OVERALL PIPELINE METRICS")
    print(f"{'=' * 80}")
    print(f"  Accuracy:   {m['accuracy']:.3f}")
    print(f"  Precision:  {m['precision']:.3f}")
    print(f"  Recall:     {m['recall']:.3f}")
    print(f"  F1 Score:   {m['f1']:.3f}")
    print(f"  TP: {m['tp']}  FP: {m['fp']}  FN: {m['fn']}  TN: {m['tn']}")
    print(f"{'=' * 80}")

    # Write results CSV
    out_path = "data/validation_1000_results.csv"
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)
    print(f"\n  Results saved to {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
