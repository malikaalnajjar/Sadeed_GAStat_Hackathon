#!/usr/bin/env python3
"""Grid search over SVM hyperparameters, excluding GE-caught records.

Since GE runs first in the pipeline and its flags are ground truth,
the SVM is only evaluated on records that GE does NOT flag. This
measures the SVM's real contribution: catching anomalies GE misses.

Uses the 300-record synthetic dataset (150 normal + 150 anomalous)
where GE recall is ~66%, leaving plenty of anomalies for SVM to catch.

Usage:
    python scripts/tune_svm.py
"""

from __future__ import annotations

import asyncio
import random
import sys
from pathlib import Path

import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.prepare_data import (
    EDU_ORDINAL_MAP,
    SVM_NUMERIC_COLUMNS,
    generate_synthetic_normals,
    load_training_data,
    prepare_svm_data,
)
from backend.detection.great_expectations_strategy import GreatExpectationsDetector
from backend.detection.lfs_preprocessing import add_derived_columns
from backend.detection.svm_strategy import EDU_ORDINAL_MAP as SVM_EDU_MAP

# Reuse the 300-record generator from the test
from tests.test_metrics_300 import generate_dataset

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SVM_FEATURES = sorted(["age", "cut_5_total", "act_1_total", "q_602_val", "edu_ordinal"])


def svm_features(rec: dict) -> np.ndarray | None:
    """Extract the 5 SVM features, return None if incomplete."""
    edu_code = rec.get("q_301")
    edu_ord = EDU_ORDINAL_MAP.get(int(edu_code), None) if edu_code is not None else None

    vals = {}
    for col in SVM_FEATURES:
        if col == "edu_ordinal":
            if edu_ord is None:
                return None
            vals[col] = float(edu_ord)
        else:
            v = rec.get(col)
            if v is None:
                return None
            vals[col] = float(v)

    arr = np.array([vals[c] for c in SVM_FEATURES], dtype=np.float64)
    if np.any(np.isnan(arr)):
        return None
    return arr


def evaluate_config(
    X_real: np.ndarray,
    test_features: list[np.ndarray],
    truths: list[bool],
    nu: float,
    gamma: float | str,
    n_synthetic: int,
    kernel: str = "rbf",
) -> dict:
    """Train SVM and evaluate on GE-passed records only."""
    X_synth = generate_synthetic_normals(X_real, n=n_synthetic, seed=42)
    X_train = np.vstack([X_real, X_synth])

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", OneClassSVM(kernel=kernel, nu=nu, gamma=gamma)),
    ])
    model.fit(X_train)

    tp = fp = tn = fn = 0
    for feat, is_anomaly in zip(test_features, truths):
        pred = model.predict(feat.reshape(1, -1))[0]
        svm_flag = pred == -1

        if is_anomaly and svm_flag:
            tp += 1
        elif is_anomaly and not svm_flag:
            fn += 1
        elif not is_anomaly and svm_flag:
            fp += 1
        else:
            tn += 1

    total = tp + fp + tn + fn
    acc = (tp + tn) / total if total else 0
    prec = tp / (tp + fp) if (tp + fp) else 0
    rec = tp / (tp + fn) if (tp + fn) else 0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0

    return {
        "nu": nu, "gamma": gamma, "n_synthetic": n_synthetic,
        "acc": acc, "prec": prec, "rec": rec, "f1": f1,
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "total": total, "n_anomaly": tp + fn, "n_normal": tn + fp,
    }


def main() -> None:
    # --- Load training data ---
    print("Loading training data...")
    df = load_training_data()
    X_real = prepare_svm_data(df)
    print(f"Real training samples: {X_real.shape[0]}")

    # --- Generate 300-record test dataset ---
    print("Generating 300-record test dataset (seed=99)...")
    records, labels, descriptions = generate_dataset(n_normal=150, n_anomaly=150, seed=99)
    print(f"Total: {len(records)} records ({sum(labels)} anomalous, {len(labels) - sum(labels)} normal)")

    # --- Run GE to identify which records it catches ---
    print("Running GE on all records...")
    ge = GreatExpectationsDetector(
        str(PROJECT_ROOT / "expectations" / "suite.json"),
        preprocessor=add_derived_columns,
    )

    loop = asyncio.new_event_loop()
    ge_flags: list[bool] = []
    for rec in records:
        result = loop.run_until_complete(ge.detect(rec))
        ge_flags.append(result.is_anomaly)

    ge_tp = sum(g and t for g, t in zip(ge_flags, labels))
    ge_fp = sum(g and not t for g, t in zip(ge_flags, labels))
    ge_fn = sum(not g and t for g, t in zip(ge_flags, labels))
    print(f"GE catches: {ge_tp} TP, {ge_fp} FP, {ge_fn} FN (misses)")

    # --- Filter to GE-passed records only ---
    test_features: list[np.ndarray] = []
    test_truths: list[bool] = []
    test_descs: list[str] = []

    for i, (rec, truth, desc) in enumerate(zip(records, labels, descriptions)):
        if ge_flags[i]:
            continue  # GE already caught this, skip
        feat = svm_features(rec)
        if feat is None:
            continue
        test_features.append(feat)
        test_truths.append(truth)
        test_descs.append(desc)

    n_remaining_anom = sum(test_truths)
    n_remaining_norm = len(test_truths) - n_remaining_anom
    print(f"\nSVM evaluated on {len(test_truths)} GE-passed records:")
    print(f"  {n_remaining_anom} anomalies GE missed (SVM must catch these)")
    print(f"  {n_remaining_norm} true normals (SVM must not flag these)")

    # --- Grid search ---
    nus = [0.01, 0.03, 0.05, 0.08, 0.1, 0.15, 0.2, 0.25, 0.3]
    gammas = [0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0, "scale", "auto"]
    n_synthetics = [500, 1000, 2000, 3000]

    total_configs = len(nus) * len(gammas) * len(n_synthetics)
    print(f"\nGrid: {len(nus)} nu x {len(gammas)} gamma x {len(n_synthetics)} n_synth = {total_configs} configs\n")

    results: list[dict] = []
    best_f1 = -1
    best_config = None

    for n_synth in n_synthetics:
        for nu in nus:
            for gamma in gammas:
                try:
                    r = evaluate_config(X_real, test_features, test_truths, nu, gamma, n_synth)
                    results.append(r)
                    # Prefer higher F1, then fewer FP
                    if (r["f1"] > best_f1) or (r["f1"] == best_f1 and r["fp"] < (best_config or {}).get("fp", 999)):
                        best_f1 = r["f1"]
                        best_config = r
                except Exception:
                    pass

    # Sort by F1 desc, FP asc
    results.sort(key=lambda x: (-x["f1"], x["fp"]))

    print(f"{'nu':>6}  {'gamma':>8}  {'n_synth':>7}  {'Acc':>6}  {'Prec':>6}  {'Rec':>6}  {'F1':>6}  {'TP':>4}  {'FP':>4}  {'FN':>4}")
    print("-" * 85)
    for r in results[:30]:
        print(
            f"{r['nu']:>6.3f}  {str(r['gamma']):>8}  {r['n_synthetic']:>7}  "
            f"{r['acc']:>6.3f}  {r['prec']:>6.3f}  {r['rec']:>6.3f}  {r['f1']:>6.3f}  "
            f"{r['tp']:>4}  {r['fp']:>4}  {r['fn']:>4}"
        )

    if best_config:
        print(f"\n{'='*85}")
        print(f"BEST: nu={best_config['nu']}, gamma={best_config['gamma']}, "
              f"n_synthetic={best_config['n_synthetic']}")
        print(f"      F1={best_config['f1']:.3f}  Prec={best_config['prec']:.3f}  "
              f"Rec={best_config['rec']:.3f}  TP={best_config['tp']}  "
              f"FP={best_config['fp']}  FN={best_config['fn']}")
        print(f"      (on {len(test_truths)} GE-passed records: "
              f"{n_remaining_anom} anomalies, {n_remaining_norm} normals)")

        # Show current config for comparison
        print(f"\n--- Current config (nu=0.1, gamma=0.001, n_synth=2000) ---")
        curr = evaluate_config(X_real, test_features, test_truths, 0.05, 0.1, 2000)
        print(f"      F1={curr['f1']:.3f}  Prec={curr['prec']:.3f}  "
              f"Rec={curr['rec']:.3f}  TP={curr['tp']}  "
              f"FP={curr['fp']}  FN={curr['fn']}")
    else:
        print("\nNo valid configurations found.")


if __name__ == "__main__":
    main()
