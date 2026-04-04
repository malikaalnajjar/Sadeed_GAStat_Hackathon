#!/usr/bin/env python3
"""Prepare LFS training data for all three Sadeed detection layers.

Reads the hackathon Excel files and produces:
  1. expectations/suite.json  — Great Expectations suite from business rules
  2. data/normal_samples.npy  — cleaned numeric array for OC-SVM training
  3. Prints candidate FEW_SHOT_EXAMPLES for llm_strategy.py (manual paste)

Usage:
    python scripts/prepare_data.py            # from project root
    python scripts/prepare_data.py --print-few-shot  # also print few-shot JSON
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths (relative to project root)
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
TRAINING_XLSX = DATA_DIR / "LFS_Training_Dataset.xlsx"
RULES_XLSX = DATA_DIR / "LFS_Business_Rules.xlsx"
METADATA_XLSX = DATA_DIR / "MetaData_LFS_Training_Dataset.xlsx"
SUITE_PATH = PROJECT_ROOT / "expectations" / "suite.json"
NORMAL_SAMPLES_PATH = DATA_DIR / "normal_samples.npy"

# ---------------------------------------------------------------------------
# Column-code mappings (from the training dataset)
# ---------------------------------------------------------------------------

# Education levels (q_301)
EDU_NO_FORMAL = 10500031      # 0 - No formal education
EDU_PRIMARY_INCOMPLETE = 10500011  # 3 - Primary incomplete
EDU_PRIMARY = 10500003        # 4 - Primary education
EDU_INTERMEDIATE = 10500017   # 5 - Intermediate education
EDU_SECONDARY = 10500019      # 7 - Secondary (general)
EDU_SECONDARY_VOC = 10500020  # 8 - Secondary (vocational)
EDU_ASSOC_DIPLOMA = 10500021  # 9 - Associate diploma
EDU_DIPLOMA = 10500023        # 11 - Diploma
EDU_BACHELOR = 10500025       # 13 - Bachelor's (3-4 years)
EDU_BACHELOR_5 = 10500026     # 14 - Bachelor's (5 years)
EDU_MASTERS = 10500009        # 18 - Master's degree
EDU_PHD = 10500010            # 19 - PhD

SECONDARY_AND_ABOVE = {
    EDU_SECONDARY, EDU_SECONDARY_VOC, EDU_ASSOC_DIPLOMA, EDU_DIPLOMA,
    EDU_BACHELOR, EDU_BACHELOR_5, EDU_MASTERS, EDU_PHD,
}
DIPLOMA_AND_ABOVE = {
    EDU_DIPLOMA, EDU_ASSOC_DIPLOMA, EDU_BACHELOR, EDU_BACHELOR_5,
    EDU_MASTERS, EDU_PHD,
}
BACHELOR_AND_ABOVE = {EDU_BACHELOR, EDU_BACHELOR_5, EDU_MASTERS, EDU_PHD}
MASTERS_AND_ABOVE = {EDU_MASTERS, EDU_PHD}
PHD_CODES = {EDU_PHD}

# Family relation codes
FAM_HEAD = 1700001
FAM_WIFE = 1700021
FAM_SON = 1700022
FAM_DAUGHTER = 1700023
FAM_DOMESTIC = 1700010
FAM_BROTHER = 1700030

# Gender codes
GENDER_MALE = 1600001
GENDER_FEMALE = 1600002

# Nationality
NAT_SAUDI = 1800001

# Marriage status
MAR_NEVER = 10600001
MAR_MARRIED = 10600002
MAR_DIVORCED = 10600003
MAR_WIDOWED = 10600004
MAR_UNKNOWN = 10600012

# Job sector (q_534)
SECTOR_PUBLIC = 99400001
SECTOR_SEMI_PUBLIC = 99400002
SECTOR_PRIVATE = 99400003
SECTOR_DOMESTIC = 99400004

# Ordinal encoding for education level (q_301 → 0-11 scale)
# Higher number = higher education level
EDU_ORDINAL_MAP: dict[int, int] = {
    EDU_NO_FORMAL: 0,
    EDU_PRIMARY_INCOMPLETE: 1,
    EDU_PRIMARY: 2,
    EDU_INTERMEDIATE: 3,
    EDU_SECONDARY: 4,
    EDU_SECONDARY_VOC: 5,
    EDU_ASSOC_DIPLOMA: 6,
    EDU_DIPLOMA: 7,
    EDU_BACHELOR: 8,
    EDU_BACHELOR_5: 9,
    EDU_MASTERS: 10,
    EDU_PHD: 11,
}

# Numeric columns for SVM (sorted alphabetically — must match inference order)
SVM_NUMERIC_COLUMNS = sorted([
    "age", "cut_5_total", "act_1_total", "q_602_val", "edu_ordinal",
])


# ---------------------------------------------------------------------------
# Step 1: Generate expectations/suite.json
# ---------------------------------------------------------------------------

def generate_suite() -> dict:
    """Build a GE expectation suite from key LFS business rules."""
    expectations: list[dict] = []

    def _add(etype: str, kwargs: dict, rule: str, notes: str) -> None:
        expectations.append({
            "expectation_type": etype,
            "kwargs": kwargs,
            "meta": {"rule": rule, "notes": notes},
        })

    # Required fields
    for col in ("age", "gender", "family_relation"):
        _add("expect_column_values_to_not_be_null",
             {"column": col}, "required_field", f"{col} must always be present")

    # Valid ranges & sets
    _add("expect_column_values_to_be_between",
         {"column": "age", "min_value": 0, "max_value": 120},
         "range_check", "Age must be in a valid range")

    _add("expect_column_values_to_be_in_set",
         {"column": "gender", "value_set": [GENDER_MALE, GENDER_FEMALE]},
         "valid_values", "1600001=Male, 1600002=Female")

    _add("expect_column_values_to_be_in_set",
         {"column": "family_relation",
          "value_set": [1700001, 1700002, 1700003, 1700004, 1700005,
                        1700006, 1700007, 1700008, 1700009, 1700010,
                        1700011, 1700021, 1700022, 1700023, 1700030]},
         "valid_values", "Valid family relation codes")

    _add("expect_column_values_to_be_in_set",
         {"column": "marage_status",
          "value_set": [MAR_NEVER, MAR_MARRIED, MAR_DIVORCED,
                        MAR_WIDOWED, MAR_UNKNOWN]},
         "valid_values",
         "1=Never married, 2=Married, 3=Divorced, 4=Widowed, 98=Unknown")

    # Age vs education (rules 2011-2016): use derived columns
    age_edu_rules = [
        ("2011", 17, SECONDARY_AND_ABOVE, "_rule_2011_min_secondary_age",
         "Secondary education and above requires age >= 17"),
        ("2012", 19, DIPLOMA_AND_ABOVE, "_rule_2012_min_diploma_age",
         "Diploma and above requires age >= 19"),
        ("2013", 21, BACHELOR_AND_ABOVE, "_rule_2013_min_bachelor_age",
         "Bachelor's degree and above requires age >= 21"),
        ("2015", 23, MASTERS_AND_ABOVE, "_rule_2015_min_masters_age",
         "Master's degree and above requires age >= 23"),
        ("2016", 25, PHD_CODES, "_rule_2016_min_phd_age",
         "PhD requires age >= 25"),
    ]
    for rule_id, _min_age, _codes, derived_col, notes in age_edu_rules:
        _add("expect_column_pair_values_A_to_be_greater_than_B",
             {"column_A": "age", "column_B": derived_col, "or_equal": True},
             rule_id, notes)

    # Rule 2001: Head of household must be >= 15
    _add("expect_column_pair_values_A_to_be_greater_than_B",
         {"column_A": "age", "column_B": "_rule_2001_min_head_age",
          "or_equal": True},
         "2001", "Head of household must be >= 15 years old")

    # Rule 2031: Spouse must be married
    _add("expect_column_pair_values_A_to_be_greater_than_B",
         {"column_A": "_rule_2031_spouse_married", "column_B": "_rule_zero",
          "or_equal": True},
         "2031", "Spouse must have married status")

    # Rule 2075: Grandparent must be >= 30
    _add("expect_column_pair_values_A_to_be_greater_than_B",
         {"column_A": "_rule_2075_grandparent_age", "column_B": "_rule_zero",
          "or_equal": True},
         "2075", "Grandparent must be >= 30 years old")

    # Salary bounds (rules 3039 + 3068)
    _add("expect_column_values_to_be_between",
         {"column": "q_602_val", "min_value": 500, "max_value": 50000,
          "mostly": 0.85},
         "3039+3068",
         "Monthly salary typically between 500-50000 SAR")

    # Work hours bounds
    _add("expect_column_values_to_be_between",
         {"column": "cut_5_total", "min_value": 1, "max_value": 84,
          "mostly": 0.85},
         "3027+3028",
         "Usual weekly hours should be 1-84")

    _add("expect_column_values_to_be_between",
         {"column": "act_1_total", "min_value": 0, "max_value": 84,
          "mostly": 0.85},
         "2088+3030",
         "Actual weekly hours should not exceed 84")

    # Rule 2078: Public sector workers must be >= 17
    _add("expect_column_pair_values_A_to_be_greater_than_B",
         {"column_A": "_rule_2078_public_sector_age", "column_B": "_rule_zero",
          "or_equal": True},
         "2078", "Public sector workers must be >= 17")

    # Rule 2141: Domestic workers must be >= 15
    _add("expect_column_pair_values_A_to_be_greater_than_B",
         {"column_A": "_rule_2141_domestic_worker_age",
          "column_B": "_rule_zero", "or_equal": True},
         "2141", "Domestic worker must be >= 15")

    # Rule 4030: Spouse should be >= 18
    _add("expect_column_pair_values_A_to_be_greater_than_B",
         {"column_A": "_rule_4030_spouse_age", "column_B": "_rule_zero",
          "or_equal": True},
         "4030", "Spouse should be >= 18")

    # Rule 4069: Under 15 cannot be married
    _add("expect_column_pair_values_A_to_be_greater_than_B",
         {"column_A": "_rule_4069_child_not_married", "column_B": "_rule_zero",
          "or_equal": True},
         "4069", "Under 15 cannot have any marital status except never married")

    # Rule 3047: University degree or higher + salary < 3000
    _add("expect_column_pair_values_A_to_be_greater_than_B",
         {"column_A": "_rule_3047_uni_salary", "column_B": "_rule_zero",
          "or_equal": True},
         "3047", "University degree or higher: salary should be at least 3000 SAR")

    # Rule 3031: Public sector + usual hours < 35
    _add("expect_column_pair_values_A_to_be_greater_than_B",
         {"column_A": "_rule_3031_public_usual_hours", "column_B": "_rule_zero",
          "or_equal": True},
         "3031", "Public sector: usual working hours should be at least 35")

    # Rule 3032: Public sector + actual hours < 35
    _add("expect_column_pair_values_A_to_be_greater_than_B",
         {"column_A": "_rule_3032_public_actual_hours", "column_B": "_rule_zero",
          "or_equal": True},
         "3032", "Public sector: actual working hours should be at least 35")

    # Rule 3035: Private sector + usual hours < 40
    _add("expect_column_pair_values_A_to_be_greater_than_B",
         {"column_A": "_rule_3035_private_usual_hours", "column_B": "_rule_zero",
          "or_equal": True},
         "3035", "Private sector: usual working hours should be at least 40")

    # Rule 3036: Private sector + actual hours < 40
    _add("expect_column_pair_values_A_to_be_greater_than_B",
         {"column_A": "_rule_3036_private_actual_hours", "column_B": "_rule_zero",
          "or_equal": True},
         "3036", "Private sector: actual working hours should be at least 40")

    suite = {
        "expectation_suite_name": "lfs_business_rules",
        "expectations": expectations,
        "meta": {
            "great_expectations_version": "0.18.0",
            "description": (
                "LFS survey validation rules derived from LFS_Business_Rules.xlsx. "
                "Covers age-education consistency, age-family relation constraints, "
                "salary bounds, work hours limits, and marital status logic. "
                "Some rules require derived columns computed during pre-processing "
                "(prefixed with _rule_)."
            ),
        },
    }
    return suite


def add_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived helper columns needed by the expectation suite.

    These columns encode conditional business rules as numeric values so that
    Great Expectations pair-comparison expectations can be used.
    """
    df = df.copy()

    # Constant zero column for comparisons
    df["_rule_zero"] = 0

    # Rule 2001: Head of household must be >= 15
    df["_rule_2001_min_head_age"] = df["family_relation"].apply(
        lambda x: 15 if x == FAM_HEAD else 0
    )

    # Rules 2011-2016: Minimum age for education levels
    df["_rule_2011_min_secondary_age"] = df["q_301"].apply(
        lambda x: 17 if x in SECONDARY_AND_ABOVE else 0
    )
    df["_rule_2012_min_diploma_age"] = df["q_301"].apply(
        lambda x: 19 if x in DIPLOMA_AND_ABOVE else 0
    )
    df["_rule_2013_min_bachelor_age"] = df["q_301"].apply(
        lambda x: 21 if x in BACHELOR_AND_ABOVE else 0
    )
    df["_rule_2015_min_masters_age"] = df["q_301"].apply(
        lambda x: 23 if x in MASTERS_AND_ABOVE else 0
    )
    df["_rule_2016_min_phd_age"] = df["q_301"].apply(
        lambda x: 25 if x in PHD_CODES else 0
    )

    # Boolean-style rules: 1 = valid, -1 = violation
    # (compared against _rule_zero=0 with or_equal=True: -1 < 0 fails, 1 >= 0 passes)

    # Rule 2031: Spouse must be married
    df["_rule_2031_spouse_married"] = df.apply(
        lambda r: (
            -1 if r["family_relation"] == FAM_WIFE and r["marage_status"] != MAR_MARRIED
            else 1
        ),
        axis=1,
    )

    # Rule 2075: Grandparent must be >= 30
    grandparent_codes = {1700007, 1700008}
    df["_rule_2075_grandparent_age"] = df.apply(
        lambda r: (
            -1 if r["family_relation"] in grandparent_codes and r["age"] < 30
            else 1
        ),
        axis=1,
    )

    # Rule 2078: Public sector workers must be >= 17
    df["_rule_2078_public_sector_age"] = df.apply(
        lambda r: (
            -1 if r.get("q_534") == SECTOR_PUBLIC and r["age"] < 17
            else 1
        ),
        axis=1,
    )

    # Rule 2141: Domestic workers must be >= 15
    df["_rule_2141_domestic_worker_age"] = df.apply(
        lambda r: (
            -1 if r["family_relation"] == FAM_DOMESTIC and r["age"] < 15
            else 1
        ),
        axis=1,
    )

    # Rule 4030: Spouse should be >= 18
    df["_rule_4030_spouse_age"] = df.apply(
        lambda r: (
            -1 if r["family_relation"] == FAM_WIFE and r["age"] < 18
            else 1
        ),
        axis=1,
    )

    return df


# ---------------------------------------------------------------------------
# Step 2: Extract numeric columns for SVM training
# ---------------------------------------------------------------------------

def load_training_data() -> pd.DataFrame:
    """Load the LFS training dataset from Excel."""
    df = pd.read_excel(TRAINING_XLSX, dtype=str)
    return df


def prepare_svm_data(df: pd.DataFrame) -> np.ndarray:
    """Extract and clean numeric columns for OC-SVM training.

    Filters out rows with extreme outliers (salary > 100000, hours > 100)
    that represent data entry errors rather than normal patterns.
    Adds ordinal-encoded education level as a feature.
    """
    numeric_df = pd.DataFrame()

    # Extract raw numeric columns (excluding edu_ordinal which is derived)
    raw_cols = [c for c in SVM_NUMERIC_COLUMNS if c != "edu_ordinal"]
    for col in raw_cols:
        if col in df.columns:
            numeric_df[col] = pd.to_numeric(df[col], errors="coerce")

    # Ordinal-encode education from q_301
    if "q_301" in df.columns:
        numeric_df["edu_ordinal"] = pd.to_numeric(df["q_301"], errors="coerce").map(EDU_ORDINAL_MAP)
    else:
        numeric_df["edu_ordinal"] = np.nan

    # Drop rows where any required numeric column is NaN
    numeric_df = numeric_df.dropna()

    # Filter out extreme outliers to keep only "normal" samples
    mask = (
        (numeric_df["q_602_val"] >= 200)
        & (numeric_df["q_602_val"] <= 100000)
        & (numeric_df["cut_5_total"] > 0)
        & (numeric_df["cut_5_total"] <= 100)
        & (numeric_df["act_1_total"] >= 0)
        & (numeric_df["act_1_total"] <= 100)
    )
    clean_df = numeric_df[mask]

    # Reorder columns to match SVM_NUMERIC_COLUMNS (sorted alphabetically)
    clean_df = clean_df[SVM_NUMERIC_COLUMNS]

    return clean_df.values.astype(np.float64)


# ---------------------------------------------------------------------------
# Step 2b: Generate synthetic normal samples from domain constraints
# ---------------------------------------------------------------------------

# Education code → (min_age, max_salary)
_EDU_CONSTRAINTS: dict[int, tuple[int, int]] = {
    EDU_NO_FORMAL:          (15, 5000),
    EDU_PRIMARY_INCOMPLETE: (15, 5000),
    EDU_PRIMARY:            (15, 5000),
    EDU_INTERMEDIATE:       (15, 8000),
    EDU_SECONDARY:          (17, 15000),
    EDU_SECONDARY_VOC:      (17, 15000),
    EDU_ASSOC_DIPLOMA:      (19, 25000),
    EDU_DIPLOMA:            (19, 25000),
    EDU_BACHELOR:           (21, 50000),
    EDU_BACHELOR_5:         (21, 50000),
    EDU_MASTERS:            (23, 50000),
    EDU_PHD:                (25, 50000),
}

# Salary floors by education tier (from training data distribution)
_EDU_SALARY_FLOOR: dict[int, int] = {
    EDU_NO_FORMAL: 500, EDU_PRIMARY_INCOMPLETE: 500, EDU_PRIMARY: 500,
    EDU_INTERMEDIATE: 500, EDU_SECONDARY: 1000, EDU_SECONDARY_VOC: 1000,
    EDU_ASSOC_DIPLOMA: 2000, EDU_DIPLOMA: 2000,
    EDU_BACHELOR: 3000, EDU_BACHELOR_5: 3000,
    EDU_MASTERS: 5000, EDU_PHD: 8000,
}


def generate_synthetic_normals(
    real_samples: np.ndarray, n: int = 2000, seed: int = 42
) -> np.ndarray:
    """Generate synthetic normal samples anchored to real data distributions.

    Fits the generation to the real data's actual ranges and distributions,
    then applies business-rule constraints as guardrails.

    Columns are in SVM_NUMERIC_COLUMNS order (sorted alphabetically):
      act_1_total, age, cut_5_total, edu_ordinal, q_602_val

    Args:
        real_samples: The real normal samples array (n_real, 5) to match.
        n: Number of synthetic samples to generate.
        seed: Random seed for reproducibility.

    Returns:
        2-D array of shape (n, 5).
    """
    rng = np.random.default_rng(seed)

    # Column indices (sorted alphabetically):
    # act_1_total=0, age=1, cut_5_total=2, edu_ordinal=3, q_602_val=4
    IDX_ACT = 0
    IDX_AGE = 1
    IDX_CUT = 2
    IDX_EDU = 3
    IDX_SAL = 4

    # Learn real distributions
    real_means = real_samples.mean(axis=0)
    real_stds = real_samples.std(axis=0)

    # Education-tier salary ceilings for realistic correlation
    _EDU_SALARY_CEIL = {
        0: 5000, 1: 5000, 2: 5000, 3: 8000,       # no edu → intermediate
        4: 15000, 5: 15000,                          # secondary
        6: 25000, 7: 25000,                          # diploma
        8: 50000, 9: 50000, 10: 50000, 11: 50000,   # bachelor+
    }
    _EDU_MIN_AGE = {
        0: 15, 1: 15, 2: 15, 3: 15,
        4: 17, 5: 17, 6: 19, 7: 19,
        8: 21, 9: 21, 10: 23, 11: 25,
    }

    # Pre-compute log salary stats
    real_log_salary = np.log(real_samples[:, IDX_SAL].clip(min=200))
    log_mean = real_log_salary.mean()
    log_std = real_log_salary.std()

    samples: list[list[float]] = []

    for _ in range(n):
        # Education: sample from real distribution (rounded to nearest int)
        edu = round(rng.normal(real_means[IDX_EDU], real_stds[IDX_EDU]))
        edu = max(0, min(11, edu))

        # Age: sample from real distribution, enforce education minimum
        age = round(rng.normal(real_means[IDX_AGE], real_stds[IDX_AGE]))
        age = max(_EDU_MIN_AGE[edu], min(74, age))

        # Usual weekly hours: match real distribution
        usual = round(rng.normal(real_means[IDX_CUT], real_stds[IDX_CUT]))
        usual = max(1, min(84, usual))

        # Actual hours: correlated with usual, allow some variance
        actual = round(usual + rng.normal(real_means[IDX_ACT] - real_means[IDX_CUT], 6))
        actual = max(0, min(84, actual))

        # Salary: log-normal, capped by education tier
        salary = round(np.exp(rng.normal(log_mean, log_std)))
        salary = max(200, min(_EDU_SALARY_CEIL[edu], salary))

        # SVM_NUMERIC_COLUMNS order: act_1_total, age, cut_5_total, edu_ordinal, q_602_val
        samples.append([float(actual), float(age), float(usual), float(edu), float(salary)])

    return np.array(samples, dtype=np.float64)


# ---------------------------------------------------------------------------
# Step 3: Generate few-shot examples for LLM strategy
# ---------------------------------------------------------------------------

def generate_few_shot_examples(df: pd.DataFrame) -> list[dict]:
    """Pick 5 representative records: 3 normal, 2 anomalous."""
    notes_col = "الملاحظة"
    key_cols = [
        "age", "gender", "family_relation", "marage_status",
        "nationality", "q_301", "q_602_val", "cut_5_total", "act_1_total",
    ]

    examples: list[dict] = []

    # Parse numeric values for key columns
    records = []
    for _, row in df.iterrows():
        record: dict = {}
        for col in key_cols:
            val = row.get(col)
            if val is not None and str(val) != "NULL" and str(val).strip():
                try:
                    record[col] = int(val)
                except ValueError:
                    try:
                        record[col] = float(val)
                    except ValueError:
                        record[col] = str(val)
            else:
                record[col] = None
        record["_note"] = str(row.get(notes_col, "")) if row.get(notes_col) else ""
        records.append(record)

    # Select normal-looking records (ones with all fields, reasonable values)
    normal_candidates = [
        r for r in records
        if r.get("q_602_val") is not None
        and r.get("cut_5_total") is not None
        and r.get("age") is not None
        and 500 < (r.get("q_602_val") or 0) < 50000
        and 10 < (r.get("cut_5_total") or 0) <= 48
        and 0 < (r.get("act_1_total") or 0) <= 48
    ]

    # Select anomalous records
    anomaly_candidates = [
        r for r in records
        if (
            (r.get("q_602_val") is not None and (r["q_602_val"] > 50000 or r["q_602_val"] < 500))
            or (r.get("cut_5_total") is not None and r["cut_5_total"] > 84)
        )
    ]

    # Pick 3 normal, 2 anomalous
    for r in normal_candidates[:3]:
        inp = {k: v for k, v in r.items() if k != "_note" and v is not None}
        examples.append({"input": inp, "label": "normal"})

    for r in anomaly_candidates[:2]:
        inp = {k: v for k, v in r.items() if k != "_note" and v is not None}
        examples.append({"input": inp, "label": "anomaly"})

    return examples


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare LFS data for Sadeed")
    parser.add_argument(
        "--print-few-shot", action="store_true",
        help="Print few-shot examples JSON to stdout",
    )
    args = parser.parse_args()

    # Step 1: Generate expectation suite
    print("[1/3] Generating expectations/suite.json ...")
    suite = generate_suite()
    SUITE_PATH.parent.mkdir(parents=True, exist_ok=True)
    SUITE_PATH.write_text(json.dumps(suite, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"      Wrote {len(suite['expectations'])} expectations to {SUITE_PATH}")

    # Step 2: Extract real + synthetic normal samples for SVM
    print("[2/3] Extracting numeric data for SVM training ...")
    df = load_training_data()
    X_real = prepare_svm_data(df)
    print(f"      Real samples: {X_real.shape[0]}")

    X_synthetic = generate_synthetic_normals(X_real, n=2000, seed=42)
    print(f"      Synthetic samples: {X_synthetic.shape[0]}")

    X = np.vstack([X_real, X_synthetic])
    np.save(NORMAL_SAMPLES_PATH, X)
    print(f"      Saved {X.shape[0]} samples x {X.shape[1]} features to {NORMAL_SAMPLES_PATH}")
    print(f"      Columns (sorted): {SVM_NUMERIC_COLUMNS}")

    # Step 3: Generate few-shot examples
    print("[3/3] Generating few-shot examples ...")
    examples = generate_few_shot_examples(df)
    print(f"      Generated {len(examples)} examples ({sum(1 for e in examples if e['label'] == 'normal')} normal, {sum(1 for e in examples if e['label'] == 'anomaly')} anomaly)")

    if args.print_few_shot:
        print("\n--- FEW_SHOT_EXAMPLES (paste into llm_strategy.py) ---")
        print(json.dumps(examples, indent=4, ensure_ascii=False))

    print("\nDone.")


if __name__ == "__main__":
    main()
