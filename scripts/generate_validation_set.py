"""
Generate 1000-record adversarial validation CSV.

Designed to stress-test every layer:
  - 750 normal (500 vanilla + 150 edge-case near GE boundaries + 100 unusual-but-valid)
  - 100 GE anomalies (mix of clear violations and barely-over-threshold)
  - 100 SVM anomalies (extreme enough for SVM>0.5, but pass GE)
  - 50 SVM false positives (MUST trigger SVM>0.5 but are semantically valid)

Usage:
    python scripts/generate_validation_set.py
"""

from __future__ import annotations

import csv
import random
from pathlib import Path

OUTPUT_PATH = Path("data/validation_1000.csv")

# Salary ranges aligned with GE rules:
#   No edu/Primary/Intermediate: 1500-5000
#   Secondary: 3000-15000
#   Diploma: 4000-25000
#   Bachelor's: 3000-50000 (floor 3000)
#   Master's: 5000-50000 (floor 5000)
#   PhD: 8000-50000 (floor 8000)
EDU_PROFILES: dict[int, tuple[str, int, int, int]] = {
    10500031: ("No formal edu", 15, 1500, 4500),
    10500003: ("Primary", 15, 1500, 4500),
    10500017: ("Intermediate", 15, 2000, 4500),
    10500019: ("Secondary", 17, 3000, 12000),
    10500023: ("Diploma", 19, 4000, 18000),
    10500025: ("Bachelor's", 21, 5000, 25000),
    10500009: ("Master's", 23, 8000, 35000),
    10500010: ("PhD", 25, 12000, 45000),
}

GENDERS = [1600001, 1600002]
FAMILY_RELATIONS = [1700001, 1700021, 1700022]
MARITAL_STATUSES = [10600001, 10600002, 10600003, 10600004]

FIELDS = [
    "age", "gender", "family_relation", "marage_status", "nationality",
    "q_301", "q_602_val", "cut_5_total", "act_1_total",
    "_expected_verdict", "_category", "_description",
]


def _base(rng: random.Random, edu_code: int) -> dict:
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


# =========================================================================
# 500 Vanilla Normal — safe middle-of-range values
# =========================================================================

def gen_vanilla_normal(rng: random.Random) -> dict:
    edu_code = rng.choice(list(EDU_PROFILES.keys()))
    _, min_age, sal_low, sal_high = EDU_PROFILES[edu_code]
    rec = _base(rng, edu_code)
    if rec["age"] < min_age:
        rec["age"] = min_age
    mid_low = sal_low + int((sal_high - sal_low) * 0.2)
    mid_high = sal_high - int((sal_high - sal_low) * 0.2)
    rec["q_602_val"] = rng.randint(mid_low, max(mid_low + 1, mid_high))
    usual = rng.randint(32, 46)
    actual = usual + rng.randint(-4, 4)
    rec["cut_5_total"] = usual
    rec["act_1_total"] = max(1, min(84, actual))
    rec["_expected_verdict"] = "normal"
    rec["_category"] = "normal"
    rec["_description"] = "Normal record"
    return rec


# =========================================================================
# 150 Edge-Case Normal — right at GE boundaries but just within
# =========================================================================

EDGE_GENERATORS: list = []


def _edge(fn):
    EDGE_GENERATORS.append(fn)
    return fn


@_edge
def edge_salary_just_under_cap_no_edu(rng: random.Random) -> dict:
    """No edu, salary 4800-4999 — just under 5000 cap."""
    edu = rng.choice([10500031, 10500003, 10500017])
    rec = _base(rng, edu)
    rec["q_602_val"] = rng.randint(4800, 4999)
    rec["cut_5_total"] = rng.randint(35, 48)
    rec["act_1_total"] = rec["cut_5_total"] + rng.randint(-3, 3)
    rec["_description"] = f"{EDU_PROFILES[edu][0]}, salary {rec['q_602_val']} (just under 5000 cap)"
    return rec


@_edge
def edge_salary_just_above_floor_phd(rng: random.Random) -> dict:
    """PhD, salary 8000-8500 — just above 8000 floor."""
    rec = _base(rng, 10500010)
    rec["q_602_val"] = rng.randint(8000, 8500)
    rec["cut_5_total"] = rng.randint(35, 48)
    rec["act_1_total"] = rec["cut_5_total"] + rng.randint(-3, 3)
    rec["_description"] = f"PhD, salary {rec['q_602_val']} (just above 8000 floor)"
    return rec


@_edge
def edge_salary_just_above_floor_masters(rng: random.Random) -> dict:
    """Master's, salary 5000-5500 — just above 5000 floor."""
    rec = _base(rng, 10500009)
    rec["q_602_val"] = rng.randint(5000, 5500)
    rec["cut_5_total"] = rng.randint(35, 48)
    rec["act_1_total"] = rec["cut_5_total"] + rng.randint(-3, 3)
    rec["_description"] = f"Master's, salary {rec['q_602_val']} (just above 5000 floor)"
    return rec


@_edge
def edge_hours_gap_38(rng: random.Random) -> dict:
    """Hours gap of 35-39 — just under 40 threshold."""
    edu_code = rng.choice(list(EDU_PROFILES.keys()))
    _, _, sal_low, sal_high = EDU_PROFILES[edu_code]
    rec = _base(rng, edu_code)
    rec["q_602_val"] = rng.randint(sal_low, sal_high)
    gap = rng.randint(35, 39)
    if rng.random() < 0.5:
        rec["cut_5_total"] = rng.randint(55, 70)
        rec["act_1_total"] = rec["cut_5_total"] - gap
    else:
        rec["cut_5_total"] = rng.randint(10, 20)
        rec["act_1_total"] = min(84, rec["cut_5_total"] + gap)
    rec["_description"] = f"Hours gap {gap} (just under 40 threshold)"
    return rec


@_edge
def edge_age_exactly_min_for_edu(rng: random.Random) -> dict:
    """Exactly minimum age for education level."""
    edu_code = rng.choice([10500019, 10500023, 10500025, 10500009, 10500010])
    _, min_age, sal_low, sal_high = EDU_PROFILES[edu_code]
    rec = _base(rng, edu_code)
    rec["age"] = min_age
    rec["family_relation"] = 1700022
    rec["marage_status"] = 10600001
    rec["q_602_val"] = rng.randint(sal_low, sal_high)
    rec["cut_5_total"] = rng.randint(35, 48)
    rec["act_1_total"] = rec["cut_5_total"] + rng.randint(-3, 3)
    rec["_description"] = f"{min_age}yo with {EDU_PROFILES[edu_code][0]} (exactly min age)"
    return rec


@_edge
def edge_salary_just_under_secondary_cap(rng: random.Random) -> dict:
    """Secondary edu, salary 14000-14999 — just under 15000 cap."""
    rec = _base(rng, 10500019)
    rec["q_602_val"] = rng.randint(14000, 14999)
    rec["cut_5_total"] = rng.randint(35, 48)
    rec["act_1_total"] = rec["cut_5_total"] + rng.randint(-3, 3)
    rec["_description"] = f"Secondary, salary {rec['q_602_val']} (just under 15000 cap)"
    return rec


# =========================================================================
# 100 Unusual-but-Valid Normal — statistically uncommon, semantically fine
# =========================================================================

UNUSUAL_GENERATORS: list = []


def _unusual(fn):
    UNUSUAL_GENERATORS.append(fn)
    return fn


@_unusual
def unusual_older_high_edu(rng: random.Random) -> dict:
    """55-64yo with PhD/Master's, normal salary/hours."""
    edu = rng.choice([10500009, 10500010])
    _, _, sal_low, sal_high = EDU_PROFILES[edu]
    rec = _base(rng, edu)
    rec["age"] = rng.randint(55, 64)
    rec["q_602_val"] = rng.randint(sal_low, sal_high)
    rec["cut_5_total"] = rng.randint(35, 48)
    rec["act_1_total"] = rec["cut_5_total"] + rng.randint(-5, 5)
    rec["_description"] = f"{rec['age']}yo {EDU_PROFILES[edu][0]}, {rec['q_602_val']} SAR"
    return rec


@_unusual
def unusual_young_low_edu_moderate_hours(rng: random.Random) -> dict:
    """17-19yo, no edu, working 45-50 hours — legal, just unusual."""
    rec = _base(rng, 10500031)
    rec["age"] = rng.randint(17, 19)
    rec["family_relation"] = 1700022
    rec["marage_status"] = 10600001
    rec["q_602_val"] = rng.randint(1500, 3500)
    rec["cut_5_total"] = rng.randint(45, 50)
    rec["act_1_total"] = rec["cut_5_total"] + rng.randint(-3, 3)
    rec["_description"] = f"{rec['age']}yo no edu, {rec['cut_5_total']} hrs"
    return rec


@_unusual
def unusual_high_salary_high_edu(rng: random.Random) -> dict:
    """PhD near top salary, long but legal hours."""
    rec = _base(rng, 10500010)
    rec["q_602_val"] = rng.randint(38000, 45000)
    rec["cut_5_total"] = rng.randint(48, 55)
    rec["act_1_total"] = rec["cut_5_total"] + rng.randint(-5, 5)
    rec["_description"] = f"PhD, {rec['q_602_val']} SAR, {rec['cut_5_total']} hrs"
    return rec


@_unusual
def unusual_widowed_head(rng: random.Random) -> dict:
    """Widowed head of household — valid but uncommon demographic."""
    edu_code = rng.choice(list(EDU_PROFILES.keys()))
    _, _, sal_low, sal_high = EDU_PROFILES[edu_code]
    rec = _base(rng, edu_code)
    rec["family_relation"] = 1700001
    rec["marage_status"] = 10600004
    rec["age"] = max(rec["age"], 30)
    rec["q_602_val"] = rng.randint(sal_low, sal_high)
    rec["cut_5_total"] = rng.randint(35, 48)
    rec["act_1_total"] = rec["cut_5_total"] + rng.randint(-3, 3)
    rec["_description"] = f"Widowed head, {rec['age']}yo, {EDU_PROFILES[edu_code][0]}"
    return rec


# =========================================================================
# 100 GE Anomalies — mix of clear and barely-over-threshold
# =========================================================================

GE_GENERATORS: list = []


def _ge(fn):
    GE_GENERATORS.append(fn)
    return fn


@_ge
def ge_salary_barely_over_no_edu(rng: random.Random) -> dict:
    """No edu, salary 5001-6000 — barely over 5000 cap."""
    edu = rng.choice([10500031, 10500003, 10500017])
    rec = _base(rng, edu)
    rec["q_602_val"] = rng.randint(5001, 6000)
    rec["cut_5_total"] = rng.randint(35, 48)
    rec["act_1_total"] = rec["cut_5_total"] + rng.randint(-3, 3)
    rec["_description"] = f"{EDU_PROFILES[edu][0]}, salary {rec['q_602_val']} (barely over 5000 cap)"
    return rec


@_ge
def ge_salary_way_over_no_edu(rng: random.Random) -> dict:
    """No edu, salary 15000-40000 — clearly over cap."""
    edu = rng.choice([10500031, 10500003, 10500017])
    rec = _base(rng, edu)
    rec["q_602_val"] = rng.randint(15000, 40000)
    rec["cut_5_total"] = rng.randint(35, 48)
    rec["act_1_total"] = rec["cut_5_total"] + rng.randint(-3, 3)
    rec["_description"] = f"{EDU_PROFILES[edu][0]}, salary {rec['q_602_val']} (way over 5000 cap)"
    return rec


@_ge
def ge_salary_barely_under_floor_phd(rng: random.Random) -> dict:
    """PhD, salary 7000-7999 — barely under 8000 floor."""
    rec = _base(rng, 10500010)
    rec["q_602_val"] = rng.randint(7000, 7999)
    rec["cut_5_total"] = rng.randint(35, 48)
    rec["act_1_total"] = rec["cut_5_total"] + rng.randint(-3, 3)
    rec["_description"] = f"PhD, salary {rec['q_602_val']} (barely under 8000 floor)"
    return rec


@_ge
def ge_salary_way_under_floor_phd(rng: random.Random) -> dict:
    """PhD, salary 500-2000 — clearly under floor."""
    rec = _base(rng, 10500010)
    rec["q_602_val"] = rng.randint(500, 2000)
    rec["cut_5_total"] = rng.randint(35, 48)
    rec["act_1_total"] = rec["cut_5_total"] + rng.randint(-3, 3)
    rec["_description"] = f"PhD, salary {rec['q_602_val']} (way under 8000 floor)"
    return rec


@_ge
def ge_salary_under_floor_masters(rng: random.Random) -> dict:
    """Master's, salary under 5000."""
    rec = _base(rng, 10500009)
    rec["q_602_val"] = rng.randint(500, 4999)
    rec["cut_5_total"] = rng.randint(35, 48)
    rec["act_1_total"] = rec["cut_5_total"] + rng.randint(-3, 3)
    rec["_description"] = f"Master's, salary {rec['q_602_val']} (under 5000 floor)"
    return rec


@_ge
def ge_salary_under_floor_bachelor(rng: random.Random) -> dict:
    """Bachelor's, salary under 3000."""
    rec = _base(rng, 10500025)
    rec["q_602_val"] = rng.randint(500, 2999)
    rec["cut_5_total"] = rng.randint(35, 48)
    rec["act_1_total"] = rec["cut_5_total"] + rng.randint(-3, 3)
    rec["_description"] = f"Bachelor's, salary {rec['q_602_val']} (under 3000 floor)"
    return rec


@_ge
def ge_hours_barely_over_threshold(rng: random.Random) -> dict:
    """Hours gap 41-45 — barely over 40."""
    edu_code = rng.choice(list(EDU_PROFILES.keys()))
    _, _, sal_low, sal_high = EDU_PROFILES[edu_code]
    rec = _base(rng, edu_code)
    rec["q_602_val"] = rng.randint(sal_low, sal_high)
    gap = rng.randint(41, 45)
    rec["cut_5_total"] = rng.randint(55, 70)
    rec["act_1_total"] = max(0, rec["cut_5_total"] - gap)
    rec["_description"] = f"Hours gap {gap} (barely over 40 threshold)"
    return rec


@_ge
def ge_hours_extreme_gap(rng: random.Random) -> dict:
    """Hours gap 60+ — extreme mismatch."""
    edu_code = rng.choice(list(EDU_PROFILES.keys()))
    _, _, sal_low, sal_high = EDU_PROFILES[edu_code]
    rec = _base(rng, edu_code)
    rec["q_602_val"] = rng.randint(sal_low, sal_high)
    rec["cut_5_total"] = rng.randint(70, 84)
    rec["act_1_total"] = rng.randint(0, 5)
    gap = rec["cut_5_total"] - rec["act_1_total"]
    rec["_description"] = f"Hours gap {gap} (extreme)"
    return rec


@_ge
def ge_age_under_edu_min(rng: random.Random) -> dict:
    """Age below education minimum."""
    edu_code = rng.choice([10500019, 10500023, 10500025, 10500009, 10500010])
    _, min_age, sal_low, sal_high = EDU_PROFILES[edu_code]
    rec = _base(rng, edu_code)
    rec["age"] = rng.randint(15, min_age - 1)
    rec["family_relation"] = 1700022
    rec["marage_status"] = 10600001
    rec["q_602_val"] = rng.randint(sal_low, sal_high)
    rec["cut_5_total"] = rng.randint(35, 48)
    rec["act_1_total"] = rec["cut_5_total"] + rng.randint(-3, 3)
    rec["_description"] = f"{rec['age']}yo with {EDU_PROFILES[edu_code][0]} (min {min_age})"
    return rec


@_ge
def ge_spouse_not_married(rng: random.Random) -> dict:
    """Spouse with non-married status."""
    edu_code = rng.choice(list(EDU_PROFILES.keys()))
    _, _, sal_low, sal_high = EDU_PROFILES[edu_code]
    rec = _base(rng, edu_code)
    rec["family_relation"] = 1700021
    rec["marage_status"] = rng.choice([10600001, 10600003, 10600004])
    rec["age"] = max(rec["age"], 18)
    rec["q_602_val"] = rng.randint(sal_low, sal_high)
    rec["cut_5_total"] = rng.randint(35, 48)
    rec["act_1_total"] = rec["cut_5_total"] + rng.randint(-3, 3)
    status = {10600001: "never married", 10600003: "divorced", 10600004: "widowed"}
    rec["_description"] = f"Spouse but {status[rec['marage_status']]}"
    return rec


@_ge
def ge_child_employed(rng: random.Random) -> dict:
    """Under 15 with salary."""
    rec = _base(rng, 10500031)
    rec["age"] = rng.randint(10, 14)
    rec["family_relation"] = 1700022
    rec["marage_status"] = 10600001
    rec["q_602_val"] = rng.randint(500, 3000)
    rec["cut_5_total"] = rng.randint(10, 30)
    rec["act_1_total"] = rec["cut_5_total"]
    rec["_description"] = f"{rec['age']}yo child earning {rec['q_602_val']} SAR"
    return rec


@_ge
def ge_secondary_over_cap(rng: random.Random) -> dict:
    """Secondary edu, salary over 15000."""
    rec = _base(rng, 10500019)
    rec["q_602_val"] = rng.randint(15001, 40000)
    rec["cut_5_total"] = rng.randint(35, 48)
    rec["act_1_total"] = rec["cut_5_total"] + rng.randint(-3, 3)
    rec["_description"] = f"Secondary, salary {rec['q_602_val']} (over 15000 cap)"
    return rec


# =========================================================================
# 100 SVM Anomalies — extreme stats, pass all GE rules, SVM should flag
# =========================================================================

SVM_GENERATORS: list = []


def _svm(fn):
    SVM_GENERATORS.append(fn)
    return fn


@_svm
def svm_elderly_extreme_hours(rng: random.Random) -> dict:
    """67+ working 75+ hours — extreme outlier."""
    edu_code = rng.choice(list(EDU_PROFILES.keys()))
    _, _, sal_low, sal_high = EDU_PROFILES[edu_code]
    rec = _base(rng, edu_code)
    rec["age"] = rng.randint(67, 80)
    hours = rng.randint(75, 84)
    rec["cut_5_total"] = hours
    rec["act_1_total"] = hours
    rec["q_602_val"] = rng.randint(sal_low, sal_high)
    rec["_description"] = f"{rec['age']}yo working {hours} hrs/wk"
    return rec


@_svm
def svm_very_few_hours_high_salary(rng: random.Random) -> dict:
    """1-2 hrs/wk at 30k+ SAR — education-appropriate but extreme combo."""
    edu = rng.choice([10500009, 10500010])
    _, _, sal_low, sal_high = EDU_PROFILES[edu]
    rec = _base(rng, edu)
    rec["cut_5_total"] = rng.randint(1, 2)
    rec["act_1_total"] = rec["cut_5_total"]
    rec["q_602_val"] = rng.randint(30000, sal_high)
    rec["_description"] = f"{rec['cut_5_total']} hrs/wk at {rec['q_602_val']} SAR ({EDU_PROFILES[edu][0]})"
    return rec


@_svm
def svm_young_extreme_hours(rng: random.Random) -> dict:
    """15-16yo working 78+ hours."""
    edu = rng.choice([10500031, 10500003])
    rec = _base(rng, edu)
    rec["age"] = rng.randint(15, 16)
    rec["family_relation"] = 1700022
    rec["marage_status"] = 10600001
    hours = rng.randint(78, 84)
    rec["cut_5_total"] = hours
    rec["act_1_total"] = hours
    rec["q_602_val"] = rng.randint(1500, 3000)
    rec["_description"] = f"{rec['age']}yo working {hours} hrs/wk"
    return rec


@_svm
def svm_extreme_salary_hours_combo(rng: random.Random) -> dict:
    """High edu salary near cap + only 1-3 hours."""
    edu = rng.choice([10500009, 10500010])
    _, _, sal_low, sal_high = EDU_PROFILES[edu]
    rec = _base(rng, edu)
    rec["q_602_val"] = rng.randint(int(sal_high * 0.8), sal_high)
    rec["cut_5_total"] = rng.randint(1, 3)
    rec["act_1_total"] = rec["cut_5_total"]
    rec["_description"] = f"{EDU_PROFILES[edu][0]}, {rec['q_602_val']} SAR, {rec['cut_5_total']} hrs"
    return rec


# =========================================================================
# 50 SVM False Positives — MUST trigger SVM>0.5 but semantically valid
# Records with unusual-but-legal statistical profiles
# =========================================================================

FP_GENERATORS: list = []


def _fp(fn):
    FP_GENERATORS.append(fn)
    return fn


@_fp
def fp_high_salary_long_hours_phd(rng: random.Random) -> dict:
    """PhD, 40-44k SAR, 55-65 hours — high-performing academic, valid."""
    rec = _base(rng, 10500010)
    rec["q_602_val"] = rng.randint(40000, 44000)
    hours = rng.randint(55, 65)
    rec["cut_5_total"] = hours
    rec["act_1_total"] = hours + rng.randint(-3, 3)
    rec["_description"] = f"PhD, {rec['q_602_val']} SAR, {hours} hrs (high-performing, valid)"
    return rec


@_fp
def fp_masters_high_salary_long_hours(rng: random.Random) -> dict:
    """Master's, 28-34k SAR, 55-62 hours — senior professional, valid."""
    rec = _base(rng, 10500009)
    rec["q_602_val"] = rng.randint(28000, 34000)
    hours = rng.randint(55, 62)
    rec["cut_5_total"] = hours
    rec["act_1_total"] = hours + rng.randint(-3, 3)
    rec["_description"] = f"Master's, {rec['q_602_val']} SAR, {hours} hrs (senior professional, valid)"
    return rec


@_fp
def fp_bachelor_high_salary_long_hours(rng: random.Random) -> dict:
    """Bachelor's, 20-24k SAR, 55-65 hours — ambitious worker, valid."""
    rec = _base(rng, 10500025)
    rec["q_602_val"] = rng.randint(20000, 24000)
    hours = rng.randint(55, 65)
    rec["cut_5_total"] = hours
    rec["act_1_total"] = hours + rng.randint(-3, 3)
    rec["_description"] = f"Bachelor's, {rec['q_602_val']} SAR, {hours} hrs (ambitious, valid)"
    return rec


@_fp
def fp_low_edu_moderate_salary_high_hours(rng: random.Random) -> dict:
    """No edu, 3500-4500 SAR, 55-65 hours — blue collar, valid."""
    rec = _base(rng, 10500031)
    rec["q_602_val"] = rng.randint(3500, 4500)
    hours = rng.randint(55, 65)
    rec["cut_5_total"] = hours
    rec["act_1_total"] = hours + rng.randint(-3, 3)
    rec["_description"] = f"No edu, {rec['q_602_val']} SAR, {hours} hrs (blue collar, valid)"
    return rec


@_fp
def fp_diploma_high_salary(rng: random.Random) -> dict:
    """Diploma, 16-18k SAR, 50-58 hours — skilled technician, valid."""
    rec = _base(rng, 10500023)
    rec["q_602_val"] = rng.randint(16000, 18000)
    hours = rng.randint(50, 58)
    rec["cut_5_total"] = hours
    rec["act_1_total"] = hours + rng.randint(-3, 3)
    rec["_description"] = f"Diploma, {rec['q_602_val']} SAR, {hours} hrs (skilled tech, valid)"
    return rec


# =========================================================================
# Main
# =========================================================================

def main() -> None:
    rng = random.Random(2026)
    records: list[dict] = []

    # 500 vanilla normal
    for _ in range(500):
        records.append(gen_vanilla_normal(rng))

    # 150 edge-case normal
    for i in range(150):
        gen = EDGE_GENERATORS[i % len(EDGE_GENERATORS)]
        rec = gen(rng)
        rec["_expected_verdict"] = "normal"
        rec["_category"] = "normal_edge"
        records.append(rec)

    # 100 unusual-but-valid normal
    for i in range(100):
        gen = UNUSUAL_GENERATORS[i % len(UNUSUAL_GENERATORS)]
        rec = gen(rng)
        rec["_expected_verdict"] = "normal"
        rec["_category"] = "normal_unusual"
        records.append(rec)

    # 100 GE anomalies
    for i in range(100):
        gen = GE_GENERATORS[i % len(GE_GENERATORS)]
        rec = gen(rng)
        rec["_expected_verdict"] = "anomaly"
        rec["_category"] = "ge_anomaly"
        records.append(rec)

    # 100 SVM anomalies
    for i in range(100):
        gen = SVM_GENERATORS[i % len(SVM_GENERATORS)]
        rec = gen(rng)
        rec["_expected_verdict"] = "anomaly"
        rec["_category"] = "svm_anomaly"
        records.append(rec)

    # 50 SVM false positives
    for i in range(50):
        gen = FP_GENERATORS[i % len(FP_GENERATORS)]
        rec = gen(rng)
        rec["_expected_verdict"] = "normal"
        rec["_category"] = "svm_false_positive"
        records.append(rec)

    # Shuffle
    rng.shuffle(records)

    # Write CSV
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(records)

    cats = {}
    for r in records:
        cats[r["_category"]] = cats.get(r["_category"], 0) + 1
    print(f"Generated {len(records)} records -> {OUTPUT_PATH}")
    for cat, count in sorted(cats.items()):
        print(f"  {cat}: {count}")


if __name__ == "__main__":
    main()
