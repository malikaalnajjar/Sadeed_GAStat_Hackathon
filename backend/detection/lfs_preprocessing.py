"""LFS-specific data preprocessing for the Great Expectations detector.

Computes derived helper columns (prefixed ``_rule_``) that encode conditional
business rules as numeric values, enabling GE pair-comparison expectations to
express ``IF condition THEN constraint`` logic.

This module is intentionally separated from the generic GE detector so that
the detector class remains reusable for non-LFS datasets.
"""

from __future__ import annotations

from typing import Any

# ---------------------------------------------------------------------------
# Code constants (from LFS_Training_Dataset.xlsx)
# ---------------------------------------------------------------------------

# Education level codes (q_301)
_EDU_SECONDARY = 10500019
_EDU_SECONDARY_VOC = 10500020
_EDU_ASSOC_DIPLOMA = 10500021
_EDU_DIPLOMA = 10500023
_EDU_BACHELOR = 10500025
_EDU_BACHELOR_5 = 10500026
_EDU_MASTERS = 10500009
_EDU_PHD = 10500010

_SECONDARY_PLUS = {
    _EDU_SECONDARY, _EDU_SECONDARY_VOC, _EDU_ASSOC_DIPLOMA, _EDU_DIPLOMA,
    _EDU_BACHELOR, _EDU_BACHELOR_5, _EDU_MASTERS, _EDU_PHD,
}
_DIPLOMA_PLUS = {
    _EDU_DIPLOMA, _EDU_ASSOC_DIPLOMA, _EDU_BACHELOR, _EDU_BACHELOR_5,
    _EDU_MASTERS, _EDU_PHD,
}
_BACHELOR_PLUS = {_EDU_BACHELOR, _EDU_BACHELOR_5, _EDU_MASTERS, _EDU_PHD}
_MASTERS_PLUS = {_EDU_MASTERS, _EDU_PHD}
_PHD_ONLY = {_EDU_PHD}

# Family relation codes
_FAM_HEAD = 1700001
_FAM_WIFE = 1700021
_FAM_DOMESTIC = 1700010
_GRANDPARENT_CODES = {1700007, 1700008}

# Marriage status
_MAR_MARRIED = 10600002
_MAR_NEVER = 10600001

# Nationality
_NAT_SAUDI = 1800001

# Job sector (q_534)
_SECTOR_PUBLIC = 99400001
_SECTOR_PRIVATE = 99400003


# Eastern Arabic (٠-٩) and Extended Arabic-Indic (۰-۹) digit mapping
_ARABIC_DIGIT_MAP = str.maketrans(
    "٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹",
    "01234567890123456789",
)


def _safe_int(value: Any) -> int | None:
    """Parse *value* as an integer, handling decimals and Arabic numerals.

    Handles:
        - Normal integers: 35 → 35
        - Decimal strings: "35.0" → 35
        - Eastern Arabic: "٣٥" → 35
        - Mixed: "١٥٠٠٠.٠" → 15000
    """
    if value is None:
        return None
    try:
        return int(value)
    except (ValueError, TypeError):
        pass
    try:
        s = str(value).strip().translate(_ARABIC_DIGIT_MAP)
        return int(float(s))
    except (ValueError, TypeError):
        return None


_NUMERIC_FIELDS = {
    "age", "gender", "family_relation", "marage_status", "nationality",
    "q_301", "q_602_val", "cut_5_total", "act_1_total", "q_534",
}

# ---------------------------------------------------------------------------
# Label-to-code mappings for form-submitted string values
# ---------------------------------------------------------------------------
# Forms display human-readable labels; the backend needs integer codes.
# Keys are lowercased and stripped for case-insensitive matching.

_GENDER_LABELS: dict[str, int] = {
    "male": 1600001,
    "female": 1600002,
    "ذكر": 1600001,
    "أنثى": 1600002,
}

_FAMILY_RELATION_LABELS: dict[str, int] = {
    "head": 1700001,
    "head of household": 1700001,
    "spouse": 1700021,
    "wife": 1700021,
    "husband": 1700021,
    "son": 1700022,
    "daughter": 1700023,
    "father": 1700004,
    "mother": 1700005,
    "brother": 1700003,
    "sister": 1700006,
    "grandfather": 1700007,
    "grandmother": 1700008,
    "grandchild": 1700009,
    "domestic worker": 1700010,
    "other relative": 1700011,
    "other": 1700030,
    "رب الأسرة": 1700001,
    "زوج": 1700021,
    "زوجة": 1700021,
    "ابن": 1700022,
    "ابنة": 1700023,
    "عامل منزلي": 1700010,
}

_MARAGE_STATUS_LABELS: dict[str, int] = {
    "never married": 10600001,
    "single": 10600001,
    "married": 10600002,
    "divorced": 10600003,
    "widowed": 10600004,
    "أعزب": 10600001,
    "لم يسبق له الزواج": 10600001,
    "متزوج": 10600002,
    "متزوجة": 10600002,
    "مطلق": 10600003,
    "مطلقة": 10600003,
    "أرمل": 10600004,
    "أرملة": 10600004,
}

_NATIONALITY_LABELS: dict[str, int] = {
    "saudi": 1800001,
    "sa": 1800001,
    "سعودي": 1800001,
    "سعودية": 1800001,
}

_EDUCATION_LABELS: dict[str, int] = {
    "no formal education": 10500031,
    "none": 10500031,
    "0": 10500031,
    "primary incomplete": 10500011,
    "1": 10500011,
    "primary": 10500003,
    "primary education": 10500003,
    "2": 10500003,
    "intermediate": 10500017,
    "3": 10500017,
    "secondary": 10500019,
    "4": 10500019,
    "secondary vocational": 10500020,
    "vocational": 10500020,
    "5": 10500020,
    "associate diploma": 10500021,
    "6": 10500021,
    "diploma": 10500023,
    "7": 10500023,
    "bachelor": 10500025,
    "bachelors": 10500025,
    "bachelor's": 10500025,
    "8": 10500025,
    "bachelor 5yr": 10500026,
    "9": 10500026,
    "master": 10500009,
    "masters": 10500009,
    "master's": 10500009,
    "10": 10500009,
    "phd": 10500010,
    "doctorate": 10500010,
    "11": 10500010,
}

_SECTOR_LABELS: dict[str, int] = {
    "public": 99400001,
    "public sector": 99400001,
    "semi-public": 99400002,
    "private": 99400003,
    "private sector": 99400003,
    "domestic worker": 99400004,
}

_LABEL_MAPS: dict[str, dict[str, int]] = {
    "gender": _GENDER_LABELS,
    "family_relation": _FAMILY_RELATION_LABELS,
    "marage_status": _MARAGE_STATUS_LABELS,
    "nationality": _NATIONALITY_LABELS,
    "q_301": _EDUCATION_LABELS,
    "q_534": _SECTOR_LABELS,
}


def add_derived_columns(data: dict[str, Any]) -> dict[str, Any]:
    """Coerce string values to integers and add derived ``_rule_*`` columns.

    Form scanners (Chrome extension, Google Forms) send all values as strings.
    GE expectations compare against integer code sets, so we must coerce first.

    Args:
        data: Raw data dict from the API request.

    Returns:
        A new dict containing all original keys plus the derived columns.
    """
    out = dict(data)

    # Normalise human-readable labels to integer codes
    for field, label_map in _LABEL_MAPS.items():
        val = out.get(field)
        if val is not None and isinstance(val, str):
            lookup = val.strip().lower()
            if lookup in label_map:
                out[field] = label_map[lookup]

    # Coerce string values to integers so GE set/range checks work
    for field in _NUMERIC_FIELDS:
        if field in out:
            parsed = _safe_int(out[field])
            if parsed is not None:
                out[field] = parsed

    age = _safe_int(out.get("age"))
    family = _safe_int(out.get("family_relation"))
    edu = _safe_int(out.get("q_301"))
    marital = _safe_int(out.get("marage_status"))
    sector = _safe_int(out.get("q_534"))

    # Constant zero for comparisons
    out["_rule_zero"] = 0

    # Rule 2001: Head of household must be >= 15
    out["_rule_2001_min_head_age"] = 15 if family == _FAM_HEAD else 0

    # Rules 2011-2016: Minimum age for education levels
    out["_rule_2011_min_secondary_age"] = 17 if edu in _SECONDARY_PLUS else 0
    out["_rule_2012_min_diploma_age"] = 19 if edu in _DIPLOMA_PLUS else 0
    out["_rule_2013_min_bachelor_age"] = 21 if edu in _BACHELOR_PLUS else 0
    out["_rule_2015_min_masters_age"] = 23 if edu in _MASTERS_PLUS else 0
    out["_rule_2016_min_phd_age"] = 25 if edu in _PHD_ONLY else 0

    # Boolean-style rules: 1 = valid (passes A >= B with B=0),
    # -1 = violation (fails A >= B with B=0).

    # Rule 2031: Spouse must be married
    if family == _FAM_WIFE and marital != _MAR_MARRIED:
        out["_rule_2031_spouse_married"] = -1
    else:
        out["_rule_2031_spouse_married"] = 1

    # Rule 2075: Grandparent must be >= 30
    if family in _GRANDPARENT_CODES and age is not None and age < 30:
        out["_rule_2075_grandparent_age"] = -1
    else:
        out["_rule_2075_grandparent_age"] = 1

    # Rule 2078: Public sector workers must be >= 17
    if sector == _SECTOR_PUBLIC and age is not None and age < 17:
        out["_rule_2078_public_sector_age"] = -1
    else:
        out["_rule_2078_public_sector_age"] = 1

    # Rule 2141: Domestic workers must be >= 15
    if family == _FAM_DOMESTIC and age is not None and age < 15:
        out["_rule_2141_domestic_worker_age"] = -1
    else:
        out["_rule_2141_domestic_worker_age"] = 1

    # Rule 4069: Under 15 cannot have any marital status except "never married"
    _MAR_NEVER = 10600001
    if age is not None and age < 15 and marital is not None and marital != _MAR_NEVER:
        out["_rule_4069_child_not_married"] = -1
    else:
        out["_rule_4069_child_not_married"] = 1

    # Rule 4030: Spouse should be >= 18
    if family == _FAM_WIFE and age is not None and age < 18:
        out["_rule_4030_spouse_age"] = -1
    else:
        out["_rule_4030_spouse_age"] = 1

    salary = _safe_int(data.get("q_602_val"))
    usual_hrs = _safe_int(out.get("cut_5_total"))
    actual_hrs = _safe_int(out.get("act_1_total"))
    nationality = _safe_int(out.get("nationality"))

    # Rule 3047: University degree or higher + salary < 3000
    if edu in _BACHELOR_PLUS and salary is not None and salary < 3000:
        out["_rule_3047_uni_salary"] = -1
    else:
        out["_rule_3047_uni_salary"] = 1

    # Rule 3031: Public sector + usual hours < 35
    if sector == _SECTOR_PUBLIC and usual_hrs is not None and usual_hrs < 35:
        out["_rule_3031_public_usual_hours"] = -1
    else:
        out["_rule_3031_public_usual_hours"] = 1

    # Rule 3032: Public sector + actual hours < 35
    if sector == _SECTOR_PUBLIC and actual_hrs is not None and actual_hrs < 35:
        out["_rule_3032_public_actual_hours"] = -1
    else:
        out["_rule_3032_public_actual_hours"] = 1

    # Rule 3035: Private sector + usual hours < 40
    if sector == _SECTOR_PRIVATE and usual_hrs is not None and usual_hrs < 40:
        out["_rule_3035_private_usual_hours"] = -1
    else:
        out["_rule_3035_private_usual_hours"] = 1

    # Rule 3036: Private sector + actual hours < 40
    if sector == _SECTOR_PRIVATE and actual_hrs is not None and actual_hrs < 40:
        out["_rule_3036_private_actual_hours"] = -1
    else:
        out["_rule_3036_private_actual_hours"] = 1

    # Rule 2053: Grandparent cannot be never married
    if family in _GRANDPARENT_CODES and marital == _MAR_NEVER:
        out["_rule_2053_grandparent_married"] = -1
    else:
        out["_rule_2053_grandparent_married"] = 1

    # Rule 2090: Never married cannot have spouse relationship
    if marital == _MAR_NEVER and family == _FAM_WIFE:
        out["_rule_2090_never_married_spouse"] = -1
    else:
        out["_rule_2090_never_married_spouse"] = 1

    # Rule 3042: Age > 30 and salary < 1000
    if age is not None and age > 30 and salary is not None and salary < 1000:
        out["_rule_3042_age_salary"] = -1
    else:
        out["_rule_3042_age_salary"] = 1

    # Rule 3043: Head of household and salary < 2000
    if family == _FAM_HEAD and salary is not None and salary < 2000:
        out["_rule_3043_head_salary"] = -1
    else:
        out["_rule_3043_head_salary"] = 1

    # Rule 3044: Saudi and salary < 3000
    if nationality == _NAT_SAUDI and salary is not None and salary < 3000:
        out["_rule_3044_saudi_salary"] = -1
    else:
        out["_rule_3044_saudi_salary"] = 1

    # Rule 3071: Age < 25 and salary > 10000
    if age is not None and age < 25 and salary is not None and salary > 10000:
        out["_rule_3071_young_high_salary"] = -1
    else:
        out["_rule_3071_young_high_salary"] = 1

    # Rule 3027: Usual working hours < 20
    if usual_hrs is not None and usual_hrs < 20:
        out["_rule_3027_low_usual_hours"] = -1
    else:
        out["_rule_3027_low_usual_hours"] = 1

    # Rule 3028: Usual working hours > 48
    if usual_hrs is not None and usual_hrs > 48:
        out["_rule_3028_high_usual_hours"] = -1
    else:
        out["_rule_3028_high_usual_hours"] = 1

    # Rule 3030: Actual working hours > 48
    if actual_hrs is not None and actual_hrs > 48:
        out["_rule_3030_high_actual_hours"] = -1
    else:
        out["_rule_3030_high_actual_hours"] = 1

    return out
