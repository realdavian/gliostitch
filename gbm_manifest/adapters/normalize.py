"""Value normalisation: raw per-dataset clinical values -> canonical vocabulary."""
from __future__ import annotations

import math
import re
from typing import Optional

from ..core.schema import EORCategory

_ALIVE_RE = re.compile(r"aliv", re.IGNORECASE)
_INT_RE = re.compile(r"-?\d+")
_NTR_RE = re.compile(r"\bntr\b|near[\s-]*total")
_NULL_TOKENS = {"", "na", "nan", "n/a", "none", "null", "not available", "unknown", "indeterminate"}


def _blank(v) -> bool:
    if v is None:
        return True
    if isinstance(v, float) and math.isnan(v):
        return True
    return str(v).strip().lower() in _NULL_TOKENS


def raw_str(v) -> Optional[str]:
    return None if _blank(v) else str(v).strip()


def to_float(v) -> Optional[float]:
    if _blank(v):
        return None
    try:
        return float(str(v).strip())
    except ValueError:
        return None


def parse_survival_days(raw) -> tuple[Optional[float], Optional[int]]:
    """BraTS dual-encoded Survival_days.
    '289' -> (289.0, 1);  'ALIVE (361 days later)' -> (361.0, 0);
    blank/unparseable -> (None, None)."""
    if _blank(raw):
        return (None, None)
    s = str(raw).strip()
    m = _INT_RE.search(s)
    days = float(m.group()) if m else None
    if _ALIVE_RE.search(s):
        return (days, 0)
    if m and s.lstrip("-").isdigit():
        return (days, 1)
    return (days, None)


def event_from_status(status) -> Optional[int]:
    """UPENN Survival_Status. Deceased -> 1, Alive -> 0, Lost -> None."""
    if _blank(status):
        return None
    s = str(status).strip().lower()
    if s.startswith("deceased"):
        return 1
    if s.startswith("alive"):
        return 0
    return None


def event_from_censored_flag(flag) -> Optional[int]:
    """RHUH right_censored (inverse of event). no -> 1, yes -> 0."""
    if _blank(flag):
        return None
    s = str(flag).strip().lower()
    if s in {"no", "n", "false", "0"}:
        return 1
    if s in {"yes", "y", "true", "1"}:
        return 0
    return None


def event_from_int(value) -> Optional[int]:
    """UCSF '1-dead 0-alive' column."""
    if _blank(value):
        return None
    try:
        iv = int(float(str(value).strip()))
    except ValueError:
        return None
    return iv if iv in (0, 1) else None


def normalize_grade(raw) -> Optional[int]:
    """Numeric grades pass through; HGG -> 4, LGG -> 2. Roman numerals accepted."""
    if _blank(raw):
        return None
    s = str(raw).strip().lower()
    roman = {"ii": 2, "iii": 3, "iv": 4}
    if s in roman:
        return roman[s]
    if s in {"hgg", "high", "high-grade", "high grade"}:
        return 4
    if s in {"lgg", "low", "low-grade", "low grade"}:
        return 2
    m = _INT_RE.search(s)
    if m:
        g = int(m.group())
        return g if g in (1, 2, 3, 4) else None
    return None


def normalize_eor_categorical(raw) -> EORCategory:
    """GTR/NTR/STR/biopsy strings (BraTS, RHUH, UCSF)."""
    if _blank(raw):
        return EORCategory.UNKNOWN
    s = str(raw).strip().lower()
    # NTR is checked first and on a word boundary: "ntr" is a substring of
    # ordinary words ("contrast"), and RHUH records near-total resection as a
    # bare "NTR". Falling through to UNKNOWN would record a known extent as
    # missing and silently inflate the unknown stratum.
    if _NTR_RE.search(s):
        return EORCategory.NTR
    if "gtr" in s or "gross" in s:
        return EORCategory.GTR
    if "str" in s or "subtotal" in s:
        return EORCategory.STR
    if "biops" in s:
        return EORCategory.BIOPSY
    return EORCategory.UNKNOWN


def normalize_eor_binary(yn) -> EORCategory:
    """UPENN GTR_over90percent Y/N -> GTR / non_GTR."""
    if _blank(yn):
        return EORCategory.UNKNOWN
    s = str(yn).strip().lower()
    if s in {"y", "yes", "true", "1"}:
        return EORCategory.GTR
    if s in {"n", "no", "false", "0"}:
        return EORCategory.NON_GTR
    return EORCategory.UNKNOWN


def normalize_idh(raw) -> Optional[str]:
    if _blank(raw):
        return None
    s = str(raw).strip().lower()
    if s.startswith("wt") or "wild" in s:
        return "wildtype"
    if s.startswith("mut"):
        return "mutant"
    return None


def normalize_mgmt(raw) -> Optional[str]:
    """Order matters: test 'unmeth' before 'meth'."""
    if _blank(raw):
        return None
    s = str(raw).strip().lower()
    if "unmeth" in s or s in {"negative", "neg"}:
        return "unmethylated"
    if "meth" in s or s in {"positive", "pos"}:
        return "methylated"
    return None
