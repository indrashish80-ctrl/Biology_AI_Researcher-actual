"""
stats_parsing.py
----------------
Deterministic (no-AI) parsers for the numbers and design cues that
scientific text reports: p-values, confidence intervals, sample sizes,
effect sizes, doses, and study-design keywords, plus a quote-grounding
check.

Why deterministic: every number an AI extractor returns is later checked
against what these parsers find in the source text. If the AI reports a
p-value that appears nowhere in the paper, it is dropped and flagged.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

_NUM = r"[-+]?\d+(?:\.\d+)?"


def normalize_numeric_text(text: str) -> str:
    """Unify minus signs / dashes / spacing so regexes see plain ASCII numbers."""
    if not text:
        return ""
    t = text.replace("\u2212", "-").replace("\u2012", "-").replace("\u2013", "-").replace("\u2014", "-")
    t = t.replace("\u2264", "<=").replace("\u2265", ">=").replace("\u00d7", "x").replace("\u00a0", " ")
    t = t.replace("\u2009", " ").replace("\u202f", " ").replace("\u2011", "-").replace("\u2010", "-")
    # v25: the Lancet's mid-dot decimal ('p=0\u00b719', 'HR 0\u00b797') was read as 'p=0' - a non-significant result
    # became 'significant'. A middle dot or bullet between digits is a decimal point.
    t = re.sub(r"(?<=\d)[\u00b7\u2219\u22c5](?=\d)", ".", t)
    return t


# ----------------------------------------------------------------------
# P-VALUES
# ----------------------------------------------------------------------

_P_RE = re.compile(
    r"\b[pP]\s*(?:-?\s*value)?\s*(<=|>=|<|>|=)\s*"
    r"(\d*\.?\d+(?:\s*[x*]\s*10\s*\^?\s*-?\d+|[eE]-?\d+)?)"
)


def _to_float(s: str) -> Optional[float]:
    s = s.strip().replace(" ", "")
    m = re.fullmatch(r"(\d*\.?\d+)[x*]10\^?(-?\d+)", s)
    try:
        if m:
            return float(m.group(1)) * (10 ** int(m.group(2)))
        return float(s)
    except ValueError:
        return None


def parse_p_values(text: str) -> List[Dict[str, Any]]:
    out = []
    for m in _P_RE.finditer(normalize_numeric_text(text)):
        v = _to_float(m.group(2))
        if v is not None and 0 <= v <= 1:
            out.append({"comparator": m.group(1), "value": v, "raw": m.group(0)})
    return out


def p_value_in_text(value: float, text: str, rel_tol: float = 0.01) -> bool:
    """True if some p-value in `text` equals `value` (within tolerance)."""
    for p in parse_p_values(text):
        if abs(p["value"] - value) <= max(rel_tol * abs(value), 1e-12):
            return True
    return False


# ----------------------------------------------------------------------
# CONFIDENCE INTERVALS
# ----------------------------------------------------------------------

_CI_RE = re.compile(
    rf"(\d{{2}}(?:\.\d)?)\s*%\s*(?:CI|C\.I\.|confidence interval)\s*[:=,]?\s*[\[\(]?\s*"
    rf"({_NUM})\s*(?:to|,|;|-)\s*({_NUM})",
    re.IGNORECASE,
)


def parse_confidence_intervals(text: str) -> List[Dict[str, Any]]:
    out = []
    for m in _CI_RE.finditer(normalize_numeric_text(text)):
        lo, hi = float(m.group(2)), float(m.group(3))
        if lo <= hi:
            out.append({"level": float(m.group(1)), "low": lo, "high": hi})
    return out


# ----------------------------------------------------------------------
# SAMPLE SIZES
# ----------------------------------------------------------------------

_N_RE = re.compile(r"\b[nN]\s*=\s*(\d[\d,]*)")
# a count of study units ('89 breast cancer patients', '40 female BALB/c mice'). Cell counts ('450000 cells per well')
# and cell-line names ('MDA-MB-231 cells', '4T1') are NOT sample sizes; neither are doses or durations.
_N_NOUN_RE = re.compile(
    r"(?<![-\w.^/])(\d[\d,]*)\s+(?!(?:mM|µM|uM|nM|pM|mg|g|kg|ml|mL|µg|ng|IU|%|h|hours?|days?|weeks?|months?|years?|times|fold|x|×)\b)"
    r"(?:(?![\d])[A-Za-z][\w/()-]*\s+){0,3}?"
    r"(plants?|mice|mouse|rats?|patients?|participants?|subjects?|animals?|individuals?|"
    r"replicates?|seedlings?|larvae|flies|fish|embryos?|volunteers?|women|men|children|cases|controls|donors|pigs|dogs|rabbits)\b",
    re.IGNORECASE)


def parse_sample_sizes(text: str) -> List[int]:
    t = normalize_numeric_text(text)
    vals = []
    for m in _N_RE.finditer(t):
        vals.append(int(m.group(1).replace(",", "")))
    for m in _N_NOUN_RE.finditer(t):
        vals.append(int(m.group(1).replace(",", "")))
    return [v for v in vals if 1 <= v <= 10_000_000]


# ----------------------------------------------------------------------
# EFFECT SIZES
# ----------------------------------------------------------------------

_EFFECT_PATTERNS: List[Tuple[str, str]] = [
    ("cohens_d", rf"Cohen'?s\s*d\s*[=:]\s*({_NUM})"),
    ("hedges_g", rf"Hedges'?\s*g\s*[=:]\s*({_NUM})"),
    ("odds_ratio", rf"\b(?:OR|odds ratio)\s*[=:]?\s*({_NUM})"),
    ("hazard_ratio", rf"\b(?:HR|hazard ratio)\s*[=:]?\s*({_NUM})"),
    ("risk_ratio", rf"\b(?:RR|risk ratio|relative risk)\s*[=:]?\s*({_NUM})"),
    ("correlation_r", rf"\br\s*=\s*({_NUM})"),
    ("r_squared", rf"\bR\^?2\s*=\s*({_NUM})"),
    ("eta_squared", rf"\b(?:eta|η)\^?2?\s*=\s*({_NUM})"),
    ("fold_change", rf"({_NUM})\s*-?\s*fold(?:\s*change)?"),
    ("percent_change", rf"(?:increased|decreased|reduced|higher|lower|improved|elevated)\s+(?:[\w-]+\s+){{0,3}}by\s+({_NUM})\s*%"),
]


def parse_effect_sizes(text: str) -> List[Dict[str, Any]]:
    t = normalize_numeric_text(text)
    out = []
    for kind, pat in _EFFECT_PATTERNS:
        for m in re.finditer(pat, t, flags=re.IGNORECASE if kind != "correlation_r" else 0):
            try:
                out.append({"type": kind, "value": float(m.group(1))})
            except ValueError:
                pass
    return out


# ----------------------------------------------------------------------
# DOSES / DURATIONS
# ----------------------------------------------------------------------

_UNIT_TO_BASE = {  # exact (case-sensitive) unit -> (base unit, multiplier)
    "M": ("M", 1.0), "mM": ("M", 1e-3), "uM": ("M", 1e-6), "µM": ("M", 1e-6), "nM": ("M", 1e-9),
    "pM": ("M", 1e-12),
    "kg": ("g", 1e3), "g": ("g", 1.0), "mg": ("g", 1e-3), "ug": ("g", 1e-6), "µg": ("g", 1e-6),
    "ng": ("g", 1e-9),
}
_DOSE_RE = re.compile(rf"(?<![\d.,])(\d{{1,3}}(?:,\d{{3}})+(?!\d)|{_NUM})\s*(mM|µM|uM|nM|pM|M|kg|mg|µg|ug|ng|g)(?:\s*/\s*(kg|l|ml|L|mL|day|d))?(?![a-zA-Z])")


def _body_measure(t: str, m: "re.Match") -> bool:
    """'BMI >= 27 kg/m2', 'weighing 70 kg', 'weight loss of 2 kg': a body measurement, never a drug dose. A dose is never
    in plain kilograms of the patient; mg/kg (dose per body weight) still counts."""
    if m.group(2) != "kg":
        return False
    if m.group(3) is None or re.match(r"\s*/\s*m\b|\s*/\s*m\s*(?:2|²|\^2)", t[m.end(2):m.end(2) + 8]):
        return True                       # bare 'kg' or 'kg/m2'
    return bool(re.search(r"\b(?:bmi|body mass|body[- ]weight|weigh\w*|lost|loss|gain\w*)\b[^.;]{0,25}$",
                          t[max(0, m.start() - 40):m.start()], re.I))


def parse_dose(text: Optional[str]) -> Optional[Dict[str, Any]]:
    """'10 mM' -> {'value':10,'unit':'mM','base_value':0.01,'base_unit':'M'}; None if unparseable."""
    if not text:
        return None
    t = normalize_numeric_text(str(text))
    # the first REAL dose: never a zero-dose control ('0 mM metformin-treated cells') or a figure panel ('Figure 3G')
    m = next((x for x in _DOSE_RE.finditer(t) if float(x.group(1).replace(",", "")) > 0
              and not re.search(r"(?:fig(?:ure)?s?\.?|panel|table)\s*[\w,\s]{0,6}$", t[max(0, x.start() - 14):x.start()], re.I)
              and not _body_measure(t, x)), None)
    if not m:
        return None
    value, unit, per = float(m.group(1).replace(",", "")), m.group(2), m.group(3)      # '1,700 mg' = 1700 mg
    base = _UNIT_TO_BASE.get(unit)
    if not base:
        return None
    return {"value": value, "unit": unit + (f"/{per}" if per else ""), "per": per,
            "base_value": value * base[1], "base_unit": base[0] + (f"/{per}" if per else "")}


_DUR_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(seconds?|secs?|minutes?|mins?|hours?|hrs?|h|days?|d|weeks?|wks?|months?|years?|yrs?)\b",
                     re.IGNORECASE)
_DUR_TO_DAYS = {"s": 1 / 86400, "m": 1 / 1440, "h": 1 / 24, "d": 1, "w": 7, "mo": 30, "y": 365}


def parse_duration_days(text: Optional[str]) -> Optional[float]:
    if not text:
        return None
    m = _DUR_RE.search(str(text))
    if not m:
        return None
    u = m.group(2).lower()
    key = "mo" if u.startswith("mo") else ("m" if u.startswith("min") else u[0])
    return float(m.group(1)) * _DUR_TO_DAYS.get(key, 1)


# ----------------------------------------------------------------------
# STUDY DESIGN CUES
# ----------------------------------------------------------------------

DESIGN_CUES: Dict[str, str] = {
    "randomized": r"\brandomi[sz]ed\b|\brandomly (assigned|allocated)\b|random number generator",
    "blinded": r"\b(double|single)[- ]blind|\bblind(ed)? (to|assessment|outcome)|\bmasked\b|\bblind to\b",
    "controlled": r"\bcontrol (group|plants|animals|samples|condition|treatment)|\bvehicle[- ]control|\bplacebo\b|"
                  r"served as (the )?control|\buntreated control|\bsham\b|wild[- ]type",
    "in_vitro": r"\bin vitro\b|cell (line|culture)|cultured cells|\bassay\b in cells",
    "in_vivo": r"\bin vivo\b|\bmice\b|\brats\b|\bplants\b|\bseedlings\b|zebrafish|drosophila|\bpatients\b",
    "cohort": r"\bcohort\b|prospective(ly)? (study|followed)|retrospective",
    "case_control": r"case[- ]control",
    "cross_sectional": r"cross[- ]sectional|survey",
    "observational": r"\bobservational\b|epidemiolog|registry|field survey|natural(ly occurring)? variation",
    "meta_analysis": r"meta[- ]analy",
    "systematic_review": r"systematic review",
    "review": r"\breview (article|of the literature)|narrative review|we review",
    "computational": r"\bsimulat|in silico|computational (model|analysis)|machine learning model",
    "case_report": r"case report|case series",
    "correlational": r"\bcorrelat(ed|ion|ions)\b|\bassociation (between|of)\b|\bGWAS\b|genome[- ]wide association",
    "replicated": r"independent(ly)? (replicat|experiment)|replicated in|biological replicates|three independent",
    "preregistered": r"pre-?regist",
    "multiple_testing_correction": r"bonferroni|holm|benjamini|false discovery rate|\bFDR\b|tukey|dunnett",
}


def detect_design_features(text: str) -> Dict[str, bool]:
    t = text or ""
    return {k: bool(re.search(p, t, flags=re.IGNORECASE)) for k, p in DESIGN_CUES.items()}


# ----------------------------------------------------------------------
# QUOTE GROUNDING
# ----------------------------------------------------------------------

def _norm_for_match(s: str) -> str:
    s = normalize_numeric_text(s or "").lower()
    s = re.sub(r"[\u2018\u2019\u201c\u201d\"'`]", "", s)
    s = re.sub(r"[^a-z0-9<>=.%\- ]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def quote_grounding(quote: str, source_text: str) -> float:
    """
    1.0 if `quote` appears (after normalisation) in the source; otherwise the
    fraction of the quote's word-trigrams that occur in the source. Used to
    reject claims whose supporting 'quote' the model made up.
    """
    q, s = _norm_for_match(quote), _norm_for_match(source_text)
    if not q or not s:
        return 0.0
    if q in s:
        return 1.0
    words = q.split()
    if len(words) < 3:
        return 0.0
    tri = [" ".join(words[i:i + 3]) for i in range(len(words) - 2)]
    return sum(1 for t in tri if t in s) / len(tri)


def numbers_in_text(text: str) -> List[float]:
    return [float(x) for x in re.findall(_NUM, normalize_numeric_text(text or ""))]


def number_in_text(value: Any, text: str, rel_tol: float = 1e-6) -> bool:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return False
    return any(abs(n - v) <= max(rel_tol * abs(v), 1e-9) or abs(abs(n) - abs(v)) <= 1e-9
               for n in numbers_in_text(text))
