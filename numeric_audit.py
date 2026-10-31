"""
numeric_audit.py  (v25)
=======================
Unit and numerical audit extensions:

1. Tie TABLE values to the abstract and the text. For every result number the abstract states (a percentage, a mean,
   an effect with its interval, a sample size), look for the same quantity in the paper's tables and results text:
       consistent     the same value appears in a table / the results
       rounded        it appears rounded to the abstract's precision (12.47 -> 12.5) - fine
       conflict       the table / results give a DIFFERENT value for the same labelled quantity (same row label or
                      the same words around it) - flagged for a person, with both values and where they are
       not found      only the abstract states it (no table to check against)
2. Keep ORIGINAL and NORMALISED values side by side. Every measured value is parsed with its unit and converted to a
   canonical unit where a conversion is exact or standard - the original text is never replaced:
       glucose mg/dL -> mmol/L (x 0.0555), cholesterol mg/dL -> mmol/L (x 0.02586), triglycerides (x 0.01129),
       creatinine mg/dL -> umol/L (x 88.4), HbA1c % (NGSP) -> mmol/mol (IFCC: (% - 2.15) x 10.929),
       mass (ug, mg, g, kg), volume (uL, mL, L), time (min, h, day, week, month, year -> days),
       concentrations (nM, uM, mM, M), dose per body weight (mg/kg), temperature (F -> C)

    parse_values(text) -> [{'raw', 'value', 'unit', 'quantity', 'normalised', 'normalised_unit'}]
    audit_paper(abstract, tables_text, results_text) -> {'checked', 'consistent', 'rounded', 'conflicts', 'not_found'}
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

NUMERIC_VERSION = "1.0"

_UNIT = (r"(?:mg/dl|mmol/l|mmol/mol|umol/l|µmol/l|μmol/l|nmol/l|pmol/l|ng/ml|pg/ml|µg/ml|ug/ml|μg/ml|mg/l|g/dl|g/l|iu/l|u/l|"
         r"mg/kg(?:/day|/d)?|µg/kg|ug/kg|mg/day|mg/d|mg|µg|ug|μg|ng|g|kg|ml|µl|ul|μl|l|nm|µm|um|μm|mm|m|mmhg|bpm|"
         r"min|minutes?|h|hrs?|hours?|days?|d|weeks?|wk|months?|mo|years?|yr|°c|°f|kcal|%)")
_VAL = re.compile(r"(?<![\w.])(-?\d+(?:[.,]\d+)?)\s*(" + _UNIT + r")(?![a-z])", re.I)
_QUANT = [("glucose", r"glucose|glyc(?:a)?emia|fpg"), ("cholesterol", r"cholesterol|ldl|hdl"), ("triglycerides", r"triglycerid\w*|\btg\b"),
          ("creatinine", r"creatinine"), ("hba1c", r"hba1c|a1c|glycated ha?emoglobin")]


def _quantity(context: str) -> Optional[str]:
    """The analyte named CLOSEST before the value ('glucose 126 mg/dL and HbA1c 7.5%' -> HbA1c for 7.5%)."""
    best, at = None, -1
    for name, rx in _QUANT:
        for m in re.finditer(rx, context, re.I):
            if m.end() > at:
                best, at = name, m.end()
    return best


def normalise(value: float, unit: str, quantity: Optional[str] = None) -> Tuple[Optional[float], Optional[str]]:
    u = unit.lower().replace("μ", "u").replace("µ", "u")
    q = quantity
    if u == "mg/dl" and q in ("glucose", "cholesterol", "triglycerides", "creatinine"):
        f = {"glucose": 0.0555, "cholesterol": 0.02586, "triglycerides": 0.01129, "creatinine": 88.4}[q]
        return round(value * f, 3), "umol/L" if q == "creatinine" else "mmol/L"
    if u == "%" and q == "hba1c":
        return round((value - 2.15) * 10.929, 1), "mmol/mol (IFCC)"
    if u == "mmol/mol" and q == "hba1c":
        return round(value / 10.929 + 2.15, 2), "% (NGSP)"
    mass = {"ng": 1e-6, "ug": 1e-3, "mg": 1.0, "g": 1e3, "kg": 1e6}
    if u in mass:
        return round(value * mass[u], 6), "mg"
    vol = {"ul": 1e-3, "ml": 1.0, "l": 1e3}
    if u in vol:
        return round(value * vol[u], 6), "mL"
    days = {"min": 1 / 1440, "minute": 1 / 1440, "minutes": 1 / 1440, "h": 1 / 24, "hr": 1 / 24, "hrs": 1 / 24,
            "hour": 1 / 24, "hours": 1 / 24, "d": 1, "day": 1, "days": 1, "week": 7, "weeks": 7, "wk": 7, "month": 30.44,
            "months": 30.44, "mo": 30.44, "year": 365.25, "years": 365.25, "yr": 365.25}
    if u in days:
        return round(value * days[u], 4), "days"
    conc = {"nm": 1e-3, "um": 1.0, "mm": 1e3}
    if u in conc and q is None:
        return round(value * conc[u], 6), "uM"
    if u == "°f":
        return round((value - 32) * 5 / 9, 2), "°C"
    if u in ("ug/kg",):
        return round(value / 1000, 6), "mg/kg"
    return value, unit


def parse_values(text: str) -> List[Dict[str, Any]]:
    t = text or ""
    out = []
    for m in _VAL.finditer(t):
        raw_v = m.group(1).replace(",", ".") if re.match(r"^-?\d+,\d{1,2}$", m.group(1)) else m.group(1).replace(",", "")
        try:
            v = float(raw_v)
        except ValueError:
            continue
        unit = m.group(2)
        ctx = t[max(0, m.start() - 60):m.start()]
        q = _quantity(ctx)
        nv, nu = normalise(v, unit, q)
        out.append({"raw": m.group(0), "value": v, "unit": unit, "quantity": q, "normalised": nv, "normalised_unit": nu,
                    "context": ctx.strip()[-60:], "pos": m.start()})
    return out


def _numbers(text: str) -> List[Tuple[float, int, int]]:
    """(value, decimals, position) of every plain number."""
    out = []
    for m in re.finditer(r"(?<![\w.])(-?\d+(?:\.\d+)?)(?![\w.])", text or ""):
        s = m.group(1)
        out.append((float(s), len(s.split(".")[1]) if "." in s else 0, m.start()))
    return out


_GENERIC = {"with", "were", "that", "from", "than", "this", "group", "groups", "patients", "versus", "compared", "mean",
            "median", "participants", "subjects", "women", "study", "trial", "total", "overall", "respectively", "both",
            "after", "before", "during", "each", "among", "between", "values", "value", "levels", "level"}


def _label_words(text: str, pos: int, end: Optional[int] = None) -> set:
    """v26.2: the words of the number's OWN clause, on both sides ('2104 (57.7%) were luminal' -> {luminal}),
    stopping at a clause delimiter or at the next / previous number - the 70 characters before a number used to pull
    in the labels of its neighbours in a list and produced false 'abstract vs table' conflicts."""
    end = pos if end is None else end
    left = text[max(0, pos - 80):pos]
    cut = max(left.rfind(c) for c in ";.:")
    m = list(re.finditer(r"\d", left))
    if m:
        cut = max(cut, m[-1].end())
    left = left[cut + 1:] if cut >= 0 else left
    right = text[end:end + 60]
    stop = min([i for i in (right.find(";"), right.find(". "), right.find(":")) if i >= 0] + [len(right)])
    nxt = re.search(r"\d", right[2:])
    if nxt:
        stop = min(stop, nxt.start() + 2)
    seg = (left + " " + right[:stop]).lower()
    return {w for w in re.findall(r"[a-z][a-z\-]{3,}", seg)} - _GENERIC


def _raw_at(text: str, pos: int) -> str:
    m = re.match(r"-?\d+(?:\.\d+)?", text[pos:])
    return m.group(0) if m else ""


def _is_pct(text: str, pos: int, raw: str) -> bool:
    return bool(re.match(r"\s*%", text[pos + len(raw):pos + len(raw) + 3]))


def audit_paper(abstract: str, tables_text: str = "", results_text: str = "") -> Dict[str, Any]:
    """Tie the abstract's result numbers to the tables / results text."""
    ab = abstract or ""
    pool = f"{tables_text or ''}\n{results_text or ''}"
    if not pool.strip():
        return {"checked": 0, "consistent": 0, "rounded": 0, "conflicts": [], "not_found": 0,
                "note": "no table / results text to check against (abstract only)", "version": NUMERIC_VERSION}
    pool_nums = _numbers(pool)
    res = {"checked": 0, "consistent": 0, "rounded": 0, "conflicts": [], "not_found": 0}
    # words that occur everywhere in this paper ('breast', 'cancer', 'grade') identify nothing: not a shared label
    from collections import Counter
    freq = Counter(re.findall(r"[a-z][a-z\-]{3,}", (ab + " " + pool).lower()))
    generic = {w for w, n in freq.items() if n >= 6}
    for v, dec, pos in _numbers(ab):
        if dec == 0 and v < 10 or 1900 <= v <= 2100:          # small counts and years carry no checkable result
            continue
        if re.match(r"\s*(?:%\s*)?(?:ci|confidence)", ab[pos + len(str(v)):pos + len(str(v)) + 14], re.I):
            continue
        res["checked"] += 1
        if any(abs(abs(pv) - abs(v)) < 1e-9 for pv, _, _ in pool_nums):
            res["consistent"] += 1
            continue
        if dec < 3 and any(round(abs(pv), dec) == round(abs(v), dec) for pv, pd, _ in pool_nums if pd > dec):
            res["rounded"] += 1
            continue
        raw = _raw_at(ab, pos)
        words = _label_words(ab, pos, pos + len(raw)) - generic
        clash = None
        if len(words) >= 2:
            for pv, pd, pp in pool_nums:
                praw = _raw_at(pool, pp)
                w2 = _label_words(pool, pp, pp + len(praw)) - generic
                overlap = words & w2
                if len(overlap) >= 2 and abs(pv - v) > max(0.06 * abs(v), 10 ** -dec) and \
                        (pv != 0 and 0.5 < abs(v / pv) < 2):
                    clash = {"abstract_value": v, "other_value": pv, "shared_label": sorted(overlap)[:4],
                             "abstract_context": ab[max(0, pos - 80):pos + 20].strip(),
                             "other_context": pool[max(0, pp - 80):pp + 20].strip()}
                    break
        if clash:
            res["conflicts"].append(clash)
        else:
            res["not_found"] += 1
    res["version"] = NUMERIC_VERSION
    return res


def side_by_side(text: str, limit: int = 30) -> List[Dict[str, Any]]:
    """Original and normalised values, for the report's appendix (only where a conversion changed something)."""
    out = []
    for x in parse_values(text):
        if x["normalised"] is not None and (x["normalised"] != x["value"] or x["normalised_unit"] != x["unit"]):
            out.append({k: x[k] for k in ("raw", "quantity", "normalised", "normalised_unit")})
        if len(out) >= limit:
            break
    return out


def markdown(audits: List[Dict[str, Any]]) -> str:
    rows = [a for a in audits if a.get("audit", {}).get("checked")]
    if not rows:
        return ""
    L = ["## Numerical audit (abstract vs tables / results)", "", "| study | checked | consistent | rounded | conflicts | only in abstract |",
         "|---|---|---|---|---|---|"]
    for a in rows:
        x = a["audit"]
        L.append(f"| {a['ref']} | {x['checked']} | {x['consistent']} | {x['rounded']} | **{len(x['conflicts'])}** | {x['not_found']} |")
    for a in rows:
        for c in a["audit"]["conflicts"][:3]:
            L.append(f"- {a['ref']}: abstract {c['abstract_value']:g} vs {c['other_value']:g} for '{' '.join(c['shared_label'])}' "
                     f"(abstract: \"…{c['abstract_context'][-70:]}\"; table/text: \"…{c['other_context'][-70:]}\") - check the paper")
    return "\n".join(L) + "\n"
