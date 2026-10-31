"""
risk_numbers.py  (v26)
======================
Deterministic arithmetic checks that a language model easily gets wrong:

  events(text)            '12 of 100 (12%) vs 20 of 100 (20%)', '12/100 vs 20/100', 'in 12 (12.0%) and 20 (20.0%) of
                          100 patients each' -> per-arm events and denominators
  risk_table(a, n1, b, n2) relative risk, absolute risk difference (percentage points), NNT / NNH, odds ratio, with a
                          Wald / Newcombe-style 95% interval for the difference
  check_percentages(text) every 'x of n (p%)' and 'x/n (p%)' is recomputed; a stated percentage that is off by more
                          than rounding is a conflict
  check_change(text)      'fell from 8.0 to 7.2 (a 10% reduction)' -> the stated relative change is recomputed
  framing(statement, ...) a RELATIVE reduction presented without its ABSOLUTE size, and 'increased by 5%' when the
                          rate went from 10% to 15% (5 percentage points = a 50% relative increase)
  meaningful(...)         statistical vs biological / clinical significance: standardised effect size (Cohen's d) when
                          an SD or an interval and n are available; ratio effects near 1; minimal important differences
                          from mcid.json (you add the ones your field uses - none are invented here)
"""
from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

RISK_VERSION = "1.0"
ROOT = Path(__file__).resolve().parent
_NUM = r"(\d{1,7}(?:,\d{3})*)"
_PCT = r"(\d{1,3}(?:\.\d+)?)\s*%"
_OF = re.compile(_NUM + r"\s*(?:of|/|out of)\s*" + _NUM + r"(?:\s*(?:patients|participants|subjects|women|men|mice|animals))?"
                 r"\s*\(\s*" + _PCT + r"\s*\)", re.I)
_OF_NOPCT = re.compile(_NUM + r"\s*(?:of|/|out of)\s*" + _NUM + r"\b", re.I)
_CHANGE = re.compile(r"(?:from|decreased from|increased from|fell from|rose from|reduced from)\s+(-?\d+(?:\.\d+)?)\s*%?\s*"
                     r"(?:[a-zA-Z/µ]+\s*)?to\s+(-?\d+(?:\.\d+)?)\s*%?[^.;]{0,40}?\(?\s*(?:an?|the)?\s*"
                     r"(\d+(?:\.\d+)?)\s*%\s*(reduction|decrease|increase|rise|fall|drop|improvement|decline)", re.I)


def _n(x: str) -> int:
    return int(x.replace(",", ""))


def risk_table(a: int, n1: int, b: int, n2: int) -> Optional[Dict[str, Any]]:
    """a / n1 = events / total in the treated arm, b / n2 in the comparator arm."""
    if min(n1, n2) <= 0 or a < 0 or b < 0 or a > n1 or b > n2:
        return None
    p1, p2 = a / n1, b / n2
    rd = p1 - p2
    se = math.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
    out: Dict[str, Any] = {"risk_treated": round(p1, 4), "risk_control": round(p2, 4),
                           "risk_difference_pp": round(100 * rd, 2),
                           "risk_difference_ci_pp": [round(100 * (rd - 1.96 * se), 2), round(100 * (rd + 1.96 * se), 2)]}
    if p2 > 0:
        rr = p1 / p2
        out["relative_risk"] = round(rr, 3)
        out["relative_change_pct"] = round(100 * (rr - 1), 1)
        if a > 0 and b > 0:
            se_l = math.sqrt(1 / a - 1 / n1 + 1 / b - 1 / n2)
            out["relative_risk_ci"] = [round(math.exp(math.log(rr) - 1.96 * se_l), 3),
                                       round(math.exp(math.log(rr) + 1.96 * se_l), 3)]
    if a not in (0, n1) and b not in (0, n2):
        out["odds_ratio"] = round((a / (n1 - a)) / (b / (n2 - b)), 3)
    if rd != 0:
        nn = 1 / abs(rd)
        out["nnt" if rd < 0 else "nnh"] = math.ceil(nn - 1e-9)
        lo, hi = out["risk_difference_ci_pp"]
        if lo < 0 < hi:
            out["nnt_note"] = "the interval for the difference includes 0: the NNT interval runs through infinity"
    return out


def events(text: str) -> List[Dict[str, Any]]:
    """Pairs of arms 'x of n' found in one sentence (the first two)."""
    out = []
    for sent in re.split(r"(?<=[.;])\s+", text or ""):
        hits = [(_n(m.group(1)), _n(m.group(2))) for m in _OF_NOPCT.finditer(sent)]
        hits = [h for h in hits if 0 <= h[0] <= h[1] and h[1] >= 5]
        if len(hits) >= 2 and re.search(r"\bvs\.?|\bversus\b|\bcompared|\band\b|\bplacebo\b|\bcontrol", sent, re.I):
            (a, n1), (b, n2) = hits[0], hits[1]
            t = risk_table(a, n1, b, n2)
            if t:
                out.append({"sentence": sent.strip()[:300], "treated": [a, n1], "control": [b, n2], **t})
    return out


def check_percentages(text: str) -> List[Dict[str, Any]]:
    """'x of n (p%)' recomputed. A difference larger than rounding to the stated decimals (+ 0.1 pp) is a conflict."""
    out = []
    for m in _OF.finditer(text or ""):
        x, n, p = _n(m.group(1)), _n(m.group(2)), float(m.group(3))
        if n == 0 or x > n:
            continue
        dec = len(m.group(3).split(".")[1]) if "." in m.group(3) else 0
        calc = 100 * x / n
        tol = 0.5 * 10 ** (-dec) + 0.1
        out.append({"text": m.group(0), "stated_pct": p, "computed_pct": round(calc, 2),
                    "ok": abs(calc - p) <= tol})
    return out


def check_change(text: str) -> List[Dict[str, Any]]:
    out = []
    for m in _CHANGE.finditer(text or ""):
        a, b, stated, word = float(m.group(1)), float(m.group(2)), float(m.group(3)), m.group(4).lower()
        if a == 0:
            continue
        rel = 100 * (b - a) / abs(a)
        pp = b - a
        sign_ok = (rel < 0) == (word in ("reduction", "decrease", "fall", "drop", "decline")) or word == "improvement"
        ok_rel = abs(abs(rel) - stated) <= max(0.6, 0.02 * stated)
        ok_pp = abs(abs(pp) - stated) <= 0.15
        out.append({"text": m.group(0)[:200], "from": a, "to": b, "stated_pct": stated, "relative_change_pct": round(rel, 1),
                    "absolute_change": round(pp, 3), "ok": sign_ok and (ok_rel or ok_pp),
                    "note": None if ok_rel else ("the stated % is the ABSOLUTE change (percentage points), not the "
                                                 "relative change" if ok_pp else "the stated change does not match from -> to")})
    return out


_REL_CLAIM = re.compile(r"(?:reduced|lowered|cut|decreased|increased|raised)\s+(?:the\s+)?(?:risk|odds|incidence|rate|hazard)"
                        r"[^.]{0,40}?\bby\s+(\d+(?:\.\d+)?)\s*%", re.I)
_PP_AMBIG = re.compile(r"(?:increased|rose|decreased|fell|dropped|declined)\s+(?:by\s+)?(\d+(?:\.\d+)?)\s*%\s*(?:from\s+)?"
                       r"[^.]{0,30}?\bfrom\s+(\d+(?:\.\d+)?)\s*%\s+to\s+(\d+(?:\.\d+)?)\s*%", re.I)


def framing(text: str) -> List[Dict[str, Any]]:
    """Relative-vs-absolute presentation problems in a report or a statement (with the paper's counts when known)."""
    out = []
    ev = events(text)
    for m in _REL_CLAIM.finditer(text or ""):
        rel = float(m.group(1))
        near = next((e for e in ev if abs(abs(e.get("relative_change_pct") or 0) - rel) <= 2), None)
        if near:
            out.append({"text": m.group(0), "kind": "relative only", "relative_pct": rel,
                        "absolute_pp": near["risk_difference_pp"], "nnt": near.get("nnt") or near.get("nnh"),
                        "advice": f"state the absolute size too: {abs(near['risk_difference_pp'])} percentage points "
                                  f"(NNT {near.get('nnt') or near.get('nnh')})"})
        elif not re.search(r"absolute|percentage points?|\bNNT\b|number needed", text or "", re.I):
            out.append({"text": m.group(0), "kind": "relative only", "relative_pct": rel, "absolute_pp": None,
                        "advice": "a relative reduction is given without the absolute risks - its size for patients cannot be judged"})
    for m in _PP_AMBIG.finditer(text or ""):
        stated, a, b = float(m.group(1)), float(m.group(2)), float(m.group(3))
        pp, rel = b - a, 100 * (b - a) / a if a else None
        if abs(abs(pp) - stated) <= 0.15 and rel is not None and abs(abs(rel) - stated) > 1:
            out.append({"text": m.group(0), "kind": "percent vs percentage points", "stated": stated,
                        "absolute_pp": round(pp, 2), "relative_pct": round(rel, 1),
                        "advice": f"'{stated:g}%' here is {abs(pp):g} percentage points = a {abs(rel):.0f}% relative change"})
    return out


# ------------------------------------------------------------------------------------------------ meaningful effects
def load_mcid() -> Dict[str, Any]:
    p = ROOT / "mcid.json"
    try:
        return {k.lower(): v for k, v in (json.loads(p.read_text(encoding="utf-8")).get("endpoints") or {}).items()}
    except Exception:
        return {}


def cohens_d(diff: float, sd: Optional[float] = None, ci: Optional[List[float]] = None, n1: Optional[int] = None,
             n2: Optional[int] = None) -> Optional[float]:
    """Standardised mean difference. From an SD, or back-calculated from a 95% interval and the group sizes."""
    if sd and sd > 0:
        return diff / sd
    if ci and n1 and n2 and n1 > 1 and n2 > 1:
        se = (ci[1] - ci[0]) / (2 * 1.96)
        if se > 0:
            sd_pooled = se / math.sqrt(1 / n1 + 1 / n2)
            return diff / sd_pooled
    return None


def meaningful(effect: Dict[str, Any], endpoint: Optional[str] = None, significant: Optional[bool] = None,
               sd: Optional[float] = None, n1: Optional[int] = None, n2: Optional[int] = None) -> Dict[str, Any]:
    """{'statistical': 'significant'|'not significant'|'unknown', 'magnitude': 'trivial'|'small'|'moderate'|'large'|
    'unknown', 'basis', 'meets_mcid': True|False|None}. A significant but trivial effect is reported as such."""
    kind, est = effect.get("kind"), effect.get("estimate")
    lo, hi = effect.get("lo"), effect.get("hi")
    if significant is None and lo is not None and hi is not None:
        null = 1.0 if kind == "ratio" else 0.0
        significant = not (lo <= null <= hi)
    out: Dict[str, Any] = {"statistical": "unknown" if significant is None else ("significant" if significant else "not significant"),
                           "magnitude": "unknown", "basis": None, "meets_mcid": None}
    if est is None:
        return out
    if kind == "ratio" and est > 0:
        r = est if est >= 1 else 1 / est
        out["magnitude"] = "trivial" if r < 1.1 else "small" if r < 1.5 else "moderate" if r < 2.5 else "large"
        out["basis"] = (f"ratio {est:g} ({100 * abs(1 - est):.0f}% relative change); BioSynthetica heuristic size bands "
                        f"1.1 / 1.5 / 2.5 (not a published standard - use mcid.json for your endpoints)")
    else:
        d = cohens_d(float(est), sd, [lo, hi] if lo is not None and hi is not None else None, n1, n2)
        if d is not None:
            ad = abs(d)
            out["magnitude"] = "trivial" if ad < 0.2 else "small" if ad < 0.5 else "moderate" if ad < 0.8 else "large"
            out["basis"] = f"standardised difference d = {d:.2f} (Cohen: 0.2 small, 0.5 moderate, 0.8 large)"
    mc = load_mcid().get((endpoint or "").lower())
    if mc and kind != "ratio" and mc.get("value") is not None:
        out["meets_mcid"] = abs(float(est)) >= float(mc["value"])
        out["mcid"] = mc
    if out["statistical"] == "significant" and (out["magnitude"] == "trivial" or out["meets_mcid"] is False):
        out["warning"] = "statistically significant but below a meaningful size - do not present it as an important effect"
    return out


def audit_text(text: Any) -> Dict[str, Any]:
    text = text if isinstance(text, str) else ("" if text is None else str(text))
    pct = check_percentages(text)
    ch = check_change(text)
    return {"events": events(text), "percentages": pct, "changes": ch, "framing": framing(text),
            "conflicts": [p for p in pct if not p["ok"]] + [c for c in ch if not c["ok"]], "version": RISK_VERSION}
