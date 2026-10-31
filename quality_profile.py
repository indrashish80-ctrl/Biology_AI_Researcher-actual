"""
quality_profile.py  (v26)
=========================
A QUALITY PROFILE per study instead of one score, kept apart from RELEVANCE (a rigorous study can still be irrelevant
to the question, and a relevant one can be weak). Read by code from the paper; every rating gives its reason.

    profile(row, text, question, library_record=None) -> {
        'risk_of_bias':        RoB 2 / ROBINS-I / SYRCLE overall + tool          (risk_of_bias.py)
        'design_fit':          does the design suit the question type?            appropriate | acceptable | weak
        'controls':            placebo / vehicle / sham / active control, blinded assessment, validated measures
        'precision':           sample size, interval width                         adequate | limited | poor | unknown
        'missing_data':        attrition %, intention-to-treat                     low | some | high | not reported
        'selective_reporting': registered? outcome switched?                       low | some concerns | high | unknown
        'reproducibility':     data / code / protocol availability statements
        'applicability':       model and population vs the question                direct | indirect | not applicable
        'funding_conflicts':   funders, industry funding, COI (study_funding.py)
        'duration':            treatment / follow-up length as written
        'relevance':           reported SEPARATELY (the screener's decision), never folded into quality
    }
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

PROFILE_VERSION = "1.0"

_DURATION = re.compile(
    r"(?:for|over|during|after|lasting|treatment (?:period|duration) of|follow[- ]up (?:of|was|period of)?|"
    r"median follow[- ]up (?:of|was)?)\s*(?:a (?:median|mean) of\s*)?(\d+(?:\.\d+)?)\s*[- ]?\s*"
    r"(hours?|h|days?|d|weeks?|wk|months?|mo|years?|yr)\b|(\d+(?:\.\d+)?)[- ](hour|day|week|month|year)s?[- ]"
    r"(?:trial|study|treatment|intervention|follow[- ]up|period|course)", re.I)
_UNIT_DAYS = {"h": 1 / 24, "hour": 1 / 24, "d": 1, "day": 1, "wk": 7, "week": 7, "mo": 30.44, "month": 30.44, "yr": 365.25,
              "year": 365.25}
_ATTRITION = re.compile(r"(\d+(?:\.\d+)?)\s*%\s*(?:of (?:participants|patients|subjects) )?(?:were )?(?:lost to follow[- ]up|"
                        r"dropped out|withdrew|discontinued|did not complete)|(?:lost to follow[- ]up|drop[- ]?outs?|attrition|"
                        r"withdrawals?)\s*(?:was|were|of|:)?\s*(\d+(?:\.\d+)?)\s*%", re.I)


def duration(text: str) -> Optional[Dict[str, Any]]:
    best = None
    for m in _DURATION.finditer(text or ""):
        num = float(m.group(1) or m.group(3))
        unit = (m.group(2) or m.group(4) or "").lower().rstrip("s")
        unit = {"hours": "hour"}.get(unit, unit)
        days = num * _UNIT_DAYS.get(unit, _UNIT_DAYS.get(unit[:2], 1))
        if days <= 0 or days > 365.25 * 60:
            continue
        cand = {"as_written": m.group(0).strip(), "days": round(days, 2)}
        if best is None or days > best["days"]:
            best = cand
    return best


def _design_fit(question_type: Optional[str], design: Optional[str]) -> Dict[str, str]:
    d = design or "unclear"
    q = question_type or "therapy"
    table = {
        "therapy": {"human_clinical": "appropriate", "review_meta": "appropriate", "human_observational": "acceptable",
                    "animal": "weak", "cell_culture": "weak", "organoid_ex_vivo": "weak", "in_silico": "weak"},
        "etiology": {"human_observational": "appropriate", "review_meta": "appropriate", "human_clinical": "appropriate",
                     "animal": "acceptable", "cell_culture": "weak"},
        "prognosis": {"human_observational": "appropriate", "human_clinical": "acceptable", "review_meta": "appropriate"},
        "diagnosis": {"human_observational": "appropriate", "human_clinical": "acceptable", "review_meta": "appropriate"},
        "mechanism": {"cell_culture": "appropriate", "animal": "appropriate", "organoid_ex_vivo": "appropriate",
                      "human_clinical": "acceptable", "human_observational": "acceptable", "in_silico": "acceptable"},
    }
    fit = table.get(q, table["therapy"]).get(d, "weak" if d != "unclear" else "unknown")
    return {"rating": fit, "reason": f"{d.replace('_', ' ')} study for a {q} question"}


def _controls(text: str) -> Dict[str, Any]:
    t = text or ""
    found = [k for k, rx in (("placebo", r"\bplacebo\b"), ("vehicle", r"\bvehicle\b"), ("sham", r"\bsham\b"),
                             ("active comparator", r"\b(?:active comparator|standard (?:of )?care|usual care)\b"),
                             ("untreated control", r"\buntreated (?:controls?|cells|mice)\b"),
                             ("blinded assessment", r"\b(?:blind(?:ed)?|masked)\b[^.]{0,40}\b(?:assess|outcome|evaluat|patholog)"),
                             ("double-blind", r"\bdouble[- ]blind"), ("validated measure", r"\bvalidated\b"))
             if re.search(rx, t, re.I)]
    rating = "adequate" if any(k in found for k in ("placebo", "vehicle", "sham", "active comparator", "untreated control")) \
        and any(k in found for k in ("blinded assessment", "double-blind")) else \
        "partial" if found else "not reported"
    return {"rating": rating, "found": found}


def _precision(row: Dict[str, Any], text: str) -> Dict[str, Any]:
    n = None
    for f in row.get("findings") or []:
        n = n or f.get("sample_size")
    if n is None:
        m = re.search(r"\b[nN]\s*=\s*(\d{1,6})\b|(\d{2,6})\s+(?:patients|participants|women|men|subjects|mice|rats)\b", text or "")
        if m:
            n = int(m.group(1) or m.group(2))
    widths = []
    for f in row.get("findings") or []:
        ci = f.get("confidence_interval")
        if isinstance(ci, (list, tuple)) and len(ci) == 2 and all(isinstance(x, (int, float)) for x in ci):
            lo, hi = ci
            if lo > 0 and hi > 0 and (f.get("effect_kind") == "ratio" or (lo < 5 and hi < 20)):
                widths.append(hi / lo)
    ratio_w = max(widths) if widths else None
    if n is None and ratio_w is None:
        return {"rating": "unknown", "n": None, "ci_ratio_width": None}
    rating = "adequate"
    reasons = []
    if n is not None and n < 30:
        rating, _ = "poor", reasons.append(f"n = {n}")
    elif n is not None and n < 100:
        rating, _ = "limited", reasons.append(f"n = {n}")
    if ratio_w is not None and ratio_w > 3:
        rating, _ = "poor", reasons.append(f"interval spans a {ratio_w:.1f}-fold range")
    elif ratio_w is not None and ratio_w > 2 and rating == "adequate":
        rating, _ = "limited", reasons.append(f"interval spans a {ratio_w:.1f}-fold range")
    return {"rating": rating, "n": n, "ci_ratio_width": round(ratio_w, 2) if ratio_w else None, "reasons": reasons}


def _missing(text: str) -> Dict[str, Any]:
    vals = [float(m.group(1) or m.group(2)) for m in _ATTRITION.finditer(text or "")]
    itt = bool(re.search(r"intention[- ]to[- ]treat|\bITT\b", text or "", re.I))
    if not vals:
        return {"rating": "not reported", "attrition_pct": None, "itt": itt}
    a = max(vals)
    return {"rating": "low" if a < 10 else "some" if a < 20 else "high", "attrition_pct": a, "itt": itt}


def _selective(row: Dict[str, Any], text: str, switched: bool) -> Dict[str, Any]:
    reg = re.findall(r"\b(NCT\d{8}|ISRCTN\d{6,}|EudraCT[\s:]*\d{4}-\d{6}-\d{2}|ACTRN\d{14}|ChiCTR[\w-]+|CTRI/\d{4}/\d+/\d+)\b",
                     text or "")
    if switched:
        return {"rating": "high", "registered": True, "registries": sorted(set(reg)),
                "reason": "the published primary outcome differs from the registered one"}
    if reg:
        return {"rating": "low", "registered": True, "registries": sorted(set(reg))}
    if row.get("evidence_type") == "human_clinical":
        return {"rating": "some concerns", "registered": False, "reason": "no trial registration found in the text"}
    return {"rating": "unknown", "registered": False}


def _repro(text: str) -> Dict[str, Any]:
    t = text or ""
    found = [k for k, rx in (("data available", r"data (?:availability|are available|is available|have been deposited|were deposited)|"
                                                r"\b(?:GSE\d{3,}|PRJNA\d+|E-MTAB-\d+|zenodo|figshare|dryad)\b"),
                             ("code available", r"\b(?:code|scripts?) (?:is |are )?(?:available|deposited)|github\.com"),
                             ("protocol available", r"\bprotocol (?:is |was )?(?:available|published|registered)\b"),
                             ("data on request", r"available (?:from the (?:corresponding )?authors? )?(?:up)?on (?:reasonable )?request"))
             if re.search(rx, t, re.I)]
    rating = "good" if any(k in found for k in ("data available", "code available")) else \
        "limited" if found else "not reported"
    return {"rating": rating, "found": found}


def _applicability(row: Dict[str, Any], frame: Dict[str, Any]) -> Dict[str, str]:
    et = row.get("evidence_type") or "unclear"
    human_q = frame.get("domain") == "human_medicine"
    if human_q and et in ("human_clinical", "human_observational", "review_meta"):
        return {"rating": "direct", "reason": "human study for a question about people"}
    if human_q and et in ("animal", "cell_culture", "organoid_ex_vivo", "in_silico", "experimental_disease_model"):
        return {"rating": "indirect", "reason": f"{et.replace('_', ' ')} evidence for a question about people"}
    return {"rating": "direct" if et != "unclear" else "unknown", "reason": f"{et.replace('_', ' ')} study"}


def profile(row: Dict[str, Any], text: str, question: str, library_record: Optional[Dict[str, Any]] = None,
            switched: bool = False) -> Dict[str, Any]:
    import study_funding as sf
    from evidence_identity import question_frame
    text = text if isinstance(text, str) else ("" if text is None else str(text))
    try:
        import query_parser as qp
        qtype = qp.parse(question).get("question_type")
    except Exception:
        qtype = None
    frame = question_frame(question)
    rob = row.get("risk_of_bias_detail") or {}
    if library_record and (library_record.get("funding") or library_record.get("coi")):
        fund, coi = library_record.get("funding"), library_record.get("coi")
    else:
        ft = sf.from_text(text or "")
        fund, coi = ft["funding"], ft["coi"]
    rel = row.get("paper_relevance") or {}
    return {"risk_of_bias": {"overall": rob.get("overall", "not assessed"), "tool": rob.get("tool")},
            "design_fit": _design_fit(qtype, row.get("evidence_type")),
            "controls": _controls(text),
            "precision": _precision(row, text),
            "missing_data": _missing(text),
            "selective_reporting": _selective(row, text, switched),
            "reproducibility": _repro(text),
            "applicability": _applicability(row, frame),
            "funding_conflicts": sf.profile(fund if isinstance(fund, dict) else None, coi,
                                            row.get("intervention"), row.get("evidence_type")),
            "duration": duration(text),
            "relevance": {"ok": rel.get("ok"), "reason": rel.get("reason"),
                          "note": "relevance is reported separately and is never part of the quality profile"},
            "version": PROFILE_VERSION}


def markdown(rows: List[Dict[str, Any]]) -> str:
    L = ["## Study quality profiles (separate from relevance)", "",
         "| study | risk of bias | design fit | controls | precision | missing data | selective reporting | "
         "reproducibility | applicability | funding / COI | duration |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        q = r.get("quality_profile")
        if not q:
            continue
        fc = q["funding_conflicts"]
        fund = ("industry" if fc["industry_funded"] else "non-industry" if fc["industry_funded"] is False else "not reported") + \
            f"; COI {fc['coi']}"
        L.append(f"| {r.get('ref')} | {q['risk_of_bias']['overall']} | {q['design_fit']['rating']} | {q['controls']['rating']} | "
                 f"{q['precision']['rating']} | {q['missing_data']['rating']} | {q['selective_reporting']['rating']} | "
                 f"{q['reproducibility']['rating']} | {q['applicability']['rating']} | {fund} | "
                 f"{(q.get('duration') or {}).get('as_written') or 'not reported'} |")
    return "\n".join(L) + "\n" if len(L) > 4 else ""
