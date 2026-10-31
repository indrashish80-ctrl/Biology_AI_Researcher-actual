"""
dose_plausibility.py  (v26)
===========================
Is a result that is real inside its experiment plausibly applicable to a living organism / a patient? Extends the
existing dose check (evidence_quality.dose_relevance: cell concentration vs human plasma level, animal mg/kg ->
human-equivalent dose with the FDA body-surface-area Km factors) with the other translation questions:

    assess(row, text, question, human_route=None) -> {
        'dose':        the existing dose_relevance result (physiological / borderline / supra-physiological ...),
        'route':       route of the study (oral, intravenous, intraperitoneal, intratumoral, topical, in vitro, ...)
                       vs the route people use, and whether the difference matters (bypasses absorption / first pass),
        'duration':    exposure duration of the study vs the question's timescale (acute assay vs chronic use),
        'species':     species named, and that metabolism / target biology can differ (never assumed equal),
        'outcome_link':does the measured outcome correspond to the real-world effect asked about?
                       (direct clinical / surrogate / mechanistic marker),
        'verdict':     'plausible' | 'uncertain' | 'poorly supported' with every reason,
    }
Unusual results are FLAGGED, never rejected.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

DOSE_VERSION = "1.0"
ROUTES = [("in vitro", r"\bin vitro\b|cultured|cell lines?|\bcells were (?:treated|exposed|incubated)"),
          ("intratumoral", r"intra-?tumou?ral"), ("intraperitoneal", r"intraperitoneal|\bi\.p\.|\bIP injection"),
          ("intravenous", r"intravenous|\bi\.v\.|infusion"), ("subcutaneous", r"subcutaneous|\bs\.c\."),
          ("intramuscular", r"intramuscular|\bi\.m\."), ("topical", r"topical|cream|ointment|applied to the skin"),
          ("inhaled", r"inhal|nebuli"), ("oral", r"\boral(?:ly)?\b|by mouth|gavage|drinking water|in the diet|tablets?|capsules?"),
          ("intrathecal", r"intrathecal")]
_OUTCOME_LEVEL = [("clinical", r"survival|mortality|death|recurrence|relapse|remission|cure|stroke|myocardial infarction|"
                               r"fracture|hospitali[sz]|quality of life|pain|disability|incidence"),
                  ("surrogate", r"HbA1c|glucose|blood pressure|LDL|cholesterol|tumou?r (?:size|volume)|response rate|viral load|"
                                r"bone mineral density|ejection fraction|weight|BMI|CRP"),
                  ("mechanistic marker", r"Ki-?67|apoptosis|proliferation|phosphorylat|expression|AMPK|mTOR|caspase|"
                                         r"viability|migration|invasion|cytokine|IL-\d+|TNF")]


def route_of(text: str) -> List[str]:
    return [name for name, rx in ROUTES if re.search(rx, text or "", re.I)]


def outcome_level(outcome: str) -> str:
    for lvl, rx in _OUTCOME_LEVEL:
        if re.search(rx, outcome or "", re.I):
            return lvl
    return "unclear"


def assess(row: Dict[str, Any], text: str, question: str, human_route: Optional[str] = None) -> Dict[str, Any]:
    from quality_profile import duration
    et = row.get("evidence_type") or "unclear"
    reasons: List[str] = []
    dose = row.get("dose_relevance") or {}
    cat = dose.get("category")
    score = 0
    if cat == "supra-physiological":
        score += 2
        reasons.append(f"dose: {dose.get('statement')}")
    elif cat == "borderline":
        score += 1
        reasons.append(f"dose borderline: {dose.get('statement')}")
    routes = route_of(text)
    if et in ("cell_culture", "organoid_ex_vivo") and "in vitro" not in routes:
        routes = ["in vitro"] + routes
    hroute = human_route or ("oral" if re.search(r"\b(?:metformin|aspirin|statin|tablet|oral)\b", question or "", re.I) else None)
    route_note = None
    if hroute and routes and et not in ("human_clinical", "human_observational"):
        main = routes[0]
        if main in ("intraperitoneal", "intratumoral", "intravenous", "intrathecal") and hroute == "oral":
            route_note = (f"{main} administration bypasses oral absorption and first-pass metabolism - exposure at the target "
                          f"can be much higher than after oral use in people")
            score += 1
        elif main == "in vitro":
            route_note = "direct exposure of cells - no absorption, distribution or metabolism"
    if route_note:
        reasons.append(route_note)
    dur = duration(text)
    long_q = bool(re.search(r"\b(?:long[- ]term|chronic|years?|prevent|incidence|survival|risk of)\b", question or "", re.I))
    dur_note = None
    if dur and long_q and dur["days"] < 7:
        dur_note = f"acute exposure ({dur['as_written']}) for a question about long-term / chronic effects"
        score += 1
    if dur_note:
        reasons.append(dur_note)
    species = None
    m = re.search(r"\b(mice|mouse|rats?|rabbits?|dogs?|pigs?|monkeys?|macaques?|zebrafish|drosophila|yeast|human)\b", text or "", re.I)
    if m and et in ("animal", "experimental_disease_model"):
        species = m.group(1).lower()
        reasons.append(f"species: {species} - drug metabolism, target expression and disease biology can differ from humans")
    measured = " ".join(str(f.get("endpoint") or f.get("outcome") or "") for f in row.get("findings") or []) or (row.get("outcome") or "")
    asked_lvl = outcome_level(question or "")
    got_lvl = outcome_level(measured)
    link = {"measured": got_lvl, "asked": asked_lvl}
    if asked_lvl == "clinical" and got_lvl in ("surrogate", "mechanistic marker"):
        link["note"] = f"the study measured a {got_lvl} ({measured[:60]}), the question asks about a clinical outcome"
        score += 1
        reasons.append(link["note"])
    verdict = "plausible" if score == 0 else "uncertain" if score <= 2 else "poorly supported"
    return {"dose": {"category": cat, "statement": dose.get("statement")}, "route": {"study": routes, "people": hroute,
            "note": route_note}, "duration": {"study": dur, "question_long_term": long_q, "note": dur_note},
            "species": species, "outcome_link": link, "verdict": verdict, "reasons": reasons, "version": DOSE_VERSION}


def markdown(rows: List[Dict[str, Any]]) -> str:
    L = ["## Dose and biological plausibility", "", "| study | verdict | reasons |", "|---|---|---|"]
    n = 0
    for r in rows:
        d = r.get("plausibility")
        if not d:
            continue
        n += 1
        L.append(f"| {r.get('ref')} | {d['verdict']} | {'; '.join(d['reasons'])[:400] or 'dose, route, duration and outcome match the question'} |")
    return ("\n".join(L) + "\n\n_Flags, not rejections: an unusual result is shown with the reasons it may not translate._\n") if n else ""
