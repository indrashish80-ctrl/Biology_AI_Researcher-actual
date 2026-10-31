"""
evidence_quality.py
-------------------
Separate, never-merged quality dimensions for every included paper (#10) and
automatic dose / exposure relevance (#11).

    paper_quality            design strength, sample size, reporting completeness
    evidence_relevance       how directly the paper addresses THIS question (screening score,
                             endpoint match, organism match)
    claim_grounding          share of the paper's claims that passed the semantic audit
    experimental_quality     controls, randomisation, blinding, replication, multiplicity control
    risk_of_bias             design-specific bias domains (RoB-2 / ROBINS / SYRCLE-style cues)
    translational_relevance  evidence level for the question's organism x dose relevance

One overall score would hide exactly the differences that matter (a strong paper can
contain weak evidence for this question, and vice versa), so each is reported on its own
with the reasons that produced it. All ratings are deterministic heuristics from the
paper text - they are screening aids, not a formal risk-of-bias assessment.

Dose / PK: each experimental concentration or dose is compared with an approximate
human exposure reference:
    in vitro / ex vivo   concentration  vs  human plasma exposure
    animal               mg/kg          ->  human-equivalent dose (FDA body-surface-area
                                            factors, 2005 guidance) vs max human daily dose
and labelled physiological / borderline / supra-physiological. A high dose is NOT called
invalid: "the result demonstrates biological activity at this concentration, but its
relevance to therapeutic human exposure is uncertain".
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from stats_parsing import detect_design_features, normalize_numeric_text, parse_p_values, parse_sample_sizes
from study_classifier import evidence_level

# --------------------------------------------------------------------------- dose / exposure
# Approximate human exposure references (published therapeutic / peak plasma ranges).
# They are deliberately conservative upper values for screening; verify for your population.
EXPOSURE_REFERENCE: Dict[str, Dict[str, Any]] = {
    "metformin": {"upper_M": 40e-6, "text": "~10-40 µM therapeutic plasma", "max_daily_mg": 2550, "mw": 129.16},
    "sirolimus|rapamycin": {"upper_M": 20e-9, "text": "~5-20 nM trough blood level", "max_daily_mg": None, "mw": 914.2},
    "resveratrol": {"upper_M": 2.5e-6, "text": "≤ ~2.5 µM peak plasma after gram doses", "max_daily_mg": 5000, "mw": 228.25},
    "curcumin": {"upper_M": 1e-6, "text": "< ~1 µM plasma (poor oral bioavailability)", "max_daily_mg": 8000, "mw": 368.38},
    "atorvastatin": {"upper_M": 0.1e-6, "text": "≤ ~0.1 µM peak plasma", "max_daily_mg": 80, "mw": 558.6},
    "simvastatin": {"upper_M": 0.1e-6, "text": "≤ ~0.1 µM peak plasma", "max_daily_mg": 80, "mw": 418.6},
    "caffeine": {"upper_M": 50e-6, "text": "~5-50 µM plasma", "max_daily_mg": 400, "mw": 194.19},
    "quercetin": {"upper_M": 1e-6, "text": "< ~1 µM plasma", "max_daily_mg": None, "mw": 302.24},
    "aspirin|acetylsalicylic|salicylate": {"upper_M": 2e-3, "text": "salicylate up to ~1-2 mM at high anti-inflammatory "
                                           "doses (far lower at low dose)", "max_daily_mg": 4000, "mw": 180.16},
}
# FDA (2005) body-surface-area conversion factors (Km); HED = animal dose x Km_animal / Km_human
KM = {"mouse": 3, "mice": 3, "hamster": 5, "rat": 6, "rats": 6, "ferret": 7, "guinea pig": 8, "rabbit": 12, "dog": 20,
      "monkey": 12, "macaque": 12, "marmoset": 6, "baboon": 20, "minipig": 35, "pig": 35}
KM_HUMAN, HUMAN_KG = 37, 60.0

_CONC = re.compile(r"(?<![\d.,])(\d{1,3}(?:,\d{3})+(?!\d)|\d+(?:\.\d+)?)\s*(mM|µM|uM|nM|pM|µg\s*/\s*mL|ug\s*/\s*mL|mg\s*/\s*mL|mg\s*/\s*L|"
                   r"mg\s*/\s*kg(?:\s*/\s*d(?:ay)?)?|mg\s*kg-1|mg\s*/\s*d(?:ay)?|g\s*/\s*d(?:ay)?)(?![A-Za-z])")
_MOLAR = {"mM": 1e-3, "µM": 1e-6, "uM": 1e-6, "nM": 1e-9, "pM": 1e-12}


def _reference_for(intervention: str) -> Optional[Dict[str, Any]]:
    low = (intervention or "").lower()
    for names, ref in EXPOSURE_REFERENCE.items():
        if re.search(rf"\b(?:{names})\b", low):
            return {"name": names.split("|")[0], **ref}
    return None


def _fmt_conc(m: float) -> str:
    for unit, f in (("mM", 1e-3), ("µM", 1e-6), ("nM", 1e-9), ("pM", 1e-12)):
        if m >= f:
            return f"{m / f:.3g} {unit}"
    return f"{m:.2g} M"


def parse_exposures(text: str) -> List[Dict[str, Any]]:
    out, seen = [], set()
    t = normalize_numeric_text(text or "").replace("μ", "µ")
    for m in _CONC.finditer(t):
        v, unit = float(m.group(1).replace(",", "")), re.sub(r"\s+", "", m.group(2))   # '1,700 mg' = 1700 mg
        key = (v, unit)
        if key in seen or v <= 0:
            continue
        if re.search(r"(?:fig(?:ure)?s?\.?|panel|table)\s*[\w,\s]{0,6}$", t[max(0, m.start() - 14):m.start()], re.I):
            continue                      # 'Figure 3G' is a figure panel, not 3 g
        seen.add(key)
        out.append({"value": v, "unit": unit, "raw": m.group(0)})
    return out


def _category(ratio: float, limits=(3.0, 10.0)) -> str:
    return "physiological" if ratio <= limits[0] else ("borderline" if ratio <= limits[1] else "supra-physiological")


_OTHER_DRUG = re.compile(r"\b(?:letrozole|anastrozole|exemestane|tamoxifen|fulvestrant|trastuzumab|pertuzumab|lapatinib|paclitaxel|"
                         r"docetaxel|doxorubicin|epirubicin|cyclophosphamide|carboplatin|cisplatin|capecitabine|gemcitabine|"
                         r"5-fu|fluorouracil|bevacizumab|everolimus|palbociclib|ribociclib|abemaciclib|insulin|glibenclamide|"
                         r"glipizide|sitagliptin|pioglitazone|statin|atorvastatin|simvastatin|aspirin|celecoxib|dexamethasone|"
                         r"prednisone|vitamin|placebo|[a-z]+(?:mab|nib|platin|taxel|rubicin|zole|stat)\b)", re.I)


def own_exposures(intervention: str, text: str) -> List[Dict[str, Any]]:
    """Doses that belong to THIS intervention: a dose written next to another drug ('letrozole 2.5 mg/day' in a
    metformin + letrozole trial) is that drug's dose, never metformin's."""
    exps = parse_exposures(text)
    toks = [t for t in re.findall(r"[a-z]{4,}", (intervention or "").lower()) if t not in ("with", "plus", "combination")]
    if not exps or not toks:
        return exps
    t = text or ""
    keep = []
    for e in exps:
        i = t.find(e["raw"])
        if i < 0:
            keep.append(e)
            continue
        before, after = t[max(0, i - 50):i].lower(), t[i:i + len(e["raw"]) + 25].lower()
        own_near = any(k in before or k in after for k in toks)
        other = _OTHER_DRUG.search(before[-30:]) or _OTHER_DRUG.search(after)
        if other and not (own_near and any(before.rfind(k) > before.rfind(other.group(0).lower()) for k in toks)):
            continue                     # the nearest drug name before the dose is another drug
        keep.append(e)
    return keep


def dose_relevance(intervention: str, text: str, model_type: str, species_text: str = "") -> Dict[str, Any]:
    if model_type == "review_meta":
        return {"category": "not applicable", "assessments": [],
                "statement": "Review - the doses belong to the studies it cites; not assessed for this paper."}
    ref = _reference_for(intervention)
    exposures = own_exposures(intervention, text)
    if not exposures:
        return {"category": "not reported", "assessments": [], "statement": "No dose or concentration found in the text."}
    if not ref:
        doses = ", ".join(e["raw"] for e in exposures[:3])
        return {"category": "not assessed", "assessments": [],
                "statement": f"No human exposure reference on file for '{intervention or 'this intervention'}'; "
                             f"relevance of {doses} to human exposure was not assessed."}
    assessments = []
    for e in exposures[:6]:
        u = e["unit"]
        a: Dict[str, Any] = {"exposure": e["raw"]}
        if u in _MOLAR or u.lower().endswith(("/ml", "/l")):
            if u in _MOLAR:
                molar = e["value"] * _MOLAR[u]
            else:
                ul = u.lower()
                # 1 µg/mL = 1 mg/L ; 1 mg/mL = 1000 mg/L
                mg_per_l = e["value"] * (1000.0 if ul.startswith("mg/ml") else 1.0)
                molar = mg_per_l / 1000.0 / ref["mw"]
            ratio = molar / ref["upper_M"]
            a.update({"kind": "concentration", "molar": _fmt_conc(molar), "reference": ref["text"],
                      "ratio": round(ratio, 1), "category": _category(ratio)})
        elif "kg" in u:
            sp = next((k for k in KM if re.search(rf"\b{k}\b", species_text or "", re.I)), "mouse" if model_type == "animal" else None)
            if not sp or not ref.get("max_daily_mg"):
                continue
            hed = e["value"] * KM[sp] / KM_HUMAN
            human_max = ref["max_daily_mg"] / HUMAN_KG
            ratio = hed / human_max
            a.update({"kind": "animal dose", "species": sp, "hed_mg_per_kg": round(hed, 2),
                      "reference": f"max human dose ~{ref['max_daily_mg']} mg/day (~{human_max:.1f} mg/kg)",
                      "ratio": round(ratio, 1), "category": _category(ratio, (1.0, 3.0))})
        elif "day" in u or u.endswith("/d"):
            if not ref.get("max_daily_mg"):
                continue
            mg = e["value"] * (1000.0 if u.lower().startswith("g") else 1.0)
            ratio = mg / ref["max_daily_mg"]
            a.update({"kind": "human daily dose", "reference": f"max human dose ~{ref['max_daily_mg']} mg/day",
                      "ratio": round(ratio, 2), "category": "physiological" if ratio <= 1.0 else "supra-physiological"})
        else:
            continue
        assessments.append(a)
    if not assessments:
        return {"category": "not assessed", "assessments": [],
                "statement": "Doses were reported in units that cannot be compared with human exposure."}
    # judge the dose of the system the finding comes from: a mouse study's mg/kg, not the mM of its cell assays
    want = {"animal": "animal dose", "experimental_disease_model": "animal dose", "human_clinical": "human daily dose",
            "human_observational": "human daily dose", "case_report": "human daily dose"}.get(model_type or "")
    if want:
        match = [a for a in assessments if a.get("kind") == want]
        if match:
            assessments = match
        elif all(a.get("kind") == "concentration" for a in assessments):
            conc = ", ".join(a["exposure"] for a in assessments[:3])
            return {"category": "not assessed", "assessments": assessments,
                    # a molar value in an animal / patient study can be a plasma level as well as a cell-assay dose
                    "statement": f"Only molar concentrations ({conc}) were found - in this {'animal' if want == 'animal dose' else 'human'} "
                                 f"study they may be cell-assay or blood / plasma concentrations; the administered "
                                 f"{'animal' if want == 'animal dose' else 'patient'} dose was not found in the text, so relevance "
                                 f"to human exposure was not assessed.",
                    "reference": ref["text"], "intervention": ref["name"]}
    order = ["physiological", "borderline", "supra-physiological"]
    worst = max(assessments, key=lambda x: order.index(x["category"]))
    hed = (f" in {worst['species']} (human-equivalent ~{worst['hed_mg_per_kg']} mg/kg)"
           if worst.get("hed_mg_per_kg") is not None else "")
    if worst["category"] == "physiological":
        stmt = (f"{worst['exposure']}{hed} is within the range of human exposure to {ref['name']} ({worst['reference']}); "
                f"the result is relevant to therapeutic exposure.")
    else:
        stmt = (f"The result demonstrates biological activity at {worst['exposure']}{hed}, about {worst['ratio']}x the "
                f"human reference ({worst['reference']}); its relevance to therapeutic human exposure is uncertain.")
    return {"category": worst["category"], "assessments": assessments, "statement": stmt,
            "reference": ref["text"], "intervention": ref["name"]}


# --------------------------------------------------------------------------- quality dimensions
def _rate(score: float, hi: float, mid: float) -> str:
    return "High" if score >= hi else ("Moderate" if score >= mid else "Low")


DESIGN_POINTS = {"human_clinical": 3.0, "human_observational": 2.0, "animal": 1.5, "plant": 1.5, "microbial": 1.5,
                 "organoid_ex_vivo": 1.0, "cell_culture": 1.0, "in_silico": 0.5, "review_meta": 1.0, "unclear": 0.5}


def quality_profile(row: Dict[str, Any], text: str, claims_total: int, claims_verified: int,
                    screening_score: Optional[float], question_endpoints: List[str], focus: str,
                    dose: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    etype = row.get("evidence_type") or "unclear"
    feats = detect_design_features(text)
    full = row.get("text_basis") not in (None, "abstract")
    n = row.get("n")
    try:
        # the matrix row's n is the single source of truth (None = not applicable / not reported); text is parsed only
        # for rows built without one, so the quality profile and the QC table can never disagree
        n = int(n) if n is not None else (None if "n" in row else
                                          (max(parse_sample_sizes(text)) if parse_sample_sizes(text) else None))
    except (TypeError, ValueError):
        n = None
    out: Dict[str, Dict[str, Any]] = {}

    # 1 paper quality
    pts, why = DESIGN_POINTS.get(etype, 0.5), [f"design: {row.get('evidence_type_label') or etype}"]
    if etype == "human_clinical" and feats.get("randomized"):
        pts += 0.5; why.append("randomised")
    if etype == "review_meta" and feats.get("meta_analysis"):
        pts += 1.0; why.append("quantitative meta-analysis")
    if n:
        big = 100 if etype in ("human_clinical", "human_observational") else 10
        if n >= big:
            pts += 1.0; why.append(f"n={n} (adequate for design)")
        else:
            why.append(f"n={n} (small for design)")
    else:
        why.append("sample size not reported")
    if parse_p_values(text) or re.search(r"\b95\s*%\s*CI\b|confidence interval", text, re.I):
        pts += 0.5; why.append("statistics reported")
    if full:
        pts += 0.5; why.append("full text read")
    else:
        why.append("abstract only - limited assessment")
    out["paper_quality"] = {"rating": _rate(pts, 4.0, 2.5), "reasons": why}

    # 2 evidence relevance (to THIS question)
    rel, why = 0.0, []
    if screening_score is not None:
        rel += 2.0 * float(screening_score); why.append(f"screening relevance {float(screening_score):.2f}")
    fams = {f.get("endpoint") for f in row.get("findings") or []}
    if question_endpoints and fams & set(question_endpoints):
        rel += 1.5; why.append("measures the endpoint asked about")
    elif question_endpoints:
        why.append("does not measure the endpoint asked about (" + ", ".join(e.replace('_', ' ') for e in question_endpoints) + ")")
    lvl = row.get("level")
    if lvl in (1, 2):
        rel += 1.0; why.append("studies the organism the question is about")
    out["evidence_relevance"] = {"rating": _rate(rel, 3.0, 1.5), "reasons": why or ["no screening data"]}

    # 3 claim grounding
    share = (claims_verified / claims_total) if claims_total else 0.0
    out["claim_grounding"] = {"rating": _rate(share, 0.8, 0.5) if claims_total else "Low",
                              "reasons": [f"{claims_verified}/{claims_total} claims passed the semantic audit"]}

    # 4 experimental quality (design-appropriate features)
    wanted = {"human_clinical": ["randomized", "blinded", "controlled", "preregistered"],
              "human_observational": ["controlled", "multiple_testing_correction"],
              "animal": ["randomized", "blinded", "controlled", "replicated"],
              "organoid_ex_vivo": ["controlled", "replicated", "multiple_testing_correction"],
              "cell_culture": ["controlled", "replicated", "multiple_testing_correction"],
              "plant": ["controlled", "replicated", "randomized"], "microbial": ["controlled", "replicated"],
              }.get(etype, ["controlled"])
    if etype == "human_observational":
        feats["controlled"] = bool(re.search(r"\badjust\w*|multivariab\w*|propensity|confound\w*", text, re.I))
    have = [w for w in wanted if feats.get(w)]
    out["experimental_quality"] = {
        "rating": _rate(len(have) / max(1, len(wanted)), 0.66, 0.33) if full or have else "Not assessable",
        "reasons": [f"present: {', '.join(have) or 'none'}", f"not found: {', '.join(w for w in wanted if w not in have) or 'none'}"]}

    # 5 risk of bias (design-specific domains)
    domains = {
        "human_clinical": {"random sequence": feats.get("randomized"), "blinding": feats.get("blinded"),
                           "allocation concealment": bool(re.search(r"allocation conceal|sealed envelope|central(?:ised|ized) randomi", text, re.I)),
                           "complete outcome data / ITT": bool(re.search(r"intention[- ]to[- ]treat|\bITT\b|lost to follow|attrition", text, re.I)),
                           "pre-registration": feats.get("preregistered") or bool(re.search(r"NCT\d{8}|ISRCTN|registered", text, re.I))},
        "human_observational": {"confounding adjusted": feats.get("controlled"),
                                "defined cohort / sampling": bool(re.search(r"population-based|consecutive|inclusion criteria|eligib", text, re.I)),
                                "exposure measured objectively": bool(re.search(r"prescription|pharmacy|records|registry", text, re.I))},
        "animal": {"randomisation": feats.get("randomized"), "blinding": feats.get("blinded"),
                   "sample size justification": bool(re.search(r"power (?:analysis|calculation)|sample size (?:was )?(?:calculated|determined)", text, re.I))},
    }.get(etype)
    if not full:
        out["risk_of_bias"] = {"rating": "Not assessable", "reasons": ["abstract only"]}
    elif domains is None:
        out["risk_of_bias"] = {"rating": "High" if etype in ("cell_culture", "organoid_ex_vivo", "in_silico") else "Not assessable",
                               "reasons": ["no blinding/randomisation framework for this design"
                                           if etype in ("cell_culture", "organoid_ex_vivo", "in_silico") else "design not assessable"]}
    else:
        met = [k for k, v in domains.items() if v]
        frac = len(met) / len(domains)
        out["risk_of_bias"] = {"rating": "Low" if frac >= 0.8 else ("Some concerns" if frac >= 0.4 else "High"),
                               "reasons": [f"addressed: {', '.join(met) or 'none'}",
                                           f"not reported: {', '.join(k for k in domains if k not in met) or 'none'}"]}

    # 6 translational relevance (level for the question's organism x dose)
    lvl = evidence_level(etype, focus, row.get("organism_group"))
    base = {1: 4, 2: 3, 3: 2, 4: 1, 5: 1, 6: 0}.get(lvl or 6, 0)
    why = [f"evidence level {('L' + str(lvl)) if lvl else 'secondary'}"]
    if dose.get("category") == "supra-physiological":
        base -= 1; why.append("supra-physiological exposure")
    elif dose.get("category") == "borderline":
        base -= 0.5; why.append("borderline exposure")
    elif dose.get("category") == "physiological":
        why.append("physiological exposure")
    out["translational_relevance"] = {"rating": "High" if base >= 3.5 else "Moderate" if base >= 2.5 else "Low" if base >= 1 else "Very low",
                                      "reasons": why}
    return out


# --------------------------------------------------------------------------- causal strength, replication (points 31, 32)
def causal_strength(row: Dict[str, Any]) -> Dict[str, Any]:
    inf, comp, n = row.get("inference_type"), row.get("comparator"), row.get("n")
    why = [f"inference: {(inf or 'unknown').replace('_', ' ')}", f"comparator: {comp or 'not reported'}",
           f"n: {n if n is not None else 'not reported'}"]
    if inf == "randomised_controlled":
        rating = "High"
    elif inf == "controlled_experiment":
        rating = "Moderate"
    elif inf == "statistical_association":
        rating = "Low"
    else:
        rating = "Very low"
    if n == 1 or comp == "none":
        rating = "Very low"
        why.append("single case or no comparator - effect cannot be attributed to the intervention")
    return {"rating": rating, "reasons": why}


def _surnames(authors: Optional[str]) -> set:
    out = set()
    for a in (authors or "").split(","):
        parts = a.strip().split()
        if parts:
            sur = parts[0] if (len(parts) > 1 and len(parts[-1]) <= 3 and parts[-1].isupper()) else parts[-1]
            out.add(sur.lower())
    return out


def apply_replication(rows: List[Dict[str, Any]]) -> None:
    """Replication as a formal property: independent group vs same group, same vs different model / species."""
    def secondary(x: Dict[str, Any]) -> bool:
        return x.get("evidence_type") == "review_meta" or bool(x.get("secondary_report"))

    for r in rows:
        if secondary(r):
            # a review relays other studies: replication of ITS findings is not a property of this paper
            r["replication"] = {"status": "not applicable (review - secondary literature)", "same_group": [],
                                "independent_same_model": [], "independent_other_model_or_species": [], "conflicting": []}
            q = r.setdefault("quality", {})
            q["replication"] = {"rating": "Not assessable", "reasons": ["review - secondary literature"]}
            q["causal_strength"] = causal_strength(r)
            continue
        def ep_key(f: Dict[str, Any]) -> Any:
            # 'expression' of METTL3 and 'expression' of a miRNA are different read-outs: never a replication / conflict
            if f.get("endpoint") in ("expression", "signalling", "protein_localization"):
                return (f.get("endpoint"), f.get("target") or f"?{id(f)}")
            return f.get("endpoint")
        skip = ("secondary_report", "combination")
        mine = [(ep_key(f), f.get("direction"), set(re.findall(r"[a-z]{4,}", (f.get("intervention") or "").lower())))
                for f in r.get("findings") or [] if f.get("evidence_role") not in skip]
        same_group, indep_same_model, indep_other_model, conflicting = [], [], [], []
        for o in rows:
            if o is r or secondary(o):
                continue                       # reviews never replicate or contradict a primary study
            for f in o.get("findings") or []:
                if f.get("evidence_role") in skip:
                    continue
                toks = set(re.findall(r"[a-z]{4,}", (f.get("intervention") or "").lower()))
                for ep, d, it in mine:
                    if ep != ep_key(f) or not (it & toks):
                        continue
                    if d != f.get("direction"):
                        conflicting.append(o["ref"])
                        continue
                    if _surnames(r.get("authors")) & _surnames(o.get("authors")):
                        same_group.append(o["ref"])
                    elif (o.get("evidence_type"), (o.get("model_profile") or {}).get("species")) == \
                            (r.get("evidence_type"), (r.get("model_profile") or {}).get("species")):
                        indep_same_model.append(o["ref"])
                    else:
                        indep_other_model.append(o["ref"])
        u = lambda xs: sorted(set(xs))  # noqa: E731
        if indep_other_model:
            status, rating = "independently replicated across models/species", "High"
        elif indep_same_model:
            status, rating = "independently replicated in the same model", "Moderate"
        elif same_group:
            status, rating = "replicated by the same research group only", "Low"
        else:
            status, rating = "not replicated within the deep-read sample", "Very low"
        r["replication"] = {"status": status, "same_group": u(same_group), "independent_same_model": u(indep_same_model),
                            "independent_other_model_or_species": u(indep_other_model), "conflicting": u(conflicting)}
        q = r.setdefault("quality", {})
        q["replication"] = {"rating": rating, "reasons": [status] + ([f"opposite results in {', '.join(u(conflicting))}"] if conflicting else [])}
        q["causal_strength"] = causal_strength(r)


# --------------------------------------------------------------------------- honest QC checklist (points 16-18, 29, 30)
def _phi(x: float) -> float:
    import math
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def qc_checklist(row: Dict[str, Any], text: str) -> Dict[str, Dict[str, str]]:
    """What could - and could NOT - be validated. Never reports 'passed' without the data to check."""
    from stats_parsing import parse_effect_sizes
    n = row.get("n")
    ps = parse_p_values(text)
    ci = re.search(r"\b9[05]\s*%\s*(?:CI|confidence interval)", text, re.I)
    d = next((e["value"] for e in parse_effect_sizes(text) if e.get("type") in ("cohens_d", "hedges_g")), None)
    out: Dict[str, Dict[str, str]] = {}
    out["statistical_information"] = {"status": "reported" if (ps or ci) else "not reported",
                                      "detail": (f"{len(ps)} p-value(s)" if ps else "no p-values") + ("; confidence interval" if ci else "")}
    if n and d and n >= 4:
        power = _phi(abs(float(d)) * ((n / 2) / 2) ** 0.5 - 1.96)
        out["statistical_power"] = {"status": f"approx. {power:.0%}",
                                    "detail": f"two-group normal approximation, d={d}, n={n} (n/2 per group), alpha=0.05 - indicative only"}
    else:
        missing = [x for x, v in (("sample size", n), ("effect size", d)) if not v] + ["variance"]
        out["statistical_power"] = {"status": "not assessable",
                                    "detail": "requires sample size, effect size, variance and alpha; missing: " + ", ".join(missing)
                                    + ". A p-value alone does not establish adequate power."}
    dose = row.get("dose_relevance") or {}
    cat = dose.get("category")
    out["dose_validity"] = {"status": "unknown" if cat in (None, "not reported", "not assessed") else cat,
                            "detail": dose.get("statement") or "no dose reported"}
    out["sample_size"] = {"status": str(n) if n is not None else str(row.get("n_display") or "not reported"),
                          "detail": "single case" if n == 1 else ("" if n else "evidence weight cannot be judged without n")}
    out["comparator"] = {"status": row.get("comparator") or "not reported",
                         "detail": "no control - causal inference weak" if row.get("comparator") in ("none", "not reported") else ""}
    fu = row.get("follow_up_days")
    out["follow_up"] = {"status": f"{fu / 30.44:.1f} months" if fu else "not reported",
                        "detail": "durability / recurrence not assessable" if not fu or fu < 365 else ""}
    if row.get("evidence_type") == "review_meta":
        # a review REPORTS other studies' experiments: none of the experimental checks apply to it
        for k in ("statistical_power", "dose_validity", "sample_size", "comparator", "follow_up"):
            out[k] = {"status": "not applicable (review)", "detail": "secondary literature - the original experiment was not examined"}
    out["experiment_origin"] = {"status": "secondary report (review)" if row.get("evidence_type") == "review_meta" else "this study",
                                "detail": "results are reported from other papers; BioSynthetica did not verify the original experiments"
                                if row.get("evidence_type") == "review_meta" else ""}
    out["text_basis"] = {"status": "abstract only" if row.get("text_basis") in (None, "abstract") else "full text",
                         "detail": "methods, dose, statistics and adverse effects cannot be fully checked from an abstract"
                         if row.get("text_basis") in (None, "abstract") else ""}
    return out
