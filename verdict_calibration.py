"""
verdict_calibration.py  (v25)
=============================
Calibrated uncertainty: every answer gets ONE of five explicit verdict categories, decided by written criteria from
the verified evidence - never by how confident a model sounds.

    strongly supported      >= 2 INDEPENDENT human randomised trials (publications of one trial count once) agree on
                            the asked endpoint and direction, together >= 300 participants, none at high risk of bias
                            alone carrying the result, and no significant human result the other way;
                            or a pooled estimate of >= 2 independent human studies whose interval excludes no-effect
                            with I-squared < 50%
    supported with limits   at least one human randomised trial OR >= 2 independent human observational studies
                            agree, nothing significant the other way, but a criterion of 'strongly supported' is missed
                            (the limits are listed: single trial, small sample, high risk of bias, observational only,
                            surrogate / short-term endpoint, heterogeneity)
    mixed                   significant human results in BOTH directions on the asked endpoint, or a significant human
                            effect against an adequately powered human null of the same design
    preliminary             evidence only from animals, cells, in-silico work, case reports, or human studies with
                            fewer than 50 participants in total / non-randomised pilots
    insufficient            no verified finding on the asked endpoint, only non-significant results from studies that
                            were not shown to be powered, or only secondary literature (reviews re-reporting studies)

The direction is reported separately: benefit/increase, decrease, no effect, or unclear.
Each verdict carries the criteria met and the criteria missed, so a reader sees WHY it is that category.

    calibrate(studies)           studies = [{'design','n','direction','significant','independent_id','risk_of_bias',
                                             'surrogate','follow_up_days','powered'}]
    from_matrix(matrix)          builds the study list from a finished evidence matrix
    gold_check(questions)        rule self-check over evidence sets built from the gold questions (gold_bench part D)
"""
from __future__ import annotations

import re
from collections import Counter, defaultdict
from typing import Any, Dict, Iterable, List, Optional

VERDICT_VERSION = "1.0"
CATEGORIES = ["strongly_supported", "supported_with_limits", "mixed", "preliminary", "insufficient"]
LABELS = {"strongly_supported": "Strongly supported", "supported_with_limits": "Supported, with limits",
          "mixed": "Mixed evidence", "preliminary": "Preliminary", "insufficient": "Insufficient evidence"}
HUMAN_RCT = {"rct", "human_clinical", "randomised_controlled", "randomized_controlled", "meta_analysis_rct"}
HUMAN_OBS = {"human_observational", "observational", "cohort", "case_control", "cross_sectional"}
PRECLIN = {"animal", "cell_culture", "in_silico", "organoid_ex_vivo", "experimental_disease_model", "plant", "microbial",
           "in_vitro", "case_report"}
STRONG_MIN_N = 300
SMALL_N = 50


def _kind(design: str) -> str:
    d = (design or "").lower()
    if d in HUMAN_RCT or re.search(r"random", d):
        return "rct"
    if d in HUMAN_OBS:
        return "obs"
    if d in ("review_meta", "review", "secondary"):
        return "secondary"
    if d in PRECLIN:
        return "preclinical"
    return "unclear"


def calibrate(studies: Iterable[Dict[str, Any]], pooled: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    st = [dict(s) for s in studies]
    for s in st:
        s["kind"] = _kind(s.get("design"))
        s["direction"] = (s.get("direction") or "unclear").lower()
        if s["direction"] in ("increase", "decrease") and s.get("significant") is False:
            s["direction"] = "no_effect"
    # independence: publications of one trial / cohort count once (the strongest-reported one stands for the group)
    groups: Dict[Any, Dict[str, Any]] = {}
    for i, s in enumerate(st):
        key = s.get("independent_id") or f"study{i}"
        cur = groups.get(key)
        if cur is None or (s.get("n") or 0) > (cur.get("n") or 0):
            groups[key] = s
    ind = list(groups.values())
    human = [s for s in ind if s["kind"] in ("rct", "obs")]
    effects = [s for s in human if s["direction"] in ("increase", "decrease")]
    dirs = Counter(s["direction"] for s in effects)
    met: List[str] = []
    missed: List[str] = []
    limits: List[str] = []
    counts = {"independent_studies": len(ind), "human": len(human), "human_rct": sum(s["kind"] == "rct" for s in human),
              "human_observational": sum(s["kind"] == "obs" for s in human),
              "preclinical": sum(s["kind"] == "preclinical" for s in ind),
              "secondary": sum(s["kind"] == "secondary" for s in ind), "publications": len(st)}

    def result(cat: str, direction: str) -> Dict[str, Any]:
        return {"category": cat, "label": LABELS[cat], "direction": direction, "criteria_met": met,
                "criteria_missed": missed, "limits": limits, "counts": counts, "version": VERDICT_VERSION}

    if not ind or all(s["kind"] == "secondary" for s in ind):
        missed.append("no primary study measured the asked endpoint" if not ind else "only secondary literature")
        return result("insufficient", "unclear")
    # MIXED: significant human effects both ways, or an effect against a powered null of the same design
    if len(dirs) == 2:
        met.append(f"significant human results in both directions ({dirs['increase']} increase, {dirs['decrease']} decrease)")
        return result("mixed", "unclear")
    if effects:
        d = effects[0]["direction"]
        powered_nulls = [s for s in human if s["direction"] == "no_effect" and (s.get("powered") or (s.get("n") or 0) >= 300)
                         and s["kind"] == effects[0]["kind"]]
        if powered_nulls:
            met.append(f"{len(effects)} human stud(ies) show a {d} but {len(powered_nulls)} adequately sized "
                       f"stud(ies) of the same design show no difference")
            return result("mixed", "unclear")
        rcts = [s for s in effects if s["kind"] == "rct"]
        obs = [s for s in effects if s["kind"] == "obs"]
        total_n = sum(s.get("n") or 0 for s in effects)
        high_rob = [s for s in effects if (s.get("risk_of_bias") or "").lower() == "high"]
        surrogate = [s for s in effects if s.get("surrogate")]
        if pooled and pooled.get("k", 0) >= 2 and pooled.get("significant") and (pooled.get("i2") or 0) < 50:
            met.append(f"pooled estimate of {pooled['k']} independent studies excludes no effect (I² {pooled.get('i2', 0):.0f}%)")
            return result("strongly_supported", d)
        if len(rcts) >= 2 and total_n >= STRONG_MIN_N and len(high_rob) < len(rcts) and not surrogate:
            met.append(f"{len(rcts)} independent randomised trials agree ({total_n} participants)")
            return result("strongly_supported", d)
        # not strong: say which criterion is missing
        if len(rcts) < 2:
            missed.append("fewer than 2 independent randomised trials")
            if rcts:
                limits.append("single randomised trial")
        if total_n < STRONG_MIN_N:
            missed.append(f"fewer than {STRONG_MIN_N} participants in the agreeing studies" +
                          (f" ({total_n})" if total_n else " (sample sizes not reported)"))
        if high_rob:
            limits.append(f"{len(high_rob)} agreeing stud(ies) at high risk of bias")
        if surrogate:
            limits.append("surrogate / short-term endpoint")
        if rcts and (total_n >= SMALL_N or any(s.get("n") is None for s in rcts)):
            met.append(f"{len(rcts)} randomised trial(s) in the direction '{d}'")
            if not total_n:
                limits.append("sample size not reported")
            return result("supported_with_limits", d)
        if len(obs) >= 2 and total_n >= SMALL_N:
            met.append(f"{len(obs)} independent observational studies agree")
            limits.append("observational evidence only (association, not causation)")
            return result("supported_with_limits", d)
        limits.append("small or single non-randomised human evidence")
        met.append(f"human evidence in the direction '{d}' ({total_n or 'unknown'} participants)")
        return result("preliminary", d)
    # no significant human effect
    pre = [s for s in ind if s["kind"] == "preclinical" and s["direction"] in ("increase", "decrease")]
    nulls = [s for s in human if s["direction"] == "no_effect"]
    powered = [s for s in nulls if s.get("powered") or (s.get("n") or 0) >= STRONG_MIN_N]
    if len(powered) >= 2 and sum(s["kind"] == "rct" for s in powered) >= 1:
        met.append(f"{len(powered)} adequately sized human studies found no significant difference")
        return result("supported_with_limits", "no_effect")
    if pre:
        pdirs = Counter(s["direction"] for s in pre)
        d = pdirs.most_common(1)[0][0] if len(pdirs) == 1 else "unclear"
        met.append(f"{len(pre)} preclinical stud(ies) only")
        if nulls:
            limits.append(f"{len(nulls)} human stud(ies) found no significant difference (power not shown)")
        return result("preliminary", d)
    if nulls:
        missed.append("only non-significant human results, from studies not shown to be powered for the effect")
        return result("insufficient", "no_effect")
    missed.append("no finding with a direction on the asked endpoint")
    return result("insufficient", "unclear")


def from_matrix(matrix: Dict[str, Any], question: str = "") -> Dict[str, Any]:
    """Study list from the evidence matrix: one entry per paper (its findings that answer the question), trials
    grouped by the matrix's trial grouping, design from the classification, risk of bias from the quality block."""
    studies = []
    for r in (matrix or {}).get("rows") or []:
        if (r.get("paper_relevance") or {}).get("ok") is False:
            continue
        fs = [f for f in r.get("findings") or [] if f.get("answers_question") is not False and f.get("direction")]
        if not fs:
            continue
        dirs = Counter((f.get("direction") or "unclear") for f in fs if f.get("direction") in ("increase", "decrease", "no_effect"))
        if not dirs:
            continue
        d = "mixed" if len({k for k in dirs if k != "no_effect"}) == 2 else dirs.most_common(1)[0][0]
        rob = ((r.get("quality") or {}).get("risk_of_bias") or {})
        rob_label = (rob.get("rating") or rob.get("label")) if isinstance(rob, dict) else rob
        det = r.get("risk_of_bias_detail") or {}
        if det.get("tool"):
            import risk_of_bias as _rob
            rob_label = _rob.label(det)
        design = r.get("evidence_type") or "unclear"
        if r.get("secondary_report"):
            design = "review_meta"
        studies.append({"ref": r.get("ref"), "design": "rct" if design == "human_clinical" and
                        re.search(r"random", f"{r.get('design') or ''} {r.get('title') or ''} {r.get('evidence_type_label') or ''}", re.I)
                        else design, "n": r.get("n") if isinstance(r.get("n"), int) else None,
                        "direction": d if d != "mixed" else "unclear", "significant": d in ("increase", "decrease"),
                        "independent_id": r.get("trial_group") or r.get("ref"),
                        "risk_of_bias": str(rob_label or "").lower() or None,
                        "surrogate": any((f.get("outcome_level") or "").lower().startswith("surrogate") for f in fs)})
    pooled = None
    try:
        import meta_analysis as ma
        res = ma.analyse((matrix or {}).get("rows") or [])
        pooled = next((p for p in res if p and p.get("k", 0) >= 2), None)
    except Exception:
        pooled = None
    out = calibrate(studies, pooled)
    out["studies"] = [{k: s.get(k) for k in ("ref", "design", "n", "direction", "independent_id", "risk_of_bias")} for s in studies]
    return out


def markdown(v: Dict[str, Any]) -> str:
    dir_txt = {"increase": "an increase", "decrease": "a decrease", "no_effect": "no effect", "unclear": "no clear direction"}
    L = [f"## Verdict: {v['label']}" + (f" ({dir_txt.get(v['direction'], v['direction'])})" if v["direction"] != "unclear" else ""),
         "", "| | |", "|---|---|"]
    c = v["counts"]
    L.append(f"| Evidence | {c['independent_studies']} independent studies ({c['publications']} publications): "
             f"{c['human_rct']} human randomised, {c['human_observational']} human observational, "
             f"{c['preclinical']} preclinical, {c['secondary']} secondary |")
    if v["criteria_met"]:
        L.append("| Criteria met | " + "; ".join(v["criteria_met"]) + " |")
    if v["criteria_missed"]:
        L.append("| Criteria not met | " + "; ".join(v["criteria_missed"]) + " |")
    if v["limits"]:
        L.append("| Limits | " + "; ".join(v["limits"]) + " |")
    L += ["", "_Categories: strongly supported · supported with limits · mixed · preliminary · insufficient - each defined "
          "by written criteria (verdict_calibration.py), tested on the gold set._", ""]
    return "\n".join(L) + "\n"


# --------------------------------------------------------------------------------------------- gold self-check
def gold_check(questions: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Evidence sets built from the gold questions (each a doctor-labelled trial result) and the category the rules
    MUST give; a violation is any mismatch. Covers single trials, duplicate publications, agreeing / conflicting
    trials, preclinical-only and empty sets."""
    cases = []
    by_outcome = defaultdict(list)
    for q in questions:
        by_outcome[q["expected_direction"]].append(q)
    for q in questions:
        sig = q["expected_direction"] != "no_effect"
        st = [{"design": "rct", "n": None, "direction": q["expected_direction"], "significant": sig,
               "independent_id": q["pmcid"]}]
        cases.append(("single trial", st, "supported_with_limits" if sig else "insufficient"))
        # the same trial published twice must not become 'strongly supported'
        cases.append(("one trial, two publications", st + [dict(st[0], n=400)], "supported_with_limits" if sig else "insufficient"))
    inc, dec = by_outcome.get("increase", []), by_outcome.get("decrease", [])
    for a, b in zip(inc[::2], inc[1::2]):
        st = [{"design": "rct", "n": 200, "direction": "increase", "significant": True, "independent_id": a["pmcid"] + a["qid"]},
              {"design": "rct", "n": 200, "direction": "increase", "significant": True, "independent_id": b["pmcid"] + b["qid"]}]
        cases.append(("two agreeing trials, 400 participants", st, "strongly_supported"))
        cases.append(("two agreeing small trials", [dict(x, n=60) for x in st], "supported_with_limits"))
    for a, b in zip(inc, dec):
        st = [{"design": "rct", "n": 200, "direction": "increase", "significant": True, "independent_id": "A" + a["qid"]},
              {"design": "rct", "n": 200, "direction": "decrease", "significant": True, "independent_id": "B" + b["qid"]}]
        cases.append(("conflicting trials", st, "mixed"))
    for q in questions[:30]:
        cases.append(("animal only", [{"design": "animal", "n": 12, "direction": "decrease", "significant": True}], "preliminary"))
        cases.append(("nothing", [], "insufficient"))
        cases.append(("review only", [{"design": "review_meta", "direction": "decrease", "significant": True}], "insufficient"))
    dist = Counter()
    bad = []
    for name, st, want in cases:
        got = calibrate(st)["category"]
        dist[got] += 1
        if got != want:
            bad.append({"case": name, "expected": want, "got": got})
    return {"cases": len(cases), "violations": len(bad), "examples": bad[:10], "distribution": dict(dist),
            "version": VERDICT_VERSION}
