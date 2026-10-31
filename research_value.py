"""
research_value.py  (v26)
========================
The evidence-driven research planner's missing step: BEFORE searching further, ask whether more evidence is likely to
CHANGE the conclusion, and if so, which evidence.

    assess(dossier) -> {
        'supported':  what the evidence already supports (category, direction, independent human studies),
        'gaps':       [{'type': 'study quality' | 'missing population' | 'contradictory outcomes' | 'lack of evidence' |
                        'imprecision', 'detail', 'query'}],
        'robustness': for a pooled effect - how many null studies of average weight would make it non-significant
                      (a fail-safe count on the pooled z, Rosenthal-style), and whether one more trial of average size
                      could move the interval across no-effect,
        'value':      'high' | 'moderate' | 'low'   (likelihood that further evidence changes the conclusion),
        'next_tasks': targeted searches for the gaps that matter,
        'stop':       True when further searching is unlikely to change the conclusion, with the reason
    }
analysis_tiers uses it: a run escalates to a deeper tier only while value is not 'low'.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

VALUE_VERSION = "1.0"


def fail_safe(pooled: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Null studies of average weight needed to pull the pooled estimate's z below 1.96."""
    sts = pooled.get("studies") or []
    if pooled.get("estimate") is None or len(sts) < 2:
        return None
    ratio = pooled.get("kind") == "ratio"
    try:
        mu = math.log(pooled["estimate"]) if ratio else pooled["estimate"]
        se = ((math.log(pooled["hi"]) - math.log(pooled["lo"])) if ratio else (pooled["hi"] - pooled["lo"])) / (2 * 1.959964)
    except (ValueError, ZeroDivisionError):
        return None
    if se <= 0:
        return None
    z = abs(mu) / se
    k = len(sts)
    # with null studies of the average weight added, the pooled precision grows ~ (k+n)/k and the estimate shrinks ~ k/(k+n):
    # z(n) = z * sqrt(k / (k + n))  ->  n = k * (z^2 / 1.96^2 - 1)
    n_null = max(0.0, k * (z * z / 1.959964 ** 2 - 1))
    one_more = z * math.sqrt(k / (k + 1))
    return {"z": round(z, 2), "k": k, "null_studies_to_lose_significance": math.ceil(n_null) if z >= 1.96 else 0,
            "one_null_trial_would_change_significance": z >= 1.96 and one_more < 1.96,
            "significant": z >= 1.96}


def assess(dossier: Dict[str, Any]) -> Dict[str, Any]:
    from evidence_identity import question_frame
    q = dossier.get("research_question") or ""
    fr = question_frame(q)
    rows = [r for r in (dossier.get("evidence_matrix") or {}).get("rows") or [] if not r.get("secondary_report")]
    v = dossier.get("calibrated_verdict") or {}
    cat = v.get("category") or "insufficient"
    human = [r for r in rows if r.get("evidence_type") in ("human_clinical", "human_observational")]
    rcts = [r for r in human if r.get("evidence_type") == "human_clinical"]
    indep = len({r.get("trial_group") or r.get("ref") for r in human})
    iv = (fr.get("exposures") or [None])[0] or ""
    from evidence_identity import question_outcome
    oc = str(question_outcome(fr) or "")
    base = " ".join(x for x in (iv, oc) if x) or q
    gaps: List[Dict[str, str]] = []
    human_q = fr.get("domain") == "human_medicine"
    if not rows:
        gaps.append({"type": "lack of evidence", "detail": "no primary study answers the question",
                     "query": base})
    if human_q and not human:
        gaps.append({"type": "missing population", "detail": "no human study - only cell / animal evidence",
                     "query": f"{base} patients clinical study"})
    elif human_q and not rcts and human:
        gaps.append({"type": "study quality", "detail": "human evidence is observational only - no randomised trial",
                     "query": f"{base} randomized controlled trial"})
    hi = [r for r in human if (r.get("risk_of_bias_detail") or {}).get("overall") == "high"]
    if human and len(hi) > len(human) / 2:
        gaps.append({"type": "study quality", "detail": f"{len(hi)} of {len(human)} human studies at high risk of bias",
                     "query": f"{base} randomized double-blind"})
    if cat == "mixed":
        gaps.append({"type": "contradictory outcomes", "detail": "studies disagree on the outcome asked about",
                     "query": f"{base} meta-analysis"})
    pooled = next((p for p in dossier.get("rct_meta_analysis") or [] if p.get("k", 0) >= 2), None)
    rob = fail_safe(pooled) if pooled else None
    if pooled and pooled.get("prediction_interval"):
        lo, hi_ = pooled["prediction_interval"]
        null = 1.0 if pooled.get("kind") == "ratio" else 0.0
        if lo <= null <= hi_:
            gaps.append({"type": "imprecision", "detail": "the prediction interval includes no effect: a new trial could "
                                                          "plausibly find nothing", "query": f"{base} trial"})
    if indep == 1:
        gaps.append({"type": "lack of evidence", "detail": "a single independent human study - no replication",
                     "query": f"{base} replication"})
    # how likely is it that more evidence changes the conclusion?
    if cat in ("insufficient", "preliminary", "mixed"):
        value = "high"
    elif cat == "supported_with_limits":
        value = "high" if (rob and rob["one_null_trial_would_change_significance"]) else "moderate"
    else:
        value = "moderate" if (rob and rob["null_studies_to_lose_significance"] <= 2) else "low"
    stop = value == "low"
    return {"supported": {"category": cat, "label": v.get("label"), "direction": v.get("direction"),
                          "independent_human_studies": indep, "randomised_trials": len(rcts)},
            "gaps": gaps, "robustness": rob, "value": value,
            "next_tasks": [g["query"] for g in gaps][:5],
            "stop": stop, "reason": ("further studies are unlikely to change the conclusion (strongly supported and "
                                     "robust to added null studies)" if stop else
                                     f"more evidence could change the conclusion: {', '.join(sorted({g['type'] for g in gaps})) or cat}"),
            "version": VALUE_VERSION}


def markdown(a: Dict[str, Any]) -> str:
    L = ["## Would more research change the answer?", "",
         f"- **Value of further evidence: {a['value']}** - {a['reason']}",
         f"- supported now: {a['supported'].get('label') or a['supported']['category']}; "
         f"{a['supported']['independent_human_studies']} independent human stud(ies), "
         f"{a['supported']['randomised_trials']} randomised"]
    r = a.get("robustness")
    if r:
        L.append(f"- robustness: pooled z = {r['z']} over {r['k']} trials; "
                 + (f"about {r['null_studies_to_lose_significance']} null trial(s) of average size would remove significance"
                    if r["significant"] else "the pooled effect is not significant"))
    for g in a["gaps"]:
        L.append(f"- gap ({g['type']}): {g['detail']} - next search: `{g['query']}`")
    return "\n".join(L) + "\n"
