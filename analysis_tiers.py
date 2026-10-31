"""
analysis_tiers.py  (v25)
========================
Three analysis tiers, and automatic escalation when the evidence is thin or conflicting.

    quick      abstracts of the top papers, fast sources, one reader, no repair round        (minutes)
    research   full texts of up to 25 papers, citation chasing, a second independent reader    (the old 'full dossier')
    deep       up to 300 screened / 50 deep-read, every source, second reader, two repair rounds

Escalation rules (escalate(dossier) -> {'escalate', 'to', 'reasons'}), checked after each tier:
    thin          fewer than 3 independent primary studies answer the question, or none in humans for a question
                  about people, or the calibrated verdict is 'insufficient' / 'preliminary'
    conflicting   the calibrated verdict is 'mixed', or the run found direct contradictions on the asked endpoint
    low reading   more than half the findings were read from abstracts only
    incomplete    fewer than 95% of AI calls succeeded (completeness_gate)
A tier is never repeated, and 'deep' is the ceiling. Every escalation and its reasons are stored in the dossier.

    run_tiered(engine, question, start='quick', ceiling='deep', **investigate_kwargs)
"""
from __future__ import annotations

import time
from typing import Any, Callable, Dict, List, Optional

TIERS_VERSION = "1.0"
ORDER = ["quick", "research", "deep"]
PARAMS: Dict[str, Dict[str, Any]] = {
    "quick": {"screen_n": 40, "deep_read_k": 8, "citation_chasing": False, "use_structures": False, "auto_heal": False,
              "abstract_only": True, "quick": True, "second_reader": "0"},
    "research": {"screen_n": 150, "deep_read_k": 25, "citation_chasing": True, "auto_heal": True,
                 "abstract_only": False, "quick": False, "second_reader": "1"},
    "deep": {"screen_n": 300, "deep_read_k": 50, "citation_chasing": True, "auto_heal": True,
             "abstract_only": False, "quick": False, "second_reader": "1"},
}


def _primary_answering(matrix: Dict[str, Any]) -> List[Dict[str, Any]]:
    out = []
    for r in (matrix or {}).get("rows") or []:
        if r.get("secondary_report") or (r.get("paper_relevance") or {}).get("ok") is False:
            continue
        if any(f.get("answers_question") is not False for f in r.get("findings") or []):
            out.append(r)
    return out


def escalate(dossier: Dict[str, Any], tier: str) -> Dict[str, Any]:
    reasons: List[str] = []
    matrix = dossier.get("evidence_matrix") or {}
    rows = _primary_answering(matrix)
    independent = {r.get("trial_group") or r.get("ref") for r in rows}
    if len(independent) < 3:
        reasons.append(f"thin evidence: {len(independent)} independent primary stud(ies) answer the question")
    q = (dossier.get("research_question") or "").lower()
    try:
        from evidence_identity import question_frame
        human_q = question_frame(q).get("domain") == "human_medicine"
    except Exception:
        human_q = False
    if human_q and not any(r.get("evidence_type") in ("human_clinical", "human_observational") for r in rows):
        reasons.append("thin evidence: no human study answers a question about people")
    v = dossier.get("calibrated_verdict") or {}
    if v.get("category") in ("insufficient", "preliminary"):
        reasons.append(f"verdict '{v.get('label')}'")
    if v.get("category") == "mixed":
        reasons.append("conflicting evidence: verdict 'Mixed evidence'")
    cs = dossier.get("contradiction_summary") or {}
    direct = cs.get("direct_contradictions") or cs.get("same_system_disagreements") or 0
    if direct:
        reasons.append(f"conflicting evidence: {direct} direct contradiction(s)")
    basis = [r.get("text_basis") for r in rows]
    if basis and sum(1 for b in basis if (b or "abstract") == "abstract") > len(basis) / 2:
        reasons.append("low reading confidence: most findings read from abstracts only")
    comp = dossier.get("completeness") or {}
    if comp and comp.get("complete") is False:
        reasons.append(f"incomplete run: AI success rate {comp.get('success_rate')}")
    i = ORDER.index(tier) if tier in ORDER else 0
    nxt = ORDER[i + 1] if i + 1 < len(ORDER) else None
    # v26: escalate only when more evidence could change the conclusion (research_value: value of further evidence)
    value = None
    try:
        import research_value as rv
        va = rv.assess(dossier)
        value = va["value"]
        if va["stop"] and reasons:
            reasons = [f"not escalated: {va['reason']}"]
            return {"escalate": False, "to": None, "reasons": reasons, "tier": tier, "value": value,
                    "next_tasks": va["next_tasks"]}
    except Exception:
        va = None
    return {"escalate": bool(reasons) and nxt is not None, "to": nxt if reasons else None, "reasons": reasons,
            "tier": tier, "value": value, "next_tasks": (va or {}).get("next_tasks")}


def kwargs_for(tier: str, base: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    kw = dict(base or {})
    p = dict(PARAMS[tier])
    p.pop("second_reader", None)
    kw.update(p)
    return kw


def run_tiered(engine: Any, question: str, start: str = "quick", ceiling: str = "deep",
               progress: Optional[Callable[[str, str], None]] = None, **investigate_kwargs) -> Dict[str, Any]:
    """Run `start`, escalate while the rules say so (never past `ceiling`). Returns the last dossier with
    dossier['tiers'] = [{'tier', 'seconds', 'decision'}]."""
    history = []
    tier = start if start in ORDER else "quick"
    top = ORDER.index(ceiling) if ceiling in ORDER else len(ORDER) - 1
    dossier: Dict[str, Any] = {}
    while True:
        # the second reader follows the tier through the engine's quick flag (off in 'quick', on otherwise) - no
        # process-wide setting, so concurrent jobs cannot change each other's tier
        if progress:
            progress("tier", f"Analysis tier: {tier}")
        t0 = time.time()
        dossier = engine.investigate(question, **kwargs_for(tier, investigate_kwargs))
        try:
            if not dossier.get("calibrated_verdict"):
                import verdict_calibration as vc
                dossier["calibrated_verdict"] = vc.from_matrix(dossier.get("evidence_matrix") or {}, question)
        except Exception:
            pass
        dec = escalate(dossier, tier)
        history.append({"tier": tier, "seconds": round(time.time() - t0, 1), "decision": dec})
        if not dec["escalate"] or ORDER.index(dec["to"]) > top:
            break
        if progress:                           # v26.1: the re-run is announced with its reason (it is not a reset)
            why = "; ".join(str(r) for r in (dec.get("reasons") or [])[:2]) or "the evidence is thin or conflicting"
            progress("pass", f"moving up from the {tier} level to the {dec['to']} level, because {why}. "
                             f"Results already found are reused where possible.")
        tier = dec["to"]
    dossier["tiers"] = history
    dossier["analysis_tier"] = tier
    return dossier


def markdown(history: List[Dict[str, Any]]) -> str:
    if not history:
        return ""
    L = ["## Analysis tiers", "", "| tier | time | next step |", "|---|---|---|"]
    for h in history:
        d = h["decision"]
        nxt = f"escalated to **{d['to']}**: " + "; ".join(d["reasons"]) if d["escalate"] else \
            ("stopped: " + ("; ".join(d["reasons"]) + " (ceiling reached)" if d["reasons"] else "evidence sufficient"))
        L.append(f"| {h['tier']} | {h['seconds']:.0f} s | {nxt} |")
    return "\n".join(L) + "\n"
