"""
hypothesis_schema.py  (v22, improvement C)
------------------------------------------
Hypothesis vs discovery. A plausible mechanism is a HYPOTHESIS, not a finding; a correlation or a significant p-value
does not by itself establish causation. This module is purely ADDITIVE: it never deletes a hypothesis and never
rewrites its text. It (1) checks that each hypothesis carries the five parts a testable hypothesis needs, (2) assigns
an honest status label, (3) caps the strongest wording the evidence allows, and (4) flags overclaiming wording.

Required parts
    supporting_evidence      claim IDs (derived_from) that actually produced it
    competing_explanations   at least one rival explanation
    testable_prediction      at least one concrete, checkable prediction
    falsification_condition  at least one result that would refute it
    experiment               the experiment needed to test it (with a control and a measured variable)

Nothing here invents science: a missing part is reported as MISSING, never filled with made-up content.
The generic confounder checklist is shown only as a labelled prompt for the researcher, never as evidence.
"""
from __future__ import annotations

import os
import re
from typing import Any, Dict, List, Optional

ENABLED = os.environ.get("BIOSYN_HYPOTHESIS_SCHEMA", "1") != "0"

PARTS = ("supporting_evidence", "competing_explanations", "testable_prediction", "falsification_condition", "experiment")
PART_LABEL = {
    "supporting_evidence": "supporting evidence (claim IDs)",
    "competing_explanations": "competing explanation(s)",
    "testable_prediction": "testable prediction",
    "falsification_condition": "falsification condition",
    "experiment": "experiment that would test it",
}

# ---- wording: what each evidence level may be called -------------------------------------------------------------
OVERCLAIM_RX = re.compile(
    r"\b(prov(?:es|ed|en)|demonstrat(?:es|ed)\s+that|establish(?:es|ed)|confirm(?:s|ed)|discover(?:y|ed|s)|"
    r"definitively|conclusively|is\s+the\s+cause\s+of|causes)\b", re.I)
CAUSAL_RX = re.compile(r"\b(causes?|caused|leads?\s+to|drives?|induces?|results?\s+in|is\s+responsible\s+for|"
                       r"mediates?|due\s+to)\b", re.I)
CORRELATIONAL_TYPES = {"correlational", "association", "observational", "cross_sectional", "case_control", "cohort",
                       "in_silico", "computational", "mechanistic", "intermediate", "inferred"}
INTERVENTIONAL_TYPES = {"interventional", "randomized", "rct", "experimental", "knockout", "knockdown", "rescue"}

# Status ladder (low -> high). A hypothesis can only be promoted by an experiment result, never by wording.
STATUS_SPECULATIVE = "speculative"            # missing parts, or no derived evidence
STATUS_TESTABLE = "testable_hypothesis"       # all five parts present; untested
STATUS_SUPPORTED = "supported_by_data"        # an experiment result supported it (HypothesisEvaluator)
STATUS_REFUTED = "refuted_by_data"
STATUS_WEAKENED = "weakened_by_data"

GENERIC_CONFOUNDERS = [
    "reverse causation (the outcome changes the exposure, not the reverse)",
    "confounding by an unmeasured shared cause",
    "selection / publication bias in the supporting studies",
    "the effect is specific to the model system (cell line, species, dose) and does not generalise",
]


def _as_list(v: Any) -> List[Any]:
    if v is None or v == "":
        return []
    if isinstance(v, (list, tuple, set)):
        return [x for x in v if x not in (None, "")]
    if isinstance(v, dict):
        return [v] if any(v.values()) else []
    return [v]


def _text(v: Any) -> str:
    return v if isinstance(v, str) else ""


def _parse_experiment(h: Dict[str, Any]) -> Dict[str, Any]:
    exp = h.get("next_experiment") or h.get("experiment")
    if isinstance(exp, str):
        s = exp.strip()
        if s.startswith("{"):
            try:
                import json
                exp = json.loads(s)
            except ValueError:
                exp = {"experiment_title": s}
        else:
            exp = {"experiment_title": s} if s else {}
    return exp if isinstance(exp, dict) else {}


def _experiment_ok(exp: Dict[str, Any]) -> bool:
    title = _text(exp.get("experiment_title")) or _text(exp.get("title")) or _text(exp.get("name"))
    controls = _text(exp.get("controls")) or _text(exp.get("control"))
    var = (_text(exp.get("independent_variable")) or _text(exp.get("dependent_variable"))
           or _text(exp.get("key_measurements")))
    return bool(title.strip() and controls.strip() and var.strip())


def _derived_ids(h: Dict[str, Any]) -> List[Any]:
    ids = _as_list(h.get("derived_from"))
    se = h.get("supporting_evidence")
    if isinstance(se, dict):
        ids += _as_list(se.get("derived_from"))
    elif isinstance(se, list):
        ids += se
    seen, out = set(), []
    for i in ids:
        k = str(i)
        if k not in seen:
            seen.add(k)
            out.append(i)
    return out


def evidence_ceiling(h: Dict[str, Any]) -> str:
    """The strongest relationship the cited evidence supports: 'none' | 'association' | 'mechanistic_proposal' | 'causal'.
    'causal' needs an interventional result on the SAME hypothesis (a recorded data test), not wording."""
    st = (h.get("status") or "").lower()
    ev = h.get("evaluation") if isinstance(h.get("evaluation"), dict) else {}
    design = str(ev.get("design") or "").lower()
    # a significant difference in a dataset is NOT causal by itself: it needs a dataset whose groups were assigned by
    # an intervention (randomised / controlled manipulation), declared in evaluation["design"]
    if st in ("supported", STATUS_SUPPORTED) and design in INTERVENTIONAL_TYPES:
        return "causal"
    if not _derived_ids(h):
        return "none"
    kinds = {str(k).lower() for k in _as_list(h.get("evidence_types"))}
    if kinds & INTERVENTIONAL_TYPES and not (kinds <= CORRELATIONAL_TYPES):
        return "mechanistic_proposal"       # interventional evidence exists, but not on THIS hypothesis yet
    ms = (h.get("mechanism_status") or "")
    if isinstance(ms, dict):
        ms = ms.get("status") or ""
    return "mechanistic_proposal" if "mechan" in str(ms).lower() else "association"


def check_hypothesis(h: Dict[str, Any]) -> Dict[str, Any]:
    """Return the structured completeness / wording audit of ONE hypothesis (the input is not modified)."""
    stmt = _text(h.get("statement"))
    present = {
        "supporting_evidence": bool(_derived_ids(h)),
        "competing_explanations": bool(_as_list(h.get("competing_explanations"))),
        "testable_prediction": bool([p for p in _as_list(h.get("predictions") or h.get("prediction")) if _text(p).strip() or isinstance(p, dict)]),
        "falsification_condition": bool([c for c in _as_list(h.get("falsification_conditions") or h.get("falsification_condition")) if _text(c).strip() or isinstance(c, dict)]),
        "experiment": _experiment_ok(_parse_experiment(h)),
    }
    missing = [p for p in PARTS if not present[p]]
    ceiling = evidence_ceiling(h)
    wording: List[str] = []
    m = OVERCLAIM_RX.search(stmt)
    if m and ceiling != "causal":
        wording.append(f"overclaiming word '{m.group(0)}' - the evidence supports at most: {ceiling.replace('_', ' ')}")
    if CAUSAL_RX.search(stmt) and ceiling in ("none", "association"):
        wording.append("causal wording, but the cited evidence is only "
                       + ("absent" if ceiling == "none" else "associational") + " - say 'is associated with' or 'may'")
    st = (h.get("status") or "").lower()
    if st in ("supported", STATUS_SUPPORTED):
        label = STATUS_SUPPORTED
    elif st in ("refuted", STATUS_REFUTED):
        label = STATUS_REFUTED
    elif st in ("weakened", STATUS_WEAKENED):
        label = STATUS_WEAKENED
    else:
        label = STATUS_TESTABLE if not missing else STATUS_SPECULATIVE
    return {
        "complete": not missing, "present": present, "missing": missing, "label": label, "evidence_ceiling": ceiling,
        "wording_issues": wording,
        "is_discovery": label == STATUS_SUPPORTED and ceiling == "causal",
        "generic_checklist": (GENERIC_CONFOUNDERS if "competing_explanations" in missing and ceiling != "causal" else []),
    }


def annotate(hypotheses: Optional[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    """Add a 'schema' key to every hypothesis (in place). Existing keys are never changed."""
    out = hypotheses if hypotheses is not None else []
    if not ENABLED:
        return out
    for h in out:
        if isinstance(h, dict):
            try:
                h["schema"] = check_hypothesis(h)
            except Exception as exc:                  # never break a run over an audit
                h["schema"] = {"error": f"{type(exc).__name__}: {exc}"}
    return out


def summary(hypotheses: Optional[List[Dict[str, Any]]]) -> Dict[str, Any]:
    hs = [h for h in hypotheses or [] if isinstance(h, dict) and isinstance(h.get("schema"), dict) and "label" in h["schema"]]
    from collections import Counter
    labels = Counter(h["schema"]["label"] for h in hs)
    miss = Counter(m for h in hs for m in h["schema"]["missing"])
    return {"n": len(hs), "labels": dict(labels), "missing_parts": dict(miss),
            "n_complete": sum(1 for h in hs if h["schema"]["complete"]),
            "n_discoveries": sum(1 for h in hs if h["schema"]["is_discovery"]),
            "n_wording_issues": sum(len(h["schema"]["wording_issues"]) for h in hs)}


def audit_issues(hypotheses: Optional[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    """Issues in the report_auditor format (no repair action - they inform, they never trigger a re-run)."""
    s = summary(hypotheses)
    issues = []
    if s["n"] and s["n_complete"] < s["n"]:
        issues.append({"code": "Q10", "severity": "minor", "action": None,
                       "repair": "labelled 'speculative' in the report",
                       "detail": f"{s['n'] - s['n_complete']} of {s['n']} hypothesis(es) lack required parts: "
                                 + ", ".join(f"{PART_LABEL[k]} x{v}" for k, v in s["missing_parts"].items())})
    if s["n_wording_issues"]:
        issues.append({"code": "Q11", "severity": "minor", "action": None, "repair": "flagged in the report",
                       "detail": f"{s['n_wording_issues']} hypothesis wording issue(s): causal / discovery wording the "
                                 f"evidence does not support"})
    return issues


def markdown(hypotheses: Optional[List[Dict[str, Any]]]) -> str:
    hs = [h for h in hypotheses or [] if isinstance(h, dict) and isinstance(h.get("schema"), dict) and "label" in h["schema"]]
    if not hs:
        return ""
    s = summary(hs)
    lines = ["\n## Hypotheses are not discoveries\n",
             "_A hypothesis becomes a finding only after an experiment supports it. Each hypothesis below is checked for "
             "the five parts a testable hypothesis needs; missing parts are reported, never invented._\n",
             f"- {s['n']} hypothesis(es): {s['n_complete']} complete, {s['n'] - s['n_complete']} speculative; "
             f"{s['n_discoveries']} supported by a data test."]
    for i, h in enumerate(hs, 1):
        sc = h["schema"]
        tag = sc["label"].replace("_", " ")
        stmt = _text(h.get("statement"))[:140]
        line = f"- **H{h.get('id', i)}** [{tag}; evidence ceiling: {sc['evidence_ceiling'].replace('_', ' ')}] {stmt}"
        if sc["missing"]:
            line += "  \n  Missing: " + ", ".join(PART_LABEL[m] for m in sc["missing"])
        for w in sc["wording_issues"]:
            line += f"  \n  Wording: {w}"
        if sc["generic_checklist"]:
            line += "  \n  Generic checklist (NOT evidence, for the researcher to consider): " + "; ".join(sc["generic_checklist"])
        lines.append(line)
    return "\n".join(lines) + "\n"
