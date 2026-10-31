"""
v22_layers.py
-------------
The single entry point that attaches the three v22 layers to a finished run. Every step is ADDITIVE and wrapped so a
failure here can never break or change a run: existing dossier keys are never modified; new keys are
dossier["evidence_record"], dossier["hypothesis_audit"] and dossier["v22"].

Switches (all default ON; set to 0 to turn a layer off):
    BIOSYN_EVIDENCE_RECORD     canonical evidence record + consistency check   (improvement B)
    BIOSYN_HYPOTHESIS_SCHEMA   hypothesis completeness / wording check         (improvement C)
    BIOSYN_GOLD_EVAL           score the run against gold/ if verified papers exist (improvement A)
"""
from __future__ import annotations

import os
from typing import Any, Dict, List


def apply(dossier: Dict[str, Any], report_text: str = "") -> Dict[str, Any]:
    """Returns {"issues": [...blocking-style issues], "markdown": "...section to append to the report"}."""
    issues: List[Dict[str, Any]] = []
    md: List[str] = []
    info: Dict[str, Any] = {"layers": [], "errors": []}

    # ---- B: canonical evidence record + consistency
    if os.environ.get("BIOSYN_EVIDENCE_RECORD", "1") != "0":
        try:
            import evidence_record as er
            matrix = dossier.get("evidence_matrix") or {}
            rec = er.build_record(matrix)
            probs = er.check_consistency(rec, matrix, report_text)
            dossier["evidence_record"] = rec
            info["layers"].append("evidence_record")
            info["record_problems"] = len(probs)
            issues += probs
            md.append(er.markdown(rec, probs))
        except Exception as exc:
            info["errors"].append(f"evidence_record: {type(exc).__name__}: {exc}")

    # ---- C: hypotheses vs discoveries
    if os.environ.get("BIOSYN_HYPOTHESIS_SCHEMA", "1") != "0":
        try:
            import hypothesis_schema as hs
            hyps = dossier.get("hypotheses") or []
            hs.annotate(hyps)
            dossier["hypothesis_audit"] = hs.summary(hyps)
            info["layers"].append("hypothesis_schema")
            md.append(hs.markdown(hyps))
            info["hypothesis_notes"] = [i["detail"] for i in hs.audit_issues(hyps)]   # informational only
        except Exception as exc:
            info["errors"].append(f"hypothesis_schema: {type(exc).__name__}: {exc}")

    # ---- A: gold-standard score (only when verified gold papers exist)
    if os.environ.get("BIOSYN_GOLD_EVAL", "1") != "0":
        try:
            import gold_eval as ge
            gold = ge.load_gold()
            if gold:
                res = ge.evaluate((dossier.get("evidence_matrix") or {}).get("rows") or [], gold, only_in_run=True)
                if res["gold_papers"]:             # only when the run read at least one gold paper
                    dossier["gold_eval"] = res
                    info["layers"].append("gold_eval")
                    md.append("\n" + ge.markdown(res))
        except Exception as exc:
            info["errors"].append(f"gold_eval: {type(exc).__name__}: {exc}")

    dossier["v22"] = info
    return {"issues": issues, "markdown": "".join(md)}
