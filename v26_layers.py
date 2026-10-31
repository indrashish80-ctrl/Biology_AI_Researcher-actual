"""
v26_layers.py
=============
The v26 layers, attached to every finished run after the v25 layers. Each is ADDITIVE and wrapped: a failure is
recorded in dossier['v26']['errors'] and never stops the run. Switch any off with BIOSYN_V26_<NAME>=0.

    LIBRARY      local-library decision (searched locally / externally, snapshot date) + registered studies
    QUALITY      quality profile per study: risk of bias, design fit, controls, precision, missing data, selective
                 reporting, reproducibility, applicability, funding + conflicts, duration - relevance kept apart
    CAUSAL       design support, confounders, reverse causation / selection / immortal-time risks, causal diagram,
                 causal wording the evidence cannot carry
    PLAUSIBILITY dose (existing check) + route + duration + species + measured-vs-asked outcome
    RISK         relative vs absolute risk, NNT, recomputed percentages and changes, statistical vs meaningful size
    CONFIDENCE   calibrated confidence of each finding's direction (gold/calibration.json)
    SUBGROUP     pre-specified subgroup analyses of pooled results (exploratory)
    REGISTRY     registered studies without a published / posted result (missing evidence)
    VALUE        would more research change the answer? (research_value)
    REVIEW       the adversarial evidence review (evidence_reviewer) - runs last and sees everything above
"""
from __future__ import annotations

import os
from typing import Any, Dict, List

V26_VERSION = "26.0"


def _on(name: str) -> bool:
    return os.getenv(f"BIOSYN_V26_{name}", "1") != "0"


def apply(dossier: Dict[str, Any], report_text: str = "", engine: Any = None) -> Dict[str, Any]:
    info: Dict[str, Any] = {"version": V26_VERSION, "layers": [], "errors": []}
    md: List[str] = []
    top: List[str] = []
    issues: List[Dict[str, Any]] = []
    db = getattr(engine, "db", None)
    router = getattr(engine, "router", None)
    matrix = dossier.get("evidence_matrix") or {}
    rows = matrix.get("rows") or []
    question = dossier.get("research_question") or ""
    texts: Dict[str, str] = {}
    if db is not None:
        try:
            from v25_layers import _texts
            texts = _texts(db, rows)
        except Exception:
            texts = {}

    def ids_of(r: Dict[str, Any]) -> Dict[str, Any]:
        """PMID / DOI / PMCID of a matrix row (rows carry an 'identifier' string; the run database has the fields)."""
        out = {"pmid": None, "doi": None, "pmcid": None}
        if db is not None and r.get("paper_id") is not None:
            try:
                p = db.get_paper(r["paper_id"]) or {}
                out.update({k: p.get(k) for k in out})
            except Exception:
                pass
        ident = str(r.get("identifier") or "")
        import re as _re
        m = _re.search(r"PMID\s*(\d+)", ident)
        out["pmid"] = out["pmid"] or (m.group(1) if m else None)
        m = _re.search(r"(10\.\d{4,9}/\S+)", ident)
        out["doi"] = out["doi"] or (m.group(1).lower() if m else None)
        return out

    def text_of(r: Dict[str, Any]) -> str:
        t = texts.get(r.get("ref")) or ""
        if not t:
            t = " ".join(f.get("quote") or "" for f in r.get("findings") or [])
        return t

    def layer(name: str, fn) -> None:
        if not _on(name):
            return
        try:
            fn()
            info["layers"].append(name.lower())
        except Exception as exc:
            info["errors"].append(f"{name.lower()}: {type(exc).__name__}: {str(exc)[:200]}")

    def library():
        pr = dossier.get("prisma") or {}
        dec = (pr.get("counts") or {}).get("local_library")
        dossier["local_library"] = dec
        dossier["registered_studies"] = pr.get("registered_studies") or []
        if dec:
            md.append("## Local library\n\n"
                      f"- searched the local library first: {dec.get('local_hits')} records, {dec.get('strong_hits')} strong "
                      f"matches; snapshot {dec.get('snapshot')}\n"
                      f"- external databases: {'searched' if dec.get('use_online') else 'NOT searched'} - "
                      + "; ".join(dec.get("reasons") or []) + "\n")

    def quality():
        import quality_profile as qp
        sw = {str(s.get("ref") or s.get("trial")) for s in (dossier.get("executive_synthesis") or {}).get("outcome_switching") or []}
        lib = None
        try:
            import library_db as L
            lib = L.Library() if L.exists() else None
        except Exception:
            lib = None
        try:
            for r in rows:
                rec = lib.by_ids(**ids_of(r)) if lib else None
                r["quality_profile"] = qp.profile(r, text_of(r), question, rec, switched=r.get("ref") in sw)
        finally:
            if lib:
                lib.close()
        md.append(qp.markdown(rows))
        ind = [r.get("ref") for r in rows if (r.get("quality_profile") or {}).get("funding_conflicts", {}).get("concern")]
        dossier["funding_concerns"] = ind

    def causal():
        import causal_checks as cc
        from evidence_identity import question_frame
        for r in rows:
            r["causal"] = cc.assess(r, text_of(r))
        diagram = cc.dag(question, [r for r in rows if not r.get("secondary_report")])
        lang = cc.language(report_text, [r for r in rows if not r.get("secondary_report")],
                           question_frame(question).get("domain") == "human_medicine")
        dossier["causal"] = {"diagram": diagram, "language_issues": lang}
        md.append(cc.markdown(rows, diagram, lang))

    def plausibility():
        import dose_plausibility as dp
        for r in rows:
            r["plausibility"] = dp.assess(r, text_of(r), question)
        m = dp.markdown(rows)
        if m:
            md.append(m)

    def risk():
        import risk_numbers as rn
        tab, mean = [], []
        for r in rows:
            au = rn.audit_text(text_of(r))
            r["risk_numbers"] = {"events": au["events"][:4], "conflicts": au["conflicts"][:6]}
            for e in au["events"][:2]:
                tab.append((r.get("ref"), e))
            for f in r.get("findings") or []:
                try:
                    from meta_analysis import extract_effect
                    eff = extract_effect(f.get("quote") or "")
                except Exception:
                    eff = None
                if eff:
                    f["effect_size_meaning"] = rn.meaningful(eff, f.get("endpoint"), f.get("significant"),
                                                             n1=None, n2=None)
                    if f["effect_size_meaning"].get("warning"):
                        mean.append((r.get("ref"), f.get("endpoint"), f["effect_size_meaning"]))
        rep = rn.audit_text(report_text)
        dossier["risk_numbers"] = {"report": {"conflicts": rep["conflicts"], "framing": rep["framing"]},
                                   "trivial_but_significant": [{"ref": a, "endpoint": b, **c} for a, b, c in mean]}
        L = ["## Absolute and relative effects", ""]
        if tab:
            L += ["| study | events (treated / control) | absolute difference | relative risk | NNT / NNH |", "|---|---|---|---|---|"]
            for ref, e in tab:
                L.append(f"| {ref} | {e['treated'][0]}/{e['treated'][1]} vs {e['control'][0]}/{e['control'][1]} | "
                         f"{e['risk_difference_pp']:+.1f} pp (95% CI {e['risk_difference_ci_pp'][0]:+.1f} to "
                         f"{e['risk_difference_ci_pp'][1]:+.1f}) | {e.get('relative_risk', 'n/a')} | "
                         f"{e.get('nnt') or e.get('nnh') or 'n/a'}{' (NNH)' if e.get('nnh') else ''} |")
        for a, b, c in mean[:8]:
            L.append(f"- {a} {b}: {c['warning']} ({c.get('basis')})")
        for c in rep["conflicts"][:6]:
            L.append(f"- report arithmetic: '{c.get('text')}' does not recompute ({c.get('computed_pct', c.get('relative_change_pct'))})")
        for fr in rep["framing"][:6]:
            L.append(f"- report framing: '{fr['text']}' - {fr['advice']}")
        if len(L) > 2:
            md.append("\n".join(L) + "\n")

    def confidence():
        import confidence_calibration as ccal
        vals = []
        for r in rows:
            for f in r.get("findings") or []:
                f["direction_confidence"] = ccal.for_finding(f, r.get("intervention") or "", r.get("comparator") or "")
                if f["direction_confidence"].get("confidence") is not None:
                    vals.append(f["direction_confidence"]["confidence"])
        try:
            import json
            from pathlib import Path
            cal = json.loads((Path(__file__).resolve().parent / "gold" / "calibration.json").read_text(encoding="utf-8"))
            ece = (cal.get("heldout") or {}).get("ece")
        except Exception:
            ece = None
        dossier["direction_confidence"] = {"findings": len(vals), "mean": round(sum(vals) / len(vals), 3) if vals else None,
                                           "heldout_ece": ece}
        if vals:
            md.append("## Calibrated reading confidence\n\n"
                      f"- {len(vals)} finding(s) re-read by code; mean calibrated confidence that the stated direction is "
                      f"right: **{sum(vals) / len(vals):.2f}** (calibration checked on the held-out gold set, ECE {ece})\n"
                      "- a confidence is a measured frequency from doctor-labelled sentences, not the model's own claim\n")

    def subgroup():
        import meta_analysis as ma
        res = []
        for p in dossier.get("rct_meta_analysis") or []:
            res += ma.subgroup_analysis(p, rows)
        dossier["subgroup_analyses"] = res
        m = ma.subgroup_markdown(res)
        if m:
            md.append(m)

    def registry():
        import publication_bias as pb
        g = pb.registry_gap(dossier.get("registered_studies") or (dossier.get("prisma") or {}).get("registered_studies") or [],
                            [{**r, "pmid": ids_of(r)["pmid"]} for r in rows])
        dossier["registry_gap"] = g
        md.append(pb.registry_gap_markdown(g))

    def value():
        import research_value as rv
        a = rv.assess(dossier)
        dossier["research_value"] = a
        md.append(rv.markdown(a))

    def review():
        import evidence_reviewer as er
        res = er.review(dossier, report_text, db=db, router=router if os.getenv("BIOSYN_AI_CRITIC", "1") != "0" else None)
        dossier["adversarial_review"] = {k: v for k, v in res.items() if k != "markdown"}
        top.append(res["markdown"])
        for i in res["issues"]:
            if i["severity"] in ("critical", "major"):
                issues.append({"code": f"V26-{i['code']}", "severity": i["severity"], "detail": i["detail"]})

    def verification():
        import sim_verifier as cv
        ver = dossier.get("claim_verification") or (dossier.get("executive_synthesis") or {}).get("claim_verification")
        if ver:
            dossier["claim_verification"] = ver
            md.append(cv.markdown(ver))
        tests = dossier.get("hypothesis_tests")
        if tests:                                     # v27: every hypothesis tested in the laboratory
            import ai_lab
            md.append(ai_lab.markdown([{"claim": t["statement"], **t} for t in tests]))

    for name, fn in (("LIBRARY", library), ("QUALITY", quality), ("CAUSAL", causal), ("PLAUSIBILITY", plausibility),
                     ("RISK", risk), ("CONFIDENCE", confidence), ("SUBGROUP", subgroup), ("REGISTRY", registry),
                     ("VALUE", value), ("VERIFY", verification), ("REVIEW", review)):
        layer(name, fn)
    dossier["v26"] = info
    # the adversarial review goes right after the report's opening banners, before the evidence
    out_text = report_text
    if top and report_text:
        lines = report_text.split("\n")
        cut = 0
        for i, ln in enumerate(lines[:60]):
            if ln.startswith("## ") and i > 0:
                cut = i
                break
        out_text = "\n".join(lines[:cut] + [top[0]] + lines[cut:]) if cut else top[0] + "\n" + report_text
    return {"issues": issues, "markdown": "\n".join(m for m in md if m), "report_text": out_text, "info": info}
