"""
v25_layers.py
=============
One entry point that attaches every v25 layer to a finished run. Like v22_layers, each layer is ADDITIVE and wrapped:
a failure is recorded in dossier['v25']['errors'] and never stops or changes the run.

Layers (switch any off with BIOSYN_V25_<NAME>=0):
    GUARD       quote / number guard and second-reader statistics           dossier['extraction_guard']
    VERDICT     calibrated verdict category                                 dossier['calibrated_verdict']
    META        REML / Hartung-Knapp meta-analysis of randomised trials     dossier['rct_meta_analysis']
    BIAS        funnel plot, Egger, trim-and-fill, registry vs published    dossier['publication_bias']
    GRAPH       evidence dependency graph                                   dossier['evidence_graph']
    SCALE       cross-scale verification chain                              dossier['cross_scale']
    KG          knowledge-graph edges in entity_relations                   dossier['knowledge_graph']
    ROB         RoB 2 / ROBINS-I / SYRCLE tables (already on matrix rows)   (report section)
    NUMERIC     abstract-vs-table numerical audit                           dossier['numeric_audit']
    STATUS      publication-status check of the included papers            dossier['publication_status']
    MEMORY      paper versions + dossier registry                           dossier['scientific_memory']
    REPRO       reproducibility manifest (research_dossier_<id>.manifest.json)
    METRICS     latency / AI calls / cost / cache hit rates                 dossier['metrics']
    SECTIONS    A published evidence / B simulation / C AI hypotheses       (top of the report)
"""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Dict, List

V25_VERSION = "25.0"


def _on(name: str) -> bool:
    return os.getenv(f"BIOSYN_V25_{name}", "1") != "0"


def _texts(db: Any, rows: List[Dict[str, Any]]) -> Dict[str, str]:
    out = {}
    for r in rows:
        pid = r.get("paper_id")
        if pid is None:
            continue
        try:
            secs = db.get_sections(pid)
            out[r.get("ref")] = "\n".join(s.get("text") or "" for s in secs)
        except Exception:
            pass
    return out


def apply(dossier: Dict[str, Any], report_text: str = "", engine: Any = None) -> Dict[str, Any]:
    """Returns {'issues': [...], 'markdown_top': str, 'markdown': str, 'report_text': str}."""
    info: Dict[str, Any] = {"version": V25_VERSION, "layers": [], "errors": []}
    md: List[str] = []
    issues: List[Dict[str, Any]] = []
    db = getattr(engine, "db", None)
    router = getattr(engine, "router", None)
    matrix = dossier.get("evidence_matrix") or {}
    rows = matrix.get("rows") or []
    rq = dossier.get("research_question_id")
    pids = dossier.get("included_paper_ids") or [r.get("paper_id") for r in rows if r.get("paper_id")]

    def layer(name: str, fn) -> None:
        if not _on(name):
            return
        try:
            fn()
            info["layers"].append(name.lower())
        except Exception as exc:
            info["errors"].append(f"{name.lower()}: {type(exc).__name__}: {str(exc)[:200]}")

    def guard():
        from claim_extraction import guard_summary
        g = guard_summary(pids)
        dossier["extraction_guard"] = g
        if g.get("proposed"):
            md.append("## Extraction guard (verbatim quotes, numbers, second reader)\n\n"
                      f"- {g['proposed']} finding(s) proposed by the AI, {g['kept']} kept, **{g['dropped']} dropped** "
                      f"(quote not verbatim in the paper, or a number the paper does not contain)"
                      + ("; reasons: " + ", ".join(f"{k} {v}" for k, v in g['drop_reasons'].items()) if g.get("drop_reasons") else "")
                      + "\n" + (f"- second reader: {g['agree']} agree, {g['disagree']} disagree (quarantined for review), "
                                f"{g['not_found']} not found by it, {g['missed_by_first']} found only by it (review queue)"
                                + (f"; agreement {g['reader_agreement']:.0%}" if g.get("reader_agreement") is not None else "")
                                if (g.get("agree") or g.get("disagree") or g.get("not_found")) else "- second reader: not run")
                      + "\n- texts read: " + ", ".join(f"{k} {v}" for k, v in (g.get("by_basis") or {}).items()) + "\n")

    def verdict():
        import verdict_calibration as vc
        v = vc.from_matrix(matrix, dossier.get("research_question") or "")
        dossier["calibrated_verdict"] = v
        md.append(vc.markdown(v))

    def meta():
        import meta_analysis as ma
        res = ma.rct_meta(rows)
        dossier["rct_meta_analysis"] = res
        if res:
            md.append(ma.advanced_markdown(res))
        if _on("BIAS"):
            import publication_bias as pb
            pooled = [p for p in (res or []) if p.get("k", 0) >= 3] or [p for p in ma.analyse(rows) if p.get("k", 0) >= 3]
            out = [pb.analyse(p) for p in pooled]
            reg = pb.registry_section(dossier.get("executive_synthesis") or {})
            dossier["publication_bias"] = {"analyses": [{k: v for k, v in o.items() if k != "funnel_svg"} for o in out],
                                           "registry": reg}
            md.append(pb.markdown(out, reg))

    def graph():
        import evidence_graph as eg
        g = eg.build(matrix, _texts(db, rows) if db is not None else {})
        dossier["evidence_graph"] = g
        md.append(eg.markdown(g))
        if g.get("double_counted"):
            issues.append({"code": "V25-DOUBLE", "severity": "major", "detail": f"{len(g['double_counted'])} primary "
                           f"stud(ies) are also counted inside an included review"})

    def scale():
        import cross_scale as cs
        c = cs.chain(matrix)
        dossier["cross_scale"] = {k: v for k, v in c.items() if k != "levels"} | {"levels": {k: len(v) for k, v in c["levels"].items()}}
        md.append(cs.markdown(c))

    def kg():
        if db is None or rq is None:
            return
        import knowledge_graph as kgm
        dossier["knowledge_graph"] = kgm.build(db, rq, dossier.get("protein_interactions"))
        md.append(kgm.markdown(kgm.edges_for(db, rq)))

    def rob():
        import risk_of_bias as rb
        md.append(rb.markdown(rows))

    def numeric():
        if db is None:
            return
        import numeric_audit as na
        aud = []
        for r in rows:
            pid = r.get("paper_id")
            if pid is None or r.get("text_basis") == "abstract":
                continue
            p = db.get_paper(pid) or {}
            tabs = "\n".join(f"{t.get('caption') or ''}\n" + "\n".join(" | ".join(x) for x in (t.get("rows") or []))
                             for t in db.get_tables(pid))
            res_txt = "\n".join(s.get("text") or "" for s in db.get_sections(pid) if s.get("section_type") in ("results", "results_discussion"))
            a = na.audit_paper(p.get("abstract") or "", tabs, res_txt)
            aud.append({"ref": r.get("ref"), "audit": a})
        dossier["numeric_audit"] = aud
        md.append(na.markdown(aud))

    def status():
        if db is None or not pids:
            return
        import publication_status as ps
        res = ps.recheck(db, older_than_days=30, fetch=not os.getenv("BIOSYN_OFFLINE"), paper_ids=pids)
        dossier["publication_status"] = res
        bad = [c for c in res.get("changed") or [] if c["status"] in ("retracted", "expression_of_concern")]
        if bad:
            issues.append({"code": "V25-RETRACTED", "severity": "critical",
                           "detail": "; ".join(f"{c.get('title') or c['paper_id']}: {c['status']}" for c in bad)})
        if res.get("changed"):
            md.append(ps.markdown(res))

    def memory():
        if db is None:
            return
        import scientific_memory as sm
        snap = sm.snapshot(db, pids)
        sm.register_dossier(db, dossier)
        dossier["scientific_memory"] = {"new_paper_versions": len(snap["new"]), "changed_papers": snap["changed"]}
        if snap["changed"]:
            imp = sm.impact(db, snap["changed"])
            dossier["scientific_memory"]["impact"] = imp
            md.append(sm.markdown(imp))

    def repro():
        import dossier_repro as dr
        m = dr.manifest(dossier)
        dossier["manifest_fingerprint"] = m["claims_fingerprint"]
        from core_utils import output_path
        path = output_path(f"research_dossier_{rq}.manifest.json")
        rp = dossier.get("report_path")
        if rp:
            path = Path(rp).with_name(f"research_dossier_{rq}.manifest.json")
        dr.write(m, str(path))
        dossier["manifest_path"] = str(path)

    def metrics():
        import run_metrics as rm
        for k, v in (getattr(engine, "run_context", None) or {}).items():
            dossier.setdefault(k, v)                     # queue wait / server load from the web server (v26)
        m = rm.collect(dossier, router)
        dossier["metrics"] = m
        rm.record_run(m, dossier.get("research_question") or "", dossier.get("analysis_tier") or dossier.get("speed") or "")
        md.append(rm.markdown(m))

    def review():
        if db is None:
            return
        import review_store as rs
        s = rs.summary(db)
        dossier["review_queue"] = s
        if s.get("open"):
            md.append(f"## Review queue\n\n{s['open']} finding(s) wait for a person (readers disagreed, a second reader found "
                      f"something the first missed, or a paper's status changed). Open them on the website: Review.\n")

    banners: List[str] = []

    def completeness():
        # v24.2 completeness gate, now wired: an INCOMPLETE banner when fewer than 95% of AI calls succeeded
        import completeness_gate as cg
        tot = router.health.totals() if router is not None and hasattr(getattr(router, "health", None), "totals") else {}
        res = cg.check_completeness(tot or {})
        dossier["completeness"] = {"complete": res["is_complete"], "success_rate": round(res["success_rate"], 3),
                                   "calls": res["calls"], "failures": res["failures"]}
        if not res["is_complete"]:
            banners.append("> " + res["banner"])
            issues.append({"code": "V25-INCOMPLETE", "severity": "major", "detail": res["banner"]})
        # v26.2: which errors happened, so a failed run explains itself (keys are redacted)
        errs = router.health.error_summary(8) if router is not None and hasattr(getattr(router, "health", None), "error_summary") else []
        if errs:
            dossier["completeness"]["errors"] = errs
            rows = "\n".join(f"| {e['provider']} | {e['kind'].replace('_', ' ')} | {e['count']} | "
                             f"{e['example'][:160].replace('|', '/')} |" for e in errs)
            md.append("## AI provider problems in this run\n\n"
                      f"{tot.get('requests', 0)} AI requests; {tot.get('requests_failed', 0)} failed on every provider; "
                      f"{tot.get('failures', 0)} provider attempts failed in total (a failed attempt is retried on the "
                      f"next provider).\n\n| Provider | Kind | Count | Example |\n|---|---|---|---|\n{rows}\n")

    def strict():
        # v24.2 strict answer layer, now wired (BIOSYN_STRICT_ANSWER=1): the bottom line from RCTs / meta-analyses only
        import strict_answer_layer as sal
        if not sal.STRICT_MODE and os.getenv("BIOSYN_STRICT_ANSWER") != "1":
            return
        claims = []
        for r in rows:
            design = "rct" if r.get("evidence_type") == "human_clinical" and re.search(
                r"random", f"{r.get('design') or ''} {r.get('title') or ''}", re.I) else \
                "meta_analysis" if r.get("secondary_report") and re.search(r"meta-?analys", r.get("title") or "", re.I) else \
                {"human_observational": "observational_human", "animal": "animal", "cell_culture": "cell_culture",
                 "in_silico": "computational"}.get(r.get("evidence_type"), "cell_culture")
            for f in r.get("findings") or []:
                claims.append({"ref": r.get("ref"), "study_type": design, "endpoint": f.get("endpoint"),
                               "direction": f.get("direction"), "quote": f.get("quote")})
        t = sal.tier_evidence(claims, strict_mode=True)
        dossier["strict_answer"] = {"answer": len(t["answer_claims"]), "context": len(t["context_claims"])}
        L = ["## Strict answer (randomised trials and meta-analyses only)", ""]
        L += [f"- {c['ref']}: {c['endpoint']} {c['direction']} - “{(c.get('quote') or '')[:150]}”" for c in t["answer_claims"][:20]] or \
             ["No randomised trial or meta-analysis answers the question; everything else is context and mechanism."]
        L.append(f"\n{len(t['context_claims'])} other finding(s) are context & mechanism (observational, animal, cell, in silico).")
        md.append("\n".join(L) + "\n")

    for name, fn in (("COMPLETENESS", completeness), ("STRICT", strict), ("GUARD", guard), ("VERDICT", verdict), ("META", meta), ("GRAPH", graph), ("SCALE", scale),
                     ("KG", kg), ("ROB", rob), ("NUMERIC", numeric), ("STATUS", status), ("MEMORY", memory),
                     ("REVIEW", review), ("REPRO", repro), ("METRICS", metrics)):
        layer(name, fn)
    top = ""
    new_report = report_text or ""
    if _on("SECTIONS"):
        try:
            import dossier_sections as ds
            top = ds.markdown(dossier)
            new_report = ds.label_report(new_report)
            info["layers"].append("sections")
        except Exception as exc:
            info["errors"].append(f"sections: {type(exc).__name__}: {str(exc)[:200]}")
    if dossier.get("tiers"):
        try:
            import analysis_tiers as at
            md.insert(0, at.markdown(dossier["tiers"]))
        except Exception:
            pass
    dossier["v25"] = info
    body = "\n".join(x for x in md if x)
    if top and new_report:
        # the three-section guide goes after the report's title and any banner lines right under it ('> Quick scan ...',
        # '> INCOMPLETE ...'), so the banners stay the first thing a reader sees
        lines = new_report.split("\n")
        i = 1
        while i < len(lines) and (not lines[i].strip() or lines[i].lstrip().startswith(">")):
            i += 1
        new_report = "\n".join(lines[:i] + banners) + "\n" + top + "\n" + "\n".join(lines[i:])
    elif banners and new_report:
        head, sep, rest = new_report.partition("\n")
        new_report = head + sep + "\n".join(banners) + "\n" + rest
    return {"issues": issues, "markdown": ("\n\n" + body) if body else "", "markdown_top": top, "report_text": new_report}
