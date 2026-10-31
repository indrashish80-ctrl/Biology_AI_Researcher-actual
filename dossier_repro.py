"""
dossier_repro.py  (v25)
=======================
Reproducible dossiers: everything needed to re-run a dossier and see exactly what changed.

The MANIFEST (research_dossier_<id>.manifest.json), written at the end of every run:
    question (as typed and as searched), date and time of every search, the queries sent to each source and the
    number of records each returned, every screening decision (included / excluded + reason), the deep-read list with
    its text basis, the settings (screen size, deep-read budget, sources, tier), the AI providers and models, the code
    versions (extractor, audit, classifier, guard, retrieval ...), and content hashes of the included papers.

    manifest(dossier) -> dict                         write(manifest, path)
    rerun(manifest, engine)                           runs the same question with the same settings
    diff(old_dossier, new_dossier) -> what changed    (papers in / out, claims added / removed / changed, verdict)

Claims are compared by CONTENT (paper identifier + endpoint + direction + quote), never by database id - the ids are
renumbered on every run, which made v24.2's regression_tracker report every claim as removed and added again.
"""
from __future__ import annotations

import hashlib
import json
import re
import time
from pathlib import Path
from typing import Any, Dict, Optional

REPRO_VERSION = "1.0"


def _versions() -> Dict[str, str]:
    try:
        import gold_bench
        return gold_bench.versions()
    except Exception:
        return {}


def manifest(dossier: Dict[str, Any], settings: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    prisma = dossier.get("prisma") or {}
    rows = (dossier.get("evidence_matrix") or {}).get("rows") or []
    screening = prisma.get("screening_table") or prisma.get("records") or []
    decisions = []
    for s in screening if isinstance(screening, list) else []:
        decisions.append({k: s.get(k) for k in ("record_id", "title", "doi", "pmid", "decision", "reason", "score",
                                                 "ft_stage", "ft_reason", "text_basis") if k in s})
    return {
        "manifest_version": REPRO_VERSION,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "question": dossier.get("research_question"),
        "question_as_typed": (dossier.get("question_normalisation") or {}).get("original"),
        "research_question_id": dossier.get("research_question_id"),
        "search": {"queries": prisma.get("queries") or prisma.get("search_queries"),
                   "by_source": prisma.get("by_source") or prisma.get("identified_by_source"),
                   "searched_at": prisma.get("searched_at") or time.strftime("%Y-%m-%d"),
                   "identified": (prisma.get("counts") or {}).get("identified")},
        "screening_decisions": decisions,
        "included": [{"ref": r.get("ref"), "title": r.get("title"), "identifier": r.get("identifier"),
                      "text_basis": r.get("text_basis")} for r in rows],
        "settings": settings or {"speed": dossier.get("speed"), "tier": dossier.get("analysis_tier")},
        "ai": {"assignments": dossier.get("ai_assignments"),
               "synthesis_model": (dossier.get("executive_synthesis") or {}).get("model")},
        "code_versions": _versions(),
        "verdict": dossier.get("calibrated_verdict") or (dossier.get("executive_synthesis") or {}).get("verdict"),
        "claims_fingerprint": fingerprint(claim_keys(dossier)),
    }


def write(m: Dict[str, Any], path: str) -> str:
    Path(path).write_text(json.dumps(m, indent=1, ensure_ascii=False, default=str), encoding="utf-8")
    return path


def _norm(s: Any) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(s or "").lower()).strip()


def claim_keys(dossier: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """{content key: finding} for every verified finding of a dossier."""
    out = {}
    for r in (dossier.get("evidence_matrix") or {}).get("rows") or []:
        pid = _norm(r.get("identifier") or r.get("title"))[:80]
        for f in r.get("findings") or []:
            q = _norm(f.get("quote"))[:160]
            key = hashlib.sha1(f"{pid}|{_norm(f.get('endpoint'))}|{q}".encode()).hexdigest()[:16]
            out[key] = {"paper": r.get("title"), "ref": r.get("ref"), "endpoint": f.get("endpoint"),
                        "direction": f.get("direction"), "quote": (f.get("quote") or "")[:200]}
    return out


def fingerprint(keys: Dict[str, Any]) -> str:
    return hashlib.sha1("|".join(sorted(f"{k}:{v.get('direction')}" for k, v in keys.items())).encode()).hexdigest()[:16]


def diff(old: Dict[str, Any], new: Dict[str, Any]) -> Dict[str, Any]:
    a, b = claim_keys(old), claim_keys(new)
    added = [b[k] for k in b if k not in a]
    removed = [a[k] for k in a if k not in b]
    changed = [{"old": a[k], "new": b[k]} for k in a if k in b and a[k]["direction"] != b[k]["direction"]]
    pa = {_norm(r.get("identifier") or r.get("title")) for r in (old.get("evidence_matrix") or {}).get("rows") or []}
    pb = {_norm(r.get("identifier") or r.get("title")) for r in (new.get("evidence_matrix") or {}).get("rows") or []}
    va = (old.get("calibrated_verdict") or {}).get("label") or (old.get("executive_synthesis") or {}).get("verdict")
    vb = (new.get("calibrated_verdict") or {}).get("label") or (new.get("executive_synthesis") or {}).get("verdict")
    return {"papers_added": sorted(pb - pa), "papers_removed": sorted(pa - pb), "claims_added": added,
            "claims_removed": removed, "claims_changed_direction": changed, "verdict_before": va, "verdict_after": vb,
            "verdict_changed": va != vb, "identical": not (added or removed or changed or pa != pb or va != vb)}


def rerun(m: Dict[str, Any], engine: Any) -> Dict[str, Any]:
    """Re-run a manifest's question with its settings. Search results can legitimately differ (new papers were
    published); diff() shows how."""
    s = dict(m.get("settings") or {})
    kw = {k: s[k] for k in ("screen_n", "deep_read_k", "sources", "citation_chasing", "use_unpaywall") if k in s}
    if s.get("speed") == "quick":
        kw.update({"quick": True, "abstract_only": True})
    return engine.investigate(m["question"], use_prisma=True, enhanced_report=True, **kw)


def markdown(d: Dict[str, Any]) -> str:
    if d.get("identical"):
        return "## Re-run comparison\n\nThe re-run reproduced the dossier exactly (same papers, findings and verdict).\n"
    L = ["## Re-run comparison", "",
         f"- papers: +{len(d['papers_added'])} / -{len(d['papers_removed'])}",
         f"- findings: +{len(d['claims_added'])} / -{len(d['claims_removed'])}; direction changed: {len(d['claims_changed_direction'])}",
         f"- verdict: {d['verdict_before']} -> {d['verdict_after']}" + (" (CHANGED)" if d["verdict_changed"] else "")]
    for c in d["claims_changed_direction"][:10]:
        L.append(f"  - {c['new']['ref']} {c['new']['endpoint']}: {c['old']['direction']} -> {c['new']['direction']}")
    return "\n".join(L) + "\n"
