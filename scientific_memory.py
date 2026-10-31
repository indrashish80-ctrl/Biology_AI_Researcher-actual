"""
scientific_memory.py  (v25)
===========================
Versioned scientific memory: when a paper is ADDED or CORRECTED, find the dossiers and claims it affects.

    paper_versions     every paper's content (title + abstract + full-text sections) is hashed; a new hash = a new
                       version (a corrected abstract, a full text replacing an abstract, an erratum re-ingested)
    dossier_registry   every finished dossier: its question, the papers it included, its claim ids, its verdict,
                       the code versions - written by register_dossier() at the end of a run

    snapshot(db)                     -> papers whose content changed since the last snapshot (new versions)
    register_dossier(db, dossier)    -> remember a finished dossier
    dossiers_citing(db, paper_ids)   -> dossiers that INCLUDED any of these papers (their verdict may change)
    candidates_for_new(db, paper_ids)-> dossiers whose QUESTION the new papers are relevant to (keyword + PICO score)
    impact(db, changed_paper_ids)    -> one report: affected dossiers, affected claims, suggested re-runs
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional

MEMORY_VERSION = "1.0"
SCHEMA = """
CREATE TABLE IF NOT EXISTS paper_versions (paper_id INTEGER NOT NULL, version INTEGER NOT NULL, content_hash TEXT NOT NULL,
    recorded_at TEXT NOT NULL, reason TEXT, PRIMARY KEY (paper_id, version));
CREATE TABLE IF NOT EXISTS dossier_registry (research_question_id INTEGER PRIMARY KEY, question TEXT, path TEXT,
    paper_ids_json TEXT, claim_ids_json TEXT, verdict_json TEXT, versions_json TEXT, created_at TEXT NOT NULL);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def ensure(db: Any) -> None:
    db._conn.executescript(SCHEMA)
    db._conn.commit()


def content_hash(db: Any, paper: Dict[str, Any]) -> str:
    h = hashlib.sha256()
    for x in (paper.get("title"), paper.get("abstract")):
        h.update((x or "").encode("utf-8", "replace") + b"\x00")
    try:
        for s in db.get_sections(paper["id"]):
            h.update((s.get("section_type") or "").encode() + b":" + (s.get("text") or "").encode("utf-8", "replace") + b"\x00")
    except Exception:
        pass
    return h.hexdigest()[:24]


def snapshot(db: Any, paper_ids: Optional[Iterable[int]] = None) -> Dict[str, Any]:
    """Record a version for every new / changed paper. Returns {'new': [...], 'changed': [...], 'unchanged': n}."""
    ensure(db)
    sql = "SELECT id, title, abstract FROM papers"
    ids = list(paper_ids) if paper_ids else None
    if ids:
        sql += f" WHERE id IN ({','.join('?' * len(ids))})"
    papers = [dict(r) for r in db._conn.execute(sql, tuple(ids or ())).fetchall()]
    new, changed, same = [], [], 0
    for p in papers:
        h = content_hash(db, p)
        last = db._conn.execute("SELECT version, content_hash FROM paper_versions WHERE paper_id=? ORDER BY version DESC LIMIT 1",
                                (p["id"],)).fetchone()
        if last and last[1] == h:
            same += 1
            continue
        v = (last[0] + 1) if last else 1
        db._conn.execute("INSERT INTO paper_versions (paper_id, version, content_hash, recorded_at, reason) VALUES (?,?,?,?,?)",
                         (p["id"], v, h, _now(), "new paper" if not last else "content changed"))
        (changed if last else new).append(p["id"])
    db._conn.commit()
    return {"new": new, "changed": changed, "unchanged": same}


def register_dossier(db: Any, dossier: Dict[str, Any], path: Optional[str] = None) -> None:
    ensure(db)
    rq = dossier.get("research_question_id")
    if rq is None:
        return
    claim_ids = sorted({cid for r in ((dossier.get("evidence_matrix") or {}).get("rows") or [])
                        for f in r.get("findings") or [] for cid in f.get("claim_ids") or []})
    try:
        import gold_bench
        versions = gold_bench.versions()
    except Exception:
        versions = {}
    db._conn.execute("INSERT OR REPLACE INTO dossier_registry (research_question_id, question, path, paper_ids_json, "
                     "claim_ids_json, verdict_json, versions_json, created_at) VALUES (?,?,?,?,?,?,?,?)",
                     (rq, dossier.get("research_question"), path or f"research_dossier_{rq}.json",
                      json.dumps(dossier.get("included_paper_ids") or []), json.dumps(claim_ids),
                      json.dumps(dossier.get("calibrated_verdict") or (dossier.get("executive_synthesis") or {}).get("verdict")),
                      json.dumps(versions), _now()))
    db._conn.commit()


def _registry(db: Any) -> List[Dict[str, Any]]:
    ensure(db)
    rows = [dict(r) for r in db._conn.execute("SELECT * FROM dossier_registry").fetchall()]
    for r in rows:
        r["paper_ids"] = json.loads(r.pop("paper_ids_json") or "[]")
        r["claim_ids"] = json.loads(r.pop("claim_ids_json") or "[]")
        r["verdict"] = json.loads(r.pop("verdict_json") or "null")
        r["versions"] = json.loads(r.pop("versions_json") or "{}")
    return rows


def dossiers_citing(db: Any, paper_ids: Iterable[int]) -> List[Dict[str, Any]]:
    ps = set(int(p) for p in paper_ids)
    out = []
    for r in _registry(db):
        hit = ps & set(r["paper_ids"])
        if hit:
            out.append({"research_question_id": r["research_question_id"], "question": r["question"], "path": r["path"],
                        "papers": sorted(hit), "verdict": (r["verdict"] or {}).get("label") if isinstance(r["verdict"], dict) else r["verdict"]})
    return out


def candidates_for_new(db: Any, paper_ids: Iterable[int], min_score: float = 0.5) -> List[Dict[str, Any]]:
    """Dossiers whose question a NEW paper may answer: the paper's title/abstract covers >= `min_score` of the
    question's weighted PICO concepts (hybrid_retrieval.pico_score)."""
    import hybrid_retrieval as hr
    import query_parser as qp
    papers = [db.get_paper(int(p)) for p in paper_ids]
    out = []
    for r in _registry(db):
        if not r.get("question"):
            continue
        parsed = qp.parse(r["question"])
        concepts = qp.search_concepts(parsed)
        for p in papers:
            if not p or p["id"] in r["paper_ids"]:
                continue
            s, roles = hr.pico_score(p.get("title") or "", p.get("abstract") or "", concepts, parsed.get("preferred_designs") or [])
            if s >= min_score and "intervention" in roles:
                out.append({"research_question_id": r["research_question_id"], "question": r["question"],
                            "paper_id": p["id"], "title": p.get("title"), "score": round(s, 3), "matched": roles})
    return sorted(out, key=lambda x: -x["score"])


def impact(db: Any, changed_paper_ids: Optional[Iterable[int]] = None) -> Dict[str, Any]:
    snap = snapshot(db) if changed_paper_ids is None else {"new": [], "changed": list(changed_paper_ids), "unchanged": None}
    changed, new = snap["changed"], snap["new"]
    affected = dossiers_citing(db, changed) if changed else []
    claims = []
    for pid in changed:
        claims += [dict(r) for r in db._conn.execute("SELECT id, research_question_id FROM claims WHERE paper_id=?", (pid,)).fetchall()]
    cands = candidates_for_new(db, new) if new else []
    return {"new_papers": new, "changed_papers": changed, "affected_dossiers": affected, "affected_claims": len(claims),
            "new_paper_candidates": cands,
            "suggested_reruns": sorted({d["research_question_id"] for d in affected} | {c["research_question_id"] for c in cands}),
            "version": MEMORY_VERSION}


def markdown(imp: Dict[str, Any]) -> str:
    L = ["## What changed in the knowledge base", "",
         f"{len(imp['new_papers'])} new paper(s), {len(imp['changed_papers'])} changed paper(s), "
         f"{imp['affected_claims']} claim(s) resting on changed papers."]
    for d in imp["affected_dossiers"]:
        L.append(f"- Dossier #{d['research_question_id']} ('{d['question']}') included changed paper(s) {d['papers']} - re-run it.")
    for c in imp["new_paper_candidates"][:20]:
        L.append(f"- New paper #{c['paper_id']} may answer dossier #{c['research_question_id']} ('{c['question']}'): "
                 f"covers {', '.join(c['matched'])} (score {c['score']}).")
    return "\n".join(L) + "\n"
