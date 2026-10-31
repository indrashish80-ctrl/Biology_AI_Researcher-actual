"""
publication_status.py  (v25)
============================
Track retractions, errata, corrections and expressions of concern - and when a paper's status changes, flag every
claim (in every research question) that depended on it.

Status values, most serious first:
    retracted               the paper was retracted (or withdrawn)           -> its claims are quarantined
    expression_of_concern   the journal issued an expression of concern      -> its claims are quarantined for review
    corrected               an erratum / correction / corrected-and-republished notice exists -> claims flagged
    current                 no notice found at the last check
    unknown                 never checked

Where the status comes from (all free, no key):
    * the publication types and correction links the search APIs already return (Europe PMC pubTypeList /
      commentCorrectionList; PubMed 'Retracted Publication', 'RetractionIn', 'ErratumIn', 'ExpressionOfConcernIn')
    * Crossref 'updated-by' / 'update-to' (which includes the Retraction Watch database since 2023), by DOI
    * the title / first lines ('RETRACTED:', 'Retraction notice', 'This article has been retracted')

    check_paper(db, paper_id, fetch=True)        -> status record (stored in table paper_status)
    recheck(db, older_than_days=30, fetch=True)  -> the 're-check status' job: every paper not checked recently
    affected_claims(db, paper_id)                -> claims and research questions resting on the paper
    status_for_claims(db, claims)                -> {paper_id: status} for a report

When a paper becomes retracted / of concern / corrected, recheck() adds a 'publication_status:<status>' flag to each
of its claims (semantic_audit quarantines retracted / concern claims), writes an audit flag per research question,
queues a review item, and returns the affected dossiers (via scientific_memory, when present).
"""
from __future__ import annotations

import json
import os
import re
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Iterable, List, Optional

STATUS_VERSION = "1.0"
SEVERITY = {"retracted": 3, "expression_of_concern": 2, "corrected": 1, "current": 0, "unknown": -1}
SCHEMA = """
CREATE TABLE IF NOT EXISTS paper_status (
    paper_id INTEGER PRIMARY KEY, status TEXT NOT NULL, notices_json TEXT, source TEXT,
    checked_at TEXT NOT NULL, changed_at TEXT, previous_status TEXT
);
"""
_PT = {
    "retracted": re.compile(r"\b(?:retracted publication|retraction of publication|withdrawn publication|retraction)\b", re.I),
    "expression_of_concern": re.compile(r"\bexpression of concern\b", re.I),
    "corrected": re.compile(r"\b(?:published erratum|erratum|corrigendum|correction|corrected and republished)\b", re.I),
}
_CC = {"RetractionIn": "retracted", "RetractionOf": None, "ErratumIn": "corrected", "ExpressionOfConcernIn": "expression_of_concern",
       "CorrectedandRepublishedIn": "corrected", "RepublishedIn": "corrected", "UpdateIn": "corrected"}
_CROSSREF = {"retraction": "retracted", "withdrawal": "retracted", "removal": "retracted",
             "expression_of_concern": "expression_of_concern", "correction": "corrected", "erratum": "corrected",
             "corrigendum": "corrected", "new_version": None, "clarification": "corrected"}


def _now() -> datetime:
    return datetime.now(timezone.utc)


def ensure(db: Any) -> None:
    db._conn.executescript(SCHEMA)
    db._conn.commit()


def worst(statuses: Iterable[Optional[str]]) -> str:
    s = [x for x in statuses if x]
    return max(s, key=lambda x: SEVERITY.get(x, -1)) if s else "current"


# ------------------------------------------------------------------------------------------- readers (pure)
def from_pub_types(pub_types: Iterable[str]) -> List[Dict[str, str]]:
    """Notices implied by indexed publication types. A NOTICE itself ('Retraction of Publication') is reported as a
    notice, not as the retracted paper - the caller decides from which side it reads."""
    out = []
    for t in pub_types or []:
        t = str(t)
        if re.search(r"retracted publication|withdrawn publication", t, re.I):
            out.append({"status": "retracted", "evidence": f"publication type '{t}'"})
        elif _PT["expression_of_concern"].search(t):
            out.append({"status": "expression_of_concern", "evidence": f"publication type '{t}'"})
        elif re.search(r"corrected and republished", t, re.I):
            out.append({"status": "corrected", "evidence": f"publication type '{t}'"})
    return out


def from_comment_corrections(items: Iterable[Dict[str, Any]]) -> List[Dict[str, str]]:
    """Europe PMC commentCorrectionList / PubMed CommentsCorrections: [{'type': 'RetractionIn', 'reference': ...}]."""
    out = []
    for it in items or []:
        typ = it.get("type") or it.get("RefType") or ""
        st = _CC.get(typ)
        if st:
            out.append({"status": st, "evidence": f"{typ}: {str(it.get('reference') or it.get('RefSource') or '')[:160]}",
                        "notice_id": it.get("id") or it.get("PMID")})
    return out


def from_crossref(message: Dict[str, Any]) -> List[Dict[str, str]]:
    """Crossref /works/{doi} message: 'updated-by' (notices about THIS work)."""
    out = []
    for u in (message or {}).get("updated-by") or []:
        st = _CROSSREF.get(str(u.get("type") or "").lower().replace(" ", "_"))
        if st:
            out.append({"status": st, "evidence": f"Crossref update: {u.get('type')} {u.get('DOI') or ''}".strip(),
                        "notice_id": u.get("DOI"), "date": str((u.get("updated") or {}).get("date-time") or "")})
    return out


def from_text(title: Optional[str], abstract: Optional[str]) -> List[Dict[str, str]]:
    t = title or ""
    out = []
    if re.match(r"\s*(?:retracted|retraction|withdrawn)\b[:\s]", t, re.I) or \
            re.search(r"\bthis article has been retracted\b|\bretraction notice\b", (abstract or "")[:800], re.I):
        out.append({"status": "retracted", "evidence": "title / abstract states the retraction"})
    if re.search(r"\bexpression of concern\b", t, re.I):
        out.append({"status": "expression_of_concern", "evidence": "title states an expression of concern"})
    if re.match(r"\s*(?:erratum|corrigendum|correction)\b", t, re.I):
        out.append({"status": "corrected", "evidence": "title is a correction notice"})
    return out


# ------------------------------------------------------------------------------------------- fetchers (network)
def _fetch_europepmc(pmid: Optional[str], doi: Optional[str]) -> Dict[str, Any]:
    from http_client import HttpClient
    q = f"EXT_ID:{pmid} AND SRC:MED" if pmid else f'DOI:"{doi}"'
    js = HttpClient(timeout=15, retries=1).get_json("https://www.ebi.ac.uk/europepmc/webservices/rest/search",
                                                     {"query": q, "resultType": "core", "format": "json", "pageSize": 1})
    res = ((js or {}).get("resultList") or {}).get("result") or []
    return res[0] if res else {}


def _fetch_crossref(doi: str) -> Dict[str, Any]:
    from http_client import HttpClient
    js = HttpClient(timeout=15, retries=1).get_json(f"https://api.crossref.org/works/{doi}")
    return (js or {}).get("message") or {}


def collect_notices(paper: Dict[str, Any], fetch: bool = True, meta: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    notices: List[Dict[str, str]] = []
    sources = []
    notices += from_text(paper.get("title"), paper.get("abstract"))
    if meta:
        notices += from_pub_types(meta.get("pub_types") or [])
        notices += from_comment_corrections(meta.get("comment_corrections") or [])
        sources.append("stored metadata")
    # v26: the local library (updated daily from PubMed) knows retractions / errata without any network call
    try:
        import library_db
        if library_db.exists():
            with library_db.Library() as lib:
                rec = lib.by_ids(pmid=paper.get("pmid"), doi=paper.get("doi"), pmcid=paper.get("pmcid"))
            if rec:
                notices += [n for n in (rec.get("notices") or []) if isinstance(n, dict) and n.get("status")]
                sources.append("local library")
    except Exception:
        pass
    if fetch and not os.getenv("BIOSYN_OFFLINE"):
        if paper.get("pmid") or paper.get("doi"):
            try:
                rec = _fetch_europepmc(paper.get("pmid"), paper.get("doi"))
                notices += from_pub_types(((rec.get("pubTypeList") or {}).get("pubType")) or [])
                notices += from_comment_corrections(((rec.get("commentCorrectionList") or {}).get("commentCorrection")) or [])
                sources.append("Europe PMC")
            except Exception as exc:
                sources.append(f"Europe PMC failed ({type(exc).__name__})")
        if paper.get("doi"):
            try:
                notices += from_crossref(_fetch_crossref(paper["doi"]))
                sources.append("Crossref")
            except Exception as exc:
                sources.append(f"Crossref failed ({type(exc).__name__})")
    seen, uniq = set(), []
    for n in notices:
        k = (n["status"], n.get("notice_id") or n["evidence"])
        if k not in seen:
            seen.add(k)
            uniq.append(n)
    return {"status": worst(n["status"] for n in uniq), "notices": uniq, "sources": sources}


# ------------------------------------------------------------------------------------------- storage + dependants
def _stored_meta(db: Any, paper: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    try:
        for key in (f"pmid:{paper.get('pmid')}", f"doi:{str(paper.get('doi') or '').lower()}", f"pmcid:{paper.get('pmcid')}"):
            row = db._conn.execute("SELECT json FROM paper_meta WHERE key=?", (key,)).fetchone()
            if row:
                return json.loads(row[0])
    except Exception:
        return None
    return None


def get_status(db: Any, paper_id: int) -> Dict[str, Any]:
    ensure(db)
    r = db._conn.execute("SELECT * FROM paper_status WHERE paper_id=?", (paper_id,)).fetchone()
    if not r:
        return {"paper_id": paper_id, "status": "unknown", "notices": []}
    d = dict(r)
    d["notices"] = json.loads(d.pop("notices_json") or "[]")
    return d


def check_paper(db: Any, paper_id: int, fetch: bool = True) -> Dict[str, Any]:
    """Check one paper, store the result, and (on a change to a worse status) flag every dependent claim."""
    ensure(db)
    paper = db.get_paper(paper_id) or {}
    prev = get_status(db, paper_id)
    res = collect_notices(paper, fetch=fetch, meta=_stored_meta(db, paper))
    now = _now().isoformat(timespec="seconds")
    changed = res["status"] != prev.get("status") and not (prev.get("status") == "unknown" and res["status"] == "current")
    db._conn.execute("INSERT OR REPLACE INTO paper_status (paper_id, status, notices_json, source, checked_at, changed_at, "
                     "previous_status) VALUES (?,?,?,?,?,?,?)",
                     (paper_id, res["status"], json.dumps(res["notices"], ensure_ascii=False), "; ".join(res["sources"]),
                      now, now if changed else prev.get("changed_at"), prev.get("status") if changed else prev.get("previous_status")))
    db._conn.commit()
    out = {"paper_id": paper_id, "title": paper.get("title"), **res, "changed": changed, "previous": prev.get("status")}
    if changed and SEVERITY.get(res["status"], 0) > max(0, SEVERITY.get(prev.get("status") or "unknown", 0)):
        out["dependents"] = flag_dependents(db, paper_id, res["status"], res["notices"])
    return out


def affected_claims(db: Any, paper_id: int) -> List[Dict[str, Any]]:
    rows = db._conn.execute("SELECT id, research_question_id, statement, quote FROM claims WHERE paper_id=?", (paper_id,)).fetchall()
    return [dict(r) for r in rows]


def flag_dependents(db: Any, paper_id: int, status: str, notices: List[Dict[str, Any]]) -> Dict[str, Any]:
    try:
        from semantic_audit import ensure_columns
        ensure_columns(db)                       # verification_status exists before it is reset
    except Exception:
        pass
    claims = affected_claims(db, paper_id)
    rqs = sorted({c["research_question_id"] for c in claims if c["research_question_id"] is not None})
    why = notices[0]["evidence"] if notices else status
    for c in claims:
        row = db._conn.execute("SELECT flags_json FROM claims WHERE id=?", (c["id"],)).fetchone()
        flags = json.loads(row[0] or "[]") if row and row[0] else []
        flags = [f for f in flags if not str(f).startswith("publication_status:")] + [f"publication_status:{status}: {why}"]
        db._conn.execute("UPDATE claims SET flags_json=?, verification_status=NULL WHERE id=?",
                         (json.dumps(flags, ensure_ascii=False), c["id"]))
    db._conn.commit()
    for rq in rqs:
        try:
            db.add_audit_flag(rq, "paper", paper_id, None, f"paper:{paper_id}",
                              f"The paper is now '{status}' ({why}); {sum(c['research_question_id'] == rq for c in claims)} "
                              f"claim(s) of this question rested on it and are re-audited.",
                              "critical" if status in ("retracted", "expression_of_concern") else "major")
        except Exception:
            pass
    try:
        import review_store
        for c in claims[:20]:
            review_store.queue_for_review(db, paper_id, c["research_question_id"], {"statement": c["statement"], "quote": c["quote"]},
                                          f"paper {status}", claim_id=c["id"])
    except Exception:
        pass
    dossiers: List[str] = []
    try:
        import scientific_memory
        dossiers = scientific_memory.dossiers_citing(db, [paper_id])
    except Exception:
        pass
    return {"claims": len(claims), "research_questions": rqs, "dossiers": dossiers}


def recheck(db: Any, older_than_days: int = 30, fetch: bool = True, limit: int = 500,
            paper_ids: Optional[List[int]] = None) -> Dict[str, Any]:
    """The 're-check status' job. Checks papers never checked or last checked more than `older_than_days` ago."""
    ensure(db)
    cutoff = (_now() - timedelta(days=older_than_days)).isoformat(timespec="seconds")
    if paper_ids:
        ids = list(paper_ids)
    else:
        ids = [r[0] for r in db._conn.execute(
            "SELECT p.id FROM papers p LEFT JOIN paper_status s ON s.paper_id = p.id "
            "WHERE s.paper_id IS NULL OR s.checked_at < ? ORDER BY p.id LIMIT ?", (cutoff, limit)).fetchall()]
    changed, counts = [], {}
    for pid in ids:
        r = check_paper(db, pid, fetch=fetch)
        counts[r["status"]] = counts.get(r["status"], 0) + 1
        if r.get("changed"):
            changed.append({k: r.get(k) for k in ("paper_id", "title", "status", "previous", "notices", "dependents")})
    return {"checked": len(ids), "by_status": counts, "changed": changed, "version": STATUS_VERSION,
            "when": _now().isoformat(timespec="seconds")}


def status_for_claims(db: Any, claims: List[Dict[str, Any]]) -> Dict[int, str]:
    ensure(db)
    ids = sorted({c.get("paper_id") for c in claims if c.get("paper_id")})
    if not ids:
        return {}
    rows = db._conn.execute(f"SELECT paper_id, status FROM paper_status WHERE paper_id IN ({','.join('?' * len(ids))})",
                            tuple(ids)).fetchall()
    return {r[0]: r[1] for r in rows}


def markdown(summary: Dict[str, Any]) -> str:
    bad = [c for c in summary.get("changed") or [] if c["status"] != "current"]
    L = ["## Publication status", "", f"Checked {summary.get('checked', 0)} paper(s): " +
         ", ".join(f"{k} {v}" for k, v in sorted((summary.get('by_status') or {}).items()))]
    for c in bad:
        L.append(f"- **{c['status'].replace('_', ' ')}**: {c.get('title') or c['paper_id']} - "
                 + "; ".join(n["evidence"] for n in (c.get("notices") or [])[:2])
                 + (f" ({(c.get('dependents') or {}).get('claims', 0)} claim(s) re-audited)" if c.get("dependents") else ""))
    return "\n".join(L) + "\n"
