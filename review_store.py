"""
review_store.py  (v25 - was an in-memory stub in v24.2)
=======================================================
Keep a person in the loop for high-stakes claims (principle 6), and turn their checks into gold data (principle 5).

A review item is one finding with its verbatim quote and source, waiting for a person:
    - findings the two readers disagreed on (independent_reader)
    - findings only the second reader extracted (possible misses of the first reader)
    - findings on a retracted / corrected paper (publication_status)
    - any finding a user flags from the website
A reviewer approves, corrects (endpoint / direction) or rejects it in seconds, because the quote and the paper are
shown next to it. Approved and corrected items export as gold DRAFTS (gold/papers/*.json, status "draft") - a person
then completes and verifies the draft; nothing becomes gold automatically.

Stored in the run database (table review_items); the old ReviewStore class is kept for callers of v24.2.
"""
from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

SCHEMA = """
CREATE TABLE IF NOT EXISTS review_items (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    paper_id INTEGER, research_question_id INTEGER, claim_id INTEGER,
    reason TEXT NOT NULL, finding_json TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'open',          -- open | approved | corrected | rejected
    reviewer TEXT, decision_json TEXT, notes TEXT,
    created_at TEXT NOT NULL, decided_at TEXT,
    UNIQUE(paper_id, claim_id, reason)
);
CREATE INDEX IF NOT EXISTS idx_review_status ON review_items(status);
"""
STATUSES = ("open", "approved", "corrected", "rejected")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def ensure(db: Any) -> None:
    db._conn.executescript(SCHEMA)
    db._conn.commit()


def queue_for_review(db: Any, paper_id: Optional[int], research_question_id: Optional[int], finding: Dict[str, Any],
                     reason: str, claim_id: Optional[int] = None) -> Optional[int]:
    """Add one item (idempotent per paper / claim / reason). Returns its id."""
    ensure(db)
    keep = {k: finding.get(k) for k in ("statement", "intervention", "subject", "outcome", "measurement", "endpoint",
                                        "direction", "effect_direction", "comparator", "quote", "p_value", "provider",
                                        "second_reader_reading") if finding.get(k) is not None}
    cur = db._conn.execute(
        "INSERT OR IGNORE INTO review_items (paper_id, research_question_id, claim_id, reason, finding_json, created_at) "
        "VALUES (?,?,?,?,?,?)", (paper_id, research_question_id, claim_id if claim_id is not None else -1, reason,
                                 json.dumps(keep, ensure_ascii=False), _now()))
    db._conn.commit()
    # an ignored duplicate leaves lastrowid at the PREVIOUS insert: only a real insert returns an id
    return cur.lastrowid if cur.rowcount == 1 else None


def list_items(db: Any, status: Optional[str] = "open", limit: int = 200) -> List[Dict[str, Any]]:
    ensure(db)
    sql = ("SELECT r.*, p.title AS paper_title, p.pmid, p.doi FROM review_items r LEFT JOIN papers p ON p.id = r.paper_id")
    args: tuple = ()
    if status:
        sql += " WHERE r.status = ?"
        args = (status,)
    sql += " ORDER BY r.id DESC LIMIT ?"
    rows = [dict(r) for r in db._conn.execute(sql, args + (limit,)).fetchall()]
    for r in rows:
        r["finding"] = json.loads(r.pop("finding_json") or "{}")
        r["decision"] = json.loads(r.pop("decision_json") or "null")
    return rows


def decide(db: Any, item_id: int, status: str, reviewer: str = "", correction: Optional[Dict[str, Any]] = None,
           notes: str = "") -> Dict[str, Any]:
    """approved | corrected (give correction={'endpoint','direction'}) | rejected."""
    if status not in STATUSES[1:]:
        raise ValueError(f"status must be one of {STATUSES[1:]}")
    if status == "corrected" and not correction:
        raise ValueError("a correction needs the corrected endpoint and/or direction")
    ensure(db)
    cur = db._conn.execute("UPDATE review_items SET status=?, reviewer=?, decision_json=?, notes=?, decided_at=? WHERE id=?",
                     (status, reviewer, json.dumps(correction or {}, ensure_ascii=False), notes, _now(), item_id))
    db._conn.commit()
    if cur.rowcount == 0:
        raise KeyError(item_id)
    return next((r for r in list_items(db, None, 100000) if r["id"] == item_id), {})


def export_gold_drafts(db: Any, folder: Any = None) -> List[str]:
    """Approved / corrected items -> gold/papers/review_<pmid|paper>.json drafts (status 'draft': a person completes
    and verifies them before they count). Existing verified files are never overwritten."""
    ensure(db)
    folder = Path(folder or Path(__file__).resolve().parent / "gold" / "papers")
    folder.mkdir(parents=True, exist_ok=True)
    by_paper: Dict[Any, List[Dict[str, Any]]] = {}
    for r in list_items(db, None, 100000):
        if r["status"] in ("approved", "corrected"):
            by_paper.setdefault(r["paper_id"], []).append(r)
    written = []
    for pid, items in by_paper.items():
        p = items[0]
        key = re.sub(r"[^A-Za-z0-9]+", "_", str(p.get("pmid") or p.get("doi") or f"paper{pid}"))[:60]
        path = folder / f"review_{key}.json"
        if path.exists():
            try:
                if json.loads(path.read_text(encoding="utf-8")).get("status") == "verified":
                    continue
            except Exception:
                pass
        findings = []
        for r in items:
            f, d = r["finding"], r.get("decision") or {}
            findings.append({"endpoint": d.get("endpoint") or f.get("outcome") or f.get("measurement") or f.get("endpoint"),
                             "direction": d.get("direction") or f.get("direction") or f.get("effect_direction"),
                             "quote": f.get("quote"), "p_value": f.get("p_value"),
                             "review": {"status": r["status"], "reviewer": r.get("reviewer"), "reason": r["reason"]}})
        draft = {"gold_id": f"review-{key}", "status": "draft", "verified_by": "", "verified_on": "",
                 "paper": {"pmid": p.get("pmid") or "", "doi": p.get("doi") or "", "title": p.get("paper_title") or ""},
                 "study_type": "", "difficult_case": ["from-review-queue"], "source_text": "",
                 "findings": findings, "non_findings": [],
                 "note": "Draft exported from the review queue. Complete study_type, check every finding against the "
                         "paper, then set status to 'verified' with verified_by / verified_on."}
        path.write_text(json.dumps(draft, indent=2, ensure_ascii=False), encoding="utf-8")
        written.append(str(path))
    return written


def summary(db: Any) -> Dict[str, int]:
    ensure(db)
    out = {s: 0 for s in STATUSES}
    for s, n in db._conn.execute("SELECT status, COUNT(*) FROM review_items GROUP BY status").fetchall():
        out[s] = n
    return out


class ReviewStore:
    """v24.2 interface (in memory) - kept so older callers keep working."""

    def __init__(self):
        self.reviews: Dict[str, Dict[str, Any]] = {}

    def submit_review(self, claim_id: str, reviewer_id: str, direction_approved: Optional[str], notes: str = "") -> Dict[str, Any]:
        review = {"claim_id": claim_id, "reviewer_id": reviewer_id, "direction_approved": direction_approved,
                  "notes": notes, "timestamp": _now(), "status": "approved" if direction_approved else "needs_work"}
        self.reviews[claim_id] = review
        return review

    def get_review(self, claim_id: str) -> Optional[Dict[str, Any]]:
        return self.reviews.get(claim_id)

    def list_reviews(self, status: Optional[str] = None) -> List[Dict[str, Any]]:
        return [r for r in self.reviews.values() if not status or r["status"] == status]

    def export_gold_draft(self, claims: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        out = []
        for claim in claims:
            review = self.get_review(claim.get("id", ""))
            if review and review["status"] == "approved":
                out.append({"claim": claim, "gold_direction": review["direction_approved"],
                            "reviewer_notes": review["notes"], "reviewed_timestamp": review["timestamp"]})
        return out
