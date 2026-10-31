"""
extraction_cache.py  (v25)
==========================
Per-paper extraction cache: a paper read once is never read again by the AI for the same purpose.

Key = (paper identifier: DOI > PMID > PMCID > title hash) + what was extracted ('design', 'claims') + the extractor
version + a hash of the exact text that was read + (for claims) the normalised question.

Invalidation rules - an entry is a miss, never a stale answer, when:
    * the extractor / guard version changes      (EXTRACTOR_VERSION, GUARD_VERSION are in the key)
    * the paper's text changes                   (abstract replaced by the full text, a corrected version)
    * the question changes                       (claims are extracted with the question's focus)
    * the user asks for a fresh run              (BIOSYN_NO_EXTRACTION_CACHE=1 or invalidate())
Entries are kept in cache/extraction_cache.db (shared by every run database). Only AI output is cached; the
deterministic checks (quote guard, audit) always run again on the cached output.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

CACHE_VERSION = "1.0"
_LOCK = threading.Lock()
_STATS = {"hits": 0, "misses": 0, "stores": 0}
_SCHEMA = """
CREATE TABLE IF NOT EXISTS extraction_cache (key TEXT PRIMARY KEY, paper_key TEXT NOT NULL, kind TEXT NOT NULL,
    version TEXT NOT NULL, value_json TEXT NOT NULL, created_at TEXT NOT NULL, hits INTEGER NOT NULL DEFAULT 0);
CREATE INDEX IF NOT EXISTS idx_xc_paper ON extraction_cache(paper_key);
"""


def _path() -> Path:
    p = Path(os.getenv("BIOSYN_EXTRACTION_CACHE") or Path(__file__).resolve().parent / "cache" / "extraction_cache.db")
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def enabled() -> bool:
    return os.getenv("BIOSYN_NO_EXTRACTION_CACHE", "0") != "1"


def _conn() -> sqlite3.Connection:
    c = sqlite3.connect(str(_path()), timeout=30)
    c.executescript(_SCHEMA)
    return c


def paper_key(paper: Dict[str, Any]) -> str:
    if paper.get("doi"):
        return "doi:" + re.sub(r"^(?:https?://(?:dx\.)?doi\.org/|doi:\s*)", "", str(paper["doi"]), flags=re.I).strip().lower()
    if paper.get("pmid"):
        return f"pmid:{paper['pmid']}"
    if paper.get("pmcid"):
        return f"pmcid:{str(paper['pmcid']).upper()}"
    t = re.sub(r"[^a-z0-9]+", " ", (paper.get("title") or "").lower()).strip()
    return "title:" + hashlib.sha1(t.encode()).hexdigest()[:16]


def _h(s: str) -> str:
    return hashlib.sha256((s or "").encode("utf-8", "replace")).hexdigest()[:20]


def make_key(paper: Dict[str, Any], kind: str, version: str, text: str, question: Optional[str] = None,
             extra: str = "") -> str:
    q = re.sub(r"\s+", " ", (question or "").lower()).strip()
    return "|".join([paper_key(paper), kind, version, _h(text), _h(q) if q else "-", _h(extra) if extra else "-"])


def get(key: str) -> Optional[Any]:
    if not enabled():
        return None
    with _LOCK:
        c = _conn()
        try:
            r = c.execute("SELECT value_json FROM extraction_cache WHERE key=?", (key,)).fetchone()
            if r:
                c.execute("UPDATE extraction_cache SET hits = hits + 1 WHERE key=?", (key,))
                c.commit()
        finally:
            c.close()
    if r:
        _STATS["hits"] += 1
        try:
            return json.loads(r[0])
        except ValueError:
            return None
    _STATS["misses"] += 1
    return None


def put(key: str, value: Any) -> None:
    if not enabled() or value is None:
        return
    pk, kind, version = (key.split("|") + ["", "", ""])[:3]
    with _LOCK:
        c = _conn()
        try:
            c.execute("INSERT OR REPLACE INTO extraction_cache (key, paper_key, kind, version, value_json, created_at, hits) "
                      "VALUES (?,?,?,?,?,?,0)", (key, pk, kind, version, json.dumps(value, ensure_ascii=False),
                                                 datetime.now(timezone.utc).isoformat(timespec="seconds")))
            c.commit()
        finally:
            c.close()
    _STATS["stores"] += 1


def invalidate(paper: Optional[Dict[str, Any]] = None, kind: Optional[str] = None) -> int:
    with _LOCK:
        c = _conn()
        try:
            if paper is None and kind is None:
                n = c.execute("DELETE FROM extraction_cache").rowcount
            elif paper is None:
                n = c.execute("DELETE FROM extraction_cache WHERE kind=?", (kind,)).rowcount
            else:
                sql = "DELETE FROM extraction_cache WHERE paper_key=?" + (" AND kind=?" if kind else "")
                n = c.execute(sql, (paper_key(paper),) + ((kind,) if kind else ())).rowcount
            c.commit()
        finally:
            c.close()
    return n


def stats() -> Dict[str, Any]:
    tot = _STATS["hits"] + _STATS["misses"]
    with _LOCK:
        c = _conn()
        try:
            n = c.execute("SELECT kind, COUNT(*) FROM extraction_cache GROUP BY kind").fetchall()
        finally:
            c.close()
    return {**_STATS, "hit_rate": round(_STATS["hits"] / tot, 3) if tot else None, "entries": dict(n),
            "path": str(_path()), "version": CACHE_VERSION}


def reset_stats() -> None:
    for k in _STATS:
        _STATS[k] = 0
