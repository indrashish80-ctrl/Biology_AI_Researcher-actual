"""
keyword_index.py  (v25)
=======================
SQLite FTS5 / BM25 keyword index beside the vector store. Embeddings find papers that MEAN the same thing; BM25 finds
papers that SAY the exact thing - a drug code (SGLT2), a gene symbol (CXCL10), a trial acronym (MA.32), a rare
disease name - which a small embedding model blurs. Both are merged by hybrid_retrieval.

    idx = KeywordIndex(db)               # lives in the run database (research_knowledge.db), table papers_fts
    idx.sync()                           # indexes new / changed papers only (content hash)
    idx.search("metformin Ki-67 breast", k=50) -> [{"paper_id", "bm25", "rank"}]

    mem = MemoryIndex(docs)              # same scoring for a candidate pool that is not in the database yet
                                         # (the screening step ranks search results before they are stored)

Weights: title 3, abstract 1.5, body 1. Tokenizer: porter stemming + unicode61 with diacritics removed, so
'tumours' matches 'tumour' and 'tumor' (via the synonym layer). Queries are built from synonym-expanded concept
groups: within a group terms are OR-ed; groups are OR-ed too (BM25 rewards documents that match more of them) - an
AND query would return nothing for a paper that words one concept differently.
"""
from __future__ import annotations

import hashlib
import re
import sqlite3
import threading
from typing import Any, Dict, Iterable, List, Optional, Sequence

INDEX_VERSION = "1.0"
_WEIGHTS = (3.0, 1.5, 1.0)
_FTS_DDL = """
CREATE VIRTUAL TABLE IF NOT EXISTS papers_fts USING fts5(
    title, abstract, body, paper_id UNINDEXED, tokenize = 'porter unicode61 remove_diacritics 2');
CREATE TABLE IF NOT EXISTS papers_fts_state (paper_id INTEGER PRIMARY KEY, content_hash TEXT NOT NULL,
    indexed_at TEXT NOT NULL);
"""


def fts5_available() -> bool:
    try:
        c = sqlite3.connect(":memory:")
        c.execute("CREATE VIRTUAL TABLE t USING fts5(x)")
        c.close()
        return True
    except sqlite3.OperationalError:
        return False


def _quote_term(t: str) -> Optional[str]:
    t = re.sub(r"[\"*^():]", " ", t or "").strip()
    t = re.sub(r"\s+", " ", t)
    if not t or not re.search(r"[A-Za-z0-9]", t):
        return None
    toks = re.findall(r"[A-Za-z0-9]+", t)
    if not toks:
        return None
    return '"' + " ".join(toks) + '"'       # a phrase query; FTS5 tokenises the inside the same way as the documents


def build_match(query: str, concepts: Optional[List[Iterable[str]]] = None, max_terms: int = 60) -> Optional[str]:
    """FTS5 MATCH expression: OR of every (synonym-expanded) term."""
    if concepts is None:
        import mesh_synonyms as ms
        concepts = ms.concepts(query)
    terms: List[str] = []
    for group in concepts:
        for t in sorted(group, key=len):
            q = _quote_term(t)
            if q and q not in terms:
                terms.append(q)
    if not terms:
        words = [w for w in re.findall(r"[A-Za-z0-9][A-Za-z0-9\-]{2,}", query or "")]
        terms = [q for q in (_quote_term(w) for w in words) if q]
    return " OR ".join(terms[:max_terms]) if terms else None


def _hash(*parts: Optional[str]) -> str:
    h = hashlib.sha1()
    for p in parts:
        h.update((p or "").encode("utf-8", "replace"))
        h.update(b"\x00")
    return h.hexdigest()


class KeywordIndex:
    """BM25 over the papers stored in the run database."""

    def __init__(self, db: Any):
        self.db = db
        self._lock = threading.RLock()
        self.ok = fts5_available()
        if self.ok:
            try:
                db._conn.executescript(_FTS_DDL)
                db._conn.commit()
            except sqlite3.OperationalError:
                self.ok = False

    def _body(self, paper_id: int, full_text: Optional[str]) -> str:
        try:
            secs = self.db.get_sections(paper_id)
            body = "\n".join(s.get("text") or "" for s in secs if s.get("section_type") not in ("abstract", "article_meta"))
        except Exception:
            body = ""
        return (body or full_text or "")[:200000]

    def index_paper(self, paper: Dict[str, Any]) -> bool:
        if not self.ok:
            return False
        pid = int(paper["id"])
        body = self._body(pid, paper.get("full_text"))
        h = _hash(paper.get("title"), paper.get("abstract"), body)
        with self._lock:
            row = self.db._conn.execute("SELECT content_hash FROM papers_fts_state WHERE paper_id=?", (pid,)).fetchone()
            if row and row[0] == h:
                return False
            self.db._conn.execute("DELETE FROM papers_fts WHERE paper_id=?", (pid,))
            self.db._conn.execute("INSERT INTO papers_fts (title, abstract, body, paper_id) VALUES (?,?,?,?)",
                                  (paper.get("title") or "", paper.get("abstract") or "", body, pid))
            self.db._conn.execute("INSERT OR REPLACE INTO papers_fts_state (paper_id, content_hash, indexed_at) "
                                  "VALUES (?,?,datetime('now'))", (pid, h))
            self.db._conn.commit()
        return True

    def sync(self, paper_ids: Optional[Sequence[int]] = None) -> Dict[str, int]:
        """Index new or changed papers (all, or the given ids). Returns counts."""
        if not self.ok:
            return {"indexed": 0, "unchanged": 0, "available": 0}
        sql = "SELECT id, title, abstract, full_text FROM papers"
        args: tuple = ()
        if paper_ids:
            sql += f" WHERE id IN ({','.join('?' * len(paper_ids))})"
            args = tuple(int(i) for i in paper_ids)
        rows = [dict(r) for r in self.db._conn.execute(sql, args).fetchall()]
        n = sum(1 for r in rows if self.index_paper(r))
        return {"indexed": n, "unchanged": len(rows) - n, "available": 1}

    def remove(self, paper_id: int) -> None:
        if self.ok:
            with self._lock:
                self.db._conn.execute("DELETE FROM papers_fts WHERE paper_id=?", (paper_id,))
                self.db._conn.execute("DELETE FROM papers_fts_state WHERE paper_id=?", (paper_id,))
                self.db._conn.commit()

    def search(self, query: str, k: int = 50, concepts: Optional[List[Iterable[str]]] = None) -> List[Dict[str, Any]]:
        if not self.ok:
            return []
        match = build_match(query, concepts)
        if not match:
            return []
        try:
            rows = self.db._conn.execute(
                "SELECT paper_id, bm25(papers_fts, ?, ?, ?) AS s FROM papers_fts WHERE papers_fts MATCH ? "
                "ORDER BY s LIMIT ?", (*_WEIGHTS, match, k)).fetchall()
        except sqlite3.OperationalError:
            return []
        # FTS5's bm25() is negative (lower = better); reported positive, larger = better
        return [{"paper_id": int(r[0]), "bm25": round(-float(r[1]), 4), "rank": i + 1} for i, r in enumerate(rows)]

    def count(self) -> int:
        if not self.ok:
            return 0
        return int(self.db._conn.execute("SELECT COUNT(*) FROM papers_fts").fetchone()[0])


class MemoryIndex:
    """The same BM25 scoring over documents held in memory: [{'id', 'title', 'abstract', 'body'?}]."""

    def __init__(self, docs: Sequence[Dict[str, Any]]):
        self.ok = fts5_available()
        self.n = 0
        if not self.ok:
            return
        self.c = sqlite3.connect(":memory:", check_same_thread=False)
        self.c.execute("CREATE VIRTUAL TABLE d USING fts5(title, abstract, body, doc_id UNINDEXED, "
                       "tokenize = 'porter unicode61 remove_diacritics 2')")
        self.c.executemany("INSERT INTO d (title, abstract, body, doc_id) VALUES (?,?,?,?)",
                           [(x.get("title") or "", x.get("abstract") or "", x.get("body") or "", str(x["id"])) for x in docs])
        self.n = len(docs)

    def search(self, query: str, k: int = 50, concepts: Optional[List[Iterable[str]]] = None) -> List[Dict[str, Any]]:
        if not self.ok or not self.n:
            return []
        match = build_match(query, concepts)
        if not match:
            return []
        try:
            rows = self.c.execute("SELECT doc_id, bm25(d, ?, ?, ?) AS s FROM d WHERE d MATCH ? ORDER BY s LIMIT ?",
                                  (*_WEIGHTS, match, k)).fetchall()
        except sqlite3.OperationalError:
            return []
        return [{"id": r[0], "bm25": round(-float(r[1]), 4), "rank": i + 1} for i, r in enumerate(rows)]
