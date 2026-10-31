"""
library_db.py  (v26)
====================
The LOCAL LITERATURE LIBRARY: a reusable research collection searched before any external API.

Where it lives (on your computer, never uploaded):
    BioSynthetica/library/            default; move it with BIOSYN_LIBRARY_DIR (e.g. a second drive)
        library.db                    records, passages, findings, provenance, sync state (SQLite + FTS5)
        vectors/                      semantic index (library_vectors.py)
        fulltext/                     open-access XML / PDF files, one copy each, with their licence
It is separate from research_knowledge.db (your questions and dossiers): rebuilding the library never touches a run.

Tables
    records           one row per paper / registry entry: identifiers (PMID, DOI, PMCID, NCT), title, abstract,
                      authors, journal, year, THREE dates kept apart (published / indexed / last revised), publication
                      types, MeSH, publication status (retracted, corrected, concern), licence, source link, retrieval
                      time, content hash, version, collection ('literature' or 'user:<name>'), deleted flag
    record_history    every change to a record: version, when, which fields, the OLD values (never silently replaced)
    passages          paragraphs of the full text / abstract with their section ('results', 'limitations' ...),
                      searchable with BM25 (passages_fts)
    records_fts       title + abstract + MeSH, BM25
    findings          question-INDEPENDENT findings extracted once per paper (extractor version + text hash),
                      reused by every later question (claim_extraction picks the relevant ones by code)
    processing        processing history per record (parsed, indexed, embedded, pre-read ...), with errors
    sync              last update, files already applied, scope
Deduplication: identifiers first (DOI, PMID, PMCID, NCT), then normalised title + year + first author.
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
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

LIBRARY_VERSION = "1.0"
ROOT = Path(__file__).resolve().parent

_SCHEMA = """
CREATE TABLE IF NOT EXISTS records (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    pmid TEXT, doi TEXT, pmcid TEXT, nct TEXT,
    title TEXT NOT NULL, abstract TEXT, authors TEXT, journal TEXT, year INTEGER,
    published_date TEXT, indexed_date TEXT, revised_date TEXT,
    pub_types_json TEXT, mesh_json TEXT, keywords_json TEXT,
    status TEXT NOT NULL DEFAULT 'current', notices_json TEXT,
    licence TEXT, licence_url TEXT, reuse TEXT,
    funding_json TEXT, coi TEXT,
    fulltext_path TEXT, fulltext_kind TEXT,
    source TEXT, source_url TEXT, retrieved_at TEXT NOT NULL,
    content_hash TEXT NOT NULL, version INTEGER NOT NULL DEFAULT 1,
    collection TEXT NOT NULL DEFAULT 'literature', title_key TEXT, first_author TEXT,
    deleted INTEGER NOT NULL DEFAULT 0, deleted_reason TEXT, field_src_json TEXT
);
CREATE UNIQUE INDEX IF NOT EXISTS ux_rec_pmid ON records(pmid, collection) WHERE pmid IS NOT NULL;
DROP INDEX IF EXISTS ix_rec_doi;
DROP INDEX IF EXISTS ix_rec_pmcid;
DROP INDEX IF EXISTS ix_rec_nct;
DROP INDEX IF EXISTS ix_rec_tkey;
DROP INDEX IF EXISTS ix_rec_coll;
CREATE INDEX IF NOT EXISTS ix_rec_doi2 ON records(doi, collection) WHERE doi IS NOT NULL;
CREATE INDEX IF NOT EXISTS ix_rec_pmcid2 ON records(pmcid, collection) WHERE pmcid IS NOT NULL;
CREATE INDEX IF NOT EXISTS ix_rec_nct2 ON records(nct, collection) WHERE nct IS NOT NULL;
CREATE INDEX IF NOT EXISTS ix_rec_tkey2 ON records(title_key, collection);
CREATE INDEX IF NOT EXISTS ix_rec_coll2 ON records(collection, deleted, status);
CREATE TABLE IF NOT EXISTS record_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT, record_id INTEGER NOT NULL, version INTEGER NOT NULL, changed_at TEXT NOT NULL,
    change TEXT NOT NULL, fields_json TEXT, old_json TEXT, source TEXT
);
CREATE INDEX IF NOT EXISTS ix_hist_rec ON record_history(record_id);
CREATE TABLE IF NOT EXISTS passages (
    id INTEGER PRIMARY KEY AUTOINCREMENT, record_id INTEGER NOT NULL, section TEXT, ord INTEGER NOT NULL,
    text TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_pass_rec ON passages(record_id, ord);
CREATE VIRTUAL TABLE IF NOT EXISTS passages_fts USING fts5(text, section, content='passages', content_rowid='id',
    tokenize='porter unicode61');
CREATE VIRTUAL TABLE IF NOT EXISTS records_fts USING fts5(title, abstract, mesh_json, content='records',
    content_rowid='id', tokenize='porter unicode61');
CREATE TABLE IF NOT EXISTS findings (
    record_id INTEGER NOT NULL, extractor_version TEXT NOT NULL, text_hash TEXT NOT NULL, findings_json TEXT NOT NULL,
    provider TEXT, created_at TEXT NOT NULL, PRIMARY KEY (record_id, extractor_version, text_hash)
);
CREATE TABLE IF NOT EXISTS processing (
    id INTEGER PRIMARY KEY AUTOINCREMENT, record_id INTEGER, step TEXT NOT NULL, status TEXT NOT NULL,
    version TEXT, detail TEXT, at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_proc_rec ON processing(record_id, step);
CREATE TABLE IF NOT EXISTS sync (key TEXT PRIMARY KEY, value TEXT, updated_at TEXT NOT NULL);
"""

# the fields whose change makes a new version (an old value is kept in record_history)
TRACKED = ("title", "abstract", "authors", "journal", "year", "published_date", "revised_date", "pub_types_json",
           "mesh_json", "status", "notices_json", "licence", "doi", "pmcid", "nct", "funding_json", "coi")


def library_dir() -> Path:
    d = Path(os.getenv("BIOSYN_LIBRARY_DIR") or ROOT / "library")
    d.mkdir(parents=True, exist_ok=True)
    return d


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def title_key(title: Optional[str]) -> str:
    t = re.sub(r"<[^>]+>", " ", title or "").lower()
    return re.sub(r"[^a-z0-9]+", "", t)[:120]


def first_author_key(authors: Any) -> str:
    if isinstance(authors, (list, tuple)):
        a = str(authors[0]) if authors else ""
    else:
        a = str(authors or "").split(";")[0].split(",")[0] if ";" in str(authors or "") else str(authors or "").split(",")[0]
    a = re.sub(r"[^A-Za-z\- ]", " ", a).strip()
    return (a.split()[-1] if a and " " in a and not a.isupper() else a.split()[0] if a else "").lower()


SOURCE_RANK = {"user": 5, "pubmed": 4, "pmc": 4, "europe_pmc": 4, "clinicaltrials": 4, "crossref": 2, "openalex": 2,
               "semantic_scholar": 1}


def source_rank(source: Optional[str], field: Optional[str] = None) -> int:
    src = (source or "").split(":")[0].lower()
    if field in ("coi", "funding_json", "licence") and src in ("pmc", "europe_pmc", "user"):
        return 6                                  # disclosures and licences are read from the full text itself
    return SOURCE_RANK.get(src, 3)


def _merge_notices(old_json: Optional[str], new_json: Optional[str]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for js in (old_json, new_json):
        try:
            for n in json.loads(js or "[]") or []:
                if n not in out:
                    out.append(n)
        except Exception:
            pass
    return out


def _norm_text(v: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", re.sub(r"<[^>]+>", "", str(v or "")).lower())


def _same(field: str, a: Any, b: Any) -> bool:
    if field in ("title", "abstract", "journal", "coi"):
        return _norm_text(a) == _norm_text(b)
    if field in ("pub_types_json", "mesh_json", "notices_json", "funding_json"):
        try:
            return json.loads(a) == json.loads(b)
        except Exception:
            return a == b
    return str(a) == str(b)


def _hash(rec: Dict[str, Any]) -> str:
    s = json.dumps({k: rec.get(k) for k in TRACKED}, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:24]


def split_passages(text: str, target_words: int = 160, max_words: int = 320) -> List[str]:
    """Paragraphs of about 160 words. A paragraph is never cut inside a sentence, so a negation or a qualifying clause
    ('... but not in patients older than 65') stays with the result it qualifies."""
    paras = [p.strip() for p in re.split(r"\n\s*\n", text or "") if p.strip()]
    out: List[str] = []
    for p in paras:
        words = p.split()
        if len(words) <= max_words:
            out.append(p)
            continue
        sents = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9(])", p)
        buf: List[str] = []
        n = 0
        for s in sents:
            w = len(s.split())
            if buf and n + w > target_words:
                out.append(" ".join(buf))
                buf, n = [], 0
            buf.append(s)
            n += w
        if buf:
            out.append(" ".join(buf))
    return out


_SCHEMA_READY: set = set()


class Library:
    """Thread-safe handle on library.db. `with Library() as lib: ...`"""

    def __init__(self, path: Optional[str] = None):
        self.dir = Path(path).parent if path else library_dir()
        self.path = Path(path) if path else self.dir / "library.db"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self.c = sqlite3.connect(str(self.path), check_same_thread=False, timeout=60)
        self.c.row_factory = sqlite3.Row
        self.c.execute("PRAGMA synchronous=NORMAL")
        self.c.execute("PRAGMA cache_size=-65536")      # 64 MB page cache per handle
        self.c.execute("PRAGMA temp_store=MEMORY")
        key = str(self.path.resolve())
        if key not in _SCHEMA_READY:                    # the schema script runs once per file per process (v26.1)
            self.c.execute("PRAGMA journal_mode=WAL")
            self.c.executescript(_SCHEMA)
            self.c.commit()
            _SCHEMA_READY.add(key)
        elif not self.c.execute("SELECT 1 FROM sqlite_master WHERE name='records'").fetchone():
            self.c.execute("PRAGMA journal_mode=WAL")   # the file was replaced since: create the schema again
            self.c.executescript(_SCHEMA)
            self.c.commit()

    # ------------------------------------------------------------------------------------------------ basics
    def __enter__(self) -> "Library":
        return self

    def __exit__(self, *a) -> None:
        self.close()

    def close(self) -> None:
        try:
            self.c.close()
        except Exception:
            pass

    def count(self, collection: Optional[str] = "literature", include_deleted: bool = False) -> int:
        q = "SELECT COUNT(*) FROM records WHERE 1=1" + ("" if include_deleted else " AND deleted=0")
        args: Tuple[Any, ...] = ()
        if collection:
            q += " AND collection=?"
            args = (collection,)
        return int(self.c.execute(q, args).fetchone()[0])

    def get_sync(self, key: str, default: Any = None) -> Any:
        r = self.c.execute("SELECT value FROM sync WHERE key=?", (key,)).fetchone()
        return json.loads(r[0]) if r and r[0] is not None else default

    def set_sync(self, key: str, value: Any) -> None:
        with self._lock:
            self.c.execute("INSERT INTO sync (key, value, updated_at) VALUES (?,?,?) ON CONFLICT(key) DO UPDATE SET "
                           "value=excluded.value, updated_at=excluded.updated_at", (key, json.dumps(value, default=str), _now()))
            self.c.commit()

    def log(self, record_id: Optional[int], step: str, status: str, version: str = "", detail: str = "") -> None:
        with self._lock:
            self.c.execute("INSERT INTO processing (record_id, step, status, version, detail, at) VALUES (?,?,?,?,?,?)",
                           (record_id, step, status, version, (detail or "")[:2000], _now()))

    # ------------------------------------------------------------------------------------------------ lookup / dedup
    def find(self, rec: Dict[str, Any]) -> Optional[sqlite3.Row]:
        """The existing record for these identifiers, or for the same title + year + first author - within the same
        collection (a copy you uploaded never merges into the shared literature record, or the reverse)."""
        coll = rec.get("collection") or "literature"
        for col in ("pmid", "doi", "pmcid", "nct"):
            v = rec.get(col)
            if v:
                r = self.c.execute(f"SELECT * FROM records WHERE {col}=? AND collection=? ORDER BY id LIMIT 1",
                                   (str(v), coll)).fetchone()
                if r:
                    return r
        tk = title_key(rec.get("title"))
        if len(tk) >= 25:
            rows = self.c.execute("SELECT * FROM records WHERE title_key=? AND collection=?", (tk, coll)).fetchall()
            fa = first_author_key(rec.get("authors"))
            yr = rec.get("year")
            for r in rows:
                year_ok = not yr or not r["year"] or abs(int(r["year"]) - int(yr)) <= 1
                auth_ok = not fa or not r["first_author"] or r["first_author"] == fa
                if year_ok and auth_ok:
                    return r
        return None

    def get(self, record_id: int) -> Optional[Dict[str, Any]]:
        r = self.c.execute("SELECT * FROM records WHERE id=?", (record_id,)).fetchone()
        return self._dict(r) if r else None

    def by_ids(self, pmid: Optional[str] = None, doi: Optional[str] = None, pmcid: Optional[str] = None,
               nct: Optional[str] = None) -> Optional[Dict[str, Any]]:
        r = self.find({"pmid": pmid, "doi": (doi or "").lower() or None, "pmcid": pmcid, "nct": nct,
                       "collection": "literature"})
        return self._dict(r) if r and not r["deleted"] else None

    @staticmethod
    def _dict(r: sqlite3.Row) -> Dict[str, Any]:
        d = dict(r)
        for k in ("pub_types_json", "mesh_json", "keywords_json", "notices_json", "funding_json"):
            try:
                d[k[:-5]] = json.loads(d.get(k) or "null")
            except Exception:
                d[k[:-5]] = None
        return d

    # ------------------------------------------------------------------------------------------------ write
    def upsert(self, rec: Dict[str, Any], source: str = "", source_url: str = "") -> Tuple[int, str]:
        """Insert or update one record. Returns (id, 'new' | 'updated' | 'unchanged' | 'merged').
        An update never overwrites silently: the old values of every changed field go to record_history."""
        r = dict(rec)
        if not str(r.get("title") or "").strip():
            raise ValueError("a record needs a title")
        if r.get("year") is not None:
            try:
                r["year"] = int(str(r["year"])[:4])
            except ValueError:
                r["year"] = None
        if r.get("pmid") is not None:
            r["pmid"] = str(r["pmid"])
        if r.get("doi"):
            r["doi"] = str(r["doi"]).strip().lower()
        for k in ("pub_types", "mesh", "keywords", "notices", "funding"):
            if k in r and not isinstance(r.get(k), str):
                v = r.pop(k)
                if v not in (None, [], {}):              # 'nothing reported' never overwrites what another source reported
                    r[f"{k}_json"] = json.dumps(v, ensure_ascii=False)
        if isinstance(r.get("authors"), (list, tuple)):
            r["authors"] = "; ".join(str(a) for a in r["authors"] if a)
        r.setdefault("status", "current")
        h = _hash(r)
        now = _now()
        with self._lock:
            old = self.find(r)
            if old is None:
                cols = ["pmid", "doi", "pmcid", "nct", "title", "abstract", "authors", "journal", "year", "published_date",
                        "indexed_date", "revised_date", "pub_types_json", "mesh_json", "keywords_json", "status",
                        "notices_json", "licence", "licence_url", "reuse", "funding_json", "coi", "fulltext_path",
                        "fulltext_kind", "collection", "field_src_json"]
                r["field_src_json"] = json.dumps({k: source for k in TRACKED if r.get(k) not in (None, "")})
                vals = [r.get(c) for c in cols]
                vals[cols.index("collection")] = r.get("collection") or "literature"
                cur = self.c.execute(
                    f"INSERT INTO records ({', '.join(cols)}, source, source_url, retrieved_at, content_hash, title_key, "
                    f"first_author) VALUES ({', '.join('?' * len(cols))}, ?,?,?,?,?,?)",
                    (*vals, source, source_url, now, h, title_key(r.get("title")), first_author_key(r.get("authors"))))
                rid = int(cur.lastrowid)
                self.c.execute("INSERT INTO record_history (record_id, version, changed_at, change, fields_json, source) "
                               "VALUES (?,?,?,?,?,?)", (rid, 1, now, "created", None, source))
                self._fts_add(rid, r.get("title"), r.get("abstract"), r.get("mesh_json"))
                return rid, "new"
            rid = int(old["id"])
            # a filled field is replaced only by a source at least as authoritative as the one that filled it
            # (PubMed's title is not overwritten by a Crossref spelling); identifiers only ever ADD; a status only
            # gets worse (a retraction is never undone by a source that has not heard of it) unless the SAME source
            # that set it says otherwise
            try:
                fsrc = json.loads(old["field_src_json"] or "{}") or {}
            except Exception:
                fsrc = {}
            new_rank = source_rank(source)
            merged, new_src = {}, dict(fsrc)
            for k in TRACKED:
                nv, ov = r.get(k), old[k]
                owner = fsrc.get(k) or old["source"]
                if nv in (None, ""):
                    merged[k] = ov
                elif ov in (None, ""):
                    merged[k], new_src[k] = nv, source
                elif _same(k, nv, ov):
                    merged[k] = ov                       # same content, other spelling / punctuation: no new version
                elif k in ("doi", "pmcid", "nct"):
                    merged[k] = ov                       # identifiers only ever ADD
                elif k == "status":
                    from publication_status import worst
                    merged[k] = nv if (source and source == owner) else worst([ov, nv])
                    if merged[k] != ov:
                        new_src[k] = source
                elif k == "notices_json":
                    merged[k] = json.dumps(_merge_notices(ov, nv), ensure_ascii=False)
                elif k in ("licence",) and source_rank(source) < 4 and old["fulltext_path"]:
                    merged[k] = ov                       # the full text's licence is kept
                elif k == "authors" and source != owner and len(str(nv).split(";")) < len(str(ov).split(";")):
                    merged[k] = ov                       # a shorter author list from another source never wins
                elif k in ("coi", "funding_json") and source_rank(source, k) > source_rank(owner, k):
                    merged[k], new_src[k] = nv, source   # the full text's own disclosure beats the index record's
                elif source == owner or new_rank > source_rank(owner):
                    merged[k], new_src[k] = nv, source   # the same source updating itself, or a better source
                else:
                    merged[k] = ov
            changed = [k for k in TRACKED if merged.get(k) != old[k] and not (merged.get(k) in (None, "") and old[k] in (None, ""))]
            if old["deleted"] and r.get("deleted") is None:
                changed.append("deleted")
            if not changed:
                self.c.execute("UPDATE records SET retrieved_at=? WHERE id=?", (now, rid))
                return rid, "unchanged"
            ver = int(old["version"]) + 1
            self.c.execute("INSERT INTO record_history (record_id, version, changed_at, change, fields_json, old_json, source) "
                           "VALUES (?,?,?,?,?,?,?)", (rid, ver, now, "updated", json.dumps(changed),
                                                      json.dumps({k: old[k] for k in changed if k != "deleted"},
                                                                 ensure_ascii=False, default=str), source))
            sets = {k: merged[k] for k in changed if k != "deleted"}
            for k in ("indexed_date", "fulltext_path", "fulltext_kind", "licence_url", "reuse"):
                if r.get(k):
                    sets[k] = r[k]
            sets.update({"version": ver, "content_hash": _hash(merged), "retrieved_at": now, "deleted": 0,
                         "field_src_json": json.dumps(new_src),
                         "title_key": title_key(merged.get("title")), "first_author": first_author_key(merged.get("authors"))})
            if source:
                sets["source"] = source
            if source_url:
                sets["source_url"] = source_url
            if not old["deleted"]:
                self._fts_remove(rid, old["title"], old["abstract"], old["mesh_json"])
            self.c.execute(f"UPDATE records SET {', '.join(k + '=?' for k in sets)} WHERE id=?", (*sets.values(), rid))
            new = {**dict(old), **sets}
            self._fts_add(rid, new.get("title"), new.get("abstract"), new.get("mesh_json"))
            return rid, "updated"

    def mark_deleted(self, pmid: str, reason: str = "deleted by PubMed (DeleteCitation)", source: str = "") -> bool:
        """PubMed deletes a citation: the record is kept, flagged and hidden from search - and the change is logged."""
        with self._lock:
            r = self.c.execute("SELECT * FROM records WHERE pmid=? AND collection='literature'", (str(pmid),)).fetchone()
            if not r or r["deleted"]:
                return False
            self._fts_remove(r["id"], r["title"], r["abstract"], r["mesh_json"])
            self.c.execute("UPDATE records SET deleted=1, deleted_reason=?, version=? WHERE id=?",
                           (reason, int(r["version"]) + 1, r["id"]))
            self.c.execute("INSERT INTO record_history (record_id, version, changed_at, change, fields_json, source) "
                           "VALUES (?,?,?,?,?,?)", (r["id"], int(r["version"]) + 1, _now(), "deleted", json.dumps([reason]),
                                                    source))
            return True

    def rollback(self) -> None:
        with self._lock:
            self.c.rollback()

    def commit(self) -> None:
        with self._lock:
            self.c.commit()

    # external-content FTS (content='records'): a row is removed with the 'delete' command and its OLD values
    def _fts_add(self, rid: int, title: Any, abstract: Any, mesh_json: Any) -> None:
        self.c.execute("INSERT INTO records_fts (rowid, title, abstract, mesh_json) VALUES (?,?,?,?)",
                       (rid, title or "", abstract or "", mesh_json or ""))

    def _fts_remove(self, rid: int, title: Any, abstract: Any, mesh_json: Any) -> None:
        self.c.execute("INSERT INTO records_fts (records_fts, rowid, title, abstract, mesh_json) VALUES ('delete',?,?,?,?)",
                       (rid, title or "", abstract or "", mesh_json or ""))

    def history(self, record_id: int) -> List[Dict[str, Any]]:
        rows = self.c.execute("SELECT * FROM record_history WHERE record_id=? ORDER BY version", (record_id,)).fetchall()
        out = []
        for h in rows:
            d = dict(h)
            for k in ("fields_json", "old_json"):
                d[k[:-5]] = json.loads(d.pop(k) or "null")
            out.append(d)
        return out

    # ------------------------------------------------------------------------------------------------ passages
    def set_passages(self, record_id: int, sections: Sequence[Tuple[str, str]]) -> int:
        """sections: [(section_type, text)]. Replaces the record's passages (the old text stays in the full-text file)."""
        with self._lock:
            old = self.c.execute("SELECT id, text, section FROM passages WHERE record_id=?", (record_id,)).fetchall()
            for o in old:
                self.c.execute("INSERT INTO passages_fts (passages_fts, rowid, text, section) VALUES ('delete', ?, ?, ?)",
                               (o["id"], o["text"], o["section"] or ""))
            self.c.execute("DELETE FROM passages WHERE record_id=?", (record_id,))
            n = 0
            for sec, text in sections:
                for p in split_passages(text):
                    cur = self.c.execute("INSERT INTO passages (record_id, section, ord, text) VALUES (?,?,?,?)",
                                         (record_id, sec, n, p))
                    self.c.execute("INSERT INTO passages_fts (rowid, text, section) VALUES (?,?,?)",
                                   (cur.lastrowid, p, sec or ""))
                    n += 1
            return n

    def passages(self, record_id: int) -> List[Dict[str, Any]]:
        return [dict(r) for r in self.c.execute("SELECT * FROM passages WHERE record_id=? ORDER BY ord", (record_id,))]

    # ------------------------------------------------------------------------------------------------ findings
    def put_findings(self, record_id: int, extractor_version: str, text_hash: str, findings: List[Dict[str, Any]],
                     provider: str = "") -> None:
        with self._lock:
            self.c.execute("INSERT OR REPLACE INTO findings VALUES (?,?,?,?,?,?)",
                           (record_id, extractor_version, text_hash, json.dumps(findings, ensure_ascii=False), provider, _now()))
            self.c.commit()

    def get_findings(self, record_id: int, extractor_version: str, text_hash: Optional[str] = None) -> Optional[List[Dict[str, Any]]]:
        q = "SELECT findings_json FROM findings WHERE record_id=? AND extractor_version=?"
        args: Tuple[Any, ...] = (record_id, extractor_version)
        if text_hash:
            q += " AND text_hash=?"
            args += (text_hash,)
        r = self.c.execute(q + " ORDER BY created_at DESC LIMIT 1", args).fetchone()
        return json.loads(r[0]) if r else None

    # ------------------------------------------------------------------------------------------------ search
    def search(self, query: str, k: int = 50, collections: Sequence[str] = ("literature",),
               concepts: Optional[List[Iterable[str]]] = None) -> List[Dict[str, Any]]:
        """BM25 over title + abstract + MeSH (synonyms expanded). Deleted records are never returned."""
        from keyword_index import build_match
        m = build_match(query, concepts)
        if not m:
            return []
        cols = list(collections) or ["literature"]
        # rank inside the full-text index first (no join while scoring every match), then fetch only the top rows;
        # deleted / other-collection rows are filtered after - if too many were filtered, rank deeper (v26.1: ~2x faster)
        ok_cols = set(cols)
        depth = max(k * 3, 60)
        try:
            while True:
                ranked = self.c.execute("SELECT rowid, bm25(records_fts, 3.0, 1.0, 1.5) FROM records_fts "
                                        "WHERE records_fts MATCH ? ORDER BY 2 LIMIT ?", (m, depth)).fetchall()
                bm = {rid: b for rid, b in ranked}
                recs: Dict[int, sqlite3.Row] = {}
                ids = list(bm)
                for i in range(0, len(ids), 900):
                    chunk = ids[i:i + 900]
                    for r in self.c.execute(f"SELECT * FROM records WHERE id IN ({','.join('?' * len(chunk))})", chunk):
                        if not r["deleted"] and r["collection"] in ok_cols:
                            recs[r["id"]] = r
                if len(recs) >= k or len(ranked) < depth:
                    break
                depth *= 4
        except sqlite3.OperationalError:
            return []
        out = []
        for rid in ids:
            r = recs.get(rid)
            if r is None:
                continue
            d = self._dict(r)
            d["bm25"] = -float(bm[rid])
            out.append(d)
            if len(out) >= k:
                break
        return out

    def search_passages(self, query: str, record_ids: Optional[Sequence[int]] = None, k: int = 30,
                        concepts: Optional[List[Iterable[str]]] = None) -> List[Dict[str, Any]]:
        from keyword_index import build_match
        m = build_match(query, concepts)
        if not m:
            return []
        q = ("SELECT p.*, bm25(passages_fts) AS bm FROM passages_fts JOIN passages p ON p.id = passages_fts.rowid "
             "WHERE passages_fts MATCH ?")
        args: List[Any] = [m]
        if record_ids:
            q += f" AND p.record_id IN ({','.join('?' * len(record_ids))})"
            args += list(record_ids)
        try:
            rows = self.c.execute(q + " ORDER BY bm LIMIT ?", (*args, k)).fetchall()
        except sqlite3.OperationalError:
            return []
        return [{**dict(r), "bm25": -float(r["bm"])} for r in rows]

    def stats(self) -> Dict[str, Any]:
        c = self.c
        by_coll = {r[0]: r[1] for r in c.execute("SELECT collection, COUNT(*) FROM records WHERE deleted=0 GROUP BY collection")}
        by_status = {r[0]: r[1] for r in c.execute("SELECT status, COUNT(*) FROM records WHERE deleted=0 GROUP BY status")}
        return {"path": str(self.path), "records": by_coll, "status": by_status,
                "deleted": c.execute("SELECT COUNT(*) FROM records WHERE deleted=1").fetchone()[0],
                "passages": c.execute("SELECT COUNT(*) FROM passages").fetchone()[0],
                "with_full_text": c.execute("SELECT COUNT(*) FROM records WHERE fulltext_path IS NOT NULL").fetchone()[0],
                "pre_read": c.execute("SELECT COUNT(DISTINCT record_id) FROM findings").fetchone()[0],
                "last_update": self.get_sync("last_update"), "scope": self.get_sync("scope"),
                "bytes": self.path.stat().st_size if self.path.exists() else 0, "version": LIBRARY_VERSION}


def to_candidate(rec: Dict[str, Any]) -> Any:
    """A library record as a retrieval.models.PaperCandidate (source 'local_library')."""
    from retrieval.models import PaperCandidate
    authors = [a.strip() for a in str(rec.get("authors") or "").split(";") if a.strip()]
    c = PaperCandidate(title=rec.get("title") or "", authors=authors, journal=rec.get("journal"), year=rec.get("year"),
                       doi=rec.get("doi"), pmid=rec.get("pmid"), pmcid=rec.get("pmcid"), abstract=rec.get("abstract"),
                       is_open_access=bool(rec.get("fulltext_path")), source_db="local_library",
                       url=rec.get("source_url"))
    if rec.get("fulltext_path") and rec.get("fulltext_kind") == "jats_xml" and Path(rec["fulltext_path"]).exists():
        c.local_xml_path = rec["fulltext_path"]
        c.has_full_text_xml = True
    c.raw_metadata = {"library_id": rec.get("id"), "pub_types": rec.get("pub_types"), "mesh": rec.get("mesh"),
                      "status": rec.get("status"), "nct": rec.get("nct"), "published_date": rec.get("published_date"),
                      "indexed_date": rec.get("indexed_date"), "revised_date": rec.get("revised_date"),
                      "licence": rec.get("licence"), "collection": rec.get("collection"),
                      "funding": rec.get("funding"), "coi": rec.get("coi")}
    return c


def exists() -> bool:
    """True when a library with at least one record is present (no folder is created by asking)."""
    d = Path(os.getenv("BIOSYN_LIBRARY_DIR") or ROOT / "library")
    p = d / "library.db"
    if not p.exists():
        return False
    try:
        c = sqlite3.connect(f"file:{p}?mode=ro", uri=True, timeout=5)
        n = c.execute("SELECT COUNT(*) FROM records WHERE deleted=0").fetchone()[0]
        c.close()
        return n > 0
    except Exception:
        return False
