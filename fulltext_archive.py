"""
fulltext_archive.py  (v25)
==========================
Every full text is downloaded ONCE, kept with its licence, and reused by every later run.

    papers/archive/index.db           one row per archived file (survives deleting the run database)
    papers/archive/<sha>.<xml|pdf>    the file itself, named by its content hash (identical copies stored once)

For each file: the identifiers it belongs to (DOI / PMID / PMCID / dedup key), the kind (jats_xml / pdf), where it
came from (PMC, Europe PMC, Unpaywall, OpenAlex ...), when, its SHA-256, and its LICENCE:

    licence_from_jats(xml)        the <license> of the article's <permissions> (CC BY, CC BY-NC, ...) or None
    licence_from_unpaywall(info)  best_oa_location.license
    reuse(licence)                'redistribute with attribution' | 'non-commercial reuse with attribution' |
                                  'no derivatives' | 'read only - do not redistribute' (unknown licence)

The run database gets a per-paper text-basis table (paper_text_basis) so that every finding knows whether it was
read from JATS XML, a PDF or only the abstract (quote_guard's confidence label).

    lookup(keys)                               -> archived record or None (download skipped)
    store(keys, content, kind, source, licence) -> record
    record_basis(db, paper_id, basis, record)  /  basis_for(db, paper_id)
"""
from __future__ import annotations

import hashlib
import re
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Union

ARCHIVE_VERSION = "1.0"
ROOT = Path(__file__).resolve().parent / "papers" / "archive"
_LOCK = threading.Lock()
_SCHEMA = """
CREATE TABLE IF NOT EXISTS files (sha256 TEXT PRIMARY KEY, kind TEXT NOT NULL, path TEXT NOT NULL, bytes INTEGER,
    source TEXT, licence TEXT, licence_url TEXT, reuse TEXT, retrieved_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS keys (key TEXT NOT NULL, sha256 TEXT NOT NULL, PRIMARY KEY (key, sha256));
CREATE INDEX IF NOT EXISTS idx_keys_key ON keys(key);
"""
_RUN_SCHEMA = """
CREATE TABLE IF NOT EXISTS paper_text_basis (paper_id INTEGER PRIMARY KEY, basis TEXT NOT NULL, sha256 TEXT,
    licence TEXT, reuse TEXT, recorded_at TEXT NOT NULL);
"""


def _root() -> Path:
    import os
    r = Path(os.getenv("BIOSYN_ARCHIVE_DIR") or ROOT)
    r.mkdir(parents=True, exist_ok=True)
    return r


def _conn() -> sqlite3.Connection:
    c = sqlite3.connect(str(_root() / "index.db"), timeout=30)
    c.row_factory = sqlite3.Row
    c.executescript(_SCHEMA)
    return c


def keys_for(doi: Any = None, pmid: Any = None, pmcid: Any = None, dedup_key: Any = None) -> List[str]:
    out = []
    if doi:
        out.append("doi:" + re.sub(r"^(?:https?://(?:dx\.)?doi\.org/|doi:\s*)", "", str(doi).strip(), flags=re.I).lower())
    if pmid:
        out.append(f"pmid:{str(pmid).strip()}")
    if pmcid:
        p = str(pmcid).strip().upper()
        out.append("pmcid:" + (p if p.startswith("PMC") else "PMC" + p))
    if dedup_key:
        out.append(f"key:{str(dedup_key).strip().lower()}")
    return out


def reuse(licence: Optional[str]) -> str:
    l = (licence or "").lower()
    if not l:
        return "read only - do not redistribute (licence unknown)"
    if re.search(r"\bnd\b|no-?deriv", l):
        return "no derivatives - quote, do not adapt"
    if re.search(r"\bnc\b|non-?commercial", l):
        return "non-commercial reuse with attribution"
    if re.search(r"cc[- ]?by|creativecommons\.org/licenses/by|cc0|public domain|publicdomain", l):
        return "redistribute with attribution" if "cc0" not in l and "public" not in l else "public domain"
    if re.search(r"open[- ]access|author manuscript|nihms|pmc", l):
        return "read only - open access to read, licence restricts redistribution"
    return "read only - do not redistribute"


def licence_from_jats(xml: str) -> Dict[str, Optional[str]]:
    """{'licence', 'licence_url'} from <permissions><license license-type=.. xlink:href=..> / <ali:license_ref>."""
    t = xml or ""
    m = re.search(r"<license\b([^>]*)>(.*?)</license>", t, re.S | re.I)
    url = None
    lic = None
    if m:
        attrs, body = m.group(1), m.group(2)
        u = re.search(r'(?:xlink:href|href)="([^"]+)"', attrs)
        url = u.group(1) if u else None
        typ = re.search(r'license-type="([^"]+)"', attrs)
        lic = typ.group(1) if typ else None
        ref = re.search(r"<ali:license_ref[^>]*>([^<]+)</ali:license_ref>", body)
        url = url or (ref.group(1).strip() if ref else None)
        if not lic:
            txt = re.sub(r"<[^>]+>", " ", body)
            cc = re.search(r"(?:Creative Commons\s+)?(?:CC[ -]?BY(?:[ -](?:NC|SA|ND)){0,2}|CC0)", txt, re.I)
            lic = cc.group(0).strip() if cc else (txt.strip()[:80] or None)
    generic = not lic or re.fullmatch(r"(?i)open[- ]?access|free|openaccess|unknown", lic.strip()) is not None
    if url and generic:
        # license-type="open-access" says only that the text is free to read; the CC URL is the actual licence
        cc = re.search(r"licenses/([a-z\-]+)/", url)
        if cc:
            lic = f"CC {cc.group(1).upper()}"
        elif re.search(r"publicdomain", url, re.I):
            lic = "public domain"
        elif not lic:
            lic = url
    return {"licence": lic, "licence_url": url}


def licence_from_unpaywall(info: Optional[Dict[str, Any]]) -> Dict[str, Optional[str]]:
    loc = (info or {}).get("best_oa_location") or {}
    lic = loc.get("license") or (info or {}).get("license")
    return {"licence": lic, "licence_url": loc.get("url_for_pdf") or loc.get("url")}


def lookup(keys: Iterable[str], kind: Optional[str] = None) -> Optional[Dict[str, Any]]:
    keys = [k for k in keys if k]
    if not keys:
        return None
    with _LOCK:
        c = _conn()
        try:
            q = (f"SELECT f.* FROM keys k JOIN files f ON f.sha256 = k.sha256 WHERE k.key IN ({','.join('?' * len(keys))})"
                 + (" AND f.kind = ?" if kind else "") + " ORDER BY CASE f.kind WHEN 'jats_xml' THEN 0 ELSE 1 END LIMIT 1")
            r = c.execute(q, tuple(keys) + ((kind,) if kind else ())).fetchone()
        finally:
            c.close()
    if not r:
        return None
    d = dict(r)
    if not Path(d["path"]).exists():
        return None
    return d


def store(keys: Iterable[str], content: Union[bytes, str, Path], kind: str, source: str = "",
          licence: Optional[str] = None, licence_url: Optional[str] = None) -> Dict[str, Any]:
    if isinstance(content, Path):
        data = content.read_bytes()
    elif isinstance(content, str):
        data = content.encode("utf-8")
    else:
        data = content
    sha = hashlib.sha256(data).hexdigest()
    ext = "xml" if kind == "jats_xml" else "pdf" if kind == "pdf" else "bin"
    path = _root() / f"{sha[:2]}" / f"{sha}.{ext}"
    if kind == "jats_xml" and not licence:
        lic = licence_from_jats(data.decode("utf-8", "replace"))
        licence, licence_url = lic["licence"], licence_url or lic["licence_url"]
    with _LOCK:
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            path.write_bytes(data)
        c = _conn()
        try:
            c.execute("INSERT OR IGNORE INTO files (sha256, kind, path, bytes, source, licence, licence_url, reuse, retrieved_at) "
                      "VALUES (?,?,?,?,?,?,?,?,?)", (sha, kind, str(path), len(data), source, licence, licence_url,
                                                    reuse(licence), datetime.now(timezone.utc).isoformat(timespec="seconds")))
            if licence:
                c.execute("UPDATE files SET licence=COALESCE(licence, ?), licence_url=COALESCE(licence_url, ?), reuse=CASE WHEN licence IS NULL THEN ? ELSE reuse END "
                          "WHERE sha256=?", (licence, licence_url, reuse(licence), sha))
            for k in keys:
                if k:
                    c.execute("INSERT OR IGNORE INTO keys (key, sha256) VALUES (?, ?)", (k, sha))
            c.commit()
            r = dict(c.execute("SELECT * FROM files WHERE sha256=?", (sha,)).fetchone())
        finally:
            c.close()
    return r


def store_candidate_file(cand: Any, path: Union[str, Path], kind: str, source: str = "",
                         unpaywall: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
    """Archive a file just downloaded for a search candidate. Never raises."""
    try:
        p = Path(path)
        if not p.exists():
            return None
        lic = licence_from_unpaywall(unpaywall) if unpaywall else {"licence": None, "licence_url": None}
        return store(keys_for(getattr(cand, "doi", None), getattr(cand, "pmid", None), getattr(cand, "pmcid", None),
                              getattr(cand, "dedup_key", None)), p, kind, source, lic["licence"], lic["licence_url"])
    except Exception:
        return None


def record_basis(db: Any, paper_id: int, basis: str, rec: Optional[Dict[str, Any]] = None) -> None:
    try:
        db._conn.executescript(_RUN_SCHEMA)
        db._conn.execute("INSERT OR REPLACE INTO paper_text_basis (paper_id, basis, sha256, licence, reuse, recorded_at) "
                         "VALUES (?,?,?,?,?,?)", (paper_id, basis, (rec or {}).get("sha256"), (rec or {}).get("licence"),
                                                  (rec or {}).get("reuse"), datetime.now(timezone.utc).isoformat(timespec="seconds")))
        db._conn.commit()
    except Exception:
        pass


def basis_for(db: Any, paper_id: int) -> Optional[str]:
    try:
        r = db._conn.execute("SELECT basis FROM paper_text_basis WHERE paper_id=?", (paper_id,)).fetchone()
    except sqlite3.OperationalError:
        return None
    if not r:
        return None
    b = r[0]
    return {"unpaywall_pdf": "full_text_pdf", "oa_pdf": "full_text_pdf", "pdf": "full_text_pdf",
            "jats_xml": "jats_xml", "abstract": "abstract"}.get(b, b)


def licences(db: Any, paper_ids: Optional[List[int]] = None) -> List[Dict[str, Any]]:
    try:
        sql = "SELECT b.paper_id, b.basis, b.licence, b.reuse, p.title FROM paper_text_basis b LEFT JOIN papers p ON p.id=b.paper_id"
        rows = [dict(r) for r in db._conn.execute(sql).fetchall()]
    except sqlite3.OperationalError:
        return []
    return [r for r in rows if paper_ids is None or r["paper_id"] in set(paper_ids)]


def summary() -> Dict[str, Any]:
    with _LOCK:
        c = _conn()
        try:
            n = c.execute("SELECT kind, COUNT(*), COALESCE(SUM(bytes),0) FROM files GROUP BY kind").fetchall()
            lic = c.execute("SELECT reuse, COUNT(*) FROM files GROUP BY reuse").fetchall()
        finally:
            c.close()
    return {"files": {k: {"n": a, "bytes": b} for k, a, b in n}, "by_reuse": {r or "unknown": k for r, k in lic},
            "path": str(_root()), "version": ARCHIVE_VERSION}
