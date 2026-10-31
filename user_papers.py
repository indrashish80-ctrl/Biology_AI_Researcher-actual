"""
user_papers.py  (v26)
=====================
Papers, reports and documents YOU add (PDF, JATS/XML, plain text / Markdown). They are stored in the local library in
a collection of their own ('user:<name>', default 'user:my-papers'):

  * kept SEPARATE from the shared literature: a research run searches them only when you tick them for that run
    (ResearchRequest.user_collections), and `share()` is the only way to move one into the literature collection
    (recorded in its history);
  * parsed by code (sections, tables for XML, funding and conflict-of-interest statements), classified (human / animal /
    cell / review ... with study_classifier), and compared with the library: the closest published records on the same
    question are listed next to it;
  * the original file is kept unchanged in library/user_files/<collection>/ ; deleting removes the copy and hides the
    record (the record's history keeps the fact that it existed).
Spreadsheets / data tables go through the existing dataset upload (deterministic statistics), not here.
"""
from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import library_db as L

USER_PAPERS_VERSION = "1.0"
ALLOWED = {".pdf", ".xml", ".nxml", ".txt", ".md"}
MAX_BYTES = 60 * 1024 * 1024


def _collection(name: Optional[str]) -> str:
    n = re.sub(r"[^a-z0-9_-]+", "-", (name or "my-papers").lower()).strip("-")[:40] or "my-papers"
    return f"user:{n}"


def _sections_from_text(text: str) -> List[tuple]:
    from pdf_ingest import split_sections
    return [(t, body) for t, _title, body in split_sections(text) if body.strip()]


def add(data: bytes, filename: str, collection: Optional[str] = None, lib: Optional[L.Library] = None) -> Dict[str, Any]:
    ext = Path(filename or "").suffix.lower()
    if ext not in ALLOWED:
        raise ValueError(f"unsupported file type '{ext or '?'}' - add a PDF, a JATS/XML article, or a .txt / .md file")
    if not data:
        raise ValueError("the file is empty")
    if len(data) > MAX_BYTES:
        raise ValueError("file larger than 60 MB")
    coll = _collection(collection.split(":", 1)[-1] if collection else None)
    own = lib is None
    lib = lib or L.Library()
    try:
        sha = hashlib.sha256(data).hexdigest()
        folder = lib.dir / "user_files" / coll.split(":", 1)[1]
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f"{sha[:16]}{ext}"
        path.write_bytes(data)
        import study_funding as sf
        try:
            return _add_parsed(lib, data, filename, ext, coll, path, sf)
        except ValueError:
            raise
        except Exception as exc:                     # a damaged PDF / malformed XML is the user's file, not a server error
            path.unlink(missing_ok=True)
            raise ValueError(f"the file could not be read ({type(exc).__name__}: {str(exc)[:160]})")
    finally:
        if own:
            lib.close()


def _add_parsed(lib, data, filename, ext, coll, path, sf):
    if True:
        if ext in (".xml", ".nxml"):
            from library_loader import parse_jats
            parsed = parse_jats(data.decode("utf-8", "replace"))
            rec, sections = parsed["record"], parsed["sections"]
            kind = "jats_xml"
        else:
            if ext == ".pdf":
                from pdf_ingest import extract_pdf_text
                text = extract_pdf_text(path)
                kind = "pdf"
            else:
                text = data.decode("utf-8", "replace")
                kind = "text"
            if len(text.strip()) < 200:
                raise ValueError("almost no text could be read (a scanned PDF needs OCR first)")
            sections = _sections_from_text(text)
            first = next((ln.strip() for ln in text.splitlines() if len(ln.strip()) > 15), Path(filename).stem)
            abstract = next((b for t, b in sections if t == "abstract"), None) or (sections[0][1][:1500] if sections else None)
            fc = sf.from_text(text)
            doi = re.search(r"\b(10\.\d{4,9}/[^\s\"<>]+[^\s\"<>.,;])", text)
            rec = {"title": first[:300], "abstract": abstract, "doi": doi.group(1).lower() if doi else None,
                   "funding": fc["funding"] or None, "coi": fc["coi"]}
        try:
            from study_classifier import classify_study
            cls = classify_study({"title": rec.get("title"), "abstract": rec.get("abstract") or ""})
        except Exception:
            cls = {"evidence_type": "unclear"}
        rec.update({"collection": coll, "fulltext_path": str(path), "fulltext_kind": kind,
                    "keywords": [cls.get("evidence_type")], "licence": "your own file (not shared)", "reuse": "private"})
        # matched only inside your collection: a user copy never merges into the shared literature record
        rid, how = lib.upsert({**rec, "title": rec.get("title") or Path(filename).stem}, source="user",
                              source_url=f"file:{path.name}")
        n = lib.set_passages(rid, sections)
        lib.log(rid, "user_upload", "ok", USER_PAPERS_VERSION, f"{filename}: {n} passages, {kind}")
        lib.commit()
        related = compare(lib, rid)
        return {"record_id": rid, "collection": coll, "status": how, "title": rec.get("title"), "passages": n,
                "basis": kind, "study_type": cls.get("evidence_type"),
                "funding": sf.profile(rec.get("funding") if isinstance(rec.get("funding"), dict) else None, rec.get("coi")),
                "related_literature": related}


def compare(lib: L.Library, record_id: int, k: int = 8) -> List[Dict[str, Any]]:
    """The published library records closest to a user paper (same question), for side-by-side reading."""
    r = lib.get(record_id)
    if not r:
        return []
    import library_search as ls
    hits = ls.search(lib, f"{r.get('title') or ''}", [(r.get("abstract") or "")[:300]], k=k, collections=("literature",))
    return [{"record_id": h["id"], "title": h.get("title"), "year": h.get("year"), "pmid": h.get("pmid"),
             "status": h.get("status"), "score": h.get("score")} for h in hits]


def list_papers(collection: Optional[str] = None) -> List[Dict[str, Any]]:
    if not L.exists():
        return []
    with L.Library() as lib:
        q = "SELECT id, title, collection, year, fulltext_kind, retrieved_at FROM records WHERE deleted=0 AND collection LIKE 'user:%'"
        args: tuple = ()
        if collection:
            q += " AND collection=?"
            args = (_collection(collection.split(":", 1)[-1]),)
        return [dict(x) for x in lib.c.execute(q + " ORDER BY id DESC", args)]


def collections() -> List[str]:
    if not L.exists():
        return []
    with L.Library() as lib:
        return [r[0] for r in lib.c.execute("SELECT DISTINCT collection FROM records WHERE deleted=0 AND collection LIKE 'user:%'")]


def delete(record_id: int) -> bool:
    with L.Library() as lib:
        r = lib.get(record_id)
        if not r or not str(r.get("collection") or "").startswith("user:"):
            return False                       # only your own uploads can be deleted here
        with lib._lock:
            lib._fts_remove(record_id, r["title"], r["abstract"], r["mesh_json"])
            lib.c.execute("UPDATE records SET deleted=1, deleted_reason='deleted by the user', version=version+1 WHERE id=?",
                          (record_id,))
            lib.c.execute("INSERT INTO record_history (record_id, version, changed_at, change, source) VALUES (?,?,?,?,?)",
                          (record_id, int(r["version"]) + 1, L._now(), "deleted by the user", "user"))
            lib.c.commit()
        try:
            if r.get("fulltext_path"):
                Path(r["fulltext_path"]).unlink(missing_ok=True)
        except Exception:
            pass
        return True


def share(record_id: int) -> Optional[int]:
    """Move one of your papers into the shared literature collection (an explicit, logged decision). When the same paper
    is already in the literature it is MERGED into that record (fields it lacks, its full text); your copy is then
    hidden with a note pointing to the shared record. Returns the shared record's id."""
    with L.Library() as lib:
        r = lib.get(record_id)
        if not r or not str(r.get("collection") or "").startswith("user:") or r.get("deleted"):
            return None
        keep = {k: r.get(k) for k in ("pmid", "doi", "pmcid", "nct", "title", "abstract", "authors", "journal", "year",
                                      "published_date", "coi", "fulltext_path", "fulltext_kind")}
        keep.update({"funding": r.get("funding"), "keywords": r.get("keywords"), "collection": "literature"})
        target, how = lib.upsert(keep, source="user", source_url=r.get("source_url") or "")
        if len(lib.passages(target)) < len(lib.passages(record_id)):      # the fuller text wins (e.g. abstract -> full)
            lib.set_passages(target, [(p["section"], p["text"]) for p in lib.passages(record_id)])
        with lib._lock:
            lib._fts_remove(record_id, r["title"], r["abstract"], r["mesh_json"])
            lib.c.execute("UPDATE records SET deleted=1, deleted_reason=?, version=version+1 WHERE id=?",
                          (f"shared into the literature as record {target}", record_id))
            lib.c.execute("INSERT INTO record_history (record_id, version, changed_at, change, old_json, source) "
                          "VALUES (?,?,?,?,?,?)", (record_id, int(r["version"]) + 1, L._now(),
                                                   f"shared by the user as literature record {target} ({how})",
                                                   f'{{"collection": "{r["collection"]}"}}', "user"))
            lib.c.commit()
        return target
