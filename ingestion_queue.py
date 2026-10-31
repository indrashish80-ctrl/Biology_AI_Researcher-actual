"""
ingestion_queue.py  (v25)
=========================
Background ingestion and incremental updates.

    Deduplicator        one canonical record per paper, even when identifiers are MISSING or only partly shared:
                        every DOI / PMID / PMCID of a record is an alias of it, and a record without any identifier
                        is matched on (normalised title, year +-1, first author's surname), then on near-identical
                        titles (>= 0.93 similar, same year +-1)
    IngestionQueue      a persistent job queue (table ingest_jobs) with a RETRY QUEUE: a failed job is retried
                        with exponential back-off (1, 2, 4 ... minutes, at most 6 h) up to 6 attempts, then parked
                        as 'failed' with its last error
    watched topics      (table watch_topics) a question / query re-searched on PubMed every day for papers added
                        since the last run; new papers are deduplicated, stored, indexed for keyword search and
                        queued for full-text retrieval and a publication-status check
    worker              start_worker(db_path) runs due jobs in a daemon thread (the web server starts it);
                        run_due(db) runs one round (tests, cron, command line)

    python ingestion_queue.py --watch "metformin breast cancer"   # add a daily topic
    python ingestion_queue.py --run                                # run everything that is due now
"""
from __future__ import annotations

import argparse
import difflib
import json
import os
import re
import threading
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

QUEUE_VERSION = "1.0"
MAX_ATTEMPTS = 6
SCHEMA = """
CREATE TABLE IF NOT EXISTS ingest_jobs (
    id INTEGER PRIMARY KEY AUTOINCREMENT, kind TEXT NOT NULL, payload_json TEXT NOT NULL, dedup TEXT,
    status TEXT NOT NULL DEFAULT 'queued', attempts INTEGER NOT NULL DEFAULT 0, next_attempt_at TEXT NOT NULL,
    last_error TEXT, result_json TEXT, created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
    UNIQUE(kind, dedup)
);
CREATE INDEX IF NOT EXISTS idx_jobs_due ON ingest_jobs(status, next_attempt_at);
CREATE TABLE IF NOT EXISTS watch_topics (
    id INTEGER PRIMARY KEY AUTOINCREMENT, query TEXT NOT NULL UNIQUE, interval_hours REAL NOT NULL DEFAULT 24,
    last_run_at TEXT, enabled INTEGER NOT NULL DEFAULT 1, research_question_id INTEGER, created_at TEXT NOT NULL
);
"""


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(d: datetime) -> str:
    return d.isoformat(timespec="seconds")


# ------------------------------------------------------------------------------------------------ dedup
def norm_title(t: Optional[str]) -> str:
    t = re.sub(r"<[^>]+>", " ", t or "").lower()
    t = re.sub(r"[^a-z0-9]+", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def _surname(authors: Any) -> str:
    a = authors[0] if isinstance(authors, (list, tuple)) and authors else (authors or "")
    a = str(a).split(",")[0].strip()
    parts = [p for p in re.split(r"\s+", a) if p]
    if not parts:
        return ""
    # 'Smith J' / 'Smith, John' / 'John Smith' -> 'smith'
    cand = parts[0] if len(parts) > 1 and len(parts[-1]) <= 2 else parts[-1]
    return re.sub(r"[^a-z]", "", cand.lower())


def _year(y: Any) -> Optional[int]:
    m = re.search(r"(19|20)\d{2}", str(y or ""))
    return int(m.group(0)) if m else None


def _ids(rec: Dict[str, Any]) -> List[str]:
    out = []
    doi = rec.get("doi")
    if doi:
        out.append("doi:" + re.sub(r"^(?:https?://(?:dx\.)?doi\.org/|doi:\s*)", "", str(doi).strip(), flags=re.I).lower())
    if rec.get("pmid"):
        m = re.search(r"\d+", str(rec["pmid"]))
        if m:
            out.append("pmid:" + m.group(0))
    if rec.get("pmcid"):
        p = str(rec["pmcid"]).upper().strip()
        out.append("pmcid:" + (p if p.startswith("PMC") else "PMC" + p))
    return out


class Deduplicator:
    """add(record) -> (canonical_id, is_new). Records are dicts (or objects with the same attributes)."""

    def __init__(self, title_threshold: float = 0.93):
        self.alias: Dict[str, int] = {}
        self.fuzzy: Dict[Tuple[str, Optional[int], str], int] = {}
        self.by_year: Dict[Optional[int], List[Tuple[str, int]]] = {}
        self.records: Dict[int, Dict[str, Any]] = {}
        self.th = title_threshold
        self._next = 0

    @staticmethod
    def _as_dict(r: Any) -> Dict[str, Any]:
        if isinstance(r, dict):
            return r
        return {k: getattr(r, k, None) for k in ("doi", "pmid", "pmcid", "title", "year", "authors")}

    def find(self, rec: Any) -> Optional[int]:
        r = self._as_dict(rec)
        for k in _ids(r):
            if k in self.alias:
                return self.alias[k]
        t, y, s = norm_title(r.get("title")), _year(r.get("year")), _surname(r.get("authors"))
        if not t or len(t) < 15:
            return None
        for yy in (y, (y or 0) - 1, (y or 0) + 1) if y else (None,):
            hit = self.fuzzy.get((t, yy, s)) if s else None
            if hit is not None:
                return hit
        for yy in ((y, y - 1, y + 1) if y else (None,)):
            for t2, cid in self.by_year.get(yy, []):
                if abs(len(t2) - len(t)) <= 0.1 * len(t) and difflib.SequenceMatcher(None, t, t2).ratio() >= self.th:
                    return cid
        return None

    def add(self, rec: Any) -> Tuple[int, bool]:
        r = self._as_dict(rec)
        cid = self.find(r)
        new = cid is None
        if new:
            cid = self._next
            self._next += 1
            self.records[cid] = dict(r)
        else:
            base = self.records[cid]
            for k, v in r.items():                       # complementary metadata fills the gaps
                if v and not base.get(k):
                    base[k] = v
        for k in _ids(r):
            self.alias.setdefault(k, cid)
        t, y, s = norm_title(r.get("title")), _year(r.get("year")), _surname(r.get("authors"))
        if t:
            if s:
                self.fuzzy.setdefault((t, y, s), cid)
            self.by_year.setdefault(y, []).append((t, cid))
        return cid, new


def dedupe(records: Iterable[Any]) -> List[Any]:
    """The first record of each duplicate group (later duplicates merge into it when they are dicts)."""
    d = Deduplicator()
    out, seen = [], set()
    for r in records:
        cid, new = d.add(r)
        if new and cid not in seen:
            seen.add(cid)
            out.append(r)
    return out


# ------------------------------------------------------------------------------------------------ queue
class IngestionQueue:
    def __init__(self, db: Any):
        self.db = db
        db._conn.executescript(SCHEMA)
        db._conn.commit()

    def enqueue(self, kind: str, payload: Dict[str, Any], dedup: Optional[str] = None, delay_s: float = 0) -> Optional[int]:
        now = _now()
        cur = self.db._conn.execute(
            "INSERT OR IGNORE INTO ingest_jobs (kind, payload_json, dedup, next_attempt_at, created_at, updated_at) "
            "VALUES (?,?,?,?,?,?)", (kind, json.dumps(payload, ensure_ascii=False), dedup or json.dumps(payload, sort_keys=True)[:300],
                                     _iso(now + timedelta(seconds=delay_s)), _iso(now), _iso(now)))
        self.db._conn.commit()
        # INSERT OR IGNORE keeps the previous lastrowid when the row is ignored: use rowcount
        return cur.lastrowid if cur.rowcount == 1 else None

    def due(self, limit: int = 20) -> List[Dict[str, Any]]:
        rows = self.db._conn.execute("SELECT * FROM ingest_jobs WHERE status IN ('queued','retry') AND next_attempt_at <= ? "
                                     "ORDER BY next_attempt_at LIMIT ?", (_iso(_now()), limit)).fetchall()
        return [dict(r) for r in rows]

    def _set(self, job_id: int, **kw) -> None:
        kw["updated_at"] = _iso(_now())
        sets = ", ".join(f"{k}=?" for k in kw)
        self.db._conn.execute(f"UPDATE ingest_jobs SET {sets} WHERE id=?", tuple(kw.values()) + (job_id,))
        self.db._conn.commit()

    def run_job(self, job: Dict[str, Any], handlers: Dict[str, Callable[[Any, Dict[str, Any]], Any]]) -> Dict[str, Any]:
        h = handlers.get(job["kind"])
        if h is None:
            self._set(job["id"], status="failed", last_error=f"no handler for {job['kind']}")
            return {"id": job["id"], "status": "failed"}
        self._set(job["id"], status="running")
        try:
            res = h(self.db, json.loads(job["payload_json"]))
            self._set(job["id"], status="done", result_json=json.dumps(res, ensure_ascii=False, default=str)[:20000],
                      attempts=job["attempts"] + 1, last_error=None)
            return {"id": job["id"], "status": "done", "result": res}
        except Exception as exc:
            n = job["attempts"] + 1
            if n >= MAX_ATTEMPTS:
                self._set(job["id"], status="failed", attempts=n, last_error=f"{type(exc).__name__}: {str(exc)[:300]}")
                return {"id": job["id"], "status": "failed", "error": str(exc)[:200]}
            wait = min(6 * 3600, 60 * (2 ** (n - 1)))
            self._set(job["id"], status="retry", attempts=n, last_error=f"{type(exc).__name__}: {str(exc)[:300]}",
                      next_attempt_at=_iso(_now() + timedelta(seconds=wait)))
            return {"id": job["id"], "status": "retry", "retry_in_s": wait, "error": str(exc)[:200]}

    def counts(self) -> Dict[str, int]:
        return {s: n for s, n in self.db._conn.execute("SELECT status, COUNT(*) FROM ingest_jobs GROUP BY status").fetchall()}

    # ---- watched topics
    def watch(self, query: str, interval_hours: float = 24, research_question_id: Optional[int] = None) -> int:
        self.db._conn.execute("INSERT OR IGNORE INTO watch_topics (query, interval_hours, research_question_id, created_at) "
                              "VALUES (?,?,?,?)", (query.strip(), interval_hours, research_question_id, _iso(_now())))
        self.db._conn.commit()
        return int(self.db._conn.execute("SELECT id FROM watch_topics WHERE query=?", (query.strip(),)).fetchone()[0])

    def topics_due(self) -> List[Dict[str, Any]]:
        rows = [dict(r) for r in self.db._conn.execute("SELECT * FROM watch_topics WHERE enabled=1").fetchall()]
        now = _now()
        out = []
        for r in rows:
            last = datetime.fromisoformat(r["last_run_at"]) if r.get("last_run_at") else None
            if last is None or now - last >= timedelta(hours=float(r["interval_hours"] or 24)):
                out.append(r)
        return out

    def schedule_topics(self) -> int:
        n = 0
        for t in self.topics_due():
            since = t.get("last_run_at") or _iso(_now() - timedelta(days=30))
            if self.enqueue("topic_update", {"topic_id": t["id"], "query": t["query"], "since": since,
                                             "research_question_id": t.get("research_question_id")},
                            dedup=f"topic:{t['id']}:{since[:10]}"):
                n += 1
        return n


# ------------------------------------------------------------------------------------------------ handlers
def _pubmed_since(query: str, since_iso: str, max_results: int = 50):
    """PubMed records added (Entrez date) since `since_iso`; uses the engine's PubMed client."""
    from retrieval.pubmed import PubMedClient
    d = since_iso[:10].replace("-", "/")
    q = f'({query}) AND ("{d}"[EDAT] : "3000"[EDAT])'
    return PubMedClient().search_pubmed(q, max_results=max_results)


def handle_topic_update(db: Any, p: Dict[str, Any]) -> Dict[str, Any]:
    if os.getenv("BIOSYN_OFFLINE"):
        raise RuntimeError("offline - PubMed not reachable (will retry)")
    from retrieval.source_manager import SourceManager
    cands = _pubmed_since(p["query"], p["since"])
    d = Deduplicator()
    for row in db._conn.execute("SELECT doi, pmid, pmcid, title, year, authors FROM papers").fetchall():
        d.add(dict(row))
    mgr = SourceManager()
    new_ids = []
    q = IngestionQueue(db)
    for c in cands:
        cid, new = d.add(c)
        if not new:
            continue
        pid = mgr.ingest_candidate(db, c, p.get("research_question_id"))
        new_ids.append(pid)
        q.enqueue("fulltext", {"paper_id": pid}, dedup=f"ft:{pid}")
        q.enqueue("status_check", {"paper_id": pid}, dedup=f"st:{pid}", delay_s=3600)
    try:
        from keyword_index import KeywordIndex
        KeywordIndex(db).sync(new_ids)
    except Exception:
        pass
    db._conn.execute("UPDATE watch_topics SET last_run_at=? WHERE id=?", (_iso(_now()), p["topic_id"]))
    db._conn.commit()
    return {"found": len(cands), "new": len(new_ids), "paper_ids": new_ids}


def handle_status_check(db: Any, p: Dict[str, Any]) -> Dict[str, Any]:
    import publication_status as ps
    return ps.check_paper(db, int(p["paper_id"]), fetch=not os.getenv("BIOSYN_OFFLINE"))


def handle_fulltext(db: Any, p: Dict[str, Any]) -> Dict[str, Any]:
    if os.getenv("BIOSYN_OFFLINE"):
        raise RuntimeError("offline - full text not reachable (will retry)")
    from retrieval.models import PaperCandidate
    from retrieval.source_manager import SourceManager
    paper = db.get_paper(int(p["paper_id"])) or {}
    if not (paper.get("pmcid") or paper.get("pmid")):
        return {"skipped": "no PMC / PubMed id"}
    cand = PaperCandidate(title=paper.get("title") or "", doi=paper.get("doi"), pmid=paper.get("pmid"),
                          pmcid=paper.get("pmcid"), abstract=paper.get("abstract"), source_db="queue")
    path = SourceManager().download_xml(cand)
    if not path:
        raise RuntimeError("no open full text yet")
    import fulltext_archive as fa
    rec = fa.store_candidate_file(cand, path, "jats_xml", "PMC / Europe PMC")
    from jats_ingest import ingest_jats
    pid = ingest_jats(db, str(path), None, source_query="ingestion queue")     # same identifiers -> same paper row
    fa.record_basis(db, int(pid or p["paper_id"]), "jats_xml", rec)
    return {"archived": (rec or {}).get("sha256")}


def handle_library_update(db: Any, p: Dict[str, Any]) -> Dict[str, Any]:
    """v26: the local library's daily job - PubMed update files (with deletions), embeddings of new / changed records,
    publication statuses pushed to the run database, then AI pre-reading of new trials when a key is configured."""
    import library_loader as ll
    res = ll.update(fetch=not os.getenv("BIOSYN_OFFLINE"), max_files=int(p.get("max_files") or 0))
    if os.getenv("BIOSYN_LIBRARY_PREREAD", "1") != "0" and any(os.getenv(k) for k in ("GROQ_API_KEY", "GEMINI_API_KEY",
                                                                                     "NVIDIA_NIM_API_KEY")):
        try:
            import library_db as L
            import preread
            with L.Library() as lib:
                res["preread"] = preread.run(lib, limit=int(p.get("preread_limit") or 50))
        except Exception as exc:
            res["preread"] = {"error": f"{type(exc).__name__}: {str(exc)[:200]}"}
    return res


HANDLERS = {"topic_update": handle_topic_update, "status_check": handle_status_check, "fulltext": handle_fulltext,
            "library_update": handle_library_update}


def schedule_library_update(q: "IngestionQueue") -> bool:
    """One library_update job per day while a library exists (BIOSYN_LIBRARY_AUTOUPDATE=0 turns it off)."""
    if os.getenv("BIOSYN_LIBRARY_AUTOUPDATE", "1") == "0":
        return False
    try:
        import library_db as L
        if not L.exists():
            return False
    except Exception:
        return False
    return q.enqueue("library_update", {}, dedup=f"library:{_now().date().isoformat()}") is not None


def run_due(db: Any, max_jobs: int = 20, handlers: Optional[Dict[str, Callable]] = None) -> Dict[str, Any]:
    q = IngestionQueue(db)
    scheduled = q.schedule_topics() + int(schedule_library_update(q))
    out = [q.run_job(j, handlers or HANDLERS) for j in q.due(max_jobs)]
    return {"scheduled_topics": scheduled, "ran": len(out), "results": out, "counts": q.counts()}


_WORKER: Optional[threading.Thread] = None
_STOP = threading.Event()


def start_worker(db_path: Optional[str] = None, every_s: float = 300) -> bool:
    """Daemon thread: run due jobs every `every_s` seconds. Idempotent. Disabled by BIOSYN_NO_INGEST_WORKER=1."""
    global _WORKER
    if os.getenv("BIOSYN_NO_INGEST_WORKER") == "1" or (_WORKER and _WORKER.is_alive()):
        return False

    def loop():
        from database import ResearchDB
        while not _STOP.is_set():
            try:
                db = ResearchDB(db_path) if db_path else ResearchDB()
                try:
                    run_due(db)
                finally:
                    db.close()
            except Exception as exc:
                print(f"  [Ingest worker] {type(exc).__name__}: {str(exc)[:120]}")
            _STOP.wait(every_s)

    _STOP.clear()                                 # a worker stopped earlier (server shut-down) can be started again
    _WORKER = threading.Thread(target=loop, name="biosyn-ingest", daemon=True)
    _WORKER.start()
    return True


def stop_worker(timeout: float = 5.0) -> None:
    _STOP.set()
    if _WORKER and _WORKER.is_alive() and _WORKER is not threading.current_thread():
        _WORKER.join(timeout)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="BioSynthetica background ingestion")
    ap.add_argument("--watch", help="add a topic re-searched every day")
    ap.add_argument("--run", action="store_true", help="run every due job now")
    ap.add_argument("--db", default=None)
    a = ap.parse_args(argv)
    from database import ResearchDB
    db = ResearchDB(a.db) if a.db else ResearchDB()
    q = IngestionQueue(db)
    if a.watch:
        print(f"watching topic #{q.watch(a.watch)}: {a.watch}")
    if a.run:
        print(json.dumps(run_due(db), indent=1, default=str)[:4000])
    print(json.dumps(q.counts()))
    db.close()
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
