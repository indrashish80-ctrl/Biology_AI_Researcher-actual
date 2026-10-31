"""
database.py
-----------
SQLite research knowledge store with thread-local connection management in WAL mode.
Provides non-blocking concurrent reads and safe serialised writes across worker threads.
"""

from __future__ import annotations

import json
import os
import sqlite3
import threading
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

try:
    from .core_utils import now_iso
except ImportError:
    from core_utils import now_iso

SCHEMA_VERSION = 1
import os as _os
# BIOSYN_DB_PATH: another knowledge-base file (the test runner uses a temporary one)
DEFAULT_DB_PATH = Path(_os.getenv("BIOSYN_DB_PATH") or Path(__file__).resolve().parent / "research_knowledge.db")

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS schema_meta (
    key TEXT PRIMARY KEY,
    value TEXT
);

CREATE TABLE IF NOT EXISTS research_questions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    question TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'open',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS papers (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    research_question_id INTEGER,
    dedup_key TEXT NOT NULL,
    title TEXT,
    authors TEXT,
    journal TEXT,
    year TEXT,
    doi TEXT,
    pmid TEXT,
    pmcid TEXT,
    abstract TEXT,
    full_text TEXT,
    is_open_access INTEGER,
    source_query TEXT,
    fetched_at TEXT NOT NULL,
    UNIQUE(dedup_key),
    FOREIGN KEY (research_question_id) REFERENCES research_questions(id)
);

CREATE TABLE IF NOT EXISTS paper_extractions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    paper_id INTEGER NOT NULL,
    organism TEXT,
    sample_size INTEGER,
    study_type TEXT,
    intervention TEXT,
    control TEXT,
    methods TEXT,
    outcomes_json TEXT,
    limitations_json TEXT,
    experimental_groups_json TEXT,
    measured_outcomes_json TEXT,
    raw_json TEXT,
    extraction_status TEXT NOT NULL,
    extracted_at TEXT NOT NULL,
    FOREIGN KEY (paper_id) REFERENCES papers(id)
);

CREATE TABLE IF NOT EXISTS datasets (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    research_question_id INTEGER,
    source TEXT NOT NULL,
    accession TEXT NOT NULL,
    title TEXT,
    description TEXT,
    organism TEXT,
    url TEXT,
    metadata_json TEXT,
    discovered_at TEXT NOT NULL,
    UNIQUE(source, accession),
    FOREIGN KEY (research_question_id) REFERENCES research_questions(id)
);

CREATE TABLE IF NOT EXISTS experiments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    research_question_id INTEGER,
    dataset_path TEXT,
    group_column TEXT,
    outcome_column TEXT,
    evidence_record_json TEXT,
    created_at TEXT NOT NULL,
    FOREIGN KEY (research_question_id) REFERENCES research_questions(id)
);

CREATE TABLE IF NOT EXISTS evidence (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    research_question_id INTEGER,
    evidence_type TEXT NOT NULL,
    statement TEXT NOT NULL,
    source_type TEXT,
    source_id INTEGER,
    confidence TEXT,
    value_json TEXT,
    created_at TEXT NOT NULL,
    FOREIGN KEY (research_question_id) REFERENCES research_questions(id)
);

CREATE TABLE IF NOT EXISTS claims (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    research_question_id INTEGER,
    paper_id INTEGER,
    statement TEXT NOT NULL,
    subject TEXT,
    population TEXT,
    organism TEXT,
    intervention TEXT,
    dose TEXT,
    duration TEXT,
    measurement TEXT,
    statistical_method TEXT,
    effect_direction TEXT,
    effect_size REAL,
    p_value REAL,
    sample_size INTEGER,
    confidence_interval TEXT,
    conditions_json TEXT,
    created_at TEXT NOT NULL,
    FOREIGN KEY (research_question_id) REFERENCES research_questions(id),
    FOREIGN KEY (paper_id) REFERENCES papers(id)
);

CREATE TABLE IF NOT EXISTS contradictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    research_question_id INTEGER,
    claim_a_id INTEGER NOT NULL,
    claim_b_id INTEGER NOT NULL,
    verdict TEXT NOT NULL,
    comparison_json TEXT,
    reasoning TEXT,
    created_at TEXT NOT NULL,
    FOREIGN KEY (claim_a_id) REFERENCES claims(id),
    FOREIGN KEY (claim_b_id) REFERENCES claims(id)
);

CREATE TABLE IF NOT EXISTS hypotheses (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    research_question_id INTEGER,
    statement TEXT NOT NULL,
    mechanism TEXT,
    predictions_json TEXT,
    required_measurement TEXT,
    expected_result TEXT,
    falsification_conditions_json TEXT,
    supporting_evidence_json TEXT,
    competing_explanations_json TEXT,
    next_experiment TEXT,
    status TEXT NOT NULL DEFAULT 'proposed',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    FOREIGN KEY (research_question_id) REFERENCES research_questions(id)
);

CREATE TABLE IF NOT EXISTS experiment_designs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    hypothesis_id INTEGER NOT NULL,
    design_json TEXT NOT NULL,
    power_notes TEXT,
    confounder_notes TEXT,
    created_at TEXT NOT NULL,
    FOREIGN KEY (hypothesis_id) REFERENCES hypotheses(id)
);

CREATE TABLE IF NOT EXISTS research_gaps (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    research_question_id INTEGER,
    description TEXT NOT NULL,
    category TEXT,
    supporting_observation TEXT,
    created_at TEXT NOT NULL,
    FOREIGN KEY (research_question_id) REFERENCES research_questions(id)
);

CREATE TABLE IF NOT EXISTS provenance (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    entity_type TEXT NOT NULL,
    entity_id INTEGER,
    provenance_type TEXT NOT NULL,
    source_ref TEXT,
    details_json TEXT,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS audit_flags (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    research_question_id INTEGER,
    entity_type TEXT,
    entity_id INTEGER,
    claim_text TEXT,
    evidence_reference TEXT,
    issue TEXT NOT NULL,
    severity TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS processed_items (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    item_type TEXT NOT NULL,
    external_id TEXT NOT NULL,
    research_question_id INTEGER,
    processed_at TEXT NOT NULL,
    UNIQUE(item_type, external_id)
);

CREATE TABLE IF NOT EXISTS provider_health_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    provider TEXT NOT NULL,
    event TEXT NOT NULL,
    latency_ms REAL,
    error_text TEXT,
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_papers_rq ON papers(research_question_id);
CREATE INDEX IF NOT EXISTS idx_claims_rq ON claims(research_question_id);
CREATE INDEX IF NOT EXISTS idx_evidence_rq ON evidence(research_question_id);
CREATE INDEX IF NOT EXISTS idx_hypotheses_rq ON hypotheses(research_question_id);
CREATE INDEX IF NOT EXISTS idx_provenance_entity ON provenance(entity_type, entity_id);
"""


class _Result:
    def __init__(self, cur: sqlite3.Cursor):
        self.description = cur.description
        self.lastrowid = cur.lastrowid
        self.rowcount = cur.rowcount
        self._rows = cur.fetchall() if cur.description is not None else []
        self._i = 0

    def fetchone(self):
        if self._i >= len(self._rows):
            return None
        row = self._rows[self._i]
        self._i += 1
        return row

    def fetchall(self):
        out = self._rows[self._i:]
        self._i = len(self._rows)
        return out

    def fetchmany(self, size: int = 1):
        out = self._rows[self._i:self._i + size]
        self._i += len(out)
        return out

    def __iter__(self):
        return iter(self.fetchall())


class _LockedConnection:
    """Thread-safe proxy that serializes access to thread-local or shared connections."""
    def __init__(self, conn_getter: Any, lock: threading.RLock):
        self._conn_getter = conn_getter
        self._lock = lock

    @property
    def _conn(self) -> sqlite3.Connection:
        if callable(self._conn_getter):
            return self._conn_getter()
        return self._conn_getter

    def execute(self, sql: str, params: Any = ()) -> _Result:
        with self._lock:
            cur = self._conn.execute(sql, params)
            return _Result(cur)

    def executemany(self, sql: str, seq: Any) -> _Result:
        with self._lock:
            cur = self._conn.executemany(sql, seq)
            return _Result(cur)

    def executescript(self, script: str) -> _Result:
        with self._lock:
            cur = self._conn.executescript(script)
            return _Result(cur)

    def commit(self) -> None:
        with self._lock:
            self._conn.commit()

    def rollback(self) -> None:
        with self._lock:
            self._conn.rollback()

    def close(self) -> None:
        pass


class ResearchDB:
    def __init__(self, db_path: Optional[str] = None):
        self.db_path = str(db_path or DEFAULT_DB_PATH)
        self._lock = threading.RLock()
        self._local = threading.local()
        self._all_conns: List[sqlite3.Connection] = []

        # Initialize schema
        self._conn = _LockedConnection(self._get_thread_conn, self._lock)
        self._init_schema()

    def _get_thread_conn(self) -> sqlite3.Connection:
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(self.db_path, check_same_thread=False, timeout=45.0)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode = WAL;")
            conn.execute("PRAGMA synchronous = NORMAL;")
            conn.execute("PRAGMA busy_timeout = 45000;")
            conn.execute("PRAGMA foreign_keys = ON;")
            self._local.conn = conn
            with self._lock:
                self._all_conns.append(conn)
        return conn

    def commit(self) -> None:
        """Commit THIS thread's connection (v26.2: reader threads publish their writes at once)."""
        self._conn.commit()

    def rollback(self) -> None:
        """Roll back THIS thread's open transaction (v26.2: a failed paper never leaves the database locked)."""
        self._conn.rollback()

    def _init_schema(self) -> None:
        with self._lock:
            conn = self._get_thread_conn()
            conn.executescript(SCHEMA_SQL)
            conn.executescript("""
            CREATE TABLE IF NOT EXISTS paper_sections (
                id INTEGER PRIMARY KEY AUTOINCREMENT, paper_id INTEGER NOT NULL,
                section_type TEXT NOT NULL, title TEXT, text TEXT NOT NULL,
                created_at TEXT NOT NULL, FOREIGN KEY (paper_id) REFERENCES papers(id)
            );
            CREATE INDEX IF NOT EXISTS idx_sections_paper ON paper_sections(paper_id);
            CREATE TABLE IF NOT EXISTS paper_tables (
                id INTEGER PRIMARY KEY AUTOINCREMENT, paper_id INTEGER NOT NULL,
                label TEXT, caption TEXT, rows_json TEXT, created_at TEXT NOT NULL,
                FOREIGN KEY (paper_id) REFERENCES papers(id)
            );
            CREATE INDEX IF NOT EXISTS idx_tables_paper ON paper_tables(paper_id);
            CREATE TABLE IF NOT EXISTS claim_groups (
                id INTEGER PRIMARY KEY AUTOINCREMENT, research_question_id INTEGER,
                group_key TEXT NOT NULL, intervention TEXT, outcome TEXT,
                member_ids_json TEXT, summary_json TEXT, explanations_json TEXT, created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS biological_entities (
                id INTEGER PRIMARY KEY AUTOINCREMENT, entity_type TEXT NOT NULL, source_db TEXT NOT NULL,
                accession TEXT NOT NULL, symbol TEXT, name TEXT, organism TEXT, metadata_json TEXT,
                UNIQUE(entity_type, source_db, accession)
            );
            CREATE TABLE IF NOT EXISTS entity_relations (
                id INTEGER PRIMARY KEY AUTOINCREMENT, from_entity_id INTEGER NOT NULL, relation TEXT NOT NULL,
                to_entity_id INTEGER NOT NULL, source_db TEXT, created_at TEXT NOT NULL,
                UNIQUE(from_entity_id, relation, to_entity_id, source_db)
            );
            CREATE TABLE IF NOT EXISTS entity_mentions (
                id INTEGER PRIMARY KEY AUTOINCREMENT, entity_id INTEGER NOT NULL, source_type TEXT NOT NULL,
                source_id INTEGER NOT NULL, mention TEXT, created_at TEXT NOT NULL,
                UNIQUE(entity_id, source_type, source_id, mention)
            );
            """)
            for col, typ in (("evidence_origin", "TEXT"), ("grounded", "INTEGER"), ("grounding_score", "REAL"),
                             ("quote", "TEXT"), ("flags_json", "TEXT"), ("nonsignificant_direction", "INTEGER"),
                             ("study_design", "TEXT"), ("source_basis", "TEXT"), ("extraction_confidence", "TEXT"),
                             ("extraction_json", "TEXT")):
                try:
                    conn.execute(f"ALTER TABLE claims ADD COLUMN {col} {typ}")
                except sqlite3.OperationalError:
                    pass
            conn.execute(
                "INSERT OR REPLACE INTO schema_meta (key, value) VALUES ('version', ?)",
                (str(SCHEMA_VERSION),),
            )
            conn.commit()

    def close(self) -> None:
        with self._lock:
            for conn in self._all_conns:
                try:
                    conn.close()
                except Exception:
                    pass
            self._all_conns.clear()
            self._local.conn = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def _row(self, row: Optional[sqlite3.Row]) -> Optional[Dict[str, Any]]:
        return dict(row) if row is not None else None

    def _rows(self, rows: Iterable[sqlite3.Row]) -> List[Dict[str, Any]]:
        return [dict(r) for r in rows]

    def _insert(self, table: str, fields: Dict[str, Any]) -> int:
        cols = ", ".join(fields.keys())
        placeholders = ", ".join("?" for _ in fields)
        with self._lock:
            cursor = self._conn.execute(
                f"INSERT INTO {table} ({cols}) VALUES ({placeholders})",
                tuple(fields.values()),
            )
            self._conn.commit()
            return cursor.lastrowid

    @staticmethod
    def _j(value: Any) -> Optional[str]:
        if value is None:
            return None
        return json.dumps(value, ensure_ascii=False)

    @staticmethod
    def _dj(value: Optional[str]) -> Any:
        if value in (None, ""):
            return None
        try:
            return json.loads(value)
        except (json.JSONDecodeError, TypeError):
            return None

    # --- RESEARCH QUESTIONS ---

    def create_research_question(self, question: str) -> int:
        existing = self._row(
            self._conn.execute(
                "SELECT * FROM research_questions WHERE question = ? ORDER BY id DESC LIMIT 1",
                (question,),
            ).fetchone()
        )
        if existing:
            return existing["id"]

        ts = now_iso()
        return self._insert(
            "research_questions",
            {"question": question, "status": "open", "created_at": ts, "updated_at": ts},
        )

    def get_research_question(self, rq_id: int) -> Optional[Dict[str, Any]]:
        return self._row(
            self._conn.execute("SELECT * FROM research_questions WHERE id = ?", (rq_id,)).fetchone()
        )

    # --- PAPERS ---

    def upsert_paper(self, research_question_id: Optional[int], paper: Dict[str, Any], source_query: str = "") -> int:
        dedup_key = (
            paper.get("doi")
            or paper.get("pmid")
            or paper.get("pmcid")
            or f"title::{(paper.get('title') or '').strip().lower()}"
        )
        existing = self._row(
            self._conn.execute("SELECT id FROM papers WHERE dedup_key = ?", (dedup_key,)).fetchone()
        )
        if existing:
            return existing["id"]

        return self._insert(
            "papers",
            {
                "research_question_id": research_question_id,
                "dedup_key": dedup_key,
                "title": paper.get("title"),
                "authors": paper.get("authors"),
                "journal": paper.get("journal"),
                "year": str(paper.get("year")) if paper.get("year") is not None else None,
                "doi": paper.get("doi"),
                "pmid": paper.get("pmid"),
                "pmcid": paper.get("pmcid"),
                "abstract": paper.get("abstract"),
                "full_text": paper.get("full_text"),
                "is_open_access": 1 if paper.get("is_open_access") else 0,
                "source_query": source_query,
                "fetched_at": now_iso(),
            },
        )

    def get_paper(self, paper_id: int) -> Optional[Dict[str, Any]]:
        return self._row(self._conn.execute("SELECT * FROM papers WHERE id = ?", (paper_id,)).fetchone())

    def list_papers(self, research_question_id: Optional[int] = None) -> List[Dict[str, Any]]:
        if research_question_id is None:
            return self._rows(self._conn.execute("SELECT * FROM papers").fetchall())
        return self._rows(
            self._conn.execute(
                "SELECT * FROM papers WHERE research_question_id = ?", (research_question_id,)
            ).fetchall()
        )

    def list_papers_summary(self, research_question_id: Optional[int] = None) -> List[Dict[str, Any]]:
        cols = "id, research_question_id, dedup_key, title, journal, year, doi, pmid, pmcid, is_open_access"
        if research_question_id is None:
            return self._rows(self._conn.execute(f"SELECT {cols} FROM papers").fetchall())
        return self._rows(
            self._conn.execute(
                f"SELECT {cols} FROM papers WHERE research_question_id = ?", (research_question_id,)
            ).fetchall()
        )

    def save_paper_extraction(self, paper_id: int, extraction: Dict[str, Any], status: str = "validated") -> int:
        study = extraction.get("study", {}) if isinstance(extraction.get("study"), dict) else {}
        return self._insert(
            "paper_extractions",
            {
                "paper_id": paper_id,
                "organism": study.get("organism") or extraction.get("organism"),
                "sample_size": study.get("sample_size") or extraction.get("sample_size"),
                "study_type": study.get("study_type") or extraction.get("study_type"),
                "intervention": extraction.get("intervention"),
                "control": extraction.get("control"),
                "methods": extraction.get("methods"),
                "outcomes_json": self._j(extraction.get("outcomes")),
                "limitations_json": self._j(extraction.get("limitations")),
                "experimental_groups_json": self._j(extraction.get("experimental_groups")),
                "measured_outcomes_json": self._j(extraction.get("measured_outcomes")),
                "raw_json": self._j(extraction),
                "extraction_status": status,
                "extracted_at": now_iso(),
            },
        )

    def get_paper_extraction(self, paper_id: int) -> Optional[Dict[str, Any]]:
        row = self._row(
            self._conn.execute(
                "SELECT * FROM paper_extractions WHERE paper_id = ? ORDER BY id DESC LIMIT 1", (paper_id,)
            ).fetchone()
        )
        if not row:
            return None
        for key in ("outcomes_json", "limitations_json", "experimental_groups_json", "measured_outcomes_json", "raw_json"):
            row[key.replace("_json", "")] = self._dj(row[key])
        return row

    def save_paper_section(self, paper_id: int, section_type: str, title: str, text: str) -> int:
        return self._insert("paper_sections", {"paper_id": paper_id, "section_type": section_type,
            "title": title, "text": text, "created_at": now_iso()})

    def clear_paper_sections(self, paper_id: int) -> None:
        self._conn.execute("DELETE FROM paper_sections WHERE paper_id = ?", (paper_id,)); self._conn.commit()

    def get_sections(self, paper_id: int, include_meta: bool = False) -> List[Dict[str, Any]]:
        rows = self._rows(self._conn.execute("SELECT * FROM paper_sections WHERE paper_id = ? ORDER BY id", (paper_id,)).fetchall())
        return rows if include_meta else [r for r in rows if r.get("section_type") != "article_meta"]

    def save_paper_table(self, paper_id: int, label: str, caption: str, rows: List[List[str]]) -> int:
        return self._insert("paper_tables", {"paper_id": paper_id, "label": label, "caption": caption,
            "rows_json": self._j(rows), "created_at": now_iso()})

    def clear_paper_tables(self, paper_id: int) -> None:
        self._conn.execute("DELETE FROM paper_tables WHERE paper_id = ?", (paper_id,)); self._conn.commit()

    def get_tables(self, paper_id: int) -> List[Dict[str, Any]]:
        rows = self._rows(self._conn.execute("SELECT * FROM paper_tables WHERE paper_id = ? ORDER BY id", (paper_id,)).fetchall())
        for r in rows: r["rows"] = self._dj(r.pop("rows_json")) or []
        return rows

    def save_paper_extraction_ext(self, paper_id: int, extraction: Dict[str, Any], status: str = "validated", basis: str = "full_text") -> int:
        eid = self.save_paper_extraction(paper_id, extraction, status=status)
        self._conn.execute("UPDATE paper_extractions SET raw_json = ? WHERE id = ?", (self._j({**extraction, "basis": basis}), eid)); self._conn.commit()
        return eid

    def get_extraction_raw(self, paper_id: int) -> Dict[str, Any]:
        row = self.get_paper_extraction(paper_id)
        return (row or {}).get("raw") or {}

    # --- EVIDENCE ---

    def add_evidence(self, research_question_id: Optional[int], evidence_type: str, statement: str,
                      source_type: Optional[str] = None, source_id: Optional[int] = None,
                      confidence: Optional[str] = None, value: Any = None) -> int:
        eid = self._insert(
            "evidence",
            {
                "research_question_id": research_question_id,
                "evidence_type": evidence_type,
                "statement": statement,
                "source_type": source_type,
                "source_id": source_id,
                "confidence": confidence,
                "value_json": self._j(value),
                "created_at": now_iso(),
            },
        )
        self.add_provenance("evidence", eid, evidence_type, source_ref=f"{source_type}:{source_id}" if source_type else None)
        return eid

    def clear_experiment_evidence(self, research_question_id: Optional[int]) -> None:
        with self._lock:
            self._conn.execute("DELETE FROM evidence WHERE research_question_id IS ? AND source_type='experiment'",
                               (research_question_id,))
            self._conn.commit()

    def list_evidence(self, research_question_id: Optional[int] = None) -> List[Dict[str, Any]]:
        rows = (
            self._rows(self._conn.execute("SELECT * FROM evidence").fetchall())
            if research_question_id is None
            else self._rows(
                self._conn.execute(
                    "SELECT * FROM evidence WHERE research_question_id = ?", (research_question_id,)
                ).fetchall()
            )
        )
        for row in rows:
            row["value"] = self._dj(row["value_json"])
        return rows

    # --- CLAIMS ---

    def add_claim(self, research_question_id: Optional[int], paper_id: Optional[int], claim: Dict[str, Any]) -> int:
        claim = dict(claim)
        claim.setdefault("effect_direction", claim.get("direction"))
        claim.setdefault("measurement", claim.get("outcome"))
        cid = self._insert(
            "claims",
            {
                "research_question_id": research_question_id,
                "paper_id": paper_id,
                "statement": claim.get("statement", ""),
                "subject": claim.get("subject"),
                "population": claim.get("population"),
                "organism": claim.get("organism"),
                "intervention": claim.get("intervention"),
                "dose": claim.get("dose"),
                "duration": claim.get("duration"),
                "measurement": claim.get("measurement"),
                "statistical_method": claim.get("statistical_method"),
                "effect_direction": claim.get("effect_direction"),
                "effect_size": claim.get("effect_size"),
                "p_value": claim.get("p_value"),
                "sample_size": claim.get("sample_size"),
                "confidence_interval": self._j(claim.get("confidence_interval")),
                "conditions_json": self._j(claim.get("conditions")),
                "created_at": now_iso(),
            },
        )
        source_ref = f"paper:{paper_id}" if paper_id else "unknown"
        self.add_provenance("claim", cid, "REPORTED_BY_PAPER" if paper_id else "AI_INFERENCE", source_ref=source_ref)
        return cid

    def add_claim_ext(self, research_question_id: Optional[int], paper_id: Optional[int], claim: Dict[str, Any]) -> int:
        cid = self.add_claim(research_question_id, paper_id, claim)
        self._conn.execute("""UPDATE claims SET evidence_origin=?, grounded=?, grounding_score=?, quote=?, flags_json=?,
            nonsignificant_direction=?, study_design=? WHERE id=?""",
            (claim.get("evidence_origin"), int(bool(claim.get("grounded"))), claim.get("grounding_score"),
             claim.get("quote"), self._j(claim.get("flags")), int(bool(claim.get("nonsignificant_direction"))),
             claim.get("study_design"), cid))
        conf = claim.get("extraction_confidence")
        if conf or claim.get("source_basis") or claim.get("second_reader"):
            # v25: how the finding was read (text basis, verbatim-quote mode, second reader) - shown next to the quote
            self._conn.execute("UPDATE claims SET source_basis=?, extraction_confidence=?, extraction_json=? WHERE id=?",
                               (claim.get("source_basis"), (conf or {}).get("label") if isinstance(conf, dict) else conf,
                                self._j({"confidence": conf, "quote_mode": claim.get("quote_mode"),
                                         "quote_span": claim.get("quote_span"),
                                         "quote_as_extracted": claim.get("quote_as_extracted"),
                                         "second_reader": claim.get("second_reader"),
                                         "second_reader_reading": claim.get("second_reader_reading")}), cid))
        self._conn.commit()
        return cid

    def delete_claims_for_paper(self, paper_id: int, research_question_id: Optional[int] = None) -> None:
        """Remove a paper's claims before it is read again. v26.2: the contradictions that point at those claims are
        removed FIRST (they are recomputed in the contradictions stage) - deleting the claims alone violated the
        foreign key, the paper was skipped, and the open transaction locked the database for the next stage."""
        where, args = ("paper_id = ?", (paper_id,)) if research_question_id is None else \
            ("paper_id = ? AND research_question_id = ?", (paper_id, research_question_id))
        with self._lock:
            try:
                ids = [r[0] for r in self._conn.execute(f"SELECT id FROM claims WHERE {where}", args).fetchall()]
                for i in range(0, len(ids), 500):
                    chunk = ids[i:i + 500]
                    marks = ",".join("?" * len(chunk))
                    self._conn.execute(f"DELETE FROM contradictions WHERE claim_a_id IN ({marks}) OR claim_b_id IN ({marks})",
                                       (*chunk, *chunk))
                self._conn.execute(f"DELETE FROM claims WHERE {where}", args)
                self._conn.commit()
            except Exception:
                self._conn.rollback()
                raise

    def claims_for_paper(self, paper_id: int) -> List[Dict[str, Any]]:
        return self._claims_query("SELECT * FROM claims WHERE paper_id = ? ORDER BY id", (paper_id,))

    def claims_with_papers(self, research_question_id: Optional[int] = None) -> List[Dict[str, Any]]:
        sql = "SELECT c.*, p.title AS paper_title, p.pmid, p.pmcid, p.doi FROM claims c LEFT JOIN papers p ON p.id=c.paper_id"
        args = ()
        if research_question_id is not None:
            sql += " WHERE c.research_question_id = ?"
            args = (research_question_id,)
        sql += " ORDER BY c.id"
        return self._claims_query(sql, args)

    paper_scope: Optional[set] = None

    def set_paper_scope(self, paper_ids: Optional[Iterable[int]]) -> None:
        self.paper_scope = set(paper_ids) if paper_ids is not None else None

    def _in_scope(self, c: Dict[str, Any]) -> bool:
        return self.paper_scope is None or c.get("paper_id") in self.paper_scope

    def verified_claims(self, research_question_id: Optional[int] = None) -> List[Dict[str, Any]]:
        from semantic_audit import ensure_audited
        ensure_audited(self, research_question_id)
        return [c for c in self.claims_with_papers(research_question_id)
                if c.get("verification_status") == "verified" and self._in_scope(c)]

    def primary_claims(self, research_question_id: Optional[int] = None) -> List[Dict[str, Any]]:
        return [c for c in self.verified_claims(research_question_id)
                if ((c.get("semantic_audit") or {}).get("evidence_role")) != "secondary_report"]

    def quarantined_claims(self, research_question_id: Optional[int] = None) -> List[Dict[str, Any]]:
        from semantic_audit import ensure_audited
        ensure_audited(self, research_question_id)
        return [c for c in self.claims_with_papers(research_question_id)
                if c.get("verification_status") == "quarantined" and self._in_scope(c)]

    def _claims_query(self, sql: str, args: tuple) -> List[Dict[str, Any]]:
        rows = self._rows(self._conn.execute(sql, args).fetchall())
        for r in rows:
            r["conditions"] = self._dj(r.get("conditions_json"))
            r["confidence_interval"] = self._dj(r.get("confidence_interval"))
            r["flags"] = self._dj(r.get("flags_json")) or []
            r["grounded"] = bool(r.get("grounded"))
            r["semantic_audit"] = self._dj(r.get("semantic_audit_json")) if r.get("semantic_audit_json") else None
            if r.get("extraction_json"):
                r["extraction"] = self._dj(r.get("extraction_json"))
        return rows

    # --- CONTRADICTIONS ---

    def add_contradiction(self, research_question_id: Optional[int], claim_a_id: int, claim_b_id: int,
                           verdict: str, comparison: Dict[str, Any], reasoning: str = "") -> int:
        return self._insert(
            "contradictions",
            {
                "research_question_id": research_question_id,
                "claim_a_id": claim_a_id,
                "claim_b_id": claim_b_id,
                "verdict": verdict,
                "comparison_json": self._j(comparison),
                "reasoning": reasoning,
                "created_at": now_iso(),
            },
        )

    def current_contradictions(self, research_question_id: Optional[int] = None) -> List[Dict[str, Any]]:
        ok = {c["id"] for c in self.verified_claims(research_question_id)}
        return [c for c in self.list_contradictions(research_question_id)
                if c.get("claim_a_id") in ok and c.get("claim_b_id") in ok]

    def list_contradictions(self, research_question_id: Optional[int] = None) -> List[Dict[str, Any]]:
        rows = (
            self._rows(self._conn.execute("SELECT * FROM contradictions").fetchall())
            if research_question_id is None
            else self._rows(
                self._conn.execute(
                    "SELECT * FROM contradictions WHERE research_question_id = ?", (research_question_id,)
                ).fetchall()
            )
        )
        for row in rows:
            row["comparison"] = self._dj(row["comparison_json"])
            row["kind"] = (row["comparison"] or {}).get("kind", "same_system") if isinstance(row["comparison"], dict) else "same_system"
        return rows

    # --- HYPOTHESES & EXPERIMENT DESIGNS ---

    def _ensure_hypothesis_columns(self) -> None:
        for col, typ in (("rank_order", "INTEGER"), ("label", "TEXT"), ("support_json", "TEXT"), ("merged_into", "INTEGER")):
            try:
                self._conn.execute(f"ALTER TABLE hypotheses ADD COLUMN {col} {typ}")
            except sqlite3.OperationalError:
                pass

    def _label_parameters(self, research_question_id: Optional[int], value: Any) -> Any:
        from reasoning_guards import label_ai_parameters, sourced_numbers
        try:
            sourced = sourced_numbers(self.verified_claims(research_question_id)) if research_question_id is not None else {}
        except Exception:
            sourced = {}
        if isinstance(value, str) and value.strip().startswith("{"):
            try:
                return json.dumps(label_ai_parameters(json.loads(value), sourced), ensure_ascii=False)
            except ValueError:
                pass
        return label_ai_parameters(value, sourced)

    def add_hypothesis(self, research_question_id: Optional[int], hypothesis: Dict[str, Any]) -> int:
        ts = now_iso()
        hypothesis = dict(hypothesis)
        for k in ("statement", "mechanism", "predictions", "falsification_conditions", "next_experiment",
                  "required_measurement", "expected_result"):
            if hypothesis.get(k):
                hypothesis[k] = self._label_parameters(research_question_id, hypothesis[k])
        if isinstance(hypothesis.get("next_experiment"), (dict, list)):
            hypothesis["next_experiment"] = json.dumps(hypothesis["next_experiment"], ensure_ascii=False)
        hid = self._insert(
            "hypotheses",
            {
                "research_question_id": research_question_id,
                "statement": hypothesis.get("statement", ""),
                "mechanism": hypothesis.get("mechanism"),
                "predictions_json": self._j(hypothesis.get("predictions")),
                "required_measurement": hypothesis.get("required_measurement"),
                "expected_result": hypothesis.get("expected_result"),
                "falsification_conditions_json": self._j(hypothesis.get("falsification_conditions")),
                "supporting_evidence_json": self._j(hypothesis.get("supporting_evidence")),
                "competing_explanations_json": self._j(hypothesis.get("competing_explanations")),
                "next_experiment": hypothesis.get("next_experiment"),
                "status": hypothesis.get("status", "proposed"),
                "created_at": ts,
                "updated_at": ts,
            },
        )
        self.add_provenance("hypothesis", hid, "HYPOTHESIS", source_ref="ai_inference")
        return hid

    def list_hypotheses(self, research_question_id: Optional[int] = None, include_merged: bool = False) -> List[Dict[str, Any]]:
        self._ensure_hypothesis_columns()
        order = " ORDER BY CASE WHEN rank_order IS NULL THEN 1 ELSE 0 END, rank_order, id"
        rows = (
            self._rows(self._conn.execute("SELECT * FROM hypotheses" + order).fetchall())
            if research_question_id is None
            else self._rows(
                self._conn.execute(
                    "SELECT * FROM hypotheses WHERE research_question_id = ?" + order, (research_question_id,)
                ).fetchall()
            )
        )
        for row in rows:
            for field in ("predictions", "falsification_conditions", "supporting_evidence", "competing_explanations"):
                row[field] = self._dj(row.get(f"{field}_json"))
            row["support"] = self._dj(row.get("support_json")) if row.get("support_json") else None
        return [r for r in rows if include_merged or r.get("status") not in ("merged", "superseded")]

    def retire_hypotheses(self, research_question_id: Optional[int]) -> None:
        self._ensure_hypothesis_columns()
        with self._lock:
            self._conn.execute("UPDATE hypotheses SET status='superseded', updated_at=? WHERE research_question_id IS ? "
                               "AND COALESCE(status, '') NOT IN ('merged', 'superseded')", (now_iso(), research_question_id))
            self._conn.commit()

    def set_hypothesis_rank(self, hypothesis_id: int, rank: int, label: str, support: Any) -> None:
        self._ensure_hypothesis_columns()
        with self._lock:
            self._conn.execute("UPDATE hypotheses SET rank_order=?, label=?, support_json=?, updated_at=? WHERE id=?",
                               (rank, label, self._j(support), now_iso(), hypothesis_id))
            self._conn.commit()

    def mark_hypothesis_merged(self, hypothesis_id: int, into_id: int) -> None:
        self._ensure_hypothesis_columns()
        with self._lock:
            self._conn.execute("UPDATE hypotheses SET status='merged', merged_into=?, updated_at=? WHERE id=?",
                               (into_id, now_iso(), hypothesis_id))
            self._conn.commit()

    def add_research_gap(self, research_question_id: Optional[int], description: str,
                          category: str = "", supporting_observation: str = "") -> int:
        return self._insert(
            "research_gaps",
            {
                "research_question_id": research_question_id,
                "description": description,
                "category": category,
                "supporting_observation": supporting_observation,
                "created_at": now_iso(),
            },
        )

    def clear_research_gaps(self, research_question_id: Optional[int]) -> None:
        with self._lock:
            self._conn.execute("DELETE FROM research_gaps WHERE research_question_id IS ?", (research_question_id,))
            self._conn.commit()

    def list_research_gaps(self, research_question_id: Optional[int] = None) -> List[Dict[str, Any]]:
        if research_question_id is None:
            rows = self._rows(self._conn.execute("SELECT * FROM research_gaps").fetchall())
        else:
            rows = self._rows(self._conn.execute(
                "SELECT * FROM research_gaps WHERE research_question_id = ? ORDER BY id", (research_question_id,)).fetchall())
        out, seen = [], set()
        for r in rows:
            k = (r.get("research_question_id"), " ".join((r.get("description") or "").lower().split()))
            if k in seen:
                continue
            seen.add(k)
            out.append(r)
        return out

    def update_hypothesis_status(self, hypothesis_id: int, status: str) -> None:
        self._conn.execute("UPDATE hypotheses SET status = ?, updated_at = ? WHERE id = ?",
                           (status, now_iso(), hypothesis_id))
        self._conn.commit()

    def add_experiment_design(self, hypothesis_id: int, design: Any, power_notes: Optional[str] = None,
                              confounder_notes: Optional[str] = None) -> int:
        row = self._conn.execute("SELECT research_question_id FROM hypotheses WHERE id=?", (hypothesis_id,)).fetchone()
        rq = row[0] if row else None
        design = self._label_parameters(rq, design)
        power_notes = self._label_parameters(rq, power_notes) if power_notes else power_notes
        confounder_notes = self._label_parameters(rq, confounder_notes) if confounder_notes else confounder_notes
        return self._insert(
            "experiment_designs",
            {
                "hypothesis_id": hypothesis_id,
                "design_json": self._j(design) or "{}",
                "power_notes": power_notes,
                "confounder_notes": confounder_notes,
                "created_at": now_iso(),
            },
        )

    # --- CONTRADICTION GROUPS ---

    def clear_contradictions(self, research_question_id: Optional[int]) -> None:
        self._conn.execute("DELETE FROM contradictions WHERE research_question_id IS ?", (research_question_id,))
        self._conn.execute("DELETE FROM claim_groups WHERE research_question_id IS ?", (research_question_id,))
        self._conn.commit()

    def save_claim_group(self, research_question_id: Optional[int], group_key: str, intervention: Optional[str],
                         outcome: Optional[str], member_ids: List[int], summary: Any, explanations: Any) -> int:
        return self._insert(
            "claim_groups",
            {
                "research_question_id": research_question_id,
                "group_key": group_key,
                "intervention": intervention,
                "outcome": outcome,
                "member_ids_json": self._j(member_ids),
                "summary_json": self._j(summary),
                "explanations_json": self._j(explanations),
                "created_at": now_iso(),
            },
        )

    # --- DATASETS ---

    def upsert_dataset(self, research_question_id: Optional[int], source: str, accession: str, title: str = "",
                       description: str = "", organism: str = "", url: str = "", metadata: Any = None) -> int:
        row = self._row(self._conn.execute("SELECT id FROM datasets WHERE source = ? AND accession = ?",
                                           (source, accession)).fetchone())
        if row:
            self._conn.execute(
                "UPDATE datasets SET title=?, description=?, organism=?, url=?, metadata_json=? WHERE id=?",
                (title, description, organism, url, self._j(metadata), row["id"]))
            self._conn.commit()
            return row["id"]
        return self._insert(
            "datasets",
            {
                "research_question_id": research_question_id, "source": source, "accession": accession,
                "title": title, "description": description, "organism": organism, "url": url,
                "metadata_json": self._j(metadata), "discovered_at": now_iso(),
            },
        )

    # --- PROVIDER HEALTH ---

    def log_provider_event(self, provider: str, event: str, latency_ms: Optional[float] = None,
                           error_text: Optional[str] = None) -> int:
        return self._insert("provider_health_log", {"provider": provider, "event": event, "latency_ms": latency_ms,
                                                    "error_text": (error_text or "")[:500] or None,
                                                    "created_at": now_iso()})

    # --- BIOLOGICAL ENTITIES ---

    def upsert_entity(self, entity_type: str, source_db: str, accession: str, symbol: str = "", name: str = "",
                      organism: str = "", metadata: Any = None) -> int:
        row = self._row(self._conn.execute("SELECT id FROM biological_entities WHERE entity_type=? AND source_db=? AND accession=?",
            (entity_type, source_db, accession)).fetchone())
        if row:
            self._conn.execute("UPDATE biological_entities SET symbol=?, name=?, organism=?, metadata_json=? WHERE id=?",
                (symbol, name, organism, self._j(metadata), row["id"]))
            self._conn.commit()
            return row["id"]
        return self._insert("biological_entities", {"entity_type": entity_type, "source_db": source_db, "accession": accession,
            "symbol": symbol, "name": name, "organism": organism, "metadata_json": self._j(metadata)})

    def link_entity(self, entity_id: int, source_type: str, source_id: int, mention: str = "") -> int:
        row = self._row(self._conn.execute("SELECT id FROM entity_mentions WHERE entity_id=? AND source_type=? AND source_id=? AND mention=?",
            (entity_id, source_type, source_id, mention)).fetchone())
        if row: return row["id"]
        return self._insert("entity_mentions", {"entity_id": entity_id, "source_type": source_type, "source_id": source_id,
            "mention": mention, "created_at": now_iso()})

    # --- PROVENANCE, AUDIT, MEMORY ---

    def add_provenance(self, entity_type: str, entity_id: int, provenance_type: str,
                        source_ref: Optional[str] = None, details: Any = None) -> int:
        return self._insert(
            "provenance",
            {
                "entity_type": entity_type,
                "entity_id": entity_id,
                "provenance_type": provenance_type,
                "source_ref": source_ref,
                "details_json": self._j(details),
                "created_at": now_iso(),
            },
        )

    def add_audit_flag(self, research_question_id: Optional[int], entity_type: str, entity_id: Optional[int],
                        claim_text: str, evidence_reference: str, issue: str, severity: str = "HIGH") -> int:
        return self._insert(
            "audit_flags",
            {
                "research_question_id": research_question_id,
                "entity_type": entity_type,
                "entity_id": entity_id,
                "claim_text": claim_text,
                "evidence_reference": evidence_reference,
                "issue": issue,
                "severity": severity,
                "created_at": now_iso(),
            },
        )

    def list_audit_flags(self, research_question_id: Optional[int] = None) -> List[Dict[str, Any]]:
        if research_question_id is None:
            return self._rows(self._conn.execute("SELECT * FROM audit_flags").fetchall())
        return self._rows(
            self._conn.execute(
                "SELECT * FROM audit_flags WHERE research_question_id = ?", (research_question_id,)
            ).fetchall()
        )

    def already_processed(self, item_type: str, external_id: str) -> bool:
        if not external_id:
            return False
        row = self._conn.execute(
            "SELECT id FROM processed_items WHERE item_type = ? AND external_id = ?",
            (item_type, external_id),
        ).fetchone()
        return row is not None

    def mark_processed(self, item_type: str, external_id: str, research_question_id: Optional[int] = None) -> None:
        if not external_id:
            return
        try:
            self._conn.execute(
                "INSERT OR IGNORE INTO processed_items (item_type, external_id, research_question_id, processed_at) "
                "VALUES (?, ?, ?, ?)",
                (item_type, external_id, research_question_id, now_iso()),
            )
            self._conn.commit()
        except sqlite3.IntegrityError:
            pass

    def research_question_summary(self, rq_id: int) -> Dict[str, Any]:
        def count(table: str) -> int:
            return self._conn.execute(
                f"SELECT COUNT(*) as n FROM {table} WHERE research_question_id = ?", (rq_id,)
            ).fetchone()["n"]

        return {
            "research_question": self.get_research_question(rq_id),
            "papers": count("papers"),
            "evidence": count("evidence"),
            "claims": count("claims"),
            "contradictions": count("contradictions"),
            "hypotheses": count("hypotheses"),
            "research_gaps": count("research_gaps"),
            "audit_flags": count("audit_flags"),
        }