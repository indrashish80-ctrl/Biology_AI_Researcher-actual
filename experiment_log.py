"""
experiment_log.py  (v25)
========================
Every in-silico experiment is logged - including the ones that FAILED or were REJECTED by the validator - with what
is needed to repeat it exactly: model id and version (hash of the card + Antimony source), parameters, seed, solver
and tolerances, the spec, the status, the error, the duration and a summary of the outputs.

    cache/experiments.db   table experiments
    log(record) -> id      get(id)      list(limit, status)      stats()
"""
from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

LOG_VERSION = "1.0"
_LOCK = threading.Lock()
_DDL = """
CREATE TABLE IF NOT EXISTS experiments (
    id INTEGER PRIMARY KEY AUTOINCREMENT, started REAL NOT NULL, finished REAL, status TEXT NOT NULL,
    model_id TEXT, model_version TEXT, seed INTEGER, solver_json TEXT, parameters_json TEXT, spec_json TEXT,
    error TEXT, duration_s REAL, summary_json TEXT, n_simulations INTEGER, research_question_id INTEGER, job_id TEXT
);
CREATE INDEX IF NOT EXISTS idx_exp_status ON experiments(status);
"""
STATUSES = ("queued", "running", "ok", "failed", "rejected", "timeout", "resource_limit")


def _db() -> sqlite3.Connection:
    p = Path(os.getenv("BIOSYN_EXPERIMENT_DB") or Path(__file__).resolve().parent / "cache" / "experiments.db")
    p.parent.mkdir(parents=True, exist_ok=True)
    c = sqlite3.connect(str(p), timeout=30)
    c.row_factory = sqlite3.Row
    c.executescript(_DDL)
    return c


def log(rec: Dict[str, Any]) -> int:
    with _LOCK:
        c = _db()
        try:
            cur = c.execute(
                "INSERT INTO experiments (started, finished, status, model_id, model_version, seed, solver_json, parameters_json, "
                "spec_json, error, duration_s, summary_json, n_simulations, research_question_id, job_id) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (rec.get("started") or time.time(), rec.get("finished"), rec.get("status") or "ok", rec.get("model_id"),
                 rec.get("model_version"), rec.get("seed"), json.dumps(rec.get("solver")), json.dumps(rec.get("parameters")),
                 json.dumps(rec.get("spec"), default=str), rec.get("error"), rec.get("duration_s"),
                 json.dumps(rec.get("summary"), default=str), rec.get("n_simulations"), rec.get("research_question_id"),
                 rec.get("job_id")))
            c.commit()
            return int(cur.lastrowid)
        finally:
            c.close()


def update(exp_id: int, **kw) -> None:
    if not kw:
        return
    enc = {k: (json.dumps(v, default=str) if k.endswith("_json") or k in ("summary", "solver", "spec", "parameters") else v)
           for k, v in kw.items()}
    enc = {(k + "_json" if k in ("summary", "solver", "spec", "parameters") else k): v for k, v in enc.items()}
    with _LOCK:
        c = _db()
        try:
            c.execute(f"UPDATE experiments SET {', '.join(f'{k}=?' for k in enc)} WHERE id=?", tuple(enc.values()) + (exp_id,))
            c.commit()
        finally:
            c.close()


def mark_interrupted() -> int:
    """Runs still 'running' / 'queued' from a previous server process are marked 'interrupted' (they cannot finish)."""
    with _LOCK:
        with _db() as c:
            n = c.execute("UPDATE experiments SET status='interrupted', error=COALESCE(error, 'the server stopped "
                          "before this run finished') WHERE status IN ('running', 'queued')").rowcount
    return n


def _row(r: sqlite3.Row) -> Dict[str, Any]:
    d = dict(r)
    for k in ("solver_json", "parameters_json", "spec_json", "summary_json"):
        d[k[:-5]] = json.loads(d.pop(k) or "null")
    return d


def get(exp_id: int) -> Optional[Dict[str, Any]]:
    with _LOCK:
        c = _db()
        try:
            r = c.execute("SELECT * FROM experiments WHERE id=?", (exp_id,)).fetchone()
        finally:
            c.close()
    return _row(r) if r else None


def list_runs(limit: int = 50, status: Optional[str] = None) -> List[Dict[str, Any]]:
    with _LOCK:
        c = _db()
        try:
            if status:
                rows = c.execute("SELECT * FROM experiments WHERE status=? ORDER BY id DESC LIMIT ?", (status, limit)).fetchall()
            else:
                rows = c.execute("SELECT * FROM experiments ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
        finally:
            c.close()
    return [_row(r) for r in rows]


def stats() -> Dict[str, int]:
    with _LOCK:
        c = _db()
        try:
            return {s: n for s, n in c.execute("SELECT status, COUNT(*) FROM experiments GROUP BY status").fetchall()}
        finally:
            c.close()
