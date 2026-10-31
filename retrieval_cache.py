"""
retrieval_cache.py  (v25)
=========================
Embedding cache and retrieval-result cache, each with an explicit invalidation rule.

Embedding cache  (cache/embeddings.db)
    key   = embedding model name + SHA-256 of the exact text
    rule  = an embedding never changes for the same model and text, so entries never expire; switching the model
            (EMBED_MODEL) makes every lookup a miss automatically. Only real models are cached (the offline hashing
            embedder is cheaper to recompute than to look up).

Retrieval cache  (cache/retrieval.db)
    key   = normalised query + mode + k + INDEX VERSION
    index version = hash of (number of indexed papers, latest index time, vector count, embedding model,
                    synonym table size, retrieval code version)
    rule  = adding, changing or removing a paper, re-embedding, changing the synonym table or the ranking code
            changes the index version, so stale results are never served. Entries also expire after 7 days.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

RCACHE_VERSION = "1.0"
_LOCK = threading.Lock()
STATS = {"emb_hits": 0, "emb_misses": 0, "ret_hits": 0, "ret_misses": 0}
RET_TTL_S = 7 * 24 * 3600


def _dir() -> Path:
    d = Path(os.getenv("BIOSYN_CACHE_DIR") or Path(__file__).resolve().parent / "cache")
    d.mkdir(parents=True, exist_ok=True)
    return d


def _c(name: str, ddl: str) -> sqlite3.Connection:
    c = sqlite3.connect(str(_dir() / name), timeout=30)
    c.executescript(ddl)
    return c


_EMB_DDL = "CREATE TABLE IF NOT EXISTS emb (key TEXT PRIMARY KEY, dim INTEGER, vec BLOB NOT NULL)"
_RET_DDL = "CREATE TABLE IF NOT EXISTS ret (key TEXT PRIMARY KEY, value TEXT NOT NULL, created REAL NOT NULL)"


def _ekey(model: str, text: str) -> str:
    return model + "|" + hashlib.sha256((text or "").encode("utf-8", "replace")).hexdigest()


def cached_embed(embedder: Any, texts: Sequence[str]):
    """Embed with the cache in front; returns a numpy array in the input order."""
    import numpy as np
    name = getattr(embedder, "name", type(embedder).__name__)
    if "hash" in str(name).lower() or os.getenv("BIOSYN_NO_EMBED_CACHE") == "1":
        return embedder.embed(list(texts))
    keys = [_ekey(str(name), t) for t in texts]
    found: Dict[str, Any] = {}
    with _LOCK:
        c = _c("embeddings.db", _EMB_DDL)
        try:
            for i in range(0, len(keys), 500):
                chunk = keys[i:i + 500]
                for k, dim, blob in c.execute(f"SELECT key, dim, vec FROM emb WHERE key IN ({','.join('?' * len(chunk))})", chunk):
                    found[k] = np.frombuffer(blob, dtype=np.float32, count=dim)
        finally:
            c.close()
    missing = [i for i, k in enumerate(keys) if k not in found]
    STATS["emb_hits"] += len(keys) - len(missing)
    STATS["emb_misses"] += len(missing)
    if missing:
        vecs = embedder.embed([texts[i] for i in missing])
        with _LOCK:
            c = _c("embeddings.db", _EMB_DDL)
            try:
                c.executemany("INSERT OR REPLACE INTO emb (key, dim, vec) VALUES (?,?,?)",
                              [(keys[i], int(v.shape[0]), np.asarray(v, dtype=np.float32).tobytes()) for i, v in zip(missing, vecs)])
                c.commit()
            finally:
                c.close()
        for i, v in zip(missing, vecs):
            found[keys[i]] = np.asarray(v, dtype=np.float32)
    return np.vstack([found[k] for k in keys]) if keys else np.zeros((0, 1), dtype=np.float32)


def index_version(db: Any = None, vector_store: Any = None) -> str:
    parts = [RCACHE_VERSION]
    try:
        import hybrid_retrieval
        parts.append(hybrid_retrieval.RETRIEVAL_VERSION)
    except Exception:
        pass
    try:
        import mesh_synonyms
        parts.append(str(mesh_synonyms.info()["terms"]))
    except Exception:
        pass
    if db is not None:
        try:
            n, last = db._conn.execute("SELECT COUNT(*), MAX(indexed_at) FROM papers_fts_state").fetchone()
            parts += [str(n), str(last)]
        except Exception:
            parts.append(str(db._conn.execute("SELECT COUNT(*), MAX(id) FROM papers").fetchone()))
    if vector_store is not None:
        try:
            parts += [str(vector_store.count()), str(getattr(vector_store.embedder, "name", ""))]
        except Exception:
            pass
    return hashlib.sha1("|".join(parts).encode()).hexdigest()[:16]


def _rkey(query: str, mode: str, k: int, version: str) -> str:
    q = re.sub(r"\s+", " ", (query or "").lower()).strip()
    return hashlib.sha256(f"{q}|{mode}|{k}|{version}".encode()).hexdigest()


def get_results(query: str, mode: str, k: int, version: str) -> Optional[List[Dict[str, Any]]]:
    if os.getenv("BIOSYN_NO_RETRIEVAL_CACHE") == "1":
        return None
    key = _rkey(query, mode, k, version)
    with _LOCK:
        c = _c("retrieval.db", _RET_DDL)
        try:
            r = c.execute("SELECT value, created FROM ret WHERE key=?", (key,)).fetchone()
        finally:
            c.close()
    if r and time.time() - r[1] < RET_TTL_S:
        STATS["ret_hits"] += 1
        return json.loads(r[0])
    STATS["ret_misses"] += 1
    return None


def put_results(query: str, mode: str, k: int, version: str, results: List[Dict[str, Any]]) -> None:
    if os.getenv("BIOSYN_NO_RETRIEVAL_CACHE") == "1":
        return
    with _LOCK:
        c = _c("retrieval.db", _RET_DDL)
        try:
            c.execute("INSERT OR REPLACE INTO ret (key, value, created) VALUES (?,?,?)",
                      (_rkey(query, mode, k, version), json.dumps(results, default=str), time.time()))
            c.execute("DELETE FROM ret WHERE created < ?", (time.time() - RET_TTL_S,))
            c.commit()
        finally:
            c.close()


def stats() -> Dict[str, Any]:
    e = STATS["emb_hits"] + STATS["emb_misses"]
    r = STATS["ret_hits"] + STATS["ret_misses"]
    return {**STATS, "embedding_hit_rate": round(STATS["emb_hits"] / e, 3) if e else None,
            "retrieval_hit_rate": round(STATS["ret_hits"] / r, 3) if r else None}
