"""
library_vectors.py  (v26.1)
===========================
Semantic index of the local library, built for hundreds of thousands to a few million records on an ordinary computer.

    library/vectors/
        index.db                 row -> record id, alive flag, text hash; embedder name, dimension, generation
        shard_00000.npy ...      float16 vectors, PREALLOCATED to 50,000 rows each (memory-mapped; rows are written in
                                 place, a shard is never rewritten - v26.0 rewrote the whole shard on every batch)
        hnsw.bin                 optional approximate index (`pip install hnswlib`)

Search without hnswlib is an exact, vectorised dot product over the memory-mapped shards (float16 -> float32 in
blocks), dead rows masked out; with hnswlib it is a few milliseconds. The row -> record map is cached in memory and
reloaded only when another process changed the index (a generation counter). 1M records at 384 dims = 0.77 GB.
"""
from __future__ import annotations

import os
import sqlite3
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

VECTORS_VERSION = "1.1"
SHARD_ROWS = 50_000
BLOCK = 50_000


class LibraryVectors:
    def __init__(self, folder: Optional[Path] = None, embedder: Any = None):
        from library_db import library_dir
        self.dir = Path(folder or library_dir() / "vectors")
        self.dir.mkdir(parents=True, exist_ok=True)
        self._embedder = embedder
        self._lock = threading.RLock()
        self.c = sqlite3.connect(str(self.dir / "index.db"), check_same_thread=False, timeout=60)
        self.c.executescript("""
            CREATE TABLE IF NOT EXISTS vec (row INTEGER PRIMARY KEY, record_id INTEGER NOT NULL, text_hash TEXT,
                                            alive INTEGER NOT NULL DEFAULT 1);
            CREATE INDEX IF NOT EXISTS ix_vec_rec ON vec(record_id, alive);
            CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);
        """)
        self.c.commit()
        self._hnsw = None
        self._map_gen: Optional[str] = None
        self._row_rec: Optional[np.ndarray] = None      # row -> record id, -1 = dead
        self._mm: Dict[int, np.ndarray] = {}
        self._f32: Dict[int, np.ndarray] = {}           # float32 copies of shards kept in RAM (budget below)
        self._ram_budget = float(os.getenv("BIOSYN_VECTOR_RAM_MB") or 1024) * 1e6

    # ------------------------------------------------------------------------------------------------ basics
    @property
    def embedder(self) -> Any:
        if self._embedder is None:
            from vector_store import get_embedder
            self._embedder = get_embedder()
        return self._embedder

    def _meta(self, key: str, value: Optional[str] = None) -> Optional[str]:
        if value is not None:
            self.c.execute("INSERT OR REPLACE INTO meta VALUES (?,?)", (key, value))
            return value
        r = self.c.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        return r[0] if r else None

    def _bump(self) -> None:
        g = int(self._meta("generation") or 0) + 1
        self._meta("generation", str(g))

    def _check_embedder(self, dim: int) -> None:
        name = getattr(self.embedder, "name", type(self.embedder).__name__)
        stored = self._meta("embedder")
        if stored is None:
            self._meta("embedder", name)
            self._meta("dim", str(dim))
        elif stored != name or int(self._meta("dim") or dim) != dim:
            raise ValueError(f"the library's vectors were made with '{stored}' ({self._meta('dim')} dims), not '{name}' "
                             f"({dim}): rebuild them with `python library_loader.py embed --rebuild`")

    def count(self) -> int:
        return int(self.c.execute("SELECT COUNT(*) FROM vec WHERE alive=1").fetchone()[0])

    def total_rows(self) -> int:
        return int(self.c.execute("SELECT COALESCE(MAX(row), -1) + 1 FROM vec").fetchone()[0])

    def missing(self, items: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """The items (record_id, text_hash) whose current text is not embedded yet (one query per call)."""
        ids = [it["record_id"] for it in items]
        have: Dict[int, set] = {}
        for i in range(0, len(ids), 900):
            chunk = ids[i:i + 900]
            for rid, th in self.c.execute(f"SELECT record_id, text_hash FROM vec WHERE alive=1 AND record_id IN "
                                          f"({','.join('?' * len(chunk))})", chunk):
                have.setdefault(rid, set()).add(th)
        return [it for it in items if it["text_hash"] not in have.get(it["record_id"], set())]

    def _shard_path(self, n: int) -> Path:
        return self.dir / f"shard_{n:05d}.npy"

    def _shard(self, n: int, dim: int, write: bool) -> np.ndarray:
        """The memory-mapped shard, created preallocated; a v26.0 (compact, rewritten) shard is upgraded once."""
        p = self._shard_path(n)
        if not p.exists():
            if not write:
                return np.zeros((0, dim), dtype=np.float16)
            arr = np.lib.format.open_memmap(p, mode="w+", dtype=np.float16, shape=(SHARD_ROWS, dim))
            del arr
        arr = np.load(p, mmap_mode="r+" if write else "r")
        if arr.shape[0] < SHARD_ROWS and write:          # v26.0 shard: copy into a preallocated file once
            old = np.array(arr)
            del arr
            tmp = p.with_suffix(".tmp.npy")
            big = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.float16, shape=(SHARD_ROWS, dim))
            big[:len(old)] = old
            big.flush()
            del big
            os.replace(tmp, p)
            arr = np.load(p, mmap_mode="r+")
        return arr

    # ------------------------------------------------------------------------------------------------ write
    def add(self, items: Sequence[Dict[str, Any]], batch: int = 256) -> int:
        """items: [{'record_id', 'text', 'text_hash'}]. Older rows of the same record are tombstoned."""
        items = [i for i in items if (i.get("text") or "").strip()]
        added = 0
        for j in range(0, len(items), batch):
            chunk = items[j:j + batch]
            vecs = np.asarray(self.embedder.embed([i["text"][:2000] for i in chunk]), dtype=np.float32)
            norms = np.linalg.norm(vecs, axis=1, keepdims=True)
            vecs = (vecs / np.where(norms == 0, 1, norms)).astype(np.float16)
            with self._lock:
                self._check_embedder(vecs.shape[1])
                dim = vecs.shape[1]
                self.c.executemany("UPDATE vec SET alive=0 WHERE record_id=?", [(it["record_id"],) for it in chunk])
                start = self.total_rows()
                pos = 0
                while pos < len(chunk):
                    row = start + pos
                    shard, off = divmod(row, SHARD_ROWS)
                    take = min(len(chunk) - pos, SHARD_ROWS - off)
                    arr = self._shard(shard, dim, write=True)
                    arr[off:off + take] = vecs[pos:pos + take]
                    arr.flush()
                    del arr
                    self.c.executemany("INSERT INTO vec (row, record_id, text_hash, alive) VALUES (?,?,?,1)",
                                       [(row + k, chunk[pos + k]["record_id"], chunk[pos + k].get("text_hash"))
                                        for k in range(take)])
                    pos += take
                self._bump()
                self.c.commit()
                self._hnsw = None
                self._mm.clear()
                self._f32.clear()
            added += len(chunk)
        return added

    def remove(self, record_id: int) -> None:
        with self._lock:
            self.c.execute("UPDATE vec SET alive=0 WHERE record_id=?", (record_id,))
            self._bump()
            self.c.commit()
            self._hnsw = None

    # ------------------------------------------------------------------------------------------------ search
    def _row_map(self) -> np.ndarray:
        gen = self._meta("generation") or "0"
        if self._row_rec is None or gen != self._map_gen:
            cache = self.dir / "rowmap.npy"
            m = None
            if self._meta("rowmap_generation") == gen and cache.exists():
                try:
                    m = np.load(cache)
                except Exception:
                    m = None
            if m is None:
                n = self.total_rows()
                m = np.full(n, -1, dtype=np.int64)
                rows = self.c.execute("SELECT row, record_id FROM vec WHERE alive=1").fetchall()
                if rows:
                    a = np.asarray(rows, dtype=np.int64)
                    m[a[:, 0]] = a[:, 1]
                try:
                    tmp = self.dir / "rowmap.tmp.npy"
                    np.save(tmp, m)
                    os.replace(tmp, cache)
                    with self._lock:
                        self._meta("rowmap_generation", gen)
                        self.c.commit()
                except Exception:
                    pass
            self._row_rec, self._map_gen = m, gen   # rows are append-only: the float32 copies stay valid
        return self._row_rec

    def query(self, text: str, k: int = 50, allowed: Optional[set] = None) -> List[Dict[str, Any]]:
        if not (text or "").strip():
            return []
        rmap = self._row_map()
        if not len(rmap) or not (rmap >= 0).any():
            return []
        q = np.asarray(self.embedder.embed([text])[0], dtype=np.float32)
        q = q / (np.linalg.norm(q) or 1.0)
        dim = int(self._meta("dim") or len(q))
        if self._use_hnsw():
            try:
                labels, dists = self._hnsw.knn_query(q, k=min(int((rmap >= 0).sum()), max(k * 4, k)))
                out = []
                for lab, dist in zip(labels[0], dists[0]):
                    rid = int(rmap[lab]) if lab < len(rmap) else -1
                    if rid < 0 or (allowed is not None and rid not in allowed):
                        continue
                    out.append({"record_id": rid, "score": round(1.0 - float(dist), 4)})
                    if len(out) >= k:
                        break
                if allowed is None:
                    return out
            except Exception:
                pass
        n = len(rmap)
        allow_mask = None
        if allowed is not None:
            allow_mask = np.isin(rmap, np.fromiter(allowed, dtype=np.int64, count=len(allowed)))
        best_s: List[np.ndarray] = []
        best_r: List[np.ndarray] = []
        for sh in range((n - 1) // SHARD_ROWS + 1):
            lo, hi = sh * SHARD_ROWS, min(n, (sh + 1) * SHARD_ROWS)
            arr = self._dense(sh, dim, hi - lo)
            for b0 in range(0, hi - lo, BLOCK):
                b1 = min(hi - lo, b0 + BLOCK)
                blk = arr[b0:b1]
                sims = (blk if blk.dtype == np.float32 else np.asarray(blk, dtype=np.float32)) @ q
                rows = np.arange(lo + b0, lo + b1)
                ok = rmap[rows] >= 0
                if allow_mask is not None:
                    ok &= allow_mask[rows]
                sims = np.where(ok, sims, -np.inf)
                kk = min(len(sims), k)
                top = np.argpartition(-sims, kk - 1)[:kk] if kk < len(sims) else np.arange(len(sims))
                best_s.append(sims[top])
                best_r.append(rows[top])
        s = np.concatenate(best_s)
        r = np.concatenate(best_r)
        order = np.argsort(-s)
        out = []
        for i in order:
            if not np.isfinite(s[i]):
                break
            out.append({"record_id": int(rmap[r[i]]), "score": round(float(s[i]), 4)})
            if len(out) >= k:
                break
        return out

    def _dense(self, sh: int, dim: int, used: int) -> np.ndarray:
        """The shard's used rows as float32 in RAM while the budget allows (float16 -> float32 conversion is the slow
        step of an exact search: ~120 ms per 50k vectors, against ~8 ms for the multiply); else the memory map."""
        a = self._f32.get(sh)
        if a is not None and len(a) >= used:
            return a
        mm = self._mm.get(sh)
        if mm is None:
            mm = self._mm[sh] = self._shard(sh, dim, write=False)
        need = used * dim * 4
        have = sum(x.nbytes for k, x in self._f32.items() if k != sh)
        if have + need <= self._ram_budget:
            a = np.asarray(mm[:used], dtype=np.float32)
            self._f32[sh] = a
            return a
        return mm

    def _use_hnsw(self) -> bool:
        if os.getenv("BIOSYN_NO_HNSW") == "1":
            return False
        if self._hnsw is not None:
            return True
        p = self.dir / "hnsw.bin"
        if not p.exists():
            return False
        try:
            import hnswlib
            if int(self._meta("hnsw_generation") or -1) != int(self._meta("generation") or 0):
                return False                       # stale: rows added / removed since it was built - exact search
            idx = hnswlib.Index(space="cosine", dim=int(self._meta("dim")))
            idx.load_index(str(p))
            idx.set_ef(100)
            self._hnsw = idx
            return True
        except Exception:
            return False

    def build_hnsw(self) -> Dict[str, Any]:
        """Approximate index over the alive rows (needs `pip install hnswlib`)."""
        try:
            import hnswlib
        except ImportError:
            return {"built": False, "reason": "pip install hnswlib to enable the approximate index"}
        rmap = self._row_map()
        alive = np.nonzero(rmap >= 0)[0]
        dim = int(self._meta("dim") or 0)
        idx = hnswlib.Index(space="cosine", dim=dim)
        idx.init_index(max_elements=max(1, len(alive)), ef_construction=200, M=16)
        for sh in range((len(rmap) - 1) // SHARD_ROWS + 1 if len(rmap) else 0):
            arr = self._shard(sh, dim, write=False)
            rows = alive[(alive >= sh * SHARD_ROWS) & (alive < (sh + 1) * SHARD_ROWS)]
            if len(rows):
                idx.add_items(np.asarray(arr[rows - sh * SHARD_ROWS], dtype=np.float32), rows)
        idx.save_index(str(self.dir / "hnsw.bin"))
        self._meta("hnsw_generation", self._meta("generation") or "0")
        self.c.commit()
        self._hnsw = None
        return {"built": True, "rows": int(len(alive))}

    def info(self) -> Dict[str, Any]:
        files = list(self.dir.glob("shard_*.npy"))
        return {"vectors": self.count(), "embedder": self._meta("embedder"), "dim": self._meta("dim"),
                "bytes": sum(f.stat().st_size for f in files), "shards": len(files),
                "hnsw": (self.dir / "hnsw.bin").exists(), "version": VECTORS_VERSION}

    def close(self) -> None:
        self._mm.clear()
        try:
            self.c.close()
        except Exception:
            pass
