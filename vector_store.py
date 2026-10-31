"""
vector_store.py
---------------
Local, zero-cost semantic search over every paper the engine has ever screened.

Storage   : ChromaDB PersistentClient on the local disk (./vector_store/)
            -> falls back to a NumPy matrix on disk if chromadb is not installed.
Embedding : BAAI/bge-small-en-v1.5 via fastembed (ONNX, CPU-only, ~130 MB, downloaded once)
            -> falls back to a deterministic hashing embedder if the model cannot be
               downloaded (offline machine). Search still works, just less semantically.

Nothing here calls a paid or hosted service. Embeddings are computed on the user's CPU.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import threading
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np

DEFAULT_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-small-en-v1.5")
DEFAULT_PATH = os.getenv("VECTOR_STORE_PATH", str(Path(__file__).resolve().parent / "vector_store"))   # next to the program, whatever folder it is started from

_TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9\-]{1,}")
_STOP = {
    "the", "of", "and", "in", "on", "to", "a", "an", "with", "by", "for", "is", "are", "was",
    "were", "this", "that", "these", "we", "our", "from", "as", "at", "be", "or", "it", "its",
    "has", "have", "had", "not", "but", "can", "which", "using", "used", "study", "results",
}


# -----------------------------------------------------------------------------
# Embedders
# -----------------------------------------------------------------------------

class HashingEmbedder:
    """Offline fallback: signed feature hashing of unigrams + bigrams, sublinear TF, L2-norm."""

    name = "hashing-512"

    def __init__(self, dim: int = 512):
        self.dim = dim

    def _vec(self, text: str) -> np.ndarray:
        toks = [t for t in _TOKEN_RE.findall((text or "").lower()) if t not in _STOP]
        feats = toks + [f"{a}_{b}" for a, b in zip(toks, toks[1:])]
        v = np.zeros(self.dim, dtype=np.float32)
        counts: Dict[str, int] = {}
        for f in feats:
            counts[f] = counts.get(f, 0) + 1
        for f, c in counts.items():
            h = int.from_bytes(hashlib.blake2b(f.encode(), digest_size=8).digest(), "little")
            idx, sign = h % self.dim, 1.0 if (h >> 63) & 1 else -1.0
            v[idx] += sign * (1.0 + math.log(c))
        n = float(np.linalg.norm(v))
        return v / n if n > 0 else v

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        return np.vstack([self._vec(t) for t in texts]) if texts else np.zeros((0, self.dim), np.float32)


class FastEmbedEmbedder:
    """BAAI/bge-small-en-v1.5 (or any fastembed model) on local CPU."""

    def __init__(self, model_name: str = DEFAULT_MODEL):
        from fastembed import TextEmbedding  # lazy import: optional dependency
        cache = os.getenv("FASTEMBED_CACHE_PATH", str(Path(__file__).resolve().parent / "cache" / "fastembed"))
        short = model_name.split("/")[-1].lower()
        on_disk = Path(cache).is_dir() and any(short in p.name.lower() for p in Path(cache).iterdir())
        self.model = None
        if on_disk:
            try:
                # the model downloaded once is loaded from disk without asking the internet first (an online check on
                # a slow or blocked connection made start-up take minutes)
                self.model = TextEmbedding(model_name=model_name, cache_dir=cache, local_files_only=True)
            except Exception:
                self.model = None
        if self.model is None:
            self.model = TextEmbedding(model_name=model_name, cache_dir=cache)       # first run: download it once
        self.name = model_name.split("/")[-1]

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 384), np.float32)
        vecs = np.array(list(self.model.embed(list(texts), batch_size=32)), dtype=np.float32)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return vecs / norms


_EMBEDDER_LOCK = threading.Lock()
_EMBEDDER: Any = None


def get_embedder() -> Any:
    """Process-wide singleton so the ONNX model is loaded once."""
    global _EMBEDDER
    with _EMBEDDER_LOCK:
        if _EMBEDDER is None:
            if os.getenv("EMBED_BACKEND", "").lower() == "hashing":
                _EMBEDDER = HashingEmbedder()
            else:
                try:
                    _EMBEDDER = FastEmbedEmbedder()
                    _EMBEDDER.embed(["warm-up"])
                except Exception as exc:  # offline / not installed
                    print(f"  [VectorStore] Local embedding model unavailable ({str(exc)[:120]}); "
                          f"using offline hashing embedder.")
                    _EMBEDDER = HashingEmbedder()
        return _EMBEDDER


# -----------------------------------------------------------------------------
# Store
# -----------------------------------------------------------------------------

def _clean_meta(meta: Dict[str, Any]) -> Dict[str, Any]:
    """Chroma metadata must be str/int/float/bool and non-None."""
    out = {}
    for k, v in (meta or {}).items():
        if v is None:
            continue
        if isinstance(v, (str, int, float, bool)):
            out[k] = v
        else:
            out[k] = json.dumps(v, ensure_ascii=False)[:2000]
    return out


class LocalVectorStore:
    def __init__(self, path: str = DEFAULT_PATH, embedder: Any = None):
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)
        self.embedder = embedder or get_embedder()
        self.collection_name = re.sub(r"[^a-zA-Z0-9_-]", "_", f"papers_{self.embedder.name}")[:60]
        self._lock = threading.Lock()
        self.backend = "numpy"
        self._col = None
        try:
            import chromadb
            from chromadb.config import Settings

            client = chromadb.PersistentClient(path=str(self.path), settings=Settings(anonymized_telemetry=False))
            self._col = client.get_or_create_collection(
                name=self.collection_name, metadata={"hnsw:space": "cosine"}, embedding_function=None
            )
            self.backend = "chromadb"
        except Exception as exc:
            self._to_numpy(exc)

    def _to_numpy(self, exc: Exception) -> None:
        """ChromaDB failed (not installed, locked, or its internal database is damaged - 'no such table: embeddings'):
        continue with the local NumPy store instead of stopping the investigation. The library is a speed-up for
        screening, never a requirement."""
        print(f"  [VectorStore] ChromaDB unavailable ({str(exc)[:100]}); using NumPy store.")
        self.backend = "numpy"
        self._col = None
        self._np_path = self.path / f"{self.collection_name}.npz"
        self._meta_path = self.path / f"{self.collection_name}.json"
        try:
            self._load_np()
        except Exception as exc2:                       # a damaged NumPy file: start an empty library
            print(f"  [VectorStore] local library unreadable ({str(exc2)[:80]}); starting an empty one.")
            self._vecs, self._ids, self._docs, self._metas = None, [], [], []

    # ---- numpy fallback persistence -------------------------------------------------
    def _load_np(self) -> None:
        if self._np_path.exists() and self._meta_path.exists():
            self._vecs = np.load(self._np_path)["v"]
            blob = json.loads(self._meta_path.read_text(encoding="utf-8"))
            self._ids, self._docs, self._metas = blob["ids"], blob["docs"], blob["metas"]
        else:
            self._vecs, self._ids, self._docs, self._metas = None, [], [], []

    def _save_np(self) -> None:
        np.savez_compressed(self._np_path, v=self._vecs if self._vecs is not None else np.zeros((0, 1)))
        self._meta_path.write_text(json.dumps({"ids": self._ids, "docs": self._docs, "metas": self._metas}),
                                   encoding="utf-8")

    # ---- public API -----------------------------------------------------------------
    def embed(self, texts: Sequence[str]) -> np.ndarray:
        # v25: embeddings are cached by (model, text hash) - a paper is embedded once per model
        try:
            from retrieval_cache import cached_embed
            return cached_embed(self.embedder, list(texts))
        except Exception:
            return self.embedder.embed(list(texts))

    def upsert(self, items: Iterable[Dict[str, Any]]) -> int:
        """items: {"id": str, "text": str, "metadata": dict}"""
        items = [i for i in items if i.get("id") and (i.get("text") or "").strip()]
        if not items:
            return 0
        # de-duplicate ids within the batch (Chroma rejects duplicates)
        uniq: Dict[str, Dict[str, Any]] = {}
        for it in items:
            uniq[str(it["id"])] = it
        items = list(uniq.values())
        vecs = self.embed([i["text"] for i in items])
        with self._lock:
            if self.backend == "chromadb":
                try:
                    self._col.upsert(
                        ids=[str(i["id"]) for i in items],
                        embeddings=vecs.tolist(),
                        documents=[i["text"][:8000] for i in items],
                        metadatas=[_clean_meta(i.get("metadata") or {}) or {"_": 1} for i in items],
                    )
                    return len(items)
                except Exception as exc:
                    self._to_numpy(exc)
            if self.backend != "chromadb":
                index = {pid: n for n, pid in enumerate(self._ids)}
                for it, v in zip(items, vecs):
                    pid = str(it["id"])
                    if pid in index:
                        self._vecs[index[pid]] = v
                        self._docs[index[pid]] = it["text"][:8000]
                        self._metas[index[pid]] = _clean_meta(it.get("metadata") or {})
                    else:
                        self._vecs = v[None, :] if self._vecs is None or len(self._vecs) == 0 else np.vstack([self._vecs, v])
                        self._ids.append(pid)
                        self._docs.append(it["text"][:8000])
                        self._metas.append(_clean_meta(it.get("metadata") or {}))
                        index[pid] = len(self._ids) - 1
                self._save_np()
        return len(items)

    def count(self) -> int:
        if self.backend == "chromadb":
            try:
                return int(self._col.count())
            except Exception as exc:
                self._to_numpy(exc)
        return len(self._ids)

    def query(self, text: str, k: int = 10, where: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        n = self.count()
        if n == 0 or not (text or "").strip():
            return []
        qv = self.embed([text])[0]
        k = max(1, min(k, n))
        if self.backend == "chromadb":
            try:
                res = self._col.query(query_embeddings=[qv.tolist()], n_results=k, where=where or None,
                                      include=["metadatas", "documents", "distances"])
                out = []
                for pid, meta, doc, dist in zip(res["ids"][0], res["metadatas"][0], res["documents"][0], res["distances"][0]):
                    out.append({"id": pid, "score": round(1.0 - float(dist), 4), "metadata": meta or {}, "document": doc})
                return out
            except Exception as exc:
                self._to_numpy(exc)
                n = self.count()
                if n == 0:
                    return []
                k = max(1, min(k, n))
        if self._vecs is None or len(self._ids) == 0:
            return []
        sims = self._vecs @ qv
        order = np.argsort(-sims)
        out = []
        for i in order:
            meta = self._metas[i]
            if where and any(meta.get(key) != val for key, val in where.items()):
                continue
            out.append({"id": self._ids[i], "score": round(float(sims[i]), 4), "metadata": meta, "document": self._docs[i]})
            if len(out) >= k:
                break
        return out

    def info(self) -> Dict[str, Any]:
        try:
            n = self.count()
        except Exception:
            n = None
        return {"backend": self.backend, "embedder": self.embedder.name,
                "collection": self.collection_name, "documents": n, "path": str(self.path)}


def cosine_scores(embedder: Any, query: str, texts: Sequence[str]) -> List[float]:
    """Similarity of `query` to each text, computed locally (used by PRISMA screening)."""
    if not texts:
        return []
    m = embedder.embed([query] + list(texts))
    return [float(x) for x in (m[1:] @ m[0])]
