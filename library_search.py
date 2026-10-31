"""
library_search.py  (v26)
========================
LOCAL-FIRST retrieval: the engine searches the local library before any external API, and decides by rule whether
the external search is needed at all.

    identify(question, queries, k, collections) -> {'candidates': [PaperCandidate], 'decision': {...}}

Search = BM25 over title/abstract/MeSH (with the synonym expansion of query_parser) for the question and every planned
query + the semantic index (library_vectors) when it was built with the same embedder, merged by reciprocal rank and
reranked on PICO match (hybrid_retrieval.pico_score). Retracted records stay in (they are flagged downstream, never
silently dropped); deleted records never appear.

Coverage rule (BIOSYN_LOCAL_FIRST = auto | only | off; default auto):
    the external search is SKIPPED only when ALL hold
      - the library was updated within BIOSYN_LIBRARY_MAX_AGE_DAYS (default 8) days,
      - at least BIOSYN_LOCAL_MIN_HITS (default 20) records match the question's intervention / exposure AND outcome
        (PICO score >= 0.5), and
      - the question's main concepts fall inside the library's scope (scope MeSH / keywords hit the matched records);
    otherwise the external sources run as before and the local hits join them (nothing is lost).
    'only' never calls external sources (offline use); 'off' ignores the library.
The decision, its reasons and the library snapshot date go into the PRISMA record of the dossier.
"""
from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence

SEARCH_VERSION = "1.0"


def policy() -> str:
    v = (os.getenv("BIOSYN_LOCAL_FIRST") or "auto").strip().lower()
    return v if v in ("auto", "only", "off") else "auto"


def _age_days(iso: Optional[str]) -> Optional[float]:
    if not iso:
        return None
    try:
        t = datetime.fromisoformat(str(iso).replace("Z", "+00:00"))
        if t.tzinfo is None:
            t = t.replace(tzinfo=timezone.utc)
        return (datetime.now(timezone.utc) - t).total_seconds() / 86400
    except Exception:
        return None


_VEC: Dict[str, Any] = {}
_VEC_LOCK = __import__("threading").Lock()


def _vectors(folder: Any) -> Any:
    """One open vector index per folder, kept between searches: its row map and memory-mapped shards stay warm and
    are reloaded only when the index changes (generation counter)."""
    import library_vectors as LV
    key = str(folder)
    try:
        ino = os.stat(os.path.join(key, "index.db")).st_ino
    except OSError:
        ino = None
    with _VEC_LOCK:
        hit = _VEC.get(key)
        if hit is not None and hit[1] == ino:
            return hit[0]
        if hit is not None:
            hit[0].close()                        # the folder was rebuilt: reopen
        v = LV.LibraryVectors(folder)
        _VEC[key] = (v, ino)
        return v


def warm() -> Optional[str]:
    """Load the library's vector index into RAM (server start-up), so the first search costs no more than the next."""
    import library_db as L
    if not L.exists():
        return None
    import time
    t = time.time()
    folder = L.library_dir() / "vectors"
    if not (folder / "index.db").exists():
        return "keyword index only (no vectors)"
    v = _vectors(folder)
    rmap = v._row_map()
    dim = int(v._meta("dim") or 0)
    n = len(rmap)
    import library_vectors as LV
    for sh in range((n - 1) // LV.SHARD_ROWS + 1 if n and dim else 0):
        v._dense(sh, dim, min(n, (sh + 1) * LV.SHARD_ROWS) - sh * LV.SHARD_ROWS)
    v._use_hnsw()                                # loads hnsw.bin when present and current
    return f"{v.count()} vectors in {time.time() - t:.1f}s"


def search(lib: Any, question: str, queries: Sequence[str] = (), k: int = 100,
           collections: Sequence[str] = ("literature",), use_vectors: bool = True) -> List[Dict[str, Any]]:
    import hybrid_retrieval as hr
    import query_parser as qp
    parsed = qp.parse(question)
    concepts = qp.search_concepts(parsed)
    groups = [set(c["terms"]) for c in concepts] or None
    rankings: List[List[int]] = []
    recs: Dict[int, Dict[str, Any]] = {}
    qs = [question, *dict.fromkeys(x for x in queries if x and x != question)][:12]

    def _kw(q: str) -> List[Dict[str, Any]]:
        if q == question or len(qs) == 1:
            return lib.search(q, k=k, collections=collections, concepts=groups if q == question else None)
        import library_db as L
        with L.Library(str(lib.path)) as own:   # SQLite releases the GIL while it searches: own read connection
            return own.search(q, k=k, collections=collections)

    if len(qs) > 1:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=min(4, len(qs))) as ex:
            results = list(ex.map(_kw, qs))
    else:
        results = [_kw(question)]
    for hits in results:                         # merged in the same order as before: results are identical
        rankings.append([h["id"] for h in hits])
        for h in hits:
            recs.setdefault(h["id"], h)
    if use_vectors and (lib.dir / "vectors" / "index.db").exists():
        try:
            vec = _vectors(lib.dir / "vectors")
            if vec.count():
                allowed = None
                if list(collections) != ["literature"]:
                    allowed = {r[0] for r in lib.c.execute(
                        f"SELECT id FROM records WHERE deleted=0 AND collection IN ({','.join('?' * len(collections))})",
                        tuple(collections))}
                vh = vec.query(question, k=k, allowed=allowed)
                ids = [h["record_id"] for h in vh]
                rankings.append(ids)
                for rid in ids:
                    if rid not in recs:
                        r = lib.get(rid)
                        if r and not r["deleted"] and r["collection"] in collections:
                            recs[rid] = r
        except ValueError:
            pass                                  # vectors made with another embedder: keyword search only
        except Exception:
            pass
    fused = hr.rrf(rankings)
    out = []
    for rid, f in sorted(fused.items(), key=lambda x: -x[1])[: k * 2]:
        r = recs.get(rid)
        if not r:
            continue
        ps, roles = hr.pico_score(r.get("title") or "", r.get("abstract") or "", concepts,
                                  parsed.get("preferred_designs") or [])
        out.append({**r, "fused": f, "pico": ps, "matched": roles})
    top = max((o["fused"] for o in out), default=1.0) or 1.0
    for o in out:
        o["score"] = round(0.45 * o["fused"] / top + 0.55 * o["pico"], 4)
    out.sort(key=lambda o: -o["score"])
    return out[:k]


def decide(lib: Any, question: str, hits: List[Dict[str, Any]], min_hits: Optional[int] = None) -> Dict[str, Any]:
    pol = policy()
    min_hits = int(min_hits or os.getenv("BIOSYN_LOCAL_MIN_HITS") or 20)
    max_age = float(os.getenv("BIOSYN_LIBRARY_MAX_AGE_DAYS") or 8)
    last = lib.get_sync("last_update")
    age = _age_days(last)
    strong = [h for h in hits if h["pico"] >= 0.5]
    scope = lib.get_sync("scope") or {}
    reasons: List[str] = []
    if pol == "off":
        return {"policy": pol, "use_online": True, "use_local": False, "reasons": ["BIOSYN_LOCAL_FIRST=off"],
                "snapshot": last, "local_hits": 0, "strong_hits": 0}
    if age is None or age > max_age:
        reasons.append(f"library last updated {('%.0f days ago' % age) if age is not None else 'never'} "
                       f"(limit {max_age:.0f} days) - newer papers may be missing")
    if len(strong) < min_hits:
        reasons.append(f"{len(strong)} local record(s) match intervention and outcome (need {min_hits})")
    if scope.get("mesh") or scope.get("keywords"):
        terms = [t.lower() for t in (scope.get("mesh") or []) + (scope.get("keywords") or [])]
        inside = any(any(t in (" ".join(h.get("mesh") or []) + " " + (h.get("title") or "") + " " +
                               (h.get("abstract") or "")).lower() for t in terms) for h in strong[:20])
        if not inside:
            reasons.append("the question falls outside the library's scope")
    use_online = pol != "only" and bool(reasons)
    return {"policy": pol, "use_online": use_online, "use_local": True,
            "reasons": reasons or [f"{len(strong)} strong local matches in a library updated "
                                   f"{age:.1f} day(s) ago - external search skipped"],
            "snapshot": last, "local_hits": len(hits), "strong_hits": len(strong), "scope": scope,
            "version": SEARCH_VERSION}


def identify(question: str, queries: Sequence[str] = (), k: int = 100,
             collections: Sequence[str] = ("literature",)) -> Optional[Dict[str, Any]]:
    """None when there is no library (or policy 'off'); else local candidates + the online-or-not decision."""
    import library_db as L
    if policy() == "off" or not L.exists():
        return None
    with L.Library() as lib:
        hits = search(lib, question, queries, k=k, collections=collections)
        dec = decide(lib, question, hits)
        cands = []
        for h in hits:
            c = L.to_candidate(h)
            c.relevance_score = h["score"]
            c.source_queries = ["local library"]
            cands.append(c)
    return {"candidates": cands, "decision": dec}


def fulltext_for(cand: Any) -> Optional[str]:
    """Path of the library's JATS full text for a candidate (by PMCID / PMID / DOI), or None."""
    import library_db as L
    from pathlib import Path
    if not L.exists():
        return None
    with L.Library() as lib:
        r = lib.by_ids(pmid=getattr(cand, "pmid", None), doi=getattr(cand, "doi", None), pmcid=getattr(cand, "pmcid", None))
    if r and r.get("fulltext_path") and r.get("fulltext_kind") == "jats_xml" and Path(r["fulltext_path"]).exists():
        return r["fulltext_path"]
    return None
