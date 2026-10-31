"""
hybrid_retrieval.py  (v25)
==========================
Keyword search + vector search, merged, then reranked.

    1. parse the question into PICO (query_parser) and expand every element with biomedical synonyms (mesh_synonyms)
    2. candidates: BM25 top-N (keyword_index) and embedding top-N (vector_store), merged by reciprocal-rank fusion
       (RRF, k = 60 - the standard parameter-free way to merge rankings whose scores are not comparable)
    3. rerank the top candidates:
         - a cross-encoder when one is installed and allowed (BIOSYN_RERANKER=cross; fastembed's
           Xenova/ms-marco-MiniLM-L-6-v2, ~80 MB, runs on the CPU), blended with the fused rank, or
         - by default, a deterministic PICO reranker: how many of the question's weighted concepts (intervention,
           outcome, population, comparator) the title and abstract contain, plus a small bonus for the design the
           question type calls for (randomised trials for a therapy question)
       Every result carries WHY it ranked where it did (matched concepts, BM25 rank, vector rank).

    HybridRetriever(docs=[{'id','title','abstract'}], vector_store=None).search(question, k=20)
    rerank_records(question, records, semantic_scores)       -> blended scores for the screening step
    evaluate(questions, corpus, ks=(1, 5, 10, 20))            -> recall@k, MRR (used by gold_bench.py)
"""
from __future__ import annotations

import os
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

RETRIEVAL_VERSION = "1.0"
RRF_K = 60


def rrf(rankings: Sequence[Sequence[Any]], k: int = RRF_K) -> Dict[Any, float]:
    """Reciprocal-rank fusion: score(d) = sum over rankings of 1 / (k + rank)."""
    out: Dict[Any, float] = {}
    for ranking in rankings:
        for r, d in enumerate(ranking, 1):
            out[d] = out.get(d, 0.0) + 1.0 / (k + r)
    return out


def _has(term: str, text: str) -> bool:
    t = re.escape(term.lower()).replace(r"\ ", r"[\s\-]+")
    return re.search(rf"(?<![a-z0-9]){t}(?:s|es)?(?![a-z0-9])", text) is not None


_DESIGN_RX = {
    "randomised controlled trial": r"\brandomi[sz]ed|\brandomly (?:assigned|allocated)|\bplacebo-controlled|\brct\b",
    "meta-analysis": r"\bmeta-?analys[ie]s|\bsystematic review",
    "cohort": r"\bcohort\b|\bprospective(?:ly)?\b|\bfollow-?up\b",
    "case-control": r"\bcase-?control\b",
    "mendelian randomisation": r"\bmendelian randomi[sz]ation\b",
    "diagnostic accuracy study": r"\bsensitivity\b.*\bspecificity\b|\barea under the (?:roc )?curve\b",
    "cross-sectional": r"\bcross-?sectional\b|\bprevalence\b",
    "registry": r"\bregistry\b|\bnationwide\b",
    "experimental (cell / animal)": r"\bmice\b|\brats?\b|\bcell lines?\b|\bin vitro\b|\bxenograft",
}


def pico_score(title: str, abstract: str, concepts: List[Dict[str, Any]], preferred: Sequence[str] = ()) -> Tuple[float, List[str]]:
    """(0..1 score, matched roles) - weighted share of the question's concepts present in title (full credit) or
    abstract (70%), plus up to 0.1 for the preferred study design."""
    tl, al = (title or "").lower(), (abstract or "").lower()
    if not concepts:
        return 0.0, []
    tot = sum(c["weight"] for c in concepts)
    got = 0.0
    matched = []
    for c in concepts:
        in_t = any(_has(t, tl) for t in c["terms"])
        in_a = in_t or any(_has(t, al) for t in c["terms"])
        if in_t:
            got += c["weight"]
        elif in_a:
            got += 0.7 * c["weight"]
        if in_a:
            matched.append(c["role"])
    score = got / tot if tot else 0.0
    bonus = 0.0
    txt = tl + " " + al
    for d in preferred[:1]:
        rx = _DESIGN_RX.get(d)
        if rx and re.search(rx, txt):
            bonus = 0.1
    return min(1.0, score * 0.9 + bonus), sorted(set(matched))


class _Cross:
    """Optional cross-encoder (fastembed). Loaded only when BIOSYN_RERANKER=cross and the model is available."""
    _inst = None
    _failed = False

    @classmethod
    def get(cls):
        if cls._failed or (os.getenv("BIOSYN_RERANKER") or "").lower() != "cross":
            return None
        if cls._inst is None:
            try:
                from fastembed.rerank.cross_encoder import TextCrossEncoder
                cls._inst = TextCrossEncoder(os.getenv("BIOSYN_RERANK_MODEL", "Xenova/ms-marco-MiniLM-L-6-v2"))
            except Exception as exc:
                print(f"  [Rerank] cross-encoder unavailable ({str(exc)[:100]}); using the PICO reranker")
                cls._failed = True
                return None
        return cls._inst


def _cross_scores(query: str, texts: List[str]) -> Optional[List[float]]:
    ce = _Cross.get()
    if ce is None or not texts:
        return None
    try:
        s = list(ce.rerank(query, texts))
        lo, hi = min(s), max(s)
        return [(x - lo) / (hi - lo) if hi > lo else 0.5 for x in s]
    except Exception:
        return None


class HybridRetriever:
    def __init__(self, docs: Optional[Sequence[Dict[str, Any]]] = None, db: Any = None, vector_store: Any = None,
                 use_vectors: bool = True):
        from keyword_index import KeywordIndex, MemoryIndex
        self.docs = {str(d["id"]): d for d in docs or []}
        self.db = db
        self.vs = vector_store if use_vectors else None
        if docs is not None:
            self.kw = MemoryIndex(list(docs))
        else:
            self.kw = KeywordIndex(db)
            self.kw.sync()
        self._mem_vecs = None
        if docs is not None and use_vectors and vector_store is None:
            self._embed_docs()

    def _embed_docs(self) -> None:
        try:
            from vector_store import get_embedder
            self.embedder = get_embedder()
            ids = list(self.docs)
            texts = [f"{self.docs[i].get('title') or ''}. {self.docs[i].get('abstract') or ''}"[:2000] for i in ids]
            import numpy as np
            vecs = []
            from retrieval_cache import cached_embed
            for j in range(0, len(texts), 64):
                vecs.append(cached_embed(self.embedder, texts[j:j + 64]))
            self._mem_ids, self._mem_vecs = ids, np.vstack(vecs) if vecs else None
        except Exception:
            self._mem_vecs = None

    def _doc(self, did: str) -> Dict[str, Any]:
        if did in self.docs:
            return self.docs[did]
        if self.db is not None:
            p = self.db.get_paper(int(did))
            if p:
                return {"id": did, "title": p.get("title"), "abstract": p.get("abstract")}
        return {"id": did}

    def _vector_ranking(self, question: str, n: int) -> List[str]:
        if self._mem_vecs is not None:
            q = self.embedder.embed([question])[0]
            sims = self._mem_vecs @ q
            order = sims.argsort()[::-1][:n]
            return [self._mem_ids[i] for i in order]
        if self.vs is not None:
            try:
                return [str((h.get("metadata") or {}).get("paper_id") or h["id"]) for h in self.vs.query(question, k=n)]
            except Exception:
                return []
        return []

    def search(self, question: str, k: int = 20, n_candidates: int = 100, rerank: bool = True) -> List[Dict[str, Any]]:
        # v25: results over the run database are cached until the index changes (retrieval_cache.index_version)
        version = None
        if self.db is not None:
            try:
                import retrieval_cache as rc
                version = rc.index_version(self.db, self.vs)
                hit = rc.get_results(question, "rerank" if rerank else "fused", k, version)
                if hit is not None:
                    return hit
            except Exception:
                version = None
        res = self._search(question, k, n_candidates, rerank)
        if version:
            try:
                import retrieval_cache as rc
                rc.put_results(question, "rerank" if rerank else "fused", k, version, res)
            except Exception:
                pass
        return res

    def _search(self, question: str, k: int, n_candidates: int, rerank: bool) -> List[Dict[str, Any]]:
        import query_parser as qp
        parsed = qp.parse(question)
        concepts = qp.search_concepts(parsed)
        groups = [set(c["terms"]) for c in concepts] or None
        bm = self.kw.search(question, k=n_candidates, concepts=groups)
        bm_ids = [str(h.get("id", h.get("paper_id"))) for h in bm]
        vec_ids = self._vector_ranking(question, n_candidates)
        fused = rrf([bm_ids, vec_ids] if vec_ids else [bm_ids])
        order = sorted(fused, key=lambda d: -fused[d])[:max(k * 3, 30)]
        top = fused[order[0]] if order else 1.0
        res = []
        for d in order:
            doc = self._doc(d)
            ps, roles = pico_score(doc.get("title") or "", doc.get("abstract") or "", concepts, parsed.get("preferred_designs") or [])
            res.append({"id": d, "fused": fused[d] / top, "pico": ps, "matched": roles,
                        "bm25_rank": bm_ids.index(d) + 1 if d in bm_ids else None,
                        "vector_rank": vec_ids.index(d) + 1 if d in vec_ids else None,
                        "title": doc.get("title")})
        if rerank and res:
            cross = _cross_scores(question, [f"{r['title'] or ''}. {(self._doc(r['id']).get('abstract') or '')[:1500]}" for r in res])
            for i, r in enumerate(res):
                r["score"] = (0.4 * r["fused"] + 0.6 * cross[i]) if cross else (0.45 * r["fused"] + 0.55 * r["pico"])
                r["reranker"] = "cross-encoder" if cross else "pico"
            res.sort(key=lambda r: -r["score"])
        else:
            for r in res:
                r["score"] = r["fused"]
        return res[:k]


def rerank_records(question: str, records: Sequence[Any], semantic: Sequence[float]) -> List[Dict[str, float]]:
    """Scores for the PRISMA screening step: BM25 (with synonyms) over the candidate pool, the existing semantic
    score and PICO coverage. Returns [{'bm25_norm', 'pico', 'matched'}] in the records' order."""
    import query_parser as qp
    from keyword_index import MemoryIndex
    parsed = qp.parse(question)
    concepts = qp.search_concepts(parsed)
    docs = [{"id": str(i), "title": getattr(c, "title", "") or "", "abstract": getattr(c, "abstract", "") or ""}
            for i, c in enumerate(records)]
    hits = MemoryIndex(docs).search(question, k=len(docs) or 1, concepts=[set(c["terms"]) for c in concepts] or None)
    best = max((h["bm25"] for h in hits), default=0.0)
    bm = {h["id"]: (h["bm25"] / best if best > 0 else 0.0) for h in hits}
    out = []
    for i, d in enumerate(docs):
        ps, roles = pico_score(d["title"], d["abstract"], concepts, parsed.get("preferred_designs") or [])
        out.append({"bm25_norm": round(bm.get(d["id"], 0.0), 4), "pico": round(ps, 4), "matched": roles})
    return out


def evaluate(questions: Sequence[Dict[str, Any]], corpus: Sequence[Dict[str, Any]], ks: Sequence[int] = (1, 5, 10, 20),
             modes: Sequence[str] = ("bm25", "vector", "hybrid", "hybrid+rerank")) -> Dict[str, Any]:
    """recall@k and MRR per retrieval mode. Each question: {'question', 'relevant': [ids]}."""
    out: Dict[str, Any] = {"questions": len(questions), "corpus": len(corpus), "modes": {}}
    r_vec = HybridRetriever(docs=corpus, use_vectors=True)
    for mode in modes:
        hits = {k: 0 for k in ks}
        rr = 0.0
        for q in questions:
            rel = set(q.get("relevant") or [])
            if mode == "bm25":
                import query_parser as qp
                concepts = qp.search_concepts(qp.parse(q["question"]))
                ranked = [h["id"] for h in r_vec.kw.search(q["question"], k=max(ks), concepts=[set(c["terms"]) for c in concepts] or None)]
            elif mode == "vector":
                ranked = r_vec._vector_ranking(q["question"], max(ks))
            elif mode == "hybrid":
                ranked = [h["id"] for h in r_vec.search(q["question"], k=max(ks), rerank=False)]
            else:
                ranked = [h["id"] for h in r_vec.search(q["question"], k=max(ks), rerank=True)]
            for k in ks:
                hits[k] += bool(rel & set(ranked[:k]))
            first = next((i for i, d in enumerate(ranked, 1) if d in rel), None)
            rr += 1.0 / first if first else 0.0
        n = len(questions) or 1
        out["modes"][mode] = {**{f"recall@{k}": round(hits[k] / n, 3) for k in ks}, "mrr": round(rr / n, 3)}
    try:
        from vector_store import get_embedder
        out["embedder"] = type(get_embedder()).__name__
    except Exception:
        out["embedder"] = None
    out["reranker"] = "cross-encoder" if _Cross.get() else "pico"
    return out
