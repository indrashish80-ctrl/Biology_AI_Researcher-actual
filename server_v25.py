"""
server_v25.py
=============
The v25 web API (included by server.py). Every endpoint is local: no AI key is needed except /api/lab/propose.

    GET  /api/v25/info                       versions, simulator backend, caches, archive, gold benchmark summary
    POST /api/query/parse        {q}         PICO + question type + the boolean query that will be searched
    GET  /api/search/hybrid?q=&k=            BM25 + vectors + rerank over the papers already stored (library)
    GET  /api/review?status=open             the review queue (findings waiting for a person)
    POST /api/review/{id}        {status, reviewer, correction, notes}
    POST /api/review/export                  approved / corrected items -> gold/papers/review_*.json drafts
    POST /api/status/recheck     {paper_ids?, older_than_days}   re-check retractions / errata / concerns
    GET  /api/memory/impact                  papers added / changed since the last snapshot and the dossiers they affect
    GET  /api/graph/{rq_id}                  knowledge-graph edges of a research question
    POST /api/dossier/diff       {a, b}      compare two finished jobs (papers, findings, verdict)
    GET  /api/metrics/trend                  speed / cost against accuracy, per code version
    GET  /api/gold/benchmark                 the last gold-standard benchmark (gold/benchmark_results.json)
    POST /api/ingest/watch       {query, interval_hours}         add a topic re-searched every day
    POST /api/ingest/run                     run every due ingestion job now
    GET  /api/ingest/status                  queue counts and watched topics
    GET  /api/lab/models                     the model library (assumptions, ranges, limits)
    POST /api/lab/validate       {spec}      validate an experiment spec (nothing runs)
    POST /api/lab/simulate       {spec}      validate, then queue the job (resource-limited); returns a job id
    GET  /api/lab/sim/{job_id}               job status / result
    POST /api/lab/propose        {hypothesis, keys}   the AI proposes a spec, the validator checks it (nothing runs)
    POST /api/lab/validate-model {model, data, fit}   fit to training points, test on held-out points
    GET  /api/lab/experiments                the experiment log (including failed and rejected runs)
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent
router = APIRouter()


def _db():
    from database import ResearchDB
    return ResearchDB()


class Q(BaseModel):
    q: str = Field(min_length=2, max_length=600)


class Decision(BaseModel):
    status: str = Field(pattern="^(approved|corrected|rejected)$")
    reviewer: str = ""
    correction: Optional[Dict[str, Any]] = None
    notes: str = ""


class Recheck(BaseModel):
    paper_ids: Optional[List[int]] = None
    older_than_days: int = Field(default=30, ge=0, le=3650)
    limit: int = Field(default=200, ge=1, le=2000)


class DiffReq(BaseModel):
    a: str
    b: str


class Watch(BaseModel):
    query: str = Field(min_length=3, max_length=400)
    interval_hours: float = Field(default=24, ge=1, le=24 * 30)


class SpecReq(BaseModel):
    spec: Dict[str, Any]


class ProposeReq(BaseModel):
    hypothesis: str = Field(min_length=8, max_length=1000)
    keys: Optional[Dict[str, Optional[str]]] = None


class ValidateModelReq(BaseModel):
    model: str
    data: Dict[str, Any]
    fit: List[str]
    holdout: float = Field(default=0.3, gt=0, lt=0.9)
    seed: int = 1


@router.get("/api/v25/info")
def info():
    out: Dict[str, Any] = {"version": "25.0"}
    try:
        import gold_bench
        out["code_versions"] = gold_bench.versions()
    except Exception as exc:
        out["code_versions"] = {"error": str(exc)}
    try:
        import simulator
        out["simulator"] = simulator.backend()
        out["models"] = sorted(simulator.ModelLibrary().cards)
    except Exception as exc:
        out["simulator"] = {"available": False, "error": str(exc)}
    for name, fn in (("extraction_cache", lambda: __import__("extraction_cache").stats()),
                     ("retrieval_cache", lambda: __import__("retrieval_cache").stats()),
                     ("archive", lambda: __import__("fulltext_archive").summary()),
                     ("synonyms", lambda: __import__("mesh_synonyms").info())):
        try:
            out[name] = fn()
        except Exception as exc:
            out[name] = {"error": str(exc)[:200]}
    p = ROOT / "gold" / "benchmark_results.json"
    if p.exists():
        try:
            b = json.loads(p.read_text(encoding="utf-8"))
            out["gold_benchmark"] = {"when": b.get("when"), "direction_accuracy_dev": (b.get("A") or {}).get("accuracy"),
                                     "direction_accuracy_heldout": (b.get("A_heldout") or {}).get("accuracy"),
                                     "retrieval": (b.get("C") or {}).get("modes")}
        except Exception:
            pass
    return out


@router.post("/api/query/parse")
def query_parse(req: Q):
    import query_parser as qp
    p = qp.parse(req.q)
    return {**p, "boolean_query": qp.boolean_query(p), "concepts": qp.search_concepts(p)}


@router.get("/api/search/hybrid")
def search_hybrid(q: str, k: int = 15):
    import hybrid_retrieval as hr
    if len(q.strip()) < 2:
        raise HTTPException(400, "type a question")
    with _db() as db:
        n = db._conn.execute("SELECT COUNT(*) FROM papers").fetchone()[0]
        if not n:
            return {"results": [], "note": "the library is empty - run a research question first"}
        res = hr.HybridRetriever(db=db, use_vectors=False).search(q, k=max(1, min(k, 50)))
    return {"results": res, "papers": n}


@router.get("/api/review")
def review_list(status: Optional[str] = "open", limit: int = 100):
    import review_store as rs
    with _db() as db:
        return {"items": rs.list_items(db, None if status in (None, "", "all") else status, max(1, min(limit, 1000))),
                "summary": rs.summary(db)}


@router.post("/api/review/export")
def review_export():
    # declared BEFORE /api/review/{item_id}: otherwise "export" is parsed as an item id and the call fails with 422
    import review_store as rs
    with _db() as db:
        return {"written": rs.export_gold_drafts(db)}


@router.post("/api/review/{item_id}")
def review_decide(item_id: int, d: Decision):
    import review_store as rs
    with _db() as db:
        try:
            return rs.decide(db, item_id, d.status, d.reviewer, d.correction, d.notes)
        except KeyError:
            raise HTTPException(404, f"review item {item_id} not found")
        except ValueError as exc:
            raise HTTPException(400, str(exc))


@router.post("/api/status/recheck")
def status_recheck(req: Recheck):
    import publication_status as ps
    with _db() as db:
        return ps.recheck(db, req.older_than_days, fetch=not os.getenv("BIOSYN_OFFLINE"), limit=req.limit,
                          paper_ids=req.paper_ids)


@router.get("/api/memory/impact")
def memory_impact():
    import scientific_memory as sm
    with _db() as db:
        return sm.impact(db)


@router.get("/api/graph/{rq_id}")
def graph(rq_id: int):
    import knowledge_graph as kg
    with _db() as db:
        e = kg.edges_for(db, rq_id)
        return {"edges": e, "summary": kg.summary(e)}


@router.post("/api/dossier/diff")
def dossier_diff(req: DiffReq):
    import re
    import sys
    import dossier_repro as dr
    # the same folder the server writes jobs to (server.RUNS), not a second hard-coded path
    runs = Path(getattr(sys.modules.get("server"), "RUNS", ROOT / "runs"))

    def load(j):
        j = re.sub(r"[^a-f0-9]", "", j)
        p = runs / j / "dossier.json"
        if not p.exists():
            raise HTTPException(404, f"job {j} not found")
        d = json.loads(p.read_text(encoding="utf-8"))
        return d.get("dossier") or d
    d = dr.diff(load(req.a), load(req.b))
    return {**d, "markdown": dr.markdown(d)}


@router.get("/api/metrics/trend")
def metrics_trend(limit: int = 20):
    import run_metrics as rm
    return {"trend": rm.trend(max(1, min(limit, 200)))}


@router.get("/api/gold/benchmark")
def gold_benchmark():
    p = ROOT / "gold" / "benchmark_results.json"
    if not p.exists():
        raise HTTPException(404, "no benchmark yet - run: python gold_bench.py")
    return json.loads(p.read_text(encoding="utf-8"))


@router.post("/api/ingest/watch")
def ingest_watch(req: Watch):
    import ingestion_queue as iq
    with _db() as db:
        q = iq.IngestionQueue(db)
        return {"topic_id": q.watch(req.query, req.interval_hours), "counts": q.counts()}


@router.post("/api/ingest/run")
def ingest_run():
    import ingestion_queue as iq
    with _db() as db:
        return iq.run_due(db)


@router.get("/api/ingest/status")
def ingest_status():
    import ingestion_queue as iq
    with _db() as db:
        q = iq.IngestionQueue(db)
        topics = [dict(r) for r in db._conn.execute("SELECT * FROM watch_topics").fetchall()]
        return {"counts": q.counts(), "topics": topics}


@router.get("/api/lab/models")
def lab_models():
    import simulator
    lib = simulator.ModelLibrary()
    return {"backend": simulator.backend(), "models": lib.catalogue(), "problems": lib.problems}


@router.post("/api/lab/validate")
def lab_validate(req: SpecReq):
    import experiment_spec as es
    import simulator
    return es.validate(req.spec, simulator.ModelLibrary())  # a malformed spec is reported as errors, never a 500


@router.post("/api/lab/simulate")
def lab_simulate(req: SpecReq):
    import job_queue as jq
    import simulator
    if not simulator.backend()["available"]:
        raise HTTPException(503, "the simulator needs Tellurium: pip install tellurium")
    q = jq.default_queue()
    jid = q.submit(req.spec)
    return q.status(jid)


@router.get("/api/lab/sim/{job_id}")
def lab_sim(job_id: str, wait: float = 0):
    import job_queue as jq
    q = jq.default_queue()
    if q.status(job_id)["status"] == "unknown":
        raise HTTPException(404, f"simulation job {job_id} not found")
    res = q.result(job_id, wait=max(0.0, min(wait, 30.0)))
    return {"status": q.status(job_id), "result": res}


@router.post("/api/lab/propose")
def lab_propose(req: ProposeReq):
    import experiment_spec as es
    import simulator
    lib = simulator.ModelLibrary()
    router_ = None
    keys = {k: (v or "").strip() or None for k, v in (req.keys or {}).items() if k in ("groq", "gemini", "nvidia")}
    if any(keys.values()):
        from ai_router import AIRouter
        router_ = AIRouter(keys=keys)
    return es.propose(req.hypothesis, lib, router_)


@router.post("/api/lab/validate-model")
def lab_validate_model(req: ValidateModelReq):
    import model_validation as mv
    import simulator
    lib = simulator.ModelLibrary()
    try:
        sim = simulator.Simulator(lib.get(req.model))
        return mv.validate(sim, req.data, req.fit, holdout=req.holdout, seed=req.seed)
    except (KeyError, ValueError) as exc:
        raise HTTPException(400, str(exc))


@router.get("/api/lab/experiments")
def lab_experiments(limit: int = 50, status: Optional[str] = None):
    import experiment_log as el
    return {"experiments": el.list_runs(max(1, min(limit, 500)), status), "stats": el.stats()}
