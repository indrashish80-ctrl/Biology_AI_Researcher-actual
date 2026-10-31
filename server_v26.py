"""
server_v26.py
=============
The v26 web API (included by server.py). Local: no AI key is needed except where noted.

    GET  /api/literature/status                    the local literature library: records, full texts, pre-read papers,
                                                   vectors, scope, last update, where it is stored
    POST /api/literature/scope    {mesh, keywords, journals, year_from, year_to, pub_types}
    POST /api/literature/update   {max_files}      run the daily update now (background thread; PubMed update files,
                                                   embeddings, statuses)
    GET  /api/literature/update                    progress / result of the last update started here
    POST /api/literature/load     {kind, path}     load files already on this computer (pubmed | pmc | trials)
    GET  /api/literature/search?q=&k=              local-first search (BM25 + vectors + PICO rerank) and the
                                                   online-or-not decision for that question
    GET  /api/literature/record/{id}               one record with its version history and passages
    POST /api/papers              multipart file + collection   add YOUR paper (PDF / XML / TXT / MD), kept separate
    GET  /api/papers?collection=                   your papers;  GET /api/papers/collections
    DELETE /api/papers/{id}                        delete one of your papers (the copy is removed)
    POST /api/papers/{id}/share                    move one of your papers into the shared literature (logged)
    POST /api/lab/agent           {goal, observations?}         queue the autonomous experiment loop (capped process)
    POST /api/lab/agent/propose   {hypothesis, keys}            the AI proposes a goal; the validator checks it
    POST /api/lab/uncertainty     {model, observable, ...}      queue a full uncertainty report
    GET  /api/calibration                          the confidence calibration (fit set, held-out ECE, reliability)
"""
from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent
router = APIRouter()
_UPDATE: Dict[str, Any] = {"running": False}
_UPDATE_LOCK = threading.Lock()


class ScopeReq(BaseModel):
    mesh: List[str] = []
    keywords: List[str] = []
    journals: List[str] = []
    year_from: Optional[int] = Field(default=None, ge=1800, le=2100)
    year_to: Optional[int] = Field(default=None, ge=1800, le=2100)
    pub_types: List[str] = []


class UpdateReq(BaseModel):
    max_files: int = Field(default=0, ge=0, le=2000)


class LoadReq(BaseModel):
    kind: str = Field(pattern="^(pubmed|pmc|trials)$")
    path: str = Field(min_length=1, max_length=1000)


class AgentReq(BaseModel):
    goal: Dict[str, Any]
    observations: Optional[Dict[str, float]] = None


class AgentProposeReq(BaseModel):
    hypothesis: str = Field(min_length=8, max_length=1000)
    keys: Optional[Dict[str, Optional[str]]] = None


class UncertaintyReq(BaseModel):
    model: str
    observable: str
    metric: str = "max"
    parameters: Dict[str, float] = {}
    t_end: float = Field(default=100, gt=0)
    uncertain: Optional[Dict[str, float]] = None
    n_mc: int = Field(default=200, ge=10, le=500)
    data: Optional[Dict[str, Any]] = None
    fit: Optional[List[str]] = None
    seed: int = 1


@router.get("/api/literature/status")
def literature_status():
    import library_db as L
    if not L.exists():
        d = Path(os.getenv("BIOSYN_LIBRARY_DIR") or ROOT / "library")
        return {"exists": False, "where": str(d),
                "how_to_start": "python library_loader.py scope --mesh \"...\" ; python library_loader.py download "
                                "pubmed-baseline --max-files 2 ; python library_loader.py embed"}
    with L.Library() as lib:
        st = lib.stats()
    try:
        import library_vectors as LV
        v = LV.LibraryVectors(Path(st["path"]).parent / "vectors")
        st["vectors"] = v.info()
        v.close()
    except Exception as exc:
        st["vectors"] = {"error": str(exc)[:200]}
    import library_search as ls
    return {"exists": True, "where": str(Path(st["path"]).parent), "policy": ls.policy(), **st}


@router.post("/api/literature/scope")
def literature_scope(req: ScopeReq):
    import library_db as L
    from library_loader import Scope
    sc = Scope(req.mesh, req.keywords, req.journals, req.year_from, req.year_to, req.pub_types)
    with L.Library() as lib:
        lib.set_sync("scope", sc.as_dict())
    return sc.as_dict()


@router.post("/api/literature/update")
def literature_update(req: UpdateReq):
    with _UPDATE_LOCK:
        if _UPDATE.get("running"):
            return {**_UPDATE, "note": "an update is already running"}
        _UPDATE.clear()
        _UPDATE.update({"running": True, "started": time.time()})

    def work():
        try:
            import library_loader as ll
            res = ll.update(fetch=not os.getenv("BIOSYN_OFFLINE"), max_files=req.max_files, log=lambda *_: None)
            _UPDATE.update({"result": json.loads(json.dumps(res, default=str))})
        except Exception as exc:
            _UPDATE.update({"error": f"{type(exc).__name__}: {str(exc)[:300]}"})
        finally:
            _UPDATE.update({"running": False, "finished": time.time()})
    threading.Thread(target=work, name="library-update", daemon=True).start()
    return dict(_UPDATE)


@router.get("/api/literature/update")
def literature_update_status():
    return dict(_UPDATE)


@router.post("/api/literature/load")
def literature_load(req: LoadReq):
    import library_db as L
    import library_loader as ll
    p = Path(req.path).expanduser()
    if not p.exists():
        raise HTTPException(404, f"not found on this computer: {p}")
    with L.Library() as lib:
        if req.kind == "pubmed":
            files = sorted(p.glob("*.xml*")) if p.is_dir() else [p]
            return ll.load_pubmed(lib, files, log=lambda *_: None)
        if req.kind == "pmc":
            return ll.load_jats(lib, [p], log=lambda *_: None)
        studies = json.loads(p.read_text(encoding="utf-8"))
        if isinstance(studies, dict):
            studies = studies.get("studies") or [studies]
        return ll.load_trials(lib, studies)


@router.get("/api/literature/search")
def literature_search(q: str, k: int = 20):
    import library_db as L
    import library_search as ls
    if len(q.strip()) < 3:
        raise HTTPException(400, "type a question")
    if not L.exists():
        return {"results": [], "note": "no local library yet"}
    with L.Library() as lib:
        hits = ls.search(lib, q, k=max(1, min(k, 100)))
        dec = ls.decide(lib, q, hits)
    keep = ("id", "title", "year", "journal", "pmid", "doi", "pmcid", "nct", "status", "score", "pico", "matched",
            "published_date", "indexed_date", "revised_date", "licence", "fulltext_kind")
    return {"results": [{x: h.get(x) for x in keep} for h in hits], "decision": dec}


@router.get("/api/literature/record/{record_id}")
def literature_record(record_id: int):
    import library_db as L
    with L.Library() as lib:
        r = lib.get(record_id)
        if not r:
            raise HTTPException(404, "record not found")
        r = {k: v for k, v in r.items() if not k.endswith("_json")}
        return {**r, "history": lib.history(record_id), "passages": lib.passages(record_id)[:60]}


@router.post("/api/papers")
async def papers_add(file: UploadFile = File(...), collection: str = Form("my-papers")):
    import user_papers as up
    data = await file.read()
    try:
        return up.add(data, file.filename or "upload", collection)
    except ValueError as exc:
        raise HTTPException(400, str(exc))


@router.get("/api/papers")
def papers_list(collection: Optional[str] = None):
    import user_papers as up
    return {"papers": up.list_papers(collection), "collections": up.collections()}


@router.get("/api/papers/collections")
def papers_collections():
    import user_papers as up
    return {"collections": up.collections()}


@router.delete("/api/papers/{record_id}")
def papers_delete(record_id: int):
    import user_papers as up
    if not up.delete(record_id):
        raise HTTPException(404, "not one of your papers")
    return {"deleted": record_id}


@router.post("/api/papers/{record_id}/share")
def papers_share(record_id: int):
    import user_papers as up
    target = up.share(record_id)
    if target is None:
        raise HTTPException(404, "not one of your papers")
    return {"shared_as": target}


@router.post("/api/lab/agent")
def lab_agent(req: AgentReq):
    import job_queue as jq
    import simulator
    if not simulator.backend()["available"]:
        raise HTTPException(503, "the simulator needs Tellurium: pip install tellurium")
    q = jq.default_queue()
    jid = q.submit({"agent": req.goal, "observations": req.observations})
    return q.status(jid)


@router.post("/api/lab/agent/propose")
def lab_agent_propose(req: AgentProposeReq):
    import lab_agent as la
    import simulator
    keys = {k: (v or "").strip() or None for k, v in (req.keys or {}).items() if k in ("groq", "gemini", "nvidia")}
    r = None
    if any(keys.values()):
        from ai_router import AIRouter
        r = AIRouter(keys=keys)
    return la.propose_goal(req.hypothesis, simulator.ModelLibrary(), r)


@router.post("/api/lab/uncertainty")
def lab_uncertainty(req: UncertaintyReq):
    import job_queue as jq
    import simulator
    if not simulator.backend()["available"]:
        raise HTTPException(503, "the simulator needs Tellurium: pip install tellurium")
    spec = req.model_dump()
    try:                                          # a wrong model / observable is a 400 now, not a failed job later
        card = simulator.ModelLibrary().get(spec.get("model_id") or spec.get("model") or "")
        obs = spec.get("observable")
        if obs and obs not in card["observables"]:
            raise ValueError(f"no observable '{obs}' (choose from {', '.join(card['observables'])})")
    except (KeyError, ValueError) as exc:
        raise HTTPException(400, str(exc).strip('"'))
    q = jq.default_queue()
    jid = q.submit({"uncertainty": spec})
    return q.status(jid)


class VerifyReq(BaseModel):
    claim: str = Field(min_length=8, max_length=1000)
    spec: Optional[Dict[str, Any]] = None              # a hand-written experiment (no AI needed)
    keys: Optional[Dict[str, Optional[str]]] = None


@router.post("/api/lab/verify")
def lab_verify(req: VerifyReq):
    """v27: test a claim in BioSynthetica's OWN laboratory. The AI (or, without a key, the rule-based planner) designs
    experiments - genes, proteins, pathogens, drugs, pathways or a purpose-built model - and writes its predictions down
    BEFORE they run; the code runs them, repeats them over plausible biology and judges them. A hand-written `spec` for
    the documented model library is still accepted."""
    keys = {k: (v or "").strip() or None for k, v in (req.keys or {}).items() if k in ("groq", "gemini", "nvidia")}
    router_ = None
    if any(keys.values()):
        from ai_router import AIRouter
        router_ = AIRouter(keys=keys)
    try:
        if req.spec:
            import sim_verifier as cv
            return cv.verify(req.claim, router=router_, spec=req.spec, time_budget_s=60)
        import ai_lab
        return ai_lab.investigate(req.claim, router=router_, time_budget_s=90)
    except Exception as exc:
        raise HTTPException(500, f"the laboratory failed: {type(exc).__name__}: {str(exc)[:200]}")


class ToolReq(BaseModel):
    tool: str
    args: Dict[str, Any] = {}


@router.get("/api/biolab/tools")
def biolab_tools():
    """v27: the laboratory's tools (genes, proteins, pathogens, drugs, pathways, custom models) and their arguments."""
    import biolab
    return {"tools": biolab.catalogue(), "version": biolab.BIOLAB_VERSION}


@router.post("/api/biolab/run")
def biolab_run(req: ToolReq):
    """v27: run one laboratory tool directly (no AI). Bad input is a 400 with the reason."""
    import biolab
    import ode_engine
    try:
        return biolab.run_tool(req.tool, req.args)
    except (ValueError, ode_engine.ModelError, KeyError) as exc:
        raise HTTPException(400, str(exc).strip('"')[:400])


@router.get("/api/calibration")
def calibration():
    p = ROOT / "gold" / "calibration.json"
    if not p.exists():
        raise HTTPException(404, "no calibration yet - run: python confidence_calibration.py")
    d = json.loads(p.read_text(encoding="utf-8"))
    d.pop("table", None)
    return d
