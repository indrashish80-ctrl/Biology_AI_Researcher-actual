"""
server.py
---------
BioSynthetica web server (FastAPI + static TypeScript front-end).
Equipped with configurable concurrent job execution, thread-safe memory isolation,
child-thread log capture, and non-blocking SSE event streaming.
"""

from __future__ import annotations

import asyncio
import contextvars
import io
import json
import os
import queue
import re
import sys
import threading
import time
import traceback
import types
import uuid
from contextlib import asynccontextmanager
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)

if not (ROOT / "retrieval" / "__init__.py").exists() and "retrieval" not in sys.modules:
    _pkg = types.ModuleType("retrieval")
    _pkg.__path__ = [str(ROOT)]
    sys.modules["retrieval"] = _pkg

STATIC = ROOT / "static"
STATIC.mkdir(parents=True, exist_ok=True)  # <--- ADD THIS LINE!

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, PlainTextResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator

from ai_router import AIRouter
from database import ResearchDB

RUNS = Path(os.getenv("BIOSYN_RUNS_DIR") or ROOT / "runs")      # BIOSYN_RUNS_DIR: tests use a temporary folder
UPLOADS = RUNS / "uploads"
UPLOADS.mkdir(parents=True, exist_ok=True)
STATIC = ROOT / "static"
ALLOW_SERVER_KEYS = os.getenv("ALLOW_SERVER_KEYS", "0") == "1"
MAX_JOBS_KEPT = 50
MAX_CONCURRENT_JOBS = max(1, int(os.getenv("BIOSYN_MAX_CONCURRENT_JOBS", "2")))

STAGES = ["planning", "screening", "retrieval", "extraction", "contradictions", "pathways", "hypotheses",
          "evidence_matrix", "protein_interactions", "deep_reasoning", "report", "agent", "done"]

_KEY_PATTERNS = [
    re.compile(r"gsk_[A-Za-z0-9]{8,}"),
    re.compile(r"nvapi-[A-Za-z0-9_\-]{8,}"),
    re.compile(r"AIza[0-9A-Za-z_\-]{20,}"),
    re.compile(r"sk-[A-Za-z0-9_\-]{16,}"),
]

# Context variable automatically propagated across ThreadPoolExecutor child threads
_JOB_SINK: contextvars.ContextVar[Optional[Any]] = contextvars.ContextVar("_JOB_SINK", default=None)


def redact(text: str, secrets: List[str]) -> str:
    for s in secrets:
        if s and len(s) >= 8:
            text = text.replace(s, "[redacted-key]")
            text = text.replace(s[:12], "[redacted-key]")
    for pat in _KEY_PATTERNS:
        text = pat.sub("[redacted-key]", text)
    return text


class _ThreadStdout(io.TextIOBase):
    def __init__(self, real):
        self.real = real
        self.local = threading.local()
        self._sinks: Dict[int, Any] = {}
        self._child_map: Dict[int, int] = {}
        self._lock = threading.Lock()

    def attach(self, sink):
        tid = threading.get_ident()
        with self._lock:
            self._sinks[tid] = sink
        self.local.sink = sink
        self.local.buf = ""
        _JOB_SINK.set(sink)

    def attach_child(self, parent_tid: int):
        """Map a child worker thread to its parent execution thread sink."""
        child_tid = threading.get_ident()
        with self._lock:
            sink = self._sinks.get(parent_tid)
            if sink is not None:
                self._sinks[child_tid] = sink
                self._child_map[child_tid] = parent_tid
        if sink is not None:
            self.local.sink = sink
            self.local.buf = ""
            _JOB_SINK.set(sink)

    def detach(self):
        tid = threading.get_ident()
        buf = getattr(self.local, "buf", "")
        sink = getattr(self.local, "sink", None)
        if sink and buf.strip():
            sink(buf)
        self.local.sink = None
        self.local.buf = ""
        with self._lock:
            self._sinks.pop(tid, None)
            for c_tid, p_tid in list(self._child_map.items()):
                if p_tid == tid or c_tid == tid:
                    self._child_map.pop(c_tid, None)
                    self._sinks.pop(c_tid, None)
        _JOB_SINK.set(None)

    def write(self, s):
        # 1. Context variable inherited by ThreadPoolExecutor workers
        sink = _JOB_SINK.get()

        # 2. Thread-local sink
        if sink is None:
            sink = getattr(self.local, "sink", None)

        # 3. Explicit child / parent thread ID registry
        if sink is None:
            tid = threading.get_ident()
            with self._lock:
                sink = self._sinks.get(tid)

        # 4. Fallback for spawned sub-threads when an active job is executing
        if sink is None and threading.current_thread() is not threading.main_thread():
            with self._lock:
                if len(self._sinks) == 1:
                    sink = next(iter(self._sinks.values()))

        if sink is None:
            return self.real.write(s)

        buf = getattr(self.local, "buf", "") + s
        while "\n" in buf:
            line, buf = buf.split("\n", 1)
            sink(line)
        self.local.buf = buf
        return len(s)

    def flush(self):
        self.real.flush()


class _NonBlockingConsole(io.TextIOBase):
    def __init__(self, real, maxlen: int = 5000):
        self.real = real
        self.q: "queue.Queue[str]" = queue.Queue(maxsize=maxlen)
        self.dropped = 0
        threading.Thread(target=self._pump, name="console-writer", daemon=True).start()
        import atexit
        atexit.register(self.drain)

    def drain(self, timeout: float = 3.0) -> None:
        end = time.time() + timeout
        while self.q.unfinished_tasks and time.time() < end:
            time.sleep(0.01)
        try:
            self.real.flush()
        except Exception:
            pass

    def _pump(self):
        while True:
            s = self.q.get()
            try:
                self.real.write(s)
                self.real.flush()
            except Exception:
                pass
            finally:
                self.q.task_done()

    def write(self, s):
        try:
            self.q.put_nowait(s)
        except queue.Full:
            self.dropped += 1
        return len(s)

    def flush(self):
        pass

    def isatty(self):
        try:
            return self.real.isatty()
        except Exception:
            return False

    @property
    def encoding(self):
        return getattr(self.real, "encoding", "utf-8")


def _windows_console_guard() -> None:
    if os.name != "nt":
        return
    try:
        import ctypes
        k32 = ctypes.windll.kernel32
        h = k32.GetStdHandle(-10)
        mode = ctypes.c_uint32()
        if k32.GetConsoleMode(h, ctypes.byref(mode)):
            k32.SetConsoleMode(h, (mode.value & ~0x0040) | 0x0080)
    except Exception:
        pass


def keep_awake(on: bool) -> bool:
    """v27: keep the computer from idle-sleeping while any run is going (Windows, macOS, Linux; keep_awake.py)."""
    import keep_awake as _ka
    if on:
        return bool(_ka.hold().get("active"))
    _ka.release()
    return True


_LAST_BEAT = [time.time()]


def _heartbeat() -> None:
    """v27: notice when the computer was asleep. The run is frozen while it sleeps, so that time is credited to each
    running job as 'paused' (the page's clock does not count it), the job log says so, and the AI providers' failure
    window and cool-downs are reset - the first requests after waking often fail while the network comes back."""
    while True:
        time.sleep(2.0)
        now = time.time()
        gap = now - _LAST_BEAT[0]
        _LAST_BEAT[0] = now
        if gap < 20.0:
            continue
        slept = gap - 2.0
        for job in list(JOBS.values()):
            if job.status == "running":
                job.paused += slept
                job.emit("info", text=f"The computer was asleep for {slept / 60:.1f} min - the run was paused and has "
                                      f"continued. That time is not counted on the clock.")
                r = getattr(job, "router", None)
                if r is not None:
                    r.__dict__.pop("_recent", None)
                    r.__dict__.pop("_groq_rest", None)
        try:
            import ai_pacing
            with ai_pacing._LOCK:
                ai_pacing._COOL_UNTIL.clear()     # cool-downs from before the sleep are stale
        except Exception:
            pass


threading.Thread(target=_heartbeat, name="sleep-watch", daemon=True).start()


if os.getenv("BIOSYN_BLOCKING_CONSOLE") != "1":
    sys.stderr = _NonBlockingConsole(sys.stderr)
    _CONSOLE = _NonBlockingConsole(sys.stdout)
else:
    _CONSOLE = sys.stdout
_windows_console_guard()
_STDOUT = _ThreadStdout(_CONSOLE)
sys.stdout = _STDOUT


class Job:
    def __init__(self, req: "ResearchRequest"):
        self.id = uuid.uuid4().hex[:12]
        self.req = req
        self.status = "queued"
        self.stage = "queued"
        self.created = time.time()
        self.started: Optional[float] = None
        self.finished: Optional[float] = None
        self.events: List[Dict[str, Any]] = []
        self.cond = threading.Condition()
        self.result: Optional[Dict[str, Any]] = None
        self.error: Optional[str] = None
        self.dir = RUNS / self.id
        self.secrets = [v for v in (req.keys.model_dump() if req.keys else {}).values() if v]
        self.cancel_event = threading.Event()               # v26: POST /api/jobs/{id}/cancel
        self.paused = 0.0                                   # v27: seconds the computer was asleep during the run
        self.router = None

    def emit(self, kind: str, **data):
        if "text" in data and isinstance(data["text"], str):
            data["text"] = redact(data["text"], self.secrets)
        with self.cond:
            self.events.append({"seq": len(self.events), "kind": kind, "t": round(time.time() - self.created, 2), **data})
            self.cond.notify_all()

    def summary(self) -> Dict[str, Any]:
        return {"id": self.id, "status": self.status, "stage": self.stage, "question": self.req.question,
                "mode": self.req.mode, "created": self.created, "started": self.started, "finished": self.finished,
                "error": self.error, "paused": round(self.paused, 1),
                "active_s": round(max(0.0, ((self.finished or time.time()) - self.started - self.paused)), 1)
                if self.started else 0.0}


JOBS: Dict[str, Job] = {}
ORDER: List[str] = []
EXECUTOR = ThreadPoolExecutor(max_workers=MAX_CONCURRENT_JOBS, thread_name_prefix="research")


class Keys(BaseModel):
    groq: Optional[str] = None
    gemini: Optional[str] = None
    nvidia: Optional[str] = None
    openalex: Optional[str] = None
    semantic_scholar: Optional[str] = None
    ncbi: Optional[str] = None                     # v27: NCBI (PubMed) API key - 10 requests/s instead of 3


ALL_SOURCES = ["europe_pmc", "pubmed", "pmc", "crossref", "openalex", "semantic_scholar", "clinicaltrials"]


class ResearchRequest(BaseModel):
    question: str = Field(min_length=8, max_length=600)
    mode: str = Field(default="prisma", pattern="^(prisma|pipeline|agent)$")
    keys: Optional[Keys] = None
    email: Optional[str] = None
    screen_n: int = Field(default=150, ge=5, le=300)
    deep_read_k: int = Field(default=25, ge=1, le=80)
    structures: bool = True
    max_papers: int = Field(default=4, ge=2, le=12)
    unpaywall: bool = True
    string_db: bool = True
    deep_reasoning: bool = True
    geo: bool = True
    link_bio: bool = True
    agent_steps: int = Field(default=8, ge=3, le=20)
    sources: Optional[List[str]] = None
    citation_chasing: bool = True
    exact: bool = False
    auto_heal: bool = True
    speed: str = Field(default="full", pattern="^(full|quick|research|deep|auto)$")   # v25: three tiers + auto escalation
    reuse: bool = True
    dataset_id: Optional[str] = None
    group_col: Optional[str] = None
    outcome_col: Optional[str] = None
    user_collections: Optional[List[str]] = None       # v26: include your own uploaded papers ('user:<name>')
    # v26.2: check hypotheses with the virtual lab / simulator during the run (also switched on when the question
    # asks to 'verify', 'simulate' or test 'in silico'); `hypothesis` = your own claim to verify
    verify_hypotheses: bool = False
    hypothesis: Optional[str] = Field(default=None, max_length=1000)

    @field_validator("question")
    @classmethod
    def _real_question(cls, v: str) -> str:
        v = re.sub(r"\s+", " ", v).strip()
        if len(v) < 8 or len(re.findall(r"[^\W\d_]{2,}", v)) < 2:
            raise ValueError("please type a research question (at least two words)")
        return v


def _keys_for(req: ResearchRequest) -> Dict[str, Optional[str]]:
    keys = (req.keys.model_dump() if req.keys else {})
    keys = {k: (v or "").strip() or None for k, v in keys.items() if k in ("groq", "gemini", "nvidia")}
    if ALLOW_SERVER_KEYS and not any(keys.values()):
        keys = {"groq": os.getenv("GROQ_API_KEY"), "gemini": os.getenv("GEMINI_API_KEY"),
                "nvidia": os.getenv("NVIDIA_NIM_API_KEY")}
    return keys


QUICK = {"screen_n": 40, "deep_read_k": 8, "unpaywall": False, "citation_chasing": False, "structures": False,
         "geo": False, "link_bio": False, "auto_heal": False}
QUICK_SOURCES = ["europe_pmc", "pubmed", "openalex"]
CACHE_INDEX = RUNS / "cache_index.json"
CACHE_MAX_AGE = 7 * 24 * 3600
_CACHE_LOCK = threading.Lock()


DEEP = {"screen_n": 300, "deep_read_k": 50, "citation_chasing": True, "auto_heal": True}


def apply_speed(req: "ResearchRequest") -> "ResearchRequest":
    if req.speed == "research":
        req.speed = "full"                       # 'research' is the v25 name of the full dossier
    if req.speed == "deep":
        for k, v in DEEP.items():
            setattr(req, k, v)
    if req.speed == "quick":
        for k, v in QUICK.items():
            setattr(req, k, v)
        req.sources = [x for x in (req.sources or ALL_SOURCES) if x in QUICK_SOURCES] or QUICK_SOURCES
    return req


def cache_key(req: "ResearchRequest") -> Optional[str]:
    if req.dataset_id or req.mode == "agent":
        return None
    import hashlib
    q = re.sub(r"[^a-z0-9 ]", "", re.sub(r"\s+", " ", req.question.lower())).strip()
    parts = [q, req.mode, req.speed, req.screen_n, req.deep_read_k, req.structures, req.unpaywall, req.string_db,
             req.deep_reasoning, req.geo, req.link_bio, sorted(req.sources or ALL_SOURCES), req.citation_chasing,
             req.exact, req.auto_heal, req.verify_hypotheses, req.hypothesis]
    return hashlib.sha256(json.dumps(parts, default=str).encode()).hexdigest()[:24]


def _cache_load() -> Dict[str, Any]:
    try:
        return json.loads(CACHE_INDEX.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def cache_lookup(key: Optional[str]) -> Optional[Dict[str, Any]]:
    if not key:
        return None
    with _CACHE_LOCK:
        hit = _cache_load().get(key)
    if not hit or time.time() - hit.get("finished", 0) > CACHE_MAX_AGE:
        return None
    if not (RUNS / hit["job_id"] / "dossier.json").exists():
        return None
    return hit


def cache_store(key: Optional[str], job: "Job") -> None:
    if not key:
        return
    with _CACHE_LOCK:
        idx = _cache_load()
        idx[key] = {"job_id": job.id, "finished": time.time(), "question": job.req.question}
        now = time.time()
        idx = {k: v for k, v in idx.items() if now - v.get("finished", 0) <= CACHE_MAX_AGE}
        try:
            RUNS.mkdir(parents=True, exist_ok=True)
            CACHE_INDEX.write_text(json.dumps(idx, indent=1), encoding="utf-8")
        except OSError:
            pass


def _jsonable(obj: Any) -> Any:
    return json.loads(json.dumps(obj, default=str, ensure_ascii=False))


def _run_job(job: Job) -> None:
    from research_engine import ResearchEngine
    from core_utils import JobCancelled
    from ai_router import AIOutage

    req = job.req
    if job.cancel_event.is_set():
        job.status = job.stage = "cancelled"
        job.error = "cancelled by the user"
        job.finished = time.time()
        job.emit("error", text="Cancelled before it started.")
        return
    job.status, job.started = "running", time.time()
    keep_awake(True)
    job.dir.mkdir(parents=True, exist_ok=True)
    _STDOUT.attach(lambda line: job.emit("log", text=line))
    engine = None
    router = None
    from evidence_identity import exact_spelling
    as_typed = exact_spelling() if req.exact else None
    if as_typed:
        as_typed.__enter__()
    try:
        keys = _keys_for(req)
        import ncbi_throttle as _nt                # v27: the NCBI key entered with the other keys
        _nt.set_api_key((req.keys.ncbi if req.keys else None) or None)
        if _nt.api_key():
            job.emit("info", text="NCBI API key in use: PubMed requests at up to 10 per second")
        if any(keys.values()):
            router = AIRouter(keys=keys)
            router.cancel_event = job.cancel_event
            job.router = router
            job.emit("info", text=f"AI providers for this run: {', '.join(router.configured_providers())}")
            if not os.getenv("BIOSYN_OFFLINE") and not os.getenv("BIOSYN_NO_PREFLIGHT"):
                # v26.2: test each key once BEFORE the run (a few seconds); a provider that fails is switched off
                # for this run with the reason, instead of failing on every paper for an hour
                job.emit("stage", stage="preflight", text="Checking your AI keys (one short test request each)")
                pf = router.preflight()             # raises (run not started) when no provider works
                for name, st in pf.items():
                    job.emit("info", text=f"{name}: ready" if st == "ok" else
                             f"{name}: switched off for this run - {redact(st, job.secrets)}")
        else:
            raise RuntimeError(NO_KEY_MESSAGE)

        def progress(stage: str, message: str):
            if job.cancel_event.is_set():
                from core_utils import JobCancelled
                raise JobCancelled("cancelled by the user")
            job.stage = stage
            job.emit("stage", stage=stage, text=message)

        lit = req.keys.model_dump() if req.keys else {}
        engine = ResearchEngine(router=router, use_ai=router is not None, contact_email=req.email,
                                progress=progress, openalex_key=lit.get("openalex"),
                                s2_key=lit.get("semantic_scholar"))
        report_path = job.dir / "RESEARCH_REPORT.md"
        # v26: queue wait and server load go into the run metrics
        engine.run_context = {"queue_wait_s": round(job.started - job.created, 2), "server_load": _server_load()}
        dataset_path = None
        if req.dataset_id:
            matches = list(UPLOADS.glob(f"{re.sub(r'[^a-f0-9]', '', req.dataset_id)}.*"))
            dataset_path = str(matches[0]) if matches else None

        if req.mode == "agent":
            progress("agent", "Autonomous ReAct agent (search -> batch read -> critic)")
            res = engine.run_agent(req.question)
            rp = ROOT / "RESEARCH_REPORT.md"
            report_md = rp.read_text(encoding="utf-8") if rp.exists() else "No report generated."
            report_path.write_text(report_md, encoding="utf-8")
            result = {"mode": "agent", "final_synthesis": res.get("final_synthesis"), "status": res.get("status"),
                      "report_markdown": report_md}
        else:
            invest = engine.investigate
            import sim_verifier as _cv
            engine._verify = bool(req.verify_hypotheses or req.hypothesis or _cv.wants_verification(req.question))
            engine._user_hypothesis = (req.hypothesis or "").strip() or None
            if engine._verify:
                job.emit("info", text="Hypotheses will be checked in the virtual lab / simulator during this run")
            if req.speed == "auto":
                # v25: start with a quick scan and escalate (research -> deep) while the evidence is thin or conflicting
                def invest(question, **kw):          # noqa: F811
                    # v26.2: ONE continuous run. It starts at the quick level and, when the evidence is thin or
                    # conflicting, widens the search INSIDE the run (research, then deep level) - the papers already
                    # read are kept and the late stages run once. v26.1 re-ran the whole pipeline per level, which
                    # looked like the run resetting near the end.
                    import analysis_tiers as AT
                    for k in ("screen_n", "deep_read_k", "citation_chasing", "use_structures", "auto_heal",
                              "abstract_only", "quick"):
                        kw.pop(k, None)
                    engine._auto_levels, engine._auto_level, engine._auto_history = ["research", "deep"], "quick", []
                    q = AT.kwargs_for("quick", kw)
                    q["auto_heal"] = False            # no restart-style repair round in Auto: it widens in place
                    progress("tier", "Auto: starting at the quick level - the search widens within this run if "
                                     "the evidence is thin or conflicting")
                    d = engine.investigate(question, **q)
                    d["tiers"] = list(engine._auto_history)
                    d["analysis_tier"] = engine._auto_level
                    return d
            dossier = invest(
                question=req.question,
                experimental_dataset=dataset_path,
                exp_group_col=req.group_col if dataset_path else None,
                exp_outcome_col=req.outcome_col if dataset_path else None,
                max_papers=req.max_papers,
                enable_geo=req.geo,
                output_report=str(report_path),
                use_prisma=req.mode == "prisma",
                screen_n=req.screen_n,
                deep_read_k=req.deep_read_k,
                use_unpaywall=req.unpaywall,
                use_string=req.string_db,
                use_structures=req.structures,
                use_deep_reasoning=req.deep_reasoning,
                link_bio=req.link_bio,
                enhanced_report=True,
                sources=[s for s in (req.sources or ALL_SOURCES) if s in ALL_SOURCES] or None,
                citation_chasing=req.citation_chasing,
                auto_heal=req.auto_heal,
                abstract_only=req.speed == "quick",
                quick=req.speed == "quick",
                user_collections=req.user_collections or None,
            )
            report_md = report_path.read_text(encoding="utf-8") if report_path.exists() else ""
            result_bug = (job.dir / "BUG_REPORT.md").read_text(encoding="utf-8") if (job.dir / "BUG_REPORT.md").exists() else None

            if dataset_path:
                shown = DATASET_NAMES.get(re.sub(r"[^a-f0-9]", "", req.dataset_id or ""))
                if shown:
                    report_md = report_md.replace(Path(dataset_path).name, shown)
                    report_path.write_text(report_md, encoding="utf-8")
                    if isinstance(dossier.get("user_experiment"), dict):
                        dossier["user_experiment"]["dataset"] = shown
                    if isinstance((dossier.get("executive_synthesis") or {}).get("user_experiment"), dict):
                        dossier["executive_synthesis"]["user_experiment"]["dataset"] = shown
            result = {"mode": req.mode, "dossier": _jsonable(dossier), "report_markdown": report_md, "bug_report": result_bug}
            if dossier.get("evidence_matrix_csv"):
                (job.dir / "evidence_matrix.csv").write_text(dossier["evidence_matrix_csv"], encoding="utf-8")

        result["providers"] = router.health.all_status() if router else {}
        (job.dir / "dossier.json").write_text(json.dumps(result, indent=2, default=str, ensure_ascii=False),
                                              encoding="utf-8")
        job.result = _jsonable(result)
        job.status, job.stage = "done", "done"
        cache_store(getattr(job, "cache_key", None), job)
        job.emit("done", text="Investigation complete.")
    except AIOutage as exc:
        job.status, job.stage = "error", "error"
        job.error = redact(str(exc), job.secrets)
        job.emit("error", text="The run was stopped early because the AI providers kept failing: " + job.error)
    except JobCancelled:
        job.status, job.stage = "cancelled", "cancelled"
        job.error = "cancelled by the user"
        job.emit("error", text="Cancelled by the user - the partial run was stopped and nothing was saved as a result.")
    except Exception as exc:
        if job.cancel_event.is_set():
            job.status, job.stage, job.error = "cancelled", "cancelled", "cancelled by the user"
            job.emit("error", text="Cancelled by the user.")
            return
        job.status, job.stage = "error", "error"
        job.error = redact(f"{type(exc).__name__}: {exc}", job.secrets)
        for line in traceback.format_exc().splitlines()[-6:]:
            job.emit("log", text=line)
        job.emit("error", text=job.error)
    finally:
        if as_typed:
            as_typed.__exit__(None, None, None)
        _STDOUT.detach()
        keep_awake(False)
        job.finished = time.time()
        if engine is not None:
            try:
                engine.close()
            except Exception:
                pass
        router = None
        job.router = None
        job.req.keys = None
        job.secrets_hint = len(job.secrets)
        job.secrets = []


def _server_load() -> Dict[str, Any]:
    now = time.time()
    js = list(JOBS.values())
    return {"running": sum(1 for j in js if j.status == "running"), "queued": sum(1 for j in js if j.status == "queued"),
            "finished_last_hour": sum(1 for j in js if j.finished and now - j.finished < 3600),
            "max_concurrent": MAX_CONCURRENT_JOBS}


def _nocache_http():
    from http_client import HttpClient
    return HttpClient(cache_dir=None, timeout=15, retries=0)


def _prune():
    while len(ORDER) > MAX_JOBS_KEPT:
        old = ORDER.pop(0)
        JOBS.pop(old, None)


async def _start_warm_up() -> None:
    def _speller():
        try:
            from spelling import vocabulary
            vocabulary()
        except Exception:
            pass
    threading.Thread(target=_speller, name="speller-warm-up", daemon=True).start()
    if not os.getenv("BIOSYN_OFFLINE"):
        try:                                      # v25: watched topics are re-searched daily (only topics you added)
            from ingestion_queue import start_worker
            start_worker()
        except Exception as exc:
            print(f"  [Ingest worker] not started: {exc}")
    if os.getenv("BIOSYN_NO_WARMUP") != "1":
        threading.Thread(target=_warm_up, name="warm-up", daemon=True).start()


@asynccontextmanager
async def _lifespan(_app):
    """Start-up and shut-down (FastAPI lifespan; replaces the deprecated @app.on_event('startup'))."""
    await _start_warm_up()
    try:
        yield
    finally:
        try:                                      # stop the daily ingestion worker cleanly
            from ingestion_queue import stop_worker
            stop_worker()
        except Exception:
            pass


app = FastAPI(title="BioSynthetica", docs_url="/api/docs", redoc_url=None, lifespan=_lifespan)
try:                                              # v25: review queue, hybrid search, status, memory, lab simulator ...
    from server_v25 import router as _v25_router
    app.include_router(_v25_router)
except Exception as _exc:                         # the main app must start even if an optional module is broken
    print(f"  [v25 API] not loaded: {type(_exc).__name__}: {_exc}")
try:                                              # v26: local literature library, your papers, lab agent, calibration
    from server_v26 import router as _v26_router
    app.include_router(_v26_router)
except Exception as _exc:
    print(f"  [v26 API] not loaded: {type(_exc).__name__}: {_exc}")
app.mount("/static", StaticFiles(directory=STATIC), name="static")


def _warm_up() -> None:
    t0 = time.time()
    try:
        import research_engine  # noqa: F401  (imported to warm it up: the first job starts faster)
        from vector_store import get_embedder
        get_embedder()
        print(f"  [Warm-up] Engine and embedding model ready in {time.time() - t0:.1f}s")
        import library_search                 # v26.1: local library index in RAM, so the first search is fast too
        info = library_search.warm()
        if info:
            print(f"  [Warm-up] Local library ready: {info}")
    except Exception as exc:
        print(f"  [Warm-up] skipped: {str(exc)[:160]}")


@app.middleware("http")
async def security_headers(request, call_next):
    resp = await call_next(request)
    resp.headers["Cache-Control"] = "no-store" if request.url.path.startswith("/api") else "no-cache"
    resp.headers["Referrer-Policy"] = "no-referrer"
    resp.headers["X-Content-Type-Options"] = "nosniff"
    return resp


APP_VERSION = "27.0"


@app.get("/api/version")
def api_version():
    """v26.2: the running server's version - the page compares it with its own, so an old copy is noticed."""
    return {"version": APP_VERSION}


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


PAGES = {"research": "research.html", "results": "results.html", "lab": "lab.html", "library": "library.html",
         "review": "review.html", "simulate": "simulate.html",
         "history": "history.html", "keys": "keys.html"}


@app.get("/{page}", include_in_schema=False)
def page(page: str):
    name = PAGES.get(page.removesuffix(".html"))
    if not name or not (STATIC / name).is_file():
        raise HTTPException(404, "Page not found")
    return FileResponse(STATIC / name)


@app.get("/api/health")
def health():
    from vector_store import DEFAULT_MODEL
    env_keys = any(os.getenv(k) for k in ("GROQ_API_KEY", "GEMINI_API_KEY", "NVIDIA_NIM_API_KEY"))
    return {
        "ok": True,
        "server_keys_present": env_keys,
        "server_keys_used": ALLOW_SERVER_KEYS and env_keys,
        "queue": sum(1 for j in JOBS.values() if j.status in ("queued", "running")),
        "embedding_model": DEFAULT_MODEL,
        "concurrency_limit": MAX_CONCURRENT_JOBS,
    }


@app.post("/api/keys/check")
def check_keys(keys: Keys):
    supplied = {k: v.strip() for k, v in keys.model_dump().items() if v and v.strip()}
    out: Dict[str, Any] = {}
    for name in ("openalex", "semantic_scholar"):
        key = supplied.pop(name, None)
        if not key:
            continue
        t0 = time.time()
        try:
            if name == "openalex":
                from retrieval.openalex import OpenAlexClient
                hits = OpenAlexClient(api_key=key, http=_nocache_http()).search("metformin", max_results=1)
            else:
                from retrieval.semantic_scholar import SemanticScholarClient
                c = SemanticScholarClient(api_key=key)
                c.http.cache_dir = None
                hits = c.search("metformin", max_results=1)
            out[name] = ({"ok": True, "latency_ms": int((time.time() - t0) * 1000)} if hits
                         else {"ok": False, "error": "no results (key rejected or service throttled)"})
        except Exception as exc:
            out[name] = {"ok": False, "error": redact(str(exc), [key])[:240]}
    ncbi = supplied.pop("ncbi", None)
    if ncbi:                                       # v27: NCBI answers an invalid key with {"error": "API key invalid"}
        t0 = time.time()
        try:
            d = _nocache_http().get_json("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi",
                                         params={"db": "pubmed", "term": "metformin", "retmax": 1, "retmode": "json",
                                                 "api_key": ncbi, "tool": "BiologyAIResearcher"})
            err = (d or {}).get("error") if isinstance(d, dict) else None
            out["ncbi"] = ({"ok": False, "error": str(err)[:200]} if err or not d else
                           {"ok": True, "latency_ms": int((time.time() - t0) * 1000)})
        except Exception as exc:
            out["ncbi"] = {"ok": False, "error": redact(str(exc), [ncbi])[:240]}
    for name, key in supplied.items():
        router = AIRouter(keys={name: key})
        t0 = time.time()
        try:
            txt = router._ask_specific("Reply with the single word OK.", name)
            out[name] = {"ok": True, "latency_ms": int((time.time() - t0) * 1000), "reply": (txt or "")[:20]}
        except Exception as exc:
            out[name] = {"ok": False, "error": redact(str(exc), [key])[:240]}
    return out


NO_KEY_MESSAGE = ("No API key entered. BioSynthetica needs at least one API key (Groq, Gemini or NVIDIA - all free) to read "
                  "papers, extract claims and write the report; it does not run without one. Add a key on the "
                  "API keys page and press Start again.")


@app.post("/api/research")
def start_research(req: ResearchRequest):
    if not any(_keys_for(req).values()):
        raise HTTPException(400, NO_KEY_MESSAGE)
    if req.dataset_id:
        _validate_dataset(req)
    req = apply_speed(req)
    job = Job(req)
    job.searched_as = None if req.exact else _spell(req.question)["suggestion"]
    job.cache_key = cache_key(req)
    hit = cache_lookup(job.cache_key) if req.reuse else None
    if hit:
        try:
            saved = json.loads((RUNS / hit["job_id"] / "dossier.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            saved = None
        if saved:
            job.dir = RUNS / hit["job_id"]
            saved["reused_from"] = {"job_id": hit["job_id"], "finished": hit["finished"]}
            job.result = _jsonable(saved)
            job.status, job.stage = "done", "done"
            job.started = job.finished = time.time()
            job.req.keys = None
            job.secrets = []
            JOBS[job.id] = job
            ORDER.append(job.id)
            _prune()
            when = time.strftime("%Y-%m-%d %H:%M", time.localtime(hit["finished"]))
            job.emit("info", text=f"Opened the saved result of the same question from {when} (no new search). "
                                  f"Untick 'Reuse saved result' to run it again.")
            job.emit("done", text="Investigation complete (saved result).")
            return {"job_id": job.id, "position": 0, "reused": True}
    JOBS[job.id] = job
    ORDER.append(job.id)
    _prune()
    running_count = sum(1 for j in JOBS.values() if j.status == "running")
    ahead = max(0, running_count - MAX_CONCURRENT_JOBS + sum(1 for j in JOBS.values() if j.status == "queued" and j.id != job.id))
    job.emit("info", text=f"Queued ({ahead} job(s) ahead)." if ahead else "Starting...")
    EXECUTOR.submit(_run_job, job)
    return {"job_id": job.id, "position": ahead}


class LabRequest(BaseModel):
    params: Dict[str, Any] = Field(default_factory=dict)


@app.get("/api/lab/tools")
def lab_tools():
    from virtual_lab import catalogue
    return catalogue()


@app.post("/api/lab/{tool}")
def lab_run(tool: str, req: LabRequest):
    from virtual_lab import TOOLS, run_tool
    if tool not in TOOLS:
        raise HTTPException(404, f"Unknown tool '{tool}'")
    from virtual_lab import NEEDS_TABLE, ONLINE_TOOLS
    params = dict(req.params or {})
    online = tool in ONLINE_TOOLS or (tool == "protein_structure" and str(params.get("source") or "pdb") != "file")
    if online and not ALLOW_ONLINE_LAB:
        raise HTTPException(403, "Online look-ups are disabled on this server")
    if tool in NEEDS_TABLE:
        did = re.sub(r"[^a-f0-9]", "", str(params.pop("dataset_id", "") or ""))
        matches = list(UPLOADS.glob(f"{did}.*")) if did else []
        if not matches:
            raise HTTPException(400, "Upload the data table first.")
        try:
            params["table"] = _read_table(matches[0])
        except Exception as exc:
            raise HTTPException(400, f"Could not read the table: {exc}")
    return _jsonable(run_tool(tool, **params))


ALLOW_ONLINE_LAB = True
DATASET_NAMES: Dict[str, str] = {}


@app.get("/api/jobs")
def list_jobs():
    return [JOBS[j].summary() for j in reversed(ORDER) if j in JOBS]


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str):
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    return {**job.summary(), "result": job.result, "exact": bool(job.req.exact),
            "searched_as": getattr(job, "searched_as", None)}


@app.post("/api/jobs/{job_id}/cancel")
def cancel_job(job_id: str):
    """v26: stop a queued or running research job at its next safe point (a stage change or an AI call)."""
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    if job.status in ("done", "error", "cancelled"):
        return {**job.summary(), "note": f"the job is already {job.status}"}
    job.cancel_event.set()
    if job.status == "queued":                  # never started: cancelled now, with the same reason as a running job
        job.status = job.stage = "cancelled"
        job.error = "cancelled by the user"
        job.finished = time.time()
    job.emit("info", text="Cancellation requested - stopping at the next safe point.")
    return job.summary()


@app.get("/api/jobs/{job_id}/events")
async def stream(job_id: str, request: Request, since: int = 0):
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(404, "Job not found")

    async def gen():
        idx = since
        last_ping = time.time()
        while True:
            if await request.is_disconnected():
                return
            batch = job.events[idx:]
            for ev in batch:
                yield f"id: {ev['seq']}\ndata: {json.dumps(ev, ensure_ascii=False)}\n\n"
            idx += len(batch)
            if job.status in ("done", "error") and idx >= len(job.events):
                yield "event: end\ndata: {}\n\n"
                return
            if time.time() - last_ping > 15:
                last_ping = time.time()
                yield ": ping\n\n"
            if not batch:
                await asyncio.sleep(0.4)

    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"X-Accel-Buffering": "no"})


@app.get("/api/jobs/{job_id}/download/{kind}")
def download(job_id: str, kind: str):
    job = JOBS.get(job_id)
    if not job or not job.dir.exists():
        raise HTTPException(404, "Job not found")
    files = {"report": ("RESEARCH_REPORT.md", "text/markdown"), "dossier": ("dossier.json", "application/json"),
             "matrix": ("evidence_matrix.csv", "text/csv"), "bug": ("BUG_REPORT.md", "text/markdown")}
    if kind not in files or not (job.dir / files[kind][0]).exists():
        raise HTTPException(404, "File not available for this run")
    name, media = files[kind]
    return FileResponse(job.dir / name, media_type=media, filename=f"biosynthetica_{job_id}_{name}")


DATASET_TYPES = (".csv", ".tsv", ".txt", ".xlsx", ".xls")


def _read_table(path: Path):
    from core_utils import read_table
    return read_table(path)


def _validate_dataset(req: "ResearchRequest") -> None:
    import pandas as pd
    matches = list(UPLOADS.glob(f"{re.sub(r'[^a-f0-9]', '', req.dataset_id or '')}.*"))
    if not matches:
        raise HTTPException(400, "The uploaded dataset was not found on the server - upload it again.")
    if not req.group_col or not req.outcome_col:
        raise HTTPException(400, "Choose the grouping column (e.g. treatment / control) and the measured outcome column.")
    try:
        df = _read_table(matches[0])
    except Exception as exc:
        raise HTTPException(400, f"Could not read the dataset: {exc}")
    for col in (req.group_col, req.outcome_col):
        if col not in df.columns:
            raise HTTPException(400, f"Column '{col}' is not in the file (columns: {', '.join(map(str, df.columns))[:300]}).")
    y = pd.to_numeric(df[req.outcome_col], errors="coerce")
    if y.notna().sum() < 4:
        raise HTTPException(400, f"The outcome column '{req.outcome_col}' needs at least 4 numeric values.")
    sizes = df.loc[y.notna(), req.group_col].value_counts()
    if len(sizes) < 2 or (sizes >= 2).sum() < 2:
        raise HTTPException(400, f"The grouping column '{req.group_col}' needs at least two groups with 2+ numeric values each "
                                 f"(found: {', '.join(f'{k} = {v}' for k, v in sizes.items())[:200] or 'none'}).")


@app.post("/api/dataset")
async def upload_dataset(file: UploadFile = File(...)):
    import pandas as pd

    ext = Path(file.filename or "").suffix.lower()
    if ext not in DATASET_TYPES:
        raise HTTPException(400, "Upload a CSV, TSV or Excel (.xlsx) file with one row per sample.")
    data = await file.read()
    if not data:
        raise HTTPException(400, "The file is empty.")
    if len(data) > 100 * 1024 * 1024:
        raise HTTPException(413, "File larger than 100 MB.")
    uid = uuid.uuid4().hex
    DATASET_NAMES[uid] = Path(file.filename or f"dataset{ext}").name[:120]
    path = UPLOADS / f"{uid}{ext}"
    path.write_bytes(data)
    try:
        df = _read_table(path)
    except Exception as exc:
        path.unlink(missing_ok=True)
        raise HTTPException(400, f"Could not read file: {exc}")
    if df.empty or len(df.columns) < 2:
        path.unlink(missing_ok=True)
        raise HTTPException(400, "The table needs at least two columns (a group column and a measured outcome) and one row per sample.")
    numeric = [str(c) for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    groupish = [str(c) for c in df.columns if str(c) not in numeric and 2 <= df[c].nunique() <= 20]
    return {"dataset_id": uid, "name": file.filename, "rows": int(len(df)), "columns": [str(c) for c in df.columns],
            "numeric_columns": numeric, "group_columns": groupish,
            "preview": _jsonable(df.head(5).astype(object).where(df.head(5).notna(), None).to_dict(orient="records"))}


@app.get("/api/library/info")
def library_info():
    from vector_store import LocalVectorStore
    return LocalVectorStore().info()


@app.get("/api/library/search")
def library_search(q: str, k: int = 15, exact: bool = False):
    from vector_store import LocalVectorStore
    if not q.strip():
        return {"results": [], "ms": 0}
    store = LocalVectorStore()
    t0 = time.perf_counter()
    fixed = _spell(q) if not exact else {"suggestion": None}
    used = fixed["suggestion"] or q
    res = store.query(used, k=max(1, min(k, 50)))
    ms = round((time.perf_counter() - t0) * 1000, 1)
    return {"results": res, "ms": ms, "info": store.info(), "query": used,
            "corrected_from": q if fixed["suggestion"] else None}


def _spell(q: str) -> Dict[str, Any]:
    from evidence_identity import normalize_question
    norm = normalize_question(q)
    fixed = norm["question"]
    changed = bool(norm.get("corrections")) and fixed.strip().lower() != q.strip().lower()
    return {"original": q, "suggestion": fixed if changed else None,
            "corrections": [list(c) for c in norm.get("corrections") or []] if changed else []}


@app.get("/api/spell")
def spell(q: str = ""):
    q = (q or "")[:600]
    t0 = time.perf_counter()
    out = _spell(q) if q.strip() else {"original": q, "suggestion": None, "corrections": []}
    out["ms"] = round((time.perf_counter() - t0) * 1000, 1)
    return out


_WORD_SUFFIX = re.compile(r"(?:ations?|ation|ions?|ing|ings|ed|es|s|ive|ively|ement|ments?)$")


def _same_word_family(a: str, b: str) -> bool:
    sa, sb = _WORD_SUFFIX.sub("", a), _WORD_SUFFIX.sub("", b)
    short, long_ = sorted((sa, sb), key=len)
    if len(long_) - len(short) > 3:
        return False
    if len(short) >= 4 and long_.startswith(short):
        return True
    return len(short) >= 5 and short[:5] == long_[:5]


@app.get("/api/suggest")
def suggest(q: str = "", limit: int = 8):
    import difflib
    q = (q or "")[:600]
    limit = max(1, min(limit, 12))
    if len(q.strip()) < 2:
        return {"suggestions": []}
    t0 = time.perf_counter()
    low = q.strip().lower()
    fixed = (_spell(q)["suggestion"] or q).strip().lower()
    out: List[Dict[str, str]] = []
    seen = set()

    def add(text: str, kind: str):
        key = text.strip().lower()
        if key and key not in seen and key != low:
            seen.add(key)
            out.append({"text": text.strip(), "kind": kind})

    past: List[str] = [j.req.question for j in JOBS.values()]
    try:
        with ResearchDB() as db:
            past += [r[0] for r in db._conn.execute(
                "SELECT question FROM research_questions ORDER BY id DESC LIMIT 400").fetchall()]
    except Exception:
        pass
    words = [w for w in re.findall(r"[a-z0-9]+", fixed) if len(w) > 2]
    scored = []
    for p in dict.fromkeys(past):
        pl = p.lower()
        if pl.startswith(fixed) or pl.startswith(low):
            scored.append((0, p))
        elif words and all(w in pl for w in words):
            scored.append((1, p))
        elif len(fixed) > 6 and difflib.SequenceMatcher(None, fixed, pl[:len(fixed) + 8]).ratio() > 0.72:
            scored.append((2, p))
    for _, p in sorted(scored, key=lambda t: t[0])[:5]:
        add(p, "history")
    m = re.search(r"([A-Za-z]{2,})$", q)
    if m and len(out) < limit:
        try:
            from spelling import vocabulary
            v = vocabulary()
            stem = m.group(1).lower()
            dom = v["domain"]
            comp = sorted((w for w in dom if w.startswith(stem) and w != stem and len(w) > len(stem)),
                          key=lambda w: (-dom[w], len(w)))
            picked: List[str] = []
            for w in comp:
                if not any(_same_word_family(w, p) for p in picked):
                    picked.append(w)
                if len(picked) >= limit - len(out):
                    break
            for w in picked:
                add(q[: m.start(1)] + w, "completion")
        except Exception:
            pass
    return {"suggestions": out[:limit], "corrected": fixed if fixed != low else None,
            "ms": round((time.perf_counter() - t0) * 1000, 1)}


@app.get("/api/history")
def history(limit: int = 20):
    with ResearchDB() as db:
        rows = db._rows(db._conn.execute(
            "SELECT id, question, created_at FROM research_questions ORDER BY id DESC LIMIT ?", (limit,)).fetchall())
        for r in rows:
            s = db.research_question_summary(r["id"])
            r.update({k: s[k] for k in ("papers", "claims", "hypotheses", "audit_flags")})
    return rows


@app.get("/api/history/{rq_id}/report", response_class=PlainTextResponse)
def history_report(rq_id: int):
    from report_generator import ResearchReportGenerator
    with ResearchDB() as db:
        if not db.get_research_question(rq_id):
            raise HTTPException(404, "Unknown research question")
        tmp = RUNS / f"history_{rq_id}.md"
        return ResearchReportGenerator(db).generate(rq_id, output_path=str(tmp))


if __name__ == "__main__":
    import uvicorn

    host = os.getenv("HOST", "127.0.0.1")
    port = int(os.getenv("PORT", "8000"))
    print(f"BioSynthetica running on http://{host}:{port} (Concurrency limit: {MAX_CONCURRENT_JOBS})")
    uvicorn.run(app, host=host, port=port, log_level="warning")
