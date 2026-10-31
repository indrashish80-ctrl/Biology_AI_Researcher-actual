"""
job_queue.py  (v25)
===================
Resource limits and a job queue for virtual-lab experiments.

Every experiment runs in its OWN process (a crashed or runaway model cannot take the web server down), with caps:
    CPU time   RLIMIT_CPU     (POSIX)          default 60 s     BIOSYN_JOB_CPU_S
    memory     RLIMIT_AS      (POSIX)          default 1024 MB  BIOSYN_JOB_MEM_MB
    wall time  timeout + kill (every platform) default 120 s    BIOSYN_JOB_WALL_S
On Windows (no `resource` module) the CPU and memory caps are a Job Object set through kernel32 with ctypes - no
pywin32 needed. Every result says which caps were active in 'caps_active'. A job that hits a cap is logged as 'timeout' / 'resource_limit'.

    q = JobQueue(workers=2)          # FIFO, at most `workers` experiments at once
    jid = q.submit(spec)             # validated first - an invalid spec is logged as 'rejected' and never queued
    q.status(jid) / q.result(jid, wait=30)
    run_spec(spec)                   # the function the worker process runs (also usable directly, in-process)
"""
from __future__ import annotations

import os
import queue
import sys
import threading
import time
import traceback
import uuid
from typing import Any, Dict, Optional

QUEUE_VERSION = "1.0"


def caps() -> Dict[str, Any]:
    return {"cpu_s": int(os.getenv("BIOSYN_JOB_CPU_S", "60")), "mem_mb": int(os.getenv("BIOSYN_JOB_MEM_MB", "1024")),
            "wall_s": float(os.getenv("BIOSYN_JOB_WALL_S", "120"))}


def _job_object_structs():
    """ctypes layouts of JOBOBJECT_EXTENDED_LIMIT_INFORMATION (winnt.h); 144 bytes on 64-bit Windows."""
    import ctypes

    class BASIC(ctypes.Structure):
        _fields_ = [("PerProcessUserTimeLimit", ctypes.c_int64), ("PerJobUserTimeLimit", ctypes.c_int64),
                    ("LimitFlags", ctypes.c_uint32), ("MinimumWorkingSetSize", ctypes.c_size_t),
                    ("MaximumWorkingSetSize", ctypes.c_size_t), ("ActiveProcessLimit", ctypes.c_uint32),
                    ("Affinity", ctypes.c_size_t), ("PriorityClass", ctypes.c_uint32), ("SchedulingClass", ctypes.c_uint32)]

    class IO(ctypes.Structure):
        _fields_ = [(n, ctypes.c_uint64) for n in ("ReadOperationCount", "WriteOperationCount", "OtherOperationCount",
                                                   "ReadTransferCount", "WriteTransferCount", "OtherTransferCount")]

    class EXTENDED(ctypes.Structure):
        _fields_ = [("BasicLimitInformation", BASIC), ("IoInfo", IO), ("ProcessMemoryLimit", ctypes.c_size_t),
                    ("JobMemoryLimit", ctypes.c_size_t), ("PeakProcessMemoryUsed", ctypes.c_size_t),
                    ("PeakJobMemoryUsed", ctypes.c_size_t)]
    return EXTENDED


_WIN_JOB = None                                   # keep the handle alive for the life of the worker process


def _apply_windows_limits(c: Dict[str, Any]) -> Dict[str, bool]:
    """Windows has no `resource` module: put this process in a Job Object with a CPU-time and a committed-memory
    limit (kernel32 via ctypes, no pywin32 needed). Windows terminates the process when either limit is exceeded."""
    global _WIN_JOB
    import ctypes
    from ctypes import wintypes
    JOB_OBJECT_LIMIT_PROCESS_TIME, JOB_OBJECT_LIMIT_PROCESS_MEMORY = 0x00000002, 0x00000100
    JobObjectExtendedLimitInformation = 9
    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    k32.CreateJobObjectW.restype = wintypes.HANDLE
    k32.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
    k32.SetInformationJobObject.restype = wintypes.BOOL
    k32.SetInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
    k32.AssignProcessToJobObject.restype = wintypes.BOOL
    k32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
    k32.GetCurrentProcess.restype = wintypes.HANDLE
    job = k32.CreateJobObjectW(None, None)
    if not job:
        return {"cpu": False, "memory": False}
    info = _job_object_structs()()
    info.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_PROCESS_TIME | JOB_OBJECT_LIMIT_PROCESS_MEMORY
    info.BasicLimitInformation.PerProcessUserTimeLimit = int(c["cpu_s"]) * 10_000_000      # 100-ns units
    info.ProcessMemoryLimit = int(c["mem_mb"]) * 1024 * 1024
    ok = k32.SetInformationJobObject(job, JobObjectExtendedLimitInformation, ctypes.byref(info), ctypes.sizeof(info))
    if ok and k32.AssignProcessToJobObject(job, k32.GetCurrentProcess()):
        _WIN_JOB = job
        return {"cpu": True, "memory": True}
    return {"cpu": False, "memory": False}


def _apply_limits(c: Dict[str, Any]) -> Dict[str, bool]:
    active = {"cpu": False, "memory": False, "wall": True}
    if os.name == "nt":
        try:
            active.update(_apply_windows_limits(c))
        except Exception:
            pass                                  # very old Windows: the wall-time cap still applies
        return active
    try:
        import resource
        resource.setrlimit(resource.RLIMIT_CPU, (c["cpu_s"], c["cpu_s"] + 5))
        active["cpu"] = True
        mem = c["mem_mb"] * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_AS, (mem, mem))
        active["memory"] = True
    except Exception:
        pass
    return active


# ------------------------------------------------------------------------------------------------ the work itself
def run_spec(spec: Dict[str, Any]) -> Dict[str, Any]:
    """Validate and run one spec (base run, sweep / grid, sensitivity, threshold, expected-result test).
    v26: {'agent': goal, 'observations': {...}} runs the autonomous loop (lab_agent); {'uncertainty': {...}} the full
    uncertainty report (model_uncertainty) - both inside the same capped worker process."""
    import experiment_spec as es
    import sensitivity as se
    import simulator as S
    lib = S.ModelLibrary()
    if isinstance(spec, dict) and spec.get("agent") is not None:
        import lab_agent
        return lab_agent.run(spec["agent"], lib, observations=spec.get("observations"))
    if isinstance(spec, dict) and spec.get("uncertainty") is not None:
        import model_uncertainty as mu
        u = spec["uncertainty"]
        try:
            r = mu.report(lib, u["model"], u.get("parameters") or {}, u["observable"], u.get("metric", "max"),
                          float(u.get("t_end", 100)), u.get("uncertain"), int(min(500, u.get("n_mc", 200))),
                          u.get("data"), u.get("fit"), int(u.get("seed", 1)))
        except (KeyError, ValueError) as exc:
            return {"status": "rejected", "errors": [f"{type(exc).__name__}: {exc}"]}
        return {"status": "ok", "uncertainty": r, "markdown": mu.markdown(r),
                "basis": "in silico - uncertainty of a model's output, not of the biology"}
    v = es.validate(spec, lib)
    if not v["valid"]:
        return {"status": "rejected", "errors": v["errors"], "warnings": v["warnings"]}
    s = v["spec"]
    sim = S.Simulator(lib.get(s["model"]))
    t_end, pts, obs = s["t_end"], s.get("points", 201), s["observables"]
    out: Dict[str, Any] = {"status": "ok", "model": s["model"], "model_version": sim.version, "spec": s,
                           "warnings": v["warnings"], "assumptions": lib.get(s["model"]).get("assumptions"),
                           "limits": lib.get(s["model"]).get("limits")}
    runs = []
    for p in es.expand(s)[:es.LIMITS["total_simulations"]]:
        r = sim.run(p, t_end=t_end, points=pts, observables=obs, seed=s.get("seed"), solver=s.get("solver", "cvode"))
        runs.append({"parameters": p, "summary": r["summary"], "numerically_ok": r["numerically_ok"]})
        if len(runs) == 1:
            out["series"] = {"time": r["time"][::max(1, len(r["time"]) // 200)],
                             **{o: r["series"][o][::max(1, len(r["time"]) // 200)] for o in obs}}
            out["solver"] = r["solver"]
    out["runs"] = runs
    metric_obs = (s.get("expected") or {}).get("observable") or obs[0]
    if s.get("sensitivity"):
        sv = s["sensitivity"]
        if sv["method"] == "morris":
            out["sensitivity"] = se.morris(sim, sv["parameters"], sv.get("observable") or metric_obs, sv.get("metric", "max"),
                                           r=int(sv.get("trajectories", 10)), seed=int(s.get("seed") or 1), t_end=t_end,
                                           points=min(pts, 401), params=s["parameters"])
        else:
            out["sensitivity"] = se.local(sim, s["parameters"], sv["parameters"], sv.get("observable") or metric_obs,
                                          sv.get("metric", "max"), t_end=t_end, points=min(pts, 401))
    if s.get("threshold"):
        th = s["threshold"]
        out["threshold"] = se.threshold(sim, th["parameter"], th["observable"], th.get("metric", "max"), float(th["target"]),
                                        float(th["low"]), float(th["high"]), t_end=t_end, points=min(pts, 401),
                                        params=s["parameters"])
    if s.get("expected"):
        out["expected"] = [{"parameters": r["parameters"], **es.evaluate_expected(s["expected"], r["summary"])} for r in runs]
    out["n_simulations"] = len(runs) + (out.get("sensitivity") or {}).get("simulations", 0) + \
        (out.get("threshold") or {}).get("simulations", 0)
    out["basis"] = "in silico - consistency with a model, never evidence that the hypothesis is true"
    return out


def _worker_main() -> int:
    """`python job_queue.py --worker`: read {'spec', 'caps'} as JSON on stdin, apply the caps, write the result on stdout.
    A plain subprocess (not multiprocessing 'spawn'), so the caller's __main__ module is never re-imported in the
    child - with spawn, a script without an `if __name__ == "__main__"` guard would run again inside every job."""
    import json
    req = json.loads(sys.stdin.read() or "{}")
    c = req.get("caps") or caps()
    active = _apply_limits(c)
    try:
        res = run_spec(req.get("spec") or {})
        res["caps_active"] = active
    except MemoryError:
        res = {"status": "resource_limit", "error": f"memory cap of {c['mem_mb']} MB reached", "caps_active": active}
    except Exception as exc:
        res = {"status": "failed", "error": f"{type(exc).__name__}: {exc}", "trace": traceback.format_exc()[-1500:],
               "caps_active": active}
    out = json.dumps(res, default=str)
    sys.stdout.write("\n__BIOSYN_RESULT__" + out + "\n")
    sys.stdout.flush()
    return 0


def run_limited(spec: Dict[str, Any], c: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Run one spec in a separate process under the caps."""
    import json
    import subprocess
    from pathlib import Path
    c = c or caps()
    here = Path(__file__).resolve()
    t0 = time.time()
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    p = subprocess.Popen([sys.executable, str(here), "--worker"], cwd=str(here.parent), stdin=subprocess.PIPE,
                         stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding="utf-8",
                         creationflags=flags)
    res: Optional[Dict[str, Any]] = None
    try:
        out, err = p.communicate(json.dumps({"spec": spec, "caps": c}, default=str), timeout=c["wall_s"])
    except subprocess.TimeoutExpired:
        p.kill()
        try:
            p.communicate(timeout=5)
        except Exception:
            pass
        res = {"status": "timeout", "error": f"wall-time cap of {c['wall_s']:g} s reached - the job was stopped",
               "caps_active": {"cpu": os.name == "posix", "memory": os.name == "posix", "wall": True}}
        out, err = "", ""
    if res is None:
        marker = "__BIOSYN_RESULT__"
        if marker in (out or ""):
            try:
                res = json.loads(out.rsplit(marker, 1)[1].strip().splitlines()[0])
            except Exception:
                res = None
        if res is None:
            code = p.returncode
            tail = (err or "").strip().splitlines()[-1:] or [""]
            # no result line: the process was killed (POSIX: negative signal code; Windows Job Object: a positive
            # exit code) - a Python error would still have printed a result, unless the traceback says otherwise
            crashed_in_python = "Traceback (most recent call last)" in (err or "") and "MemoryError" not in (err or "")
            res = {"status": "failed" if crashed_in_python or code == 0 else "resource_limit",
                   "error": f"the experiment process ended without a result (exit code {code}; a CPU or memory cap may have "
                            f"been hit) {tail[0][:300]}".strip()}
    res["duration_s"] = round(time.time() - t0, 2)
    res["caps"] = c
    return res


class JobQueue:
    def __init__(self, workers: int = 2, isolated: bool = True):
        self.q: "queue.Queue[str]" = queue.Queue()
        self.jobs: Dict[str, Dict[str, Any]] = {}
        self.isolated = isolated
        self._lock = threading.Lock()
        try:                    # v26.2: a run that was going when the server last stopped can never finish: say so
            import experiment_log as el
            el.mark_interrupted()
        except Exception:
            pass
        self._threads = [threading.Thread(target=self._loop, daemon=True, name=f"biosyn-lab-{i}") for i in range(max(1, workers))]
        for t in self._threads:
            t.start()

    def submit(self, spec: Dict[str, Any], research_question_id: Optional[int] = None) -> str:
        import experiment_log as el
        import experiment_spec as es
        import simulator as S
        jid = uuid.uuid4().hex[:12]
        lib = S.ModelLibrary()
        if isinstance(spec, dict) and (spec.get("agent") is not None or spec.get("uncertainty") is not None):
            # v26 agent plans / uncertainty reports: their own validators, same queue, same caps, same log
            import lab_agent
            if spec.get("agent") is not None:
                errs = lab_agent.validate_goal(spec["agent"] if isinstance(spec["agent"], dict) else {}, lib)
            else:
                u = spec["uncertainty"] if isinstance(spec["uncertainty"], dict) else {}
                errs = [] if u.get("model") in lib.cards and u.get("observable") in (lib.cards.get(u.get("model")) or {}).get("observables", {}) \
                    else ["uncertainty needs a library 'model' and one of its 'observable's"]
            v = {"valid": not errs, "errors": errs, "spec": spec, "n_simulations": None}
        else:
            v = es.validate(spec, lib)
        rec = {"model_id": spec.get("model"), "spec": spec, "parameters": spec.get("parameters"), "seed": spec.get("seed"),
               "research_question_id": research_question_id, "job_id": jid}
        if not v["valid"]:
            exp = el.log({**rec, "status": "rejected", "error": "; ".join(v["errors"]), "finished": time.time()})
            with self._lock:
                self.jobs[jid] = {"id": jid, "status": "rejected", "errors": v["errors"], "experiment_id": exp}
            return jid
        if v["spec"].get("model") in lib.cards:
            rec["model_version"] = S.model_hash(lib.get(v["spec"]["model"]))
        exp = el.log({**rec, "status": "queued", "n_simulations": v["n_simulations"]})
        with self._lock:
            self.jobs[jid] = {"id": jid, "status": "queued", "spec": v["spec"], "experiment_id": exp, "submitted": time.time()}
        self.q.put(jid)
        return jid

    def _loop(self) -> None:
        import experiment_log as el
        while True:
            jid = self.q.get()
            with self._lock:
                job = self.jobs[jid]
                job["status"] = "running"
            el.update(job["experiment_id"], status="running")
            t0 = time.time()
            try:
                res = run_limited(job["spec"]) if self.isolated else run_spec(job["spec"])
            except Exception as exc:
                res = {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}
            st = res.get("status", "failed")
            summ = {"runs": [r.get("summary") for r in (res.get("runs") or [])[:20]],
                    "expected": res.get("expected"), "threshold": res.get("threshold"),
                    "sensitivity_ranking": (res.get("sensitivity") or {}).get("ranking")}
            el.update(job["experiment_id"], status=st, finished=time.time(), duration_s=round(time.time() - t0, 2),
                      error=res.get("error"), summary=summ, solver=res.get("solver"), n_simulations=res.get("n_simulations"))
            with self._lock:
                job.update({"status": st, "result": res, "finished": time.time()})

    def status(self, jid: str) -> Dict[str, Any]:
        with self._lock:
            j = dict(self.jobs.get(jid) or {"id": jid, "status": "unknown"})
        j.pop("result", None)
        return j

    def result(self, jid: str, wait: float = 0) -> Optional[Dict[str, Any]]:
        end = time.time() + wait
        while True:
            with self._lock:
                j = self.jobs.get(jid)
                if j and j["status"] not in ("queued", "running"):
                    return j.get("result") or {"status": j["status"], "errors": j.get("errors")}
            if time.time() >= end:
                return None
            time.sleep(0.1)


_QUEUE: Optional[JobQueue] = None


def default_queue() -> JobQueue:
    global _QUEUE
    if _QUEUE is None:
        _QUEUE = JobQueue(workers=int(os.getenv("BIOSYN_LAB_WORKERS", "2")))
    return _QUEUE


if __name__ == "__main__" and "--worker" in sys.argv:
    sys.exit(_worker_main())
