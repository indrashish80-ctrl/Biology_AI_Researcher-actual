"""
run_metrics.py  (v25)
=====================
Speed and cost tracked AGAINST accuracy, per code version.

Per run (from the engine's own instrumentation - stage timings, provider health, cache counters):
    latency       total run time; p50 / p95 of stage times; p50 / p95 of single AI-call latency
    AI usage      calls, failures, retries, rate-limit hits, tokens in / out, per provider
    cost          tokens x price per million tokens from prices.json (BIOSYN_PRICES); free-tier keys cost 0 and the
                  report says so - no price is guessed
    caches        AI-answer cache, extraction cache, embedding cache, retrieval cache, web cache hit rates
Per benchmark (gold_bench.py): accuracy numbers with the same code versions.

Everything is appended to cache/metrics.db, and trend() returns one row per code version with the run medians next
to the latest benchmark accuracy - so a change that makes runs faster but less accurate is visible.

    collect(dossier, router) -> metrics dict (also stored in dossier['metrics'])
    record_run(metrics) / record_accuracy(benchmark) / trend()
"""
from __future__ import annotations

import json
import os
import sqlite3
import statistics
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

METRICS_VERSION = "1.0"
_LOCK = threading.Lock()
_DDL = """
CREATE TABLE IF NOT EXISTS runs (id INTEGER PRIMARY KEY AUTOINCREMENT, at REAL NOT NULL, versions TEXT NOT NULL,
    question TEXT, tier TEXT, metrics_json TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS accuracy (id INTEGER PRIMARY KEY AUTOINCREMENT, at REAL NOT NULL, versions TEXT NOT NULL,
    results_json TEXT NOT NULL);
"""


def _db() -> sqlite3.Connection:
    p = Path(os.getenv("BIOSYN_METRICS_DB") or Path(__file__).resolve().parent / "cache" / "metrics.db")
    p.parent.mkdir(parents=True, exist_ok=True)
    c = sqlite3.connect(str(p), timeout=30)
    c.executescript(_DDL)
    return c


def pct(xs: Sequence[float], q: float) -> Optional[float]:
    xs = sorted(x for x in xs if x is not None)
    if not xs:
        return None
    k = (len(xs) - 1) * q
    f, c = int(k), min(int(k) + 1, len(xs) - 1)
    return round(xs[f] + (xs[c] - xs[f]) * (k - f), 2)


def prices() -> Dict[str, Dict[str, float]]:
    """{'groq': {'in': $ per 1M input tokens, 'out': ...}, ...} from prices.json / BIOSYN_PRICES; empty = free tier."""
    src = os.getenv("BIOSYN_PRICES") or str(Path(__file__).resolve().parent / "prices.json")
    try:
        if src.strip().startswith("{"):
            return json.loads(src)
        p = Path(src)
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
    except Exception:
        return {}


def code_versions() -> Dict[str, str]:
    try:
        import gold_bench
        return gold_bench.versions()
    except Exception:
        return {}


def _dir_bytes(p: Any) -> Optional[int]:
    from pathlib import Path
    p = Path(p)
    if not p.exists():
        return None
    if p.is_file():
        return p.stat().st_size
    tot = 0
    for f in p.rglob("*"):
        try:
            if f.is_file():
                tot += f.stat().st_size
        except OSError:
            pass
    return tot


def gpu() -> Optional[List[Dict[str, Any]]]:
    """GPU use from nvidia-smi when one is installed (BioSynthetica itself needs no GPU)."""
    import shutil
    import subprocess
    exe = shutil.which("nvidia-smi")
    if not exe:
        return None
    try:
        out = subprocess.run([exe, "--query-gpu=name,utilization.gpu,memory.used,memory.total", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=3).stdout
        rows = []
        for ln in out.strip().splitlines():
            n, u, mu, mt = [x.strip() for x in ln.split(",")]
            rows.append({"name": n, "util_pct": float(u), "mem_used_mb": float(mu), "mem_total_mb": float(mt)})
        return rows
    except Exception:
        return None


def resources(dossier: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """CPU time, peak memory, disk used by the knowledge base / caches / library / runs, GPU, queue wait, throughput."""
    import os
    from pathlib import Path
    out: Dict[str, Any] = {"cpu_seconds_process": round(time.process_time(), 2)}
    try:
        import resource as _r
        rss = _r.getrusage(_r.RUSAGE_SELF).ru_maxrss
        out["peak_memory_mb"] = round(rss / 1024 / (1024 if os.uname().sysname == "Darwin" else 1), 1)
    except Exception:
        try:
            import psutil
            out["peak_memory_mb"] = round(psutil.Process().memory_info().peak_wset / 2 ** 20, 1)   # Windows
        except Exception:
            out["peak_memory_mb"] = None
    try:
        import psutil
        out["cpu_count"] = psutil.cpu_count()
        out["system_memory_used_pct"] = psutil.virtual_memory().percent
    except Exception:
        out["cpu_count"] = os.cpu_count()
    root = Path(__file__).resolve().parent
    disk = {}
    try:
        from database import DEFAULT_DB_PATH
        disk["knowledge_base"] = _dir_bytes(DEFAULT_DB_PATH)
    except Exception:
        pass
    disk["caches"] = _dir_bytes(os.getenv("BIOSYN_CACHE_DIR") or root / "cache")
    disk["library"] = _dir_bytes(os.getenv("BIOSYN_LIBRARY_DIR") or root / "library")
    disk["runs"] = _dir_bytes(os.getenv("BIOSYN_RUNS_DIR") or root / "runs")
    out["disk_bytes"] = {k: v for k, v in disk.items() if v is not None}
    out["gpu"] = gpu()
    d = dossier or {}
    out["queue_wait_s"] = d.get("queue_wait_s")
    out["server_load"] = d.get("server_load")
    return out


def collect(dossier: Dict[str, Any], router: Any = None) -> Dict[str, Any]:
    timings = dossier.get("timings") or []
    stage_s = [t.get("seconds") for t in timings if t.get("seconds") is not None]
    out: Dict[str, Any] = {"version": METRICS_VERSION, "when": time.strftime("%Y-%m-%d %H:%M:%S"),
                           "total_seconds": dossier.get("timings_total_seconds") or round(sum(stage_s), 1),
                           "stage_p50_s": pct(stage_s, 0.5), "stage_p95_s": pct(stage_s, 0.95),
                           "stages": {t["stage"]: t.get("seconds") for t in timings if t.get("stage")}}
    agg = {k: sum(int(t.get(k) or 0) for t in timings) for k in
           ("ai_calls", "ai_failures", "ai_retries", "ai_rate_limited", "ai_tokens_in", "ai_tokens_out", "ai_cache_hits",
            "ai_tokens_cached", "requests", "cache_hits", "cpu_ms")}
    out.update(agg)
    lat: List[float] = []
    per_provider: Dict[str, Any] = {}
    health = getattr(router, "health", None)
    if health is not None:
        for name, st in getattr(health, "_stats", {}).items():
            lat += list(st.latencies_ms)
            per_provider[name] = {"calls": st.total_calls, "failures": st.total_failures,
                                  "p50_ms": pct(st.latencies_ms, 0.5), "p95_ms": pct(st.latencies_ms, 0.95)}
    out["ai_call_p50_ms"], out["ai_call_p95_ms"] = pct(lat, 0.5), pct(lat, 0.95)
    out["providers"] = per_provider
    pr = prices()
    if pr:
        # tokens are not split per provider by the router: the blended price of the providers used is applied
        used = [p for p in per_provider if p in pr] or list(pr)
        pin = statistics.mean(pr[p].get("in", 0) for p in used) if used else 0
        pout = statistics.mean(pr[p].get("out", 0) for p in used) if used else 0
        out["cost_usd"] = round(agg["ai_tokens_in"] / 1e6 * pin + agg["ai_tokens_out"] / 1e6 * pout, 4)
        out["cost_basis"] = "prices.json"
    else:
        out["cost_usd"] = 0.0
        out["cost_basis"] = "free-tier keys (no prices.json): token counts are reported, money cost is 0"
    caches: Dict[str, Any] = {}
    calls = agg["ai_calls"] + agg["ai_cache_hits"]
    caches["ai_answers"] = round(agg["ai_cache_hits"] / calls, 3) if calls else None
    web = agg["requests"] + agg["cache_hits"]
    caches["web"] = round(agg["cache_hits"] / web, 3) if web else None
    try:
        import extraction_cache
        caches["extraction"] = extraction_cache.stats().get("hit_rate")
    except Exception:
        pass
    try:
        import retrieval_cache
        rs = retrieval_cache.stats()
        caches["embedding"], caches["retrieval"] = rs.get("embedding_hit_rate"), rs.get("retrieval_hit_rate")
    except Exception:
        pass
    # provider-side prompt caching: the share of input tokens the provider reported as served from its cache
    caches["provider_prompt_cache"] = round(agg["ai_tokens_cached"] / agg["ai_tokens_in"], 3) if agg["ai_tokens_in"] else None
    try:
        import library_db
        if library_db.exists():
            with library_db.Library() as _lib:
                out["library_pre_read_papers"] = _lib.stats().get("pre_read")      # a count, not a rate
    except Exception:
        pass
    out["cache_hit_rate"] = caches
    out["cpu_seconds_run"] = round(agg.get("cpu_ms", 0) / 1000, 2)
    try:
        out["resources"] = resources(dossier)
    except Exception as exc:
        out["resources"] = {"error": str(exc)[:200]}
    m = dossier.get("evidence_matrix") or {}
    out["papers"] = dossier.get("papers_analyzed")
    out["verified_claims"] = ((m.get("totals") or {}).get("verified_claims"))
    if out.get("papers"):
        out["seconds_per_paper"] = round(out["total_seconds"] / out["papers"], 1) if out["total_seconds"] else None
        out["ai_calls_per_paper"] = round(agg["ai_calls"] / out["papers"], 2)
    return out


def record_run(metrics: Dict[str, Any], question: str = "", tier: str = "") -> None:
    with _LOCK:
        c = _db()
        try:
            c.execute("INSERT INTO runs (at, versions, question, tier, metrics_json) VALUES (?,?,?,?,?)",
                      (time.time(), json.dumps(code_versions(), sort_keys=True), question[:300], tier,
                       json.dumps(metrics, default=str)))
            c.commit()
        finally:
            c.close()


def record_accuracy(bench: Dict[str, Any]) -> None:
    keep = {}
    for part in ("A", "A_heldout", "B", "C", "D"):
        v = bench.get(part)
        if isinstance(v, dict):
            keep[part] = {k: v.get(k) for k in ("accuracy", "opposite_direction", "n", "finding_recall",
                                                 "direction_accuracy_found", "citation_accuracy", "unsupported_claim_rate",
                                                 "violations", "modes") if k in v}
    with _LOCK:
        c = _db()
        try:
            c.execute("INSERT INTO accuracy (at, versions, results_json) VALUES (?,?,?)",
                      (time.time(), json.dumps(bench.get("versions") or code_versions(), sort_keys=True), json.dumps(keep)))
            c.commit()
        finally:
            c.close()


def trend(limit: int = 20) -> List[Dict[str, Any]]:
    with _LOCK:
        c = _db()
        try:
            runs = c.execute("SELECT versions, metrics_json FROM runs ORDER BY id").fetchall()
            acc = c.execute("SELECT versions, results_json, at FROM accuracy ORDER BY id").fetchall()
        finally:
            c.close()
    by_v: Dict[str, Dict[str, Any]] = {}
    for v, mj in runs:
        m = json.loads(mj)
        d = by_v.setdefault(v, {"runs": 0, "seconds": [], "ai_calls": [], "cost": []})
        d["runs"] += 1
        d["seconds"].append(m.get("total_seconds"))
        d["ai_calls"].append(m.get("ai_calls"))
        d["cost"].append(m.get("cost_usd"))
    for v, rj, at in acc:
        by_v.setdefault(v, {"runs": 0, "seconds": [], "ai_calls": [], "cost": []})["accuracy"] = json.loads(rj)
    out = []
    for v, d in list(by_v.items())[-limit:]:
        out.append({"versions": json.loads(v), "runs": d["runs"], "median_seconds": pct(d["seconds"], 0.5),
                    "p95_seconds": pct(d["seconds"], 0.95), "median_ai_calls": pct(d["ai_calls"], 0.5),
                    "median_cost_usd": pct(d["cost"], 0.5), "accuracy": d.get("accuracy")})
    return out


def markdown(m: Dict[str, Any]) -> str:
    f = lambda v, u="": "n/a" if v is None else f"{v}{u}"
    ch = m.get("cache_hit_rate") or {}
    L = ["## Run metrics", "", "| | |", "|---|---|",
         f"| total time | {f(m.get('total_seconds'), ' s')} (stage p50 {f(m.get('stage_p50_s'), ' s')}, p95 {f(m.get('stage_p95_s'), ' s')}) |",
         f"| AI calls | {m.get('ai_calls', 0)} ({m.get('ai_failures', 0)} failed, {m.get('ai_retries', 0)} retries, "
         f"{m.get('ai_rate_limited', 0)} rate-limited); call latency p50 {f(m.get('ai_call_p50_ms'), ' ms')}, p95 {f(m.get('ai_call_p95_ms'), ' ms')} |",
         f"| tokens | {m.get('ai_tokens_in', 0):,} in / {m.get('ai_tokens_out', 0):,} out |",
         f"| cost | ${m.get('cost_usd', 0):.4f} ({m.get('cost_basis')}) |",
         "| cache hit rate | " + ", ".join(f"{k} {v:.0%}" for k, v in ch.items() if v is not None) + " |"]
    if m.get("seconds_per_paper"):
        L.append(f"| per paper | {m['seconds_per_paper']} s, {m.get('ai_calls_per_paper')} AI calls |")
    if m.get("ai_tokens_cached"):
        L.append(f"| provider prompt cache | {m['ai_tokens_cached']:,} input tokens served from the provider's cache |")
    r = m.get("resources") or {}
    if r and "error" not in r:
        disk = r.get("disk_bytes") or {}
        L.append(f"| CPU / memory | {f(m.get('cpu_seconds_run'), ' s')} CPU in this run; peak memory {f(r.get('peak_memory_mb'), ' MB')} |")
        if disk:
            L.append("| disk | " + ", ".join(f"{k.replace('_', ' ')} {v / 2**20:.1f} MB" for k, v in disk.items()) + " |")
        L.append(f"| GPU | {', '.join(g['name'] + ' ' + str(g['util_pct']) + '%' for g in r['gpu']) if r.get('gpu') else 'none used'} |")
        if r.get("queue_wait_s") is not None:
            L.append(f"| queue wait | {r['queue_wait_s']} s" + (f"; server: {r['server_load']}" if r.get("server_load") else "") + " |")
    return "\n".join(L) + "\n"
