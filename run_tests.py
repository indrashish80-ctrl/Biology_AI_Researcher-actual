"""
run_tests.py - run every test file (no internet or keys needed) and print one line per file.

    python run_tests.py            # all files, in parallel
    python run_tests.py v24 v25    # only files whose name contains one of the words
Exit code 0 only when every file passes.
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def _one(path: Path) -> tuple:
    env = dict(os.environ, BIOSYN_OFFLINE="1", BIOSYN_NO_WARMUP="1", PYTHONIOENCODING="utf-8")
    # every test file gets its own temporary knowledge base, output folder, caches and paper folder: the tests run in
    # parallel (no shared database) and never write research_dossier_*.json, research_knowledge.db, papers/raw/*
    # or cache/*.db into the project folder
    import tempfile
    tmp = Path(tempfile.mkdtemp(prefix=f"biosyn_{path.stem}_"))
    for key, val in (("BIOSYN_DB_PATH", tmp / "research_knowledge.db"), ("BIOSYN_OUTPUT_DIR", tmp / "out"),
                     ("BIOSYN_RAW_DIR", tmp / "raw"), ("BIOSYN_CACHE_DIR", tmp / "cache"),
                     ("BIOSYN_METRICS_DB", tmp / "metrics.db"), ("BIOSYN_EXTRACTION_CACHE", tmp / "extraction_cache.db"),
                     ("BIOSYN_ARCHIVE_DIR", tmp / "archive"), ("BIOSYN_EXPERIMENT_DB", tmp / "experiments.db"),
                     ("BIOSYN_RUNS_DIR", tmp / "runs"), ("VECTOR_STORE_PATH", tmp / "vector_store"),
                     ("BIOSYN_NO_INGEST_WORKER", "1")):
        env.setdefault(key, str(val))
    t = time.time()
    try:
        p = subprocess.run([sys.executable, str(path)], cwd=str(ROOT), env=env, capture_output=True, text=True,
                           encoding="utf-8", errors="replace", timeout=1800)
        out, rc = p.stdout + p.stderr, p.returncode
    except subprocess.TimeoutExpired:
        out, rc = "TIMEOUT", 124
    finally:
        import shutil
        shutil.rmtree(tmp, ignore_errors=True)
    passes = sum(1 for ln in out.splitlines() if ln.startswith("PASS"))
    fails = [ln for ln in out.splitlines() if ln.startswith("FAIL") or ln.startswith("Traceback")]
    return path.name, rc, passes, fails, round(time.time() - t, 1), out


def main(argv: list) -> int:
    files = sorted((ROOT / "tests").glob("test_*.py"))
    if argv:
        files = [f for f in files if any(a in f.name for a in argv)]
    with ThreadPoolExecutor(max_workers=max(2, (os.cpu_count() or 2))) as ex:
        results = list(ex.map(_one, files))
    bad = 0
    total = 0
    for name, rc, passes, fails, secs, out in results:
        total += passes
        status = "ok  " if rc == 0 and not fails else "FAIL"
        bad += status == "FAIL"
        print(f"{status} {name:34s} {passes:4d} checks  {secs:6.1f}s")
        if status == "FAIL":
            for ln in (fails or out.splitlines()[-15:])[:15]:
                print("       " + ln[:220])
    print(f"\n{len(results) - bad}/{len(results)} files pass, {total} checks")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
