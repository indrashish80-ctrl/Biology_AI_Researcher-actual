"""
bench_library.py  (v26.1)
=========================
Measure the local library's speed ON YOUR COMPUTER, on a synthetic library, without touching your real one.

    python bench_library.py                    # 200,000 records (about 1 minute to build, ~0.5 GB in a temp folder)
    python bench_library.py --records 1000000  # 1 million (about 5 minutes, ~2.5 GB)
    python bench_library.py --hnsw             # also time the approximate index (pip install hnswlib)

It reports the build rate and the median / 95th-percentile time of keyword, semantic and hybrid search, the full
local-first decision, and an identifier lookup. The synthetic records use the offline hashing embedder; with the
real embedding model, add the time to embed one question (about 5-20 ms on a CPU).
"""
from __future__ import annotations

import argparse
import os
import random
import shutil
import statistics
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent

DRUGS = ["metformin", "atorvastatin", "aspirin", "insulin glargine", "semaglutide", "tamoxifen", "letrozole", "lisinopril",
         "warfarin", "apixaban", "vitamin D", "omega-3 fatty acids", "empagliflozin", "dapagliflozin", "rosuvastatin",
         "methotrexate", "adalimumab", "pembrolizumab", "nivolumab", "trastuzumab"]
CONDS = ["breast cancer", "type 2 diabetes", "stroke", "myocardial infarction", "colorectal cancer", "heart failure",
         "atrial fibrillation", "rheumatoid arthritis", "melanoma", "chronic kidney disease", "obesity", "hypertension",
         "non-small cell lung cancer", "osteoporosis", "depression", "asthma"]
OUTS = ["mortality", "HbA1c", "Ki-67", "progression-free survival", "major adverse cardiovascular events", "body weight",
        "eGFR decline", "fracture risk", "recurrence", "hospitalisation", "quality of life", "blood pressure"]
DESIGNS = [("Randomized Controlled Trial", "randomised controlled trial"), ("Meta-Analysis", "meta-analysis"),
           ("Observational Study", "cohort study"), ("Journal Article", "case-control study"), ("Review", "review")]
FILLER = ("patients were enrolled across centres and followed for a median period; baseline characteristics were "
          "balanced; adverse events were recorded and analysed by intention to treat; sensitivity analyses gave "
          "similar results").split()
QUESTIONS = ["Does metformin reduce Ki-67 in breast cancer?",
             "Effect of atorvastatin on major adverse cardiovascular events in type 2 diabetes",
             "Does semaglutide reduce body weight in obesity?",
             "Apixaban versus warfarin for stroke in atrial fibrillation",
             "Does vitamin D lower fracture risk in osteoporosis?",
             "pembrolizumab progression-free survival non-small cell lung cancer",
             "empagliflozin heart failure hospitalisation",
             "tamoxifen recurrence breast cancer"]


def build(n: int) -> dict:
    import library_db as L
    import library_vectors as LV
    import vector_store as vs
    random.seed(7)
    t0 = time.time()
    lib = L.Library()
    for i in range(n):
        d, c, o = random.choice(DRUGS), random.choice(CONDS), random.choice(OUTS)
        pt, dw = random.choice(DESIGNS)
        hr = round(random.uniform(0.5, 1.4), 2)
        lib.upsert({"pmid": str(10_000_000 + i), "title": f"Effect of {d} on {o} in {c}: a {dw} (study {i})",
                    "abstract": (f"Background: {d} may influence {o} in {c}. Methods: a {dw} of {random.randint(40, 12000)} "
                                 f"participants. Results: {d} changed {o} (hazard ratio {hr}, 95% CI {round(hr - 0.15, 2)} "
                                 f"to {round(hr + 0.15, 2)}). " + " ".join(random.sample(FILLER, 18)) + "."),
                    "year": random.randint(1995, 2026), "authors": [f"Author{random.randint(1, 50000)} A"],
                    "journal": "J Synthetic Med", "pub_types": [pt], "mesh": [c.title(), d.title()]}, source="pubmed")
        if i % 5000 == 4999:
            lib.commit()
            print(f"\r  records {i + 1:,}/{n:,}", end="", flush=True)
    lib.commit()
    lib.set_sync("last_update", datetime.now(timezone.utc).isoformat())
    t1 = time.time()
    print()
    vec = LV.LibraryVectors(lib.dir / "vectors", embedder=vs.HashingEmbedder(384))
    items = [{"record_id": r[0], "text": f"{r[1]} {r[2]}", "text_hash": str(r[0])}
             for r in lib.c.execute("SELECT id, title, abstract FROM records")]
    vec.add(items, batch=2000)
    t2 = time.time()
    out = {"records": n, "records_per_s": round(n / (t1 - t0)), "vectors_per_s": round(n / (t2 - t1)),
           "db_mb": round(lib.path.stat().st_size / 1e6), "vectors_mb": round(vec.info()["bytes"] / 1e6)}
    vec.close()
    lib.close()
    return out


def _time(fn, reps: int = 3):
    ts = []
    for _ in range(reps):
        for q in QUESTIONS:
            t = time.perf_counter()
            fn(q)
            ts.append((time.perf_counter() - t) * 1000)
    ts.sort()
    return statistics.median(ts), ts[int(0.95 * (len(ts) - 1))]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--records", type=int, default=200_000)
    ap.add_argument("--hnsw", action="store_true")
    ap.add_argument("--keep", action="store_true", help="keep the temporary library folder")
    a = ap.parse_args(argv)
    tmp = Path(tempfile.mkdtemp(prefix="biosyn_bench_"))
    os.environ.update({"BIOSYN_LIBRARY_DIR": str(tmp), "EMBED_BACKEND": "hashing", "BIOSYN_LOCAL_FIRST": "only"})
    sys.path.insert(0, str(ROOT))
    import vector_store as vs
    vs._EMBEDDER = vs.HashingEmbedder(384)
    import library_db as L
    import library_search as LS
    try:
        print(f"Building a synthetic library of {a.records:,} records in {tmp} ...")
        b = build(a.records)
        print(f"  {b['records_per_s']:,} records/s, {b['vectors_per_s']:,} vectors/s; library {b['db_mb']} MB + "
              f"vectors {b['vectors_mb']} MB")
        t = time.time()
        print(f"  warm-up (what the server does at start): {LS.warm()} ({time.time() - t:.1f}s)")
        import query_parser as qp
        qp.parse(QUESTIONS[0])                        # spelling vocabulary (also warmed at server start)
        lib = L.Library()
        vec = LS._vectors(lib.dir / "vectors")
        rows = [("keyword search (BM25, top 100)", lambda q: lib.search(q, k=100)),
                (f"semantic search (exact, {vec.count():,} vectors)", lambda q: vec.query(q, k=100)),
                ("hybrid local search", lambda q: LS.search(lib, q, k=100)),
                ("hybrid + 4 planned queries", lambda q: LS.search(lib, q, [q + " randomised", q + " meta-analysis",
                                                                             q + " cohort", q + " trial"], k=100)),
                ("local-first decision (identify)", lambda q: LS.identify(q, k=100)),
                ("lookup by PMID", lambda q: lib.by_ids(pmid="10000123"))]
        print(f"\n{'':44s}{'median':>10s}{'p95':>10s}")
        for name, fn in rows:
            med, p95 = _time(fn)
            print(f"{name:44s}{med:8.1f}ms{p95:8.1f}ms")
        if a.hnsw:
            r = vec.build_hnsw()
            if r.get("built"):
                vec._hnsw = None
                med, p95 = _time(lambda q: vec.query(q, k=100))
                print(f"{'semantic search (hnswlib)':44s}{med:8.1f}ms{p95:8.1f}ms")
            else:
                print("  hnswlib:", r.get("reason"))
        lib.close()
    finally:
        if not a.keep:
            for v in list(LS._VEC.values()):
                v[0].close()
            shutil.rmtree(tmp, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
