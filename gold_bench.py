"""
gold_bench.py  (v25)
====================
Measure the engine against humans (principle 5) - a measured error rate instead of a claimed one.

Runs the engine's OWN components on the verified gold set (gold/papers/*.json, gold/questions.jsonl) and reports:

  A. direction reading   on the annotators' evidence sentences: does the engine's code read 'increase' /
                         'decrease' / 'no significant difference' the way the doctors did? (direction_in_text +
                         the significance rule + the arm rule that decides which group a change belongs to)
  B. extraction          the whole extraction step on each gold abstract (rules mode by default; --ai uses your keys):
                         finding recall, direction accuracy of found findings, citation accuracy (quote verbatim in the
                         paper), unsupported-claim rate (quote not in the paper, a number not in the quote, or a
                         direction that contradicts the annotators' label for the same sentence)
  C. retrieval           recall@1/5/10/20 and MRR of BM25, vectors, hybrid and hybrid + rerank over the gold corpus
  D. verdicts            the calibrated verdict categories on single-trial evidence sets built from the gold questions

    python gold_bench.py                    # A + B (rules) + C + D, writes gold/benchmark_results.json
    python gold_bench.py --only A           # one part
    python gold_bench.py --ai               # B with the AI extractor (GROQ_API_KEY / GEMINI_API_KEY / NVIDIA_NIM_API_KEY)

Every number is stored with the code versions that produced it (and appended to the metrics store), so a fix can be
shown to help - or not.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tempfile
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
os.environ.setdefault("BIOSYN_OFFLINE", "1")
os.environ.setdefault("BIOSYN_NO_WARMUP", "1")


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9%.<>=\- ]+", " ", (s or "").lower())).strip()


def load(gold_dir: Path = HERE / "gold"):
    import gold_eval as ge
    papers = ge.load_gold(gold_dir)
    qpath = gold_dir / "questions.jsonl"
    questions = [json.loads(x) for x in qpath.read_text(encoding="utf-8").splitlines() if x.strip()] if qpath.exists() else []
    cpath = gold_dir / "corpus" / "abstracts.jsonl"
    corpus = [json.loads(x) for x in cpath.read_text(encoding="utf-8").splitlines() if x.strip()] if cpath.exists() else []
    return papers, questions, corpus


# ------------------------------------------------------------------------------------------------ A. direction
_CHANGE_NOUN = re.compile(r"^\s*(?:the\s+|a\s+|>?\s*\d+\s*%\s+)?(?:reduction|decrease|decline|loss|fall|drop|lowering)s?\s+(?:in|of)\s+(.+)$", re.I)
_CHANGE_SUFFIX = re.compile(r"^\s*(?:>?\s*\d+\s*%\s+)?(.+?)\s+(?:reduction|decrease|decline|loss|fall|drop|lowering)s?\b(.*)$", re.I)


def read_direction(sentence: str, endpoint: str, intervention: str = "", comparator: str = "",
                   context: str = "") -> Dict[str, Any]:
    """The engine's reading of one evidence sentence, on the endpoint's OWN scale and relative to the comparator."""
    # an outcome phrased as a change ('Reduction in body weight') is read on its inner quantity and flipped back
    cn = _CHANGE_NOUN.match(endpoint or "")
    inner = cn.group(1) if cn else None
    if not cn:
        cs = _CHANGE_SUFFIX.match(endpoint or "")
        if cs and len(cs.group(1)) > 2:
            inner = (cs.group(1) + " " + cs.group(2)).strip()
    if inner and not cn:
        # 'blood loss', 'pain reductions': first read as ONE endpoint (its own change word is not a cue);
        # only when that finds nothing is it read as a change of the inner quantity ('weight loss was 6.6% vs 4.3%')
        whole = _read_core(sentence, endpoint, intervention, comparator, context, allow_fallback=False)
        if whole["direction"] not in ("none", None):
            return whole
    if inner:
        r = read_direction(sentence, inner, intervention, comparator, context)
        r["direction"] = {"increase": "decrease", "decrease": "increase"}.get(r["direction"], r["direction"])
        return r
    return _read_core(sentence, endpoint, intervention, comparator, context)


def _read_core(sentence: str, endpoint: str, intervention: str = "", comparator: str = "",
               context: str = "", allow_fallback: bool = True) -> Dict[str, Any]:
    import semantic_audit as s
    import arm_direction as ad
    oi = s.canonical_outcome(endpoint)
    sentence = ad.focus_clause(sentence, endpoint)        # the clause about this endpoint's time point / subgroup
    res = s.direction_in_text(sentence, oi, endpoint)
    raw = res.get("direction")
    located = bool(res.get("endpoint_found"))
    # direction_in_text answers on the endpoint family's canonical scale; the gold label is on the endpoint as named
    if raw in ("increase", "decrease") and int(oi.get("polarity", 1) or 1) < 0:
        raw = s.flip(raw)
    fallback = False
    if raw is None and allow_fallback:
        raw = s.sentence_direction(sentence)
        fallback = raw is not None
    stat = s.statistical_reading(sentence)
    if raw is None and stat.get("significant"):
        # a signed effect with its interval: 'body weight change was -6.4 kg (95% CI -9.4 to -3.4)'
        try:
            from meta_analysis import extract_effect
            eff = extract_effect(sentence)
            if eff and eff["kind"] == "difference" and eff["estimate"] != 0:
                raw = "increase" if eff["estimate"] > 0 else "decrease"
            elif eff and eff["kind"] == "ratio" and eff["estimate"] != 1:
                raw = "increase" if eff["estimate"] > 1 else "decrease"
        except Exception:
            pass
    arm = ad.relative_direction(sentence, raw, intervention, comparator, endpoint, stat.get("significant"), context)
    d = arm["direction"]
    if d in ("increase", "decrease") and stat.get("significant") is False:
        d = "no_effect"
    if d is None and stat.get("significant") is False:
        d = "no_effect"                                  # a reported non-significant test about this comparison
    return {"direction": d or "none", "raw": raw, "significant": stat.get("significant"), "arm": arm.get("rule"),
            "endpoint_located": located, "fallback": fallback}


def bench_direction(papers: List[Dict[str, Any]]) -> Dict[str, Any]:
    conf = Counter()
    by_tag: Dict[str, Counter] = defaultdict(Counter)
    errors = []
    located = fallback_n = fallback_ok = 0
    for g in papers:
        tags = g.get("difficult_case") or ["ordinary"]
        for f in g.get("findings") or []:
            r = read_direction(f["quote"], f["endpoint"], f.get("intervention", ""), f.get("comparator", ""),
                               g.get("source_text") or "")
            ok = r["direction"] == f["direction"]
            conf[(f["direction"], r["direction"])] += 1
            located += r["endpoint_located"]
            if r["fallback"]:
                fallback_n += 1
                fallback_ok += ok
            for t in tags:
                by_tag[t]["n"] += 1
                by_tag[t]["ok"] += ok
            if not ok and len(errors) < 60:
                errors.append({"paper": g["paper"].get("pmcid") or g["gold_id"], "endpoint": f["endpoint"],
                               "intervention": f.get("intervention"), "comparator": f.get("comparator"),
                               "gold": f["direction"], "engine": r["direction"], "raw": r["raw"], "arm_rule": r["arm"],
                               "sentence": f["quote"][:300]})
    n = sum(conf.values())
    ok = sum(v for (g, p), v in conf.items() if g == p)
    labels = ["increase", "decrease", "no_effect"]
    per = {}
    for lab in labels:
        tp = conf[(lab, lab)]
        fn = sum(v for (g, p), v in conf.items() if g == lab and p != lab)
        fp = sum(v for (g, p), v in conf.items() if p == lab and g != lab)
        per[lab] = {"precision": round(tp / (tp + fp), 3) if tp + fp else None,
                    "recall": round(tp / (tp + fn), 3) if tp + fn else None, "n": tp + fn}
    # the error that matters most: an effect read in the OPPOSITE direction
    opposite = conf[("increase", "decrease")] + conf[("decrease", "increase")]
    invented = conf[("no_effect", "increase")] + conf[("no_effect", "decrease")]
    return {"n": n, "accuracy": round(ok / n, 3) if n else None, "per_label": per,
            "opposite_direction": opposite, "effect_read_from_null": invented,
            "not_read": sum(v for (g, p), v in conf.items() if p in ("none", "unclear", "mixed")),
            "endpoint_located": located, "sentence_fallback": fallback_n,
            "sentence_fallback_accuracy": round(fallback_ok / fallback_n, 3) if fallback_n else None,
            "confusion": {f"{g}->{p}": v for (g, p), v in sorted(conf.items())},
            "by_tag": {t: {"n": c["n"], "accuracy": round(c["ok"] / c["n"], 3)} for t, c in sorted(by_tag.items())},
            "errors": errors}


def load_heldout(gold_dir: Path = HERE / "gold") -> List[Dict[str, Any]]:
    """The untouched held-out set as gold-paper-like dicts (doctor-annotated, never used for tuning)."""
    p = gold_dir / "heldout" / "ei_heldout.jsonl"
    out = []
    for line in (p.read_text(encoding="utf-8").splitlines() if p.exists() else []):
        if line.strip():
            r = json.loads(line)
            out.append({"gold_id": "heldout-" + r["pmcid"], "paper": {"pmcid": r["pmcid"], "title": r["title"]},
                        "source_text": r["abstract"], "findings": r["findings"], "difficult_case": ["heldout"]})
    return out


# ------------------------------------------------------------------------------------------------ B. extraction
def _paper_db(g: Dict[str, Any], tmp: Path):
    from database import ResearchDB
    db = ResearchDB(str(tmp / f"{g['gold_id']}.db"))
    pid = db.upsert_paper(None, {"title": g["paper"].get("title"), "abstract": g["source_text"],
                                 "pmcid": g["paper"].get("pmcid"), "doi": g["paper"].get("doi"), "year": "2016",
                                 "authors": "", "journal": ""}, source_query="gold")
    db.save_paper_section(pid, "abstract", "Abstract", g["source_text"])
    return db, pid


def _router_from_env():
    if not any(os.getenv(k) for k in ("GROQ_API_KEY", "GEMINI_API_KEY", "NVIDIA_NIM_API_KEY")):
        return None
    from ai_router import AIRouter
    return AIRouter()


def bench_extraction(papers: List[Dict[str, Any]], use_ai: bool = False, limit: Optional[int] = None) -> Dict[str, Any]:
    import quote_guard as qg
    from claim_extraction import ClaimExtractor, guard_summary
    router = _router_from_env() if use_ai else None
    tmp = Path(tempfile.mkdtemp(prefix="biosyn_gold_"))
    n_gold = found = dir_ok = 0
    claims_total = verbatim = unsupported = matched_claims = 0
    misses, wrong = [], []
    t0 = time.time()
    for g in papers[:limit] if limit else papers:
        db, pid = _paper_db(g, tmp)
        try:
            q = None
            ClaimExtractor(db, router).extract(pid, None, q, force=True, max_claims=12)
            claims = db.claims_for_paper(pid)
        finally:
            pass
        src = g["source_text"]
        used = set()
        for c in claims:
            claims_total += 1
            if qg.locate(c.get("quote") or "", src):
                verbatim += 1
            else:
                unsupported += 1
        for f in g.get("findings") or []:
            n_gold += 1
            ev = _norm(f["quote"])
            hit = None
            for i, c in enumerate(claims):
                cq = _norm(c.get("quote") or "")
                if cq and (cq in ev or ev in cq or (len(cq) > 40 and cq[:40] in ev)):
                    hit = (i, c)
                    break
            if hit is None:
                if len(misses) < 40:
                    misses.append({"paper": g["paper"].get("pmcid"), "endpoint": f["endpoint"], "evidence": f["quote"][:200]})
                continue
            found += 1
            used.add(hit[0])
            # the claim's direction, read for THIS endpoint the way the engine's audit reads it
            r = read_direction(hit[1].get("quote") or "", f["endpoint"], f.get("intervention", ""), f.get("comparator", ""),
                               src)
            if r["direction"] == f["direction"]:
                dir_ok += 1
            elif r["direction"] in ("increase", "decrease", "no_effect"):
                unsupported += 1
                if len(wrong) < 40:
                    wrong.append({"paper": g["paper"].get("pmcid"), "endpoint": f["endpoint"], "gold": f["direction"],
                                  "engine": r["direction"], "quote": (hit[1].get("quote") or "")[:200]})
        matched_claims += len(used)
        db.close()
    rate = lambda a, b: round(a / b, 3) if b else None
    return {"mode": "ai" if router else "rules", "papers": len(papers[:limit] if limit else papers),
            "gold_findings": n_gold, "found": found, "finding_recall": rate(found, n_gold),
            "direction_accuracy_found": rate(dir_ok, found),
            "claims_extracted": claims_total, "claims_matching_gold": matched_claims,
            "citation_accuracy": rate(verbatim, claims_total),
            "unsupported_claim_rate": rate(unsupported, claims_total),
            "note": "Evidence Inference annotates selected prompts, not every result: a claim that matches no gold finding is "
                    "'unannotated', not counted as false.",
            "guard": guard_summary(), "seconds": round(time.time() - t0, 1), "misses": misses, "wrong": wrong}


# ------------------------------------------------------------------------------------------------ C / D
def bench_retrieval(questions, corpus) -> Dict[str, Any]:
    import hybrid_retrieval as hr
    return hr.evaluate(questions, corpus)


def bench_verdicts(questions) -> Dict[str, Any]:
    import verdict_calibration as vc
    return vc.gold_check(questions)


def versions() -> Dict[str, str]:
    out = {}
    for mod, attr in (("semantic_audit", "AUDIT_VERSION"), ("claim_extraction", "EXTRACTOR_VERSION"),
                      ("quote_guard", "GUARD_VERSION"), ("hybrid_retrieval", "RETRIEVAL_VERSION"),
                      ("query_parser", "PARSER_VERSION"), ("arm_direction", "ARM_VERSION"),
                      ("verdict_calibration", "VERDICT_VERSION"), ("study_classifier", "CLASSIFIER_VERSION")):
        try:
            out[mod] = str(getattr(__import__(mod), attr))
        except Exception:
            pass
    return out


def markdown(res: Dict[str, Any]) -> str:
    f = lambda v: "n/a" if v is None else f"{v:.1%}"
    L = ["# Gold-standard benchmark", "", f"Run {res['when']} · code versions: " +
         ", ".join(f"{k} {v}" for k, v in res["versions"].items()), ""]
    A = res.get("A")
    if A:
        L += ["## A. Direction reading (engine code vs the annotators)", "",
              f"- {A['n']} evidence sentences · accuracy **{f(A['accuracy'])}**",
              f"- opposite-direction errors: **{A['opposite_direction']}** · effect read from a null result: "
              f"{A['effect_read_from_null']} · not read: {A['not_read']}",
              f"- endpoint located in the sentence: {A['endpoint_located']} of {A['n']}; sentence-level fallback used "
              f"{A['sentence_fallback']} times (accuracy {f(A['sentence_fallback_accuracy'])})", "",
              "| label | precision | recall | n |", "|---|---|---|---|"]
        L += [f"| {k} | {f(v['precision'])} | {f(v['recall'])} | {v['n']} |" for k, v in A["per_label"].items()]
        L += ["", "| difficult case | n | accuracy |", "|---|---|---|"]
        L += [f"| {t} | {v['n']} | {f(v['accuracy'])} |" for t, v in A["by_tag"].items()]
        L.append("")
    H = res.get("A_heldout")
    if H:
        L += ["### A'. The same, on the UNTOUCHED held-out set (never used to tune the engine)", "",
              f"- {H['n']} doctor-annotated evidence sentences · accuracy **{f(H['accuracy'])}** · "
              f"opposite-direction errors **{H['opposite_direction']}** ({f(H['opposite_direction'] / H['n'] if H['n'] else None)}) · "
              f"effect read from a null result {H['effect_read_from_null']} · not read {H['not_read']}",
              "- This is the number to quote: the development set (A) was used to find and fix reader errors, so its "
              "accuracy is optimistic.", ""]
    B = res.get("B")
    if B:
        L += [f"## B. Extraction ({B['mode']} mode, {B['papers']} abstracts)", "",
              f"- finding recall **{f(B['finding_recall'])}** ({B['found']} of {B['gold_findings']})",
              f"- direction accuracy of found findings **{f(B['direction_accuracy_found'])}**",
              f"- citation accuracy (quote verbatim in the paper) **{f(B['citation_accuracy'])}** over {B['claims_extracted']} claims",
              f"- unsupported-claim rate **{f(B['unsupported_claim_rate'])}**",
              f"- guard: {B['guard'].get('dropped', 0)} of {B['guard'].get('proposed', 0)} proposed findings dropped "
              f"(no verbatim quote / number not in the paper)"]
        if B["mode"] != "ai":
            L.append("- rules mode reads only the sentences the no-AI fallback recognises, so recall is low by design; it "
                     "measures the fallback, NOT the AI extractor. Run `python gold_bench.py --only B --ai` with a key to "
                     "measure real extraction recall and the guard's drop rate.")
        L.append("")
    C = res.get("C")
    if C:
        L += [f"## C. Retrieval ({C['questions']} questions, {C['corpus']} abstracts, embedder {C.get('embedder')}, "
              f"reranker {C.get('reranker')})", "", "| mode | R@1 | R@5 | R@10 | R@20 | MRR |", "|---|---|---|---|---|---|"]
        for m, v in C["modes"].items():
            L.append(f"| {m} | {f(v['recall@1'])} | {f(v['recall@5'])} | {f(v['recall@10'])} | {f(v['recall@20'])} | {v['mrr']:.3f} |")
        L.append("")
    D = res.get("D")
    if D:
        L += ["## D. Verdict calibration", "", f"- {D['cases']} cases · rule violations: **{D['violations']}**",
              "- " + ", ".join(f"{k}: {v}" for k, v in sorted(D["distribution"].items())), ""]
    L += ["Caveats: the gold questions reuse the annotators' wording of intervention / comparator / outcome, so retrieval "
          "is an easy lexical test; all gold papers are randomised trials (Evidence Inference), so cell and animal "
          "evidence is covered only by the metamorphic fuzzer and the replayed dossiers in tests/."]
    return "\n".join(L) + "\n"


def run(parts: str = "ABCD", use_ai: bool = False, limit: Optional[int] = None) -> Dict[str, Any]:
    papers, questions, corpus = load()
    res: Dict[str, Any] = {"when": time.strftime("%Y-%m-%d %H:%M"), "versions": versions(),
                           "gold_papers": len(papers), "gold_questions": len(questions)}
    if "A" in parts:
        res["A"] = bench_direction(papers)
        held = load_heldout()
        if held:
            h = bench_direction(held)
            h.pop("by_tag", None)
            res["A_heldout"] = h
    if "B" in parts:
        res["B"] = bench_extraction(papers, use_ai, limit)
    if "C" in parts and questions and corpus:
        res["C"] = bench_retrieval(questions, corpus)
    if "D" in parts and questions:
        res["D"] = bench_verdicts(questions)
    try:
        import run_metrics
        run_metrics.record_accuracy(res)
    except Exception:
        pass
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Gold-standard benchmark")
    ap.add_argument("--only", default="ABCD")
    ap.add_argument("--ai", action="store_true")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--out", default=str(HERE / "gold" / "benchmark_results.json"))
    a = ap.parse_args(argv)
    res = run(a.only.upper(), a.ai, a.limit)
    Path(a.out).write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
    md = markdown(res)
    Path(a.out).with_suffix(".md").write_text(md, encoding="utf-8")
    print(md)
    return 0


if __name__ == "__main__":
    sys.exit(main())
