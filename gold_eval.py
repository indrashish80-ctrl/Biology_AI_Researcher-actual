"""
gold_eval.py  (v22, improvement A)
----------------------------------
Gold-standard evaluation. The audit system checks CONSISTENCY; this checks CORRECTNESS against papers a person has
read and verified. It only READS a finished dossier - it never calls the pipeline, so it cannot change a run.

Metrics
    study-classification accuracy   predicted evidence type == the verified type
    finding recall / missed         verified findings the pipeline found / failed to find
    finding precision               reported findings that match a verified finding
    false-positive claims           reported findings that match a verified NON-finding (a known trap) or nothing
    citation accuracy               reported quote really occurs in the paper (or in a verified quote)
    difficult cases                 the same numbers, per difficult-case tag (e.g. "abstract-only", "negative result")

Gold files:  gold/papers/*.json  (format: gold/README.md, template: gold/TEMPLATE.json)
Only files with  "status": "verified"  (a person checked every field against the paper) count in the headline score.
"synthetic" files test the scorer itself; "template" / "draft" files are ignored.

    python gold_eval.py --dossier research_dossier_3.json [--gold gold] [--json out.json]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

COUNTED = ("verified",)
HERE = Path(__file__).resolve().parent


def _norm(s: Any) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(s or "").lower()).strip()


def _doi(s: Any) -> str:
    return re.sub(r"^(?:https?://(?:dx\.)?doi\.org/|doi:\s*)", "", str(s or "").strip(), flags=re.I).lower()


def load_gold(folder: Any = None, statuses: tuple = COUNTED) -> List[Dict[str, Any]]:
    root = Path(folder) if folder else HERE / "gold"
    out = []
    for p in sorted((root / "papers").glob("*.json")) if (root / "papers").is_dir() else []:
        try:
            g = json.loads(p.read_text(encoding="utf-8"))
        except ValueError:
            continue
        if isinstance(g, dict) and g.get("status") in statuses:
            g["_file"] = p.name
            out.append(g)
    return out


def validate_gold(g: Dict[str, Any]) -> List[str]:
    """Problems with one gold file (so a bad file cannot silently distort the scores)."""
    errs = []
    try:
        from study_classifier import EVIDENCE_TYPES
        if g.get("study_type") not in EVIDENCE_TYPES:
            errs.append(f"study_type {g.get('study_type')!r} is not one of {sorted(EVIDENCE_TYPES)}")
    except Exception:
        pass
    p = g.get("paper") or {}
    if not (p.get("pmid") or p.get("doi") or p.get("pmcid") or p.get("title")):
        errs.append("paper needs a pmid, pmcid, doi or title")
    if g.get("status") == "verified" and not (g.get("verified_by") and g.get("verified_on")):
        errs.append("a 'verified' file must say who verified it (verified_by) and when (verified_on)")
    for i, f in enumerate(g.get("findings") or []):
        if not (f.get("endpoint") and f.get("direction")):
            errs.append(f"findings[{i}] needs endpoint and direction")
        if not f.get("quote"):
            errs.append(f"findings[{i}] needs the exact quote it was verified against")
    return errs


def _match_paper(row: Dict[str, Any], g: Dict[str, Any]) -> bool:
    p = g.get("paper") or {}
    ident = str(row.get("identifier") or "")
    if p.get("pmcid") and (re.search(rf"\b{re.escape(str(p['pmcid']))}\b", ident, re.I)
                           or str(row.get("pmcid") or "").upper() == str(p["pmcid"]).upper()):
        return True
    if p.get("pmid") and re.search(rf"\bPMID\s+{re.escape(str(p['pmid']))}\b", ident):
        return True
    if p.get("doi") and (_doi(p["doi"]) == _doi(ident) or _doi(p["doi"]) in _doi(row.get("link"))):
        return True
    return bool(p.get("title") and _norm(p["title"]) and _norm(p["title"]) == _norm(row.get("title")))


def _canon_endpoint(s: Any) -> str:
    try:
        from semantic_audit import canonical_outcome
        fam = canonical_outcome(str(s or ""))["family"]
        return fam if fam and not str(fam).startswith("other:") else _norm(s)
    except Exception:
        return _norm(s)


def _same_finding(pred: Dict[str, Any], gold: Dict[str, Any]) -> bool:
    pe, ge = _canon_endpoint(pred.get("endpoint")), _canon_endpoint(gold.get("endpoint"))
    pd_, gd = _norm(pred.get("direction")), _norm(gold.get("direction"))
    return bool(pe) and (pe == ge or _norm(gold.get("endpoint")) in _norm(pred.get("endpoint"))) and pd_ == gd


def _quote_ok(quote: Any, g: Dict[str, Any]) -> Optional[bool]:
    q = _norm(quote)
    if not q:
        return None
    src = _norm(g.get("source_text"))
    if src:
        return q in src
    gq = [_norm(f.get("quote")) for f in g.get("findings") or [] if f.get("quote")]
    return any(q in x or x in q for x in gq) if gq else None


def evaluate(rows: List[Dict[str, Any]], gold: List[Dict[str, Any]], only_in_run: bool = False) -> Dict[str, Any]:
    """rows = evidence_matrix['rows'] of a finished run. Returns metrics, per-paper detail and per-tag breakdown.

    only_in_run=True scores only the gold papers the run actually read (used after every research run: a gold paper
    on another topic is not a miss of THIS run); False also counts absent gold papers as missed (retrieval check)."""
    if only_in_run:
        gold = [g for g in gold if any(_match_paper(r, g) for r in rows)]
    per, cls_n, cls_ok = [], 0, 0
    tp = fp_nothing = fp_trap = missed = 0
    cit_n = cit_ok = 0
    tags: Dict[str, Dict[str, int]] = defaultdict(lambda: {"papers": 0, "missed": 0, "false_positive": 0, "found": 0})
    not_found = []
    for g in gold:
        row = next((r for r in rows if _match_paper(r, g)), None)
        tag_list = g.get("difficult_case") if isinstance(g.get("difficult_case"), list) else \
            ([g["difficult_case"]] if g.get("difficult_case") else ["ordinary"])
        if row is None:
            not_found.append(g.get("gold_id") or g.get("_file"))
            m = len(g.get("findings") or [])
            missed += m
            for t in tag_list:
                tags[t]["papers"] += 1
                tags[t]["missed"] += m
            per.append({"gold_id": g.get("gold_id") or g.get("_file"), "in_run": False, "missed": m})
            continue
        cls_n += 1
        type_ok = row.get("evidence_type") == g.get("study_type")
        cls_ok += type_ok
        preds = row.get("findings") or []
        found = missed_here = fp_here = 0
        used = set()
        for gf in g.get("findings") or []:
            hit = next((i for i, pf in enumerate(preds) if i not in used and _same_finding(pf, gf)), None)
            if hit is None:
                missed_here += 1
            else:
                used.add(hit)
                found += 1
        for i, pf in enumerate(preds):
            ok = _quote_ok(pf.get("quote"), g)
            if ok is not None:
                cit_n += 1
                cit_ok += bool(ok)
            if i in used:
                continue
            trap = any(_same_finding(pf, nf) for nf in g.get("non_findings") or [])
            if trap:
                fp_trap += 1
            else:
                fp_nothing += 1
            fp_here += 1
        tp += found
        missed += missed_here
        for t in tag_list:
            tags[t]["papers"] += 1
            tags[t]["found"] += found
            tags[t]["missed"] += missed_here
            tags[t]["false_positive"] += fp_here
        per.append({"gold_id": g.get("gold_id") or g.get("_file"), "in_run": True, "type_ok": type_ok,
                    "predicted_type": row.get("evidence_type"), "gold_type": g.get("study_type"),
                    "found": found, "missed": missed_here, "false_positive": fp_here})
    n_pred = tp + fp_nothing + fp_trap
    n_gold = tp + missed
    r = lambda a, b: round(a / b, 3) if b else None
    return {
        "gold_papers": len(gold), "gold_papers_in_run": cls_n, "gold_papers_not_in_run": not_found,
        "study_classification_accuracy": r(cls_ok, cls_n),
        "finding_recall": r(tp, n_gold), "finding_precision": r(tp, n_pred),
        "findings_found": tp, "findings_missed": missed,
        "false_positive_claims": fp_trap + fp_nothing, "false_positive_known_traps": fp_trap,
        "false_positive_unmatched": fp_nothing,
        "citation_accuracy": r(cit_ok, cit_n), "citations_checked": cit_n,
        "difficult_cases": dict(tags), "per_paper": per,
    }


def markdown(res: Dict[str, Any]) -> str:
    f = lambda v: "n/a" if v is None else f"{v:.0%}"
    lines = ["## Gold-standard evaluation\n",
             f"- Verified papers: {res['gold_papers']} ({res['gold_papers_in_run']} present in this run)",
             f"- Study-classification accuracy: {f(res['study_classification_accuracy'])}",
             f"- Finding recall: {f(res['finding_recall'])} ({res['findings_missed']} missed) · precision: {f(res['finding_precision'])}",
             f"- False-positive claims: {res['false_positive_claims']} (known traps hit: {res['false_positive_known_traps']})",
             f"- Citation accuracy: {f(res['citation_accuracy'])} over {res['citations_checked']} quotes"]
    for t, v in sorted(res["difficult_cases"].items()):
        lines.append(f"  - {t}: {v['papers']} paper(s), found {v['found']}, missed {v['missed']}, false positives {v['false_positive']}")
    return "\n".join(lines) + "\n"


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--dossier", required=True, help="research_dossier_<id>.json from a finished run")
    ap.add_argument("--gold", default=None, help="gold folder (default: ./gold)")
    ap.add_argument("--include-synthetic", action="store_true", help="also score synthetic scorer-test files")
    ap.add_argument("--json", default=None, help="write the full result here")
    a = ap.parse_args(argv)
    dossier = json.loads(Path(a.dossier).read_text(encoding="utf-8"))
    statuses = COUNTED + (("synthetic",) if a.include_synthetic else ())
    gold = load_gold(a.gold, statuses)
    bad = [(g.get("_file"), e) for g in gold for e in validate_gold(g)]
    for fn, e in bad:
        print(f"[gold] {fn}: {e}", file=sys.stderr)
    if not gold:
        print("No verified gold papers found. See gold/README.md - add papers a person has checked, status 'verified'.")
        return 2
    res = evaluate((dossier.get("evidence_matrix") or {}).get("rows") or [], gold)
    print(markdown(res))
    if a.json:
        Path(a.json).write_text(json.dumps(res, indent=2), encoding="utf-8")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
