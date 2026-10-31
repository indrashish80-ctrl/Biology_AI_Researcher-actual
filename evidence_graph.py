"""
evidence_graph.py  (v25)
========================
Evidence dependency graph: which "independent" studies are not independent.

Edges between papers (each with its evidence):
    same_trial         publications of one trial (registry id, trial name, identical statistics - the matrix's own
                       trial grouping)
    shared_dataset     the same public dataset: GEO / SRA / ArrayExpress / dbGaP / PRIDE / BioProject accessions,
                       or TCGA / METABRIC / GTEx / CCLE / DepMap
    overlapping_cohort the same named cohort or registry (UK Biobank, NHANES, SEER, Framingham, Nurses' Health Study,
                       Women's Health Initiative, ARIC, MESA, CPRD, Danish / Swedish national registries, ...);
                       an overlap is likely, not certain, and is labelled so
    review_includes    a review / meta-analysis that includes (cites by author + year, PMID or DOI) a primary study
                       that the run ALSO counts - the same evidence counted twice

From the graph: clusters of mutually dependent papers (connected components over same_trial, shared_dataset and
overlapping_cohort) and the EFFECTIVE number of independent sources = number of clusters among primary papers.

    build(matrix, texts={ref: full text}) -> {'nodes', 'edges', 'clusters', 'effective_independent', 'double_counted'}
    mermaid(graph) -> a diagram for the report
"""
from __future__ import annotations

import re
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set

GRAPH_VERSION = "1.0"
ACCESSION = re.compile(r"\b(GSE\d{3,8}|GDS\d{3,6}|SR[APRX]\d{5,9}|ERP\d{5,9}|PRJ(?:NA|EB|DB)\d{3,9}|E-[A-Z]{4}-\d{1,6}|"
                       r"phs\d{6}(?:\.v\d+)?|PXD\d{6}|EGAS\d{8,11})\b")
NAMED_DATA = re.compile(r"\b(TCGA|The Cancer Genome Atlas|METABRIC|GTEx|CCLE|DepMap|GDSC|ICGC|TARGET|CPTAC)\b", re.I)
COHORTS = [
    ("UK Biobank", r"\bUK ?Biobank\b"), ("NHANES", r"\bNHANES\b|National Health and Nutrition Examination Survey"),
    ("SEER", r"\bSEER\b|Surveillance,? Epidemiology,? and End Results"), ("Framingham", r"\bFramingham\b"),
    ("Nurses' Health Study", r"Nurses'? Health Study|\bNHS ?II?\b"), ("Health Professionals Follow-up Study", r"Health Professionals Follow-?up"),
    ("Women's Health Initiative", r"Women'?s Health Initiative|\bWHI\b"), ("ARIC", r"\bARIC\b|Atherosclerosis Risk in Communities"),
    ("MESA", r"\bMESA\b|Multi-Ethnic Study of Atherosclerosis"), ("CPRD", r"\bCPRD\b|Clinical Practice Research Datalink|\bGPRD\b"),
    ("Danish national registries", r"Danish (?:National )?(?:Patient |Prescription |Cancer )?Regist"),
    ("Swedish national registries", r"Swedish (?:National )?(?:Patient |Prescribed Drug |Cancer )?Regist"),
    ("Taiwan NHIRD", r"\bNHIRD\b|National Health Insurance Research Database"), ("VA (Veterans Affairs)", r"Veterans (?:Affairs|Health Administration)"),
    ("Medicare / SEER-Medicare", r"\bSEER-Medicare\b|\bMedicare\b"), ("Rotterdam Study", r"Rotterdam Study"),
    ("EPIC", r"\bEPIC\b|European Prospective Investigation into Cancer"), ("Million Women Study", r"Million Women Study"),
    ("Kaiser Permanente", r"Kaiser Permanente"), ("China Kadoorie Biobank", r"China Kadoorie"), ("FinnGen", r"\bFinnGen\b"),
    ("Optum / MarketScan", r"\bOptum\b|MarketScan"), ("Hong Kong CDARS", r"\bCDARS\b"),
]


def _first_surname(authors: Any) -> Optional[str]:
    a = authors[0] if isinstance(authors, (list, tuple)) and authors else str(authors or "").split(",")[0]
    parts = [p for p in re.split(r"\s+", str(a).strip()) if p]
    if not parts:
        return None
    s = parts[0] if len(parts) > 1 and len(parts[-1]) <= 2 else parts[-1]
    s = re.sub(r"[^A-Za-z\-']", "", s)
    return s if len(s) >= 2 else None


def _year(r: Dict[str, Any]) -> Optional[str]:
    m = re.search(r"(19|20)\d{2}", str(r.get("year") or r.get("study") or ""))
    return m.group(0) if m else None


def extract_sources(text: str) -> Dict[str, Set[str]]:
    t = text or ""
    return {"datasets": {m.group(1).upper() for m in ACCESSION.finditer(t)} |
                        {m.group(1).upper().replace("THE CANCER GENOME ATLAS", "TCGA") for m in NAMED_DATA.finditer(t)},
            "cohorts": {name for name, rx in COHORTS if re.search(rx, t)}}


def _cites(review_text: str, target: Dict[str, Any]) -> Optional[str]:
    t = review_text or ""
    for key in ("pmid", "doi"):
        v = target.get(key)
        if v and str(v) in t:
            return f"{key} {v}"
    sn, yr = _first_surname(target.get("authors")), _year(target)
    if sn and yr and re.search(rf"\b{re.escape(sn)}\b(?:\s+et\s+al\.?|\s+(?:and|&)\s+[A-Z][\w'\-]+)?[^.;()]{{0,30}}?\(?{yr}[a-z]?\b", t):
        return f"cites {sn} {yr}"
    return None


def build(matrix: Dict[str, Any], texts: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    rows = (matrix or {}).get("rows") or []
    texts = texts or {}
    nodes, edges = [], []
    src: Dict[str, Dict[str, Set[str]]] = {}
    for r in rows:
        ref = r.get("ref")
        txt = " ".join(x for x in (texts.get(ref) or "", r.get("title") or "", r.get("quote") or "",
                                    " ".join(f.get("quote") or "" for f in r.get("findings") or [])) if x)
        src[ref] = extract_sources(txt)
        nodes.append({"ref": ref, "title": (r.get("title") or "")[:100], "design": r.get("evidence_type"),
                      "secondary": bool(r.get("secondary_report")), "trial_group": r.get("trial_group"),
                      "datasets": sorted(src[ref]["datasets"]), "cohorts": sorted(src[ref]["cohorts"])})
    by_ref = {r.get("ref"): r for r in rows}
    refs = [r.get("ref") for r in rows]
    for i, a in enumerate(refs):
        for b in refs[i + 1:]:
            ra, rb = by_ref[a], by_ref[b]
            if ra.get("trial_group") and ra.get("trial_group") == rb.get("trial_group"):
                edges.append({"a": a, "b": b, "type": "same_trial", "certainty": "established",
                              "why": ra.get("trial_basis") or f"trial group {ra['trial_group']}"})
            shared = src[a]["datasets"] & src[b]["datasets"]
            if shared:
                edges.append({"a": a, "b": b, "type": "shared_dataset", "certainty": "established",
                              "why": ", ".join(sorted(shared))})
            coh = src[a]["cohorts"] & src[b]["cohorts"]
            if coh:
                edges.append({"a": a, "b": b, "type": "overlapping_cohort", "certainty": "likely",
                              "why": ", ".join(sorted(coh))})
    double = []
    for r in rows:
        if not r.get("secondary_report"):
            continue
        rtext = texts.get(r.get("ref")) or ""
        for p in rows:
            if p is r or p.get("secondary_report"):
                continue
            why = _cites(rtext, p) if rtext else None
            if why:
                edges.append({"a": r.get("ref"), "b": p.get("ref"), "type": "review_includes", "certainty": "likely", "why": why})
                double.append({"review": r.get("ref"), "primary": p.get("ref"), "why": why})
    # clusters of dependent primary papers (union-find)
    parent = {x: x for x in refs}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for e in edges:
        if e["type"] in ("same_trial", "shared_dataset", "overlapping_cohort"):
            ra, rb = find(e["a"]), find(e["b"])
            if ra != rb:
                parent[rb] = ra
    groups: Dict[str, List[str]] = defaultdict(list)
    for x in refs:
        groups[find(x)].append(x)
    primary = [x for x in refs if not by_ref[x].get("secondary_report")]
    clusters = [g for g in groups.values() if len(g) > 1]
    effective = len({find(x) for x in primary})
    return {"nodes": nodes, "edges": edges, "clusters": clusters, "papers": len(refs), "primary": len(primary),
            "effective_independent": effective, "double_counted": double, "version": GRAPH_VERSION}


def mermaid(g: Dict[str, Any], max_edges: int = 60) -> str:
    style = {"same_trial": "==>", "shared_dataset": "-->", "overlapping_cohort": "-.->", "review_includes": "-->"}
    L = ["graph LR"]
    for n in g["nodes"]:
        shape = f"{n['ref']}[[\"{n['ref']} review\"]]" if n["secondary"] else f"{n['ref']}[\"{n['ref']}\"]"
        L.append(f"  {shape}")
    for e in g["edges"][:max_edges]:
        L.append(f"  {e['a']} {style.get(e['type'], '-->')}|{e['type'].replace('_', ' ')}| {e['b']}")
    return "\n".join(L)


def markdown(g: Dict[str, Any]) -> str:
    if not g or not g.get("nodes"):
        return ""
    L = ["## Evidence dependency graph", "",
         f"{g['primary']} primary paper(s) = **{g['effective_independent']} independent source(s)** after joining publications "
         f"of one trial, papers on one dataset and papers from one named cohort."]
    for e in g["edges"]:
        L.append(f"- {e['a']} ↔ {e['b']}: **{e['type'].replace('_', ' ')}** ({e['certainty']}) - {e['why']}")
    if g["double_counted"]:
        L.append("")
        L.append("Counted twice (a review that includes a primary study the run also counts): " +
                 "; ".join(f"{d['review']} includes {d['primary']}" for d in g["double_counted"]))
    if g["edges"]:
        L += ["", "```mermaid", mermaid(g), "```"]
    return "\n".join(L) + "\n"
