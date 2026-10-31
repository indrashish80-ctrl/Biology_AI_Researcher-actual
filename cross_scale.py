"""
cross_scale.py  (v25)
=====================
Cross-scale verification: molecular -> pathway -> cell -> tissue -> organism -> human. For the intervention the
question asks about, which LEVELS have verified evidence, and which LINKS between adjacent levels are actually
tested (the same intervention moves the same direction-consistent endpoint at both levels) versus assumed.

    level        what puts a finding there
    molecular    expression / phosphorylation / binding / enzyme activity of a named gene or protein (in silico too)
    pathway      a signalling pathway or a mechanism arrow (X via Y) read from the quote
    cell         cell-culture / organoid findings (viability, proliferation, apoptosis, migration ...)
    tissue       ex vivo tissue, biopsy markers (Ki-67 in a tumour sample, histology), organ-level measures in animals
    organism     whole-animal outcomes (tumour growth, survival, weight, behaviour)
    human        human trial / cohort findings on the asked endpoint

A link is
    tested       verified findings at BOTH levels for the same intervention, same endpoint family (or the mechanism
                 the higher level names), directions consistent
    discordant   both levels measured, directions opposite
    untested     one or both levels have no verified finding - the chain is an assumption there

    chain(matrix, intervention=None) -> {'levels': {level: [...]}, 'links': [...], 'weakest_link', 'summary'}
"""
from __future__ import annotations

import re
from collections import defaultdict
from typing import Any, Dict, List, Optional

SCALE_VERSION = "1.0"
LEVELS = ["molecular", "pathway", "cell", "tissue", "organism", "human"]
_MOLECULAR = re.compile(r"\b(?:expression|mrna|protein levels?|phosphorylat\w*|binding|enzyme activity|transcription|"
                        r"methylation|acetylation|ubiquitin\w*|kinase activity|docking|affinity)\b", re.I)
_PATHWAY = re.compile(r"\b(?:pathway|signal(?:l)?ing|axis|cascade|via|through|mediated by|dependent)\b", re.I)
_TISSUE = re.compile(r"\b(?:biopsy|biopsies|tissue|histolog\w*|immunohistochem\w*|ki-?67|tumou?r (?:sample|specimen|section)s?|"
                     r"organ|liver|kidney|muscle|adipose|skin) ", re.I)


def level_of(row: Dict[str, Any], f: Dict[str, Any]) -> str:
    mt = (f.get("model_type") or row.get("evidence_type") or "").lower()
    q = f"{f.get('quote') or ''} {f.get('measurement') or ''} {f.get('endpoint') or ''}"
    ep = (f.get("endpoint") or "").lower()
    if mt in ("in_silico",):
        return "molecular"
    if mt in ("human_clinical", "human_observational", "case_report"):
        if _TISSUE.search(q) and ep in ("proliferation", "apoptosis", "expression"):
            return "tissue"
        return "human"
    if mt in ("animal", "experimental_disease_model"):
        if ep in ("expression",) or _MOLECULAR.search(q) and ep not in ("tumour_growth", "survival", "body_weight"):
            return "molecular" if ep == "expression" else "tissue"
        return "organism"
    if mt in ("organoid_ex_vivo",):
        return "tissue"
    if mt in ("cell_culture", "microbial", "plant"):
        if ep == "expression" or (_MOLECULAR.search(q) and ep not in ("viability", "proliferation", "apoptosis", "migration")):
            return "molecular"
        if _PATHWAY.search(q) and ep in ("signalling", "pathway"):
            return "pathway"
        return "cell"
    return "cell"


def _consistent(d1: Optional[str], d2: Optional[str]) -> Optional[bool]:
    if d1 not in ("increase", "decrease") or d2 not in ("increase", "decrease"):
        return None
    return d1 == d2


def chain(matrix: Dict[str, Any], intervention: Optional[str] = None) -> Dict[str, Any]:
    by_level: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    iv = (intervention or "").lower()
    for r in (matrix or {}).get("rows") or []:
        if r.get("secondary_report"):
            continue
        for f in r.get("findings") or []:
            if iv and iv not in str(f.get("intervention") or r.get("intervention") or "").lower():
                continue
            lv = level_of(r, f)
            by_level[lv].append({"ref": r.get("ref"), "endpoint": f.get("endpoint"), "direction": f.get("direction"),
                                 "statement": (f.get("statement") or "")[:160], "model": r.get("model")})
            if f.get("mechanism") or (f.get("target") and _PATHWAY.search(f.get("quote") or "")):
                by_level["pathway"].append({"ref": r.get("ref"), "endpoint": f.get("target") or f.get("endpoint"),
                                            "direction": f.get("direction"), "statement": (f.get("quote") or "")[:160],
                                            "model": r.get("model")})
    links = []
    for a, b in zip(LEVELS, LEVELS[1:]):
        A, B = by_level.get(a) or [], by_level.get(b) or []
        if not A or not B:
            links.append({"from": a, "to": b, "status": "untested",
                          "why": f"no verified {a} finding" if not A else f"no verified {b} finding"})
            continue
        pairs = [(x, y) for x in A for y in B if x["endpoint"] and x["endpoint"] == y["endpoint"]]
        if not pairs:
            links.append({"from": a, "to": b, "status": "untested",
                          "why": "both levels measured, but not the same endpoint - the connection is assumed"})
            continue
        cons = [_consistent(x["direction"], y["direction"]) for x, y in pairs]
        if any(c is False for c in cons) and not any(c is True for c in cons):
            st = "discordant"
        elif any(c is True for c in cons):
            st = "tested"
        else:
            st = "untested"
        x, y = pairs[0]
        links.append({"from": a, "to": b, "status": st, "endpoint": x["endpoint"],
                      "why": f"{x['ref']} ({a}: {x['direction']}) vs {y['ref']} ({b}: {y['direction']})"})
    present = [lv for lv in LEVELS if by_level.get(lv)]
    weakest = next((l for l in links if l["status"] in ("discordant", "untested") and l["from"] in present
                    and l["to"] in present), None)
    tested = sum(l["status"] == "tested" for l in links)
    summary = (f"Evidence at {len(present)} of 6 levels ({', '.join(present) or 'none'}); {tested} of 5 links tested"
               + (f"; weakest link: {weakest['from']} -> {weakest['to']} ({weakest['status']})" if weakest else ""))
    return {"levels": {lv: by_level.get(lv, []) for lv in LEVELS}, "links": links, "weakest_link": weakest,
            "summary": summary, "version": SCALE_VERSION}


def markdown(c: Dict[str, Any]) -> str:
    if not c:
        return ""
    sym = {"tested": "✔ tested", "discordant": "✖ discordant", "untested": "… untested"}
    L = ["## Cross-scale verification", "", c["summary"], "", "| level | verified findings | link to next level |", "|---|---|---|"]
    links = {l["from"]: l for l in c["links"]}
    for lv in LEVELS:
        items = c["levels"].get(lv) or []
        cell = "; ".join(f"{x['ref']} {x['endpoint']} {x['direction']}" for x in items[:4]) + (f" (+{len(items) - 4})" if len(items) > 4 else "")
        l = links.get(lv)
        L.append(f"| {lv} | {cell or '—'} | {(sym[l['status']] + ': ' + l['why']) if l else ''} |")
    return "\n".join(L) + "\n"
