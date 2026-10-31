"""
evidence_record.py  (v22, improvement B)
----------------------------------------
ONE canonical evidence record with STABLE identifiers, built once from the evidence matrix. Every count a report
states about studies, experiments, outcomes (findings), claims and citations can be derived from it, and a
consistency check compares what the report actually printed against it.

Why stable IDs: the matrix labels (S1, E1, E1.F2) are positional - the same paper is S3 in one run and S5 in the next,
so two runs (or two report sections) cannot be compared. These IDs are content hashes:

    STU-xxxxxxxx           study       PMID, else DOI, else PMCID, else title+year
    EXP-xxxxxxxx-yyyyyy    experiment  study + model system + intervention
    OUT-xxxxxxxx-yyyyyy-zzzzzz  outcome  experiment + endpoint + target + direction + subgroup
    CIT-xxxxxxxx-wwwwww    citation    study + the exact quoted passage

This module is ADDITIVE and read-only with respect to the matrix: it never changes it.
"""
from __future__ import annotations

import hashlib
import os
import re
from collections import Counter
from typing import Any, Dict, List, Optional

ENABLED = os.environ.get("BIOSYN_EVIDENCE_RECORD", "1") != "0"
VERSION = 1


def _h(*parts: Any, n: int = 8) -> str:
    s = "|".join(re.sub(r"\s+", " ", str(p if p is not None else "")).strip().lower() for p in parts)
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:n]


def _norm_doi(d: Any) -> str:
    return re.sub(r"^(?:https?://(?:dx\.)?doi\.org/|doi:\s*)", "", str(d or "").strip(), flags=re.I).lower()


def study_id(row: Dict[str, Any]) -> str:
    ident = str(row.get("identifier") or "")
    m = re.match(r"PMID\s+(\d+)", ident)
    if m:
        return "STU-" + _h("pmid", m.group(1))
    if ident.lower().startswith("doi:"):
        return "STU-" + _h("doi", _norm_doi(ident))
    if re.match(r"PMC\d+", ident, re.I):
        return "STU-" + _h("pmcid", ident)
    return "STU-" + _h("title", row.get("title"), row.get("study"))


def build_record(matrix: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Canonical record from an evidence matrix (the dict build_matrix returns)."""
    matrix = matrix or {}
    studies, experiments, outcomes, citations = [], {}, [], {}
    claim_ids_seen = []
    for row in matrix.get("rows") or []:
        sid = study_id(row)
        studies.append({
            "study_id": sid, "ref": row.get("ref"), "paper_id": row.get("paper_id"), "identifier": row.get("identifier"),
            "title": row.get("title"), "study_label": row.get("study"), "evidence_type": row.get("evidence_type"),
            "bucket": row.get("bucket"), "level": row.get("level"), "secondary_report": bool(row.get("secondary_report")),
            "n_claims": row.get("n_claims") or 0, "n_quarantined": row.get("n_quarantined") or 0,
            "link": row.get("link"),
        })
        local = {}                                    # positional experiment label (E1) -> canonical id
        for f in row.get("findings") or []:
            elabel = f.get("experiment")
            if elabel not in local:
                eid = f"EXP-{sid[4:]}-" + _h(f.get("model_type"), f.get("model_detail"), f.get("intervention"), n=6)
                n = 2
                while eid in experiments and experiments[eid]["_label"] != (sid, elabel):   # same hash, different experiment
                    eid = f"EXP-{sid[4:]}-" + _h(f.get("model_type"), f.get("model_detail"), f.get("intervention"), n, n=6)
                    n += 1
                local[elabel] = eid
                experiments.setdefault(eid, {"experiment_id": eid, "study_id": sid, "model_type": f.get("model_type"),
                                             "model_detail": f.get("model_detail"), "intervention": f.get("intervention"),
                                             "_label": (sid, elabel)})
            eid = local[elabel]
            oid = f"OUT-{eid[4:]}-" + _h(f.get("endpoint"), f.get("target"), f.get("direction"), f.get("subgroup"), n=6)
            quote = f.get("quote") or ""
            cid = f"CIT-{sid[4:]}-" + _h(quote or f.get("statement"), n=6)
            if cid not in citations:
                citations[cid] = {"citation_id": cid, "study_id": sid, "quote": quote, "source_section": f.get("source_section"),
                                  "link": row.get("link"), "identifier": row.get("identifier")}
            claim_ids_seen += list(f.get("claim_ids") or [])
            outcomes.append({
                "outcome_id": oid, "experiment_id": eid, "study_id": sid, "citation_id": cid, "ref": f.get("id"),
                "endpoint": f.get("endpoint"), "direction": f.get("direction"), "p_value": f.get("p_value"),
                "claim_ids": list(f.get("claim_ids") or []), "statement": f.get("statement"),
                "answers_question": bool(f.get("answers_question")), "evidence_role": f.get("evidence_role") or "direct",
                "model_type": f.get("model_type"),
            })
    for e in experiments.values():
        e.pop("_label", None)
    rec = {"version": VERSION, "studies": studies, "experiments": list(experiments.values()), "outcomes": outcomes,
           "citations": list(citations.values())}
    rec["counts"] = counts(rec, matrix)
    return rec


def counts(rec: Dict[str, Any], matrix: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Every headline count, DERIVED from the record (the single source for report sections)."""
    st, ex, out = rec["studies"], rec["experiments"], rec["outcomes"]
    return {
        "papers": len(st),
        "primary_studies": sum(1 for s in st if not s["secondary_report"]),
        "experiments": len(ex),
        "independent_findings": len(out),
        "verified_claims": sum(s["n_claims"] for s in st),
        "verified_claims_primary": sum(s["n_claims"] for s in st if not s["secondary_report"]),
        "verified_claims_secondary": sum(s["n_claims"] for s in st if s["secondary_report"]),
        "findings_answering_question": sum(1 for o in out if o["answers_question"]),
        "findings_context_only": sum(1 for o in out if not o["answers_question"]),
        "quarantined_claims_in_included": sum(s["n_quarantined"] for s in st),
        "citations": len(rec["citations"]),
        "distinct_claim_ids": len({c for o in out for c in o["claim_ids"]}),
        "by_evidence_type": dict(Counter(s["evidence_type"] for s in st)),
        "by_direction": dict(Counter(o["direction"] or "unclear" for o in out)),
    }


# ----------------------------------------------------------------------------------------------- consistency
_ACC_RX = re.compile(r"\*\*Evidence accounting:\*\*\s*(\d+)\s+papers\s*\((\d+)\s+primary studies\)\s*·\s*(\d+)\s+experiments\s*·\s*"
                     r"\*\*(\d+)\s+independent findings\*\*\s*·\s*(\d+)\s+verified claims", re.S)


def parse_accounting(report_text: str) -> Optional[Dict[str, int]]:
    m = _ACC_RX.search(report_text or "")
    if not m:
        return None
    k = ("papers", "primary_studies", "experiments", "independent_findings", "verified_claims")
    return {a: int(b) for a, b in zip(k, m.groups())}


def check_consistency(rec: Dict[str, Any], matrix: Optional[Dict[str, Any]] = None,
                      report_text: str = "") -> List[Dict[str, Any]]:
    """Problems found (empty list = consistent). Each: {code, severity, detail}."""
    probs: List[Dict[str, Any]] = []

    def add(detail: str, severity: str = "major") -> None:
        probs.append({"code": "Q9", "severity": severity, "detail": detail, "action": None,
                      "repair": "flagged for the developer"})

    # 1. identifiers are unique and every reference resolves
    for key, idf in (("studies", "study_id"), ("experiments", "experiment_id"), ("outcomes", "outcome_id"),
                     ("citations", "citation_id")):
        c = Counter(x[idf] for x in rec[key])
        dup = [k for k, v in c.items() if v > 1]
        if dup:
            add(f"duplicate {idf}: {', '.join(dup[:3])}")
    sids = {s["study_id"] for s in rec["studies"]}
    eids = {e["experiment_id"] for e in rec["experiments"]}
    cids = {c["citation_id"] for c in rec["citations"]}
    for o in rec["outcomes"]:
        if o["study_id"] not in sids or o["experiment_id"] not in eids or o["citation_id"] not in cids:
            add(f"outcome {o['outcome_id']} points at a missing study / experiment / citation")
            break
    for o in rec["outcomes"]:
        if not o["claim_ids"]:
            add(f"outcome {o['outcome_id']} has no claim behind it (every finding must trace to a claim)", "critical")
            break
    # 2. the record's counts equal the matrix totals
    tot = (matrix or {}).get("totals") or {}
    cnt = rec["counts"]
    for k in ("papers", "primary_studies", "experiments", "independent_findings", "verified_claims",
              "findings_answering_question", "findings_context_only"):
        if k in tot and tot[k] != cnt[k]:
            add(f"matrix totals say {k}={tot[k]} but the evidence record derives {cnt[k]}", "critical")
    # 3. the numbers the report printed equal the record
    acc = parse_accounting(report_text)
    if acc:
        for k, v in acc.items():
            if v != cnt[k]:
                add(f"the report states {v} {k.replace('_', ' ')} but the evidence record derives {cnt[k]}", "critical")
    return probs


def markdown(rec: Dict[str, Any], problems: List[Dict[str, Any]]) -> str:
    c = rec["counts"]
    lines = ["\n## Evidence record (canonical IDs)\n",
             "_Every count in this report is derivable from one evidence record with stable identifiers "
             "(study STU-, experiment EXP-, outcome OUT-, citation CIT-) that do not change between runs._\n",
             f"- {c['papers']} studies ({c['primary_studies']} primary) · {c['experiments']} experiments · "
             f"{c['independent_findings']} outcomes · {c['citations']} citations · {c['distinct_claim_ids']} claims traced",
             ]
    lines.append("- Consistency check: " + ("all counts and references agree." if not problems else
                 f"{len(problems)} problem(s): " + "; ".join(p["detail"] for p in problems[:3])))
    return "\n".join(lines) + "\n"
