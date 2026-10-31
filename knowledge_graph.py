"""
knowledge_graph.py  (v25)
=========================
Wires up the (until now empty) entity_relations table: every VERIFIED claim becomes a typed edge between two
entities, with the RELATION TYPE and the EVIDENCE TYPE on the edge, plus the claim and paper it came from.

    edge  intervention --[increases | decreases | no_effect_on | associated_with | modulates]--> endpoint
          evidence_type: human_clinical / human_observational / animal / cell_culture / in_silico / review / database
          direction, significance, claim_id, paper_id, research_question_id, quote (provenance)
    edge  protein --[interacts_with]--> protein                  (STRING; evidence type 'database:STRING')
    edge  intervention --[acts_via]--> mediator                   (a mechanism the quote names: 'X via Y')

Entities are de-duplicated by normalised name (and accession when the entity linker resolved one).

    build(db, research_question_id, string_net=None) -> {'entities', 'edges'}
    neighbours(db, name) / edges_for(db, research_question_id) / export(db, rq) for the website
"""
from __future__ import annotations

import json
import re
import sqlite3
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

KG_VERSION = "1.0"
_EXTRA_COLS = (("relation_type", "TEXT"), ("evidence_type", "TEXT"), ("direction", "TEXT"), ("significant", "INTEGER"),
               ("claim_id", "INTEGER"), ("paper_id", "INTEGER"), ("research_question_id", "INTEGER"),
               ("evidence_json", "TEXT"))
REL = {"increase": "increases", "decrease": "decreases", "no_effect": "no_effect_on", "mixed": "modulates",
       "unclear": "associated_with"}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def ensure(db: Any) -> None:
    for col, typ in _EXTRA_COLS:
        try:
            db._conn.execute(f"ALTER TABLE entity_relations ADD COLUMN {col} {typ}")
        except sqlite3.OperationalError:
            pass
    db._conn.execute("CREATE INDEX IF NOT EXISTS idx_rel_rq ON entity_relations(research_question_id)")
    db._conn.commit()


def _norm(name: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w\-+/ ]", " ", (name or "").lower())).strip()


def entity(db: Any, etype: str, name: str, accession: Optional[str] = None, source_db: str = "biosynthetica") -> Optional[int]:
    n = _norm(name)
    if not n or len(n) < 2:
        return None
    return db.upsert_entity(etype, source_db, accession or f"{etype}:{n}", symbol=name[:60], name=name[:200])


def _edge(db: Any, a: int, rel: str, b: int, source_db: Optional[str] = None, **kw) -> Optional[int]:
    """One edge per (claim, relation): the table's UNIQUE(from, relation, to, source_db) is kept distinct per claim by
    a source key 'claim:<id>', so two papers' identical findings stay two edges with their own provenance."""
    src = source_db or (f"claim:{kw['claim_id']}" if kw.get("claim_id") else "biosynthetica")
    cols = ["from_entity_id", "relation", "to_entity_id", "source_db", "created_at", "relation_type"] + list(kw)
    vals = [a, rel, b, src, _now(), rel] + list(kw.values())
    try:
        cur = db._conn.execute(f"INSERT OR IGNORE INTO entity_relations ({', '.join(cols)}) "
                               f"VALUES ({', '.join('?' * len(cols))})", vals)
        return cur.lastrowid if cur.rowcount else None
    except sqlite3.IntegrityError:
        return None


def build(db: Any, research_question_id: Optional[int], string_net: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    ensure(db)
    claims = db.verified_claims(research_question_id)
    n_edges = 0
    ents = set()
    for c in claims:
        au = c.get("semantic_audit") or {}
        iv = c.get("intervention") or c.get("subject")
        ep = au.get("canonical_outcome") or c.get("measurement")
        if not iv or not ep or str(ep).startswith("other:") and not c.get("measurement"):
            continue
        a = entity(db, "intervention", iv)
        b = entity(db, "endpoint", str(ep).replace("other:", "") if str(ep).startswith("other:") else str(ep))
        if not a or not b:
            continue
        ents |= {a, b}
        d = au.get("canonical_direction") or c.get("effect_direction") or "unclear"
        ev = au.get("claim_model") or c.get("study_design") or "unclear"
        if au.get("evidence_role") == "secondary_report":
            ev = "review"
        rel = REL.get(d, "associated_with")
        if au.get("inference_type") == "statistical_association" and rel in ("increases", "decreases"):
            rel = "associated_with_" + rel
        sig = None if c.get("p_value") is None else int(float(c["p_value"]) < 0.05)
        if _edge(db, a, rel, b, evidence_type=ev, direction=d, significant=sig, claim_id=c["id"], paper_id=c.get("paper_id"),
                 research_question_id=research_question_id,
                 evidence_json=json.dumps({"quote": (c.get("quote") or "")[:400], "paper": c.get("paper_title"),
                                           "pmid": c.get("pmid"), "doi": c.get("doi")}, ensure_ascii=False)):
            n_edges += 1
        med = (au.get("mechanism") or {}).get("mediator") if isinstance(au.get("mechanism"), dict) else None
        if med:
            m = entity(db, "mediator", med)
            if m and _edge(db, a, "acts_via", m, evidence_type=ev, direction=d, claim_id=c["id"], paper_id=c.get("paper_id"),
                           research_question_id=research_question_id):
                n_edges += 1
    for e in (string_net or {}).get("edges") or []:
        a = entity(db, "protein", e.get("a") or "", source_db="STRING")
        b = entity(db, "protein", e.get("b") or "", source_db="STRING")
        if a and b and _edge(db, a, "interacts_with", b, evidence_type="database:STRING", source_db="STRING",
                             research_question_id=research_question_id,
                             evidence_json=json.dumps({"score": e.get("score")})):
            n_edges += 1
            ents |= {a, b}
    db._conn.commit()
    return {"entities": len(ents), "edges": n_edges, "claims": len(claims), "version": KG_VERSION}


def edges_for(db: Any, research_question_id: Optional[int] = None, limit: int = 2000) -> List[Dict[str, Any]]:
    ensure(db)
    sql = ("SELECT r.*, a.name AS from_name, a.entity_type AS from_type, b.name AS to_name, b.entity_type AS to_type "
           "FROM entity_relations r JOIN biological_entities a ON a.id = r.from_entity_id "
           "JOIN biological_entities b ON b.id = r.to_entity_id")
    args: tuple = ()
    if research_question_id is not None:
        sql += " WHERE r.research_question_id = ?"
        args = (research_question_id,)
    rows = [dict(r) for r in db._conn.execute(sql + " LIMIT ?", args + (limit,)).fetchall()]
    for r in rows:
        r["evidence"] = json.loads(r.pop("evidence_json") or "null")
    return rows


def neighbours(db: Any, name: str) -> List[Dict[str, Any]]:
    n = _norm(name)
    return [e for e in edges_for(db) if _norm(e["from_name"]) == n or _norm(e["to_name"]) == n]


def summary(edges: List[Dict[str, Any]]) -> Dict[str, Any]:
    from collections import Counter
    return {"edges": len(edges), "by_relation": dict(Counter(e["relation"] for e in edges)),
            "by_evidence_type": dict(Counter(e.get("evidence_type") or "?" for e in edges))}


def markdown(edges: List[Dict[str, Any]], limit: int = 25) -> str:
    if not edges:
        return ""
    s = summary(edges)
    L = ["## Knowledge graph", "", f"{s['edges']} typed edge(s): " + ", ".join(f"{k} {v}" for k, v in s["by_relation"].items())
         + " · evidence: " + ", ".join(f"{k} {v}" for k, v in s["by_evidence_type"].items()), "",
         "| from | relation | to | evidence type | claim |", "|---|---|---|---|---|"]
    for e in edges[:limit]:
        L.append(f"| {e['from_name']} | {e['relation']} | {e['to_name']} | {e.get('evidence_type') or ''} | "
                 f"{('C' + str(e['claim_id'])) if e.get('claim_id') else ''} |")
    return "\n".join(L) + "\n"
