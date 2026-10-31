"""
preread.py  (v26)
=================
QUESTION-INDEPENDENT extraction: the AI reads each library paper ONCE and tabulates every result it reports; later
questions reuse that reading and select the relevant findings by CODE (no AI call for papers already read).

    run(lib, limit=200, designs=['rct','meta'])        background / overnight job (library_loader preread)
    lookup(paper, question) -> [findings] | None       used by claim_extraction before any AI call

Every pre-read finding passes the same checks as a question-time finding:
  * quote_guard: the quote must be verbatim in the paper, every number in the finding must be in the quote;
  * a second, independent reader (independent_reader, a different prompt / provider) when BIOSYN_SECOND_READER is not 0
    - its agreement is stored with each finding, so question time does not need to re-read the paper.
Cache key: library record + PREREAD_VERSION (which includes the extractor version) + a hash of the text read. A new
extractor version, or a changed full text, makes the paper be read again; a correction / retraction is handled by the
publication-status layer as for any paper.
"""
from __future__ import annotations

import hashlib
import os
import re
from typing import Any, Dict, List, Optional, Sequence

PREREAD_BASE = "preread-1"


def version() -> str:
    from claim_extraction import EXTRACTOR_VERSION
    return f"{PREREAD_BASE}/x{EXTRACTOR_VERSION}"


def prompt(text: str, n_max: int = 12) -> str:
    from claim_extraction import prompt_claims
    rules = prompt_claims("Article", "", None, "", n_max=n_max).split("- Return ONLY valid JSON:")[0]
    return (rules.replace(f"Extract up to {n_max} core biological findings (include null results).",
                          f"Extract EVERY result this study reports, up to {n_max} (include null results, secondary "
                          f"endpoints and subgroup results). This reading will be reused for many different questions.")
            + "- Also give, when the text states them: \"comparator\", \"population\" (patients / animals / cells), "
              "\"timepoint\", \"subgroup\", \"effect\" (the effect size as written) and \"n\".\n"
            + "- Return ONLY valid JSON:\n"
            + '{"claims": [{"claim": "...", "subject": "...", "comparator": "...", "population": "...", "outcome": "...",'
              ' "direction": "increase", "p_value": 0.05, "effect": "...", "timepoint": "...", "subgroup": null, "n": null,'
              ' "quote": "verbatim text"}]}\n\n'
            + "TEXT:\n" + text)


def _source(lib: Any, rid: int) -> str:
    paras = lib.passages(rid)
    keep = [p["text"] for p in paras if (p.get("section") or "") not in ("references", "introduction")]
    if not keep:
        r = lib.get(rid) or {}
        keep = [r.get("abstract") or ""]
    return "\n\n".join(keep)


def _thash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:20]


def _design_ok(rec: Dict[str, Any], designs: Sequence[str]) -> bool:
    if not designs:
        return True
    pts = " ".join(rec.get("pub_types") or []).lower() + " " + (rec.get("title") or "").lower()
    want = {"rct": r"randomi[sz]ed controlled trial|randomi[sz]ed", "meta": r"meta-analysis|systematic review",
            "trial": r"clinical trial", "cohort": r"cohort|observational", "any": r"."}
    return any(re.search(want.get(d, re.escape(d)), pts) for d in designs)


def read_one(lib: Any, rid: int, router: Any) -> Dict[str, Any]:
    import quote_guard as qg
    from claim_extraction import validate_claims
    from llm_json import parse_json_response
    text = _source(lib, rid)
    if len(text) < 200:
        return {"record_id": rid, "status": "too_short"}
    th = _thash(text)
    if lib.get_findings(rid, version(), th) is not None:
        return {"record_id": rid, "status": "cached"}
    rec = lib.get(rid) or {}
    basis = "jats_xml" if rec.get("fulltext_kind") == "jats_xml" else ("full_text_pdf" if rec.get("fulltext_kind") == "pdf"
                                                                       else "abstract")
    chunks = [text[i:i + 9000] for i in range(0, min(len(text), 27000), 9000)]
    claims: List[Dict[str, Any]] = []
    for ch in chunks:
        try:
            resp = router.ask(prompt(ch), task="extraction")
            obj = parse_json_response(resp)
            raw = obj.get("claims") if isinstance(obj, dict) else None
            for c in validate_claims(obj).get("claims", []):
                # validate_claims keeps the core fields; the extra pre-read fields are copied from the raw item
                src = next((r for r in raw or [] if isinstance(r, dict) and (r.get("quote") or "").strip().strip('"\'')[:60]
                            == (c.get("quote") or "")[:60]), {})
                for k in ("comparator", "population", "timepoint", "subgroup", "effect", "n"):
                    if src.get(k) not in (None, ""):
                        c[k] = src[k]
                claims.append(c)
        except Exception as exc:
            lib.log(rid, "preread", "error", version(), f"{type(exc).__name__}: {str(exc)[:300]}")
    kept, dropped = [], []
    for c in claims:
        g = qg.guard_claim(c, text, basis)
        (kept if g["keep"] else dropped).append(g["claim"] if g["keep"] else {**g["claim"], "drop_reason": g["reason"]})
    reader = None
    try:
        import independent_reader as ir
        if kept and ir.enabled(False):
            second = ir.second_read(router, chunks, None, text, basis)
            reader = ir.compare(kept, second)
    except Exception as exc:
        reader = {"error": f"{type(exc).__name__}: {str(exc)[:120]}"}
    for c in kept:
        c["preread"] = version()
        c["text_basis"] = basis
        c.setdefault("second_reader", "not_run" if reader is None or "error" in (reader or {}) else c.get("second_reader"))
    lib.put_findings(rid, version(), th, kept, provider=",".join(getattr(router, "configured_providers", lambda: [])()))
    lib.log(rid, "preread", "ok", version(), f"{len(claims)} proposed, {len(kept)} kept, {len(dropped)} dropped")
    return {"record_id": rid, "status": "ok", "proposed": len(claims), "kept": len(kept), "dropped": len(dropped)}


def run(lib: Any, limit: int = 200, designs: Sequence[str] = ("rct", "meta"), router: Any = None) -> Dict[str, Any]:
    if router is None:
        from ai_router import AIRouter
        router = AIRouter()
        if not router.configured_providers():
            return {"read": 0, "note": "no AI key configured (GROQ_API_KEY / GEMINI_API_KEY / NVIDIA_NIM_API_KEY)"}
    done = {r[0] for r in lib.c.execute("SELECT record_id FROM findings WHERE extractor_version=?", (version(),))}
    todo = []
    for r in lib.c.execute("SELECT id FROM records WHERE deleted=0 AND collection='literature' AND status != 'retracted' "
                           "ORDER BY (fulltext_path IS NULL), id DESC"):
        if r[0] in done:
            continue
        rec = lib.get(r[0])
        if rec and _design_ok(rec, designs):
            todo.append(r[0])
        if len(todo) >= limit:
            break
    out = [read_one(lib, rid, router) for rid in todo]
    return {"read": sum(1 for o in out if o["status"] == "ok"), "skipped": sum(1 for o in out if o["status"] != "ok"),
            "kept": sum(o.get("kept", 0) for o in out), "dropped": sum(o.get("dropped", 0) for o in out),
            "version": version()}


# ------------------------------------------------------------------------------------------------ question time
def select_for_question(findings: List[Dict[str, Any]], question: Optional[str]) -> List[Dict[str, Any]]:
    """The pre-read findings that concern the question: the finding names the question's intervention / exposure OR its
    outcome (synonyms included). With no usable question terms, all findings are returned."""
    if not question:
        return list(findings)
    import query_parser as qp
    p = qp.parse(question)
    concepts = qp.search_concepts(p)
    roles: Dict[str, List[set]] = {}
    for c in concepts:                              # a role can appear more than once (e.g. Ki-67 AND the disease)
        roles.setdefault(c.get("role"), []).append({t.lower() for t in c.get("terms") or []})
    iv = set().union(*(roles.get("intervention") or []), *(roles.get("exposure") or []))
    disease = set().union(*(roles.get("disease") or []))
    outs = roles.get("outcome") or []
    specific = [g for g in outs if g != disease]    # the disease is the outcome only when nothing more specific is asked
    oc = set().union(*(specific or outs))
    if not iv and not oc:
        return list(findings)
    out = []
    for f in findings:
        blob = " ".join(str(f.get(k) or "") for k in ("subject", "intervention", "outcome", "statement", "quote")).lower()
        hit_iv = any(t and t in blob for t in iv)
        hit_oc = any(t and t in blob for t in oc)
        if (hit_iv and (hit_oc or not oc)) or (hit_oc and not iv):
            out.append(f)
    return out


def lookup(paper: Dict[str, Any], question: Optional[str]) -> Optional[List[Dict[str, Any]]]:
    """Pre-read findings for a run-database paper (matched by PMID / DOI / PMCID), filtered for the question.
    None when the paper was not pre-read with the current version (the normal AI extraction then runs).
    An empty list is also returned as None: no relevant pre-read finding means the question-focused reading runs."""
    if os.getenv("BIOSYN_NO_PREREAD") == "1":
        return None
    try:
        import library_db as L
        if not L.exists():
            return None
        with L.Library() as lib:
            rec = lib.by_ids(pmid=paper.get("pmid"), doi=paper.get("doi"), pmcid=paper.get("pmcid"))
            if not rec:
                return None
            fs = lib.get_findings(rec["id"], version())
    except Exception:
        return None
    if not fs:
        return None
    sel = select_for_question(fs, question)
    return [dict(f, from_library=True) for f in sel] or None
