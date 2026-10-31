"""
evidence_reviewer.py  (v26)
===========================
The ADVERSARIAL EVIDENCE REVIEW: a separate stage of every dossier whose only job is to find reasons the draft
conclusion might be wrong. It inspects the SOURCE evidence (paper texts, the screening ledger, the local library),
not the first model's reasoning.

Checks (all deterministic; the AI critic in 8 is optional and its every point must quote a source):
  1 missing contradictory studies   records on the same intervention + outcome that were screened out, not deep-read,
                                    or are in the local library but not included, whose own abstract reads the OPPOSITE
                                    direction (or no effect) compared with the conclusion
  2 unsupported causal claims       causal wording the designs cannot carry (causal_checks.language)
  3 misquoted / misread results     every included quote is re-located in its paper (quote_guard) and re-read by code;
                                    a direction the code reads differently is listed (calibrated confidence attached)
  4 numerical errors                stated percentages / changes that do not recompute; relative effects without the
                                    absolute size (risk_numbers); abstract-vs-table conflicts (numeric_audit)
  5 model confusion                 sentences about people supported only by cell / animal studies
  6 overstated certainty            'proves / establishes / definitively / confirms' or a 'Yes' when the calibrated
                                    verdict is mixed, preliminary or insufficient
  7 omitted limitations             high risk of bias, abstract-only reading, a single study, supra-physiological doses,
                                    retracted / corrected papers - not mentioned in the report
  8 AI critic (optional, a key)     sees the conclusion and the SOURCE QUOTES only; an objection whose quote is not in a
                                    paper is discarded

    review(dossier, report_text, db=None, router=None) -> {'issues': [...], 'verdict_status': 'stands' |
                                                           'stands with caveats' | 'under review', 'markdown': ...}
"""
from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

REVIEW_VERSION = "1.0"
_OVERSTATE = re.compile(r"\b(?:proves?|proven|establish(?:es|ed)? (?:that|beyond)|definitive(?:ly)?|conclusively|"
                        r"undeniabl[ey]|beyond (?:any|reasonable) doubt|confirms? that|clearly demonstrates?|is a cure)\b", re.I)
_DOSE_PHRASE = re.compile(r"\bhumans?[- ](?:reference|exposure|equivalent|plasma|dose|therapeutic|cmax|levels?)\w*", re.I)
_HEDGE = re.compile(r"\b(?:uncertain|not established|unknown|unclear|cannot be (?:assumed|extrapolated)|may not|untested|"
                    r"not (?:been )?shown|remains? to be)\b", re.I)
_PEOPLE = re.compile(r"\b(?:patients?|people|humans?|women|men|adults|children|clinical(?:ly)?)\b", re.I)


def _issue(code: str, severity: str, check: str, detail: str, evidence: Any = None) -> Dict[str, Any]:
    return {"code": code, "severity": severity, "check": check, "detail": detail, "evidence": evidence}


def _concl_direction(dossier: Dict[str, Any]) -> Optional[str]:
    v = dossier.get("calibrated_verdict") or {}
    d = v.get("direction")
    return d if d in ("increase", "decrease", "no_effect") else None


def _missing_contradictions(dossier: Dict[str, Any], question: str) -> List[Dict[str, Any]]:
    import semantic_audit as sa
    concl = _concl_direction(dossier)
    if concl is None:
        return []
    prisma = dossier.get("prisma") or {}
    included_titles = {str(r.get("title") or "").lower()[:80] for r in (dossier.get("evidence_matrix") or {}).get("rows") or []}
    cands = []
    for row in prisma.get("screening_table") or []:
        if row.get("decision") == "included":
            continue
        cands.append({"title": row.get("title"), "abstract": None, "pmid": row.get("pmid"), "doi": row.get("doi"),
                      "why_not_included": f"{row.get('decision')}: {row.get('reason') or ''}"[:160], "source": "screening"})
    try:
        import library_db as L
        import library_search as ls
        if L.exists():
            with L.Library() as lib:
                for h in ls.search(lib, question, k=40):
                    cands.append({"title": h.get("title"), "abstract": h.get("abstract"), "pmid": h.get("pmid"),
                                  "doi": h.get("doi"), "why_not_included": "in the local library, not in this run",
                                  "source": "library", "status": h.get("status")})
    except Exception:
        pass
    from evidence_identity import question_frame
    fr = question_frame(question)
    from evidence_identity import question_outcome
    endpoint = question_outcome(fr)
    iv_terms: List[str] = []
    try:
        import query_parser as qp
        for cpt in qp.search_concepts(qp.parse(question)):
            if cpt.get("role") in ("intervention", "exposure"):
                iv_terms += [t for t in cpt.get("terms") or [] if len(t) > 3]
    except Exception:
        pass
    out, seen = [], set()
    for c in cands:
        t = str(c.get("title") or "").lower()[:80]
        if not t or t in included_titles or t in seen:
            continue
        seen.add(t)
        text = f"{c.get('title') or ''}. {c.get('abstract') or ''}"
        # a contradiction only counts when the record is about the SAME intervention and reports the SAME endpoint
        if iv_terms and not any(re.search(r"\b" + re.escape(t), text, re.I) for t in iv_terms):
            continue
        if not endpoint:
            continue
        oi = sa.canonical_outcome(endpoint)
        res = sa.direction_in_text(text, oi, endpoint)
        if not res.get("endpoint_found"):
            continue
        d = res.get("direction")
        if d and d != concl and not (d == "no_effect" and concl == "no_effect"):
            out.append({**{k: c[k] for k in ("title", "pmid", "doi", "why_not_included", "source")}, "reads": d,
                        "conclusion": concl})
    return out[:15]


def _source_text(db: Any, paper_id: Any) -> str:
    """The SAME text the extractor quoted from: every section, the stored abstract and the tables (paper_extraction.
    gather_parts) - a quote from a table or from the abstract is not a misquote."""
    if db is None or paper_id is None:
        return ""
    try:
        from paper_extraction import gather_parts
        paper = db.get_paper(paper_id) or {}
        parts = gather_parts(db, paper)
        return "\n\n".join(str(parts.get(k) or "") for k in ("abstract", "introduction", "methods", "results",
                                                             "results_discussion", "discussion", "conclusion",
                                                             "limitations", "other", "tables"))
    except Exception:
        return ""


def _prose(report_text: str) -> List[str]:
    """Sentences the REPORT itself asserts: table rows, block quotes, code and quoted paper text are left out (a
    paper's own 'confirms that' inside a quotation is not the report overstating)."""
    out = []
    for line in (report_text or "").splitlines():
        s = line.strip()
        if not s or s.startswith(("|", ">", "```", "#", "<")):
            continue
        s = re.sub(r"^[-*+]\s+|^\d+\.\s+", "", s)
        s = re.sub(r"[\"“][^\"”]{3,}[\"”]", " ", s)           # quoted spans
        out += [x for x in re.split(r"(?<=[.!?])\s+", s) if x.strip()]
    return out


def _bad_status(dossier: Dict[str, Any], db: Any) -> set:
    """Paper ids of the run that are retracted or under an expression of concern (publication_status table)."""
    if db is None:
        return set()
    try:
        rows = db._conn.execute("SELECT paper_id FROM paper_status WHERE status IN ('retracted', 'expression_of_concern')").fetchall()
        return {r[0] for r in rows}
    except Exception:
        return set()


def review(dossier: Dict[str, Any], report_text: str = "", db: Any = None, router: Any = None) -> Dict[str, Any]:
    import causal_checks as cc
    import quote_guard as qg
    import risk_numbers as rn
    issues: List[Dict[str, Any]] = []
    question = dossier.get("research_question") or ""
    rows = (dossier.get("evidence_matrix") or {}).get("rows") or []
    prim = [r for r in rows if not r.get("secondary_report")]
    verdict = dossier.get("calibrated_verdict") or {}
    cat = verdict.get("category")

    # 1 missing contradictory studies
    try:
        mc = _missing_contradictions(dossier, question)
        if mc:
            issues.append(_issue("R1-MISSING-CONTRA", "major", "missing contradictory studies",
                                 f"{len(mc)} record(s) not in the evidence read the outcome differently from the conclusion "
                                 f"- the conclusion was not tested against them", mc))
    except Exception as exc:
        issues.append(_issue("R1-ERROR", "minor", "missing contradictory studies", f"check failed: {exc}"))

    # 2 unsupported causal claims
    from evidence_identity import question_frame
    human_q = question_frame(question).get("domain") == "human_medicine"
    lang = cc.language(report_text, prim, human_q)
    if lang:
        issues.append(_issue("R2-CAUSAL", "major", "unsupported causal claims",
                             f"{len(lang)} sentence(s) state causation the study designs cannot support", lang[:8]))

    # 3 misquoted / misread results
    misq, misread = [], []
    for r in prim:
        src = _source_text(db, r.get("paper_id"))
        for f in r.get("findings") or []:
            q = f.get("quote") or ""
            # v26.2: the SAME rule as the extraction guard - exact, or near-verbatim (>= 93% alike: hyphenation,
            # spacing, ligatures). The exact-only test called guard-accepted quotes 'misquoted' (critical).
            if src and q and qg.locate(q, src) is None and qg.snap(q, src) is None:
                misq.append({"ref": r.get("ref"), "quote": q[:200]})
            dc = f.get("direction_confidence") or {}
            if dc.get("agrees_with_finding") is False:
                misread.append({"ref": r.get("ref"), "endpoint": f.get("endpoint"), "finding": f.get("direction"),
                                "code_reads": dc.get("code_reading"), "quote": q[:200]})
    if misq:
        issues.append(_issue("R3-MISQUOTE", "critical", "misquoted results",
                             f"{len(misq)} quote(s) are not found in their paper", misq[:10]))
    if misread:
        issues.append(_issue("R3-MISREAD", "major", "possibly misread results",
                             f"{len(misread)} finding(s) where the deterministic reader reads the quote differently - "
                             f"check before relying on them", misread[:10]))

    # 4 numerical errors
    au = rn.audit_text(report_text)
    if au["conflicts"]:
        issues.append(_issue("R4-ARITH", "major", "arithmetic in the report",
                             f"{len(au['conflicts'])} stated percentage(s) / change(s) do not recompute", au["conflicts"][:8]))
    if au["framing"]:
        issues.append(_issue("R4-FRAMING", "minor", "relative vs absolute effects",
                             f"{len(au['framing'])} effect(s) given as a relative change only or as % instead of percentage points",
                             au["framing"][:8]))
    na = dossier.get("numeric_audit") or []
    tab_conf = [{"ref": a.get("ref"), **c} for a in (na if isinstance(na, list) else [{"audit": na}])
                for c in ((a.get("audit") or {}).get("conflicts") or [])]
    if tab_conf:
        issues.append(_issue("R4-TABLE", "major", "abstract vs table", f"{len(tab_conf)} value(s) in an abstract "
                             f"differ from the paper's own tables", tab_conf[:8]))

    # 5 model confusion
    nonhuman = {r.get("ref") for r in prim if r.get("evidence_type") in ("cell_culture", "animal", "organoid_ex_vivo",
                                                                          "in_silico", "experimental_disease_model")}
    conf = []
    for sent in _prose(report_text):
        refs = set(re.findall(r"\bS\d+\b", sent))
        # v26.2: a dose comparison ('the human reference', 'human exposure', 'human-equivalent dose') is not a claim
        # about people, and a hedged sentence ('its relevance ... is uncertain') is a caution, not an overclaim
        bare = _DOSE_PHRASE.sub(" ", sent)
        if _HEDGE.search(sent):
            continue
        if refs and refs <= nonhuman and _PEOPLE.search(bare) and not re.search(r"\b(?:cells?|mice|mouse|rats?|in vitro|"
                                                                                r"animal|xenograft|cultured)\b", sent, re.I):
            conf.append({"sentence": sent.strip()[:250], "refs": sorted(refs)})
    if conf:
        issues.append(_issue("R5-MODEL", "major", "model confusion",
                             f"{len(conf)} sentence(s) about people rest only on cell / animal studies", conf[:8]))

    # 6 overstated certainty
    over = [s.strip()[:250] for s in _prose(report_text) if _OVERSTATE.search(s)]
    weak = cat in ("mixed", "preliminary", "insufficient")
    yes = bool(re.search(r"\*\*Verdict:\*\*\s*(?:Yes|Proven)", report_text or "", re.I))
    if over or (weak and yes):
        issues.append(_issue("R6-CERTAINTY", "major" if weak else "minor", "overstated certainty",
                             (f"the calibrated verdict is '{verdict.get('label')}' but the report says 'Yes'; " if weak and yes else "")
                             + (f"{len(over)} sentence(s) use certainty words" if over else ""), over[:6]))

    # 7 omitted limitations
    low = (report_text or "").lower()
    omitted = []
    hi_rob = [r.get("ref") for r in prim if (r.get("risk_of_bias_detail") or {}).get("overall") == "high"]
    if hi_rob and "risk of bias" not in low and "high risk" not in low:
        omitted.append(f"high risk of bias in {', '.join(map(str, hi_rob[:6]))}")
    abs_only = [r.get("ref") for r in prim if (r.get("text_basis") or "abstract") == "abstract"]
    if abs_only and len(abs_only) > len(prim) / 2 and "abstract" not in low:
        omitted.append(f"{len(abs_only)} of {len(prim)} studies read from the abstract only")
    human_rows = [r for r in prim if r.get("evidence_type") in ("human_clinical", "human_observational")]
    if human_rows and len({r.get("trial_group") or r.get("ref") for r in human_rows}) == 1:
        if "single" not in low and "one trial" not in low and "one study" not in low:
            omitted.append("the human evidence comes from a single study")
    supra = [r.get("ref") for r in prim if (r.get("dose_relevance") or {}).get("category") == "supra-physiological"]
    if supra and "supra" not in low and "above human" not in low and "dose" not in low:
        omitted.append(f"supra-physiological doses in {', '.join(map(str, supra[:6]))}")
    bad = [r.get("ref") for r in prim if r.get("paper_id") in _bad_status(dossier, db)]
    if bad and "retract" not in low and "concern" not in low:
        omitted.append(f"retracted / expression of concern: {', '.join(map(str, bad))}")
    if omitted:
        issues.append(_issue("R7-LIMITS", "major", "omitted limitations", "; ".join(omitted)))

    # 8 optional AI critic - sees the conclusion and the source quotes only
    ai_points = []
    if router is not None and prim:
        try:
            ai_points = _ai_critic(router, dossier, prim, db)
            if ai_points:
                issues.append(_issue("R8-AI", "minor", "AI critic (source-quoted objections)",
                                     f"{len(ai_points)} objection(s), each backed by a verbatim source quote", ai_points[:8]))
        except Exception as exc:
            issues.append(_issue("R8-ERROR", "minor", "AI critic", f"not run: {type(exc).__name__}"))

    sev = {i["severity"] for i in issues}
    status = "under review" if "critical" in sev else "stands with caveats" if "major" in sev else "stands"
    res = {"issues": issues, "verdict_status": status, "checks_run": 8 if router is not None else 7,
           "version": REVIEW_VERSION}
    res["markdown"] = markdown(res, verdict)
    return res


def _ai_critic(router: Any, dossier: Dict[str, Any], rows: List[Dict[str, Any]], db: Any) -> List[Dict[str, Any]]:
    import quote_guard as qg
    from llm_json import parse_json_response
    v = dossier.get("calibrated_verdict") or {}
    quotes = []
    for r in rows[:12]:
        for f in (r.get("findings") or [])[:3]:
            if f.get("quote"):
                quotes.append(f"[{r.get('ref')}, {r.get('evidence_type')}] {f['quote'][:400]}")
    prompt = ("You are an ADVERSARIAL reviewer. Your only goal is to find reasons the conclusion below could be WRONG. "
              "Use ONLY the source quotes. For each objection give the exact quote it rests on (copied character for "
              "character) and say which problem it shows: misread result, overstated certainty, wrong population or model, "
              "confounding, dose not relevant, contradiction ignored, or another named problem. If the conclusion holds, "
              "return an empty list.\n"
              f"Conclusion: {v.get('label')} - {v.get('summary') or ''}\n\nSOURCE QUOTES:\n" + "\n".join(quotes) +
              '\n\nReturn ONLY JSON: {"objections": [{"problem": "...", "explanation": "...", "quote": "verbatim"}]}')
    obj = parse_json_response(router.ask(prompt, task="critique"))
    src = "\n".join(quotes)
    out = []
    for o in (obj or {}).get("objections") or []:
        q = (o.get("quote") or "").strip()
        if q and qg.locate(q, src) is not None:          # an objection without a real quote is discarded
            out.append({"problem": o.get("problem"), "explanation": str(o.get("explanation") or "")[:400], "quote": q[:300]})
    return out


def markdown(res: Dict[str, Any], verdict: Dict[str, Any]) -> str:
    L = ["## Adversarial evidence review", "",
         f"**Verdict status after review: {res['verdict_status']}** (verdict: {verdict.get('label') or 'n/a'}). "
         f"This stage looks only for reasons the conclusion could be wrong, from the source evidence.", ""]
    if not res["issues"]:
        L.append("No problem found by the checks (missing contradictions, causal wording, quotes, arithmetic, model "
                 "confusion, certainty, limitations).")
    for i in res["issues"]:
        L.append(f"- **[{i['severity']}] {i['check']}** - {i['detail']}")
        ev = i.get("evidence")
        if isinstance(ev, list):
            for e in ev[:5]:
                if isinstance(e, dict):
                    txt = e.get("sentence") or e.get("title") or e.get("quote") or e.get("text") or json.dumps(e, default=str)[:200]
                    extra = f" (reads {e['reads']}; {e.get('why_not_included')})" if e.get("reads") else \
                        (f" - {e['why']}" if e.get("why") else "")
                    L.append(f"  - {str(txt)[:220]}{extra}")
                else:
                    L.append(f"  - {str(e)[:220]}")
    return "\n".join(L) + "\n"
