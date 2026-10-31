"""
report_validator.py
-------------------
"A polished report must never outrun the reliability of its evidence pipeline."

  * evidence availability   for every evidence type: found & deep-read / found but not deep-read
                            (budget) / found but inaccessible / found but excluded / not screened
                            (budget) / not found in the searched sources  - so "no human trials
                            identified" is never confused with "no human trials exist" (point 39)
  * budget limitation       how much of the eligible literature was actually read (points 12, 40)
  * certainty caps          the reported certainty can never exceed what the evidence allows
  * safe bottom line        a deterministic, fully traceable fallback answer
  * validate_report         hard checks before the report is published (point 43)

Certainty uses BioSynthetica's INTERNAL framework (GRADE-inspired). It is not a formal GRADE
assessment and is never labelled as one (point 38).
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from study_classifier import EVIDENCE_TYPES
from core_utils import ref_key

CERTAINTY = ["very low", "low", "moderate", "high"]
CERTAINTY_LABEL = "BioSynthetica certainty (internal framework, GRADE-inspired - not a formal GRADE assessment)"
HUMAN_TYPES = ("human_clinical", "human_observational")


# --------------------------------------------------------------------------- availability

# how a direction is written in the report: a null result is 'no statistically significant difference', never 'no effect'
# (a non-significant test does not show that the true effect is zero)
DIR_WORD = {"increase": "increase", "decrease": "decrease", "no_effect": "no statistically significant difference",
            "unclear": "unclear"}


def clip_sentence(text: Any, n: int = 200) -> str:
    """v26.2: shorten a reason without cutting a word in half or leaving a bracket open ('... the line is not of
    breast Their findings' read as a broken sentence in the report)."""
    t = re.sub(r"\s+", " ", str(text or "")).strip()
    if len(t) > n:
        t = t[:n].rsplit(" ", 1)[0].rstrip(",;:- ") + "…"
    t += ")" * max(0, t.count("(") - t.count(")"))
    return t


def evidence_availability(prisma: Optional[Dict[str, Any]], rows: List[Dict[str, Any]], focus: str) -> Dict[str, Any]:
    """Separate states for each evidence type, from the PRISMA ledger (point 39)."""
    if not prisma or not prisma.get("ledger"):
        return {"search": "non-systematic retrieval (legacy mode) - search coverage unknown",
                "types": {}, "note": "Absence of a study type here may simply mean it was not searched for."}
    ledger = prisma["ledger"]
    c = prisma.get("counts") or {}
    by_type: Dict[str, Dict[str, int]] = {}
    for rec in ledger:
        t = rec.get("design_guess") or "unclear"
        s = by_type.setdefault(t, {"deep_read": 0, "not_deep_read_budget": 0, "inaccessible": 0, "excluded_screening": 0,
                                   "excluded_fulltext_other": 0, "not_screened_budget": 0})
        if not rec.get("screened"):
            s["not_screened_budget"] += 1
        elif rec.get("screen_decision") == "excluded":
            s["excluded_screening"] += 1
        elif rec.get("fulltext_stage") == "included":
            s["deep_read"] += 1
        elif rec.get("fulltext_stage") == "excluded":
            if "no retrievable text" in (rec.get("fulltext_reason") or ""):
                s["inaccessible"] += 1
            else:
                s["excluded_fulltext_other"] += 1
        else:
            s["not_deep_read_budget"] += 1
    types = {}
    for t in ("human_clinical", "human_observational", "case_report", "animal", "experimental_disease_model",
              "organoid_ex_vivo", "cell_culture", "in_silico", "review_meta"):
        s = by_type.get(t, {})
        included = sum(1 for r in rows if r.get("evidence_type") == t)
        if included:
            state = f"found and deep-read ({included})"
        elif s.get("not_deep_read_budget"):
            state = f"found but NOT deep-read - deep-read budget ({s['not_deep_read_budget']} eligible record(s))"
        elif s.get("inaccessible"):
            state = f"found but inaccessible - no retrievable text ({s['inaccessible']})"
        elif s.get("excluded_fulltext_other"):
            state = f"found but excluded at full text ({s['excluded_fulltext_other']})"
        elif s.get("excluded_screening"):
            state = f"found but excluded at title/abstract screening ({s['excluded_screening']})"
        elif s.get("not_screened_budget"):
            state = f"retrieved but NOT screened - screening budget ({s['not_screened_budget']})"
        else:
            state = "not found in the searched sources (absence of evidence in this search, not evidence of absence)"
        types[t] = {"label": EVIDENCE_TYPES.get(t, t), "state": state, **s, "included": included}
    sources = c.get("sources_used") or []
    return {"search": f"systematic search of {len(sources)} source(s): {', '.join(sources)}", "types": types,
            "note": "Design of non-deep-read records is estimated from title/abstract only."}


# --------------------------------------------------------------------------- budget
def budget_limitation(prisma: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not prisma or not prisma.get("counts"):
        return {"constrained": True, "statement": "Evidence synthesis is constrained: a non-systematic retrieval was used, "
                                                  "so the included papers are not a complete review of the literature."}
    c = prisma["counts"]
    elig, read = c.get("eligible") or 0, c.get("deep_read") or 0
    frac = read / elig if elig else 1.0
    nsc = c.get("not_screened") or 0
    constrained = frac < 1.0 or nsc > 0
    stmt = (f"Evidence synthesis is constrained by the deep-read budget: {read} of {elig} eligible records "
            f"({frac:.0%}) were deep-read" + (f" and {nsc} retrieved records were not screened (screening budget)" if nsc else "")
            + ". Conclusions describe this examined subset, not the whole eligible literature, and this is not a complete systematic review.")
    return {"constrained": constrained, "fraction_read": round(frac, 3), "eligible": elig, "deep_read": read,
            "not_screened": nsc, "statement": stmt if constrained else "All eligible records were deep-read."}


# --------------------------------------------------------------------------- certainty caps
def certainty_cap(rows: List[Dict[str, Any]], prisma: Optional[Dict[str, Any]], focus: str,
                  frame: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    frame = frame or {}
    primary = [r for r in rows if r.get("level") is not None and r.get("n_findings")]
    reasons, cap = [], "high"

    def lower(to: str, why: str):
        nonlocal cap
        if CERTAINTY.index(to) < CERTAINTY.index(cap):
            cap = to
        reasons.append(why)

    if not primary:
        lower("very low", "no primary study with a verified finding")
    direct = [r for r in primary if r.get("level") in (1, 2)]
    if focus in ("human", "general") and not any(r.get("evidence_type") in HUMAN_TYPES or
                                                (r.get("organism_group") == "human" and r.get("level") == 2) for r in primary):
        lower("low", "no human evidence - no human efficacy claim can be made")
    elif not direct:
        lower("low", "no direct evidence in the organism the question is about")
    if primary and all(r.get("inference_type") in ("case_observation", "temporal_observation", None) for r in primary):
        lower("very low", "only uncontrolled observations / case reports")
    if len(primary) == 1:
        lower("low", "a single primary study")
    if primary and sum(1 for r in primary if r.get("text_basis") in (None, "abstract")) * 2 >= len(primary):
        lower("low", "half or more of the studies were verified from abstracts only")
    b = budget_limitation(prisma)
    if not prisma or not prisma.get("counts"):
        lower("moderate", "non-systematic retrieval (not a PRISMA search) - the sample may not represent the literature")
    if b.get("fraction_read", 1.0) < 0.5:
        lower("low", f"only {b.get('fraction_read', 0):.0%} of eligible records were deep-read")
    if primary and not any((r.get("replication") or {}).get("status", "").startswith("independently") for r in primary):
        lower("moderate", "no independent replication within the deep-read sample")
    findings = [f for r in primary for f in r.get("findings") or []]
    disease = frame.get("disease") or "the disease"
    if frame.get("intent") == "etiology" and findings and not any(f.get("evidence_role") == "disease_risk" for f in findings):
        lower("low", f"no study in the sample measured {disease} occurrence or risk - mechanistic and intermediate findings "
                     f"cannot establish causes")
    elif findings and not any(f.get("answers_question", True) for f in findings):
        lower("low", "no verified finding measures the outcome the question asks about (mechanistic / context findings only)")
    if focus in ("human", "general") and frame.get("intent") != "etiology":
        # GRADE-style: human evidence that ANSWERS the question only from observational studies starts at low;
        # a single human study answering it is also low
        h_ans = [r for r in primary if r.get("evidence_type") in HUMAN_TYPES
                 and any(f.get("answers_question", True) for f in r.get("findings") or [])]
        if h_ans and not any(r.get("evidence_type") == "human_clinical" for r in h_ans):
            lower("low", "the human evidence that answers the question is observational only (no controlled trial finding)")
        elif len(h_ans) == 1:
            lower("low", "only one human study answers the question")
    if frame.get("heterogeneous"):
        subs = {x for r in primary for x in (r.get("disease_subtypes") or []) if x != "not specified"}
        if len(subs) > 1:
            lower("moderate", f"findings come from different {disease} subtypes ({', '.join(sorted(subs))}) and cannot be pooled")
    return {"max": cap, "reasons": reasons}


def apply_cap(certainty: str, cap: Dict[str, Any]) -> (str, Optional[str]):
    c = (certainty or "low").lower().strip()
    if c not in CERTAINTY:
        c = "low"
    if CERTAINTY.index(c) > CERTAINTY.index(cap["max"]):
        return cap["max"], f"Certainty lowered from '{c}' to '{cap['max']}': " + "; ".join(cap["reasons"])
    return c, None


# --------------------------------------------------------------------------- safe bottom line
TYPE_SHORT = {"human_clinical": "human clinical", "human_observational": "human observational", "animal": "animal",
              "cell_culture": "cell culture", "organoid_ex_vivo": "organoid / ex vivo", "in_silico": "in silico",
              "case_report": "case report", "experimental_disease_model": "induced disease model", "plant": "plant",
              "microbial": "microbial"}


def safe_bottom_line(question: str, rows: List[Dict[str, Any]], focus: str, q_info: Dict[str, Any],
                     availability: Dict[str, Any], budget: Dict[str, Any], frame: Optional[Dict[str, Any]] = None) -> str:
    from clinical_context import meets_cure
    frame = frame or {}
    # A study with a CRITICAL methodological flag (a cell line that is not the tissue the paper claims) is excluded
    # from the conclusion, not merely flagged beside it: the flagged finding used to keep appearing in the bottom
    # line as evidence of an effect while the validation section said it could not be about this disease.
    excluded = {r["ref"]: [f["statement"] for f in (r.get("method_flags") or []) if f.get("severity") == "critical"
                           for f in [f]] for r in rows
                if any(f.get("severity") == "critical" for f in (r.get("method_flags") or []))}
    primary = [r for r in rows if r.get("level") is not None and r.get("n_findings") and r["ref"] not in excluded]
    sample = f"S1–S{len(rows)}" if rows else "no studies"
    if not primary:
        # when the only candidate studies were excluded for a methodological flaw, say so: 'nothing was found' and
        # 'what was found cannot be used' are different statements to a reader
        why = ""
        if excluded:
            flags = sorted({clip_sentence(v[0], 200) for v in excluded.values() if v})
            why = (f" {len(excluded)} study(ies) were excluded for a methodological flaw "
                   f"({', '.join(sorted(excluded, key=ref_key))}): " + ("; ".join(flags) if flags else "")
                   + " This is not absence of evidence - it is evidence that cannot be used for this question.")
        return (f"Within the deep-read sample ({sample}), no verified primary finding addresses this question; "
                f"no conclusion can be drawn (absence of evidence, not evidence of absence).{why}")
    disease = frame.get("disease") or (q_info.get("disease_terms") or ["the condition"])[0]
    if frame.get("intent") == "etiology":
        risk = [(r, f) for r in primary for f in r.get("findings") or [] if f.get("evidence_role") == "disease_risk"]
        other = [(r, f) for r in primary for f in r.get("findings") or [] if f.get("evidence_role") != "disease_risk"]
        parts = []
        if risk:
            ex = sorted({f.get("intervention") or "an exposure" for _, f in risk})
            parts.append(f"Within the deep-read sample, {', '.join(ex)} {'was' if len(ex) == 1 else 'were'} associated with {disease} "
                         f"occurrence or risk [{', '.join(sorted({r['ref'] for r, _ in risk}))}]; association is not demonstrated causation.")
        else:
            parts.append(f"Within the deep-read sample ({sample}), no study measured {disease} occurrence or risk, so no cause of "
                         f"{disease} can be established from this evidence.")
        if other:
            kinds = sorted({(f.get("evidence_role") or "direct").replace("_", " ") for _, f in other})
            parts.append(f"The other verified findings are {', '.join(kinds)} observations "
                         f"[{', '.join(sorted({r['ref'] for r, _ in other}))}], which do not show that a factor causes {disease}.")
        if frame.get("heterogeneous"):
            subs = sorted({x for r in primary for x in (r.get("disease_subtypes") or []) if x != "not specified"})
            parts.append(f"{disease.capitalize()} is heterogeneous; subtypes covered: {', '.join(subs) or 'not specified by the studies'}.")
        if budget.get("constrained"):
            parts.append(f"Only {budget['deep_read']} of {budget['eligible']} eligible records were deep-read."
                         if budget.get("eligible") else "Retrieval was non-systematic, so the sample may not represent the literature.")
        return " ".join(parts)
    best = max(primary, key=lambda r: ((r.get("best_outcome") or {}).get("level") or -1))
    bo = best.get("best_outcome") or {}
    human = [r for r in primary if r.get("bucket") == "human" or any(
        (f.get("model_type") or "").startswith("human") for f in r.get("findings") or [])]
    parts = []
    if q_info.get("asks_cure"):
        cured = [r for r in primary if meets_cure((r.get("best_outcome") or {}).get("level"), r.get("follow_up_days"))]
        disease = (q_info.get("disease_terms") or ["the condition"])[0]

        def _sub(r):                       # leukemia is not one disease: say which subtype each study is about
            s_ = [x for x in (r.get("disease_subtypes") or []) if x != "not specified"]
            return f"{', '.join(s_)} " if s_ and (frame or {}).get("heterogeneous") else ""
        if cured:
            parts.append("Sustained remission meeting the operational cure definition (complete response sustained "
                         "≥12 months) was reported in " + "; ".join(
                             f"{_sub(r)}{r.get('evidence_type_label', 'study').lower()} [{r['ref']}]" for r in cured) + ".")
            others = [r for r in primary if r not in cured and (r.get("best_outcome") or {}).get("level") is not None]
            if others:
                parts.append("Other studies reported shorter or weaker outcomes: " + "; ".join(
                    f"{_sub(r)}'{(r.get('best_outcome') or {}).get('label', 'outcome')}' [{r['ref']}]" for r in others) + ".")
            if (frame or {}).get("heterogeneous"):
                parts.append(f"{disease.capitalize()} is not one disease: curability differs by subtype, and subtypes "
                             f"not studied here are not covered by this answer.")
        elif bo.get("key") == "cure" or (bo.get("level") or 0) >= 6:
            # a study SAYS 'cured' but does not state the follow-up the operational definition needs
            parts.append(f"{_sub(best)}{best.get('evidence_type_label', 'A study')} [{best['ref']}] reports patients as "
                         f"'{bo.get('label')}', but the follow-up needed for the operational cure definition (sustained "
                         f"≥12 months) is not stated, so a cure cannot be confirmed from this evidence.")
            if (frame or {}).get("heterogeneous"):
                parts.append(f"{disease.capitalize()} is not one disease: curability differs by subtype.")
        else:
            fu = best.get("follow_up_days")
            parts.append(f"No evidence demonstrating that {disease} is curable was identified in the deep-read sample; the strongest "
                         f"reported outcome was '{bo.get('label', 'not stated')}'"
                         + (f" after {fu / 30.44:.1f} months" if fu else "")
                         + f" in {best.get('evidence_type_label', 'a study').lower()} evidence [{best['ref']}]. "
                         f"This does not show that {disease} cannot be cured (absence of evidence, not evidence of absence).")
    else:
        # count only FINDINGS that measure the question's own endpoint (proliferation / Ki-67), one vote per study and
        # direction, with the system named - never a pooled per-paper direction that mixes apoptosis UP with growth DOWN
        q_end = set((frame or {}).get("endpoints") or [])
        related = {"proliferation": {"proliferation", "tumour_growth"}, "tumour_growth": {"tumour_growth", "proliferation"}}
        want = set().union(*[related.get(e, {e}) for e in q_end]) if q_end else set()
        from semantic_audit import FAMILY_LABEL
        dirs: Dict[str, Dict[str, set]] = {}
        for r in primary:
            for f in r.get("findings") or []:
                if not f.get("answers_question", True) or (want and f.get("endpoint") not in want):
                    continue
                if f.get("direction") not in ("increase", "decrease", "no_effect"):
                    continue
                # the system of THIS finding's experiment (an animal paper's cell assay is cell evidence)
                mt = f.get("model_type") or r.get("evidence_type")
                sysname = TYPE_SHORT.get(mt, (mt or "study").replace("_", " "))
                sg = f.get("subgroup")
                ref = r["ref"] + (f" - {'overall population' if sg == 'overall' else 'only in ' + sg}" if sg else "")
                dirs.setdefault(f["direction"], {}).setdefault(sysname, set()).add(ref)
        label = " / ".join(sorted({FAMILY_LABEL.get(e, e) for e in want})) or "the question's outcome"
        if dirs:
            parts.append(f"Within the deep-read sample, verified findings that measure {label} point: " + "; ".join(
                f"{DIR_WORD.get(k, k.replace('_', ' '))} (" + ", ".join(f"{sysn} [{', '.join(sorted(refs, key=ref_key))}]" for sysn, refs in sorted(v.items())) + ")"
                for k, v in dirs.items()) + ".")
            # a study whose effect appears only in a subgroup while the whole population shows none: say so plainly
            # same study AND same endpoint (the asked one): a subgroup effect next to an overall null on THAT endpoint
            def _on_q(f):
                return f.get("answers_question", True) and (not want or f.get("endpoint") in want)
            limited = sorted({r["ref"] for r in primary
                              if any(_on_q(f) and f.get("subgroup") and f.get("subgroup") != "overall"
                                     and f.get("direction") in ("increase", "decrease") for f in r.get("findings") or [])
                              and any(_on_q(f) and f.get("direction") == "no_effect" and f.get("subgroup") in (None, "overall")
                                      for f in r.get("findings") or [])}, key=ref_key)
            if limited:
                parts.append(f"In {', '.join(limited)} the effect was confined to a subgroup; the overall population showed no "
                             f"significant change, so the result does not apply to all patients.")
            if dirs.get("no_effect"):
                parts.append("Null results are reported, not discarded: they weigh against a general effect (a non-significant result is "
                             "not proof that the true effect is zero).")
        else:
            parts.append(f"Within the deep-read sample, no verified finding measures {label}.")
        ctx_only = sorted({r["ref"] for r in primary for f in r.get("findings") or []
                           if not f.get("answers_question", True) or (want and f.get("endpoint") not in want)}
                          - {ref for v in dirs.values() for refs in v.values() for ref in refs}, key=ref_key)
        if ctx_only:
            def _what(ref):
                r = next((x for x in primary if x["ref"] == ref), {})
                fs = sorted({f"{FAMILY_LABEL.get(f.get('endpoint'), str(f.get('endpoint') or '?').replace('other:', '').replace('_', ' '))} "
                             f"{DIR_WORD.get(f.get('direction') or 'unclear', 'unclear')}" for f in r.get("findings") or []})
                mts = sorted({f.get("model_type") for f in r.get("findings") or [] if f.get("model_type")}) or [r.get("evidence_type")]
                kind = " + ".join(TYPE_SHORT.get(m, (m or "study").replace("_", " ")) for m in mts)
                return f"{ref} ({kind}: {', '.join(fs[:2])})" if fs else ref
            # studies with the same kind of result are listed together ('S8, S10, S11 (human observational: recurrence
            # incidence decrease)'), not as thirteen near-identical entries
            grouped: Dict[str, List[str]] = {}
            for x in ctx_only:
                w = _what(x)
                key = w[len(x):].strip() if w.startswith(x) else ""
                grouped.setdefault(key, []).append(x)
            parts.append(f"Other verified findings measure different endpoints or test combination regimens, and do not answer "
                         f"the question directly: "
                         f"{'; '.join(', '.join(refs) + (' ' + k if k else '') for k, refs in grouped.items())}.")
    if excluded:
        why = sorted({clip_sentence(v[0], 200).rstrip(".") for v in excluded.values() if v})
        parts.append(f"Excluded from this conclusion: {', '.join(sorted(excluded, key=ref_key))} - "
                     + (why[0] if len(why) == 1 else "; ".join(why))
                     + ". Their findings are listed in the evidence matrix and in the methodological flags, but they "
                       "cannot support a conclusion about this disease.")
    if focus in ("human", "general") and not human:
        hs = (availability.get("types", {}).get("human_clinical") or {}).get("state", "not assessed")
        parts.append(f"No human evidence was deep-read (human clinical studies: {hs}), so no human efficacy claim can be made.")
    if budget.get("constrained"):
        parts.append(f"Only {budget['deep_read']} of {budget['eligible']} eligible records were deep-read."
                         if budget.get("eligible") else "Retrieval was non-systematic, so the sample may not represent the literature.")
    return " ".join(parts)


# --------------------------------------------------------------------------- final validation
def validate_report(synthesis: Dict[str, Any], matrix: Dict[str, Any], prisma: Optional[Dict[str, Any]],
                    string_net: Optional[Dict[str, Any]], hypotheses: List[Dict[str, Any]],
                    contradictions: Optional[List[Dict[str, Any]]] = None,
                    verified_ids: Optional[set] = None, gaps: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
    import re
    rows = matrix.get("rows") or []
    refs = {r["ref"] for r in rows}
    checks = []

    def add(name, ok, detail, severity="critical", skipped=False, layer="technical"):
        # every check is reported as passed / failed / skipped (not applicable to this run) - never a blanket 'passed'
        # an 'info' item is a disclosure (e.g. how many statements were withheld) - 'noted', never a failure
        # layer: 'technical' = the report is internally consistent (counts, citations, rules); 'evidence' = each finding
        # was re-read against its source passage. Neither is SCIENTIFIC validity (that needs independent expert review).
        status = "skipped" if skipped else ("passed" if ok else ("noted" if severity == "info" else "failed"))
        checks.append({"check": name, "ok": bool(ok) or skipped or status == "noted", "detail": detail, "severity": severity,
                       "status": status, "layer": layer})

    cons = (prisma or {}).get("consistency")
    # recompute every PRISMA identity at publication time - a stored 'ok' is not trusted
    errs = []
    if prisma and prisma.get("counts") and prisma.get("ledger") is not None:
        from prisma_screening import validate_prisma
        pids = prisma.get("paper_ids") or [x.get("paper_id") for x in prisma["ledger"] if x.get("included")]
        errs = validate_prisma(prisma["counts"], prisma["ledger"], pids)
        c = prisma["counts"]
        if "identified_databases" in c:
            calc = c["identified_databases"] + c.get("identified_citation_chasing", 0) - c["duplicates_removed"]
            if calc != c["after_dedup"]:
                errs.append(f"{c['identified_databases']} + {c.get('identified_citation_chasing', 0)} - {c['duplicates_removed']} = "
                            f"{calc}, but {c['after_dedup']} unique records are reported")
    add("PRISMA arithmetic reconciles (recomputed)", (cons is None and not prisma) or (not errs and (cons or {}).get("ok", True)),
        "legacy retrieval (no PRISMA)" if not prisma else ("all identities hold" if not errs and (cons or {}).get("ok", True)
                                                          else "; ".join(errs or (cons or {}).get("errors", []))))
    if prisma and prisma.get("counts"):
        inc = prisma["counts"].get("included")
        acc = len(rows) + len(matrix.get("off_question") or [])
        add("included studies = evidence rows + off-question exclusions", inc in (None, acc),
            f"{inc} included = {len(rows)} on-question + {len(matrix.get('off_question') or [])} off-question")
    if contradictions is not None and verified_ids is not None:
        bad = sorted({cid for c in contradictions for cid in (c.get("claim_a_id"), c.get("claim_b_id")) if cid not in verified_ids})
        add("every contradiction refers to a displayed verified claim", not bad, f"unknown claim IDs: {bad}" if bad else "ok")
    if gaps is not None and verified_ids is not None:
        badg = sorted({x for g in gaps for x in (g.get("derived_from") or []) if str(x).lstrip("C").isdigit()
                       and int(str(x).lstrip("C")) not in verified_ids})
        add("every gap derives from displayed verified claims", not badg, f"unknown: {badg}" if badg else "ok", "major")
    text = " ".join(str(synthesis.get(k) or "") for k in ("executive_summary", "bottom_line", "human_evidence", "animal_evidence",
                                                         "in_vitro_evidence", "translational_gap"))
    cited = set(re.findall(r"\bS\d+\b", text))
    add("every citation points to an included study", cited <= refs, f"unknown: {sorted(cited - refs)}" if cited - refs else "ok")
    cap = synthesis.get("certainty_cap") or {}
    add("certainty within the evidence-based cap", CERTAINTY.index(synthesis.get("certainty", "low")) <= CERTAINTY.index(cap.get("max", "high")),
        f"{synthesis.get('certainty')} <= {cap.get('max', 'high')}")
    blocked = synthesis.get("blocked_statements") or []
    add("no unsupported statement published", "⛔" not in text or all(b.get("replacement") for b in blocked),
        f"{len(blocked)} statement(s) withheld by hard rules" if blocked else "none withheld", "info")
    untraced = [t for t in synthesis.get("traceability") or [] if not t.get("sources")]
    add("every cited conclusion is traceable to a passage", not untraced, f"{len(untraced)} untraceable" if untraced else "ok")
    add("deep-read budget limitation stated", bool(synthesis.get("budget_limitation")), (synthesis.get("budget_limitation") or {}).get("statement", ""), "major")
    add("evidence-availability states reported", bool(synthesis.get("evidence_availability")), "absence of evidence vs evidence of absence", "major")
    abstract_rows = [r["ref"] for r in rows if r.get("text_basis") in (None, "abstract")]
    add("abstract-only evidence disclosed", True, f"abstract-only: {', '.join(abstract_rows) or 'none'}", "info")
    add("STRING shown only as association", not string_net or string_net.get("evidence_link") in ("linked", "background", None),
        (string_net or {}).get("evidence_link_note", "no STRING query"), "major",
        skipped=not string_net or not string_net.get("edges"))
    unlabeled = [h.get("title") for h in hypotheses or [] if not h.get("evidence_status")]
    add("hypotheses labelled (evidence-derived vs AI)", not unlabeled, f"unlabelled: {unlabeled}" if unlabeled else "ok", "major")
    add("certainty labelled as internal framework (not formal GRADE)", True, CERTAINTY_LABEL, "info")
    from reasoning_guards import (ABSENT_TYPE, CAUSE_STMT, CAUSE_NEGATED, QC_CLAIM, _rows_with_findings, _sentences,
                                  absent_type_conflict)
    full = text + " " + str(synthesis.get("certainty_rationale") or "")
    # the engine's own answer (Layer 1: verdict, level-by-level answer, discordance) is checked like the AI text
    full += " " + " ".join(str(synthesis.get(k) or "") for k in ("verdict", "level_answer", "level_discordance"))
    # the guard's own withheld / rewritten markers quote the rule ('says there is no human evidence') - not report text
    published = re.sub(r"⛔ \[statement withheld[^\]]*\]|\[rewritten by evidence rules\]", " ", full)
    contra = []
    for sent in _sentences(published):
        for m in ABSENT_TYPE.finditer(sent):
            contra += [r["ref"] for r in absent_type_conflict(sent, m, _rows_with_findings(rows))[1]]
    add("no 'no human/epidemiological evidence' statement while such evidence is present (answer + synthesis text)", not contra,
        f"contradicted by {sorted(set(contra))}" if contra else "consistent")
    # a stored direction must agree with its own quote (a reversed result - 'blunted the Ki-67 rise' stored as
    # 'Ki-67 increased' - is a hard failure, not a warning)
    from semantic_audit import canonical_outcome as _co, direction_in_text as _dit
    reversed_ = []
    for r in rows:
        if r.get("evidence_type") == "review_meta":
            continue
        for f in r.get("findings") or []:
            d0 = f.get("direction")
            if d0 not in ("increase", "decrease") or not f.get("quote"):
                continue
            meas = f.get("measurement") or str(f.get("endpoint") or "").replace("_", " ")
            oi = _co(meas)
            if oi["family"] != f.get("endpoint"):
                oi = _co(str(f.get("endpoint") or "").replace("_", " "))
            d1 = _dit(f["quote"], oi, meas).get("direction")
            if d1 in ("increase", "decrease") and d1 != d0:
                reversed_.append(f"{r['ref']} {f.get('finding') or ''} ({f.get('endpoint')}: stored {d0}, quote reads {d1})")
    add("every verified direction agrees with its own quote", not reversed_,
        "; ".join(reversed_[:5]) if reversed_ else f"{sum(len(r.get('findings') or []) for r in rows)} findings rechecked",
        layer="evidence")
    # ---- PRIORITY 1: evidence correctness. The report FAILS when a finding or the conclusion conflicts with the evidence.
    from semantic_audit import refine_endpoint as _refine, statistical_reading as _stat, _BETWEEN_NULL as _BN, _WITHIN_CHANGE as _WC
    nonsig, wrong_target = [], []
    for r in rows:
        for f in r.get("findings") or []:
            q = f.get("quote") or ""
            if f.get("direction") in ("increase", "decrease") and q and _stat(q)["significant"] is False:
                nonsig.append(f"{r['ref']} ({f.get('endpoint')}: {_stat(q)['basis']})")
            elif f.get("direction") in ("increase", "decrease") and q and _BN.search(q) and _WC.search(q):
                nonsig.append(f"{r['ref']} ({f.get('endpoint')}: change from baseline in both arms, no difference between arms)")
            if q and f.get("endpoint") in ("proliferation", "viability", "apoptosis", "tumour_growth", "glycaemic",
                                           "metastasis_invasion"):
                meas = f.get("measurement") or str(f.get("endpoint")).replace("_", " ")
                oi = _co(meas)
                if oi["family"] != f.get("endpoint"):
                    oi = _co(str(f.get("endpoint")).replace("_", " "))
                oi2, _, rej = _refine(oi, meas, q)
                if rej or oi2["family"] != oi["family"]:
                    wrong_target.append(f"{r['ref']} ({f.get('endpoint')} -> {rej or oi2['family'].replace('_', ' ')})")
    add("no non-significant result is reported as an effect (P >= 0.05 or a CI containing no-difference)", not nonsig,
        "; ".join(nonsig[:5]) if nonsig else "every directional finding is statistically significant in its own quote",
        layer="evidence")
    add("every finding measures what its endpoint says (immune-cell proliferation is not tumour proliferation; a baseline "
        "subgroup is not an outcome)", not wrong_target, "; ".join(wrong_target[:5]) if wrong_target else "ok", layer="evidence")
    vc = verdict_conflict(synthesis, matrix)
    add("the verdict agrees with the verified human evidence on the asked endpoint", not vc["conflict"], vc["detail"],
        skipped=vc["skipped"], layer="evidence")
    from reasoning_guards import subgroup_merge, SURROGATE_LEAP
    rows_by = {r["ref"]: r for r in rows}
    merged, surrogate = [], []
    for sent in _sentences(published):
        crefs = [x for x in re.findall(r"\bS\d+\b", sent) if x in rows_by]
        if not crefs:
            continue
        if "separate subgroups" not in sent and subgroup_merge(sent, [rows_by[x] for x in crefs]):
            merged.append(sent[:90])
        if SURROGATE_LEAP.search(sent) and "short-term marker" not in sent:
            surrogate.append(sent[:90])
    for h in hypotheses or []:
        st = str(h.get("statement") or "")
        crefs = [x["ref"] for x in ((h.get("support") or {}).get("supporting") or []) if x.get("ref") in rows_by]
        if crefs and "separately reported subgroups" not in st and subgroup_merge(st, [rows_by[x] for x in crefs]):
            merged.append(f"hypothesis {h.get('title') or ''}"[:90])
    add("no combined subgroup that the source reports only as separate subgroups (e.g. HER2-positive AND HOMA > 2.8)",
        not merged, "; ".join(merged[:3]) if merged else "every subgroup is reported as the paper reported it", layer="evidence")
    add("a short-term marker (Ki-67) is never presented as a patient outcome", not surrogate,
        "; ".join(surrogate[:3]) if surrogate else "ok", "major", layer="evidence")
    # ---- CROSS-SECTION CONSISTENCY: one canonical record per study and per finding, checked wherever it is reused
    design_split, within_conflict = [], []
    for r in rows:
        et = r.get("evidence_type")
        hum = {f.get("model_type") for f in r.get("findings") or []
               if str(f.get("model_type") or "").startswith("human")}
        if et in ("human_clinical", "human_observational") and hum - {et}:
            design_split.append(f"{r['ref']} filed as {et.replace('_', ' ')} but has {', '.join(sorted(hum - {et}))} findings")
        by_ep: Dict[str, set] = {}
        for f in r.get("findings") or []:
            if f.get("answers_question", True) and f.get("direction") in ("increase", "decrease") and not f.get("subgroup") \
                    and not f.get("subgroups_separate"):
                sysk = "human" if str(f.get("model_type") or "").startswith("human") else f.get("model_type")
                by_ep.setdefault(f"{f.get('endpoint')}|{sysk}", set()).add(f["direction"])
        within_conflict += [f"{r['ref']} ({k.split('|')[0]}: increase AND decrease, no subgroup distinguishes them)"
                            for k, v in by_ep.items() if len(v) > 1]
    add("one design per study: a study's findings are never filed as both clinical and observational", not design_split,
        "; ".join(design_split[:4]) if design_split else "ok", layer="evidence")
    add("no study reports opposite directions for the same endpoint without a subgroup / time / analysis that separates them",
        not within_conflict, "; ".join(within_conflict[:4]) if within_conflict else "ok", "major", layer="evidence")
    # the Layer 1 tally must equal a recount from the evidence matrix (the same verified dataset)
    m_cnt = re.search(r"of (\d+) human finding\(s\) on this endpoint", str(synthesis.get("level_answer") or ""))
    if m_cnt:
        from evidence_identity import question_frame as _qf2
        fr2 = matrix.get("question_frame") or {}
        want2 = set(fr2.get("endpoints") or _qf2(fr2.get("question") or "").get("endpoints") or [])
        if want2 & {"proliferation", "tumour_growth", "viability"}:
            want2 |= {"proliferation", "tumour_growth", "viability"}     # the answer's 'growth' group
        recount = sum(1 for r in rows if r.get("evidence_type") != "review_meta" and not r.get("secondary_report")
                      for f in r.get("findings") or []
                      if str(f.get("model_type") or "").startswith("human") and f.get("answers_question", True) is not False
                      and (not want2 or f.get("endpoint") in want2) and f.get("direction") in ("increase", "decrease", "no_effect"))
        add("the Layer 1 tally equals a recount from the evidence matrix", int(m_cnt.group(1)) == recount,
            f"answer says {m_cnt.group(1)}, the matrix has {recount}", layer="evidence")
    from reasoning_guards import BUCKET_FIELDS, prove_sentence, EFFICACY as _EFF
    unproved = []
    for fld in BUCKET_FIELDS:
        for sent in _sentences(str(synthesis.get(fld) or "")):
            if not re.search(r"\bS\d+\b", sent) and _EFF.search(sent) and "withheld" not in sent and "Verified" not in sent \
                    and not prove_sentence(sent, rows)["refs"] and re.search(r"\b(?:increas|decreas|reduc|lower|higher)\w*", sent, re.I):
                unproved.append(f"{fld.replace('_', ' ')}: {sent[:70]}")
    add("every study-type summary (human / animal / cell) is backed by the verified findings it describes", not unproved,
        "; ".join(unproved[:3]) if unproved else "ok", layer="evidence")
    trials = matrix.get("trials") or {}
    add("publications of the same trial are identified (not counted as independent studies)", True,
        trials.get("statement") or "no human publications", "info", layer="evidence")
    # one study registry: every count in the report comes from it
    reg = study_registry(matrix, prisma)
    add("study counts reconcile (deep-read = included + excluded with a reason)", reg["reconciles"], reg["statement"])
    add("no blanket quality-control claim", not QC_CLAIM.search(full), "ok" if not QC_CLAIM.search(full) else "blanket QC sentence found")
    fr = matrix.get("question_frame") or {}
    if fr.get("intent") == "etiology":
        has_risk = any(f.get("evidence_role") == "disease_risk" for r in rows for f in r.get("findings") or [])
        causal = [x.group(0) for x in re.finditer(r"[^.]*\.", full) if CAUSE_STMT.search(x.group(0))
                  and not CAUSE_NEGATED.search(x.group(0)) and "[rewritten by evidence rules]" not in x.group(0)
                  and not re.search(r"\bassociat", x.group(0), re.I)]
        add("no cause asserted without disease-occurrence/risk evidence", has_risk or not causal,
            "ok" if has_risk or not causal else f"{len(causal)} causal sentence(s) without disease-risk evidence")
    # ---- v24: four checks for the faults a real vitiligo report shipped with while every check passed -----
    # (1) a measurement made on cells reported as a patient outcome ("melanocyte death" -> "patient survival")
    _question = str((matrix.get("question") or synthesis.get("question") or fr.get("question") or ""))
    _PATIENT_EP = ("patient_survival", "recurrence_incidence", "clinical_response")
    _LAB_MODEL = ("cell_culture", "organoid_ex_vivo", "in_silico")
    _cell_as_patient = [f"{r['ref']}: {f.get('endpoint') or f.get('canonical_outcome')} measured in "
                        f"{str(f.get('model_type') or '').replace('_', ' ')}"
                        for r in rows for f in (r.get("findings") or [])
                        if f.get("model_type") in _LAB_MODEL and f.get("canonical_outcome") in _PATIENT_EP]
    add("no cell or in-silico measurement is reported as a patient outcome", not _cell_as_patient,
        "; ".join(_cell_as_patient[:4]) if _cell_as_patient else "ok", "critical", layer="evidence")

    # (2) the intervention column must hold interventions - not fragments, instruments, pathways or the question
    try:
        from claim_extraction import implausible_subject
        _bad_iv = [f"{r['ref']}: {why}" for r in rows for f in (r.get("findings") or [])
                   for why in [implausible_subject(f.get("intervention") or f.get("subject"), _question)] if why]
    except Exception:
        _bad_iv = []
    add("every published intervention is something that was administered or manipulated", not _bad_iv,
        "; ".join(_bad_iv[:4]) if _bad_iv else "ok", "critical", layer="evidence")

    # (3) a cancer endpoint on a question and a paper that are not about cancer
    _CANCER_W = re.compile(r"\b(?:cancers?|carcinomas?|tumou?rs?|neoplas\w*|melanomas?|sarcomas?|lymphomas?|"
                           r"leuk(?:a)?emi\w*|myelomas?|glioma\w*|metasta\w*|malignan\w*|oncolog\w*)\b", re.I)
    _q_cancer = bool(_CANCER_W.search(f"{_question} {' '.join(fr.get('disease_terms') or [])} {fr.get('disease') or ''}"))
    _onc = [f"{r['ref']}: {f.get('endpoint') or f.get('canonical_outcome')}"
            for r in rows for f in (r.get("findings") or [])
            if f.get("canonical_outcome") in ("metastasis_invasion", "tumour_growth")
            and not _q_cancer and not _CANCER_W.search(str(r.get("title") or ""))]
    add("a cancer endpoint appears only where the question or the paper is about cancer", not _onc,
        "; ".join(_onc[:4]) if _onc else "ok", "major", layer="evidence")

    # (4) no proposed human experiment would harm the people in it. A replication hypothesis that carries a
    # verified HARMFUL direction into a randomised human trial is not a study anyone may run.
    _unethical = []
    for h in (hypotheses or (synthesis or {}).get("hypotheses") or []):
        _txt = f"{h.get('statement') or ''} {h.get('predictions') or ''} {(h.get('experiment') or {}).get('model') or ''}"
        if not re.search(r"randomi[sz]ed controlled \(human\)|in (?:humans|patients)|human (?:trial|study)", _txt, re.I):
            continue
        if re.search(r"\b(?:decreas\w*|reduc\w*|shorten\w*|worsen\w*|lower\w*)\b[^.]{0,60}"
                     r"\b(?:patient survival|overall survival|survival|life expectancy)\b", _txt, re.I) or \
           re.search(r"\b(?:increas\w*|rais\w*)\b[^.]{0,60}\b(?:mortality|death rate|recurrence)\b", _txt, re.I):
            _unethical.append(f"{h.get('id') or h.get('hypothesis_id') or '?'}: {(h.get('statement') or '')[:120]}")
    add("no proposed human experiment predicts harm to the people in it", not _unethical,
        "; ".join(_unethical[:3]) if _unethical else "ok", "critical", layer="evidence")

    # ---- methodological traps: a critical flag means the published claim cannot be about the question's disease
    _mf = [f for r in rows for f in (r.get("method_flags") or [])]
    excluded_refs = {r["ref"] for r in rows
                     if any(f.get("severity") == "critical" for f in (r.get("method_flags") or []))}
    _crit = [f for f in _mf if f.get("severity") == "critical"]
    _crit_cited = [f for f in _crit if f.get("ref") and any(fd.get("direction") in ("increase", "decrease", "no_effect")
                                                            for fd in (next((r for r in rows if r["ref"] == f["ref"]), {})
                                                                       .get("findings") or []))]
    add("no published finding comes from a misidentified cell line (ICLAC register)", not _crit_cited,
        "; ".join(f"{f['ref']}: {f['statement'][:120]}" for f in _crit_cited) if _crit_cited
        else (f"{len(_mf)} methodological flag(s), none critical to a published finding" if _mf else "none found"),
        "critical", layer="evidence")
    _pains = [f for f in _mf if f.get("kind") == "assay_interference" and f.get("applies_to_lab_result")]
    add("laboratory results for an assay-interference compound carry the caveat", True,
        "; ".join(sorted({f"{f.get('ref')}: {f.get('compound')}" for f in _pains})) if _pains else "none",
        "info", layer="evidence")
    _sw = (synthesis or {}).get("outcome_switching") or []
    add("every trial's published primary outcome matches the one it registered", not _sw,
        "; ".join(f"{x.get('ref')} ({x.get('nct')}): registered '{(x.get('registered') or ['?'])[0][:60]}'" for x in _sw)
        if _sw else "no mismatch found (checked where a registry entry was available)", "major", layer="evidence")
    # ---- the pooled estimate must agree with the verdict it is published beside
    try:
        from meta_analysis import analyse
        _pooled = [r for r in analyse(rows) if r["bucket"] == "human" and r["i2"] < 75]
        _say = str((synthesis or {}).get("level_answer") or "") + " " + str((synthesis or {}).get("verdict") or "")
        _bad = []
        for pr in _pooled:
            if not pr["significant"] and re.search(r"\bpoint to (?:a reduction|an increase)|supported by randomised",
                                                   _say, re.I):
                _bad.append(f"pooled {pr['endpoint']} {pr['estimate']:.2f} (95% CI {pr['lo']:.2f}-{pr['hi']:.2f}) "
                            f"includes no difference, but the answer states an effect")
        add("the pooled estimate agrees with the published answer", not _bad, "; ".join(_bad) or
            (f"{len(_pooled)} pooled estimate(s) consistent" if _pooled else "nothing poolable"),
            "critical", layer="evidence", skipped=not _pooled)
    except Exception as exc:
        add("the pooled estimate agrees with the published answer", True, f"not run: {str(exc)[:80]}", "major",
            skipped=True, layer="evidence")
    add("off-question papers excluded from the evidence buckets", True,
        f"{len(matrix.get('off_question') or [])} paper(s) excluded: " +
        "; ".join(f"{x['ref']} {x['reason']}" for x in (matrix.get("off_question") or [])[:6]) if matrix.get("off_question") else "none", "info")
    failed = [c for c in checks if c["status"] == "failed" and c["severity"] == "critical"]
    warn = [c for c in checks if c["status"] == "failed" and c["severity"] != "critical"]
    n_pass = sum(1 for c in checks if c["status"] == "passed")
    n_skip = sum(1 for c in checks if c["status"] == "skipped")
    n_note = sum(1 for c in checks if c["status"] == "noted")
    tech = [c for c in checks if c.get("layer") == "technical" and c["status"] in ("passed", "failed")]
    evid = [c for c in checks if c.get("layer") == "evidence" and c["status"] in ("passed", "failed")]
    tally = (f"{n_pass} of {len(checks)} checks passed, {len(failed)} critical failure(s), {len(warn)} warning(s), "
             f"{n_note} disclosure(s), {n_skip} not applicable to this run (see the full list in the Audit layer)")
    layers = (f"Technical consistency (counts, citations, rules): {sum(c['status'] == 'passed' for c in tech)} of {len(tech)} passed · "
              f"Evidence checks (each finding re-read against its source passage - direction, significance, endpoint, "
              f"verdict): {sum(c['status'] == 'passed' for c in evid)} of {len(evid)} passed · Scientific validity: NOT "
              f"assessed by the software - passing these checks shows the report is consistent with the papers it read, not "
              f"that the sample is complete or the interpretation is right (that needs independent expert review).")
    # What the status MEANS, said plainly. A report with an unresolved critical failure, or with findings excluded
    # from its own conclusion, is not 'publication ready': the status now says which it is, and what was excluded.
    if failed:
        status = "NOT READY TO PUBLISH - a critical check failed"
        summary = f"{status} - {tally}: " + "; ".join(f"{c['check']} ({c['detail']})" for c in failed)
    elif warn:
        status = "Publishable with the warnings below"
        summary = (f"{status} - {tally}. Warnings: "
                   + "; ".join(f"{c['check']} ({c['detail']})" for c in warn[:3]) + ".")
    else:
        status = "Checks passed"
        summary = f"Critical pre-publication checks passed - {tally}."
    if excluded_refs:
        summary += (f" {len(excluded_refs)} study(ies) were EXCLUDED from the conclusion for a methodological flaw "
                    f"({', '.join(sorted(excluded_refs, key=ref_key))}); their findings remain listed in the evidence "
                    f"matrix but support no conclusion.")
    return {"passed": not failed, "status": status, "checks": checks, "registry": reg, "tally": tally,
            "layers": layers, "excluded_from_conclusion": sorted(excluded_refs, key=ref_key), "summary": summary}


def verdict_conflict(synthesis: Dict[str, Any], matrix: Dict[str, Any]) -> Dict[str, Any]:
    """Does the one-line verdict contradict the verified HUMAN findings on the asked endpoint?
    'Yes' with a human null / opposite result, 'Not established' with only supporting human results, or 'mixed' when
    the human findings all point one way - each is a hard failure."""
    verdict = str(synthesis.get("verdict") or "")
    fr = matrix.get("question_frame") or synthesis.get("question_frame") or {}
    q = fr.get("question") or ""
    if not verdict or not q:
        return {"conflict": False, "skipped": True, "detail": "no verdict / question to check"}
    from report_generator import expected_direction
    from evidence_identity import question_frame
    exp = expected_direction(q)
    if exp not in ("increase", "decrease"):
        return {"conflict": False, "skipped": True, "detail": "the question asks for no direction (or a cure) - checked elsewhere"}
    want = set(fr.get("endpoints") or question_frame(q).get("endpoints") or [])
    if want & {"proliferation", "tumour_growth"}:
        want |= {"proliferation", "tumour_growth"}
    hf = [(r, f) for r in matrix.get("rows") or [] if r.get("evidence_type") != "review_meta" for f in r.get("findings") or []
          if str(f.get("model_type") or r.get("evidence_type") or "").startswith("human") and f.get("answers_question", True)
          and (not want or f.get("endpoint") in want)]
    sup = sorted({r["ref"] for r, f in hf if f.get("direction") == exp}, key=ref_key)
    neg = sorted({r["ref"] for r, f in hf if f.get("direction") in ("no_effect", "increase", "decrease") and f.get("direction") != exp}, key=ref_key)
    v = verdict.lower()
    conflict = ""
    if v.startswith("yes") and neg:
        conflict = f"the verdict says yes, but human findings {', '.join(neg)} show no significant difference or the opposite"
    elif v.startswith("not established") and sup and not neg:
        conflict = f"the verdict says not established, but every human finding ({', '.join(sup)}) supports the effect"
    elif "mixed" in v and not (sup and neg):
        conflict = (f"the verdict says mixed, but the human findings all point one way "
                    f"({'supporting: ' + ', '.join(sup) if sup else 'not supporting: ' + ', '.join(neg)})")
    return {"conflict": bool(conflict), "skipped": not hf,
            "detail": conflict or (f"verdict '{verdict[:70]}' - human findings supporting: {', '.join(sup) or 'none'}; "
                                   f"no significant difference / opposite: {', '.join(neg) or 'none'}" if hf
                                   else "no human finding on the asked endpoint")}


def study_registry(matrix: Dict[str, Any], prisma: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """The one source of truth for study counts: deep-read -> excluded (with the reason) -> included -> primary / reviews.
    The report, the scope line and the validator all quote this."""
    rows = matrix.get("rows") or []
    off = matrix.get("off_question") or []
    reviews = [r for r in rows if r.get("evidence_type") == "review_meta"]
    with_findings = [r for r in rows if r.get("evidence_type") != "review_meta" and (r.get("n_findings") or r.get("findings"))]
    no_finding = [r for r in rows if r.get("evidence_type") != "review_meta" and not (r.get("n_findings") or r.get("findings"))]
    c = (prisma or {}).get("counts") or {}
    deep = c.get("included") if c.get("included") is not None else len(rows) + len(off)
    excluded = [{"ref": x.get("ref"), "title": (x.get("title") or "")[:90], "reason": x.get("reason") or "excluded"} for x in off]
    reconciles = deep == len(rows) + len(off)
    steps = []
    if c.get("after_dedup") is not None:
        # recomputable from the PRISMA ledger: identified - duplicates = unique = screened + not screened (budget)
        steps.append(f"{c.get('identified', '?')} identified - {c.get('duplicates_removed', '?')} duplicates = "
                     f"{c['after_dedup']} unique")
        steps.append(f"{c.get('screened', '?')} screened + {c.get('not_screened', 0)} not screened (screening budget)")
        if c.get("excluded_screening") is not None:
            steps.append(f"{c['excluded_screening']} excluded at title/abstract")
    if c.get("eligible") is not None:
        if c.get("after_dedup") is None:
            steps.append(f"{c.get('screened', '?')} screened")
        steps.append(f"{c['eligible']} eligible" + (f" ({c['eligible_not_deep_read']} not deep-read: reading budget)"
                                                   if c.get("eligible_not_deep_read") is not None else ""))
    n_abs = sum(1 for r in rows if r.get("text_basis") in (None, "abstract"))
    # 'deep-read' must not suggest full texts were examined when only abstracts were available
    steps.append(f"{deep} paper(s) deep-read" + (" (abstracts only - no full text)" if rows and n_abs == len(rows) else
                                                 f" ({n_abs} from the abstract only)" if n_abs else " (full text)"))
    if excluded:
        steps.append(f"{len(excluded)} excluded after reading (" + "; ".join(
            f"{x['title'] or x['ref']}: {x['reason']}" for x in excluded[:3]) + (" ..." if len(excluded) > 3 else "") + ")")
    steps.append(f"{len(rows)} included (S1-S{len(rows)})" if rows else "0 included")
    statement = (" -> ".join(steps) + f" = {len(with_findings)} primary studies with verified findings + {len(no_finding)} "
                 f"primary studies with no verified finding + {len(reviews)} review(s) (secondary literature, not counted as "
                 f"independent evidence)")
    if not reconciles:
        statement += f" (MISMATCH: {deep} deep-read but {len(rows)} included + {len(off)} excluded)"
    trials = matrix.get("trials") or {}
    if trials.get("groups"):
        statement += f"; {trials['statement']}"
    return {"deep_read": deep, "included": len(rows), "excluded": excluded, "primary_with_findings": len(with_findings),
            "primary_without_findings": len(no_finding), "reviews": len(reviews), "screened": c.get("screened"),
            "eligible": c.get("eligible"), "reconciles": reconciles, "statement": statement, "abstract_only": n_abs}
