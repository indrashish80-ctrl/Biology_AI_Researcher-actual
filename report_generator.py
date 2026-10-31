"""
report_generator.py
-------------------
Compiles scientific research dossiers with balanced representation across papers.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional
from database import ResearchDB
from contradiction_detection import contradiction_counts, contradiction_vocabulary  # noqa: F401  (re-exported)

# the semantic components of a claim (textual grounding only says the quote exists in the paper)
_SEMANTIC_PARTS = (("intervention_match", "intervention"), ("population_model_match", "model"),
                   ("outcome_match", "endpoint"), ("direction_match", "direction"), ("question_match", "question"))


_EVIDENCE_KIND = {"OBSERVED": "Dataset found", "DATASET_FOUND": "Dataset found", "DATASET": "Dataset found"}


def evidence_line(e: Dict[str, Any]) -> str:
    """A GEO series that was FOUND is not an observation: it is labelled 'Dataset found (not analysed)'."""
    kind = _EVIDENCE_KIND.get((e.get("evidence_type") or "").upper(), (e.get("evidence_type") or "Evidence").replace("_", " ").title())
    st = (e.get("statement") or "").replace("Empirical series ", "Public dataset ").replace(
        " identified to evaluate conflicting claims on ", " found (not analysed here) that could help evaluate the disagreement on ")
    return f"- **{kind}:** {st}"


def semantic_grounding(c: Dict[str, Any]) -> str:
    """'5/5 (intervention, model, endpoint, direction, question supported) · causal support: observation only'."""
    a = c.get("semantic_audit") or {}
    gp = a.get("grounding_profile") or {}
    if not gp:
        return "not assessed"
    ok = [lab for k, lab in _SEMANTIC_PARTS if (gp.get(k) or {}).get("status") == "supported"]
    bad = [lab for k, lab in _SEMANTIC_PARTS if (gp.get(k) or {}).get("status") == "not supported"]
    return (f"{len(ok)}/{len(_SEMANTIC_PARTS)} ({', '.join(ok) or 'none'} supported"
            + (f"; NOT supported: {', '.join(bad)}" if bad else "") + ")"
            + f" · causal support: {(gp.get('causal_support') or {}).get('status', 'not assessed')}")


def claim_headline(c: Dict[str, Any]) -> str:
    """Association / observation claims are headlined with what the source supports ('metformin use was associated with
    longer / better survival (observational; association ...)'), never with a bare causal 'survival increased'."""
    a = c.get("semantic_audit") or {}
    if (a.get("claim_status_key") in ("association", "observation", "causal_unsupported")
            or a.get("evidence_role") == "secondary_report") and a.get("evidence_statement"):
        # a review's report is always headlined as one ('A review reports: ...'), whatever its status
        return a["evidence_statement"]
    return c.get("statement") or ""

# ---------------------------------------------------------------------------------------------------------------
# The answer to the question, written from the verified findings only, level by level (people / animals / cells).
# A cell or mouse result is never presented as an effect in people, and a null clinical trial is never hidden behind
# positive laboratory results.
# ---------------------------------------------------------------------------------------------------------------
_LEVEL = {"human_clinical": "human", "human_observational": "human", "animal": "animal", "cell_culture": "cell",
          "organoid_ex_vivo": "cell", "plant": "plant", "microbial": "microbial", "in_silico": "in silico"}
_LEVEL_NAME = {"human": "people", "animal": "animals", "cell": "cells / organoids", "plant": "plants",
               "microbial": "microbes", "in silico": "computer models", "other": "other systems"}
_DOWN_Q = re.compile(r"\b(?:reduc\w*|decreas\w*|lower\w*|inhibit\w*|suppress\w*|prevent\w*|slow\w*|block\w*|"
                     r"protect\w* against|shrink\w*|kill\w*|treat\w*|cure\w*|help\w*|benefit\w*|"
                     # the same ask in the words other fields use: 'attenuate fibrosis', 'alleviate drought stress',
                     # 'mitigate injury', 'abolish the response', 'delay progression'
                     r"attenuat\w*|ameliorat\w*|alleviat\w*|mitigat\w*|abolish\w*|abrogat\w*|diminish\w*|"
                     r"dampen\w*|curb\w*|limit\w*|counteract\w*|reverse\w*|revers\w*|delay\w*|impair\w*|"
                     r"eradicat\w*|clear\w*|resolv\w*|relieve\w*|soften\w*|minimi[sz]\w*)\b", re.I)
_UP_Q = re.compile(r"\b(?:increas\w*|rais\w*|enhanc\w*|promot\w*|induc\w*|stimulat\w*|caus\w*|trigger\w*|"
                   r"worsen\w*|accelerat\w*|"
                   # 'improve tolerance', 'restore function', 'sensitise cells', 'boost yield'
                   r"improv\w*|restor\w*|rescue\w*|rescu\w*|sensiti[sz]\w*|potentiat\w*|augment\w*|boost\w*|"
                   r"elevat\w*|upregulat\w*|activat\w*|strengthen\w*|confer\w*|exacerbat\w*|aggravat\w*|"
                   r"maximi[sz]\w*|amplif\w*|facilitat\w*)\b", re.I)


def expected_direction(question: str) -> Optional[str]:
    """'Does metformin reduce ...' -> 'decrease'; 'Does benzene cause ...' -> 'increase'; an open question -> None."""
    q = question or ""
    if re.match(r"\s*(?:what|which|how|why|who|where|when)\b", q, re.I) and not re.search(r"\bdoes\b|\bdo\b", q, re.I):
        return None
    if re.search(r"\b(?:cure|cured|curable|treat|treated|treatment|helps?|benefit)\b", q, re.I):
        return "cure"
    down, up = len(_DOWN_Q.findall(q)), len(_UP_Q.findall(q))
    return "decrease" if down > up else "increase" if up > down else None


def _stance(c: Dict[str, Any], expected: Optional[str]) -> str:
    d = (c.get("canonical_direction") or c.get("effect_direction") or "unclear").lower()
    if d in ("no_effect", "none", "null"):
        return "null"
    if expected is None or d not in ("increase", "decrease"):
        return "describes"
    if expected == "cure":
        fam = (c.get("canonical_outcome") or "")
        good = "increase" if fam in ("clinical_response",) else "decrease"
        return "supports" if d == good else "opposes"
    return "supports" if d == expected else "opposes"


# endpoints people care about clinically, and which direction is the BENEFIT for each
_CLINICAL_GOOD = {"patient_survival": "increase", "clinical_response": "increase", "recurrence_incidence": "decrease"}
_GROUP = {"proliferation": "growth", "tumour_growth": "growth", "viability": "growth"}


def _endpoint_group(fam: Optional[str]) -> str:
    return _GROUP.get(fam or "", fam or "")


_OUTCOME_WORDS = {("patient_survival", "increase"): "longer survival", ("patient_survival", "decrease"): "shorter survival",
                  ("recurrence_incidence", "decrease"): "less recurrence", ("recurrence_incidence", "increase"): "more recurrence",
                  ("clinical_response", "increase"): "a better clinical response",
                  ("clinical_response", "decrease"): "a worse clinical response"}
_OUTCOME_NAME = {"patient_survival": "survival", "recurrence_incidence": "recurrence", "clinical_response": "clinical response"}


def _n_papers(cs: List[Dict[str, Any]]) -> int:
    return len({c.get("paper_id") or ("C", c.get("id")) for c in cs}) or 1


def _n_independent(cs: List[Dict[str, Any]]) -> int:
    """Publications of the same trial / cohort (T1) count once."""
    return len({c.get("trial") or ("P", c.get("paper_id") or ("C", c.get("id"))) for c in cs}) or 1


def _cid_key(c: Dict[str, Any]) -> int:
    try:
        return int(str(c.get("id")).lstrip("C"))
    except (TypeError, ValueError):
        return 10 ** 9


def _how_many(cs: List[Dict[str, Any]], one: str, many: str) -> str:
    """'an observational study' / '3 observational studies' / '7 observational publications (1 independent cohort)'."""
    n, k = _n_papers(cs), _n_independent(cs)
    if n == 1:
        return one
    if k < n:
        unit = "trial" if "trial" in many else "cohort"
        return f"{n} {many.replace('studies', 'publications').replace('trials', 'trial publications')} " \
               f"({k} independent {unit}{'' if k == 1 else 's'})"
    return f"{n} {many}"


def _clinical_phrase(c: Dict[str, Any], group: Optional[List[Dict[str, Any]]] = None) -> str:
    """'a randomised trial found no significant benefit on survival (C29)' / '7 observational studies reported an
    association with longer survival (C64, C66 ...; association, not proof of cause)' - benefit-aware for each endpoint,
    and every study with the same result is counted and cited (never one ID standing for seven studies)."""
    cs = group or [c]
    fam = c.get("canonical_outcome") or ""
    d = (c.get("canonical_direction") or c.get("effect_direction") or "").lower()
    good = _CLINICAL_GOOD.get(fam)
    what = _OUTCOME_NAME.get(fam) or fam.replace("_", " ") or "a clinical outcome"
    cs = sorted(cs, key=_cid_key)
    rct = c.get("claim_model") == "human_clinical"
    design = _how_many(cs, "a randomised trial", "randomised trials") if rct else \
        _how_many(cs, "an observational study", "observational studies")
    if d == "no_effect":
        verdict = f"found no statistically significant difference in {what}"
    elif good:
        change = _OUTCOME_WORDS.get((fam, d)) or f"{'better' if d == good else 'worse'} {what}"
        verdict = f"reported {change}" if rct else f"reported an association with {change} (association, not proof of cause)"
    else:
        verdict = f"reported {what} {d.replace('_', ' ')}"
    ids = ", ".join(f"C{x.get('id')}" for x in cs[:8]) + (" ..." if len(cs) > 8 else "")
    return f"{design} {verdict} ({ids})"


def _primary(c: Dict[str, Any]) -> bool:
    return ((c.get("semantic_audit") or {}).get("evidence_role") != "secondary_report")


def _answering(c: Dict[str, Any]) -> bool:
    role = (c.get("semantic_audit") or {}).get("evidence_role")
    return role in (None, "direct", "disease_risk")


# marks where the answer paragraph starts in the detailed report, so the engine can swap in the matrix-derived
# answer. Layer 1 counts MERGED findings (one per experiment and endpoint); a raw claim list counts duplicates
# separately, which is how '10 human findings' and '11 human findings' appeared in one report.
ANSWER_MARK = "<!--ANSWER-->"


def level_answer(question: str, claims: List[Dict[str, Any]], quarantined: Optional[List[Dict[str, Any]]] = None,
                 papers: int = 0, medical: Optional[bool] = None) -> Dict[str, Any]:
    """{'text': the answer paragraph, 'levels': {level: [claims]}, 'discordant': bool, 'discordance': text}.
    Deterministic and conservative: it only restates what the verified findings show, at the level they show it."""
    expected = expected_direction(question)
    prim = [c for c in claims if _primary(c)]
    ans = [c for c in prim if _answering(c)]
    if prim and not ans:
        # nothing measures what was asked: say so, and list what WAS measured without drawing a conclusion from it
        by: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for c in prim:
            by[_LEVEL.get(c.get("claim_model") or "", "other")].append(c)
        seen_s: set = set()
        parts = []
        for lvl, cs in by.items():
            uniq = [c for c in cs if not ((c.get("statement") or "").lower() in seen_s or seen_s.add((c.get("statement") or "").lower()))]
            parts.append(f"in {_LEVEL_NAME.get(lvl, lvl)}: " + "; ".join(f"{(c.get('statement') or '').rstrip('.')} (C{c.get('id')})"
                                                                          for c in uniq[:3]))
        text = ("**No verified finding measures what the question asks**, so the papers read cannot answer it directly. "
                "The verified findings measure related endpoints (context only, not an answer) - " + " · ".join(parts) + ". "
                "This summarises the papers read, not the whole literature.")
        return {"text": text, "levels": dict(by), "discordant": False, "discordance": "", "answered": False,
                "verdict": "Not answered by the papers read - no verified finding measures what was asked."}
    levels: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for c in ans:
        levels[_LEVEL.get(c.get("claim_model") or "", "other")].append(c)
    if medical is None:
        medical = bool(re.search(r"\b(?:humans?|patients?|people|women|men|children|clinical|disease|cancer|syndrome|"
                                 r"disorder|infection|diabet\w*|leuk\w*|vitiligo|leukoderma)\b", question or "", re.I)) or \
            bool(levels.get("human"))

    def ids(cs: List[Dict[str, Any]]) -> str:
        cs = sorted(cs, key=_cid_key)
        return ", ".join(f"C{c.get('id')}" for c in cs[:8]) + (" ..." if len(cs) > 8 else "")

    def stmts(cs: List[Dict[str, Any]], n: int = 3) -> str:
        # the same result from several studies is one statement citing ALL of them ('... (C2, C4, C6; 3 studies)')
        order: List[str] = []
        by_key: Dict[str, List[Dict[str, Any]]] = {}
        for c in cs:
            key = (c.get("statement") or "").strip().lower()
            if key not in by_key:
                order.append(key)
                by_key[key] = []
            by_key[key].append(c)
        out = []
        for key in order[:n]:
            g = sorted(by_key[key], key=_cid_key)
            k, ind = _n_papers(g), _n_independent(g)
            count = (f"; {k} studies" + (f", {ind} independent" if ind < k else "") + ")") if k > 1 else ")"
            out.append(f"{(g[0].get('statement') or '').rstrip('.')} ({ids(g)}" + count)
        return "; ".join(out)

    if not claims:
        reasons = Counter()
        for q in quarantined or []:
            for r in (q.get("semantic_audit") or {}).get("reasons") or []:
                key = r.split(":")[1].strip() if r.startswith("relevance_gate:") and ":" in r[15:] else r.split(":")[0]
                reasons[key.replace("_", " ")] += 1
        top = ", ".join(f"{k} ({n})" for k, n in reasons.most_common(3))
        text = (f"**This run could not answer the question.** {papers} paper(s) were read; "
                + (f"{len(quarantined or [])} extracted finding(s) all failed verification (main reasons: {top}). "
                   if quarantined else "no finding could be extracted and verified. ")
                + "Nothing in this report should be read as an answer. A new run with more papers (the default now reads more), "
                  "a more specific question (name the treatment, exposure or organism), or a check of the question's spelling "
                  "usually fixes this.")
        return {"text": text, "levels": {}, "discordant": False, "discordance": "", "answered": False,
                "verdict": "Could not be answered - no finding passed verification."}

    parts: List[str] = []
    human = levels.get("human", [])
    # endpoint-aware: a question about a LABORATORY endpoint (proliferation, apoptosis ...) is answered in people only by
    # human findings on that endpoint; survival / recurrence results are other clinical outcomes, reported separately
    # and judged by their own benefit direction (survival UP = benefit), never as 'does not show the effect'
    from semantic_audit import canonical_outcome as _co
    q_fam = _co(question or "").get("family") or ""
    human_other: List[Dict[str, Any]] = []
    if q_fam and not q_fam.startswith("other:") and q_fam not in _CLINICAL_GOOD:
        other_end = [c for c in human if c.get("canonical_outcome")
                     and _endpoint_group(c.get("canonical_outcome")) != _endpoint_group(q_fam)]
        # 'other CLINICAL outcomes' are clinical outcomes only (survival, recurrence, response, glycaemia, weight ...):
        # a gene-expression read-out in trial tissue is a molecular result, never listed as a clinical outcome
        human_other = [c for c in other_end if c.get("canonical_outcome") in _CLINICAL_GOOD or
                       c.get("canonical_outcome") in ("glycaemic", "body_weight", "inflammation", "cognition_behaviour")]
        human = [c for c in human if c not in other_end]
        if not human:
            levels.pop("human", None)
        else:
            levels["human"] = human
    pre = levels.get("animal", []) + levels.get("cell", [])
    st = {lvl: Counter(_stance(c, expected) for c in cs) for lvl, cs in levels.items()}
    h_sup, h_neg = st.get("human", Counter())["supports"], st.get("human", Counter())["null"] + st.get("human", Counter())["opposes"]
    p_sup = sum(st.get(lv, Counter())["supports"] for lv in ("animal", "cell"))
    p_all = len(pre)

    def pre_ids() -> str:
        # the IDs listed after 'N of M show a reduction' are the N that do; the others are named separately
        sup = [c for c in pre if _stance(c, expected) == "supports"]
        oth = [c for c in pre if c not in sup]
        return ids(sup) + (f"; other direction / no significant difference: {ids(oth)}" if oth else "")
    rct_first = sorted(human, key=lambda c: 0 if c.get("claim_model") == "human_clinical" else 1)
    if expected is None:
        for lvl, cs in levels.items():
            parts.append(f"In {_LEVEL_NAME.get(lvl, lvl)}: {stmts(cs)}.")
        text = ("The verified findings, by level of evidence: " + " ".join(parts) +
                " These are the findings of the papers read, not a complete review of the literature.")
        return {"text": text, "levels": dict(levels), "discordant": False, "discordance": "", "answered": True,
                "verdict": "Descriptive question - see the findings by level below (no direction was asked)."}
    word = {"decrease": "a reduction", "increase": "an increase", "cure": "a benefit"}[expected]
    discordant = bool(human and h_sup == 0 and h_neg > 0 and p_sup > 0)
    if human and h_sup == 0 and h_neg > 0:
        parts.append(f"**The papers read do not establish this effect in people.** The human studies found no statistically significant {word}: "
                     f"{stmts(rct_first)}.")
        if p_sup:
            parts.append(f"Laboratory studies do report {word} ({p_sup} of {p_all} cell / animal finding(s): {pre_ids()}), so the "
                         f"effect is plausible in experimental systems - but that is not demonstrated benefit in humans.")
    elif human and h_sup > 0 and h_neg == 0:
        observational_only = all(c.get("claim_model") == "human_observational" for c in human if _stance(c, expected) == "supports")
        parts.append(f"**Human studies among the papers read point to {word}:** {stmts(rct_first)}."
                     + (" These are observational studies: they show an association, not proof of cause and effect."
                        if observational_only else ""))
        if pre:
            # say how many laboratory findings point the same way - never imply all of them do
            parts.append(f"Laboratory findings: {p_sup} of {p_all} show {word} ({pre_ids()}); they do not add evidence "
                         f"about people.")
    elif human:
        # every counted finding is listed under its own group, so the numbers and the list always agree
        sup_l = [c for c in rct_first if _stance(c, expected) == "supports"]
        nul_l = [c for c in rct_first if _stance(c, expected) == "null"]
        opp_l = [c for c in rct_first if _stance(c, expected) == "opposes"]
        groups = [f"{len(sup_l)} show {word} ({stmts(sup_l, 99)})"]
        if nul_l:
            groups.append(f"{len(nul_l)} show no statistically significant difference ({stmts(nul_l, 99)})")
        if opp_l:
            groups.append(f"{len(opp_l)} show the opposite direction ({stmts(opp_l, 99)})")
        parts.append(f"**The human evidence is mixed:** of {len(sup_l) + len(nul_l) + len(opp_l)} human finding(s) on this "
                     f"endpoint, " + "; ".join(groups) + ". A non-significant result is not proof of no effect; a randomised "
                     "trial outweighs observational data where both exist.")
    elif pre:
        if medical:
            label = _co(question or "").get("label") or "this endpoint"
            parts.append((f"**No human study among the papers read measured {label} itself.** "
                          if human_other else "**No study among the papers read measured this in people.** ")
                         + f"The verified findings on it come only from "
                         f"{' and '.join(_LEVEL_NAME[l] for l in ('cell', 'animal') if levels.get(l))} "
                         f"({p_sup} of {p_all} show {word}: {pre_ids()}), which cannot establish an effect in humans.")
        else:
            parts.append(f"In {' and '.join(_LEVEL_NAME[l] for l in ('cell', 'animal') if levels.get(l))}: "
                         f"{p_sup} of {p_all} verified finding(s) show {word} ({stmts(pre)}).")
    for lvl in ("plant", "microbial", "in silico", "other"):
        if levels.get(lvl):
            s = st[lvl]
            parts.append(f"In {_LEVEL_NAME[lvl]}: {s['supports']} of {len(levels[lvl])} verified finding(s) show {word} "
                         f"({stmts(levels[lvl], 2)}).")
    if human_other:
        grouped: Dict[Any, List[Dict[str, Any]]] = {}
        for c in sorted(human_other, key=lambda c: 0 if c.get("claim_model") == "human_clinical" else 1):
            key = (c.get("claim_model"), c.get("canonical_outcome"), (c.get("canonical_direction") or c.get("effect_direction") or "").lower())
            grouped.setdefault(key, []).append(c)
        other_txt = [_clinical_phrase(cs[0], cs) for cs in grouped.values()]
        parts.append("**Other clinical outcomes in people** (not the endpoint asked about): " + "; ".join(other_txt[:4]) + ".")
    parts.append("This summarises the papers read, not the whole literature.")
    discord = (f"Preclinical evidence is predominantly supportive ({p_sup} of {p_all} cell / animal findings show {word}), "
               f"whereas the strongest clinical evidence ({ids(rct_first[:2])}) does not show the effect. They come from "
               f"different systems, so this is an evidence-level conflict (a cross-context disagreement, not a direct contradiction) - the answer depends on the level of evidence."
               if discordant else "")
    rct_null_other = [c for c in human_other if c.get("claim_model") == "human_clinical"
                      and (c.get("canonical_direction") or "").lower() == "no_effect"]
    if not discord and rct_null_other and p_sup:
        discordant = True
        obs_good = [c for c in human_other if c.get("claim_model") == "human_observational"
                    and (c.get("canonical_direction") or "").lower() == _CLINICAL_GOOD.get(c.get("canonical_outcome") or "")]
        discord = (f"Preclinical evidence is predominantly supportive ({p_sup} of {p_all} cell / animal findings show {word}), "
                   f"but the randomised clinical evidence ({ids(rct_null_other[:2])}) found no significant benefit on a clinical "
                   f"outcome" + (f", while observational data ({ids(obs_good[:2])}) report an association with benefit "
                                 f"(confounding cannot be excluded)" if obs_good else "")
                   + (". No human study among the papers read measured the asked endpoint itself." if not human else
                      f". Human studies that measured the asked endpoint itself are reported above ({ids(human)})."))
    # one plain verdict line (Layer 1 of the dossier): never stronger than the human evidence on the asked endpoint
    rct = [c for c in human if c.get("claim_model") == "human_clinical"]
    rct_sup = [c for c in rct if _stance(c, expected) == "supports"]
    if human and h_sup and not h_neg:
        verdict = ("Yes in people, within the papers read - supported by randomised human evidence."
                   if rct_sup else "Possibly - an association in people (observational evidence only), not proven.")
    elif human and h_neg and not h_sup:
        verdict = ("Not established in people - human studies found no statistically significant effect"
                   + (", although laboratory studies support it." if p_sup else "."))
    elif human:
        verdict = "Uncertain - the human evidence is mixed."
    elif pre and p_sup:
        verdict = ("Possibly, but not established in humans - only laboratory (cell / animal) evidence measured it."
                   if medical else "Supported in the experimental systems studied (see the findings by level).")
    elif pre:
        verdict = "Not supported by the laboratory evidence read."
    elif human_other:
        verdict = (f"Not answered for {_co(question or '').get('label') or 'this endpoint'} - the human studies read measured "
                   f"other clinical outcomes only (see below).")
    else:
        verdict = "Not answered by the papers read."
    if expected == "cure":
        # a cure question is answered against the cure definition, not by counting 'improvement' findings: those
        # measure repigmentation / response / recurrence - different outcomes, not tests of cure
        from clinical_context import OUTCOME_LADDER, meets_cure, CURE_MIN_FOLLOW_UP_DAYS
        lvl_of = {k: lv for lv, k, _, _ in OUTCOME_LADDER}
        cured = [c for c in ans if meets_cure(lvl_of.get((c.get("semantic_audit") or {}).get("outcome_level")),
                                              (c.get("semantic_audit") or {}).get("follow_up_days"))]
        months = CURE_MIN_FOLLOW_UP_DAYS // 30
        if cured:
            verdict = (f"A sustained remission meeting the cure definition (complete response, no recurrence for ≥{months} months) "
                       f"is reported in {ids(cured)} - within the papers read, not proof for every patient or subtype.")
        else:
            verdict = (f"No sustained cure was demonstrated in the papers read (cure = complete response with no recurrence for "
                       f"≥{months} months after treatment). This is not evidence that the condition cannot be cured.")
            parts.insert(0, f"**Against the cure definition:** no verified finding reports a complete response sustained ≥{months} "
                            f"months. The findings below concern improvement, response or recurrence - different outcomes, not "
                            f"tests of cure.")
    return {"text": " ".join(parts), "levels": dict(levels), "discordant": discordant, "discordance": discord,
            "answered": True, "verdict": verdict}


def _kind_counts(hypotheses: List[Dict[str, Any]]) -> str:
    from reasoning_guards import hypothesis_kind
    names = {"replication": "replication", "mechanistic": "evidence-based mechanistic", "network_candidate": "network candidate",
             "mechanistic_ai": "AI-proposed mechanistic"}
    cnt = Counter(hypothesis_kind(h)[0] for h in hypotheses or [])
    return ", ".join(f"{n} {names[k]}" for k, n in cnt.items()) or "none"


class ResearchReportGenerator:
    def __init__(self, db: ResearchDB):
        self.db = db

    def _get_balanced_claims(self, claims: List[Dict[str, Any]], max_total: int = 20) -> List[Dict[str, Any]]:
        """Round-robin selection across papers to prevent 1 paper from monopolizing the report."""
        by_paper: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
        for c in claims:
            pid = c.get("paper_id") or 0
            by_paper[pid].append(c)

        balanced: List[Dict[str, Any]] = []
        papers = list(by_paper.keys())
        idx = 0
        while len(balanced) < max_total and any(by_paper.values()):
            curr_pid = papers[idx % len(papers)]
            if by_paper[curr_pid]:
                balanced.append(by_paper[curr_pid].pop(0))
            idx += 1
            if not any(by_paper.values()):
                break
        return balanced

    def generate(self, research_question_id: int, output_path: str = "RESEARCH_REPORT.md") -> str:
        rq = self.db.get_research_question(research_question_id)
        question = rq["question"] if rq else "Unknown"
        summary = self.db.research_question_summary(research_question_id)
        raw_claims = self.db.verified_claims(research_question_id)          # the only evidence shown as verified
        primary_claims = [c for c in raw_claims if (c.get("semantic_audit") or {}).get("evidence_role") != "secondary_report"]
        quarantined = self.db.quarantined_claims(research_question_id)
        claims = self._get_balanced_claims(raw_claims, max_total=20)
        # every claim referenced elsewhere in the report (contradictions) is shown in full, under its real ID
        shown = {c["id"] for c in claims}
        referenced = {cid for cd in self.db.current_contradictions(research_question_id) for cid in (cd.get("claim_a_id"), cd.get("claim_b_id"))}
        claims += [c for c in raw_claims if c["id"] in referenced and c["id"] not in shown]
        contradictions = self.db.current_contradictions(research_question_id)
        hypotheses = self.db.list_hypotheses(research_question_id)
        gaps = self.db.list_research_gaps(research_question_id)
        audit_flags = self.db.list_audit_flags(research_question_id)
        evidence = self.db.list_evidence(research_question_id)

        md = []
        md.append("================================================================================")
        md.append("                         AUTONOMOUS SCIENTIFIC RESEARCH REPORT                  ")
        md.append("================================================================================\n")
        md.append(f"**Research Question:** {question.replace('<', '&lt;').replace('>', '&gt;')}\n")
        ans = level_answer(question, raw_claims, quarantined, int(summary.get("papers", 0) or 0))
        md.append("## Answer to the research question\n")
        md.append(ANSWER_MARK + "\n")        # replaced with the evidence matrix's answer once that exists, so the
        md.append(ans["text"] + "\n")        # dossier and this report can never show two different tallies
        md.append("## Executive Summary\n")
        md.append(
            f"This autonomous investigation deep-read **{summary.get('papers', 0)} papers** (every paper read for this "
            f"question, including any later excluded as off-question or retracted - the evidence matrix lists the included "
            f"studies S1, S2 ...) "
            f"and kept **{len(raw_claims)} verified claims** (quote found in the paper AND the quote supports the claim's "
            f"intervention, model, endpoint, direction, magnitude and significance). "
            f"**{len(quarantined)} claims were quarantined** and are excluded from every synthesis step.\n"
            f"- Cross-study comparison: **{contradiction_vocabulary(contradictions, primary_claims)}**. A direct contradiction "
            f"needs the same system, endpoint and context; cross-context disagreements and evidence-level conflicts are real "
            f"disagreements in the literature that are explained (or not) by differences in system, population, dose or design.\n"
            f"- Formulated **{len(hypotheses)} testable {'hypothesis' if len(hypotheses) == 1 else 'hypotheses'}** ({_kind_counts(hypotheses)}) with experimental validation specifications.\n"
            f"- Identified **{len(gaps)} key unaddressed research {'gap' if len(gaps) == 1 else 'gaps'}**.\n"
            f"- Recorded **{len(audit_flags)} deterministic audit {'flag' if len(audit_flags) == 1 else 'flags'}** for quality assurance.\n"
        )

        md.append("## 1. Verified Literature Evidence (Balanced Cross-Study View)\n")
        md.append("_Claims are numbered by their permanent ID (C#); contradictions, hypotheses and gaps cite the same IDs._\n")
        n_secondary = sum(1 for c in raw_claims if ((c.get("semantic_audit") or {}).get("evidence_role") == "secondary_report"))
        md.append(f"_{len(raw_claims) - n_secondary} claim(s) come from primary studies; {n_secondary} are results REPORTED BY "
                  f"REVIEWS (secondary literature - the original experiment was not verified by BioSynthetica)._\n")
        for c in claims:
            md.append(f"### C{c.get('id')} · {claim_headline(c)}")
            md.append(f"- **Paper:** {c.get('paper_title') or 'Unknown'} (PMID: {c.get('pmid') or 'N/A'}, PMCID: {c.get('pmcid') or 'N/A'})")
            md.append(f"- **Intervention / Factor:** `{c.get('intervention')}` -> **Outcome:** `{c.get('measurement')}`")
            md.append(f"- **Direction:** `{c.get('effect_direction')}` | **p-value:** `{c.get('p_value') if c.get('p_value') is not None else 'Unspecified'}`")
            md.append(f"- **Verbatim Grounded Quote:** *\"{c.get('quote')}\"*")
            md.append(f"- **Source section:** {c.get('source_section') or 'unknown'} | **Model:** "
                      f"{(c.get('claim_model') or 'unknown').replace('_', ' ')} | **Canonical:** "
                      f"{(c.get('canonical_outcome') or '?').replace('_', ' ')} {c.get('canonical_direction') or ''}")
            a = c.get("semantic_audit") or {}
            md.append(f"- **Claim status:** {a.get('claim_status', '—')} | **Tags:** {', '.join(a.get('tags') or []) or '—'} | "
                      f"**Inference:** {(a.get('inference_type') or '—').replace('_', ' ')}")
            gp = a.get("grounding_profile") or {}
            if gp:
                md.append("- **Grounding profile:** " + "; ".join(f"{k.replace('_', ' ')}: {v.get('status')}" for k, v in gp.items()))
            if a.get("evidence_statement") and a.get("evidence_statement") != c.get("statement"):
                md.append(f"- **What the source supports:** {a['evidence_statement']}")
            ident = a.get("identity") or {}
            if ident:
                st = ident.get("statistical_evidence")
                md.append(f"- **Evidence identity:** origin: {ident.get('original_experiment')} · type: {ident.get('evidence_type')} · "
                          f"species: {ident.get('species')} · disease: {ident.get('disease')} · subtype: "
                          f"{', '.join(ident.get('disease_subtype') or [])} · cell/tissue: {', '.join(ident.get('cell_tissue') or [])} · "
                          f"exposure: {ident.get('intervention_exposure')} · comparator: {ident.get('comparator')} · dose: "
                          f"{ident.get('dose')} · duration: {ident.get('duration')} · statistics: "
                          f"{st if isinstance(st, str) else ', '.join(str(x) for x in (st.get('p_values') or []) + (st.get('ratios') or []))} · "
                          f"causal wording: {', '.join((ident.get('causal_language') or {}).get('causal') or []) or 'none'}"
                          f" · target disease: {ident.get('target_disease') or 'not stated'} · subgroup / comorbidity: "
                          f"{', '.join(ident.get('comorbidity') or []) or 'none'} · readout: "
                          f"{(ident.get('readout') or 'not stated').replace('_', ' ')}")
            if a.get("evidence_role_label"):
                md.append(f"- **Role in answering the question:** {a['evidence_role_label']}")
            md.append(f"- **Textual grounding:** `{c.get('grounding_score', 1.0)}` (the quote occurs in the paper - text "
                      f"correspondence, NOT scientific relevance)")
            md.append(f"- **Semantic grounding:** {semantic_grounding(c)} (does the quote support the claim's intervention, "
                      f"model, endpoint, direction and question?)\n")
        rest = [c for c in raw_claims if c["id"] not in {x["id"] for x in claims}]
        if rest:
            md.append("### Other verified claims (compact list)\n")
            md.extend(f"- C{c['id']}: {claim_headline(c)} — *{(c.get('paper_title') or '')[:60]}*" for c in rest)
            md.append("")

        md.append("## 1b. Quarantined Claims (excluded from all synthesis)\n")
        off = [q for q in quarantined if any(r.startswith("relevance_gate") for r in (q.get("semantic_audit") or {}).get("reasons") or [])]
        if off:
            md.append(f"**{len(off)} claim(s) excluded as off-question** (their quote may match the paper exactly, but the disease, "
                      f"model, exposure or outcome does not match the research question):\n")
            for q in off[:40]:
                rs = [r.replace("relevance_gate: ", "") for r in (q.get("semantic_audit") or {}).get("reasons") or [] if r.startswith("relevance_gate")]
                md.append(f"- **C{q.get('id')}** {q.get('statement')} — *{(q.get('paper_title') or 'Unknown')[:70]}* — {'; '.join(rs)}")
            md.append("")
            md.append("**Other quarantined claims:**\n")
        quarantined = [q for q in quarantined if q not in off]
        if quarantined:
            for q in quarantined[:40]:
                reasons = (q.get("semantic_audit") or {}).get("reasons") or ["failed verification"]
                md.append(f"- **C{q.get('id')}** {q.get('statement')} — *{(q.get('paper_title') or 'Unknown')[:70]}*")
                md.append(f"  - Reason: {'; '.join(reasons)}")
        else:
            md.append("- None." if off else "- None - every extracted claim passed textual, semantic and relevance verification.")
        md.append("")

        md.append("## 2. Experimental Data & Evidence Synthesis\n")
        exp_evidence = [e for e in evidence if e.get("source_type") in ("dataset", "experiment")]
        if exp_evidence:
            for e in exp_evidence:
                md.append(evidence_line(e))
        else:
            md.append("*No external user experimental dataset was linked for this investigation run.*")
        md.append("")

        md.append("## 3. Cross-Study Agreements & Contradictions\n")
        md.append(f"**{contradiction_vocabulary(contradictions, primary_claims)}**\n")
        same_sys = [c for c in contradictions if c.get("kind") != "model_mismatch"]
        cross_sys = [c for c in contradictions if c.get("kind") == "model_mismatch"]
        if cross_sys:
            md.append(f"_{len(cross_sys)} opposite result(s) come from different experimental systems (e.g. patients vs cells); "
                      f"they are listed after the same-system disagreements and are not contradictions._\n")
        if ans.get("discordance"):
            md.append(f"**Across levels of evidence:** {ans['discordance']}\n")
        if contradictions:
            for i, cd in enumerate(same_sys + cross_sys, 1):
                label = ("Cross-context disagreement - different experimental systems" if cd.get("kind") == "model_mismatch"
                         else "Direct contradiction" if cd.get("verdict") == "TRUE_CONTRADICTION"
                         else "Cross-context disagreement" if cd.get("verdict") == "CONTEXT_DIFFERENCE"
                         else "Unresolved disagreement")
                md.append(f"### Disagreement #{i}: {label}")
                md.append(f"- **Between Claims:** Claim C{cd.get('claim_a_id')} vs Claim C{cd.get('claim_b_id')}")
                md.append(f"- **Why (context comparator):** {cd.get('reasoning')}")
                comp = (cd.get("comparison") or {}).get("comparison") or []
                if comp:
                    md.append("- **Compared on:** " + "; ".join(f"{r['dimension'].replace('_', ' ')} = {r['status']}" for r in comp))
                md.append("")
        else:
            md.append("- No two findings from the same experimental system disagree in direction"
                      + (" (see the cross-level note above)." if ans.get("discordance") else ".") + "\n")

        # order: Evidence -> verified claims -> GAPS -> replication hypotheses -> mechanistic hypotheses
        md.append("## 4. Unaddressed Research Gaps\n")
        if not gaps:
            md.append("- None recorded for this run" + (" (no verified finding to compare against)." if not raw_claims else ".") + "\n")
        for i, g in enumerate(gaps, 1):
            md.append(f"{i}. **[{(g.get('category') or 'General').upper()}]** {g.get('description')}")
            md.append(f"   *Supporting Observation:* {g.get('supporting_observation')}\n")

        md.append("## 5. Proposed Hypotheses for Testing (not conclusions)\n")
        md.append("_Hypotheses are proposals to test next, not results of this investigation. Replication hypotheses ask "
                  "whether a verified finding holds elsewhere (they explain nothing); mechanistic hypotheses propose how an "
                  "effect arises and are labelled by whether their steps are verified._\n")
        if not hypotheses:
            md.append("- No hypothesis was formed: " + ("no verified finding was available to build one on."
                                                       if not raw_claims else "none passed the evidence checks.") + "\n")
        from reasoning_guards import KIND_SHORT, fix_falsification, hypothesis_kind
        for h in hypotheses:
            fix_falsification(h)                 # a criterion that would SUPPORT the hypothesis is never printed
            if not h.get("kind"):
                h["kind"], h["kind_label"] = hypothesis_kind(h)
        groups = [("replication", "5a. Replication / translation hypotheses"), ("mechanistic", "5b. Mechanistic hypotheses (evidence-based)"),
                  ("network_candidate", "5c. Mechanism candidates (protein network)"),
                  ("mechanistic_ai", "5d. Mechanistic hypotheses proposed by the AI (not established)")]
        for kind, head in groups:
            hs_ = [h for h in hypotheses if h.get("kind") == kind]
            if not hs_:
                continue
            md.append(f"### {head}\n")
            for i, h in enumerate(hs_, 1):
                sup = h.get("support") or {}
                md.append(f"#### {h.get('title') or h.get('label') or 'H' + str(i)}: {h.get('statement')}")
                md.append(f"- **Type:** {h.get('kind_label') or KIND_SHORT.get(kind)}")
                md.append("- **Status:** " + (f"derived from {sup.get('n_papers', 0)} paper(s) with verified findings (score {sup.get('score')})"
                                              if sup.get("score") else "**proposal only - no verified finding supports it yet** (AI hypothesis, L7)"))
                md.append(f"- **Mechanism:** {h.get('mechanism')}")
                md.append("- **Key Predictions:**")
                for p in h.get("predictions") or []:
                    md.append(f"  * {p}")
                md.append("- **Falsification Conditions:**")
                for f in h.get("falsification_conditions") or []:
                    md.append(f"  * {f}")
                exp = h.get("next_experiment")
                if exp:
                    from enhanced_report import experiment_text
                    md.append(f"- **Proposed Experiment Specification:** {experiment_text(exp)}\n")

        md.append("## 6. Adversarial Audit & Quality Control Flags\n")
        if audit_flags:
            for f in audit_flags:
                ref = str(f.get("evidence_reference") or "")
                pm = re.fullmatch(r"paper:(\d+)", ref)
                if pm:
                    # an internal row id means nothing to the reader: name the claim and the paper instead
                    p_ = self.db.get_paper(int(pm.group(1))) or {}
                    ref = ((f"claim C{f.get('entity_id')} · " if f.get("entity_type") == "claim" and f.get("entity_id") else "")
                           + f"paper: {(p_.get('title') or 'unknown')[:70]}"
                           + (f" (PMID {p_.get('pmid')})" if p_.get("pmid") else ""))
                md.append(f"- **[{f.get('severity')}]** {f.get('issue')} (Reference: {ref})")
        else:
            md.append("- *No audit flags were raised.*")

        report_str = "\n".join(md)
        from core_utils import output_path as _out
        Path(_out(output_path)).write_text(report_str, encoding="utf-8")
        return report_str