"""
reasoning_guards.py
-------------------
Deterministic safeguards applied to everything an AI model writes.

#3  Evidence-level leaps        "cell experiment -> therefore human mechanism" and
                                "survival association -> therefore reduced proliferation" are
                                detected sentence by sentence and annotated in place.
#12 Mechanism evidence levels   every mechanistic step is labelled
                                  M1 candidate (STRING / database association only)
                                  M2 experimental association (a verified study observed it)
                                  M3 causal evidence (genetic/pharmacological perturbation) in a model
                                  M4 causal evidence in the organism the question is about
                                  L7 AI hypothesis (no study, no database support)
#13 Hypothesis consolidation    near-duplicate hypotheses are merged, then ranked by verified
                                support weighted by evidence level - never "best hypothesis".
#14 AI-proposed parameters      doses, sample sizes, thresholds, effect sizes, durations and
                                power that no verified source supports are labelled
                                "AI-PROPOSED EXPERIMENTAL PARAMETER".
#15 Traceability                every cited statement -> study -> experiment/finding -> claim ->
                                exact source passage.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Tuple

from semantic_audit import FAMILY_RX, FAMILY_LABEL

CITE_RE = re.compile(r"\[(S\d+(?:\s*,\s*S\d+)*)\]")
HUMAN_SENT = re.compile(r"\b(in (?:humans?|patients|people|women|men|participants)|humans?|patients?|(?<![-\w])clinical(?:ly)?(?! significance| relevance)|"
                        r"in people|women|men|participants?)\b", re.I)
ABSENCE = re.compile(r"\b(no (?:\w+ ){0,3}(?:study|studies|trial|trials) (?:directly )?(?:measured|assessed|reported)|"
                     r"no (?:\w+ ){0,2}(?:endpoint|marker|measure|outcome)s? (?:was|were) (?:\w+ )?(?:assessed|measured|reported)|"
                     r"without (?:a |any )?(?:\w+ ){0,2}(?:endpoint|marker|measure|outcome)|"
                     r"not (?:directly )?(?:measured|assessed|reported)|did not (?:directly )?(?:measure|assess|evaluate|examine|"
                     r"report|test|include)|absent|lack(?:s|ing)? of|"
                     r"none (?:of (?:them|these|the \w+) )?(?:measured|assessed|reported)|other endpoints only|"
                     r"not the endpoint asked about|measured (?:\w+ ){0,3}itself|"
                     r"without (?:a |any )?(?:\w+ ){0,2}(?:read-?outs?|data|assessment|measurement)|no (?:\w+ ){0,2}read-?outs?|"
                     r"(?:a )?distinct (?:endpoint|outcome) from|does not address|do not address|not equivalent to|"
                     r"is not (?:the same as|a measure of))\b", re.I)
PERTURB = re.compile(r"\b(knock-?down|knock-?out|siRNA|shRNA|CRISPR|deletion|deficien\w+|null mice|-/-|inhibitor|"
                     r"antagonist|blocked by|abolished|abrogated|rescued|reversed|dominant[- ]negative|"
                     r"was required|is required|were required|dependent on|depleti\w+|silenc\w+)\b", re.I)
LEVEL_WEIGHT = {1: 6, 2: 5, 3: 4, 4: 3, 5: 2, 6: 1}
TYPE_NAME = {"human_clinical": "human clinical", "human_observational": "human observational", "animal": "animal",
             "organoid_ex_vivo": "organoid/ex vivo", "cell_culture": "cell culture", "in_silico": "in silico",
             "review_meta": "review", "plant": "plant", "microbial": "microbial", "unclear": "unclear design",
             "case_report": "case report", "experimental_disease_model": "experimental disease model"}
GUARD_FAMILIES = [f for f, _, _ in FAMILY_RX if f not in ("expression", "signalling")]


def _sentences(text: str) -> List[str]:
    return [s for s in re.split(r"(?<=[.!?])\s+(?=[A-Z\[(])", text or "") if s.strip()]


def _cited(sent: str) -> List[str]:
    return [r.strip() for m in CITE_RE.finditer(sent) for r in m.group(1).split(",")]


def _append_note(sent: str, note: str) -> str:
    """Put a bracketed note INSIDE the sentence (before its final full stop), so a later sentence split keeps it with
    the sentence it belongs to - it was being read as the start of the NEXT sentence."""
    m = re.search(r"([.!?])(\s*)$", sent)
    if m:
        return sent[:m.start()].rstrip() + f" {note}{m.group(1)}"
    return sent.rstrip() + f" {note}"


_SAMPLE_PHRASE = re.compile(r"\b(?:in|within|from|of) (?:the |this )?(?:retrieved|deep-read|examined|included|reviewed|current) "
                            r"(?:sample|set|literature)\b|\b(?:the )?(?:retrieved|deep-read) sample\b", re.I)
# 'leading to low certainty', 'resulting in imprecise estimates': a chain about the EVIDENCE, not about biology
_QUALITY_CHAIN = re.compile(r"(?:leading to|leads? to|resulting in|results? in)\s+(?:\w+\s+){0,2}?(?:low|limited|reduced|"
                            r"uncertain\w*|certainty|confidence|imprecis\w*|heterogene\w*|bias|inconclusive|wide confidence)", re.I)
# people described by a condition ('non-diabetic women', 'insulin-resistant patients', 'without diabetes') - the sentence is
# not ABOUT that disease
_POP_DESCRIPTOR = re.compile(r"\b(?:non-?diabetic|diabetic (?:patients|women|men|participants|status|state)|(?:with|without) "
                             r"(?:type 2 )?diabetes|insulin[- ]resist\w*|hyperinsulin\w*|metabolic (?:status|state|differences?|"
                             r"context|profile)|obes\w*|overweight)\b", re.I)
# 'reduced Ki-67, which would improve survival' / 'translates into better outcomes': a surrogate read as a patient benefit
SURROGATE_LEAP = re.compile(r"(?:ki-?67|proliferat\w*)[^.;]{0,90}\b(?:translat\w* (?:in)?to|predicts?|leads? to|would (?:improve|"
                            r"reduce|prolong)|improv\w* (?:patient )?(?:survival|outcomes?|prognosis)|clinical benefit|better "
                            r"(?:survival|outcomes?|prognosis))", re.I)
# a sentence that says something was NOT available is true when nothing of that kind was verified ('No deep-read animal
# studies were available in the retrieved sample') - it is never an 'evidence-type mischaracterisation'
ABSENCE_SENT = re.compile(r"\b(?:no|none of the|not any)\b[^.]{0,90}?\b(?:(?:were|was|are|is) (?:not )?(?:available|found|identified|"
                          r"deep-read|included|retrieved|verified|examined|read)|remained|after (?:quality )?filtering)\b|"
                          r"\b(?:were|was) not (?:deep-read|available|included|retrieved|verified|examined)\b", re.I)
# population descriptors are not endpoints: 'insulin-resistant women', 'women with HOMA > 2.8', 'non-diabetic patients',
# 'invasive breast cancer', 'invasive disease-free survival', the verb 'yield' ('yield conflicting directions')
_DESCRIPTOR = re.compile(r"\b(?:insulin[- ]resist\w*|hyperinsulin\w*|insulin[- ](?:dependent|independent|sensitive|mediated))"
                         r"(?:\s+(?:women|men|patients|participants|subjects|people|individuals|subgroups?|cohorts?))?|"
                         r"\b(?:with|of|and)\s+(?:a\s+)?HOMA(?:-IR)?\s*(?:index\s*)?[><=≥≤]\s*[\d.]+|"
                         r"\b(?:metabolic|glycaemic|glycemic) (?:baselines?|status|profiles?|states?|phenotypes?)|"
                         r"\b(?:non-?|minimally )?invasive\b(?![\s-]+(?:capacity|ability|potential|phenotype|behaviou?r|properties|growth))|"
                         r"\byield(?:s|ed)?\s+(?:conflict\w*|similar|different|mixed|inconsistent|divergent|contradictory|comparable|"
                         r"results?|findings|no|a|an|the)\b", re.I)


def _families_in(text: str) -> List[str]:
    text = _DESCRIPTOR.sub(" ", text or "")
    out = [f for f, rx, _ in FAMILY_RX if f in GUARD_FAMILIES and rx.search(text or "")]
    if "clinical_response" in out and len(out) > 1:
        # 'an improvement in survival' / 'improved overall survival': the generic word qualifies the other endpoint
        cr = next(rx for f, rx, _ in FAMILY_RX if f == "clinical_response")
        if all(re.fullmatch(r"improv\w*", m.group(0), re.I) for m in cr.finditer(text or "")):
            out.remove("clinical_response")
    return out


def _cited_clauses(sent: str) -> List[Tuple[str, List[str]]]:
    """(clause, refs) for the clauses of a sentence. A contrast clause without its own citation ('..., whereas trials in
    people found no survival benefit') is not about the studies cited in the other clause."""
    parts = re.split(r";|,\s*(?:while|whereas|but|although|though)\b|\b(?:whereas|while|although)\b", sent)
    out = []
    for p_ in parts:
        out.append((p_, _cited(p_)))
    if len(out) > 1 and any(r for _, r in out):
        return [(c, r) for c, r in out if r]
    return [(sent, _cited(sent))]


# --------------------------------------------------------------------------- #3 leap guards
NEG_CONCL = re.compile(
    r"\b(?:cannot be (?:cured|treated|reversed)|can ?not be (?:cured|treated)|not curable|incurable|is not (?:effective|curable|treatable)|"
    r"are not effective|is ineffective|are ineffective|does not work|do not work|has no effect|have no effect|no (?:effective )?"
    r"(?:treatment|therapy|cure) (?:exists|is available)|provides? no benefit|(?:does|do) not support (?:that )?(?:\w+ ){0,4}"
    r"(?:can|could|is|are) (?:be )?(?:cured|curable|effective)|(?:shows?|demonstrates?|proves?|indicates?) (?:that )?(?:\w+ ){0,3}"
    r"(?:cannot|can't|is not|are not|does not))\b", re.I)
CURE_POS = re.compile(r"\b(cur(?:e|ed|es|able|ative)|eradicat\w*|permanent(?:ly)? (?:resolv|clear)\w*)\b", re.I)
CURE_NEGATED = re.compile(r"(?:no (?:evidence|study|studies|trial|data)|not (?:been )?(?:shown|demonstrated|established|identified)|"
                          r"without|absence of|did not|does not show|cannot be|can ?not be|not curable|whether|if)[^.]{0,80}$", re.I)
EFFICACY = re.compile(r"\b(effective|efficacy|works?|benefit\w*|improv\w*|reduc\w*|increas\w*|decreas\w*|cur\w*|treat\w*|"
                      r"recover\w*|respon\w*|remission|associated|caus\w*|led to|leads to|prevent\w*|lower\w*|rais\w*)\b", re.I)
LIMITATION = re.compile(r"\b(no (?:verified )?evidence|not identified|was not found|were not found|limited|sample|uncertain|"
                        r"insufficient|budget|abstract|absence of evidence|cannot be drawn|no conclusion|not established|"
                        r"not assessed|unknown|caution|scope|"
                        # honest limitation / uncertainty sentences are never 'uncited conclusions'
                        r"(?:does|do|did) not (?:directly )?(?:demonstrate|show|establish|measure|capture|address)|"
                        r"not (?:directly )?(?:demonstrated|shown|measured|established|tested)|no human (?:study|studies|trial|trials|"
                        r"evidence|data)|whether|remains? (?:unclear|unknown|uncertain|untested)|certainty|"
                        r"(?:is|are|remains?) (?:low|weak|sparse|limited)|translat\w*|relevance|significance|extrapolat\w*|"
                        r"cannot (?:capture|be extrapolated|be generali[sz]ed|substitute)|(?:do|does|did|may) not (?:reflect|apply|"
                        r"translate|represent)|non-?physiological|supra-?physiological|limited relevance|caveat)", re.I)
SUBSTANTIVE_FIELDS = ("executive_summary", "bottom_line")

# --- evidence-identity rules (leukemia review) ---------------------------------------------------
TYPE_WORDS = [
    ("in_silico", r"\b(?:computational|in silico|bioinformatic\w*|modell?ing (?:data|stud\w+|results?)|simulations?|docking)\b"),
    ("animal", r"\b(?:animal (?:data|stud\w+|models?|experiments?|evidence)|(?:in |only in |among )animals|animals? (?:show|showed|"
               r"report\w*)|mice|mouse|murine|rats?|rodents?|xenograft (?:stud\w+|data|models?))\b"),
    ("cell_culture", r"\b(?:cell[- ](?:culture|line|based)s?|in vitro|cultured cells)\b"),
    # a trial's 'unstratified cohort' is not observational evidence: only a cohort STUDY / design is
    ("human_observational", r"\b(?:epidemiolog\w*|(?:prospective|retrospective|population|observational|registry)[- ]?(?:based )?"
                            r"cohorts?|cohort (?:stud\w+|data|analys\w+)|case-control|population[- ]based|observational (?:data|stud\w+|evidence)|"
                            r"exposed workers|occupational (?:data|stud\w+|cohorts?))\b"),
    ("human_clinical", r"\b(?:clinical trials?|randomi[sz]ed|trial data|clinical (?:data|evidence|stud\w+|findings)|"
                       r"(?:presurgical|window|window-of-opportunity) trials?)\b"),
]
TYPE_GROUP = {"in_silico": {"in_silico"}, "animal": {"animal", "experimental_disease_model"},
              "cell_culture": {"cell_culture", "organoid_ex_vivo"}, "human_observational": {"human_observational", "case_report"},
              "human_clinical": {"human_clinical"}}
ABSENT_TYPE = re.compile(r"\b(?:no|without|lack(?:s|ing)? of|absence of)\s+(?:\w+\s+){0,2}?"
                         r"(human|epidemiological|clinical|observational|animal|in vivo|cell[- ]culture|in vitro|cell)\b"
                         r"(?:\s+(?:or|and|/)\s+(?:\w+\s+)?(human|epidemiological|clinical|observational|animal|in vivo))?"
                         r"[^.]{0,30}?\b(studies|study|evidence|data|trials?)\b"
                         r"|\b(human|epidemiological|clinical) (?:studies|evidence|data) (?:are|is|were|was) "
                         r"(?:absent|lacking|unavailable|not available|missing)", re.I)
ABSENT_MAP = {"human": {"human_clinical", "human_observational", "case_report"}, "epidemiological": {"human_observational"},
              "observational": {"human_observational"}, "clinical": {"human_clinical"}, "animal": {"animal", "experimental_disease_model"},
              "in vivo": {"animal", "experimental_disease_model", "human_clinical", "human_observational"},
              "cell": {"cell_culture"}, "cell-culture": {"cell_culture"}, "cell culture": {"cell_culture"}, "in vitro": {"cell_culture", "organoid_ex_vivo"}}
CAUSE_STMT = re.compile(r"\b(?:causes?|caused|causing|causal(?:ly)?|cause of|a?etiolog\w*|responsible for|drives?|driver|"
                        r"triggers?|plausible causes?|risk factors?|leukemogenic|carcinogenic)\b", re.I)
CAUSE_NEGATED = re.compile(r"\b(?:not (?:been )?(?:established|demonstrated|shown|proven)|does not (?:establish|show|demonstrate)|"
                           r"do not (?:establish|show|demonstrate)|cannot (?:establish|be inferred)|no (?:study|evidence|finding)|"
                           r"whether|unknown|unclear|not a demonstrated|is not evidence)\b", re.I)
CHAIN = re.compile(r"(?:→|->|\bleading to\b|\bleads? to\b|\bresulting in\b|\bresults? in\b|\bwhich (?:then |in turn )?"
                   r"(?:causes?|leads? to|drives?|promotes?)\b|\bin turn\b|\bultimately\b)", re.I)
QC_CLAIM = re.compile(r"\b(?:all|every|each)\b[^.]{0,60}\b(?:met|meet|meets|passed|pass|satisf\w+|fulfil\w*)\b[^.]{0,90}"
                      r"\b(?:power|dose|criteria|quality|grounding|validity)\b|\b(?:adequately|sufficiently|well)[- ]powered\b|"
                      r"\bmet (?:the )?statistical power\b|\bdose validity (?:was|were) (?:met|confirmed|established)\b", re.I)


def _rows_with_findings(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [r for r in rows if (r.get("n_findings") or len(r.get("findings") or [])) > 0]


def _row_types(r: Dict[str, Any]) -> set:
    return {r.get("evidence_type")} | {f.get("model_type") for f in r.get("findings") or [] if f.get("model_type")}


def _qc_summary(rows: List[Dict[str, Any]]) -> str:
    qc = [r.get("qc") or {} for r in rows if r.get("qc")]
    n = len(qc) or len(rows)
    power = sum(1 for q in qc if (q.get("statistical_power") or {}).get("status", "").startswith("approx"))
    dose = sum(1 for q in qc if (q.get("dose_validity") or {}).get("status") not in (None, "unknown"))
    stats = sum(1 for q in qc if (q.get("statistical_information") or {}).get("status") == "reported")
    sec = sum(1 for r in rows if r.get("evidence_type") == "review_meta")
    return (f"Quality checks could NOT be confirmed for every claim: statistics reported for {stats}/{n} studies, statistical "
            f"power assessable for {power}/{n}, dose reported/assessable for {dose}/{n}"
            + (f"; {sec} source(s) are reviews reporting other studies' experiments" if sec else "")
            + " (see the QC checklist).")


def _etiology_support(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    risk, causal, other = [], [], []
    for r in rows:
        for f in r.get("findings") or []:
            role = f.get("evidence_role") or "direct"
            if role == "disease_risk":
                (causal if f.get("inference_type") in ("randomised_controlled", "controlled_experiment") else risk).append((r, f))
            else:
                other.append((r, f))
    return {"risk": risk, "causal": causal, "other": other}


def _evidence_of_absence(rows: List[Dict[str, Any]]) -> bool:
    """Studies that TESTED the intervention and found no effect, at a direct level (L1/L2)."""
    return any(f.get("direction") == "no_effect" and (f.get("level") in (1, 2) or r.get("level") in (1, 2))
               for r in rows for f in (r.get("findings") or []))


def _cure_supported(rows: List[Dict[str, Any]]) -> bool:
    from clinical_context import meets_cure
    return any(meets_cure((r.get("best_outcome") or {}).get("level"), r.get("follow_up_days")) for r in rows)


def absent_type_conflict(sent: str, am: "re.Match", found_rows: List[Dict[str, Any]]) -> Tuple[List[str], List[Dict[str, Any]]]:
    """(kinds, rows that contradict) for a 'no <type> evidence' phrase. Reviews never count; an endpoint-specific
    absence ('no human study measured proliferation', 'absence of clinical trial findings measuring Ki-67') is
    contradicted only by a study of that type that measured that endpoint."""
    kinds = [k for k in am.groups() if k and k.lower() not in ("studies", "study", "evidence", "data", "trial", "trials")]
    present, wants = [], set()
    for k in kinds:
        want = ABSENT_MAP.get(k.lower().replace("-", " "), set())
        wants |= want
        present += [r for r in found_rows if _row_types(r) & want and r not in present]
    present = [r for r in present if r.get("evidence_type") != "review_meta"]
    if wants & {"human_clinical", "human_observational"}:
        # a veterinary case report (a buffalo) is a case report but not HUMAN evidence
        present = [r for r in present if r.get("bucket") == "human" or (r.get("bucket") is None and r.get("evidence_type") != "case_report")
                   or any(str(f.get("model_type") or "").startswith("human") for f in r.get("findings") or [])]
    # the endpoint of THIS absence: the clause it stands in ('no human study measured proliferation; one trial found no
    # survival benefit' is about proliferation only)
    clause = re.split(r";|,\s*(?:but|while|whereas|although)\b|\b(?:whereas|but|although)\b", sent[am.start():], maxsplit=1)[0]
    e_fams = set(_families_in(clause))
    if e_fams:
        if e_fams & {"proliferation", "tumour_growth"}:
            e_fams |= {"proliferation", "tumour_growth"}
        present = [r for r in present if any((f.get("model_type") or r.get("evidence_type")) in wants
                                             and f.get("endpoint") in e_fams for f in r.get("findings") or [])]
    return kinds, present


DOSE_CLAIM = re.compile(r"\b(?:doses?|dosing|dosages?|concentrations?|exposures?)\b[^.;]{0,60}?\b(?:exceed\w*|higher than|"
                        r"greater than|above|beyond|supra-?physiolog\w*|unrealistic\w*|not (?:clinically )?achievable|far from)\b|"
                        r"\bsupra-?physiolog\w*\b", re.I)


def dose_statement_check(sent: str, rows: List[Dict[str, Any]]) -> Optional[Dict[str, str]]:
    """'(Mouse) doses exceed human exposure' is checked against the dose assessments of the studies read: when no dose
    was assessed it is not checkable, when the assessed doses are within human exposure it contradicts the report."""
    assessed = [(r["ref"], r.get("dose_relevance") or {}) for r in rows
                if (r.get("dose_relevance") or {}).get("category") in ("physiological", "borderline", "supra-physiological")]
    supra = [ref for ref, d in assessed if d.get("category") == "supra-physiological"]
    if supra:
        return None
    if not assessed:
        return {"rule": "dose statement not checkable: no study in the sample reports a dose that could be compared with "
                        "human exposure", "replacement": None}
    lines = "; ".join(f"{ref}: {d.get('category')}" for ref, d in assessed[:6])
    return {"rule": "dose statement contradicts the dose check of the studies read (no assessed dose is above human exposure)",
            "replacement": f"The doses that could be checked are within or near human exposure ({lines}) "
                           f"[{', '.join(ref for ref, _ in assessed[:6])}]."}


_SPECIES = {"rats": r"\brats?\b", "mice": r"\b(?:mice|mouse|murine)\b", "dogs": r"\b(?:dogs?|canine)\b",
            "pigs": r"\b(?:pigs?|porcine|swine)\b", "zebrafish": r"\bzebrafish\b", "monkeys": r"\b(?:monkeys?|macaques?|primates?)\b"}


def species_not_in_sample(sent: str, rows: List[Dict[str, Any]]) -> Optional[str]:
    """The species a sentence names (rats, mice ...) when NO study with a verified finding is in that species."""
    clean = re.sub(r"\[[^\]]*\]", " ", sent)
    named = [sp for sp, rx in _SPECIES.items() if re.search(rx, clean, re.I)]
    if not named or not rows:
        return None
    txt = " ".join(f"{r.get('title') or ''} {r.get('model') or ''} {r.get('species') or ''} {r.get('organism') or ''} "
                   + " ".join(f"{f.get('quote') or ''} {f.get('statement') or ''}" for f in r.get("findings") or [])
                   for r in rows)
    for sp in named:
        if not re.search(_SPECIES[sp], txt, re.I) and not (sp == "mice" and re.search(r"mus musculus", txt, re.I)) \
                and not (sp == "rats" and re.search(r"rattus", txt, re.I)):
            return sp
    return None


def guard_text(text: str, rows_by_ref: Dict[str, Dict[str, Any]], focus: str, field: str = "",
               ctx: Optional[Dict[str, Any]] = None) -> Tuple[str, List[str]]:
    """HARD rules, sentence by sentence (points 3, 5, 6, 36, 37). A violating sentence is never published:
    it is rewritten into the defensible statement (absence of evidence, cure not established) or withheld.
    Returns (text, notes); withheld originals are appended to ctx['blocked'] for the audit trail."""
    if not isinstance(text, str) or not text:
        return text, []
    from deep_reasoner import normalize_citations
    text = re.sub("[\u2010\u2011\u2012\u2013]", "-", normalize_citations(text))   # 'human‑derived' (unicode hyphen)       # '(S8)', '[S1.E1.F1]', '([S5.E2.F1], [S9])' are citations too
    ctx = ctx if ctx is not None else {}
    all_rows = ctx.get("rows") or list(rows_by_ref.values())
    any_human = any(r.get("bucket") == "human" and r.get("n_findings", 1) for r in all_rows)
    disease = (ctx.get("disease_terms") or ["the condition"])[0]
    out, notes = [], []
    for sent in _sentences(text):
        if not re.search(r"[A-Za-z]{3}", re.sub(r"\[[^\]]*\]", " ", sent)):
            continue                     # an empty fragment / a bare citation is not a statement: dropped, never 'withheld'
        refs = [r for r in _cited(sent) if r in rows_by_ref]
        core = _SAMPLE_PHRASE.sub(" ", sent)          # 'in the retrieved sample' is not a limitation statement
        if not refs and field in SUBSTANTIVE_FIELDS + ("conflicts_explained",) + BUCKET_FIELDS and EFFICACY.search(sent) \
                and not LIMITATION.search(core) and not ABSENCE.search(sent):
            # PROVE before withholding: an uncited summary sentence is checked against the verified findings; when they
            # support it, the citations are added (and a wrong 'most' becomes the exact count) - only a sentence nothing
            # supports is withheld as untraceable
            proof = prove_sentence(sent, all_rows)
            if proof["refs"]:
                sent = _append_note(proof["text"], f"[{', '.join(proof['refs'])}]")
                ctx.setdefault("proved", []).append({"field": field or "summary", "refs": proof["refs"],
                                                      "how": proof["how"]})
                notes.append(f"{field or 'summary'}: citations added by the evidence check ({proof['how']})")
                refs = proof["refs"]
        bucket_unproved = None
        if not refs and field in BUCKET_FIELDS and EFFICACY.search(sent) and not LIMITATION.search(_SAMPLE_PHRASE.sub(" ", sent)) \
                and not ABSENCE.search(sent) and not ABSENCE_SENT.search(sent):
            bucket_unproved = True          # an uncited bucket summary nothing proves: replaced from the verified findings
        rows = [rows_by_ref[r] for r in refs]
        scope_rows = rows or all_rows
        tags, replacement = [], None
        if bucket_unproved:
            done = ctx.setdefault("bucket_summaries", set())
            tags.append(f"unsupported summary: no verified {field.replace('_evidence', '').replace('_', ' ')} finding states this")
            if field not in done:
                done.add(field)
                replacement = bucket_summary(all_rows, field)
        if refs:
            # citation mismatch: the sentence is about a different disease, which the cited study never mentions
            # ('Benzene causes leukemia [S2]' where S2 is a metformin / breast-cancer cell paper)
            from evidence_identity import DISEASES, match_disease
            d_sent = match_disease(_POP_DESCRIPTOR.sub(" ", sent))   # 'non-diabetic women' is a population, not the topic
            q_dis = (ctx.get("frame") or {}).get("disease")
            if d_sent and d_sent != q_dis and d_sent in DISEASES:
                rx = DISEASES[d_sent]
                row_txt = " ".join(f"{r.get('title') or ''} {r.get('study') or ''} {r.get('quote') or ''} {r.get('key_finding') or ''} "
                                   + " ".join(f"{f.get('statement') or ''} {f.get('quote') or ''}" for f in (r.get("findings") or []))
                                   for r in rows)
                if not (rx["rx_words"].search(row_txt) or rx["rx_acr"].search(row_txt)):
                    tags.append(f"citation mismatch: the sentence is about {d_sent}, which {', '.join(refs)} never mention(s)")
            human_rows = [r for r in rows if r.get("bucket") == "human"]
            # 'animal models using human breast cancer cells', 'human-derived MCF-7 cells' describe a MODEL, not people
            model_words = re.sub(r"\bhuman(?:[- ]derived)?\s+(?:\w+\s+){0,4}?(?:cells?|cell lines?|xenografts?|organoids?|"
                                 r"models?|systems?|cultures?|lines?|tumou?r (?:cells|lines|xenografts)|tissue explants?)\b|\b(?:in|using)\s+(?:\w+\s+){0,2}"
                                 r"(?:animal|mouse|murine|rat|xenograft|cell[- ]culture|in vitro)\s+(?:models?|systems?)[^,.;]*",
                                 " ", sent, flags=re.I)
            review_rows = [r for r in rows if r.get("evidence_type") == "review_meta"]
            self_labelled = bool(review_rows) and len(review_rows) == len(rows) and \
                re.search(r"\b(?:reviews?|secondary|meta-?analys\w*|reported by)\b", sent, re.I)
            explanatory = re.search(r"\b(?:differences? (?:in|between)|divergen\w*|discrepan\w*|explained by|explain\w* (?:the|this)|"
                                    r"reflects? (?:the )?differences?|in vitro (?:\w+ )?(?:vs\.?|versus)|whole (?:human )?organism)\b",
                                    sent, re.I) and not re.search(r"\b(?:in|among) (?:humans?|patients|people|women|men)\b[^.;]{0,40}"
                                                                  r"\b(?:reduc|increas|decreas|inhibit|improv|lower)\w*", sent, re.I)
            if (field == "human_evidence" or HUMAN_SENT.search(model_words)) and focus in ("human", "general") and not human_rows \
                    and not LIMITATION.search(sent) and not self_labelled and not explanatory \
                    and (field == "human_evidence" or EFFICACY.search(model_words)):
                kinds = sorted({TYPE_NAME.get(r.get("evidence_type"), r.get("evidence_type") or "?") for r in rows})
                tags.append(f"model leap: stated for humans but supported only by {', '.join(kinds)} evidence")
            if not ABSENCE.search(sent) and not LIMITATION.search(sent):
                rel = lambda f_: {f_} | ({"proliferation", "tumour_growth"} if f_ in ("proliferation", "tumour_growth") else set())
                missing, fams, measured_all = [], [], set()
                for clause, crefs in _cited_clauses(sent):
                    crows = [rows_by_ref[r_] for r_ in crefs if r_ in rows_by_ref] or rows
                    measured = {f.get("endpoint") for r in crows for f in (r.get("findings") or [])}
                    measured_all |= measured
                    cf = _families_in(clause)
                    fams += [f_ for f_ in cf if f_ not in fams]
                    missing += [f_ for f_ in cf if measured and not (measured & rel(f_)) and f_ not in missing]
                measured = measured_all
                inferred = re.search(r"\b(?:indicat\w*|suggest\w*|impl(?:y|ies|ying)|therefore|thus|hence|which means|"
                                     r"meaning|reflect\w*|consistent with|implies)\b", sent, re.I)
                if missing and (len(missing) == len(fams) or inferred):      # one endpoint INFERRED from another
                    tags.append(f"endpoint leap: cited studies measured {', '.join(sorted(FAMILY_LABEL.get(m, str(m).replace('other:', '')) for m in measured if m))} "
                                f"- not {', '.join(FAMILY_LABEL.get(f_, str(f_).replace('other:', '')) for f_ in missing)}")
                elif missing:
                    # a list where most endpoints ARE supported: keep the sentence, flag the unsupported part
                    sent = _append_note(sent, f"[note: {', '.join(FAMILY_LABEL.get(f_, str(f_).replace('other:', '')) for f_ in missing)} not measured in the cited studies]")
        elif focus in ("human", "general") and HUMAN_SENT.search(sent) and EFFICACY.search(sent) and not any_human \
                and not LIMITATION.search(sent):
            tags.append("human efficacy statement with no human evidence in the deep-read sample")
        # a short-term marker (Ki-67) presented as a patient outcome: keep the sentence, say what it does not show
        if refs and SURROGATE_LEAP.search(sent) and "[note: Ki-67" not in sent \
                and not LIMITATION.search(re.sub(r"\[(?:note|scope):[^\]]*\]", " ", sent)) \
                and not any(f.get("endpoint") in ("patient_survival", "recurrence_incidence") and f.get("direction") == "increase"
                            for r in rows for f in r.get("findings") or []):
            sent = _append_note(sent, "[note: Ki-67 / proliferation is a short-term marker; these studies did not show that its "
                                      "change improves survival or recurrence]")
            notes.append("surrogate scope: short-term marker not presented as a patient outcome")
        # separate subgroup analyses merged into one population ('HER2-positive women with insulin resistance (HOMA > 2.8)'
        # when the paper reports HOMA > 2.8 and HER2-positive as two separate subgroups): annotate, do not withhold
        if refs:
            merged = subgroup_merge(sent, rows)
            if merged:
                sent = _append_note(sent, merged)
                notes.append("subgroup scope: " + merged)
        # absence of evidence must never become evidence of absence (points 5, 6)
        if NEG_CONCL.search(sent) and not _evidence_of_absence(scope_rows):
            tags.append("absence of evidence presented as evidence of absence")
            replacement = (f"No evidence demonstrating that {disease} is curable was identified in the deep-read sample; "
                           f"this is absence of evidence, not evidence that it cannot be cured." if ctx.get("asks_cure") else
                           "No verified evidence of benefit was identified in the deep-read sample; this is absence of evidence, "
                           "not evidence of no effect.")
        # recovery / improvement is not cure (points 4, 27, 28)
        m = CURE_POS.search(sent)
        if m and not CURE_NEGATED.search(sent[:m.start()]) and not NEG_CONCL.search(sent) and not _cure_supported(scope_rows):
            best = max(scope_rows, key=lambda r: ((r.get("best_outcome") or {}).get("level") or -1)) if scope_rows else {}
            bo = (best.get("best_outcome") or {}).get("label", "not stated")
            fu = best.get("follow_up_days")
            tags.append("cure claimed but no study reports sustained remission with adequate follow-up")
            replacement = (f"No study in the deep-read sample reports an outcome meeting the cure definition; the strongest reported "
                           f"outcome is '{bo}'" + (f" after {fu / 30.44:.1f} months" if fu else "") +
                           (f" [{best['ref']}]" if best.get("ref") else "") + ".")
        frame = ctx.get("frame") or {}
        dname = frame.get("disease") or disease
        limitation = bool(LIMITATION.search(sent))
        found_rows = _rows_with_findings(all_rows)
        # a cited study with NO verified finding cannot be characterised in the synthesis (leukemia S8)
        empty = [r["ref"] for r in rows if not (r.get("n_findings") or r.get("findings"))]
        if empty and not limitation:
            tags.append(f"cites {', '.join(empty)}, which has no verified finding - its result cannot be characterised")
        # evidence-type characterisation must match the verified evidence ('computational data hint ...')
        if not ABSENT_TYPE.search(sent) and not ABSENCE_SENT.search(sent):
            # 'animal models using MCF-7 cells' names two systems: it is mischaracterised only if NONE of them is cited
            named = [t for t, rx in TYPE_WORDS if re.search(rx, sent, re.I)]
            pool = [r for r in (rows if refs else found_rows) if (r.get("n_findings") or r.get("findings"))]
            if named and not any(any(_row_types(r) & TYPE_GROUP[t] for r in pool) for t in named):
                t = named[0]
                tags.append(f"evidence-type mischaracterised: the sentence describes {t.replace('_', ' ')} evidence but no "
                            f"verified {t.replace('_', ' ')} finding {'is cited' if refs else 'exists in the sample'}")
            elif named:
                # 'X in humans; larger effects are seen only in animals [S1-S3]': a SEPARATE clause about a system with no
                # verified finding anywhere in the sample is unsupported, even when another clause is cited correctly
                for clause in re.split(r";|\s+(?:while|whereas|but)\s+|,\s+(?:and\s+)?(?=\w+ (?:studies|data|models?|effects?)\b)",
                                       re.sub(r"\[[^\]]*\]", " ", sent)):
                    cn = [t for t, rx in TYPE_WORDS if re.search(rx, clause, re.I)]
                    if len(cn) == 1 and EFFICACY.search(clause) and not ABSENT_TYPE.search(clause) \
                            and not ABSENCE_SENT.search(clause) and not LIMITATION.search(clause) \
                            and not any(_row_types(r) & TYPE_GROUP[cn[0]] for r in found_rows):
                        tags.append(f"evidence-type mischaracterised: '{clause.strip()[:80]}' describes "
                                    f"{cn[0].replace('_', ' ')} evidence but no verified {cn[0].replace('_', ' ')} finding exists in "
                                    f"the sample")
                        break
        # clause by clause: 'trial data [S1, S2] suggest ..., while animal xenograft studies [S4] show ...' - each clause's
        # citations must be of the kind of system that clause names (S4 is a human trial, not an animal study)
        if refs and not any(t.startswith("evidence-type mischaracterised") for t in tags):
            for clause, crefs in _cited_clauses(sent):
                cn = [t for t, rx in TYPE_WORDS if re.search(rx, re.sub(r"\[[^\]]*\]", " ", clause), re.I)]
                crows = [rows_by_ref[r_] for r_ in crefs if r_ in rows_by_ref]
                if len(cn) == 1 and crows and len(_cited_clauses(sent)) > 1 and \
                        not any(_row_types(r) & TYPE_GROUP[cn[0]] for r in crows):
                    tags.append(f"evidence-type mischaracterised: the clause '{clause.strip()[:70]}' describes "
                                f"{cn[0].replace('_', ' ')} evidence but cites {', '.join(crefs)} "
                                f"({', '.join(sorted({TYPE_NAME.get(r.get('evidence_type'), r.get('evidence_type') or '?') for r in crows}))})")
                    break
        # an uncited explanation of a disagreement is the AI's interpretation: published, but labelled as such
        if field == "conflicts_explained" and not refs and not tags and "[AI interpretation" not in sent:
            sent = _append_note(sent, "[AI interpretation - not checked against the papers]")
        # a statement about doses must agree with the report's own dose check ('Mouse doses exceed human exposure' while
        # the only assessed animal dose is within the human range)
        if DOSE_CLAIM.search(sent) and not LIMITATION.search(sent):
            verdict = dose_statement_check(sent, all_rows)
            if verdict:
                tags.append(verdict["rule"])
                replacement = verdict["replacement"]
        # a statement about one species needs a study in that species ('Null rat study ...' with no rat study read)
        sp = species_not_in_sample(sent, found_rows)
        if sp:
            tags.append(f"no verified finding in {sp} in the sample - the sentence describes a study that was not read")
        # 'no human / epidemiological studies' while such studies are in the sample (internal contradiction)
        am = ABSENT_TYPE.search(sent)
        # 'no human study directly measured proliferation' is an endpoint-specific absence, not 'no human studies'
        if am and re.search(r"\b(?:study|studies|trials?|data|evidence)\s+(?:\w+\s+)?(?:directly\s+)?(?:measured|assessed|reported|"
                            r"examined|evaluated|tested|addressed)\b", sent[am.start():], re.I):
            am = None
        if am:
            kinds, present = absent_type_conflict(sent, am, found_rows)
            if present:
                hs = {"human_observational": sum(1 for r in present if r.get("evidence_type") == "human_observational"),
                      "human_clinical": sum(1 for r in present if r.get("evidence_type") == "human_clinical")}
                tags.append(f"internal contradiction: says there is no {'/'.join(kinds)} evidence, but the sample contains "
                            f"{', '.join(r['ref'] for r in present)}")
                replacement = (f"The deep-read sample does contain {'/'.join(kinds)} evidence: "
                               + "; ".join(f"{r['ref']} ({TYPE_NAME.get(r.get('evidence_type'), r.get('evidence_type'))})" for r in present)
                               + (f" (human: {hs.get('human_observational', 0)} observational/epidemiological, "
                                  f"{hs.get('human_clinical', 0)} clinical)" if hs and ({'human', 'epidemiological', 'clinical', 'observational'}
                                                                                        & {k.lower() for k in kinds}) else "") + ".")
        # causes questions: a mechanism or an association is not a demonstrated cause
        if frame.get("intent") == "etiology" and CAUSE_STMT.search(sent) and not CAUSE_NEGATED.search(sent):
            sup = _etiology_support(rows if refs else all_rows)
            risk_ok = "risk factor" in sent.lower() and not re.search(r"\bcaus", sent, re.I)
            if not sup["causal"] and not (risk_ok and sup["risk"]):
                cref = refs or sorted({r["ref"] for r, _ in sup["risk"] + sup["other"]})[:6]
                if sup["risk"]:
                    ex = sorted({(f.get("intervention") or "the exposure") for _, f in sup["risk"]})
                    tags.append("association presented as causation")
                    replacement = (f"Within the deep-read sample, {', '.join(ex)} {'was' if len(ex) == 1 else 'were'} associated with "
                                   f"{dname} occurrence or risk" + (f" [{', '.join(sorted({r['ref'] for r, _ in sup['risk']}))}]" if sup['risk'] else "")
                                   + "; this is association, not demonstrated causation.")
                elif not sup["other"]:
                    tags.append("cause asserted, but the cited studies have no verified finding on it")
                    replacement = ("The cited studies" + (f" [{', '.join(cref)}]" if cref else "") + f" contain no verified finding "
                                   f"that establishes a cause of {dname}.")
                else:
                    kinds = sorted({(f.get("evidence_role") or "direct").replace("_", " ") for _, f in sup["other"]})
                    tags.append("mechanism / intermediate effect presented as a cause of the disease")
                    replacement = (f"The deep-read sample does not establish causes of {dname}: the cited evidence is "
                                   f"{', '.join(kinds)} evidence" + (f" [{', '.join(cref)}]" if cref else "") +
                                   f", which shows biological mechanisms or intermediate effects - not that any factor causes {dname}.")
        # exposure -> intermediate -> disease chains need every link
        if dname and CHAIN.search(sent) and re.search(re.escape(dname.split()[0][:6]), sent, re.I) and not CAUSE_NEGATED.search(sent) \
                and not _QUALITY_CHAIN.search(sent):
            sup = _etiology_support(rows if refs else all_rows)
            if not sup["risk"] and not sup["causal"]:
                head = CHAIN.split(sent, maxsplit=1)[0].rstrip(" ,;")
                tags.append(f"causal chain: the final link to {dname} is not shown by the cited evidence")
                replacement = replacement or (f"{head} [{', '.join(refs)}]" if refs and "[" not in head else head) + \
                    f"; whether this leads to {dname} was not shown in the deep-read sample."
        # blanket QC statements are never published (they were false in earlier dossiers)
        if QC_CLAIM.search(sent):
            tags.append("unsupported quality-control statement")
            replacement = _qc_summary(all_rows)
        # every substantive conclusion must be traceable to a study (point 36)
        if field in SUBSTANTIVE_FIELDS and not refs and not tags and EFFICACY.search(sent) and not LIMITATION.search(sent) \
                and not ABSENCE.search(sent) and not re.search(r"\b(?:cannot be (?:confirmed|established|determined|concluded)|"
                                                               r"due to the absence|absence of|lack of|not (?:yet )?(?:been )?"
                                                               r"(?:confirmed|established|demonstrated))\b", sent, re.I) \
                and not re.search(r"\bSTRING(?:-db)?\b", sent):      # database-sourced statements cite STRING, not a study
            tags.append("untraceable: substantive statement without a cited study")
        if tags and not replacement and field in BUCKET_FIELDS:
            # a withheld human / animal / cell summary is replaced by what the verified findings of that kind show (when
            # there are any) - the reader gets the evidence, not an empty slot
            done = ctx.setdefault("bucket_summaries", set())
            want = _BUCKET_MODELS.get(field, ())
            if field not in done and any((f.get("model_type") or r.get("evidence_type")) in want
                                         for r in all_rows for f in r.get("findings") or []):
                done.add(field)
                replacement = bucket_summary(all_rows, field)
        if tags:
            new = (replacement + " [rewritten by evidence rules]") if replacement else f"⛔ [statement withheld - {'; '.join(tags)}]"
            ctx.setdefault("blocked", []).append({"field": field or "summary", "original": sent, "rules": tags,
                                                  "replacement": replacement})
            notes.extend(f"{field or 'summary'}: {t}" for t in tags)
            sent = new
        elif frame.get("heterogeneous") and refs and (EFFICACY.search(sent) or CAUSE_STMT.search(sent)) and not limitation:
            # heterogeneous disease (leukemia = AML/ALL/CML/CLL ...): say which subtype the evidence is about
            subs = sorted({x for r in rows for x in (r.get("disease_subtypes") or []) if x != "not specified"})
            unspecified = list(dict.fromkeys(r["ref"] for r in rows
                                             if not [x for x in (r.get("disease_subtypes") or []) if x != "not specified"]))
            scope = (f"evidence from {', '.join(subs)} only" if subs and not unspecified else
                     f"subtype not specified in {', '.join(unspecified)}" + (f"; {', '.join(subs)} in the others" if subs else ""))
            if "[scope:" not in sent:
                sent = _append_note(sent, f"[scope: {scope} - not {dname} in general]")
                ctx.setdefault("annotations", []).append({"field": field or "summary", "note": scope})
        out.append(sent)
    return " ".join(out), notes


def guard_synthesis(data: Dict[str, Any], rows: List[Dict[str, Any]], focus: str,
                    ctx: Optional[Dict[str, Any]] = None) -> List[str]:
    rows_by_ref = {r["ref"]: r for r in rows}
    ctx = dict(ctx or {})
    ctx["rows"] = rows
    ctx.setdefault("blocked", [])
    notes: List[str] = []
    for key in ("executive_summary", "bottom_line", "human_evidence", "animal_evidence", "in_vitro_evidence",
                "plant_evidence", "microbial_evidence", "translational_gap", "certainty_rationale"):
        if isinstance(data.get(key), str):
            data[key] = soften_overstatement(data[key])
            data[key], n = guard_text(data[key], rows_by_ref, focus, key, ctx)
            notes += n
    for key in ("conflicts_explained", "key_uncertainties"):
        if isinstance(data.get(key), list):
            new = []
            for item in data[key]:
                t, n = guard_text(soften_overstatement(item) if isinstance(item, str) else item, rows_by_ref, focus, key, ctx)
                new.append(t)
                notes += n
            data[key] = new
    # a mechanism step may only cite a study with a verified finding ('PRKAA1 inhibits MTOR [S4]' where S4 has none)
    if isinstance(data.get("mechanistic_chain"), list):
        fixed = []
        for step in data["mechanistic_chain"]:
            if not isinstance(step, str):
                fixed.append(step)
                continue
            from deep_reasoner import normalize_citations
            step = normalize_citations(step)
            bad = [r for r in _cited(step) if r not in rows_by_ref or not (rows_by_ref[r].get("n_findings")
                                                                           or rows_by_ref[r].get("findings"))]
            # a cited study must also MENTION the step's genes / proteins in its verified findings: 'PRKAA1 inhibits MTOR
            # [S4]' where S4's only finding is a Ki-67 result does not support the step
            genes = {_SYN.get(g.upper(), g.upper()) for g in _GENE_TOKEN.findall(re.sub(r"\[[^\]]*\]|\(STRING[^)]*\)", " ", step))
                     if g.upper() not in _NOT_NODE}
            for r in _cited(step):
                if r in bad or r not in rows_by_ref or not genes:
                    continue
                row = rows_by_ref[r]
                txt = " ".join(f"{f.get('statement') or ''} {f.get('quote') or ''} {f.get('target') or ''}"
                               for f in row.get("findings") or []) + f" {row.get('quote') or ''} {row.get('key_finding') or ''}"
                found = {_SYN.get(g.upper(), g.upper()) for g in _GENE_TOKEN.findall(txt)}
                if not genes & found:
                    bad.append(r)
            if bad:
                for r in bad:
                    step = re.sub(rf"\s*\[{r}\]|(?<=[\[,])\s*{r}\s*,|,\s*{r}(?=\])", "", step)
                step = re.sub(r"\s*\[\s*\]", "", re.sub(r"\[\s+", "[", step)).rstrip()
                step = _append_note(step, f"[citation to {', '.join(bad)} removed: no verified finding of {', '.join(bad)} "
                                          f"concerns this step]")
                notes.append(f"mechanistic_chain: removed citation(s) to {', '.join(bad)} (no verified finding)")
            fixed.append(step)
        data["mechanistic_chain"] = fixed
    data["blocked_statements"] = ctx["blocked"]
    data["scope_annotations"] = ctx.get("annotations", [])
    return notes


# --------------------------------------------------------------------------- #12 mechanism levels
MECH_LABEL = {"M1": "M1 candidate (database association only)", "M2": "M2 experimental association",
              "M3": "M3 causal evidence (perturbation) in a model system", "M4": "M4 causal evidence in the target organism",
              "L7": "L7 AI hypothesis (no supporting study)"}


# synonyms so 'AMPK' in the text and 'PRKAA1' in STRING are one node
_SYN = {"AMPK": "PRKAA1", "AMPKA1": "PRKAA1", "PRKAA1": "PRKAA1", "HER2": "ERBB2", "ERBB2": "ERBB2", "HER-2": "ERBB2",
        "RB": "RB1", "RB1": "RB1", "PRB": "RB1", "P53": "TP53", "TP53": "TP53", "MTORC1": "MTOR", "MTOR": "MTOR",
        "EGFR": "EGFR", "ERBB1": "EGFR", "LKB1": "STK11", "STK11": "STK11", "NFKB": "NFKB1", "NF-KB": "NFKB1",
        "AKT": "AKT1", "AKT1": "AKT1", "KI67": "MKI67", "KI-67": "MKI67", "MKI67": "MKI67", "C-MYC": "MYC", "MYC": "MYC",
        "P21": "CDKN1A", "CDKN1A": "CDKN1A", "MTORC2": "MTOR"}
_PROCESS = re.compile(r"\b(cell[- ]cycle(?: arrest)?|proliferation|tumou?r growth|apoptosis|viability|cell death|"
                      r"migration|invasion|metastasis|senescence|autophagy|glycolysis|oxidative phosphorylation)\b", re.I)
_GENE_TOKEN = re.compile(r"\b(?:Rb|pRb|p\d{2}|mTOR(?:C[12])?|Akt\d?|Ki-?67|c-Myc|[A-Z][A-Z0-9]{1,8}(?:-[A-Z0-9]{1,4})?)\b")
_NOT_NODE = {"STRING", "AI", "L7", "M1", "M2", "M3", "M4", "DNA", "RNA", "ATP", "HR", "OR", "CI", "ER", "PR", "TNBC", "MCF",
             "SKBR3", "EMT6", "T47D", "MDA", "MB", "MCF-7", "AND", "THE"}
_PERTURB_NEAR = r"(?:knock-?down|knock-?out|si-?RNA|sh-?RNA|CRISPR|inhibitor|inhibition|blockade|deletion|deficien\w*|silenc\w*|-/-)"
_ACTS = r"(?:inhibit|reduc|decreas|suppress|block|attenuat|lower|activat|increas|induc|enhanc|stimulat|promot|abolish|impair)\w*"
_ORDER = {"L7": 0, "M1": 1, "M2": 2, "M3": 3, "M4": 4}


def _node(tok: str) -> str:
    t = tok.upper().replace(" ", "")
    return _SYN.get(t, t)


def _nodes(text: str, agents: List[str]) -> List[tuple]:
    """Entities of a mechanism step in reading order: (position, node, kind) with kind agent / gene / process."""
    out = []
    for a in agents:
        for m in re.finditer(rf"\b{re.escape(a)}\w*", text, re.I):
            out.append((m.start(), a.lower(), "agent"))
    for m in _GENE_TOKEN.finditer(text):
        if m.group(0).upper() not in _NOT_NODE and not re.fullmatch(r"[SC]\d+|L\d|M\d", m.group(0)):
            out.append((m.start(), _node(m.group(0)), "gene"))
    for m in _PROCESS.finditer(text):
        out.append((m.start(), re.sub(r"[- ]", " ", m.group(0).lower()).replace("tumor", "tumour").replace(" arrest", ""), "process"))
    out.sort()
    seen, uniq = set(), []
    for pos, n, k in out:
        if n not in seen:
            seen.add(n)
            uniq.append((pos, n, k))
    return uniq


def _sentences_of(rows: List[Dict[str, Any]]) -> List[tuple]:
    """(row, sentence, model) - the model is the system of the FINDING the sentence comes from (a cell assay inside a
    paper that also has a mouse experiment is cell culture), the paper's type only for its own summary text."""
    sents = []
    for r in rows:
        parts = [(f"{r.get('quote') or ''} {r.get('key_finding') or ''}", r.get("evidence_type"))] + \
                [(f"{f.get('quote') or ''} {f.get('statement') or ''}", f.get("model_type") or r.get("evidence_type"))
                 for f in (r.get("findings") or [])]
        for blob, model in parts:
            for s_ in re.split(r"(?<=[.;!?])\s+", blob):
                if s_.strip():
                    sents.append((r, s_, model))
    return sents


def _mentions(node: str, kind: str, text: str) -> Optional[re.Match]:
    if kind == "process":
        words = node.split()
        stem = re.escape(words[0][:7]) + (r"\w*[- ]" + re.escape(words[1][:4]) if len(words) > 1 else "")
        return re.search(rf"\b{stem}", text, re.I)
    names = [node] + [k for k, v in _SYN.items() if v == node]
    for n in names:
        m = re.search(rf"(?<![A-Za-z0-9]){re.escape(n)}(?![A-Za-z])", text, re.I)
        if m:
            return m
    return None


# read-out markers and subtype-defining receptors: never a mechanism node linked by co-mention
_MARKER_NODES = {"MKI67", "PCNA", "proliferation", "apoptosis", "TUNEL", "ERBB2", "ESR1", "PGR"}


def _edge(a: tuple, b: tuple, sents: List[tuple], pairs: set) -> Dict[str, Any]:
    """Evidence for ONE edge A -> B. M2: a verified passage names A acting on B; M3/M4: A was perturbed in that
    passage (M4 in people); agent edges inside a combination regimen are never credited to the agent alone."""
    best = {"from": a[1], "to": b[1], "level": "L7", "basis": "no verified passage links these two"}
    for r, sent, model in sents:
        ma, mb = _mentions(a[1], a[2], sent), _mentions(b[1], b[2], sent)
        mname = TYPE_NAME.get(model, (model or "?").replace("_", " "))
        if not (ma and mb):
            continue
        if a[2] == "agent":
            if re.search(r"\bcombin\w*|\bin combination\b|\bplus\b|\btogether with\b", sent, re.I):
                cand = {"level": "L7", "basis": f"{r.get('ref')}: only a combination regimen was tested - not {a[1]} alone"}
            elif re.search(rf"{re.escape(a[1][:6])}\w*[^.;]{{0,80}}{_ACTS}", sent, re.I):
                cand = {"level": "M2", "basis": f"{r.get('ref')} ({mname}): "
                                                 f"{a[1]} changed {b[1]} in a verified passage (association within the experiment)"}
            else:
                continue
        else:
            a_rx = re.escape(sent[ma.start():ma.end()])
            # the perturbation must target A itself ('CDK4 knockdown', 'a CDK4/6 inhibitor', 'inhibition of CDK4')
            if re.search(rf"{a_rx}(?:/\w+)?[- ]?{_PERTURB_NEAR}|{_PERTURB_NEAR} of (?:the )?{a_rx}", sent, re.I):
                lvl = "M4" if model in ("human_clinical", "human_observational") else "M3"
                cand = {"level": lvl, "basis": f"{r.get('ref')} ({mname}): {a[1]} was "
                                                f"perturbed and {b[1]} changed"}
            elif {a[1], b[1]} & _MARKER_NODES:
                # Ki-67 IS the proliferation read-out; HER2 / ER name the tumour subtype: naming them in the same sentence
                # as another node is not a step of a mechanism
                continue
            elif re.search(r"\b(?:increas|decreas|reduc|induc|inhibit|suppress|activat|elevat|lower|upregulat|downregulat|"
                           r"enhanc|attenuat|block|restor)\w*", sent, re.I):
                cand = {"level": "M2", "basis": f"{r.get('ref')} ({mname}): "
                                                 f"{a[1]} and {b[1]} changed together in a verified passage (co-change, not cause)"}
            else:
                continue                     # merely named together: no evidence for an arrow
        if _ORDER[cand["level"]] > _ORDER[best["level"]] or best["basis"].startswith("no verified"):
            best.update(cand)
    if best["level"] == "L7" and frozenset((a[1], b[1])) in pairs:
        best.update({"level": "M1", "basis": "STRING-db association only - the proteins can interact; not a causal link"})
    return best


def label_mechanism(steps: List[Any], rows: List[Dict[str, Any]], string_edges: Optional[List[Dict[str, Any]]] = None,
                    agents: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    """Every arrow of a proposed pathway is judged ON ITS OWN (causal evidence graph). A step is only as strong as its
    weakest arrow: 'CDK4 inhibition -> lower Rb phosphorylation' can be M3 in a model while 'metformin -> CDK4' is
    unverified - the step is then shown as unverified with the M3 arrow listed inside it."""
    rows_by_ref = {r["ref"]: r for r in rows}
    pairs = {frozenset((_node(e.get("a", "")), _node(e.get("b", "")))) for e in (string_edges or [])}
    agents = [a for a in (agents or []) if a] or sorted({(f.get("intervention") or "").split()[0].lower()
                                                         for r in rows for f in (r.get("findings") or [])
                                                         if (f.get("intervention") or "").split()
                                                         and not (f.get("intervention") or "")[0].isupper()})
    out = []
    for step in steps or []:
        text = step if isinstance(step, str) else json.dumps(step)
        refs = [r for r in _cited(text) if r in rows_by_ref]
        cited = [rows_by_ref[r] for r in refs]
        nodes = _nodes(text, agents)
        agent_nodes = [n for n in nodes if n[2] == "agent"]
        rest = [n for n in nodes if n[2] != "agent"]
        chain = (agent_nodes[:1] + rest) if agent_nodes else rest
        sents = _sentences_of(cited)
        edges = [_edge(chain[i], chain[i + 1], sents, pairs) for i in range(len(chain) - 1)] if (sents or not refs) else []
        if not edges and refs:
            # one entity only (or the cited studies carry no passages): the cited study's own design decides
            perturbed = [r for r in refs if rows_by_ref[r].get("perturbation")]
            if perturbed:
                direct = [r for r in perturbed if rows_by_ref[r].get("level") in (1, 2)]
                code = "M4" if direct else "M3"
                basis = ", ".join(f"{r} ({TYPE_NAME.get(rows_by_ref[r].get('evidence_type'), '?')}: "
                                  f"{', '.join(rows_by_ref[r]['perturbation'][:2])})" for r in (direct or perturbed))
            else:
                code = "M2"
                basis = ", ".join(f"{r} ({TYPE_NAME.get(rows_by_ref[r].get('evidence_type'), '?')})" for r in refs)
            out.append({"step": text, "level": code, "label": MECH_LABEL[code], "basis": basis, "edges": []})
            continue
        if edges:
            weakest = min(edges, key=lambda e: _ORDER[e["level"]])
            code = weakest["level"]
            strong = [e for e in edges if _ORDER[e["level"]] >= 3]
            basis = "; ".join(f"{e['from']} → {e['to']}: {e['level']} ({e['basis']})" for e in edges)
            if strong and _ORDER[code] < 2:
                basis = (f"part of this step has causal evidence in a model ({', '.join(e['from'] + ' → ' + e['to'] for e in strong)}), "
                         f"but other links are unverified - " + basis)
        elif "STRING" in text.upper() or any(len(p & {n[1] for n in nodes}) == 2 for p in pairs):
            code, basis = "M1", "STRING-db association - indicates physical/functional association, not a causal cascade"
        elif refs:
            code, basis = "M2", ", ".join(f"{r} ({TYPE_NAME.get(rows_by_ref[r].get('evidence_type'), '?')})" for r in refs)
        else:
            code, basis = "L7", "no verified study or database support"
        out.append({"step": text, "level": code, "label": MECH_LABEL[code], "basis": basis, "edges": edges})
    return out


def chain_summary(levels: List[Dict[str, Any]], agents: Optional[List[str]] = None) -> str:
    """One honest sentence about the whole proposed pathway."""
    edges = [e for x in levels for e in (x.get("edges") or [])]
    if not edges:
        return ""
    agent_set = {a.lower() for a in (agents or [])}
    agent_edges = [e for e in edges if e["from"] in agent_set]
    causal = [e for e in edges if _ORDER[e["level"]] >= 3]
    weak_agent = [e for e in agent_edges if _ORDER[e["level"]] < 2]
    if causal and weak_agent:
        return (f"The downstream part of this pathway has causal evidence in model systems "
                f"({', '.join(e['from'] + ' → ' + e['to'] + ' ' + e['level'] for e in causal[:3])}), but the link between "
                f"{weak_agent[0]['from']} and the upstream molecular events ({', '.join(e['to'] for e in weak_agent[:3])}) "
                f"is not verified. The pathway as a whole is a hypothesis.")
    if weak_agent:
        return (f"The link between {weak_agent[0]['from']} and {', '.join(e['to'] for e in weak_agent[:3])} is not verified "
                f"in the papers read; the pathway as a whole is a hypothesis.")
    if all(_ORDER[e["level"]] >= 2 for e in edges):
        return "Every link of this pathway is named in verified passages; associations (M2) still need perturbation tests."
    return "Some links of this pathway are not verified; the pathway as a whole is a hypothesis."


# words that overstate a sample ('consistently', 'all studies') - the synthesis only covers the deep-read papers
_OVERSTATE = [(re.compile(r"\bconsistently\b", re.I), "generally (in the deep-read studies)"),
              # 'consistent anti-growth effects' (not 'consistent with'): most, not all, studies agree
              (re.compile(r"\b(?:highly |very |remarkably )?consistent\s+((?:[\w\-]+\s+){0,2}?(?:effects?|reductions?|decreases?|"
                          r"increases?|results?|findings?|evidence|inhibition|suppression|benefits?))\b", re.I),
               r"\1 (in most of the deep-read studies)"),
              (re.compile(r"\buniformly\b|\binvariably\b", re.I), "generally (in the deep-read studies)"),
              (re.compile(r"\b(?:all|every) (?:the )?(studies|papers|experiments|models)\b", re.I), r"the deep-read \1 here"),
              (re.compile(r"\bclearly (?:shows?|demonstrates?|establishes?)\b", re.I), "reports"),
              (re.compile(r"\bproves?\b", re.I), "suggests")]


def soften_overstatement(text: str) -> str:
    for rx, rep in _OVERSTATE:
        text = rx.sub(rep, text)
    return text


# --------------------------------------------------------------------------- #13 hypotheses
_ENTITY_WORDS = r"insulin|igf-?1r?|ampk|mtor\w*|lkb1|stk11|foxa1|hmgb1|rage|mmp-?\d+|emt|er\+|er-positive|tnbc|triple-negative|" \
                r"her2|p53|tp53|nf-?kb|stat3|akt|pi3k|hif-?1|vegf|complex i|oxphos|glycolysis|warburg|inflammation|" \
                r"apoptosis|autophagy|senescence|immune|t cells?|macrophages?|microbiome|estrogen|oestrogen"
_NOT_ENTITY = {"DNA", "RNA", "ATP", "AND", "THE", "WITH", "NOT", "FOR", "MCF", "REQUIRED", "NOT", "ONLY", "ALL", "AI",
               "AIM", "NOTE", "YES", "NO", "HUMAN", "CELL", "CELLS", "MRNA", "RCT"}


_TRANSLOC = re.compile(r"\b[tT]\(\s*\d{1,2}\s*;\s*\d{1,2}\s*\)|\b(?:inv|del)\(\d{1,2}\)|\b[A-Z][A-Z0-9]{1,6}[-:/][A-Z][A-Z0-9]{1,6}\b")


def hypothesis_entities(h: Dict[str, Any]) -> set:
    text = f"{h.get('statement') or ''} {h.get('mechanism') or ''}"
    ents = {e.lower().replace("-", "") for e in re.findall(_ENTITY_WORDS, text, re.I)}
    ents |= {re.sub(r"\s+", "", t.lower()).replace("-", "").replace(":", "").replace("/", "") for t in _TRANSLOC.findall(text)}
    ents |= {s.lower().replace("-", "") for s in re.findall(r"\b[A-Z][A-Z0-9]{2,9}(?:-\d+)?\b", text)
             if s not in _NOT_ENTITY and not re.fullmatch(r"[CSHML]\d+", s)}     # C12 / S4 / H1 / M3 are references
    if {"insulin", "igf1"} & ents:
        ents.add("insulin/igf1")
    return ents


def _similar(a: Dict[str, Any], b: Dict[str, Any]) -> bool:
    # hypotheses derived from DIFFERENT verified claims about different interventions are different hypotheses,
    # however similar their template wording ('X increases ... and replicates in an independent study')
    da, db_ = set(_declared(a)), set(_declared(b))
    if da and db_ and not (da & db_):
        return False
    ea, eb = hypothesis_entities(a), hypothesis_entities(b)
    if ea and eb:
        inter = len(ea & eb)
        if inter / len(ea | eb) >= 0.5 or (inter >= 2 and inter / min(len(ea), len(eb)) >= 0.6):
            return True
    ta = set(re.findall(r"[a-z]{4,}", (a.get("statement") or "").lower()))
    tb = set(re.findall(r"[a-z]{4,}", (b.get("statement") or "").lower()))
    return bool(ta and tb) and len(ta & tb) / min(len(ta), len(tb)) >= 0.7


def _label(h: Dict[str, Any]) -> str:
    # an evidence-derived replication hypothesis is labelled by what it tests, not as a 'mechanism'
    st = h.get("statement") or ""
    m = re.match(r"(.+?) (?:increases|decreases|does not change|changes) (.+?) \((.+?); observed in (C\d+)", st)
    if m:
        step = re.search(r"in an independent (.+?) study", st)
        return (f"{m.group(1)} → {m.group(2)} " + (f"— test in {step.group(1)}" if step else
                                                    f"({m.group(3).split(',')[-1].strip()}, {m.group(4)})"))
    m = re.match(r"(.+?) (?:reduces|increases|decreases|changes|affects) (.+?) through (.+?) \(", st)
    if m:
        return f"{m.group(1)} → {m.group(3)} → {m.group(2)}"
    text = f"{h.get('statement')} {h.get('mechanism')}"
    found = [(m.start(), m.group(0)) for m in re.finditer(_ENTITY_WORDS, text, re.I)]
    found += [(m.start(), m.group(0)) for m in re.finditer(r"\b[A-Z][A-Z0-9]{2,9}(?:-\d+)?\b", text)]
    ents = [e for _, e in sorted(found) if e.upper() not in _NOT_ENTITY and not re.fullmatch(r"[CS]\d+(?:\.\w+)*", e)]
    seen, keep = set(), []
    for e in ents:
        k = e.lower().replace("-", "")
        if k not in seen:
            seen.add(k)
            keep.append(e if e.isupper() else e.lower())
    if keep:
        return "/".join(keep[:3]) + " mechanism"
    words = re.findall(r"[A-Za-z0-9\-]+", h.get("statement") or "")
    return " ".join(words[:6]) + (" …" if len(words) > 6 else "") if words else "unnamed hypothesis"


def _declared(h: Dict[str, Any]) -> List[int]:
    se = h.get("supporting_evidence")
    raw = (se.get("derived_from") if isinstance(se, dict) else None) or h.get("derived_from") or []
    return [int(m.group(1)) for v in raw for m in CLAIM_ID_RE.finditer(str(v))] if raw else []


def hypothesis_support(h: Dict[str, Any], evidence: List[Dict[str, Any]]) -> Dict[str, Any]:
    """evidence: verified claims with 'level' and 'ref' (paper label). One paper counts once (its best level).
    Declared derivations (claim IDs) are used when present; otherwise related claims are matched by entity."""
    ents = hypothesis_entities(h) - {"inflammation", "apoptosis"}
    declared = set(_declared(h))
    per_paper: Dict[Any, Dict[str, Any]] = {}
    for c in evidence:
        text = f"{c.get('intervention') or ''} {c.get('measurement') or ''} {c.get('statement') or ''} {c.get('quote') or ''}".lower().replace("-", "")
        if declared:
            if c.get("id") not in declared:
                continue
        elif not ents or not any(e in text for e in ents):
            continue
        key = c.get("paper_id")
        lvl = c.get("level")
        if lvl and (key not in per_paper or lvl < per_paper[key]["level"]):
            per_paper[key] = {"level": lvl, "ref": c.get("ref") or f"paper {key}", "claim_id": c.get("id")}
    score = sum(LEVEL_WEIGHT.get(v["level"], 0) for v in per_paper.values())
    return {"score": score, "n_papers": len(per_paper), "basis": "declared derivation" if declared else "entity match",
            "supporting": sorted(({"ref": v["ref"], "level": f"L{v['level']}", "claim_id": v["claim_id"]}
                                  for v in per_paper.values()), key=lambda x: x["level"])}


_NULL_HYP = re.compile(r"\b(?:does|do|did|will) not (?:significantly |meaningfully )?(?:change|affect|alter|improve|reduce|"
                       r"increase|decrease|lower|influence)|\bno (?:significant |clinically meaningful |meaningful )?"
                       r"(?:effect|change|difference|benefit|improvement)\b|\bnull result holds\b|\bhas no effect\b", re.I)
_NO_DIFF = re.compile(r"\b(?:finds?|shows?|reports?|observes?|detects?)\s+(?:no|non-?significant)\s+(?:\w+\s+){0,2}?"
                      r"(?:difference|change|effect|benefit|association)|\bno (?:significant )?(?:difference|change|effect)\b", re.I)
_POWERED = re.compile(r"\b(?:adequately|sufficiently|well)[- ]powered\b|\bminimal (?:important|clinically important) "
                      r"(?:effect|difference)\b|\bconfidence interval\b", re.I)


def fix_falsification(h: Dict[str, Any]) -> Dict[str, Any]:
    """A falsification criterion must be able to refute the hypothesis.
    - 'X does not change survival' is refuted by a SIGNIFICANT CHANGE - 'another study finds no difference' supports it.
    - 'X reduces Ki-67' is refuted by a null result only if the study was adequately powered for a pre-specified minimal
      important effect and its confidence interval excludes that effect ('no difference' alone proves nothing)."""
    stmt = str(h.get("statement") or "")
    fals = h.get("falsification_conditions")
    if isinstance(fals, str):
        fals = [fals]
    fals = [str(x) for x in (fals or []) if x]
    m = re.search(r"\b(?:change|affect|alter|improve|reduce|increase|decrease|lower|influence|effect on|difference in|"
                  r"benefit on)\s+([\w\- ]{3,40}?)(?=\s*(?:\(|,|;|\.|$| and | in | with ))", stmt, re.I)
    endpoint = (m.group(1).strip() if m else "the outcome")
    null_hyp = bool(_NULL_HYP.search(stmt)) and not re.search(r"\b(?:reduces|increases|decreases|lowers|improves|"
                                                             r"inhibits|suppresses|promotes)\b", stmt.split(" and ")[0], re.I)
    new = []
    if null_hyp:
        new = [x for x in fals if not _NO_DIFF.search(x)]
        if len(new) < len(fals) or not new:
            new.append(f"An adequately powered study finds a statistically significant change in {endpoint} (a confidence "
                       f"interval that excludes no effect) - finding 'no difference' would SUPPORT this hypothesis, not "
                       f"falsify it.")
    else:
        for x in fals:
            if _NO_DIFF.search(x) and not _POWERED.search(x):
                x = (x.rstrip(". ") + " - in a study adequately powered for a pre-specified minimal important effect, with a "
                     "confidence interval that excludes that effect (a small or underpowered null result is inconclusive).")
            new.append(x)
    if new != fals:
        h["falsification_conditions"] = new
        h["falsification_fixed"] = True
    return h


def consolidate_hypotheses(hyps: List[Dict[str, Any]], evidence: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Merge overlapping hypotheses, then rank by evidence support (weighted by level)."""
    clusters: List[List[Dict[str, Any]]] = []
    for h in hyps:
        for cl in clusters:
            if any(_similar(h, x) for x in cl):
                cl.append(h)
                break
        else:
            clusters.append([h])
    merged = []
    for cl in clusters:
        supports = [(hypothesis_support(h, evidence), h) for h in cl]
        sup, rep = max(supports, key=lambda x: (x[0]["score"], len(x[1].get("statement") or "")))
        out = dict(rep)
        for k in ("predictions", "falsification_conditions"):
            vals, seen = [], set()
            for h in cl:
                for v in (h.get(k) or []):
                    key = re.sub(r"\W+", " ", str(v).lower()).strip()
                    if key and key not in seen:
                        seen.add(key)
                        vals.append(v)
            out[k] = vals
        out["merged_from"] = [h.get("statement") for h in cl if h is not rep]
        out["support"] = sup
        out["label"] = _label(rep)
        merged.append(out)
    merged.sort(key=lambda h: (-h["support"]["score"], -h["support"]["n_papers"]))
    by_id = {c.get("id"): c for c in evidence}
    src = sourced_numbers(evidence)
    for h in merged:
        fix_falsification(h)     # every hypothesis, AI-written ones included
    for h in merged:      # also label hypotheses stored before labelling existed (idempotent)
        for k in ("statement", "mechanism", "predictions", "falsification_conditions", "next_experiment"):
            v = h.get(k)
            if isinstance(v, str) and v.strip().startswith("{"):
                try:
                    h[k] = json.dumps(label_ai_parameters(json.loads(v), src), ensure_ascii=False)
                    continue
                except ValueError:
                    pass
            if v:
                h[k] = label_ai_parameters(v, src)
    for i, h in enumerate(merged, 1):
        h["rank"] = i
        h["code"] = f"H{i}"
        sup = h["support"]
        if not sup["score"]:
            h["evidence_status"] = "AI hypothesis (L7) - no verified supporting finding"
        elif sup.get("basis") == "declared derivation":
            h["evidence_status"] = "derived from verified claims"
        else:
            h["evidence_status"] = "related verified evidence (entity match - derivation not declared)"
        h["derived_from"] = [f"C{x}" for x in _declared(h)] or [f"C{s['claim_id']}" for s in sup["supporting"]]
        h["derivation"] = f"H{i} ← " + (" + ".join(f"{s['ref']} claim C{s['claim_id']} ({s['level']})" for s in sup["supporting"])
                                         or "no verified claim (AI hypothesis)")
        # mechanism status is always recomputed from the CURRENT evidence (stored labels may be from an older rule set)
        h["mechanism_status"] = mechanism_status(
            {"mechanism": f"{h.get('statement') or ''} {h.get('mechanism') or ''}",
             "derived_from": [int(x[1:]) for x in h["derived_from"] if x[1:].isdigit()]}, by_id)
        if h["mechanism_status"].get("untraced") and sup["score"]:
            h["evidence_status"] = ("AI-generated mechanistic hypothesis informed by literature - derived from verified claims, but "
                                    "not traced: " + ", ".join(h["mechanism_status"]["untraced"][:6]))
        ders = [by_id[int(x[1:])] for x in h["derived_from"] if x[1:].isdigit() and int(x[1:]) in by_id]
        subs = sorted({s_ for c in ders for s_ in (((c.get("semantic_audit") or {}).get("identity") or {}).get("disease_subtype") or [])
                       if s_ != "not specified"})
        roles = sorted({(c.get("semantic_audit") or {}).get("evidence_role") or "direct" for c in ders})
        h["scope"] = {"subtypes": subs or ["not specified"], "evidence_roles": roles,
                      "note": ("derived only from " + "/".join(r.replace("_", " ") for r in roles) + " evidence - not a demonstrated cause")
                      if ders and not ({"disease_risk", "direct"} & set(roles)) else ""}
        h["ai_assumptions"] = ai_assumptions(h)
        # a mechanism counts as evidence-based only if a derived claim's own passage shows HOW (mediator evidence)
        h["_mech_ev"] = any((c.get("semantic_audit") or {}).get("mechanism_evidence") for c in ders)
        # what KIND of hypothesis this is (a replication test is not an explanation; an AI mechanism is not evidence)
        h["kind"], h["kind_label"] = hypothesis_kind(h)
        h["title"] = f"H{i} — {KIND_SHORT[h['kind']]}: {h['label']}"
    # replication tests first, then evidence-based mechanisms, then AI-proposed mechanisms; rank kept within each
    order = {"replication": 0, "mechanistic": 1, "network_candidate": 2, "mechanistic_ai": 3}
    merged.sort(key=lambda h: (order.get(h["kind"], 4), h["rank"]))
    for i, h in enumerate(merged, 1):
        h["rank"], h["code"] = i, f"H{i}"
        h["title"] = f"H{i} — {KIND_SHORT[h['kind']]}: {h['label']}"
        h["derivation"] = re.sub(r"^H\d+ ←", f"H{i} ←", h.get("derivation") or "")
    return merged


KIND_SHORT = {"replication": "Replication test", "mechanistic": "Mechanism (evidence-based)",
              "network_candidate": "Mechanism candidate (network)", "mechanistic_ai": "Mechanism (AI-proposed, unverified)"}
KIND_LABEL = {
    "replication": "Replication / translation hypothesis - asks whether a verified finding holds in another system or "
                   "population. It does not explain WHY the effect occurs; no mechanism is proposed.",
    "mechanistic": "Mechanistic hypothesis - every step is named in verified source passages (M2 association or M3 "
                   "perturbation). It proposes how the effect arises; the full chain is still to be tested.",
    "network_candidate": "Mechanism candidate - a verified effect plus a protein-network association (M1). The network "
                         "link shows the proteins can interact, not that the agent acts through them.",
    "mechanistic_ai": "Mechanistic hypothesis proposed by the AI - one or more steps are not in any verified source. "
                      "Not established; treat as an idea to test.",
}


def hypothesis_kind(h: Dict[str, Any]) -> tuple:
    if h.get("kind") in KIND_LABEL:                         # set by the generator (e.g. network candidates)
        return h["kind"], KIND_LABEL[h["kind"]]
    ms = (h.get("mechanism_status") or ((h.get("supporting_evidence") or {}) if isinstance(h.get("supporting_evidence"), dict)
                                         else {}).get("mechanism_status") or {}).get("status") or ""
    mech, stmt = str(h.get("mechanism") or ""), str(h.get("statement") or "")
    if ms.startswith("no mechanism proposed") or "Not specified - no mechanistic evidence" in mech:
        # a hypothesis read back from the database has no 'kind' / 'mechanism_status': its own text says what it is
        k = "replication"
    elif re.search(r"\bVerified: [^.]*\(C\d+", mech) and re.search(r";\s*M[23]: ", stmt):
        k = "mechanistic"                 # built from a verified mediator passage (M2 stated route / M3 perturbation)
    elif ms == "observed in source" and (h.get("support") or {}).get("score") and h.get("_mech_ev"):
        k = "mechanistic"
    else:
        k = "mechanistic_ai"
    return k, KIND_LABEL[k]


# --------------------------------------------------------------------------- #14 AI-proposed parameters
AI_TAG = "[AI-PROPOSED EXPERIMENTAL PARAMETER]"
_PARAM_RE = re.compile(
    r"(?<![\w.])(?:"
    r"n\s*(?:=|>=|≥|of)\s*\d+(?:\s*(?:per (?:group|arm)|mice|rats|animals|patients|participants|replicates))?|"
    r"\d+\s*(?:per (?:group|arm)|biological replicates|technical replicates|replicates|independent (?:experiments|blocks|replicates)|"
    r"mice|rats|animals|patients|participants|subjects|women|men)|"
    r"(?:sample size|power|alpha)\s*(?:of|=|:)?\s*\d+(?:\.\d+)?\s*%?|\d+(?:\.\d+)?\s*%\s*power|"
    r"\d+(?:\.\d+)?\s*(?:mg/kg(?:/day)?|mg|g/day|mg/day|µM|uM|μM|mM|nM|µg/mL|ug/mL|mg/mL|IU)|"
    r"(?:HOMA-?IR|HbA1c|BMI|Ki-?67|glucose|insulin|IC50|EC50|threshold|cut-?off|effect size|cohen'?s d|d)\s*(?:[<>]=?|≥|≤|=|of)\s*\d+(?:\.\d+)?\s*%?|"
    r"\d+(?:\.\d+)?\s*%\s*(?:reduction|increase|decrease|difference|change)|"
    r"(?:>=|<=|[<>≥≤])\s*\d+(?:\.\d+)?\s*%?(?:\s+[a-z]+)?|"
    r"\d+(?:\.\d+)?\s*(?:-|to)?\s*\d*\s*(?:hours|hrs?|h|days?|weeks?|wks?|months?|years?)\b"
    r")", re.I)


def label_ai_parameters(value: Any, sourced: Optional[Dict[str, str]] = None) -> Any:
    """Recursively tag numeric experimental parameters. A parameter whose number+unit appears
    in verified evidence is tagged with that source instead. Idempotent; p-value conventions are left alone."""
    sourced = sourced or {}
    if isinstance(value, dict):
        return {k: label_ai_parameters(v, sourced) for k, v in value.items()}
    if isinstance(value, list):
        return [label_ai_parameters(v, sourced) for v in value]
    if not isinstance(value, str) or AI_TAG in value or "[source:" in value:
        return value

    def tag(m):
        raw = m.group(0)
        if re.search(r"\bp\s*$", m.string[max(0, m.start() - 4):m.start()], re.I):
            return raw                      # 'p < 0.05' is a statistical convention, not a design assumption
        key = re.sub(r"\s+", "", raw.lower())
        src = sourced.get(key)
        if src and re.search(rf"\b{re.escape(src.split()[-1])}\b", m.string):
            return raw                      # the sentence already cites that claim ('observed in C191')
        tail = " " if raw != raw.rstrip() else ""             # 'alpha = 0.05 [TAG] to detect', never '0.05  [TAG]to'
        return f"{raw.rstrip()} [source: {src}]{tail}" if src else f"{raw.rstrip()} {AI_TAG}{tail}"

    return _PARAM_RE.sub(tag, value)


def sourced_numbers(verified_claims: List[Dict[str, Any]]) -> Dict[str, str]:
    """number+unit strings that occur in verified quotes -> 'verified claim C<id>'."""
    out: Dict[str, str] = {}
    for c in verified_claims:
        for m in _PARAM_RE.finditer(c.get("quote") or ""):
            out.setdefault(re.sub(r"\s+", "", m.group(0).lower()), f"verified claim C{c.get('id')}")
    return out


# --------------------------------------------------------------------------- #15 traceability
def build_traceability(synthesis: Dict[str, Any], matrix: Dict[str, Any], claims_by_id: Dict[int, Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Conclusion -> study -> experiment/finding -> claim -> exact source passage, for every cited sentence."""
    rows = {r["ref"]: r for r in matrix.get("rows") or []}
    trace = []
    seen_sent: set = set()
    first_listed: Dict[str, str] = {}          # study ref -> the T# whose entry first listed its findings
    listed_ids: Dict[str, set] = {}            # study ref -> findings already shown (each finding is shown once)
    for field in ("bottom_line", "executive_summary", "human_evidence", "animal_evidence", "in_vitro_evidence",
                  "plant_evidence", "microbial_evidence"):
        for sent in _sentences(synthesis.get(field) or ""):
            refs = list(dict.fromkeys(r for r in _cited(sent) if r in rows))
            key = re.sub(r"\W+", " ", sent).strip().lower()
            if not refs or key in seen_sent:
                continue                       # the same sentence in two summary fields is traced once
            seen_sent.add(key)
            t_id = f"T{len(trace) + 1}"
            # every endpoint family the sentence names, read-outs included ('METTL3 mRNA expression', 'histone methylation')
            fams = {f_ for f_, rx_, _ in FAMILY_RX if rx_.search(sent)}
            links = []
            for ref in refs:
                r = rows[ref]
                fl = r.get("findings") or []
                rel = {"proliferation", "tumour_growth", "viability"}
                wanted = fams | (rel if fams & rel else set())
                if wanted and any(f.get("endpoint") in wanted for f in fl):
                    fl = [f for f in fl if f.get("endpoint") in wanted]     # only the findings this sentence is about
                fid = lambda f: f"{f.get('experiment')}.{f.get('finding') or f.get('id')}|{f.get('endpoint')}|{f.get('direction')}"
                new_fl = [f for f in fl if fid(f) not in listed_ids.get(ref, set())]
                if ref in first_listed and (not fams or (fl and not new_fl)):
                    # a sentence naming no endpoint, or whose findings were all shown already, points to the earlier entry
                    links.append({"ref": ref, "paper": r.get("title"), "identifier": r.get("identifier"), "link": r.get("link"),
                                  "evidence_type": r.get("evidence_type_label"), "level": r.get("level_label"),
                                  "findings": [], "see": first_listed[ref]})
                    continue
                first_listed.setdefault(ref, t_id)
                listed_ids.setdefault(ref, set()).update(fid(f) for f in new_fl)
                fl = new_fl
                fs = []
                for f in fl:
                    cl = []
                    for cid in f.get("claim_ids") or []:
                        c = claims_by_id.get(cid) or {}
                        cl.append({"claim_id": f"C{cid}", "statement": c.get("statement"), "quote": c.get("quote"),
                                   "section": c.get("source_section"), "grounding": c.get("grounding_score"),
                                   "checks": {k: v.get("ok") for k, v in ((c.get("semantic_audit") or {}).get("checks") or {}).items()}})
                    fs.append({"finding": f.get("id"), "experiment": f"{ref}.{f.get('experiment')}",
                               "model": f"{(f.get('model_type') or '').replace('_', ' ')} {f.get('model_detail') or ''}".strip(),
                               "endpoint": f.get("endpoint"), "direction": f.get("direction"), "claims": cl})
                links.append({"ref": ref, "paper": r.get("title"), "identifier": r.get("identifier"), "link": r.get("link"),
                              "evidence_type": r.get("evidence_type_label"), "level": r.get("level_label"), "findings": fs})
            trace.append({"field": field, "statement": sent, "sources": links})
    return trace


def traceability_markdown(trace: List[Dict[str, Any]]) -> str:
    if not trace:
        return "_No cited statements to trace._\n"
    md = []
    for i, t in enumerate(trace, 1):
        md.append(f"**T{i}. Conclusion ({t['field'].replace('_', ' ')}):** {t['statement']}\n")
        for s in t["sources"]:
            md.append(f"- **{s['ref']}** → paper: [{s['paper']}]({s['link']}) · {s['identifier']} · {s['evidence_type']} · {s['level']}"
                      if s.get("link") else f"- **{s['ref']}** → paper: {s['paper']} · {s['identifier']} · {s['evidence_type']} · {s['level']}")
            if s.get("see"):
                md.append(f"  - _findings and source passages listed under {s['see']}_")
                continue
            if not s["findings"]:
                md.append("  - _no verified finding for this study_")
            for f in s["findings"]:
                md.append(f"  - experiment {f['experiment']} ({f['model'] or 'model not stated'}) → finding {f['finding']}: "
                          f"{(f['endpoint'] or '').replace('_', ' ')} {f['direction'] or ''}")
                for c in f["claims"]:
                    md.append(f"    - claim {c['claim_id']}: {c['statement']}")
                    md.append(f"      - source passage ({c.get('section') or 'section unknown'}, grounding {c.get('grounding')}): "
                              f"“{(c.get('quote') or '').strip()}”")
        md.append("")
    return "\n".join(md) + "\n"


# --------------------------------------------------------------------------- hypothesis / gap grounding (points 19-23, 35)
CLAIM_ID_RE = re.compile(r"\bC(\d+)\b")


def parse_derived(value: Any, valid_ids: set) -> List[int]:
    ids = []
    for v in (value if isinstance(value, list) else [value] if value else []):
        for m in CLAIM_ID_RE.finditer(str(v)):
            i = int(m.group(1))
            if i in valid_ids and i not in ids:
                ids.append(i)
    return ids


# --------------------------------------------------------------------------- one label per link, not per chain
# A hypothesis's mechanism is usually several steps: 'metformin -> mTOR -| proliferation; HER2+ cells exhibit
# elevated mTOR activity, making them more responsive'. The first step was observed in a source; the second is an
# assertion no source states. One label for the whole chain let the unverified step inherit the verified one's
# standing. Each step is now judged and labelled on its own.
_EDGE_SPLIT = re.compile(r";|\s+(?:and thereby|thereby|which in turn|leading to|resulting in|making them|so that)\s+|"
                         r",\s+(?=(?:which|because|since|as)\b)")
_REL_WORDS = re.compile(r"->|-\||\u2192|\u22a3|\b(?:inhibit\w*|activat\w*|increas\w*|decreas\w*|reduc\w*|suppress\w*|"
                        r"elevat\w*|higher|lower|greater|more responsive|less responsive|upregulat\w*|downregulat\w*|"
                        r"phosphorylat\w*|block\w*|drive\w*|promot\w*)\b", re.I)


def mechanism_edges(mechanism: str) -> List[str]:
    """The individual steps of a mechanism statement, in order."""
    txt = re.sub(r"^\s*(?:observed in source|proposed|hypothesi[sz]ed)\s*:?\s*", "", str(mechanism or ""), flags=re.I)
    parts = [p.strip(" .,;") for p in _EDGE_SPLIT.split(txt) if p and p.strip(" .,;")]
    return [p for p in parts if _REL_WORDS.search(p) or len(p.split()) >= 3]


def mechanism_edge_status(h: Dict[str, Any], claims_by_id: Dict[int, Dict[str, Any]]) -> List[Dict[str, Any]]:
    """One verdict per step: is this step itself stated by a cited source passage?

    A step counts as observed only when every entity it names appears in a cited passage AND that passage also
    states a relation between them. Entities alone are not enough: 'HER2', 'mTOR' and 'proliferation' can all appear
    in the sources without any source saying that HER2-positive cells have higher mTOR activity.
    """
    derived = [claims_by_id[i] for i in h.get("derived_from") or [] if i in claims_by_id]
    sents: List[str] = []
    for c in derived:
        for part in re.split(r"(?<=[.;])\s+", f"{c.get('quote') or ''} {c.get('source_context') or ''}"):
            if part.strip():
                sents.append(part.lower())
    out = []
    for edge in mechanism_edges(h.get("mechanism") or ""):
        ents = hypothesis_entities({"statement": "", "mechanism": edge}) - {"inflammation", "apoptosis"}
        if not ents:
            out.append({"step": edge, "status": "unverified", "why": "names no entity that could be checked"})
            continue
        hit = None
        for sent in sents:
            flat = sent.replace("-", "").replace(" ", "")
            if all(e.lower().replace("-", "").replace(" ", "") in flat for e in ents) and _REL_WORDS.search(sent):
                hit = sent
                break
        if hit:
            out.append({"step": edge, "status": "observed in source",
                        "why": "a cited passage states this relation between " + ", ".join(sorted(ents))})
        else:
            missing = sorted(e for e in ents
                             if e.lower().replace("-", "").replace(" ", "")
                             not in " ".join(sents).replace("-", "").replace(" ", ""))
            out.append({"step": edge, "status": "unverified",
                        "why": ("not stated by any cited passage; " + ("missing from the sources: "
                                + ", ".join(missing) if missing
                                else "the entities appear separately but no passage states this relation"))})
    return out


def mechanism_status(h: Dict[str, Any], claims_by_id: Dict[int, Dict[str, Any]]) -> Dict[str, str]:
    """Mechanism observed in a source vs inferred by AI vs proposed for future testing (point 21)."""
    derived = [claims_by_id[i] for i in h.get("derived_from") or [] if i in claims_by_id]
    if "Not specified - no mechanistic evidence" in (h.get("mechanism") or ""):
        # evidence-derived replication hypothesis: no mechanism is claimed at all
        return {"status": "no mechanism proposed - tests replication of a verified finding",
                "detail": ("derived from " + ", ".join(f"C{c['id']}" for c in derived)) if derived else "no mechanism claimed",
                "untraced": []}
    ents = hypothesis_entities({"statement": "", "mechanism": h.get("mechanism") or ""}) - {"inflammation", "apoptosis"}
    if not derived:
        return {"status": "proposed for future testing", "detail": "no verified claim supports this mechanism",
                "untraced": sorted(ents)}
    quotes = " ".join(f"{c.get('quote') or ''} {c.get('source_context') or ''}" for c in derived).lower()
    quotes = quotes.replace("-", "").replace(":", "").replace("/", "").replace(" ", "") + " " + quotes.replace("-", "")
    seen = [e for e in ents if e in quotes]
    missing = sorted(set(ents) - set(seen))
    if ents and not missing:
        # EVERY named entity must be in the source passages - one gene from the source does not make the rest observed
        edges = mechanism_edge_status(h, claims_by_id)
        weak = [e for e in edges if e["status"] != "observed in source"]
        if weak:
            # a chain is only as strong as its weakest step: one unverified step makes the chain a hypothesis
            return {"status": "AI-generated mechanistic hypothesis informed by literature",
                    "detail": f"{len(weak)} of {len(edges)} steps are not stated by any cited passage: "
                              + "; ".join(f"\u201c{w['step'][:60]}\u201d ({w['why']})" for w in weak[:2]),
                    "untraced": sorted(ents), "edges": edges}
        return {"status": "observed in source", "detail": f"every named entity appears in the source passages of "
                                                          f"{', '.join('C' + str(c['id']) for c in derived)}",
                "untraced": [], "edges": edges}
    return {"status": "AI-generated mechanistic hypothesis informed by literature",
            "detail": ("not traced to the cited source passages: " + ", ".join(missing)) if missing
            else "the mechanism names no entity that can be traced to the cited claims", "untraced": missing}


# the phrase in front of the tag: a decimal point inside a number is part of it ('alpha = 0.05', never just '05')
AI_PARAM_RE = re.compile(r"((?:[^;:,.()\[\]\\]|(?<=\d)\.(?=\d)){0,80}?\S)\s*\[AI-PROPOSED EXPERIMENTAL PARAMETER\]")


def ai_assumptions(h: Dict[str, Any]) -> List[str]:
    """Every AI-proposed number, pulled out so it is shown apart from evidence (point 19)."""
    blob = json.dumps({k: h.get(k) for k in ("statement", "mechanism", "predictions", "falsification_conditions", "next_experiment")},
                      ensure_ascii=False)
    out = []
    text = blob.replace('\\"', '"')
    for m in AI_PARAM_RE.finditer(text):
        s = m.group(1)
        if m.start(1) > 0 and text[m.start(1) - 1].isalnum():
            s = s.split(" ", 1)[1] if " " in s else ""          # never start inside a word ('t non-diabetic ...')
        s = " ".join(s.strip(' "{}').split()[-8:])               # the parameter and the few words that name it
        if s and re.search(r"\d", s) and s not in out:
            out.append(s)
    return out


PK_RE = re.compile(r"\b(pharmacokinetic\w*|\bPK\b|plasma (?:level|concentration)s?|bioavailability|absorption|half-life|clearance)\b", re.I)


GAP_CHAIN = re.compile(r"\b(?:leading to|leads? to|resulting in|results? in|causing|causes?|development of|progression to|"
                       r"subsequent development|risk of|onset of|transformation (?:to|into))\b", re.I)


def _identity_line(c: Dict[str, Any]) -> str:
    a = c.get("semantic_audit") or {}
    i = a.get("identity") or {}
    from semantic_audit import FAMILY_LABEL, canonical_outcome
    out = i.get('outcome') or c.get('measurement') or '?'
    fam = c.get("canonical_outcome") or a.get("canonical_outcome")
    if fam and not str(fam).startswith("other:") and canonical_outcome(out).get("family") != fam:
        out = f"{out} (= {FAMILY_LABEL.get(fam, fam)})"        # 'tumor growth in MCF-7 cells' is proliferation
    return (f"C{c['id']}: {i.get('intervention_exposure') or c.get('intervention') or '?'} -> "
            f"{out} {i.get('direction') or c.get('canonical_direction') or ''} in "
            f"{(i.get('claim_model') or c.get('claim_model') or '?').replace('_', ' ')} "
            f"({i.get('disease') or '?'}; subtype {', '.join(i.get('disease_subtype') or ['not specified'])}; "
            f"role {(a.get('evidence_role') or 'direct').replace('_', ' ')})").replace("  ", " ")


def _vague(c: Dict[str, Any]) -> Optional[str]:
    a = c.get("semantic_audit") or {}
    i = a.get("identity") or {}
    fam = a.get("canonical_outcome") or c.get("canonical_outcome") or ""
    if i.get("intervention_is_question_words"):
        return "no named exposure"
    if (fam.startswith("other") or fam in ("expression", "signalling")) and not a.get("canonical_target"):
        return f"outcome '{c.get('measurement') or fam}' names no gene/protein"
    return None


def validate_gaps(gaps: List[Dict[str, Any]], verified: List[Dict[str, Any]],
                  frame: Optional[Dict[str, Any]] = None) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Keep only defensible research gaps (points 22, 23, 35) - each anchored to specific verified claims,
    and never assuming an untested causal link (leukemia review: 'aberrations leading to leukemia')."""
    frame = frame or {}
    from clinical_context import normalize_intervention
    valid_ids = {c["id"] for c in verified}
    by_id = {c["id"]: c for c in verified}
    combos = [((c.get("semantic_audit") or {}).get("combination_components") or [], c) for c in verified]
    combos = [(cs, c) for cs, c in combos if cs]
    isolated = {(c.get("intervention") or "").lower() for c in verified if not (c.get("semantic_audit") or {}).get("combination_components")}
    profiles = [normalize_intervention(c.get("intervention") or "", f"{c.get('quote') or ''}") for c in verified]
    systemic = any(p["systemic_drug"] and not p["combination_components"] for p in profiles)
    kept, notes = [], []
    seen_keys: List[str] = []           # identical gaps (repeated by the model or across heal rounds) are listed once
    for g in gaps:
        g = dict(g)
        text = g.get("description") or ""
        derived = parse_derived(g.get("derived_from") or g.get("supporting_observation"), valid_ids)
        # a gap about one component of a regimen inherits the attribution problem -> rewrite at regimen level
        for comps, c in combos:
            hit = [x for x in comps if x.lower() in text.lower() and x.lower() not in isolated]
            if hit:
                outcome = (c.get("semantic_audit") or {}).get("outcome_label") or c.get("measurement") or "the outcome"
                g["description"] = (f"The treatment regimen combining {', '.join(comps)} lacks dose-response or component-isolation "
                                    f"evidence identifying which component, if any, contributed to the observed {outcome}.")
                g["rewritten_from"] = text
                derived = derived or [c["id"]]
                notes.append(f"gap rewritten at regimen level (component '{hit[0]}' was never tested alone)")
                break
        covered = _gap_already_covered(g["description"], verified)
        if covered:
            notes.append(f"gap removed: it says a system is missing, but verified findings from that system exist ({covered})")
            continue
        if PK_RE.search(g["description"]) and not systemic:
            kinds = sorted({p["type"] for p in profiles}) or ["unspecified"]
            notes.append(f"pharmacokinetic gap removed: the interventions are {', '.join(kinds)} "
                         f"(regimen/procedure/local), so PK is not the limiting evidence")
            continue
        g["derived_from"] = [f"C{i}" for i in derived]
        if derived:
            g["evidence_status"] = "derived from verified claims " + ", ".join(f"C{i}" for i in derived)
            g["supporting_observation"] = "; ".join(_identity_line(by_id[i]) for i in derived)
            vague = [(i, _vague(by_id[i])) for i in derived if _vague(by_id[i])]
            if vague and len(vague) == len(derived):
                g["anchoring"] = "weak"
                g["evidence_status"] = ("AI-proposed gap, weakly anchored - the supporting observation is non-specific ("
                                        + "; ".join(f"C{i}: {why}" for i, why in vague) + ")")
                notes.append(f"gap weakly anchored: {'; '.join(why for _, why in vague)}")
            dname = frame.get("disease")
            if dname and GAP_CHAIN.search(g["description"]) and re.search(re.escape(dname.split()[0][:6]), g["description"], re.I):
                roles = {(by_id[i].get("semantic_audit") or {}).get("evidence_role") or "direct" for i in derived}
                if "disease_risk" not in roles:
                    g["chain_note"] = (f"The supporting claim(s) show only {', '.join(sorted(r.replace('_', ' ') for r in roles))} "
                                       f"evidence; the link to {dname} is itself untested in the retrieved evidence - this gap is "
                                       f"about establishing that link, not a known step.")
                    g["evidence_status"] += " · first link only (disease link not shown)"
                    notes.append(f"gap assumes an untested link to {dname} - annotated")
        else:
            g["evidence_status"] = "AI-proposed gap (not derived from a verified claim)"
            g["category"] = "ai_proposed"
        key = re.sub(r"[^a-z0-9 ]", "", (g.get("description") or "").lower())
        key = " ".join(w for w in key.split() if len(w) > 2)
        if not key or any(key == k or _jaccard(key, k) >= 0.85 for k in seen_keys):
            notes.append("duplicate gap removed")
            continue
        seen_keys.append(key)
        kept.append(g)
    return kept, notes


_GAP_MODEL = [("animal", r"animal|in vivo|mouse|mice|rodent|xenograft"),
              ("cell_culture", r"cell[- ]culture|in vitro|cell lines?"),
              ("human_clinical", r"randomi[sz]ed|clinical trial|controlled (?:human|trial)"),
              ("human_observational", r"human observational|observational|cohort|epidemiolog\w*"),
              ("human", r"human|clinical|patients?")]
_GROWTH = {"proliferation", "tumour_growth", "viability"}


def _gap_already_covered(text: str, verified: List[Dict[str, Any]]) -> str:
    """'No animal (in vivo) evidence on metformin and proliferation' is false when a verified primary finding from an
    animal experiment on the same agent and endpoint exists in ANY deep-read paper (study-level coverage)."""
    m = re.search(r"\bno\s+(.{0,60}?)\s+(?:evidence|data|studies|study)\s+(?:on|of|for|about|linking|testing)\s+(.+?)"
                  r"(?:\s+in the deep-read sample|\.|$)", text or "", re.I)
    if not m:
        return ""
    which = m.group(1).lower()
    model = next((k for k, rx in _GAP_MODEL if re.search(rx, which)), None)
    if not model:
        return ""
    topic = m.group(2).lower()
    from semantic_audit import canonical_outcome
    fam = canonical_outcome(topic)["family"]
    fams = _GROWTH if fam in _GROWTH else {fam}
    hits = []
    for c in verified:
        if (c.get("semantic_audit") or {}).get("evidence_role") == "secondary_report":
            continue
        cm = c.get("claim_model") or ""
        same_model = cm == model or (model == "human" and cm.startswith("human")) or \
            (model == "animal" and cm == "experimental_disease_model") or (model == "cell_culture" and cm == "organoid_ex_vivo")
        agent = ((c.get("intervention") or "").lower().split() or [""])[0]
        if same_model and agent and agent in topic and (c.get("canonical_outcome") in fams):
            hits.append(f"C{c['id']}")
    return ", ".join(hits[:5])


def _jaccard(a: str, b: str) -> float:
    sa, sb = set(a.split()), set(b.split())
    return len(sa & sb) / max(1, len(sa | sb))


# --------------------------------------------------------------------------- gaps that never go missing
def evidence_gaps(matrix: Dict[str, Any], synthesis: Dict[str, Any], question: str = "") -> List[Dict[str, Any]]:
    """Research gaps derived deterministically from the evidence itself, so an empty AI gap list never produces 'No
    research gaps' while the report lists open questions. Each gap names its basis:
      - CONSISTENCY  human findings on the asked endpoint point different ways (named claims)
      - SURROGATE    the asked endpoint was measured only short-term / as a surrogate, with no clinical outcome in the
                     same population (or a clinical outcome without the asked endpoint)
      - DOSE         no included study reports the dose / exposure that produced its result
      - COVERAGE     evidence types that were found but not read (budget) or not found at all - a coverage gap, kept
                     apart from biological uncertainty"""
    from semantic_audit import FAMILY_LABEL
    rows = [r for r in (matrix or {}).get("rows") or [] if r.get("evidence_type") != "review_meta"]
    frame = (synthesis or {}).get("question_frame") or (synthesis or {}).get("frame") or {}
    want = set(frame.get("endpoints") or [])
    if not want and question:
        from evidence_identity import question_frame
        want = set(question_frame(question).get("endpoints") or [])
    if want & {"proliferation", "tumour_growth"}:
        want |= {"proliferation", "tumour_growth"}
    gaps: List[Dict[str, Any]] = []
    hf = [(r, f) for r in rows for f in r.get("findings") or []
          if str(f.get("model_type") or r.get("evidence_type") or "").startswith("human") and f.get("answers_question", True)]
    on_q = [(r, f) for r, f in hf if not want or f.get("endpoint") in want]
    dirs = {}
    for r, f in on_q:
        dirs.setdefault(f.get("direction"), []).append((r, f))
    label = " / ".join(sorted({FAMILY_LABEL.get(e, e) for e in want})) or "the question's outcome"
    if len([d for d in dirs if d in ("increase", "decrease", "no_effect")]) > 1:
        ids = sorted({c for _, f in on_q for c in f.get("claim_ids") or []})
        subs = sorted({s for r, _ in on_q for s in (r.get("disease_subtypes") or []) if s != "not specified"})
        gaps.append({"gap_type": "CONSISTENCY", "derived_from": [f"C{i}" for i in ids],
                     "description": (f"Human studies disagree on {label}: "
                                     + "; ".join(f"{k.replace('_', ' ')} in {', '.join(sorted({r['ref'] for r, _ in v}))}"
                                                 for k, v in dirs.items() if k) + ". Whether the difference is explained by "
                                     + (f"subtype ({', '.join(subs)}), " if subs else "") + "metabolic state, lesion stage or chance "
                                     "has not been tested with a prespecified interaction analysis."),
                     "evidence_status": "derived from verified claims (deterministic)",
                     "supporting_observation": "; ".join(f"{r['ref']}: {f.get('statement') or ''}"[:160] for r, f in on_q[:4])})
    if on_q:
        fu = [r.get("follow_up_days") for r, _ in on_q if r.get("follow_up_days")]
        clinical = [(r, f) for r, f in hf if f.get("endpoint") in ("patient_survival", "recurrence_incidence", "clinical_response")]
        if want and want <= {"proliferation", "tumour_growth"} and (not fu or max(fu) <= 120):
            gaps.append({"gap_type": "SURROGATE", "derived_from": [f"C{c}" for _, f in on_q for c in (f.get("claim_ids") or [])][:8],
                         "description": (f"{label.capitalize()} was measured only over a short window"
                                         + (f" (at most {max(fu) / 30.44:.1f} months)" if fu else "") + "; whether a change in this "
                                         "surrogate marker predicts recurrence or survival in the same patients is not shown by "
                                         "the studies read" + (" (the survival trials read did not measure it)" if clinical else "") + "."),
                         "evidence_status": "derived from verified claims (deterministic)"})
    if rows and not any((r.get("dose_relevance") or {}).get("category") not in (None, "not reported", "not applicable")
                        for r in rows):
        gaps.append({"gap_type": "DOSE", "derived_from": [],
                     "description": "No included study reports the dose or exposure in the text that was read, so dose-response "
                                    "and the relevance of the exposure to therapeutic use cannot be judged.",
                     "evidence_status": "derived from the dose check of every included study (deterministic)"})
    # the report's own 'Key uncertainties' become gaps when the verified evidence touches them (each with its claims);
    # an uncertainty no verified finding touches is reported as such, never silently dropped
    covered = {g["gap_type"] for g in gaps}
    for u in ((synthesis or {}).get("key_uncertainties") or [])[:6]:
        if not isinstance(u, str) or not u.strip():
            continue
        ufams = set(_families_in(u))
        terms = [t for t in ("her2", "er-positive", "triple", "subtype", "insulin", "homa", "diabet", "metabolic", "dose",
                             "duration", "long-term", "mechanis", "ampk", "mtor", "survival", "recurrence") if t in u.lower()]
        hit = [(r, f) for r in rows for f in r.get("findings") or []
               if (f.get("endpoint") in ufams) or (terms and any(t in str(r.get("disease_subtypes")).lower() or
                                                                t in str(f.get("statement")).lower() for t in terms))]
        if "dose" in terms and "DOSE" in covered or ("subtype" in terms or "her2" in terms) and "CONSISTENCY" in covered:
            continue                       # already stated by a deterministic gap
        gaps.append({"gap_type": "OPEN QUESTION", "derived_from": sorted({f"C{c}" for _, f in hit for c in (f.get("claim_ids") or [])})[:8],
                     "description": u.strip().rstrip(".") + ".",
                     "evidence_status": (f"from the report's key uncertainties; touched by verified findings in "
                                         f"{', '.join(sorted({r['ref'] for r, _ in hit}))}" if hit else
                                         "from the report's key uncertainties; no verified finding in this run addresses it")})
    av = ((synthesis or {}).get("evidence_availability") or {}).get("types") or {}
    unread = [v["label"] for v in av.values() if "NOT deep-read" in str(v.get("state")) or "NOT screened" in str(v.get("state"))]
    if unread:
        gaps.append({"gap_type": "COVERAGE", "derived_from": [],
                     "description": ("Evidence-coverage gap (not a biological unknown): "
                                     + ", ".join(unread) + " records were found but not read in this run, so their results are "
                                     "not part of this answer. A full run with a larger reading budget would close this gap."),
                     "evidence_status": "derived from the PRISMA ledger (deterministic)"})
    return gaps


# --------------------------------------------------------------------------- separate subgroups are not an intersection
_SG_STOP = {"those", "women", "men", "patients", "participants", "subjects", "with", "who", "in", "the", "of", "lowest", "highest",
            "top", "bottom", "low", "high", "tumors", "tumours", "quintile", "quartile", "tertile", "levels", "level", "having",
            "whose", "had", "free", "and", "a", "an", "group", "cases"}
_SG_SYNONYM = {"homa": r"homa|insulin[- ]resist\w*|hyperinsulin\w*", "her2-positive": r"her2[- ]?(?:positive|\+)|her2\b",
               "er-positive": r"\ber[- ]?(?:positive|\+)|oestrogen receptor|estrogen receptor", "bmi": r"\bbmi\b|obes\w*|overweight"}


def _sg_terms(item: str) -> List[str]:
    toks = [t for t in re.findall(r"[A-Za-z][\w\-]*", item) if t.lower() not in _SG_STOP and len(t) > 1]
    return [_SG_SYNONYM.get(t.lower(), re.escape(t.lower())) for t in toks]


def subgroup_merge(sent: str, rows: List[Dict[str, Any]]) -> Optional[str]:
    """A note when the sentence joins two subgroups that the cited study analysed separately, else None."""
    low = sent.lower()
    if re.search(r"\b(?:separately|respectively|each (?:subgroup|analysed|analyzed)|as separate|in separate)\b", low):
        return None
    for r in rows:
        for f in r.get("findings") or []:
            items = f.get("subgroups_separate") or []
            hit = [it for it in items if any(re.search(rf"(?<![\w-]){t}", low) for t in _sg_terms(it))]
            if len(hit) >= 2:
                return (f"[note: {r.get('ref')} reports these as separate subgroups ({'; '.join(hit[:3])}) - the effect was "
                        f"not reported in patients who meet all of them at once]")
    return None


# --------------------------------------------------------------------------- prove an uncited sentence
_DIR_WORDS = {"decrease": r"\b(?:reduc\w*|decreas\w*|lower\w*|inhibit\w*|suppress\w*|declin\w*|fell|drop\w*)\b",
              "increase": r"\b(?:increas\w*|rais\w*|higher|elevat\w*|promot\w*|enhanc\w*|improv\w*)\b",
              "no_effect": r"\b(?:no (?:significant |statistically significant )?(?:effect|change|difference|benefit|impact)|"
                           r"did not (?:affect|change|reduce|alter)|not significant\w*|unchanged|null)\b"}


_SYS_HUMAN = r"\b(?:humans?|patients?|women|men|trials?|clinical|people|participants)\b"
_SYS_ANIMAL = r"\b(?:mice|mouse|rats?|animals?|rodents?|xenografts?|in vivo)\b"
_SYS_CELL = r"\b(?:cells?|in vitro|cell lines?|cultures?)\b"
# a magnitude comparison across systems ('larger effects are seen in animals') cannot be checked: effect sizes on
# different endpoints / systems are not comparable, so such a sentence is never 'proved'
_CROSS_MAGNITUDE = re.compile(r"\b(?:larger|greater|stronger|bigger|more pronounced|more marked|smaller|weaker|modest(?:ly)?er)\s+"
                              r"(?:effects?|reductions?|benefits?|responses?|changes?)\b", re.I)
_HEDGE_QUANT = re.compile(r"\b(?:some|one|a few|several|a minority|\d+ of \d+|isolated|single)\b", re.I)


def _system_of(text: str) -> Optional[str]:
    return ("human" if re.search(_SYS_HUMAN, text, re.I) else "animal" if re.search(_SYS_ANIMAL, text, re.I)
            else "cell" if re.search(_SYS_CELL, text, re.I) else None)


def _prove_clause(clause: str, fams: set, sysw: Optional[str], rows: List[Dict[str, Any]], dirs: set):
    by_dir: Dict[str, set] = {}
    measured: set = set()                   # every study that measured the endpoint, whatever its direction
    for r in rows:
        if r.get("evidence_type") == "review_meta":
            continue
        for f in r.get("findings") or []:
            mt = str(f.get("model_type") or r.get("evidence_type") or "")
            if sysw == "human" and not mt.startswith("human"):
                continue
            if sysw == "animal" and mt not in ("animal", "experimental_disease_model"):
                continue
            if sysw == "cell" and mt not in ("cell_culture", "organoid_ex_vivo"):
                continue
            if f.get("endpoint") in fams and f.get("direction") in ("increase", "decrease", "no_effect"):
                measured.add(r["ref"])
            if f.get("endpoint") in fams and f.get("direction") in dirs:
                by_dir.setdefault(f["direction"], set()).add(r["ref"])
    return by_dir, measured


def prove_sentence(sent: str, rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Which verified findings support an uncited sentence? Endpoint(s) named in the sentence x direction(s) it states x
    (when named) the system (patients / animals / cells). Returns {'refs', 'text', 'how'}; refs == [] means nothing
    in the evidence supports it. A quantifier the counts contradict ('most studies showed no effect' when 2 of 5 did)
    is replaced by the exact count.

    Clause by clause: 'X in humans; larger effects only in animals' is two statements, and EACH must be supported in
    its own system. A clause stating an effect that most findings in that system do NOT show ('reduces tumour growth
    in animals' when 3 of 4 animal studies found no difference) is not supported, and a magnitude comparison across
    systems is never treated as proved."""
    if _CROSS_MAGNITUDE.search(sent):
        return {"refs": [], "text": sent, "how": ""}
    fams_all = set(_families_in(sent))
    if fams_all & {"proliferation", "tumour_growth"}:
        fams_all |= {"proliferation", "tumour_growth"}

    def dirs_of(text: str) -> set:
        dirs = {d for d, rx in _DIR_WORDS.items() if re.search(rx, text, re.I)}
        if "no_effect" in dirs:                 # 'did not reduce' is a null, not a decrease
            if not re.search(r"\b(?:only|but|while|whereas|and)\b[^.]{0,80}" + _DIR_WORDS["decrease"], text, re.I):
                dirs.discard("decrease")
            dirs.discard("increase") if not re.search(_DIR_WORDS["increase"], re.sub(_DIR_WORDS["no_effect"], " ", text), re.I) else None
        return dirs

    if not fams_all or not dirs_of(sent):
        return {"refs": [], "text": sent, "how": ""}
    clauses = [c for c in re.split(r";|\s+(?:while|whereas|but)\s+", re.sub(r"\[[^\]]*\]", " ", sent)) if c.strip()]
    sys_sent = _system_of(sent)
    all_refs: set = set()
    hows, text = [], sent
    for clause in clauses:
        fams = set(_families_in(clause)) or fams_all
        if fams & {"proliferation", "tumour_growth"}:
            fams |= {"proliferation", "tumour_growth"}
        dirs = dirs_of(clause)
        sysw = _system_of(clause) or (sys_sent if len(clauses) == 1 else None)
        if not dirs:
            if _system_of(clause) and EFFICACY.search(clause):
                return {"refs": [], "text": sent, "how": ""}     # an effect claim about a system with no checkable direction
            continue
        by_dir, measured = _prove_clause(clause, fams, sysw, rows, dirs)
        if set(by_dir) != dirs:             # every direction the clause states must be found in the evidence
            return {"refs": [], "text": sent, "how": ""}
        total = len(measured) or 1
        m = re.search(r"\bmost (?:of the )?(studies|trials|findings|papers)\b([^.;]{0,60})", clause, re.I)
        if m:
            said = next((d for d, rx in _DIR_WORDS.items() if re.search(rx, m.group(2), re.I)), None)
            if said in by_dir and len(by_dir[said]) * 2 <= total:
                i = text.find(m.group(0))
                if i >= 0:
                    text = text[:i] + f"{len(by_dir[said])} of {total} {m.group(1)}" + text[i + len(m.group(0)) - len(m.group(2)):]
        elif sysw and not _HEDGE_QUANT.search(clause) and \
                all(len(v) * 2 < total for d, v in by_dir.items() if d in ("increase", "decrease")) and \
                any(d in ("increase", "decrease") for d in by_dir):
            # 'metformin reduces tumour growth in animals' when only 1 of 4 animal studies shows it: not representative
            return {"refs": [], "text": sent, "how": ""}
        all_refs |= {x for v in by_dir.values() for x in v}
        hows.append("; ".join(f"{d.replace('_', ' ')}: {', '.join(sorted(v, key=lambda x: int(re.sub(r'[^0-9]', '', x) or 0)))}"
                              for d, v in by_dir.items()))
    if not all_refs:
        return {"refs": [], "text": sent, "how": ""}
    refs = sorted(all_refs, key=lambda x: int(re.sub(r"\D", "", x) or 0))
    return {"refs": refs, "text": text, "how": " | ".join(h for h in hows if h)}


# --------------------------------------------------------------------------- bucket summaries from the verified findings
BUCKET_FIELDS = ("human_evidence", "animal_evidence", "in_vitro_evidence", "plant_evidence", "microbial_evidence")
_BUCKET_MODELS = {"human_evidence": ("human_clinical", "human_observational", "case_report"),
                  "animal_evidence": ("animal", "experimental_disease_model"),
                  "in_vitro_evidence": ("cell_culture", "organoid_ex_vivo"), "plant_evidence": ("plant",),
                  "microbial_evidence": ("microbial",)}


def bucket_summary(rows: List[Dict[str, Any]], field: str) -> str:
    """What the verified findings of one kind of system actually show - the same data as the evidence matrix."""
    from semantic_audit import FAMILY_LABEL
    from report_validator import DIR_WORD
    want = _BUCKET_MODELS.get(field, ())
    parts = []
    for r in rows:
        if r.get("evidence_type") == "review_meta":
            continue
        fs = [f for f in r.get("findings") or [] if (f.get("model_type") or r.get("evidence_type")) in want]
        if fs:
            parts.append(f"{r['ref']}: " + "; ".join(
                f"{FAMILY_LABEL.get(f.get('endpoint'), str(f.get('endpoint') or '?').replace('other:', ''))} "
                f"{DIR_WORD.get(f.get('direction') or 'unclear', 'unclear')}" for f in fs[:3]))
    kind = field.replace("_evidence", "").replace("_", " ").replace("in vitro", "cell / organoid")
    if not parts:
        return f"No verified {kind} finding is in the deep-read sample."
    return f"Verified {kind} findings (from the evidence matrix): " + " · ".join(parts[:6]) + \
        (" ..." if len(parts) > 6 else "") + "."
