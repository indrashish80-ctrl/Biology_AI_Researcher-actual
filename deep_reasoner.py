"""
deep_reasoner.py
----------------
Executive-summary synthesis with a free open-weight REASONING model
(AIRouter.ask_reasoning -> Groq gpt-oss-120b / Qwen, Gemini thinking, NVIDIA NIM).

Grounding discipline (same spirit as claim_extraction.py):
  * the model only sees the evidence matrix, contradiction verdicts, STRING edges
    and bucket counts - never free web text;
  * every citation must be an [S#] that exists in the matrix; anything else is
    stripped and logged as an audit note;
  * certainty is capped by the evidence: a question about human health with no
    human studies -> at most "low"; likewise a plant / microbial / animal question
    with no study in that organism group -> at most "low".
  * works for ALL biology: the question's focus organism decides what counts as
    direct evidence (humans for medicine, plants for plant science, ...).
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

from evidence_matrix import (BUCKET_LABELS, BUCKET_ORDER, HUMAN_PATTERNS, ANIMAL_PATTERNS, MICROBIAL_PATTERNS,
                             PLANT_PATTERNS, bucket_direction_summary)
from llm_json import parse_json_response

CERTAINTY = ["very low", "low", "moderate", "high"]

_HEALTH_RE = re.compile(
    r"\b(patients?|clinical|disease|disorder|syndrome|cancer|tumou?r|carcinoma|diabet|obes|cardio|heart|stroke|"
    r"alzheimer|parkinson|dementia|depress|anxiety|infection|therapy|therapeutic|treatment|treat|drug|medicin|dose|"
    r"dosage|vaccine|surgery|mortality|morbidity|prognos|diagnos|symptom|health|human|people|women|men|children|"
    r"pregnan|elderly|adults?|covid|hiv|asthma|arthritis|hypertension|metformin|aspirin|statin)\w*", re.I)
FOCUS_LABEL = {"human": "human health", "animal": "animals", "plant": "plants", "microbial": "microorganisms",
               "general": "general biology"}


def question_focus(question: str) -> str:
    """Which organism group the question is about: human | animal | plant | microbial | general."""
    q = question or ""

    def hits(patterns):
        return sum(len(re.findall(p, q, flags=re.I)) for p in patterns)

    scores = {"plant": hits(PLANT_PATTERNS), "microbial": hits(MICROBIAL_PATTERNS), "animal": hits(ANIMAL_PATTERNS)}
    best = max(scores, key=scores.get)
    human = hits(HUMAN_PATTERNS) + len(_HEALTH_RE.findall(q))
    if scores[best] and scores[best] >= human:
        return best
    if human:
        return "human"
    return "general"

PROMPT = """You are a senior biomedical evidence synthesist. Think step by step, then answer.

RESEARCH QUESTION:
{question}

QUESTION FOCUS (organism group the answer must be about): {focus}

EVIDENCE HIERARCHY for this question (every study below carries its level):
{hierarchy}
  S  = review / meta-analysis (summarises other studies; NOT an independent finding)
  L7 = your own hypothesis or speculation - must be labelled "AI hypothesis (L7)" in the text.

VERIFIED EVIDENCE - the RETRIEVED SAMPLE ({sample}). Only claims that passed quote-level and
semantic verification are listed; quarantined claims are excluded. Cite studies as [S1], [S2] ...
It is a sample of the literature, not all of it. {totals}
{matrix}

EVIDENCE BUCKET COUNTS (one primary study = one vote, by its representative verified finding):
{buckets}

CROSS-STUDY CONTRADICTIONS:
{contradictions}

STRING-db PROTEIN INTERACTIONS (combined score 0-1; association, NOT causal direction):
{string}

{context_block}

HARD RULES (a deterministic checker WITHHOLDS or rewrites every violating sentence):
- Absence of evidence is not evidence of absence. Never write that something cannot be cured, does not work or
  is ineffective unless a listed study TESTED it and found no effect. Write "no evidence demonstrating ... was identified".
- Recovery / improvement / response is not cure. Use the outcome rung each study actually reports and its follow-up.
- A combination regimen is evidence for the regimen only - never for one of its components.
- An uncontrolled observation after treatment is not causation: write "was followed by", not "caused/led to/cured".
- Do not generalise an induced, subtype-specific or other-species model to the disease in general.
- Every substantive statement must cite [S#]; uncited conclusions are withheld. Cite the STUDY ([S1]), not the
  finding id (write [S1], never (S1.E1.F1)).
- A statement about humans may cite only L1-L2 studies. Cell, organoid, animal and in silico findings
  must be attributed to their model ("in MCF-7 cells [S3]") and never presented as a human mechanism.
- Never infer one endpoint from another: an association with survival or recurrence is not evidence
  about proliferation; viability is not apoptosis; expression is not activity.
- A high experimental dose shows biological activity at that dose; say its relevance to human exposure
  is uncertain (see dose relevance) - do not call it invalid, and do not ignore it.
- Mechanistic chain: write ONE arrow per step ("A -> B [S#]"). Every arrow is checked on its own; a downstream arrow
  with causal evidence (e.g. CDK4 inhibition -> Rb) says nothing about whether the agent reaches A. A STRING link is
  an association only and never shows that the agent acts through those proteins.
- Describe only the deep-read sample: never "consistently", "uniformly", "all studies" or "proves".
- A finding marked "subgroup: X" holds only in X - never state it for all patients. A finding with
  direction "no_effect" is a real null result: report it and weigh it against the positive findings.
- Use only the verified findings listed. Do not introduce doses, sample sizes, thresholds or mechanisms
  that are not in the list unless you label them "AI hypothesis (L7)".

TASK - multi-step biological deduction:
1. Weigh evidence by how DIRECTLY it studies the organism and setting in the question focus. Keep the
   buckets (human, animal, in vitro, plant, microbial) separate. For human-health questions never let animal
   or cell data stand in for a human conclusion and say explicitly when human evidence is absent or weak.
   For plant, microbial, animal, ecological or evolutionary questions judge directness the same way
   (e.g. field vs greenhouse, wild population vs lab strain, whole organism vs isolated cells).
2. Propose a HYPOTHETICAL mechanistic chain (intervention -> molecular target(s) -> pathway -> phenotype) and label it
   explicitly as a speculative hypothesis. STRING protein-protein interaction scores indicate physical or functional
   ASSOCIATION, not directional causal cascades in vivo: two proteins binding (e.g. TP53-HMGB1 0.98) does NOT show
   that the intervention causes that sequence in a patient. Use STRING only as support for plausibility, never as
   proof of treatment-induced causality. Every step that is more than a hypothesis must be backed by an [S#] study.
3. Explain disagreements (organism, dose, design, power) instead of averaging them away.
4. Rate overall certainty with BioSynthetica's internal framework (GRADE-inspired, NOT a formal GRADE assessment):
   high | moderate | low | very low - never above the certainty ceiling given above.

Rules: cite only [S#] labels above; no invented studies, numbers, or p-values; if evidence is thin, say so.
SCOPE every statement to the retrieved sample. You have NOT seen the whole literature, so never write
"no trial has ever...", "there is no evidence that..." or "studies have never...". Write instead, e.g.:
"Within the retrieved sample ({sample}), no trial directly measured tumour proliferation markers (e.g. Ki-67)."
Name the measurable endpoint/biomarker the studies actually used (e.g. Ki-67, HbA1c, MIC, stomatal conductance).

Return ONLY valid JSON:
{{
  "executive_summary": "150-250 words of plain-language markdown with [S#] citations",
  "bottom_line": "one sentence answer to the question",
  "human_evidence": "what human studies show (or that there are none) with [S#]",
  "animal_evidence": "what animal studies show with [S#]",
  "in_vitro_evidence": "what cell/in vitro studies show with [S#]",
  "plant_evidence": "what plant studies show with [S#] (or 'none')",
  "microbial_evidence": "what microbial studies show with [S#] (or 'none')",
  "translational_gap": "how far findings from other models/settings transfer to the organism the question is about",
  "mechanistic_chain": ["step 1 [S#]", "step 2 (STRING: A-B 0.92)", "..."],
  "conflicts_explained": ["..."],
  "certainty": "high|moderate|low|very low",
  "certainty_rationale": "...",
  "key_uncertainties": ["..."]
}}"""


# how much of the sample the reasoning prompt carries, from most to least. A model that refuses the first size is
# asked again at the next one, so a long sample costs detail rather than the whole synthesis.
PROMPT_TIERS = [
    {"rows": 20, "findings": 4, "contra": 40, "string": 15, "context": 100_000},
    {"rows": 14, "findings": 3, "contra": 24, "string": 10, "context": 12_000},
    {"rows": 9, "findings": 2, "contra": 14, "string": 6, "context": 6_000},
    {"rows": 6, "findings": 2, "contra": 8, "string": 4, "context": 3_000},
    # the floor: a short instruction set and the strongest studies only, so even an 8k-token model gets a synthesis
    {"rows": 4, "findings": 1, "contra": 4, "string": 0, "context": 0, "compact": 1},
]
# the full instruction block is about 6,000 characters on its own. At the smallest size it is replaced by this
# summary of the same rules, so a small-context model can still write a synthesis instead of the report losing one.
COMPACT_PROMPT = """You are a careful evidence reviewer. Question: {question}
Focus: {focus}. Sample: {sample}
Verified findings (the ONLY evidence you may use - cite as [S1], [S2]):
{matrix}
{totals}
Rules: never state a human effect that only animal or cell studies show; never call an association a cause; never
present a short-term marker as a patient outcome; say "within the sample read" not "no evidence exists"; cite a study
for every substantive sentence; a non-significant result is not proof of no effect.
Return ONLY this JSON:
{{"executive_summary":"2-3 sentences with citations","bottom_line":"one sentence with citations",
"human_evidence":"...","animal_evidence":"...","in_vitro_evidence":"...","translational_gap":"...",
"mechanistic_chain":[],"conflicts_explained":[],"certainty":"low","certainty_rationale":"...",
"key_uncertainties":["..."]}}"""
PROMPT_CHAR_BUDGET = 26_000      # about 6-7 thousand tokens: inside the context window of the usual reasoning models


def _too_large(error_text: str) -> bool:
    """Is this 'your prompt is too long' rather than another failure?"""
    t = (error_text or "").lower()
    return ("413" in t or "too large" in t or "request_too_large" in t or "context length" in t
            or "too many tokens" in t or "maximum context" in t or "reduce the length" in t
            or "exceeds" in t and "token" in t)


def _matrix_for_prompt(rows: List[Dict[str, Any]], max_rows: int = 20, max_findings: int = 4) -> str:
    lines = []
    for r in rows[:max_rows]:
        lvl = (r.get("level_label") or "").split(" ")[0] or "?"
        dose = (r.get("dose_relevance") or {}).get("category")
        lines.append(
            f"[{r['ref']}] {lvl} {r.get('evidence_type_label') or BUCKET_LABELS.get(r['bucket'], r['bucket'])} | "
            f"model={r.get('model') or '-'} | n={r.get('n') or '?'} | text={r.get('text_basis')}"
            + (f" | dose relevance: {dose}" if dose and dose not in ("not reported",) else "")
            + (f" | {r['n_quarantined']} claim(s) quarantined (not shown)" if r.get("n_quarantined") else "")
            + f" | comparator: {r.get('comparator') or 'not reported'} | follow-up: "
            + (f"{r['follow_up_days'] / 30.44:.1f} months" if r.get("follow_up_days") else "not reported")
            + (f" | best outcome: {(r.get('best_outcome') or {}).get('label')}" if (r.get("best_outcome") or {}).get("level") is not None else "")
            + (f" | disease studied: {', '.join((r.get('disease_context') or {}).get('cause_or_induction') or (r.get('disease_context') or {}).get('definition_variants') or [])}"
               if (r.get("disease_context") or {}).get("definition_variants") or (r.get("disease_context") or {}).get("cause_or_induction") else "")
        )
        findings = r.get("findings")
        if findings is None and r.get("key_finding"):      # legacy matrices without findings
            findings = [{"id": r["ref"], "model_type": "", "model_detail": "", "endpoint": r.get("outcome"),
                         "direction": r.get("direction"), "p_value": r.get("p_value"), "statement": r.get("key_finding")}]
        if not findings:
            lines.append("    (no verified finding)")
        for f in (findings or [])[:max_findings]:
            lines.append(f"    {f.get('id')}: {(f.get('model_type') or '').replace('_', ' ')} {f.get('model_detail') or ''} | "
                         f"{(f.get('endpoint') or '?').replace('_', ' ')} -> {f.get('direction') or 'unclear'} "
                         f"(p={f.get('p_value')}) | role: {f.get('evidence_role') or 'direct'} | "
                         f"subtype: {', '.join(f.get('subtypes') or []) or 'not specified'} | "
                         f"subgroup: {f.get('subgroup') or 'not stated'} | status: {f.get('claim_status') or '-'} | "
                         f"{', '.join(f.get('tags') or [])} | {(f.get('statement') or '')[:180]}")
    return "\n".join(lines) or "(no included studies)"


def _totals_text(matrix: Dict[str, Any]) -> str:
    t = matrix.get("totals") or {}
    if not t:
        return ""
    return (f"({t.get('papers')} papers, {t.get('experiments')} experiments, {t.get('independent_findings')} independent "
            f"findings from {t.get('verified_claims')} verified claims; {t.get('quarantined_claims')} claims quarantined)")


def _hierarchy_text(focus: str) -> str:
    from study_classifier import HUMAN_LEVEL_NAMES, LEVEL_NAMES
    names = HUMAN_LEVEL_NAMES if focus in ("human", "general") else LEVEL_NAMES
    return "\n".join(f"  {v}" for k, v in names.items() if k <= 6)


def _sample_label(rows: List[Dict[str, Any]]) -> str:
    n = len(rows)
    return f"S1–S{n}, {n} {'study' if n == 1 else 'studies'}" if n else "no studies"


# Sentences that generalise a 10-paper sample to the whole literature.
_ABSOLUTE_RE = re.compile(
    r"(?<![Ww]ithin the retrieved sample)[^.]*\b(?:"
    r"no (?:\w+ ){0,3}(?:trials?|studies|evidence|data|research)\b[^.]{0,40}\b(?:ever|to date|have been|has been|exists?)|"
    r"(?:has|have) never been|never (?:been )?(?:studied|tested|measured|shown|reported)|"
    r"there (?:is|are) no (?:\w+ ){0,2}(?:trials?|studies|evidence|data)\b(?![^.]*\b(?:retrieved|included|this sample|in the sample|S\d))"
    r")[^.]*\.", re.I)


def absolute_claims(data: Dict[str, Any]) -> List[str]:
    """Return sentences that claim absence/presence of evidence beyond the retrieved sample."""
    found: List[str] = []
    for v in data.values():
        for t in (v if isinstance(v, list) else [v]):
            if isinstance(t, str):
                found.extend(m.group(0).strip() for m in _ABSOLUTE_RE.finditer(t))
    return found


_FINDING_CITE = re.compile(r"[\(\[]\s*((?:S\d+(?:\.E\d+(?:\.F\d+)?)?)(?:\s*[,;]\s*(?:and\s+)?S\d+(?:\.E\d+(?:\.F\d+)?)?)*)\s*[\)\]]")


def normalize_citations(text: Any) -> Any:
    """Reasoning models often cite the finding ids they were shown - '(S1.E1.F1)', '[S4.E1.F1, S6.E2.F1]',
    '(S3)'. Every such citation becomes the study citation '[S1]' the evidence rules check, so a correctly
    cited sentence is not withheld as 'untraceable'."""
    if isinstance(text, list):
        return [normalize_citations(t) for t in text]
    if isinstance(text, dict):
        return {k: normalize_citations(v) for k, v in text.items()}
    if not isinstance(text, str):
        return text

    def fix(m):
        refs = list(dict.fromkeys(re.findall(r"S\d+", m.group(1))))
        return "[" + ", ".join(refs) + "]"
    return _FINDING_CITE.sub(fix, text)


def _scrub_citations(text: Any, valid: set, bad: List[str]) -> Any:
    if isinstance(text, list):
        return [_scrub_citations(t, valid, bad) for t in text]
    if not isinstance(text, str):
        return text

    def expand(token: str) -> List[str]:
        m = re.fullmatch(r"S(\d+)\s*[-\u2013]\s*S?(\d+)", token)
        if m:   # "S1-S3" / "S1–3"
            lo, hi = int(m.group(1)), int(m.group(2))
            if 0 < hi - lo < 50:
                return [f"S{i}" for i in range(lo, hi + 1)]
        return [token]

    def fix(m):
        refs = [r for x in re.split(r"[,;]", m.group(1)) if x.strip() for r in expand(x.strip())]
        keep = [x for x in refs if x in valid]
        bad.extend(x for x in refs if x not in valid)
        if keep:
            return f"[{', '.join(keep)}]"
        # the reader must see that the sentence lost its only citation (it was citing a study that is not in the report)
        return "[note: its citation pointed to a study that is not in this report and was removed]"

    return re.sub(r"\[(S\d+(?:\s*(?:[,;]|[-\u2013])\s*S?\d+)*)\]", fix, text)


class DeepReasoner:
    def __init__(self, router: Any = None):
        self.router = router

    def synthesize(self, question: str, matrix: Dict[str, Any], contradictions: List[Dict[str, Any]],
                   string_net: Optional[Dict[str, Any]] = None, prisma: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        rows = matrix.get("rows") or []
        buckets = bucket_direction_summary(matrix)
        counts = matrix.get("bucket_counts") or {}
        focus = question_focus(question)
        ev = self._evidence_context(question, rows, prisma, focus)

        if self.router is None:
            out = self._deterministic(question, rows, counts, buckets, string_net, focus, matrix.get("totals"))
            return self._finalize(out, rows, focus, ev, question)

        _same = [c for c in (contradictions or []) if (c.get("kind") or (c.get("comparison") or {}).get("kind")) != "model_mismatch"]
        _cross = len(contradictions or []) - len(_same)
        contra_txt = ("\n".join(
            f"- claim {c.get('claim_a_id')} vs {c.get('claim_b_id')}: {c.get('verdict')} ({c.get('reasoning')})"
            for c in _same[:8]) or "None detected within the same experimental system.") + (
            f"\n({_cross} opposite result(s) across different experimental systems, e.g. patients vs cells - these are "
            f"CROSS-CONTEXT DISAGREEMENTS, not direct contradictions: name them as such and say which difference (system, "
            f"population, dose, endpoint) may explain them; never call the literature 'consistent' because of them.)" if _cross else "")
        edges = (string_net or {}).get("edges") or []
        string_txt = "\n".join(f"- {e['a']} — {e['b']}: {e['score']} ({e.get('top_evidence') or 'combined'})"
                               for e in edges[:15]) or "No STRING interactions retrieved."
        def _build(tier: "Dict[str, int]") -> str:
            if tier.get("compact"):
                return COMPACT_PROMPT.format(
                    question=question, focus=FOCUS_LABEL[focus], sample=_sample_label(rows),
                    matrix=_matrix_for_prompt(rows, tier["rows"], tier["findings"]), totals=_totals_text(matrix))
            return PROMPT.format(
                question=question, focus=FOCUS_LABEL[focus],
                matrix=_matrix_for_prompt(rows, tier["rows"], tier["findings"]), sample=_sample_label(rows),
                hierarchy=_hierarchy_text(focus), totals=_totals_text(matrix),
                buckets=json.dumps({BUCKET_LABELS[k]: v for k, v in buckets.items()}, ensure_ascii=False),
                contradictions="\n".join(contra_txt.splitlines()[:tier["contra"]]) or contra_txt,
                string="\n".join(string_txt.splitlines()[:tier["string"]]) or string_txt,
                context_block=ev["prompt_block"][:tier["context"]])

        # A reasoning model that refuses an over-long prompt used to cost the whole qualitative synthesis: the report
        # fell back to a deterministic tally and lost every explanation. Now the prompt is trimmed and asked again -
        # fewer studies listed, fewer findings each, a shorter context block - and only a failure at the smallest
        # size falls back. The trimming is recorded, so a reader knows the synthesis saw part of the sample.
        data, res, used_tier, size_notes = None, None, None, []
        start = 0
        for i, tier in enumerate(PROMPT_TIERS):
            if len(_build(tier)) <= PROMPT_CHAR_BUDGET:
                start = i
                break
        else:
            start = len(PROMPT_TIERS) - 1
        last_exc: Optional[Exception] = None
        for tier in PROMPT_TIERS[start:]:
            prompt = _build(tier)
            try:
                res = self.router.ask_reasoning(prompt)
                data = parse_json_response(res.get("content") or "")
                if not isinstance(data, dict) or not data.get("executive_summary"):
                    raise ValueError("reasoning model returned no executive_summary")
                used_tier = tier
                break
            except Exception as exc:
                last_exc = exc
                data = None
                if not _too_large(str(exc)):
                    break                 # a different failure: trimming will not help
                size_notes.append(f"the reasoning model refused a {len(prompt)} character prompt")
        if data is None:
            out = self._deterministic(question, rows, counts, buckets, string_net, focus, matrix.get("totals"),
                                      why="keys were supplied, but every reasoning model tried returned no usable answer")
            out["notes"].append(f"Reasoning model unavailable or malformed output: {str(last_exc)[:200]}")
            if size_notes:
                out["notes"].append("Prompt size: " + "; ".join(size_notes[:3])
                                    + " - the synthesis below is the deterministic tally.")
            return self._finalize(out, rows, focus, ev, question)
        if used_tier and used_tier is not PROMPT_TIERS[0]:
            n_shown = min(used_tier["rows"], len(rows))
            size_notes.append(f"the synthesis was written from the {n_shown} most relevant of {len(rows)} studies "
                              f"(the prompt had to be shortened for the model's context limit); every study is still "
                              f"listed in the evidence matrix and counted in the tallies")

        valid = {r["ref"] for r in rows}
        bad: List[str] = []
        for k in list(data.keys()):
            data[k] = _scrub_citations(normalize_citations(data[k]), valid, bad)

        certainty = str(data.get("certainty") or "low").lower().strip()
        if certainty not in CERTAINTY:
            certainty = "low"
        notes = list(size_notes)      # 'the synthesis was written from the N most relevant of M studies'
        direct = self._direct_count(focus, counts)
        if direct == 0 and CERTAINTY.index(certainty) > 1:
            what = ("human" if focus == "human" else BUCKET_LABELS[focus].lower() if focus in BUCKET_LABELS
                    else "whole-organism")
            notes.append(f"Certainty capped from '{certainty}' to 'low': no {what} studies in the included set.")
            certainty = "low"
        if bad:
            notes.append(f"Removed {len(bad)} citation(s) to studies not in the evidence matrix: {sorted(set(bad))}")
        overreach = absolute_claims(data)
        if overreach:
            notes.append(f"Scope warning: {len(overreach)} statement(s) about missing evidence refer only to the "
                         f"retrieved sample ({_sample_label(rows)}), not to all published research - e.g. "
                         f"\"{overreach[0][:160]}\"")

        data["certainty"] = certainty
        data["question_focus"] = focus
        raw_data = json.loads(json.dumps(data, default=str))
        g_notes = self._apply_guards(data, rows, focus, string_net, ev["guard_ctx"])
        blocked = data.get("blocked_statements") or []
        if blocked and getattr(self, "reflect", True):
            # one reflection round: the statements the evidence rules withheld or rewrote go back to the drafting model
            # with the reason, and the revision is kept only if it breaks FEWER rules (the hard rules still run on it)
            revised = self._reflect(prompt, raw_data, blocked, valid)
            if revised is not None:
                revised["certainty"] = certainty
                revised["question_focus"] = focus
                r_notes = self._apply_guards(revised, rows, focus, string_net, ev["guard_ctx"])
                if len(revised.get("blocked_statements") or []) < len(blocked):
                    notes.append(f"Reflection: {len(blocked)} statement(s) broke evidence rules; the model revised its text "
                                 f"and {len(revised.get('blocked_statements') or [])} still had to be withheld or rewritten.")
                    data, g_notes = revised, r_notes
        notes += g_notes
        out = {
            **data,
            "reasoning_trace": res.get("reasoning"),
            "provider": res.get("provider"),
            "model": res.get("model"),
            "notes": notes,
            "mode": "deep_reasoning",
        }
        return self._finalize(out, rows, focus, ev, question, guarded=True)

    # ------------------------------------------------------------------ evidence context + final hard checks
    @staticmethod
    def _evidence_context(question: str, rows: List[Dict[str, Any]], prisma: Optional[Dict[str, Any]], focus: str) -> Dict[str, Any]:
        from clinical_context import cure_definition, disease_definition_check, parse_question
        from report_validator import budget_limitation, certainty_cap, evidence_availability
        q_info = parse_question(question)
        from evidence_identity import question_frame
        frame = question_frame(question)
        avail = evidence_availability(prisma, rows, focus)
        budget = budget_limitation(prisma)
        cap = certainty_cap(rows, prisma, focus, frame)
        cure = cure_definition() if q_info["asks_cure"] else None
        ddef = disease_definition_check(q_info["disease_terms"], rows) if q_info["disease_terms"] else None
        block = []
        if cure:
            block.append("OPERATIONAL OUTCOME DEFINITION (decided before looking at results):\n  " + cure["definition"] +
                         "\n  Outcome ladder (never interchangeable): " + " < ".join(cure["ladder"]))
        block.append("SEARCH COVERAGE - what was and was NOT examined (" + avail["search"] + "):\n" +
                     "\n".join(f"  {v['label']}: {v['state']}" for v in (avail.get("types") or {}).values()))
        block.append("DEEP-READ BUDGET: " + budget["statement"])
        if ddef:
            block.append("DISEASE DEFINITION IN EACH STUDY: " + ddef["statement"] + "\n" +
                         "\n".join(f"  [{p['ref']}] {p['studied']} ({p['species']}{', induced model' if p['induced'] else ''})" for p in ddef["per_study"]))
        if frame.get("intent") == "etiology":
            block.append(f"THIS IS A CAUSES QUESTION. Each finding carries an evidence role. Only 'disease_risk' findings measure "
                         f"{frame.get('disease') or 'disease'} occurrence/risk; 'mechanistic' and 'intermediate' findings (ROS, apoptosis, "
                         f"chromosome damage, gene expression) are NOT causes of the disease. Association is not causation. "
                         f"Never write 'X causes/is a cause of {frame.get('disease') or 'the disease'}' or 'X -> Y -> disease' unless "
                         f"a disease_risk finding from a controlled design supports it.")
        if frame.get("heterogeneous"):
            block.append(f"{(frame.get('disease') or '').upper()} IS HETEROGENEOUS: state which subtype each finding concerns "
                         f"(listed per study); never generalise one subtype to the disease as a whole.")
        hs = {"human_clinical": sum(1 for r in rows if r.get("evidence_type") == "human_clinical"),
              "human_observational": sum(1 for r in rows if r.get("evidence_type") == "human_observational")}
        block.append(f"HUMAN EVIDENCE PRESENT: {hs['human_clinical']} clinical, {hs['human_observational']} observational/epidemiological "
                     f"study(ies). Do not describe evidence types that are not listed; do not characterise studies with no verified finding.")
        block.append("CERTAINTY CEILING (set by the evidence, you cannot exceed it): " + cap["max"] +
                     ("  - because: " + "; ".join(cap["reasons"]) if cap["reasons"] else ""))
        return {"q_info": q_info, "availability": avail, "budget": budget, "cap": cap, "cure": cure, "ddef": ddef,
                "prompt_block": "\n\n".join(block),
                "frame": frame,
                "guard_ctx": {"asks_cure": q_info["asks_cure"], "disease_terms": q_info["disease_terms"], "frame": frame,
                              "human_split": hs}}

    def _finalize(self, out: Dict[str, Any], rows: List[Dict[str, Any]], focus: str, ev: Dict[str, Any], question: str,
                  guarded: bool = False) -> Dict[str, Any]:
        from report_validator import CERTAINTY_LABEL, apply_cap, safe_bottom_line
        from reasoning_guards import guard_synthesis
        notes = list(out.get("notes") or [])
        if not guarded:          # deterministic text is checked by the same hard rules
            notes += [f"Guard - {n}" for n in guard_synthesis(out, rows, focus, ev["guard_ctx"])]
        bl = out.get("bottom_line") or ""
        if not guarded:          # no reasoning model: the bottom line is always the deterministic, traceable one
            out["bottom_line"] = safe_bottom_line(question, rows, focus, ev["q_info"], ev["availability"], ev["budget"], ev["frame"])
        elif not bl.strip() or "⛔" in bl or "[rewritten" in bl or (ev.get("frame") or {}).get("intent") == "etiology" \
                or not re.search(r"\[S\d+", bl):        # a bottom line that cites no study is not traceable
            # the most-read sentence: if a hard rule had to touch it, or the question asks for CAUSES (where the
            # model tends to turn mechanisms / associations into causes), it is the deterministic, traceable statement
            out["bottom_line"] = safe_bottom_line(question, rows, focus, ev["q_info"], ev["availability"], ev["budget"], ev["frame"])
            notes.append("Bottom line replaced by the deterministic evidence-based statement "
                         + ("(causes question: association / mechanism / cause kept apart)."
                            if "⛔" not in bl and "[rewritten" not in bl and bl.strip() else "(the generated one broke a hard rule)."))
        c, msg = apply_cap(out.get("certainty") or "low", ev["cap"])
        if msg:
            notes.append(msg)
        out["certainty"] = c
        if ev["cap"]["reasons"]:
            out["certainty_rationale"] = ((out.get("certainty_rationale") or "").rstrip(". ") +
                                          ". Evidence ceiling: " + "; ".join(ev["cap"]["reasons"]) + ".").lstrip(". ")
        out.update({"notes": notes, "certainty_label": CERTAINTY_LABEL, "certainty_cap": ev["cap"],
                    "evidence_availability": ev["availability"], "budget_limitation": ev["budget"],
                    "cure_definition": ev["cure"], "disease_definition": ev["ddef"], "question_info": ev["q_info"],
                    "question_frame": {k: ev["frame"].get(k) for k in ("question", "intent", "disease", "heterogeneous", "exposures", "endpoints")},
                    "question_normalisation": ev["frame"]["normalisation"]})
        return out

    def _reflect(self, prompt: str, draft: Dict[str, Any], blocked: List[Dict[str, Any]], valid: set) -> Optional[Dict[str, Any]]:
        """Send the rule violations back to the reasoning model once; None if no usable revision comes back."""
        if self.router is None:
            return None
        issues = "\n".join(f"- [{b.get('field')}] \"{str(b.get('original'))[:300]}\" -> broke: {'; '.join(b.get('rules') or [])}"
                           for b in blocked[:12])
        rev_prompt = (prompt + "\n\nYOUR PREVIOUS DRAFT (JSON):\n" + json.dumps(draft, ensure_ascii=False, default=str)[:12000] +
                      "\n\nAn automatic evidence check rejected these statements:\n" + issues +
                      "\n\nRewrite the WHOLE JSON once. Fix every listed statement so it is true for the evidence matrix above: "
                      "cite the study for every conclusion, never describe a cell or animal result as an effect in people, say "
                      "which endpoint an absence refers to ('no human study measured proliferation'), and never claim that a "
                      "study type is absent when the matrix lists one. Return only the JSON.")
        try:
            res = self.router.ask_reasoning(rev_prompt)
            data = parse_json_response(res.get("content") or "")
        except Exception:
            return None
        if not isinstance(data, dict) or not data.get("executive_summary"):
            return None
        bad: List[str] = []
        for k in list(data.keys()):
            data[k] = _scrub_citations(normalize_citations(data[k]), valid, bad)
        return data

    @staticmethod
    def _apply_guards(data: Dict[str, Any], rows: List[Dict[str, Any]], focus: str,
                      string_net: Optional[Dict[str, Any]], ctx: Optional[Dict[str, Any]] = None) -> List[str]:
        """Hard evidence rules and mechanism evidence levels on the AI text."""
        from reasoning_guards import chain_summary, guard_synthesis, label_mechanism
        notes = [f"Guard - {n}" for n in guard_synthesis(data, rows, focus, ctx)]
        steps = data.get("mechanistic_chain") or []
        if isinstance(steps, str):
            steps = [steps]
        agents = [e.split()[0] for e in (((ctx or {}).get("frame") or {}).get("exposures") or []) if e.split()]
        labelled = label_mechanism(steps, rows, (string_net or {}).get("edges"), agents=agents)
        data["mechanistic_chain_levels"] = labelled
        data["mechanistic_chain_summary"] = chain_summary(labelled, agents)
        data["mechanistic_chain"] = [f"[{x['level']}] {x['step']}" for x in labelled]
        weak = [x for x in labelled if x["level"] in ("M1", "L7")]
        if weak:
            notes.append(f"{len(weak)} mechanistic step(s) are candidate/AI-hypothesis level only (M1/L7), "
                         f"not demonstrated by a verified study.")
        return notes

    @staticmethod
    def _direct_count(focus: str, counts: Dict[str, int]) -> int:
        """Studies that directly address the question's organism group."""
        if focus in ("human", "animal", "plant", "microbial"):
            return counts.get(focus, 0)
        return sum(counts.get(b, 0) for b in ("human", "animal", "plant", "microbial"))

    # ------------------------------------------------------------------ no-AI fallback
    def _deterministic(self, question, rows, counts, buckets, string_net, focus="general", totals=None,
                       why: str = "no AI key supplied") -> Dict[str, Any]:
        from evidence_identity import question_frame
        from semantic_audit import FAMILY_LABEL
        q_end = set(question_frame(question).get("endpoints") or [])
        want = q_end | ({"tumour_growth"} if "proliferation" in q_end else set()) | ({"proliferation"} if "tumour_growth" in q_end else set())
        label = " / ".join(sorted(FAMILY_LABEL.get(e, e) for e in want)) or "the question's outcome"

        from evidence_matrix import bucket_of_model

        def describe(b):
            # Paper -> Experiment -> Finding: a paper is listed in every bucket where it has a verified FINDING (an animal
            # paper with a cell assay counts for both); primary studies only (reviews have no level); directions are
            # counted per finding on the question's own endpoint - apoptosis UP and proliferation DOWN are never pooled
            prim = [r for r in rows if r.get("level") is not None]
            fin = {r["ref"]: [f for f in r.get("findings") or [] if bucket_of_model(f.get("model_type"), r["bucket"]) == b]
                   for r in prim}
            sub = [r for r in prim if fin[r["ref"]] or (r["bucket"] == b and not r.get("n_findings"))]
            moved = [r["ref"] for r in prim if r["bucket"] == b and r.get("n_findings") and not fin[r["ref"]]]
            if not sub:
                txt = f"No {BUCKET_LABELS[b].lower()} findings in the retrieved sample."
                if moved:
                    txt = txt[:-1] + (f" ({', '.join(moved)} {'is' if len(moved) == 1 else 'are'} classified "
                                      f"{BUCKET_LABELS[b].lower()} but {'its' if len(moved) == 1 else 'their'} verified findings "
                                      f"come from other experimental systems).")
                return txt
            on_q: Dict[str, set] = {}
            other = set()
            for r in sub:
                for f in fin[r["ref"]]:
                    if f.get("answers_question", True) and (not want or f.get("endpoint") in want) \
                            and f.get("direction") in ("increase", "decrease", "no_effect"):
                        sg = f.get("subgroup")
                        on_q.setdefault(f["direction"].replace("_", " "), set()).add(
                            r["ref"] + (f" ({'overall' if sg == 'overall' else 'subgroup: ' + sg})" if sg else ""))
                    else:
                        other.add(r["ref"])
            with_f = [r["ref"] for r in sub if fin[r["ref"]]]
            without = [r["ref"] for r in sub if not fin[r["ref"]]]
            # v24 fix 5: this count includes studies filed under ANOTHER design that also ran experiments in this
            # system (an animal paper with a cell experiment). The bucket table counts one design per study, so the
            # two numbers differ on purpose - say so, instead of printing two different totals under one label.
            borrowed = [r["ref"] for r in sub if r.get("bucket") != b]
            txt = f"{len(sub)} {BUCKET_LABELS[b].lower()} {'study' if len(sub) == 1 else 'studies'}" + \
                (f" [{', '.join(with_f[:8])}]" if with_f else "") + \
                (f" (no verified finding: {', '.join(without)} - not characterised)" if without else "")
            if on_q:
                txt += f"; on {label}: " + "; ".join(f"{k} [{', '.join(sorted(v))}]" for k, v in on_q.items())
            else:
                txt += f"; none measured {label}"
            other -= {x.split(" (")[0] for v in on_q.values() for x in v}
            if other:
                txt += f"; other endpoints only: {', '.join(sorted(other))}"
            if moved:
                txt += (f"; {', '.join(moved)} {'is' if len(moved) == 1 else 'are'} classified {BUCKET_LABELS[b].lower()} but "
                        f"{'its' if len(moved) == 1 else 'their'} verified findings come from other systems")
            if borrowed:
                txt += (f" (includes {', '.join(sorted(borrowed))}, filed under another design but also reporting "
                        f"{BUCKET_LABELS[b].lower()} experiments - the evidence-by-bucket table counts one design "
                        f"per study, so it lists {len([r for r in sub if r.get('bucket') == b])})")
            return txt + "."

        edges = (string_net or {}).get("edges") or []
        from reasoning_guards import label_mechanism
        chain_steps = [f"{e['a']} — {e['b']} associated in STRING (score {e['score']}; association, not causal direction)"
                       for e in edges[:4]]
        chain_levels = label_mechanism(chain_steps, rows, edges)
        chain = [f"[{x['level']}] {x['step']}" for x in chain_levels]
        certainty = "low" if self._direct_count(focus, counts) else "very low"
        shown = [b for b in BUCKET_ORDER if b != "other" and (counts.get(b) or b == focus)] or ["human", "animal", "in_vitro"]
        summary = (
            f"{_totals_text({'totals': totals}) if totals else ''} "
            f"Within the retrieved sample ({_sample_label(rows)}): "
            + " ".join(describe(b) for b in shown) + "\n\n"
            f"This summary was generated deterministically ({why}), so it reports the direction of "
            "evidence per bucket without interpretation. "
            + ("Add a Groq, Gemini or NVIDIA key for a reasoned synthesis." if why == "no AI key supplied"
               else "Re-run later, or check the key / provider status on the API Key page.")
        )
        return {
            "executive_summary": summary,
            "bottom_line": ("Evidence tallied by bucket; interpretive synthesis requires an AI key." if why == "no AI key supplied"
                            else "Evidence tallied by bucket; the AI reasoning model returned no usable answer, so no "
                                 "interpretive synthesis was written."),
            "human_evidence": describe("human"), "animal_evidence": describe("animal"),
            "in_vitro_evidence": describe("in_vitro"),
            "plant_evidence": describe("plant"), "microbial_evidence": describe("microbial"),
            "question_focus": focus, "mechanistic_chain_levels": chain_levels,
            "translational_gap": "Not assessed without a reasoning model.",
            "mechanistic_chain": chain, "conflicts_explained": [],
            "certainty": certainty, "certainty_rationale": "Deterministic tally only.",
            "key_uncertainties": [], "reasoning_trace": None, "provider": None, "model": None,
            "notes": [], "mode": "deterministic",
        }
