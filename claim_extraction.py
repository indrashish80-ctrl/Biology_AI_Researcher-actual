"""
claim_extraction.py
-------------------
Extracts quote-grounded claims with per-paper quota balancing,
robust syntactic clause de-nesting, and passive-voice inversion resolvers.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

from paper_extraction import _clip, gather_parts
from stats_parsing import p_value_in_text, parse_p_values, quote_grounding

GROUNDING_THRESHOLD = 0.70
EXTRACTOR_VERSION = "7"   # 7 (v25): verbatim-quote guard, numbers check, second reader

DIRECTION_SYNONYMS = {
    "increase": "increase", "increases": "increase", "increasing": "increase",
    "promotes": "increase", "enhances": "increase", "elevates": "increase", "higher": "increase",
    "stimulated": "increase", "activates": "increase", "upregulates": "increase",
    "decrease": "decrease", "decreases": "decrease", "decreasing": "decrease",
    "inhibits": "decrease", "reduces": "decrease", "suppresses": "decrease", "lower": "decrease",
    "downregulates": "decrease", "attenuates": "decrease",
    "no_effect": "no_effect", "no effect": "no_effect", "neutral": "no_effect", "none": "no_effect",
    "mixed": "mixed", "unclear": "unclear",
}

STAT_SENTENCE_RE = re.compile(
    r"([^.?!;\n]*\b(?:p\s*[<=><]|increased|decreased|reduced|elevated|fold|inhibited|promoted|enhanced|suppressed)[^.?!;\n]*[.?!])",
    re.IGNORECASE,
)


def derive_default_subject(question: Optional[str]) -> Optional[str]:
    if not question:
        return None
    from evidence_identity import question_frame
    ex = question_frame(question).get("exposures") or []
    return ex[0] if ex else None


_SUBJECT_PATTERNS = [
    # Active treatment phrasing
    re.compile(r"\b(?:exposure to|exposed to|treatment with|treated with|treating with|administration of|supplementation with|"
               r"incubation with|co-?cultur\w* with|addition of|inhibition of|knock-?down of|overexpression of|deletion of|loss of)\s+"
               r"((?:[A-Za-z0-9][\w\-/()]*)(?:\s+(?!and\b|or\b|in\b|for\b|at\b|was\b|were\b|increased|decreased|reduced|"
               r"significantly\b|markedly\b|\w+ly\b|(?:improved|enhanced|promoted|inhibited|suppressed|induced)\b)"
               r"[A-Za-z0-9][\w\-/()]*){0,3})", re.I),
    # Passive-voice inversion: "Y was reduced/inhibited/promoted by X"
    re.compile(r"\b(?:was|were|is|are)\s+(?:\w+ly\s+)?(?:increased|decreased|reduced|inhibited|induced|enhanced|"
               r"suppressed|promoted|elevated|lowered|stimulated|blocked|attenuated)\s+by\s+(?:the\s+|an?\s+)?"
               r"([A-Za-z0-9][\w\-/()]+(?:\s+[A-Za-z0-9][\w\-/()]+){0,2})", re.I),
    # Followed by subordinate that-clauses
    re.compile(r"\bthat\s+(?:the\s+)?([A-Za-z0-9][\w\-/]*(?:\s+(?:and|plus|or)\s+[A-Za-z0-9][\w\-/]*)?)\s+(?:\w+ly\s+)?"
               r"(?:inhibited|increased|decreased|reduced|suppressed|enhanced|promoted|induced|augmented|attenuated|abolished|"
               r"blocked|improved)\b", re.I),
    # Agentive noun constructs
    re.compile(r"\b(?:ability|capacity|potential|propensity) of ([A-Za-z0-9][\w\-/]*(?:\s+[A-Za-z0-9][\w\-/]*)?) to "
               r"(?:activate|induce|promote|drive|stimulate|inhibit|suppress|increase|decrease|reduce)\b", re.I),
    # Treatment as head-noun: "Metformin treatment decreased..."
    re.compile(r"^(?:the\s+)?([A-Z0-9][\w\-/()]*(?:\s+[A-Za-z0-9][\w\-/()]*){0,2}?)\s+(?:treatment|exposure|administration|"
               r"supplementation|knock-?down|overexpression|deficiency|inhibition)\b"),
    # Direct active verbs: "Metformin significantly reduced..."
    re.compile(r"^(?:the\s+)?([A-Z0-9][\w\-/()]*(?:\s+(?!significantly|markedly|strongly|also|further|was|were|is|are)"
               r"[a-z][\w\-/()]*)?)\s+(?:significantly\s+|markedly\s+|strongly\s+)?(?:increased|decreased|reduced|"
               r"inhibited|induced|enhanced|suppressed|promoted|elevated|lowered|caused|triggered|was associated|were associated)\b"
               r"(?=\s+(?!and\b|or\b|in\b|by\b|after\b|at\b|with\b|over\b|during\b|from\b|to\b|significantly\b)[A-Za-z0-9])"),
    # Driver mediators: "X acts as a driver/regulator of Y"
    re.compile(r"^(?:the\s+)?((?:[\w\-/]+\s+){0,3}?[\w\-/]+(?:\s+(?:and|or)\s+[\w\-/]+)?)\s+(?:remains?|are|is|was|were|act as|acts as)\s+"
               r"(?:a\s+|the\s+)?(?:\w+\s+){0,2}?(?:drivers?|mediators?|regulators?|promoters?|inducers?|determinants?|causes?|"
               r"activators?|inhibitors?)\s+of\b", re.I),
    # Present-tense active verbs
    re.compile(r"^(?:the\s+)?([A-Za-z0-9][\w\-/]*(?:\s+(?:and|or)\s+[A-Za-z0-9][\w\-/]*)?)\s+(?:drives?|mediates?|promotes?|"
               r"induces?|activates?|stimulates?|triggers?)\s+(?!that\b)[a-z]", re.I),
    # Setting clauses followed by exposure
    re.compile(r",\s+(?:the\s+)?([A-Za-z0-9][\w\-/()]*(?:\s+(?!was\b|were\b)[a-z][\w\-/()]*)?)\s+(?:was|were)\s+"
               r"(?:significantly\s+|positively\s+|inversely\s+|independently\s+)?associated with\b"),
]

_CONNECTIVE = re.compile(r"^(?:(?:moreover|furthermore|additionally|in addition|however|importantly|notably|also|instead|thus|"
                         r"therefore|finally|interestingly|in agreement|similarly|consequently|in contrast|together|overall|"
                         r"results?|conclusions?|in the present study|in this study)\b[,:]?\s+)+", re.I)
_DOSE_PREFIX = re.compile(r"^(?:\d+(?:[.,]\d+)?\s*(?:mm|µm|um|nm|pm|mg/kg|mg/ml|µg/ml|ug/ml|ng/ml|mg|g|µg|%|mmol/l)\s+)+", re.I)
_ABBR_DEF = re.compile(r"\b([A-Za-z][\w\-]*(?:\s+[A-Za-z][\w\-]*){0,3})\s*\(([A-Z][A-Za-z0-9\-]{1,9})\)")


def abbreviations(text: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for m in _ABBR_DEF.finditer(text or ""):
        words, abbr = m.group(1).split(), m.group(2)
        for k in range(1, len(words) + 1):
            full = " ".join(words[-k:])
            if full[0].lower() == abbr[0].lower():
                out.setdefault(abbr, full)
                break
    return out


def _mentions(sentence: str, subject: Optional[str], abbr: Dict[str, str]) -> bool:
    if not subject:
        return False
    core = [w for w in re.findall(r"[A-Za-z][\w\-]{2,}", subject.lower()) if w not in ("dose", "group", "treatment", "therapy")]
    low = (sentence or "").lower()
    if core and all(re.search(rf"\b{re.escape(w)}", low) for w in core[:2]):
        return True
    return any(re.search(rf"\b{re.escape(a)}\b", sentence or "") for a, full in abbr.items()
               if core and core[0] in full.lower())


_JUNK_SUBJECT = re.compile(r"(?:since|because|although|though|when|where|while|whereas|thus|therefore|however|moreover|"
                           r"furthermore|additionally|also|this|these|that|those|it|they|we|here|there|combined|combination"
                           r"(?: regimen| therapy| treatment)?(?: of)?|regimen|treatment|therapy|the combination(?: of)?|"
                           r"both|either|neither|each|the|a|an|and|or)", re.I)


# v24 fix 2 - what can and cannot be an "intervention".
# A real report showed "While the ->", "pi3k pathway leads ->", "melanocytes ->" and the research question
# itself in the intervention column. None of these is something a study DID to anything.
_INSTRUMENT = re.compile(r"\b(?:incucyte|xcelligence|seahorse|operetta|opera phenix|cytation|clariostar|"
                         r"facs(?:calibur|aria)?|flow cytometer|flow cytometry|confocal(?: microscope)?|"
                         r"microscope|spectrophotometer|plate reader|elisa reader|qpcr machine|"
                         r"western blot(?:ting)?|immunohistochemistry|immunofluorescence|rna-?seq|mass spectrometry)\b", re.I)
# an instrument tacked onto a real agent: "CXCL10 using IncuCyte" -> "CXCL10"
_BY_INSTRUMENT = re.compile(r"\s*\b(?:using|by|via|with|on|measured (?:by|with|using)|assessed (?:by|with|using))\s+"
                            r"(?:the\s+|an?\s+)?(?=[^,;]*\b(?:incucyte|xcelligence|seahorse|facs|flow cytometr\w+|"
                            r"confocal|microscop\w+|spectrophotometer|plate reader|western blot\w*|immunohistochemistry|"
                            r"immunofluorescence|rna-?seq|mass spectrometry|qpcr|elisa)\b)[^,;]*", re.I)
# a sentence that was cut mid-clause
_FRAGMENT_START = re.compile(r"^(?:while|whereas|although|though|because|since|when|where|if|as|thus|therefore|"
                             r"however|moreover|furthermore|additionally|also|in contrast|by contrast|"
                             r"interestingly|notably|importantly|conversely|similarly|consistent with|"
                             r"in (?:addition|line|summary|conclusion|this study|the present))\b", re.I)
_FRAGMENT_END = re.compile(r"\b(?:leads?|leading|results?|resulting|causes?|causing|shows?|showed|showing|"
                           r"suggests?|indicates?|reveals?|demonstrates?|was|were|is|are|has|have|had|"
                           r"\w+ly)$", re.I)
# a biological process, pathway or cell type is the thing being studied, not the thing that was done
_NOT_AN_AGENT = re.compile(r"^(?:the\s+)?(?:[\w\-/]+\s+){0,4}"
                           r"(?:signal(?:l)?ing(?: pathway| cascade| axis)?|pathway|cascade|axis|signal(?:l)?ing)$", re.I)
_BARE_CELL = re.compile(r"^(?:the\s+)?(?:human |murine |primary |cultured |normal )?"
                        r"(?:melanocytes?|keratinocytes?|fibroblasts?|hepatocytes?|cardiomyocytes?|"
                        r"lymphocytes?|macrophages?|monocytes?|neutrophils?|t cells?|b cells?|cd8\+? t cells?|"
                        r"cells?|tissues?|skin|epidermis|serum|plasma|blood)$", re.I)


def strip_instrument(subj: Optional[str]) -> Optional[str]:
    """'CXCL10 using IncuCyte' -> 'CXCL10'. The instrument is how it was measured, not what was given."""
    if not subj:
        return subj
    out = _BY_INSTRUMENT.sub("", subj).strip(" ,;:-")
    return out or None


def implausible_subject(subj: Optional[str], question: Optional[str] = None) -> Optional[str]:
    """Why this text cannot be an intervention, or None when it can be. Plain words - it is printed."""
    t = (subj or "").strip()
    if not t:
        return None
    low = t.lower().strip(" .,;:")
    _norm = lambda x: re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]+", " ", (x or "").lower())).strip()
    if question:
        q, lw = _norm(question), _norm(t)
        if q and lw and (lw == q or (len(lw.split()) >= 5 and lw in q)):
            return "the research question itself was extracted as the intervention"
    if re.match(r"^(?:can|could|does|do|did|is|are|was|were|what|which|how|why|when|should|would|will)\b", low) \
            and len(low.split()) >= 4:
        return f"'{t}' is a question, not an intervention"
    if _FRAGMENT_START.search(low):
        return f"'{t}' is the start of a sentence, not an intervention"
    if len(low.split()) > 1 and _FRAGMENT_END.search(low):
        return f"'{t}' is a cut-off sentence fragment, not an intervention"
    if _INSTRUMENT.fullmatch(low) or (_INSTRUMENT.search(low) and not strip_instrument(t)):
        return f"'{t}' is an instrument or assay, not an intervention"
    if _NOT_AN_AGENT.match(low):
        return f"'{t}' is a pathway or process, not something that was administered"
    if _BARE_CELL.match(low):
        return f"'{t}' is the material studied, not an intervention applied to it"
    return None


def clean_subject(subj: Optional[str], abbr: Optional[Dict[str, str]] = None) -> Optional[str]:
    if not subj:
        return subj
    subj = _DOSE_PREFIX.sub("", subj.strip()).strip(" ,;:")
    words = subj.split()
    combo = bool(re.search(r"\bin combination\b|\bcombined\b|\bplus\b|\+|\band\b|\bwith\b", subj, re.I)) and \
        not re.search(r"\bcompared with\b", subj, re.I)
    for i, w in enumerate(words[1:], 1):
        if re.fullmatch(r"(?:\w+ly|among|who|which|that|per|before|after|resulted|induced|exhibited|showed|promoted|increased|"
                        r"decreased|reduced|inhibited|enhanced|suppressed|individuals|patients|\d[\w.]*|mg|mm|g|was|were|is|are|"
                        r"has|have|had|caused|led" + ("" if combo else r"|compared|comparing|versus|vs\.?|to|alone|only|in|on|"
                        r"at|for|group|groups|arm|users?") + ")", w, re.I):
            subj = " ".join(words[:i])
            break
    if abbr and subj in abbr:
        subj = abbr[subj]
    subj = strip_instrument(subj)                      # 'X using IncuCyte' -> 'X'   (v24 fix 2)
    if subj and _JUNK_SUBJECT.fullmatch(subj.strip()):
        return None
    return subj or None


_TREATED = re.compile(r"\b(?:were|was) (?:treated|managed|randomi[sz]ed to receive|randomi[sz]ed to|given) (?:with |to )?"
                      r"([A-Za-z0-9][\w\-/]*(?:\s+(?!and\b|or\b|in\b|for\b|at\b|until\b|before\b|after\b|or\b)[A-Za-z0-9][\w\-/]*){0,3})"
                      r"|\b(?:received|underwent) ([A-Za-z0-9][\w\-/]*(?:\s+(?!and\b|or\b|in\b|for\b|at\b)[A-Za-z0-9][\w\-/]*){0,3})"
                      r"|\b(?:stopped|discontinued|withdrew|ceased) ([A-Za-z0-9][\w\-/]*)", re.I)


def treatment_in(sentence: str) -> Optional[str]:
    m = _TREATED.search(sentence or "")
    if not m:
        return subject_from_sentence(sentence)
    if m.group(3):
        return f"{m.group(3)} discontinuation"
    return (m.group(1) or m.group(2) or "").strip(" ,.;") or None


def _strip_appositives(s: str) -> str:
    """Strips parenthetical or appositive non-restrictive descriptors that break head-verb binding."""
    s = re.sub(r"\([^)]*\)", "", s)
    s = re.sub(r",\s*(?:an?|the)\s+[\w\s\-]+,\s*", ", ", s)
    return s


def subject_from_sentence(sentence: str) -> Optional[str]:
    from evidence_identity import is_question_word_subject
    sentence = _CONNECTIVE.sub("", (sentence or "").strip())
    sentence = re.sub("[\u2010\u2011\u2012\u2013]", "-", sentence)
    clean_s = _strip_appositives(sentence)

    # 1. Setting clause de-nesting: "In metformin-treated cells, X increased..."
    lead = re.match(r"^(?:in|under|upon|with|after|following|during|among|in the presence of|in the absence of)\s+[^,]{3,70},\s+", sentence, re.I)
    if lead:
        rest = sentence[lead.end():]
        for rx in _SUBJECT_PATTERNS:
            m = rx.search(rest)
            if m:
                subj = re.sub(r"^(?:the|a|an|this|these|our)\s+", "", m.group(1).strip(" ,;:"), flags=re.I)
                if subj and not is_question_word_subject(subj) and not re.match(r"^(?:we|it|this|these|there)$", subj, re.I):
                    return subj

    # 2. Match primary syntactic patterns across sentence and appositive-stripped sentence
    for candidate_str in (sentence, clean_s):
        for rx in _SUBJECT_PATTERNS:
            m = rx.search(candidate_str)
            if m:
                subj = re.sub(r"^(?:the|a|an|this|these|our)\s+", "", m.group(1).strip(" ,;:"), flags=re.I)
                if subj and not is_question_word_subject(subj) and not re.match(r"^(?:we|it|this|these|there|results?|data|"
                                                                                   r"analysis|study|cells?|patients?)$", subj, re.I):
                    return subj
    return None


def prompt_claims(chunk_type: str, text: str, question: Optional[str], stats_context: str, n_max: int = 4) -> str:
    # v26 prompt caching: the long, identical instructions come FIRST and everything that changes per call (question,
    # statistics context, section text) LAST, so providers that cache a shared prompt prefix (Gemini implicit caching,
    # Groq / OpenAI-compatible prompt caching) can reuse it; cached tokens are counted in the run metrics
    focus = f'Focus on findings relevant to: "{question}"\n' if question else ""
    return f"""You are extracting biological claims from a peer-reviewed biology paper.

Extract up to {n_max} core biological findings (include null results).
Rules:
- "quote" must be a VERBATIM sentence fragment copied directly from the text proving the claim.
- Only extract results of THIS study (not findings the authors cite from other papers).
- "outcome" is the measured quantity itself, written positively: "cell growth", "apoptosis",
  "cell viability", "Ki-67", "tumour volume", "relative water content", "stomatal conductance" - never "growth inhibition" or "cytotoxicity".
- "direction" is what happened to THAT quantity: one of increase, decrease, no_effect, mixed, unclear.
  "increased growth inhibition" -> outcome "cell growth", direction "decrease".
  "increased cytotoxicity" -> outcome "cell viability", direction "decrease".
  Apoptosis and viability are different outcomes - never swap them.
- "subject" is the factor/exposure/drug NAMED IN THE TEXT (e.g. "benzene", "metformin", "silicon", "FLT3-ITD") - never words
  from the research question ("What are", "causes") and never empty. If the text names none, skip the finding.
- For passive constructions ("Y was significantly attenuated by X"), extract X as the subject and Y as the outcome.
- Extract only findings about the biological system in the research question; skip findings about unrelated organs or species.
- "associated with / correlated with" is an association: never turn it into "causes".
- Results relative to a comparator keep it: "metformin blunted the Ki-67 rise seen with placebo" -> outcome "Ki-67",
  direction "decrease", and the claim says "relative to placebo (smaller rise)". Never "Ki-67 increased".
- NULL results are findings: "did not significantly change Ki-67" -> direction "no_effect". Always extract them.
- If the result holds only in a subgroup ("in women with HOMA > 2.8", "in HER2-positive tumours"), keep that subgroup
  in "claim" and in "quote" - never generalise it to everyone.
- The subject must be the factor that CAUSED the change in the quote. Adjectives are never outcomes.
- Doses and time points of ONE drug ("0.5, 5, 25 mM", "24 to 96 hours", "time- and dose-dependent") are not a combination.
- "claim" must say the same thing as subject + outcome + direction, and name the model exactly as the quote does.
- Return ONLY valid JSON:
{{
  "claims": [
    {{
      "claim": "Summary of finding",
      "subject": "Manipulated factor, drug, or gene",
      "outcome": "Measured endpoint",
      "direction": "increase",
      "p_value": 0.05 or null,
      "quote": "verbatim text copied from chunk"
    }}
  ]
}}

{focus}Extracted statistics context: {stats_context}

SECTION: {chunk_type}
TEXT:
{text}"""


BATCH_PAPERS = 3
BATCH_CHAR_BUDGET = 9000
BATCH_MAX_CHARS_PER_PAPER = 3500


def prompt_claims_batch(items: List[Tuple[str, str]], question: Optional[str], n_max: int = 3) -> str:
    body = "\n\n".join(f"=== PAPER {tag} ===\n{text}" for tag, text in items)
    single = prompt_claims("Abstract", "", question, "", n_max=n_max)
    rules = single.split("- Return ONLY valid JSON:")[0]
    tags = ", ".join(tag for tag, _ in items)
    return (rules
            + f"There are {len(items)} separate papers below, tagged {tags}. Read each one on its own and extract up "
              f"to {n_max} findings FROM THAT PAPER ONLY. Never mix findings between papers, and never copy a quote "
              f"from one paper into another.\n"
            + "- Return ONLY valid JSON:\n"
            + '{\n  "papers": [\n    {"id": "' + items[0][0] + '", "claims": [\n'
            + '      {"claim": "...", "subject": "...", "outcome": "...", "direction": "increase",\n'
            + '       "p_value": 0.05, "quote": "verbatim text from THIS paper"}\n    ]}\n  ]\n}\n\n'
            + body)


def parse_claims_batch(response: str, tags: List[str]) -> Dict[str, List[Dict[str, Any]]]:
    from llm_json import parse_json_response
    out: Dict[str, List[Dict[str, Any]]] = {}
    try:
        obj = parse_json_response(response)
    except Exception:
        return out
    if not isinstance(obj, dict):
        return out
    papers = obj.get("papers")
    if not isinstance(papers, list):
        return out
    want = {str(t).strip().upper() for t in tags}
    for entry in papers:
        if not isinstance(entry, dict):
            continue
        tag = str(entry.get("id") or entry.get("paper") or "").strip().upper()
        if tag not in want:
            continue
        try:
            claims = validate_claims({"claims": entry.get("claims") or []}).get("claims", [])
        except Exception:
            claims = []
        if claims:
            out.setdefault(tag, []).extend(claims)
    return out


def batchable(parts: Dict[str, Any]) -> Optional[str]:
    if (parts or {}).get("basis") == "none":
        return None
    text = (parts.get("abstract") or "").strip()
    if not text or len(text) > BATCH_MAX_CHARS_PER_PAPER:
        return None
    if any((parts.get(k) or "").strip() for k in ("results", "results_discussion", "tables")):
        return None
    return text


def prefetch_claims(db: Any, router: Any, paper_ids: List[int], question: Optional[str],
                    batch_size: int = BATCH_PAPERS, log: Any = None) -> Dict[int, List[Dict[str, Any]]]:
    if not router or not paper_ids:
        return {}
    pending: List[Tuple[int, str]] = []
    for pid in paper_ids:
        try:
            paper = db.get_paper(pid)
            if not paper:
                continue
            text = batchable(gather_parts(db, paper))
        except Exception:
            text = None
        if text:
            pending.append((pid, text))
    if len(pending) < 2:
        return {}
    batches: List[List[Tuple[int, str]]] = []
    cur: List[Tuple[int, str]] = []
    used = 0
    for pid, text in pending:
        if cur and (len(cur) >= batch_size or used + len(text) > BATCH_CHAR_BUDGET):
            batches.append(cur)
            cur, used = [], 0
        cur.append((pid, text))
        used += len(text)
    if cur:
        batches.append(cur)
    batches = [b for b in batches if len(b) >= 2]
    if not batches:
        return {}
    prompts, tagmaps = [], []
    for b in batches:
        tagmap = {f"P{i + 1}": pid for i, (pid, _) in enumerate(b)}
        prompts.append(prompt_claims_batch([(f"P{i + 1}", t) for i, (_, t) in enumerate(b)], question))
        tagmaps.append(tagmap)
    try:
        responses = router.ask_parallel(prompts, max_workers=min(len(prompts), 4))
    except Exception:
        return {}
    out: Dict[int, List[Dict[str, Any]]] = {}
    for resp, tagmap in zip(responses, tagmaps):
        if not resp:
            continue
        for tag, claims in parse_claims_batch(resp, list(tagmap)).items():
            pid = tagmap.get(tag)
            if pid is not None and claims:
                out.setdefault(pid, []).extend(claims)
    if log:
        saved = sum(len(b) for b in batches) - len(batches)
        log(f"  -> Batched reading: {sum(len(b) for b in batches)} short papers in {len(batches)} AI call(s) "
            f"({saved} fewer calls); {len(out)} paper(s) answered.")
    return out


def validate_claims(obj: Any) -> Dict[str, Any]:
    if not isinstance(obj, dict):
        raise ValueError("Need JSON object")
    for k in ("synthesis", "result", "data", "output"):
        if k in obj and isinstance(obj[k], dict):
            obj = obj[k]
            break
    raw_claims = obj.get("claims") or obj.get("findings") or []
    if not isinstance(raw_claims, list):
        raise ValueError("Expected 'claims' list")

    clean = []
    for c in raw_claims:
        if not isinstance(c, dict):
            continue
        statement = c.get("claim") or c.get("statement") or c.get("finding")
        intervention = c.get("subject") or c.get("intervention") or c.get("treatment")
        if intervention and _JUNK_SUBJECT.fullmatch(str(intervention).strip()):
            intervention = None
        outcome = c.get("outcome") or c.get("measurement") or c.get("variable")
        quote = c.get("quote") or c.get("evidence")
        raw_dir = str(c.get("direction") or "unclear").lower().strip()
        direction = DIRECTION_SYNONYMS.get(raw_dir, "unclear")
        if not statement or not quote:
            continue
        clean.append({
            "statement": str(statement).strip(),
            "subject": str(intervention).strip() if intervention else None,
            "intervention": str(intervention).strip() if intervention else None,
            "outcome": str(outcome).strip() if outcome else None,
            "direction": direction,
            "quote": re.sub(r"^\d{1,3}\s+(?=[A-Z][a-z])", "", str(quote).strip().strip('"\'')),
            "p_value": c.get("p_value"),
            "evidence_origin": "this_study",
        })
    return {"claims": clean}


_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z(\[])|(?<=[.!?])\s+(?=\d{1,3}\s+[A-Z][a-z])|\n+")
_RESULT_WORD = re.compile(r"\bp\s*[<=>≤]|\b(?:increased|decreased|reduced|elevated|fold|inhibited|promoted|enhanced|suppressed|"
                          r"induced|lowered|lower|higher|smaller|larger|greater|fewer|less|more|declined|rose|improved|worsened|"
                          r"attenuated|abolished|blocked|impaired|downregulated|upregulated|associated with|odds ratio|relative risk|"
                          r"hazard ratio|significantly|achieved|maintained|attained|cured|remained|sustained)\b|"
                          r"\b\d+(?:\.\d+)?\s*%", re.I)

_HEADER_LINE = re.compile(r"^\s*\[[^\]\n]{1,300}\]\s*$", re.M)


def all_sentences(text: str) -> List[str]:
    text = _HEADER_LINE.sub("\n", text or "")
    out = []
    for s in _SENT_SPLIT.split(text):
        s = s.strip()
        if s and len(s) > 3:
            out.append(s if s[-1] in ".!?" else s + ".")
    return out


def result_sentences(text: str) -> List[str]:
    out = []
    for s in all_sentences(text):
        if _RESULT_WORD.search(s):
            out.append(s if s[-1] in ".!?" else s + ".")
    return out


def extract_deterministic_claims(text: str, default_subject: Optional[str] = None, max_claims: int = 5) -> List[Dict[str, Any]]:
    from semantic_audit import canonical_outcome, direction_in_text, FAMILY_LABEL, FAMILY_RX as _FRX
    claims = []
    seen = set()
    abbr = abbreviations(text)
    sents = all_sentences(text)
    for i, sentence in enumerate(sents):
        if not _RESULT_WORD.search(sentence):
            continue
        if len(sentence) < 30 or len(sentence) > 300 or sentence in seen:
            continue
        oinfo = canonical_outcome(sentence)
        if oinfo["family"].startswith("other:"):
            continue
        d = direction_in_text(sentence, oinfo, oinfo.get("matched") or "")
        direction = d.get("direction")
        if direction not in ("increase", "decrease", "no_effect"):
            continue
        outcome = oinfo.get("matched") or oinfo["label"]
        inverse_word = any(f == oinfo["family"] and neg.search(oinfo.get("matched") or "") for f, _, neg in _FRX)
        if oinfo["polarity"] < 0 or inverse_word:
            outcome = {"viability": "cell viability", "patient_survival": "survival",
                       "tumour_growth": "tumour growth"}.get(oinfo["family"], oinfo["label"])

        subject = clean_subject(subject_from_sentence(sentence), abbr)
        if not subject and i > 0 and oinfo["family"] in ("clinical_response", "patient_survival", "recurrence_incidence",
                                                          "proliferation", "tumour_growth", "viability", "apoptosis"):
            subject = clean_subject(treatment_in(sents[i - 1]), abbr)
        if not subject and default_subject:
            prev = sents[i - 1] if i > 0 else ""
            follows = re.match(r"^(?:this|these|such|it|its|they|their|both|the (?:effect|treatment|drug|compound))\b", sentence, re.I)
            if _mentions(sentence, default_subject, abbr) or (follows and _mentions(prev, default_subject, abbr)):
                subject = default_subject
        if not subject:
            continue
        p_vals = parse_p_values(sentence)
        p_val = p_vals[0]["value"] if p_vals else None
        quote = sentence.strip()
        seen.add(sentence)
        fam_label = FAMILY_LABEL.get(oinfo["family"], outcome)
        shown = fam_label if fam_label.lower() == str(outcome).lower() else f"{fam_label} ({outcome})"
        occurred = oinfo["family"] == "clinical_response" and direction == "increase" and not re.search(
            r"\b(?:increas|higher|greater|more|improv|enhanc)\w*", sentence, re.I)
        rate = re.search(r"\d+(?:\.\d+)?\s*%", sentence)
        if direction == "no_effect":
            statement = f"{subject}: no significant effect on {shown}."
        elif occurred:
            statement = f"{subject}: {outcome} reported" + (f" ({rate.group(0)})" if rate else "") + "."
        else:
            statement = f"{subject}: {shown} {'increased' if direction == 'increase' else 'decreased'}."
        if d.get("relative_to_control"):
            cmp_ = re.search(r"\b(placebo|control|vehicle|untreated|standard care)\b", sentence, re.I)
            statement = statement.rstrip(".") + (f" relative to {cmp_.group(1).lower()} (a smaller rise than in the "
                                                 f"{cmp_.group(1).lower()} group, not a fall from baseline)."
                                                 if cmp_ else " relative to the comparator (a smaller rise, not a fall from baseline).")

        claims.append({
            "statement": statement,
            "subject": subject,
            "intervention": subject,
            "outcome": outcome,
            "direction": direction,
            "quote": quote,
            "p_value": p_val,
            "evidence_origin": "this_study",
            "grounded": True,
            "grounding_score": 1.0,
            "deterministic_fallback": True,
        })
        if len(claims) >= max_claims:
            break
    return claims


def _as_int(v: Any) -> Optional[int]:
    if v is None or isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        return int(v) if v > 0 else None
    m = re.search(r"\d[\d,]*", str(v))
    return int(m.group(0).replace(",", "")) if m else None


def verify_claim(c: Dict[str, Any], source: str, study: Dict[str, Any]) -> Dict[str, Any]:
    flags: List[str] = []
    quote = (c.get("quote") or "").strip().strip('"\'')
    g = quote_grounding(quote, source)
    c["grounding_score"] = round(g, 2)
    c["grounded"] = g >= GROUNDING_THRESHOLD or c.get("deterministic_fallback", False)

    if not c["grounded"]:
        flags.append(f"quote not grounded in source text (score={g:.2f})")

    parsed_p = parse_p_values(quote)
    p = c.get("p_value")
    comp = None
    if p is not None:
        try:
            p = float(p)
        except (TypeError, ValueError):
            p = None

    if parsed_p:
        first = parsed_p[0]
        if p is None or abs(first["value"] - p) > 1e-12:
            p, comp = first["value"], first["comparator"]
    elif p is not None and not p_value_in_text(p, source):
        p, comp = None, None

    c["p_value"], c["p_comparator"] = p, comp
    # v25 principle 2: effect size, interval and sample size are parsed by CODE from the verbatim quote; the model's
    # numbers are only kept when the quote itself carries them (quote_guard already cleared any that it does not)
    try:
        from meta_analysis import extract_effect
        eff = extract_effect(quote)
        if eff:
            c["effect_size"], c["confidence_interval"] = eff["estimate"], [eff["lo"], eff["hi"]]
            c["effect_kind"], c["ci_level"] = eff["kind"], eff["level"]
    except Exception:
        pass
    if not c.get("sample_size"):
        try:
            from stats_parsing import parse_sample_sizes
            ns = parse_sample_sizes(quote)
            if ns:
                c["sample_size"] = max(ns)
        except Exception:
            pass
    for key, src_key in (("organism", "organism"), ("sample_size", "sample_size")):
        if not c.get(key) and study.get(src_key):
            c[key] = study[src_key]
    c["sample_size"] = _as_int(c.get("sample_size"))
    if not c.get("population"):
        c["population"] = study.get("tissue_or_cell_type") or c.get("organism")

    if p is not None and comp in ("<", "<=", "=") and p < 0.05 and c["direction"] == "no_effect":
        flags.append("direction 'no_effect' conflicts with p < 0.05")
        c["direction"] = "unclear"
    if p is not None and comp in ("=", ">", ">=") and p >= 0.05 and c["direction"] in ("increase", "decrease"):
        flags.append("directional claim lacks statistical significance (p >= 0.05)")
        c["nonsignificant_direction"] = True

    c["study_design"] = study.get("study_type")
    c["flags"] = flags
    return c


def get_evidence_chunks(parts: Dict[str, Any], max_paras: int = 4) -> List[Tuple[str, str]]:
    chunks = []
    if parts.get("abstract"):
        chunks.append(("Abstract", parts["abstract"]))
    body = "\n\n".join(x for x in (parts.get("results", ""), parts.get("discussion", "")) if x)
    paras = [p.strip() for p in body.split("\n\n") if len(p.strip()) > 80]
    for p in paras[:max_paras]:
        chunks.append(("Results Paragraph", p))
    return chunks


GROUP_CHARS = 7000


def group_chunks(chunks: List[Tuple[str, str]], budget: int = GROUP_CHARS) -> List[Tuple[str, str, int]]:
    groups: List[List[Tuple[str, str]]] = []
    size = 0
    for c_type, c_text in chunks:
        t = _clip(c_text, 3000)[0]
        if groups and size + len(t) <= budget:
            groups[-1].append((c_type, t))
            size += len(t)
        else:
            groups.append([(c_type, t)])
            size = len(t)
    out = []
    for g in groups:
        label = " + ".join(dict.fromkeys(c for c, _ in g))
        text = "\n\n".join(f"[{c}]\n{t}" for c, t in g) if len(g) > 1 else g[0][1]
        out.append((label, text, min(8, 4 * len(g))))
    return out


# ---------------------------------------------------------------- v25 guard statistics (per process, per paper)
import threading as _threading
GUARD_STATS: Dict[int, Dict[str, Any]] = {}
_GUARD_LOCK = _threading.Lock()


def record_guard_stats(paper_id: int, proposed: int, kept: int, dropped: List[Dict[str, Any]],
                       reader: Optional[Dict[str, Any]], basis: Optional[str]) -> None:
    with _GUARD_LOCK:
        GUARD_STATS[paper_id] = {
            "proposed": proposed, "kept": kept, "dropped": len(dropped), "basis": basis,
            "drop_reasons": [str(d.get("drop_reason")) for d in dropped][:20],
            "dropped_quotes": [str(d.get("quote") or "")[:200] for d in dropped][:10],
            "second_reader": ({k: v for k, v in reader.items() if k != "missed_by_first"} |
                              {"missed_by_first": len(reader.get("missed_by_first") or [])}) if reader else None}


def guard_summary(paper_ids: Optional[List[int]] = None) -> Dict[str, Any]:
    """Run-level totals of the quote / number guard and the second reader (for the dossier)."""
    with _GUARD_LOCK:
        rows = {k: v for k, v in GUARD_STATS.items() if paper_ids is None or k in set(paper_ids)}
    tot = {"papers": len(rows), "proposed": 0, "kept": 0, "dropped": 0, "agree": 0, "disagree": 0, "not_found": 0,
           "missed_by_first": 0, "by_basis": {}, "drop_reasons": {}}
    for v in rows.values():
        for k in ("proposed", "kept", "dropped"):
            tot[k] += v.get(k) or 0
        b = v.get("basis") or "unknown"
        tot["by_basis"][b] = tot["by_basis"].get(b, 0) + 1
        for r in v.get("drop_reasons") or []:
            key = r.split(":")[0]
            tot["drop_reasons"][key] = tot["drop_reasons"].get(key, 0) + 1
        sr = v.get("second_reader") or {}
        for k in ("agree", "disagree", "not_found", "missed_by_first"):
            tot[k] += int(sr.get(k) or 0)
    judged = tot["agree"] + tot["disagree"]
    tot["reader_agreement"] = round(tot["agree"] / judged, 3) if judged else None
    tot["drop_rate"] = round(tot["dropped"] / tot["proposed"], 3) if tot["proposed"] else None
    return tot


def text_basis(db: Any, paper_id: int, parts: Dict[str, Any]) -> str:
    """How this paper's text was obtained: 'jats_xml' | 'full_text_pdf' | 'abstract' (principle 4)."""
    try:
        import fulltext_archive
        b = fulltext_archive.basis_for(db, paper_id)
        if b:
            return b
    except Exception:
        pass
    if parts.get("basis") != "full_text":
        return "abstract"
    try:
        secs = db.get_sections(paper_id)
        if any(s.get("section_type") == "article_meta" for s in secs) or db.get_tables(paper_id):
            return "jats_xml"
    except Exception:
        pass
    return "full_text_pdf"


class ClaimExtractor:
    def __init__(self, db: Any, router: Any = None, providers: Optional[List[str]] = None,
                 prefetched: Optional[Dict[int, List[Dict[str, Any]]]] = None):
        self.db = db
        self.router = router
        self.providers = providers
        self.prefetched = prefetched or {}

    def extract(
        self,
        paper_id: int,
        research_question_id: Optional[int] = None,
        question: Optional[str] = None,
        force: bool = False,
        max_claims: int = 7,
    ) -> Dict[str, Any]:
        paper = self.db.get_paper(paper_id)
        if not paper:
            raise KeyError(f"Paper {paper_id} not found")

        mode_flag = f"extractor:{EXTRACTOR_VERSION}:{'ai' if self.router else 'rules'}"
        existing = [c for c in self.db.claims_for_paper(paper_id) if c.get("research_question_id") == research_question_id]
        if existing and not force:
            ok_flags = {f"extractor:{EXTRACTOR_VERSION}:ai"} | (set() if self.router else {f"extractor:{EXTRACTOR_VERSION}:rules"})
            current = all(ok_flags & set(c.get("flags") or []) for c in existing)
            if current:
                return {"paper_id": paper_id, "skipped": True, "claims": len(existing)}
            force = True

        parts = gather_parts(self.db, paper)
        if parts["basis"] == "none":
            return {"paper_id": paper_id, "status": "no_text", "claims": 0}

        # v25: the source is built with a section map, so every verbatim quote also knows which section it came from
        import quote_guard as qg
        source_all, section_at = qg.section_locator([
            ("abstract", parts.get("abstract")), ("results", parts.get("results")),
            ("results_discussion", parts.get("results_discussion")), ("other", parts.get("other")),
            ("table", parts.get("tables")), ("discussion", parts.get("discussion"))])
        basis = text_basis(self.db, paper_id, parts)
        raw = self.db.get_extraction_raw(paper_id)
        study = dict(raw.get("study") or {})
        # v26: question-aware passage selection (every results / discussion / table paragraph is scored; whole
        # paragraphs, original order, abstract always kept) instead of the first four paragraphs
        try:
            import passage_select
            chunks = passage_select.select(question, parts)
        except Exception:
            chunks = get_evidence_chunks(parts)
        all_claims = []
        from_library = False

        pre = (getattr(self, "prefetched", None) or {}).get(paper_id)
        # v26: a paper the library already read (question-independent pre-read) is not sent to the AI again: its
        # findings for THIS question are selected by code, then pass the same guard as any other finding
        lib_findings = None
        if not pre and not force:
            try:
                import preread
                lib_findings = preread.lookup(paper, question)
            except Exception:
                lib_findings = None
        # v25: per-paper extraction cache (paper id + extractor version + exact text + question): a paper already read
        # for this question is not sent to the AI again; the code checks below still run on the cached output
        import extraction_cache as xc
        _xkey = xc.make_key(paper, "claims", EXTRACTOR_VERSION, "\n".join(t for _, t in chunks), question)
        cached = xc.get(_xkey) if (self.router and not pre and not force) else None
        if pre:
            all_claims.extend(pre)
        elif lib_findings:
            all_claims.extend(lib_findings)
            from_library = True
        elif cached:
            all_claims.extend(cached)
        elif self.router:
            prompts = [prompt_claims(c_type, c_text, question, "", n_max=n) for c_type, c_text, n in group_chunks(chunks)]
            responses = self.router.ask_parallel(prompts, max_workers=len(prompts) or 1)
            for r in responses:
                if not r:
                    continue
                try:
                    from llm_json import parse_json_response
                    parsed = validate_claims(parse_json_response(r))
                    all_claims.extend(parsed.get("claims", []))
                except Exception:
                    continue
            if all_claims:
                xc.put(_xkey, all_claims)

        if not all_claims:
            subj = derive_default_subject(question)
            all_claims.extend(extract_deterministic_claims(source_all, default_subject=subj, max_claims=max_claims))

        # v25 principle 1: code, not the model, checks every quote is verbatim and every number is in it.
        # A finding whose quote is not in the paper, or that states a number the paper does not contain, is DROPPED.
        guarded, dropped = [], []
        for c in all_claims:
            g = qg.guard_claim(c, source_all, basis, section_at)
            (guarded if g["keep"] else dropped).append(g["claim"] if g["keep"] else {**g["claim"], "drop_reason": g["reason"]})
        # v25 principle 3: a second, independent reader (different provider / different prompt) reads the same text
        reader = None
        lib_read = from_library and all(c.get("second_reader") in ("agree", "disagree", "not_found") for c in guarded)
        if lib_read:
            # the pre-read already had its second reading: reuse its agreement instead of reading again
            reader = {"from_library": True, "agree": sum(1 for c in guarded if c.get("second_reader") == "agree"),
                      "disagree": sum(1 for c in guarded if c.get("second_reader") == "disagree")}
            for c in guarded:
                c["extraction_confidence"] = qg.source_confidence(
                    basis, section_at((c.get("quote_span") or [0])[0]), c.get("quote_mode") or "exact", c.get("second_reader"),
                    any(str(f).startswith("numbers_corrected") for f in c.get("flags") or []))
        if self.router and guarded and not lib_read and not any(c.get("deterministic_fallback") for c in guarded):
            import independent_reader as ir
            if ir.enabled(bool(getattr(self, "quick", False))):
                try:
                    texts = [t for _, t, _ in group_chunks(chunks)]
                    second = ir.second_read(self.router, texts, question, source_all, basis)
                    reader = ir.compare(guarded, second)
                    pol = ir.policy()
                    for c in guarded:
                        c["extraction_confidence"] = qg.source_confidence(
                            basis, section_at((c.get("quote_span") or [0])[0]), c.get("quote_mode") or "exact",
                            c.get("second_reader"), any(str(f).startswith("numbers_corrected") for f in c.get("flags") or []))
                    if pol == "drop":
                        dropped += [{**c, "drop_reason": "reader_disagreement"} for c in guarded if c.get("second_reader") == "disagree"]
                        guarded = [c for c in guarded if c.get("second_reader") != "disagree"]
                    elif pol == "keep":
                        for c in guarded:
                            c["flags"] = [f for f in c.get("flags") or [] if not str(f).startswith("reader_disagreement")]
                    if pol == "quarantine":
                        try:
                            import review_store
                            for c in guarded:
                                if c.get("second_reader") == "disagree":
                                    review_store.queue_for_review(self.db, paper_id, research_question_id, c,
                                                                  "independent readers disagree")
                        except Exception:
                            pass
                    if reader.get("missed_by_first"):
                        try:
                            import review_store
                            for m in reader["missed_by_first"][:5]:
                                review_store.queue_for_review(self.db, paper_id, research_question_id, m,
                                                              "second reader only - not in the first reading")
                        except Exception:
                            pass
                except Exception as exc:
                    reader = {"error": f"{type(exc).__name__}: {str(exc)[:120]}"}
        record_guard_stats(paper_id, len(all_claims), len(guarded), dropped, reader, basis)
        all_claims = guarded

        if force:
            self.db.delete_claims_for_paper(paper_id, research_question_id)

        seen = set()
        saved = 0
        for c in all_claims:
            c = verify_claim(c, source_all, study)
            c["flags"] = list(c.get("flags") or []) + [mode_flag if not c.get("deterministic_fallback")
                                                       else f"extractor:{EXTRACTOR_VERSION}:rules"]
            key = (str(c.get("intervention")).lower(), str(c.get("outcome")).lower(), c.get("direction"))
            if key in seen:
                continue
            seen.add(key)
            self.db.add_claim_ext(research_question_id, paper_id, c)
            saved += 1
            if saved >= max_claims:
                break

        return {"paper_id": paper_id, "status": "ok", "claims": saved}