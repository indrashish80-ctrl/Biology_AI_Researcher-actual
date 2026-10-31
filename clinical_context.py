"""
clinical_context.py
-------------------
Deterministic extraction of the CONTEXT a finding depends on. Without it, a polished
report can silently over-generalise.

  * outcome ladder        improvement < partial response < recovery/response < complete response
                          < remission < sustained remission < cure            (recovery != cure)
  * cure definition       what a question like "is X curable?" must be answered against
  * follow-up             how long after treatment the outcome was observed
  * comparator            none / unclear / placebo / vehicle / ... (uncontrolled = weaker inference)
  * sample size           n, or explicitly "not reported"
  * combination therapy   "A, B and C were given; recovery observed" is NOT evidence for A alone
  * inference type        randomised controlled > controlled > association > temporal observation > case
  * disease context       cause/induction, subtype, anatomical site, severity, disease duration
  * disease definition    what each study actually means by the disease term in the question
  * intervention profile  name -> type -> route -> formulation -> dose -> schedule -> components
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from stats_parsing import normalize_numeric_text, parse_sample_sizes

_R = lambda p: re.compile(p, re.I)  # noqa: E731

# --------------------------------------------------------------------------- question intent
CURE_Q = _R(r"\b(cur(?:e|es|ed|able|ability|ative)|incurable|eradicat\w*|permanent(?:ly)? (?:resolv|clear)\w*|"
            r"remission|reversible|reversal)\b")
_DISEASE_WORD = _R(r"\b([a-z][a-z\-]*(?:derma|itis|osis|oma|emia|aemia|pathy|plegia|algia|trophy|iasis|ism)|vitiligo|"
                   r"psoriasis|eczema|cancer|carcinoma|tumou?rs?|diabetes|obesity|asthma|malaria|tuberculosis|"
                   r"infection|disease|syndrome|disorder|alopecia|acne|dementia|depression|hypertension|"
                   r"leuk(?:a)?emia|lymphoma|sepsis|stroke)\b")
# words with a disease-like ending that name normal biology ('gravitropism', 'apoptosis', 'symbiosis', 'stoma')
_NOT_DISEASE = re.compile(r"(?:tropism|metabolism|mechanism|organism|morphism|mutualism|commensalism|catabolism|anabolism|"
                          r"parasitism|dimorphism|symbiosis|mitosis|meiosis|osmosis|homeostasis|cytosis|(?:apo|ro)ptosis|"
                          r"biosis|genesis|stoma|rhizoma|plasma|chiasma|neuroma(?=s)|diploma|aroma|coma(?=tose))$", re.I)
# a disease named after its describer or its place ('Chagas disease', "Crohn's disease", 'Lyme disease')
_EPONYM = re.compile(r"\b[A-Z][a-z]{2,}(?:['\u2019]s|s['\u2019])?\s+"
                     r"(?:disease|syndrome|disorder|sarcoma|lymphoma|palsy|chorea|ataxia|an(?:a)?emia|encephalopathy|"
                     r"dystrophy|lesions?|cirrhosis|fever)\b")
_Q_STOP = set("is are was were can could does do did the a an of in on for to and or with by how what which why "
              "be cured curable cure treatable treated effective work works there any".split())


def parse_question(question: str) -> Dict[str, Any]:
    q = question or ""
    m = re.search(r"\b(?:is|are)\s+(.+?)\s+(?:curable|treatable|reversible|preventable)\b|"
                  r"\bcan\s+(.+?)\s+be\s+(?:cured|reversed|treated|prevented)\b|\bcure\s+(?:for|of)\s+(.+?)[?.]?$", q, re.I)
    phrase = next((g for g in (m.groups() if m else []) if g), None)
    # a disease named after a person or place is ONE term: 'Chagas disease', "Parkinson's disease", 'Lyme disease' -
    # never the generic word 'disease' plus a stray 'Chagas' that then looks like a drug
    eponyms = [re.sub(r"\s+", " ", m.group(0)).lower() for m in _EPONYM.finditer(q)]
    generic = {w for e in eponyms for w in e.split()[-1:]}
    terms = eponyms + [w.lower() for w in _DISEASE_WORD.findall(q)
                       if not _NOT_DISEASE.search(w) and w.lower() not in generic]
    if phrase:
        # a word that is already part of an eponymous name is not a separate disease term
        parts = {w for e in eponyms for w in e.replace("'s", "").replace("\u2019s", "").split()}
        terms = [w for w in re.findall(r"[a-z][a-z\-]{3,}", phrase.lower())
                 if w not in _Q_STOP and w not in parts] + terms
    seen, disease_terms = set(), []
    for t in terms:
        if t not in seen:
            seen.add(t)
            disease_terms.append(t)
    return {"asks_cure": bool(CURE_Q.search(q)), "disease_terms": disease_terms[:3]}


# --------------------------------------------------------------------------- outcome ladder
OUTCOME_LADDER = [
    # level, key, label, pattern
    (0, "no_change", "no change / worsening", r"\bno (?:improvement|change|response)|worsen\w*|progress(?:ed|ion) of (?:the )?disease|relaps\w*|recurr\w*"),
    (1, "improvement", "improvement", r"\bimprov\w*|ameliorat\w*|better|reduc\w* (?:in )?(?:severity|lesion|symptom)|less severe|partial(?:ly)? improv\w*"),
    (2, "partial_response", "partial response / partial recovery",
     r"\bpartial (?:recovery|response|repigmentation|remission|resolution|healing|regression)|reduc\w* (?:in )?lesion size|regress\w*"),
    (3, "response", "recovery / response",
     r"\brecover\w*|respon(?:se|ded|der)s?\b|repigment\w*|heal\w*|resolution of symptoms|clinical(?:ly)? (?:better|response)"),
    (4, "complete_response", "complete response",
     r"\b(?:complete|total|full) (?:response|repigmentation|resolution|healing|recovery|clearance|regression)|disappear\w*|cleared"),
    (5, "remission", "remission", r"\bremission\b"),
    (6, "sustained_remission", "sustained remission (no recurrence on follow-up)",
     r"\b(?:sustained|durable|long[- ]term|lasting|persistent) (?:remission|response|repigmentation|resolution|recovery)|"
     r"no (?:recurrence|relapse)|recurrence[- ]free(?! (?:survival|interval))|relapse[- ]free(?! (?:survival|interval))|"
     r"did not (?:recur|relapse)"),          # 'recurrence-free SURVIVAL' is a time-to-event endpoint, not remission
    (7, "cure", "cure", r"\bcur(?:e|ed|es|ative)\b|eradicat\w*"),
]
CURE_MIN_FOLLOW_UP_DAYS = 365      # default operational threshold; disease-specific definitions may differ


def outcome_level(text: str) -> Dict[str, Any]:
    """Highest outcome rung literally stated in the text (no inference)."""
    best = {"level": None, "key": None, "label": "not stated"}
    for lvl, key, label, pat in OUTCOME_LADDER:
        if lvl == 0:
            continue
        for m in re.finditer(pat, text or "", re.I):
            before = (text or "")[max(0, m.start() - 30):m.start()]
            if re.search(r"\b(?:no|not|without|failed to|did not|lack of|absence of)\s+(?:\w+\s+){0,2}$", before, re.I):
                continue                       # "no complete response", "did not recover"
            best = {"level": lvl, "key": key, "label": label}
            break
    # a 'partial' qualifier caps unqualified recovery/repigmentation wording at partial response
    if best["level"] in (3, 4) and re.search(r"\bpartial(?:ly)?\b", text or "", re.I) and \
            not re.search(r"\b(?:complete|total|full)\b", text or "", re.I):
        lvl, key, label, _ = OUTCOME_LADDER[2]
        best = {"level": lvl, "key": key, "label": label}
    return best


def cure_definition(follow_up_days: int = CURE_MIN_FOLLOW_UP_DAYS) -> Dict[str, Any]:
    return {
        "definition": (f"Cure = complete response that is sustained without recurrence for at least "
                       f"{follow_up_days // 30} months of follow-up after treatment ends (operational default; "
                       f"disease-specific criteria may be stricter)."),
        "requires": ["complete response (not improvement / partial or unspecified 'recovery')",
                     "sustained remission with documented absence of recurrence",
                     f"follow-up of at least {follow_up_days // 30} months", "a defined, validated outcome measure"],
        "ladder": [label for _, _, label, _ in OUTCOME_LADDER[1:]],
        "min_follow_up_days": follow_up_days,
    }


def meets_cure(outcome_lvl: Optional[int], follow_up_days: Optional[float]) -> bool:
    return bool(outcome_lvl is not None and outcome_lvl >= 6 and follow_up_days and follow_up_days >= CURE_MIN_FOLLOW_UP_DAYS)


# --------------------------------------------------------------------------- follow-up, comparator, n
_FU = _R(r"(?:follow(?:ed)?[- ]up(?: period)?(?: of| for| at)?|after|for|over|within|at|during)\s+(?:a\s+)?(?:median\s+|mean\s+)?"
         r"(\d+(?:\.\d+)?)\s*(days?|d\b|weeks?|wks?|months?|mo\b|years?|yrs?)")
_UNIT_DAYS = {"d": 1, "day": 1, "week": 7, "wk": 7, "month": 30.44, "mo": 30.44, "year": 365.25, "yr": 365.25}


# '10-year event-free survival', '5-year overall survival', 'two-year relapse-free' = outcome measured at that time
_FU_RATE = _R(r"\b(\d+(?:\.\d+)?|one|two|three|four|five|six|seven|eight|nine|ten|fifteen|twenty)[- ](years?|months?)[- ]"
              r"(?:event-free|overall|relapse-free|disease-free|progression-free|leukemia-free|leukaemia-free|recurrence-free|"
              r"cumulative|treatment-free|remission|survival|cure|follow)")
_WORDNUM = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
            "fifteen": 15, "twenty": 20}


def follow_up_days(text: str) -> Optional[float]:
    vals = []
    t = normalize_numeric_text(text or "")
    for m in _FU.finditer(t):
        lead = m.group(0).split()[0].lower()
        before, after = t[max(0, m.start() - 80):m.start()], t[m.end():m.end() + 40]
        if lead in ("for", "over", "during") and not re.search(r"follow", before[-30:], re.I) and (
                re.search(r"\b(?:randomi[sz]ed|treated|received|receiving|given|administered|took|taking|fed|dosed|exposed|"
                          r"supplemented|assigned)\b", before, re.I)
                or re.search(r"^\s*(?:before|prior to|preceding)\s+(?:surgery|resection|operation|biopsy)", after, re.I)):
            continue                     # 'randomized to metformin for 2 weeks before surgery' = treatment time, not follow-up
        unit = m.group(2).lower().rstrip("s")
        vals.append(float(m.group(1)) * _UNIT_DAYS.get(unit, _UNIT_DAYS.get(unit[:2], 1)))
    for m in _FU_RATE.finditer(t):
        n = _WORDNUM.get(m.group(1).lower()) or float(m.group(1))
        unit = m.group(2).lower().rstrip("s")
        vals.append(float(n) * _UNIT_DAYS.get(unit, 1))
    return max(vals) if vals else None


_COMPARATORS = [("placebo", r"\bplacebo\b"), ("vehicle", r"\b(?:vehicle|sham)\b"),
                ("untreated control", r"\b(?:untreated|non-?treated|nontreated|mock[- ]treated)\b|no[- ]treatment (?:group|arm|control)|"
                                      r"\b0\s*mM (?:control|group)"),
                ("standard care", r"\b(?:standard (?:of )?care|usual care)\b"), ("non-users", r"\bnon-?users\b|never users"),
                # 'compared to the control group' is a control, not an active comparator
                ("control group", r"\bcontrol (?:group|arm|animals|patients|subjects|cells|mice|rats|condition)s?\b|"
                                  r"\b(?:compared (?:with|to)|than|versus|vs\.?)\s+(?:the |their |respective )?controls?\b"),
                ("other study groups", r"\bcompared (?:with|to) (?:all )?(?:the )?other (?:groups|arms|treatment groups)\b"),
                ("active comparator", r"\bcompared (?:with|to) (?!baseline|pre)\w+|versus\b|\bvs\.?\b")]
_NO_COMPARATOR = _R(r"\b(?:case report|case series|single[- ]arm|uncontrolled|open[- ]label single|before[- ]and[- ]after|"
                    r"pre[- ]?post|no control)\b")


def comparator(text: str, etype: Optional[str] = None) -> str:
    t = text or ""
    for name, rx in _COMPARATORS:
        if re.search(rx, t, re.I):
            return name
    if etype == "case_report" or _NO_COMPARATOR.search(t):
        return "none"
    return "not reported"


_PER_GROUP = re.compile(r"\b[nN]\s*=\s*(\d+)\s*(?:per|in each|for each|each|/)\s*(?:group|arm|condition|cohort)\b|"
                        r"\b(\d+)\s+(?:patients|participants|women|men|animals|mice|rats|subjects)\s+(?:per|in each)\s+(?:group|arm)\b")


def sample_size(text: str, extracted: Any = None, full_text: str = "") -> Optional[int]:
    """Total sample size. A per-group number ('n = 24 per group', also when it is only in the Methods = full_text) is
    never the total when the abstract states the total ('Participants (n = 57) were randomized'); an extracted value
    equal to the per-group number is corrected."""
    per_group = {int(a or b) for a, b in _PER_GROUP.findall(f"{text or ''} {full_text or ''}")}
    sizes = parse_sample_sizes(text or "")
    totals = [x for x in sizes if x not in per_group]
    try:
        if extracted not in (None, "", 0):
            ex = int(extracted)
            if ex in per_group and totals:
                return max(totals)
            if totals and ex not in sizes and not re.search(rf"(?<![\d.]){ex}(?![\d.])", f"{text or ''} {full_text or ''}"):
                return max(totals)       # the paper states its own total; an extracted number it never mentions loses
            return ex
    except (TypeError, ValueError):
        pass
    if totals:
        return max(totals)
    if sizes:
        return max(sizes)
    if re.search(r"\b(?:a|one|single) (?:\d+[- ]year[- ]old )?(?:\w+ )?(?:buffalo|cow|dog|cat|horse|patient|woman|man|boy|girl|case)\b", text or "", re.I):
        return 1
    return None


# --------------------------------------------------------------------------- combination therapy
_TREAT_LEAD = _R(r"(?:treated|treatment|administered|given|received|receiving|therapy|regimen|managed|supplemented)"
                 r"\s+(?:\w+\s+){0,3}?(?:with|using|of|by|consisting of|comprising)?\s*")
_GENERIC_COMPONENT = re.compile(r"^(?:the|a|an|and|was|were|is|are|recovery|observed|after|it|they|animal|patient|cow|buffalo)$", re.I)


_NOT_AGENT = re.compile(r"\b(?:was|were|is|are|be|been|measured|assessed|analy[sz]ed|evaluated|determined|observed|examined|"
                        r"detected|quantified|that|which|indicating|showing|figure|fig|table|panel|group|groups|control|controls|"
                        r"cells?|patients?|mice|rats|samples?|levels?|rate|index|indices|before|after|during|for|until|"
                        r"days?|weeks?|months?|hours?|h|min|surgery|but|significantly|increased|decreased|reduced|proteins?|genes?|dose|doses|dosage|time|times|manners?|dependent|concentrations?|"
                        r"mmol|µmol|umol|nmol|mol|mm|µm|um|nm|l|ml|µl)\b|-dependent|-\s*$|^\d", re.I)
_DOSE_ONLY = re.compile(r"^\s*(?:(?:\d+(?:[.,]\d+)?|to|or|from)\s*(?:mm|µm|um|nm|mmol(?:/| )?l|µmol(?:/| )?l|umol(?:/| )?l|nmol(?:/| )?l|mmol|µmol|umol|nmol|mg/kg|mg/day|mg|g|µg|ng/ml|µg/ml|mg/ml|%|iu)?\s*)+$", re.I)
_DOSE_STRIP = re.compile(r"(?:^|\s)\d+(?:[.,]\d+)?\s*(?:mm|µm|um|nm|mmol(?:/| )?l|µmol(?:/| )?l|umol(?:/| )?l|nmol(?:/| )?l|mmol|µmol|umol|nmol|mg/kg(?:/day)?|mg/day|mg|g|µg|ng/ml|µg/ml|mg/ml|%|iu)(?![\w])", re.I)
_PLACE_STRIP = re.compile(r"\b(?:in vitro|in vivo|in culture|alone|only|daily|orally|twice daily|once daily)\b", re.I)


def _agent(part: str) -> Optional[str]:
    """A regimen component that is an actual agent: 'metformin in vitro' -> 'metformin'; '25' / '0 mM' -> None (a dose
    in a series); 'proliferation' / 'MALAT1 expression were measured' / 'Figure 1 A' -> None (read-outs, clauses, labels)."""
    from semantic_audit import canonical_outcome
    p = (part or "").strip(" ,;:()[]")
    if not p or _DOSE_ONLY.match(p):
        return None
    p = _PLACE_STRIP.sub(" ", _DOSE_STRIP.sub(" ", p))
    p = re.sub(r"\s*\([^)]*$", "", p)                                   # 'endocrine therapy (HR' - unbalanced bracket
    p = re.sub(r"^(?:combined|including|such as|e\.g\.|i\.e\.|both|either|concurrent|adjuvant|neoadjuvant)\s+", "", p.strip(), flags=re.I)
    p = re.sub(r"\s+", " ", p).strip(" ,;:()[]-")
    # 'when tamoxifen', 'tamoxifen an', 'used chemotherapy': clause words around a name are not part of it
    for _ in range(3):
        p = re.sub(r"^(?:when|where|while|whereas|used|using|which|that|then|since|also|only|and|or|with|an?|the|in|of|to)\s+", "",
                   p, flags=re.I)
        p = re.sub(r"\s+(?:when|where|while|whereas|used|which|that|then|alone|an?|the|and|or|with|in|of|to|was|were|is|are)$", "",
                   p, flags=re.I)
    if re.fullmatch(r"(?:when|where|while|whereas|used|using|which|that|then|since|also|only|alone|combined|combination(?: regimen)?"
                    r"(?: of)?|regimen|modalit(?:y|ies)|two modalities)", p, re.I):
        return None
    if re.fullmatch(r"(?:the |a |an )?(?:standard|conventional|usual|current|other|several|various)?\s*(?:of\s+)?"
                    r"(?:care|therap(?:y|ies)|treatments?|regimens?|drugs?|agents?|medications?)", p, re.I):
        return None                    # a generic word, not a named agent
    if re.fullmatch(r"[A-Z]{2,5}\d{1,3}[A-Z]?", p) and p.lower() not in _DRUG_ABBR:
        return None                    # 'GRP78', 'CD16', 'AKT1': a gene / protein symbol, not a treatment
    if len(p) < 3 or len(p) > 40 or len(p.split()) > 4 or _NOT_AGENT.search(p) or _GENERIC_COMPONENT.match(p):
        return None
    if not canonical_outcome(p)["family"].startswith("other:"):
        return None                    # an outcome (proliferation, apoptosis, expression) is not a treatment
    if _NOT_TREATMENT.search(p):
        return None                    # 'according to the protocol', 'tumor stage', 'log-rank test', 'type 2 diabetes', 'there'
    return p


# words that make a list item a design / covariate / disease / statistics phrase, never a regimen component
_NOT_TREATMENT = re.compile(
    r"\b(?:according|protocol|guidelines?|stage|status|grade|age|sex|gender|bmi|body mass|test|tests|log-?rank|"
    r"analys[ie]s|ratio|hazard|odds|p|both|there|here|these|those|this|several|various|other|others|many|some|"
    r"pre-?clinical|respectively|diabet\w*|cancers?|"
    r"tumou?rs?|carcinomas?|diseases?|disorders?|syndromes?|obesity|type [12]|cd\d+\w*|receptors?|subtypes?|"
    r"in vitro|in vivo|vitro|vivo|untreated|vehicle|placebo|medium|media|single-agent|monotherapy|shrank|shrunk|grew|regressed|died|survived|"
    # a measurement procedure is not a treatment ('baseline biopsy + post-treatment surgical specimen')
    r"biops\w*|specimens?|samples?|sampling|post-?treatment|pre-?treatment|baseline|follow-?up|resection|core needle|"
    # dates and enrolment periods ('March 2013 + follow-up to October 2020')
    r"january|february|march|april|may|june|july|august|september|october|november|december|(?:19|20)\d\d|"
    # verbs / abstract nouns picked up from a sentence ('show significant', 'explores opportunities', 'resistance')
    r"shows?|showed|shown|significant\w*|explor\w*|opportunit\w*|resistan\w*|respon\w*|outcomes?|benefits?|effects?|"
    r"prognosis|survival|"
    # how a drug acts is not a second drug ('through indirect and direct mechanisms of action')
    r"mechanisms?|mechanistic|actions?|modes?|pathways?|direct|indirect|through|via|insulin-mediated|insulin-dependent)\b", re.I)


def treatment_components(text: str) -> List[str]:
    """Components of a regimen named together in one sentence, e.g.
    'treated with Curan BH, copper sulphate and mineral mixture along with supportive treatment'."""
    comps: List[str] = []
    text = re.sub(r"\bH\s*2\s*O\s*2\b", "H2O2", text or "")
    for sent in re.split(r"(?<=[.;])\s+", text):
        # 'combining a ketogenic diet and metformin', 'the ketogenic diet plus metformin group', 'co-treatment of BC cells
        # with metformin and flavone', 'MET-sensitized cells followed by cisplatin', 'metformin ... enhanced
        # tamoxifen-induced inhibition' - two agents acting together, however the sentence words it
        extra = []
        for rx in (r"\bcombin(?:ing|ed)\s+(?:a |an |the )?((?:[\w\-]+\s+)?[\w\-]+?)\s+(?:and|with|plus)\s+(?:a |an |the )?([\w\-]{3,})",
                   r"\b((?:[\w\-]+\s+)?[\w\-]+)\s+plus\s+([\w\-]{3,})",      # never ' + ' ('IFN-γ + NKT cells' are markers)
                   r"\bco-?treat\w*\s+(?:of\s+(?:[\w\-]+\s+){1,3}?)?with\s+([\w\-]{3,})\s+(?:and|plus)\s+([\w\-]{3,})",
                   r"\b([\w]{3,})[\-\u2010\u2011]sensiti[sz]ed\b[^.;]{0,80}?\bfollowed by\s+([\w\-]{3,})",
                   r"\b([\w\-]{3,})\s+(?:and|plus)\s+([\w\-]{3,})\s+(?:combination|co-?treatment|combined treatment)\b",
                   r"\bcombination of\b[^.;]{0,60}?\b((?:[\w\-]+\s+)?[\w\-]+)\s+(?:and|with|plus)\s+([\w\-]{3,})"):
            for cm2 in re.finditer(rx, sent, re.I):
                pair = []
                for x in cm2.groups():
                    x = re.sub(r"^(?:low[- ]carbohydrate|high[- ]fat)\s+", "", x.strip(), flags=re.I)
                    a = _agent(x)
                    if a and not re.search(r"\b(?:group|arm|cells?|mice|rats?|patients?)$", a, re.I):
                        pair.append(a)
                if len({a.lower() for a in pair}) >= 2:
                    extra.extend(pair)
        em = re.search(r"\b(?:enhanc|potentiat|augment|sensiti[sz])\w*\s+(?:the\s+)?(?:\w+\s+){0,3}?([A-Za-z][\w]{2,})[\-\u2010\u2011]"
                       r"(?:induced|mediated)\b", sent, re.I)
        if em:
            other = _agent(em.group(1))
            pre = [w for w in re.findall(r"[A-Za-z][\w\-]{2,}", sent[:em.start()])
                   if w.lower() not in ("either", "neither", "both", "treatment", "therapy", "gene", "deletion", "knockdown",
                                        "administration", "the", "and", "with")]
            first = next((a for a in (_agent(w) for w in pre) if a and not re.fullmatch(r"[A-Z0-9\-]{2,}", a)), None)
            if other and first and other.lower() != first.lower():
                extra.extend([first, other])
        if extra:
            comps.extend(extra)
            continue
        # 'When metformin was combined with tamoxifen', 'the combination of metformin and tamoxifen'
        cm = re.search(r"\b([A-Za-z][\w\-]{2,})\s+(?:(?:was|were|is|are)\s+)?(?:combined|co-?treated|co-?administered|"
                       r"in combination) with\s+([A-Za-z][\w\-]{2,})|"
                       r"\bcombination of\s+([A-Za-z][\w\-]{2,})\s+(?:and|with|plus)\s+([A-Za-z][\w\-]{2,})", sent, re.I)
        if cm:
            pair = [x for x in cm.groups() if x]
            agents = [a for a in (_agent(x) for x in pair) if a]
            if len({a.lower() for a in agents}) >= 2:
                comps.extend(agents)
                continue
        m = _TREAT_LEAD.search(sent)
        if not m:
            continue
        tail = sent[m.end():]
        tail = re.split(r"[.;]|\b(?:did|does|do|was|were|is|are|had|has|have|significantly|alter\w*|increas\w*|decreas\w*|reduc\w*|"
                        r"inhibit\w*|promot\w*|suppress\w*|improv\w*|but|whereas|while|however|for|after|until|resulted|leading|led|which|and then|with (?:complete|partial)|recovery|"
                        r"improvement|was observed|were observed)\b", tail, maxsplit=1, flags=re.I)[0]
        parts = re.split(r",\s*(?:and\s+)?|\s+and\s+|\s+along with\s+|\s+plus\s+|\s+together with\s+|\s+in combination with\s+|"
                         r"\s+combined with\s+|\s*\+\s*|\s+as well as\s+", tail, flags=re.I)
        parts = [re.sub(r"^\s*(?:with|of|by|the|a|an|oral|topical)\s+", "", p.strip(" ()"), flags=re.I) for p in parts]
        agents = []
        for p in parts:
            a = _agent(p)
            if a and a.lower() not in [x.lower() for x in agents]:
                agents.append(a)
        if len(agents) >= 2:                 # two DIFFERENT agents - not a dose series of one drug, not a list of read-outs
            comps.extend(agents)
    out, seen = [], set()
    for c in comps:
        k = _DRUG_ABBR.get(c.lower(), c.lower()).split()[-1]       # 'MET' and 'metformin' are one component
        if k not in seen:
            seen.add(k)
            out.append(c)
    return out


# common drug abbreviations in papers ('OHT' = 4-hydroxytamoxifen, the active form of tamoxifen)
_DRUG_ABBR = {"oht": "4-hydroxytamoxifen tamoxifen", "4-oht": "4-hydroxytamoxifen tamoxifen", "tam": "tamoxifen",
              "met": "metformin", "mf": "metformin", "dox": "doxorubicin", "5-fu": "fluorouracil", "atra": "all-trans retinoic acid",
              "ato": "arsenic trioxide", "ptx": "paclitaxel", "ddp": "cisplatin", "cddp": "cisplatin", "gem": "gemcitabine",
              "rapa": "rapamycin", "nac": "n-acetylcysteine", "e2": "estradiol", "egcg": "epigallocatechin gallate"}


_REFERS_TO_REGIMEN = re.compile(r"\b(?:the|this|both) (?:combination|combined treatment|co-?treatment|regimen|drug combination)\b|"
                                r"\bboth (?:drugs|agents|treatments)\b|\btogether\b", re.I)
_ALONE = re.compile(r"\b(?:alone or (?:together|in combination|combined)|either alone|each alone|as single agents?|"
                    r"(?:individually|separately) and in combination)\b", re.I)


def attribution(intervention: str, quote: str, context: str = "", abbr: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    """Does the evidence isolate THIS intervention, or only a regimen that contained it?"""
    comps = treatment_components(quote)
    if not comps and _REFERS_TO_REGIMEN.search(quote or ""):
        comps = treatment_components(context)          # 'the combination ...' refers back to the regimen named before
    if len(comps) < 2:
        return {"combination": False, "components": comps, "isolated": True}
    if _ALONE.search(quote or ""):
        # 'metformin and palbociclib alone or together inhibited ...': each agent was tested on its own as well
        return {"combination": True, "components": comps, "isolated": True, "named": comps}
    it = (intervention or "").lower()
    it_words = re.findall(r"[a-z][a-z0-9\-]{3,}", it)
    amap = {**_DRUG_ABBR, **{k.lower(): v.lower() for k, v in (abbr or {}).items()}}

    def is_named(c: str) -> bool:
        names = [c.lower()] + ([amap[c.lower()]] if c.lower() in amap else [])     # 'OHT' -> 4-hydroxytamoxifen
        return any(n in it or any(w in it for w in re.findall(r"[a-z][a-z0-9\-]{3,}", n))
                   or any(w in n for w in it_words) or (len(n) >= 3 and any(w.startswith(n) for w in it_words))
                   for n in names)
    named = [c for c in comps if is_named(c)]
    covers_all = len(named) >= len(comps) or re.search(r"combin|regimen|together|plus|\+|and", it) and len(named) >= 2
    return {"combination": True, "components": comps, "isolated": bool(covers_all), "named": named}


# --------------------------------------------------------------------------- inference strength
INFERENCE_RANK = ["case_observation", "temporal_observation", "statistical_association", "controlled_experiment",
                  "randomised_controlled"]
INFERENCE_LABEL = {"randomised_controlled": "randomised controlled comparison", "controlled_experiment": "controlled comparison",
                   "statistical_association": "statistical association", "temporal_observation": "observed after treatment (uncontrolled)",
                   "case_observation": "single-case / case-series observation"}
_RAND = _R(r"\brandomi[sz]\w*|randomly (?:assigned|allocated)")
_CONTROL = _R(r"\b(?:versus|vs\.?|compared (?:with|to)|relative to|than (?:the )?(?:control|placebo|vehicle|untreated)|"
              r"placebo|vehicle|sham|untreated|control (?:group|arm|animals|cells|mice|rats|patients|subjects)|wild[- ]type)\b")
# 'HR' / 'OR' only in capitals: with re.I, every 'or' in a sentence was read as an odds ratio
_ASSOC = _R(r"\b(?:associat\w*|correlat\w*|linked to|odds ratio|hazard ratio|relative risk|(?-i:\bHR\b)|(?-i:\bOR\b)|risk of)\b")
_TEMPORAL = _R(r"\b(?:observed after|after (?:\d+|two|three|one|several) (?:days?|weeks?|months?)|following (?:the )?treatment|"
               r"post[- ]?treatment|after treatment|at follow[- ]up|recovered after|improved after)\b")
CAUSAL_VERBS = _R(r"\b(?:caus\w*|led to|leads to|lead to|resulted in|results in|result in|cured?|cures|curing|is effective|"
                  r"was effective|were effective|efficac\w*|responsible for|due to|because of|attributable to|"
                  r"produced (?:a )?(?:recovery|cure|remission)|treats|successfully treat\w*)\b")


def _assoc_about_outcome(q: str, outcome_rx: Optional["re.Pattern"]) -> bool:
    """An association phrase that links the agent to THIS outcome. 'M2 polarization-associated cytokines' (a compound
    adjective) and '... increased apoptosis, which were associated with increased oxidative stress' / 'an
    antiproliferative effect associated with cell cycle arrest' (the outcome is named BEFORE the phrase and the phrase
    links it to something else) do not make the result an association."""
    for m in _ASSOC.finditer(q):
        if m.start() > 0 and q[m.start() - 1] in "-\u2010\u2011":
            continue
        if outcome_rx is not None and re.match(r"(?i)associat|correlat|linked", m.group(0)):
            before = outcome_rx.search(q[:m.start()])
            after = outcome_rx.search(q[m.end():])
            if before and not after:
                continue
        return True
    return False


def inference_type(quote: str, context: str, etype: Optional[str], outcome_rx: Optional["re.Pattern"] = None) -> str:
    q, c = quote or "", context or ""
    if etype == "case_report":
        return "case_observation"
    if _RAND.search(q) or (etype == "human_clinical" and _RAND.search(c) and _CONTROL.search(q + " " + c)):
        return "randomised_controlled"
    if _CONTROL.search(q):
        return "controlled_experiment"
    if _assoc_about_outcome(q, outcome_rx):
        return "statistical_association"
    if _TEMPORAL.search(q):
        return "temporal_observation"
    if etype in ("cell_culture", "organoid_ex_vivo", "animal", "experimental_disease_model", "plant", "microbial"):
        # a laboratory experiment compares treated with untreated by design; 'associated' elsewhere in the passage
        # does not turn a cell or mouse result into a statistical association
        return "controlled_experiment"
    if _CONTROL.search(c):
        return "controlled_experiment"
    if _ASSOC.search(c) or etype == "human_observational":
        return "statistical_association"
    if etype in ("cell_culture", "organoid_ex_vivo", "human_clinical"):
        return "controlled_experiment"      # lab assays / trials are compared with a control by design
    return "temporal_observation"


def causal_supported(statement: str, inference: str) -> Dict[str, Any]:
    causal = bool(CAUSAL_VERBS.search(statement or ""))
    ok = (not causal) or inference in ("randomised_controlled", "controlled_experiment")
    return {"causal_language": causal, "supported": ok, "inference": inference}


# --------------------------------------------------------------------------- disease context & definition
# subtype labels that mean a subtype on their own
_SUBTYPE_STANDALONE = _R(r"\b(non-?segmental|segmental|triple-negative|TNBC|ER[+-]|HER2[+-]|HER2-positive|HER2-negative|"
                         r"luminal [AB]|basal-like|hormone receptor-positive|stage [IV0-4]+[ABC]?|grade [1-4])(?![\w-])")
# words that are a subtype ONLY when they qualify the disease itself ('metastatic breast cancer', 'acute leukemia');
# 'type 2' next to 'diabetes', 'acute inflammation' or 'active cancer research' are not subtypes of the disease
_SUBTYPE_CONTEXT = r"(generali[sz]ed|locali[sz]ed|acral|acrofacial|focal|universal|mucosal|stable|active|progressive|chronic|acute|" \
                   r"early|advanced|refractory|recurrent|metastatic|invasive|inflammatory|type [12])"
_SUBTYPE = _R(r"\b(non-?segmental|segmental|generali[sz]ed|locali[sz]ed|acral|acrofacial|focal|universal|mucosal|"
              r"stable|active|progressive|chronic|acute|early|advanced|refractory|recurrent|metastatic|triple-negative|"
              r"ER[+-]|HER2[+-]?|type [12]|stage [IV0-4]+|grade [1-4])\b")       # kept for callers that scan single phrases
_SITE = _R(r"\b(face|facial|forehead|perioral|lips?|eyelids?|scalp|neck|trunk|back|chest|abdomen|hands?|fingers?|feet|"
           r"foot|toes?|acral|limbs?|arms?|legs?|genital|mucosa|oral(?!(?:ly)?\s+(?:\w+\s+)?(?:administ\w*|gavage|dos\w*|intake|"
           r"route|treatment|therap\w*|use|drugs?|agents?|medications?|antidiabetic\w*|hypoglyc\w*|metformin|supplement\w*|"
           r"glucose|tablets?|delivery))|skin|muzzle|udder|teats?|perineum)\b")       # 'oral administration' is a route
_SEVERITY = _R(r"\b(mild|moderate|severe|extensive|limited|widespread|\d+\s*% (?:body surface|BSA))\b")
# '-induced' is a disease induction only when a disease / model follows ('DMBA-induced mammary tumours', 'monobenzone-
# induced depigmentation'); 'metformin-induced apoptosis' and 'HMGB1-induced EMT' describe effects, not the disease
_DISEASE_NOUN = (r"(?:model|tumou?rs?(?!\s+(?:growth|volume|cells?|size|regression|inhibition|suppression|burden|weight|"
                 r"progression|response))|tumou?rigenesis|carcinomas?|cancers?|carcinogenesis|colitis|diabetes|vitiligo|leukod?erma|"
                 r"depigmentation|lesions?|disease|injury|arthritis|fibrosis|obesity|hypertension|inflammation of|dermatitis|"
                 r"hepatitis|nephropathy|neuropathy|asthma|mammary (?:tumou?rs?|carcinomas?|cancers?))")
_CAUSE = _R(r"\b([\w-]+?)[- ]induced\s+(?:[\w-]+\s+){0,2}?" + _DISEASE_NOUN + r"\b|\b(chemically[- ]induced|occupational|idiopathic|"
            r"autoimmune|post-?inflammatory|congenital|hereditary|genetic|drug-related|iatrogenic|infectious|nutritional|"
            r"copper[- ]deficien\w+|traumatic|contact|chemical)\b")
_CAUSE_BARE_NEEDS_DISEASE = {"chemical", "genetic", "contact", "occupational", "infectious", "nutritional", "traumatic",
                             "hereditary", "congenital", "autoimmune", "idiopathic"}
_QUALIFIERS = {"primary", "invasive", "resistant", "drug-resistant", "metastatic", "early", "advanced", "luminal", "inflammatory",
               "male", "female", "familial", "hereditary", "recurrent", "refractory", "triple-negative", "her2-positive",
               "her2-negative", "hormone", "receptor-positive", "operable", "locally", "ductal", "lobular", "mammary",
               "non-segmental", "segmental", "chemical", "occupational", "idiopathic", "acute", "chronic", "childhood",
               "myeloid", "lymphoblastic", "lymphocytic", "type",
               # organ / tissue of origin ('triple-negative breast cancer', 'colorectal cancer')
               "breast", "lung", "colorectal", "colon", "rectal", "prostate", "ovarian", "pancreatic", "gastric", "liver",
               "hepatocellular", "renal", "bladder", "cervical", "endometrial", "thyroid", "oral", "skin", "brain", "bone",
               "blood", "esophageal", "oesophageal", "head", "neck", "small-cell", "non-small-cell", "cutaneous", "uveal"}


_TRIAL_WORDS = re.compile(r"randomi[sz]\w*|placebo|trial|treatment|treated|control|study|intervention|allocated|assigned|"
                          r"metformin|drug|both|each|two|three|active|experimental|comparator|group|vs\.?|versus", re.I)


def _no_trial_arms(text: str) -> str:
    """'in the metformin arm', 'placebo arm', 'both arms': a TRIAL arm, never the anatomical arm."""
    def repl(m):
        around = text[max(0, m.start() - 45):m.end() + 45]
        return " " if _TRIAL_WORDS.search(around) else m.group(0)
    return re.sub(r"\barms?\b", repl, text or "", flags=re.I)


# v24 fix 6 - what may qualify a disease name.
# A real report printed "pmel17 vitiligo", "ccl8 vitiligo" and "skin vitiligo" as distinct disease definitions,
# then warned that the studies "do not share one disease definition" on the strength of its own noise.
# A gene or protein symbol is not a form of the disease, and the organ the disease is defined by adds nothing.
_MODEL_ID = re.compile(r"(?=.*\d)(?=.*[A-Za-z])[A-Za-z0-9-]{2,8}|C57BL[/-]?6[A-Za-z]*|BALB[/-]?c|NOD|SCID|NSG|nude", re.I)
# a gene / protein symbol: letters then digits (Pmel17, MED15, STAT3, IL6, CCL8)
_GENE_SYMBOL = re.compile(r"[A-Za-z]{2,7}-?\d{1,3}[A-Za-z]?", re.I)
_REDUNDANT_SITE = {"skin", "cutaneous", "epidermis", "epidermal", "dermal", "dermis", "body", "human", "patient",
                   "patients", "mouse", "murine", "advanced", "general"}


def disease_context(text: str, disease_terms: List[str]) -> Dict[str, Any]:
    t = text or ""
    variants, near, used = [], [], []
    for term in sorted(disease_terms or [], key=len, reverse=True):     # 'breast cancer' before 'cancer'
        for m in re.finditer(rf"((?:[\w-]+\s+){{0,2}}){re.escape(term)}\w*", t, re.I):
            span = (m.start() + len(m.group(1)), m.end())
            if any(a <= span[0] < b for a, b in used):
                continue                   # this 'cancer' is part of an already counted 'breast cancer'
            used.append(span)
            mods = [w for w in m.group(1).split() if w.lower() not in _Q_STOP]
            # keep only real qualifiers ('triple-negative', 'invasive') and model IDs ('4T1'); never numbers ('89'),
            # verbs or filler ('that promotes', 'combat', 'various')
            keep = [w for w in mods
                    if w.lower() not in _REDUNDANT_SITE
                    and (w.lower() in _QUALIFIERS or _SUBTYPE_STANDALONE.fullmatch(w)
                         or (_MODEL_ID.fullmatch(w) and not _GENE_SYMBOL.fullmatch(w)))]
            variants.append((" ".join(keep[-2:]) + " " + term).strip().lower())
            near.append(m.group(0))
    causes = []
    for m in _CAUSE.finditer(t):
        if m.group(1):
            if m.group(1).lower() not in ("drug", "cell", "self", "experimentally", "chemically"):
                causes.append(m.group(1) + "-induced")
        elif m.group(2):
            word = m.group(2).lower()
            # a bare cause word counts only when it qualifies the disease itself ('chemical leukoderma')
            if word in _CAUSE_BARE_NEEDS_DISEASE and not any(re.search(rf"\b{re.escape(word)}\s+(?:[\w-]+\s+)?{re.escape(term)}", t, re.I)
                                                             for term in (disease_terms or [])):
                continue
            causes.append(word)
    subtypes = [m.group(1) for m in _SUBTYPE_STANDALONE.finditer(t)]
    for term in disease_terms or []:
        subtypes += [m.group(1) for m in re.finditer(rf"\b{_SUBTYPE_CONTEXT}\s+(?:[\w-]+\s+)?{re.escape(term)}", t, re.I)]

    def uniq(xs):
        out = []
        for x in xs:
            if x and x.lower() not in [o.lower() for o in out]:
                out.append(x)
        return out
    return {"definition_variants": uniq(variants)[:5], "cause_or_induction": uniq(causes)[:4],
            "subtype": uniq(subtypes)[:4],
            "anatomical_site": uniq(m.group(0).lower() for m in _SITE.finditer(_no_trial_arms(" ".join(near) or t)))[:5],
            # severity only when it qualifies the disease / lesions ('mild vitiligo', 'severe disease'), never 'mild
            # hypoglycaemia' or 'moderate concentrations'
            "severity": uniq(m.group(1).lower() for m in re.finditer(
                r"\b(mild|moderate|severe|extensive|limited|widespread|\d+\s*% (?:body surface|BSA))\b(?:\s+(?:[\w-]+\s+){0,2}?"
                r"(?:disease|forms?|cases?|lesions?|symptoms?|stage|grade|illness|depigmentation|"
                + "|".join(re.escape(x) for x in (disease_terms or ["$^"])) + r")\b|(?<=BSA)|(?<=surface))", t, re.I))[:3],
            "disease_duration": (lambda m: m.group(0) if m else None)(
                re.search(r"(?:disease|lesion|symptom)s?\s+(?:duration|for|of)\s+(?:\w+\s+){0,2}\d+\s*(?:days?|weeks?|months?|years?)", t, re.I))}


def disease_definition_check(question_terms: List[str], rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """What each included study means by the disease in the question; flags heterogeneity."""
    per = []
    labels = set()
    for r in rows:
        dc = r.get("disease_context") or {}
        label = "; ".join(filter(None, [", ".join(dc.get("cause_or_induction") or []),
                                         ", ".join(dc.get("definition_variants") or [])])) or "not specified"
        per.append({"ref": r["ref"], "studied": label, "species": (r.get("model_profile") or {}).get("species"),
                    # an induced model is induced whatever the paper's design label ('MNU-induced mammary tumours' in a
                    # rat study), so the table never says 'mnu-induced' next to 'Induced model: no'
                    "induced": r.get("evidence_type") == "experimental_disease_model" or
                               any(str(x).endswith("-induced") for x in dc.get("cause_or_induction") or [])})
        labels.add(label)
    # a different CAUSE / induction (laser-induced vs idiopathic) or an induced model is a different disease
    # definition; subtype wording (HER2-positive, operable) is shown but is not by itself a different disease
    causes = {tuple(sorted((r.get("disease_context") or {}).get("cause_or_induction") or [])) for r in rows}
    hetero = len(causes) > 1 or any(p["induced"] for p in per)
    return {"question_terms": question_terms, "per_study": per, "heterogeneous": hetero,
            "statement": ("Included studies do not share one disease definition (" + " | ".join(sorted(labels)) +
                          "); results are reported per definition and must NOT be pooled as one disease.")
            if hetero else "No study uses a different cause or an induced form of the disease; subtype wording is listed per study."}


# --------------------------------------------------------------------------- intervention normalisation
_SYNONYMS = {"metformin hydrochloride": "metformin", "metformin hcl": "metformin", "acetylsalicylic acid": "aspirin",
             "sirolimus": "rapamycin", "cuso4": "copper sulphate", "copper sulfate": "copper sulphate",
             "vitamin d3": "cholecalciferol", "narrow-band uvb": "nb-uvb", "narrowband uvb": "nb-uvb"}
_TYPE = [("procedure", r"graft\w*|surgery|surgical|transplant\w*|excision|laser|phototherapy|uvb|uva|puva|nb-uvb|light therapy|"
                       r"radiotherapy|radiation|ablation|dermabrasion|microneedl\w*|cryotherapy|implant\w*|stimulation|device"),
         ("biological", r"\w+mab\b|antibod\w*|vaccin\w*|stem cell\w*|cell therap\w*|platelet-rich|prp\b|melanocyte (?:transplant|suspension)"),
         ("supplement / herbal", r"vitamin|mineral|supplement\w*|herbal|extract|ayurved\w*|homeopath\w*|curan|tonic|feed additive"),
         ("behavioural", r"exercise|diet\w*|training|lifestyle|counsel\w*")]
_ROUTE = _R(r"\b(oral(?:ly)?|topical(?:ly)?|intravenous(?:ly)?|i\.?v\.?|intramuscular(?:ly)?|i\.?m\.?|subcutaneous(?:ly)?|s\.?c\.?|"
            r"intralesional(?:ly)?|intradermal(?:ly)?|inhaled|transdermal|systemic(?:ally)?|local(?:ly)?|parenteral(?:ly)?|"
            r"in (?:the )?(?:feed|diet|water))\b")
_FORM = _R(r"\b(cream|ointment|gel|lotion|tablets?|capsules?|injection|solution|suspension|powder|bolus|drench|spray|patch)\b")
_SCHED = _R(r"\b(once daily|twice daily|three times daily|daily|weekly|monthly|bid|tid|qd|every \d+ (?:hours|days|weeks)|"
            r"for \d+ (?:days|weeks|months)|single dose)\b")


def _names_intervention(name: str, comps: List[str]) -> bool:
    """A regimen is this intervention's only when one of its components IS the intervention: 'chemotherapy + X' read
    from an introduction is not a metformin regimen. With no named intervention there is nothing to check."""
    toks = [t for t in re.findall(r"[a-z][a-z0-9\-]{2,}", (name or "").lower()) if t not in ("with", "plus", "and", "combination")]
    if not toks:
        return True
    for c in comps:
        cl = c.lower()
        full = _DRUG_ABBR.get(cl, cl)
        if any(t in full or full in t for t in toks):
            return True
    return False


def normalize_intervention(name: str, text: str = "") -> Dict[str, Any]:
    from evidence_quality import parse_exposures
    raw = (name or "").strip()
    low = raw.lower()
    canon = _SYNONYMS.get(low, low)
    comps = treatment_components(text) if text else []
    itype = next((t for t, rx in _TYPE if re.search(rx, f"{low}", re.I)), None) or \
        next((t for t, rx in _TYPE if re.search(rx, f"{low} {text[:300]}", re.I) and re.search(rx, low, re.I)), None) or \
        ("drug / compound" if low else "unspecified")
    route = _ROUTE.search(text or "")
    route_s = route.group(1).lower() if route else None
    systemic = itype == "drug / compound" and (route_s is None or not re.match(r"topical|local|intralesional|intradermal|transdermal", route_s))
    return {"name": raw, "canonical": canon, "type": itype, "route": route_s or "not reported",
            "formulation": (lambda m: m.group(1).lower() if m else "not reported")(_FORM.search(text or "")),
            "dose": ", ".join(e["raw"] for e in parse_exposures(text or "")[:2])
                    or (lambda m: m.group(0) if m else "not reported")(
                        re.search(r"(?<![\d.,])(?:\d{1,3}(?:,\d{3})+(?!\d)|\d+(?:\.\d+)?)\s*(?:mg|g|µg|mcg|IU|mL)\b", text or "")),
            "schedule": (lambda m: m.group(1).lower() if m else "not reported")(_SCHED.search(text or "")),
            "combination_components": comps if len(comps) >= 2 and _names_intervention(raw, comps) else [],
            "systemic_drug": bool(systemic)}
