"""
query_parser.py  (v25)
======================
Pull the population, intervention, comparator, outcome (PICO) and question type out of a question BEFORE searching,
so that every later step - query building, synonym expansion, screening, reranking, the verdict - knows what it is
looking for. Deterministic (no AI call); builds on evidence_identity.question_frame for the disease and endpoint
lexicons and adds what that frame lacked: an explicit comparator, the population phrase and the question type.

    parse("Does metformin compared with placebo reduce Ki-67 in women with breast cancer?")
    -> {"population": "women with breast cancer", "intervention": "metformin", "comparator": "placebo",
        "outcome": "Ki-67", "question_type": "therapy", "expected_direction": "decrease", ...}

Question types: therapy | prevention | etiology | diagnosis | prognosis | mechanism | prevalence | cure | safety |
comparison | general. Each type has a preferred study design (used by the reranker and the verdict).
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

PARSER_VERSION = "1.0"

TYPES = [
    ("cure", r"\b(?:cur(?:e|ed|es|able|ing)|eradicat\w*|permanent(?:ly)? (?:remission|resolution))\b"),
    ("diagnosis", r"\b(?:diagnos\w*|detect\w*|sensitivity|specificity|accuracy of|screening test|biomarker for|auc|roc)\b"),
    ("prognosis", r"\b(?:prognos\w*|predict\w*|life expectancy|outcome of patients|survival (?:of|in) patients with)\b"),
    ("prevalence", r"\b(?:prevalen\w*|incidence of|how common|how many people|frequency of|epidemiolog\w*)\b"),
    ("mechanism", r"\b(?:how does|how do|mechanism\w*|through (?:which|what)|by what|pathway\w*|mediat\w*|signal\w*|"
                  r"why does|molecular basis)\b"),
    ("safety", r"\b(?:safe|safety|side[- ]effects?|adverse|toxicit\w*|harm\w*|risk of (?:taking|using))\b"),
    ("etiology", r"\b(?:caus\w*|risk factors?|associated with|association|linked to|contribut\w* to|exposure to|"
                 r"increase the risk|raises? the risk)\b"),
    ("prevention", r"\b(?:prevent\w*|protect\w* against|lower the risk|reduce the risk|prophyla\w*)\b"),
    ("comparison", r"\b(?:compared (?:with|to)|versus|vs\.?|better than|worse than|or)\b"),
    ("therapy", r"\b(?:treat\w*|therap\w*|does .+ (?:reduce|increase|improve|lower|raise|change|affect|alter|slow|"
                r"inhibit|suppress|enhance|extend|shorten)|effect of|efficacy|effective)\b"),
]
PREFERRED_DESIGN = {
    "therapy": ["randomised controlled trial", "meta-analysis"], "prevention": ["randomised controlled trial", "cohort"],
    "cure": ["randomised controlled trial", "cohort"], "safety": ["randomised controlled trial", "cohort", "case-control"],
    "etiology": ["cohort", "case-control", "mendelian randomisation"], "diagnosis": ["diagnostic accuracy study"],
    "prognosis": ["cohort"], "prevalence": ["cross-sectional", "registry"],
    "mechanism": ["experimental (cell / animal)"], "comparison": ["randomised controlled trial", "meta-analysis"],
    "general": [],
}
_DOWN = r"(?:reduc\w*|decreas\w*|lower\w*|inhibit\w*|suppress\w*|slow\w*|shorten\w*|prevent\w*|attenuat\w*|block\w*|kill\w*)"
_UP = r"(?:increas\w*|rais\w*|improv\w*|enhanc\w*|extend\w*|prolong\w*|promot\w*|stimulat\w*|boost\w*|induc\w*|restor\w*)"
_CHANGE = rf"(?:{_DOWN}|{_UP}|chang\w*|affect\w*|alter\w*|influenc\w*|modif\w*|impact\w*)"
_COMP = re.compile(r"\b(?:compared (?:with|to)|versus|vs\.?|relative to|rather than|instead of|over)\s+"
                   r"((?:(?!\b(?:reduce|increase|improve|lower|raise|change|affect|alter|in|among|for|on|at)\b)[\w\-+/,. ]){2,60}?)"
                   r"(?=\s+(?:" + _CHANGE + r"|in|among|for|on|at)\b|[?.,]|$)", re.I)
_HEAD = (r"(?:adults?|children|infants?|neonates?|women|men|people|persons|patients|participants|subjects|humans?|mice|"
         r"rats?|dogs?|cats?|cattle|cows?|pigs?|zebrafish|plants?|seedlings?|cells?|elderly|pregnan\w+|smokers|athletes|"
         r"workers|volunteers|survivors|users|populations?|cohorts?|individuals|girls|boys|adolescents|veterans|"
         r"nurses|residents|inpatients|outpatients)")
_POP = re.compile(r"\b(?:in|among|for)\s+((?:[\w\-']+\s+){0,3}?" + _HEAD + r"\b[\w\-,' ]{0,60}?)"
                  r"(?=[?.]|$|\s+(?:compared|versus|vs|and how|who|treated))", re.I)
_ETIOLOGY = re.compile(r"^\s*(?:what|which (?:factors?|exposures?|genes?))\s+(?:causes?|leads? to|triggers?|drives?|"
                       r"increases? the risk of|raises? the risk of|contributes? to)\s+(.+?)\s*[?.]?$", re.I)
_EXPOSED_SUFFIX = re.compile(r"-(?:exposed|treated|induced|related|associated|dependent|based|containing)\b", re.I)
_AGENT_VERB = re.compile(rf"^\s*(?:does|do|did|can|could|will|would|is|are|might|may|should)\s+(?:the\s+|a\s+|an\s+)?"
                         rf"(.+?)\s+(?:(?:significantly|really|actually|still|also)\s+)?({_CHANGE})\b\s*(?:the\s+)?(.*)$", re.I)
_EFFECT_OF = re.compile(r"\b(?:effects?|impact|influence|role|efficacy|benefit)s?\s+of\s+(.+?)\s+(?:on|in|for|against)\s+(.+?)(?:[?.]|$)", re.I)


def _clean(s: Optional[str]) -> Optional[str]:
    if not s:
        return None
    s = re.sub(r"\s+", " ", s).strip(" ,.;?")
    s = re.sub(r"^(?:the|a|an)\s+", "", s, flags=re.I)
    return s or None


def question_type(q: str) -> str:
    low = q.lower()
    for name, rx in TYPES:
        if name == "comparison" and not re.search(r"\b(?:compared|versus|vs\.?|better than|worse than)\b", low):
            continue
        if re.search(rx, low):
            return name
    return "general"


def parse(question: str) -> Dict[str, Any]:
    """PICO + question type. Fields absent from the question are None (never guessed)."""
    from evidence_identity import question_frame
    frame = question_frame(question)
    q = frame["question"]
    out: Dict[str, Any] = {"question": q, "question_type": question_type(q), "population": None,
                           "intervention": None, "comparator": None, "outcome": None, "expected_direction": None,
                           "disease": frame.get("disease"), "domain": frame.get("domain"),
                           "organism": frame.get("organism_restriction"), "version": PARSER_VERSION}
    work = q
    m = _COMP.search(work)
    if m:
        out["comparator"] = _clean(m.group(1))
        work = work[:m.start()] + " " + work[m.end():]
    pm = _POP.search(work)
    if pm:
        out["population"] = _clean(pm.group(1))
        work = work[:pm.start()] + " " + work[pm.end():]
    et = _ETIOLOGY.match(work)
    am = None if et else _AGENT_VERB.match(work)
    if et:
        out["outcome"] = _clean(et.group(1))
        if out["population"]:
            agent = re.search(r"([\w\-]+)-(?:exposed|treated)\b", out["population"], re.I)
            if agent:
                out["intervention"] = agent.group(1)
        if out["question_type"] == "general":
            out["question_type"] = "etiology"
    elif am:
        out["intervention"] = _clean(am.group(1))
        verb = am.group(2).lower()
        out["outcome"] = _clean(am.group(3))
        if re.fullmatch(_DOWN, verb):
            out["expected_direction"] = "decrease"
        elif re.fullmatch(_UP, verb):
            out["expected_direction"] = "increase"
    else:
        em = _EFFECT_OF.search(work)
        if em:
            out["intervention"], out["outcome"] = _clean(em.group(1)), _clean(em.group(2))
    if not out["intervention"] and frame.get("exposures"):
        out["intervention"] = frame["exposures"][0]
    if out["intervention"]:
        out["intervention"] = _clean(_EXPOSED_SUFFIX.sub("", out["intervention"]))
    if not out["outcome"]:
        out["outcome"] = frame.get("endpoint_phrase") or (frame.get("endpoints") or [None])[0]
    if out["outcome"] and out["population"] and out["population"] in out["outcome"]:
        out["outcome"] = _clean(out["outcome"].replace(out["population"], ""))
    out["population_inferred"] = False
    if not out["population"] and frame.get("disease"):
        out["population"] = f"people with {frame['disease']}" if frame.get("domain") == "human_medicine" else frame["disease"]
        out["population_inferred"] = True
    if out["outcome"] and re.fullmatch(r"[a-z_]+", out["outcome"]) and "_" in out["outcome"]:
        out["outcome"] = out["outcome"].replace("_", " ")          # an internal family name ('clinical_response')
    if out["question_type"] == "general" and out["intervention"] and out["outcome"]:
        out["question_type"] = "therapy" if frame.get("domain") == "human_medicine" else "mechanism"
    # 'does X compared with Y reduce Z' is a therapy question that names its comparator
    out["is_comparison"] = bool(out["comparator"])
    if out["question_type"] == "comparison" and out["intervention"] and out["outcome"]:
        out["question_type"] = "therapy" if frame.get("domain") in ("human_medicine", None, "general_biology") else "mechanism"
    out["preferred_designs"] = PREFERRED_DESIGN.get(out["question_type"], [])
    out["missing"] = [k for k in ("population", "intervention", "comparator", "outcome") if not out[k]]
    return out


def search_concepts(parsed: Dict[str, Any]) -> List[Dict[str, Any]]:
    """The PICO elements as synonym-expanded concept groups, with a weight each (intervention and outcome matter most)."""
    import mesh_synonyms as ms
    weights = {"intervention": 1.0, "outcome": 0.9, "disease": 0.8, "population": 0.6, "comparator": 0.3}
    out = []
    for role, w in weights.items():
        txt = parsed.get(role)
        if role == "disease":
            # an inferred population ('people with vitiligo') is searched as the disease itself
            txt = (parsed.get("disease") or "").split(" / ")[0] if parsed.get("population_inferred") else None
        if not txt:
            continue
        for terms in ms.concepts(txt):
            if terms:
                out.append({"role": role, "weight": w, "terms": sorted(terms)})
    return out


def boolean_query(parsed: Dict[str, Any], max_terms: int = 6) -> str:
    """A PubMed / Europe PMC style query: each concept an OR group, concepts joined by AND (comparator left out -
    many papers do not name it in the abstract)."""
    groups = []
    for c in search_concepts(parsed):
        if c["role"] == "comparator" or (c["role"] == "population" and parsed.get("population_inferred")):
            continue
        if c["role"] == "population" and not set(c["terms"]) - {"people", "patients", "humans", "adults", "subjects",
                                                                  "participants", "persons", "individuals"}:
            continue                    # 'people' narrows nothing in a literature search
        terms = [t for t in c["terms"] if len(t) > 2][:max_terms]
        if terms:
            groups.append("(" + " OR ".join(f'"{t}"' if " " in t else t for t in terms) + ")")
    return " AND ".join(groups)
