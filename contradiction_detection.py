"""
contradiction_detection.py
--------------------------
Cross-study disagreement detection with a CONTEXT COMPARATOR.

Opposite arrows are not enough. Before two verified claims are called a
TRUE_CONTRADICTION they are compared on every dimension that can legitimately
change a biological result:

    species . model . cell line . disease subtype . metabolic state . dose .
    exposure duration . comparator . co-treatment . endpoint . experimental conditions

    TRUE_CONTRADICTION       comparable on every dimension that is known, and the critical
                             ones (species, model, endpoint) are known and identical
    CONTEXT_DIFFERENCE       at least one dimension differs - the difference is a candidate
                             explanation, not a conflict
    UNRESOLVED_DISAGREEMENT  opposite results, but comparability cannot be established
                             (critical information missing, or related-but-different endpoints)

Every stored verdict carries the full comparison table and a plain-language reason
saying WHY the studies are (or are not) comparable.
Only VERIFIED claims enter (db.verified_claims), compared on CANONICAL endpoint and
direction (so "growth inhibition UP" and "growth DOWN" agree).
"""

from __future__ import annotations

import re
from collections import defaultdict
from itertools import combinations
from typing import Any, Dict, List, Optional, Set

from semantic_audit import RELATED_OPPOSED, canonical_outcome
from stats_parsing import parse_dose, parse_duration_days

DIRECTIONAL = ("increase", "decrease", "no_effect")
VERDICTS = ("TRUE_CONTRADICTION", "CONTEXT_DIFFERENCE", "UNRESOLVED_DISAGREEMENT")

STOP_WORDS = {
    "the", "of", "and", "in", "on", "to", "a", "an", "with", "by", "for", "effect",
    "effects", "treatment", "treated", "application", "applied", "concentration", "exposure",
    "response", "responses", "during", "after", "via", "through", "associated", "cell", "cells",
}


def tokens(s: Optional[str]) -> Set[str]:
    if not s:
        return set()
    words = re.findall(r"[a-zA-Z0-9\-]+", s.lower())
    return {w for w in words if w not in STOP_WORDS and len(w) > 2}


def containment(a: Set[str], b: Set[str]) -> float:
    return len(a & b) / min(len(a), len(b)) if a and b else 0.0


COMBINATION_RE = re.compile(
    r"\+|\bplus\b|\bcombin\w*|\bco-?(?:treat|administ)\w*|\band\b|\bwith\b|/", re.IGNORECASE)


def is_combination(s: Optional[str]) -> bool:
    return bool(s and COMBINATION_RE.search(s))


def _canon(c: Dict[str, Any]) -> Dict[str, Any]:
    fam = c.get("canonical_outcome")
    info = canonical_outcome(c.get("measurement") or c.get("outcome") or c.get("statement"))
    return {"family": fam or info["family"], "target": info.get("target")}


_GENE_TOKEN = re.compile(r"\b(?:cyclin [A-E]\d?|p\d{2}|[A-Z][A-Z0-9]{1,7}(?:-[A-Z0-9]{1,3})?)\b")
_NOT_TARGET = {"PBCCS", "PBCC", "MCF7", "MCF-7", "MFI", "NKT", "IFN", "CD107A", "PD-1", "RNA", "DNA", "MRNA", "QPCR", "EMT",
               "T", "MET", "BC", "TNBC", "HER2"}


def _target_of(c: Dict[str, Any]) -> Optional[str]:
    """The gene / protein whose expression or activity the claim measures, from its outcome then its quote."""
    for src in (c.get("measurement") or c.get("outcome") or "", c.get("quote") or ""):
        for m in _GENE_TOKEN.finditer(src):
            tok = m.group(0)
            if tok.upper() not in _NOT_TARGET and not tok.isdigit():
                return tok.upper().replace("-", "")
    return None


def same_topic(a: Dict[str, Any], b: Dict[str, Any], threshold: float = 0.40) -> bool:
    """Same intervention (monotherapy never grouped with combinations) and same - or directly
    related - canonical endpoint."""
    ia = a.get("intervention") or a.get("subject")
    ib = b.get("intervention") or b.get("subject")
    if is_combination(ia) != is_combination(ib):
        return False
    if containment(tokens(ia), tokens(ib)) < threshold:
        return False
    ca, cb = _canon(a), _canon(b)
    oa, ob = ca["family"].startswith("other:"), cb["family"].startswith("other:")
    if oa != ob:
        return False                  # a recognised endpoint is never compared with an unrecognised one
    if oa and ob:
        return containment(tokens(a.get("measurement") or a.get("outcome")),
                           tokens(b.get("measurement") or b.get("outcome"))) >= max(threshold, 0.6)
    if ca["family"] == cb["family"]:
        if ca["family"] in ("expression", "signalling"):
            # 'expression' / 'activation' are generic: cyclin D1 DOWN and MIP1a UP are different read-outs. Only the
            # SAME named gene/protein is comparable.
            ta = _target_of(a)
            tb = _target_of(b)
            return bool(ta and tb and ta == tb)
        return (ca["target"] or "") == (cb["target"] or "") or not (ca["target"] and cb["target"])
    return frozenset((ca["family"], cb["family"])) in RELATED_OPPOSED


def effective_direction(c: Dict[str, Any]) -> str:
    """Canonical direction; directional claims without significance count as 'unclear'."""
    if c.get("nonsignificant_direction"):
        return "unclear"
    return c.get("canonical_direction") or c.get("effect_direction") or c.get("direction") or "unclear"


def group_claims(claims: List[Dict[str, Any]], threshold: float = 0.40) -> List[List[Dict[str, Any]]]:
    parent = list(range(len(claims)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i, j in combinations(range(len(claims)), 2):
        if same_topic(claims[i], claims[j], threshold):
            parent[find(i)] = find(j)
    buckets: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for i, c in enumerate(claims):
        buckets[find(i)].append(c)
    return sorted(buckets.values(), key=lambda g: -len(g))


# --------------------------------------------------------------------------- context extraction
_SPECIES = [("human", r"\b(patients?|humans?|women|men|participants?|homo sapiens|people)\b"),
            ("mouse", r"\b(mice|mouse|murine|mus musculus)\b"), ("rat", r"\b(rats?|rattus)\b"),
            ("zebrafish", r"\bzebrafish\b"), ("fly", r"\bdrosophila\b"), ("worm", r"\bc\. ?elegans\b"),
            ("pig", r"\b(pigs?|porcine)\b"), ("dog", r"\b(dogs?|canine)\b"), ("primate", r"\b(macaques?|primates?)\b"),
            ("yeast", r"\b(yeast|saccharomyces)\b"), ("e. coli", r"\b(e\. ?coli|escherichia)\b"),
            ("arabidopsis", r"\barabidopsis\b"), ("rice", r"\b(rice|oryza)\b"), ("maize", r"\b(maize|zea mays)\b")]
_HUMAN_CELLS = re.compile(r"\b(MCF-?7|MDA-MB-\d+|T-?47D|HeLa|HEK-?293|A549|HepG2|HCT-?116|LNCaP|PC-?3|SK-BR-3|BT-?474|"
                          r"SH-SY5Y|U-?87|Caco-?2|THP-1|Jurkat|human\s+\w*\s*cells?)\b", re.I)
_CELL_LINES = re.compile(r"\b(MCF-?7|MDA-MB-\d+|T-?47D|HeLa|HEK-?293\w*|A549|HepG2|HCT-?116|SK-BR-3|BT-?474|4T1|LNCaP|PC-?3|"
                         r"SH-SY5Y|U-?87|Caco-?2|THP-1|Jurkat|3T3|RAW ?264\.7)\b", re.I)
_SUBTYPES = [("ER+", r"\b(ER\+|ER-positive|oestrogen receptor-positive|estrogen receptor-positive|hormone receptor-positive|HR\+)"),
             ("ER-", r"\b(ER-negative|ER-(?!\w)|estrogen receptor-negative|oestrogen receptor-negative)"),
             ("HER2+", r"\b(HER2\+|HER2-positive|ERBB2-positive|HER2-overexpressing)"),
             ("TNBC", r"\b(triple-negative|TNBC|basal-like)"), ("luminal", r"\bluminal [AB]\b")]
# 'diabetic' only as a POPULATION: 'non-diabetic', 'anti-diabetic drug', 'a diabetes medication' are not diabetic patients
_METABOLIC = [("non-diabetic", r"\bnon-?diabet\w*"),
              ("diabetic", r"(?<!non-)(?<!non)(?<!anti-)(?<!anti)\b(diabet\w*|type 2 diabetes|T2DM?)\b"
                           r"(?!\s+(?:drugs?|medications?|agents?|treatments?|therap\w*|prevention|management))"),
              ("obese/high-fat", r"\b(obes\w*|high-fat diet|HFD|diet-induced obes\w*)\b"), ("lean", r"\b(lean|normal[- ]weight)\b"),
              ("insulin-resistant", r"\b(insulin[- ]resist\w*|hyperinsulin\w*)\b"),
              ("high glucose", r"\b(high[- ]glucose|hyperglyc\w*)\b"), ("low glucose", r"\b(low[- ]glucose|glucose[- ]depriv\w*|glucose[- ]free)\b"),
              ("normoglycaemic", r"\bnormoglyc\w*")]
_COMPARATORS = [("placebo", r"\bplacebo\b"), ("vehicle", r"\b(vehicle|PBS|saline|DMSO)\b"), ("untreated", r"\buntreated\b"),
                ("standard care", r"\b(standard (?:of )?care|usual care)\b"),
                ("non-users", r"\b(non-?users|never users|unexposed)\b"),
                ("active comparator", r"\b(sulfonylurea\w*|insulin users)\b")]
_CONDITIONS = [("hypoxia", r"\bhypoxi\w*"), ("normoxia", r"\bnormoxi\w*"), ("serum-free", r"\bserum[- ](?:free|starv\w*)"),
               ("3D culture", r"\b(3D|three-dimensional|spheroid\w*)\b"), ("2D culture", r"\b(2D|monolayer)\b")]


_NO_DIABETES = re.compile(r"\b(?:without|no|not having|free of|absence of)\s+(?:known\s+|a history of\s+|pre-?existing\s+|"
                          r"type (?:2|II)\s+)?diabet\w*", re.I)


_TREATS_DIABETES = re.compile(r"\b(?:used|approved|prescribed|first-line|drug|medication|therapy|treatment)\s+(?:\w+\s+){0,3}?"
                              r"(?:for|to treat|in the treatment of|of)\s+(?:type (?:2|II)\s+)?diabet\w*", re.I)
_LESION_STAGE = [("premalignant / in situ", r"\b(premalignan\w*|pre-?neoplas\w*|atypical (?:ductal |lobular )?hyperplasia|"
                                             r"in situ|DCIS|LCIS|intraepithelial neoplasia)\b"),
                 ("invasive", r"\b(invasive|infiltrating|operable|early(?:-stage)? breast cancer|metastatic|advanced)\b")]


def _set_from(patterns, text: str) -> Set[str]:
    text = text or ""
    if patterns is _METABOLIC:
        text = _NO_DIABETES.sub(" non-diabetic ", text)          # 'women without diabetes' is non-diabetic, not diabetic
        text = _TREATS_DIABETES.sub(" ", text)                    # 'a drug used to treat type 2 diabetes' names no population
    return {name for name, rx in patterns if re.search(rx, text, re.I)}


def _canon_subtypes(raw: Set[str]) -> Set[str]:
    """'tnbc', 'triple-negative', 'TNBC' -> {'TNBC'}; 'her2-positive', 'HER2+' -> {'HER2+'} (names compared, not spellings)."""
    out: Set[str] = set()
    for x in raw:
        hit = _set_from(_SUBTYPES, x)
        out |= hit if hit else {x.lower()}
    return out


def claim_context(c: Dict[str, Any], paper_text: str = "") -> Dict[str, Any]:
    """Context of ONE claim. Local text (quote + passage) wins; paper-level text (title/abstract)
    fills disease subtype and metabolic state, which are usually stated once per paper."""
    local = f"{c.get('quote') or ''} {c.get('source_context') or ''} {c.get('statement') or ''}"
    paper = f"{c.get('paper_title') or ''} {paper_text or ''}"
    model = c.get("claim_model") or ""
    species = _set_from(_SPECIES, local)
    if model in ("cell_culture", "organoid_ex_vivo") and _HUMAN_CELLS.search(local):
        species = {"human"}
    elif model == "animal":
        species = species - {"human"} or species
    if not species:
        species = _set_from(_SPECIES, f"{c.get('organism') or ''} {paper}")
    dose = parse_dose(c.get("dose")) or parse_dose(c.get("quote")) or parse_dose(c.get("source_context"))
    dur = parse_duration_days(c.get("duration")) or parse_duration_days(c.get("quote"))
    partner = None
    # a co-treatment must be named as one ('in combination with X', 'X plus Y'); the '+' in marker notation
    # ('IFN-γ + NKT cells', 'PD-1 +', 'CD8+') is a cell phenotype, not a second drug
    from clinical_context import _agent
    m = re.search(r"\b(?:in combination with|combined with|plus|co-?treat\w* with|together with)\s+([A-Za-z0-9][\w\-]*(?:\s+[A-Za-z][\w\-]*)?)",
                  local, re.I)
    agent = _agent(m.group(1)) if m else None
    if agent or is_combination(c.get("intervention")):
        partner = (agent or "combination").lower()
    canon = _canon(c)
    from clinical_context import disease_context, follow_up_days, normalize_intervention, outcome_level
    dctx = disease_context(f"{local} {paper}", [])
    iprof = normalize_intervention(c.get("intervention") or "", local)
    comps = (c.get("semantic_audit") or {}).get("combination_components") or iprof["combination_components"]
    return {
        "intervention": {"canonical": iprof["canonical"], "route": iprof["route"], "type": iprof["type"],
                         "components": sorted(x.lower() for x in comps)},
        "disease_induction": set(dctx["cause_or_induction"]),
        "outcome_definition": outcome_level(c.get("quote") or "").get("key"),
        "anatomical_site": set(dctx["anatomical_site"]),
        "severity": set(dctx["severity"]),
        "follow_up_days": follow_up_days(f"{c.get('quote') or ''} {c.get('source_context') or ''}"),
        "species": species, "model": model or None, "cell_line": {x.upper().replace("-", "") for x in _CELL_LINES.findall(local)},
        "subtype": _canon_subtypes((_set_from(_SUBTYPES, local) or _set_from(_SUBTYPES, paper)) | set(dctx["subtype"]) |
                                   {x for x in (((c.get("semantic_audit") or {}).get("identity") or {}).get("disease_subtype") or [])
                                    if x != "not specified"}),
        "evidence_role": (c.get("semantic_audit") or {}).get("evidence_role"),
        "metabolic_state": _set_from(_METABOLIC, local) or _set_from(_METABOLIC, paper),
        # premalignant / in-situ lesions and invasive cancer are different diseases for a proliferation endpoint
        "lesion_stage": _set_from(_LESION_STAGE, c.get("quote") or "") or _set_from(_LESION_STAGE, local),
        "dose": dose, "duration_days": dur,
        "comparator": _set_from(_COMPARATORS, local),
        "co_treatment": partner, "endpoint": canon,
        "conditions": _set_from(_CONDITIONS, local),
    }


def _fmt(v: Any) -> str:
    if v is None or v == set() or v == "":
        return "not reported"
    if isinstance(v, set):
        return ", ".join(sorted(v))
    if isinstance(v, dict) and "value" in v:
        return f"{v['value']:g} {v['unit']}"
    if isinstance(v, dict) and "family" in v:
        return v["family"].replace("_", " ") + (f" ({v['target']})" if v.get("target") else "")
    if isinstance(v, float):
        return f"{v:g} days"
    return str(v).replace("_", " ")


def compare_contexts(xa: Dict[str, Any], xb: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows = []

    def add(dim, a, b, status, note=""):
        rows.append({"dimension": dim, "a": _fmt(a), "b": _fmt(b), "status": status, "note": note})

    def set_dim(dim, a, b):
        if not a or not b:
            add(dim, a, b, "unknown")
        elif a == b:
            add(dim, a, b, "same")
        elif a & b:
            # 'TNBC' vs 'ER+, HER2+, TNBC': only partly the same population / model - not comparable as 'same'
            add(dim, a, b, "different", f"only partly overlapping (shared: {', '.join(sorted(a & b))})")
        else:
            add(dim, a, b, "different")

    ia, ib = xa["intervention"], xb["intervention"]
    if ia["components"] != ib["components"]:
        add("intervention", " + ".join(ia["components"]) or ia["canonical"], " + ".join(ib["components"]) or ib["canonical"],
            "different", "single agent vs regimen, or different regimens")
    elif ia["route"] != "not reported" and ib["route"] != "not reported" and ia["route"] != ib["route"]:
        add("intervention", f"{ia['canonical']} ({ia['route']})", f"{ib['canonical']} ({ib['route']})", "different", "different route")
    else:
        add("intervention", ia["canonical"], ib["canonical"], "same")
    set_dim("species", xa["species"], xb["species"])
    if not xa["model"] or not xb["model"]:
        add("model", xa["model"], xb["model"], "unknown")
    else:
        add("model", xa["model"], xb["model"], "same" if xa["model"] == xb["model"] else "different")
    if xa.get("evidence_role") and xb.get("evidence_role") and xa["evidence_role"] != xb["evidence_role"]:
        add("evidence_role", xa["evidence_role"], xb["evidence_role"], "different",
            "a mechanistic / intermediate result and a disease-level result do not test the same thing")
    if xa["cell_line"] or xb["cell_line"]:
        set_dim("cell_line", xa["cell_line"], xb["cell_line"])
    set_dim("subtype", xa["subtype"], xb["subtype"])
    set_dim("metabolic_state", xa["metabolic_state"], xb["metabolic_state"])
    if xa.get("lesion_stage") or xb.get("lesion_stage"):
        set_dim("lesion_stage", xa.get("lesion_stage") or set(), xb.get("lesion_stage") or set())
    da, db_ = xa["dose"], xb["dose"]
    if not da or not db_:
        add("dose", da, db_, "unknown")
    elif da["base_unit"] != db_["base_unit"]:
        add("dose", da, db_, "different", "different dose units (e.g. concentration vs mg/kg) - not directly comparable")
    else:
        ratio = max(da["base_value"], db_["base_value"]) / max(1e-30, min(da["base_value"], db_["base_value"]))
        # 700 mg vs 1500 mg is a different dose (2.1x): only near-identical doses count as 'same'
        add("dose", da, db_, "different" if ratio > 1.25 else "same", f"{ratio:.1f}x apart")
    ta, tb = xa["duration_days"], xb["duration_days"]
    if not ta or not tb:
        add("duration", ta, tb, "unknown")
    else:
        ratio = max(ta, tb) / max(1e-9, min(ta, tb))
        add("duration", ta, tb, "different" if ratio > 1.5 else "same", f"{ratio:.1f}x apart")
    set_dim("comparator", xa["comparator"], xb["comparator"])
    pa, pb = xa["co_treatment"], xb["co_treatment"]
    add("co_treatment", pa or "none", pb or "none", "same" if (pa or "") == (pb or "") else "different")
    ea, eb = xa["endpoint"], xb["endpoint"]
    if ea["family"] == eb["family"] and (ea.get("target") or "") == (eb.get("target") or ""):
        add("endpoint", ea, eb, "same")
    elif frozenset((ea["family"], eb["family"])) in RELATED_OPPOSED:
        add("endpoint", ea, eb, "related", "related but different endpoints (e.g. viability vs apoptosis)")
    else:
        add("endpoint", ea, eb, "different")
    if xa["conditions"] or xb["conditions"]:
        set_dim("conditions", xa["conditions"], xb["conditions"])
    set_dim("disease_induction", xa["disease_induction"], xb["disease_induction"])
    oa, ob = xa["outcome_definition"], xb["outcome_definition"]
    if oa and ob:
        add("outcome_definition", oa, ob, "same" if oa == ob else "different", "different outcome rung (e.g. improvement vs complete response)")
    else:
        add("outcome_definition", oa, ob, "unknown")
    set_dim("anatomical_site", xa["anatomical_site"], xb["anatomical_site"])
    set_dim("severity", xa["severity"], xb["severity"])
    fa, fb = xa["follow_up_days"], xb["follow_up_days"]
    if fa and fb:
        r = max(fa, fb) / max(1e-9, min(fa, fb))
        add("follow_up", f"{fa / 30.44:.1f} months", f"{fb / 30.44:.1f} months", "different" if r > 3 else "same", f"{r:.1f}x apart")
    else:
        add("follow_up", fa, fb, "unknown")
    return rows


CRITICAL = ("species", "model", "endpoint")


# experimental-system tiers: a result in one tier does not test a result in another (patients vs a dish of cells)
SYSTEM_TIER = {"human_clinical": "human", "human_observational": "human", "animal": "animal (in vivo)",
               "organoid_ex_vivo": "organoid / ex vivo", "cell_culture": "cell culture (in vitro)", "in_silico": "in silico",
               "plant": "plant", "microbial": "microbial"}


def system_tier(c: Dict[str, Any]) -> Optional[str]:
    return SYSTEM_TIER.get(c.get("claim_model") or "")


def adjudicate(a: Dict[str, Any], b: Dict[str, Any], text_a: str = "", text_b: str = "") -> Dict[str, Any]:
    res = _adjudicate(a, b, text_a, text_b)
    ta, tb = system_tier(a), system_tier(b)
    if ta and tb and ta != tb:
        # model-tier gate: never a contradiction, and never counted as a disagreement in the headline
        res["verdict"] = "CONTEXT_DIFFERENCE"
        res["kind"] = "model_mismatch"
        res["why"] = (f"Different experimental systems ({ta} vs {tb}) - a cross-context disagreement, not a direct contradiction: a result in one system does not "
                      f"test the result in the other. " + res["why"])
    else:
        res["kind"] = "same_system"
    return res


def _result_word(a: Dict[str, Any], b: Dict[str, Any]) -> str:
    """'opposite results' only for an increase vs a decrease; an effect vs no significant difference is a different result."""
    da = (a.get("canonical_direction") or a.get("effect_direction") or "").lower()
    db_ = (b.get("canonical_direction") or b.get("effect_direction") or "").lower()
    if {da, db_} == {"increase", "decrease"}:
        return "opposite results"
    if "no_effect" in (da, db_):
        return "different results (an effect in one study, no statistically significant difference in the other)"
    return "different results"


def _adjudicate(a: Dict[str, Any], b: Dict[str, Any], text_a: str = "", text_b: str = "") -> Dict[str, Any]:
    xa, xb = claim_context(a, text_a), claim_context(b, text_b)
    rows = compare_contexts(xa, xb)
    by = {r["dimension"]: r for r in rows}
    different = [r for r in rows if r["status"] == "different"]
    unknown_critical = [d for d in CRITICAL if by.get(d, {}).get("status") == "unknown"]
    # dimensions that are usually unreported do not by themselves block a verdict
    rarely = {"disease_induction", "anatomical_site", "severity", "outcome_definition", "follow_up", "subtype", "metabolic_state"}
    unknown_other = [r["dimension"] for r in rows if r["status"] == "unknown" and r["dimension"] not in CRITICAL and r["dimension"] not in rarely]
    same = [r["dimension"] for r in rows if r["status"] == "same"]

    if different:
        verdict = "CONTEXT_DIFFERENCE"
        why = ("Not comparable - the studies differ in " +
               "; ".join(f"{r['dimension'].replace('_', ' ')} ({r['a']} vs {r['b']})" for r in different) +
               ". These differences are candidate explanations for the " + _result_word(a, b) + ".")
    elif by.get("endpoint", {}).get("status") == "related":
        verdict = "UNRESOLVED_DISAGREEMENT"
        why = (f"Related but different endpoints ({by['endpoint']['a']} vs {by['endpoint']['b']}); "
               "a true contradiction needs the same endpoint.")
    elif unknown_critical or len(unknown_other) >= 3:
        verdict = "UNRESOLVED_DISAGREEMENT"
        missing = unknown_critical + unknown_other
        why = ("Comparability cannot be established - not reported for at least one study: "
               + ", ".join(m.replace("_", " ") for m in missing) + ".")
    else:
        verdict = "TRUE_CONTRADICTION"
        why = ("Comparable: same " + ", ".join(s.replace("_", " ") for s in same)
               + (f"; not reported: {', '.join(u.replace('_', ' ') for u in unknown_other)}" if unknown_other else "")
               + " - yet the results point in opposite directions.")
    return {"verdict": verdict, "why": why, "comparison": rows,
            "differences": [{"factor": r["dimension"], "a": r["a"], "b": r["b"]} for r in different],
            "unknown_fields": unknown_critical + unknown_other, "comparable_on": same}


def compare_claims(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    """Backward-compatible wrapper."""
    return adjudicate(a, b)


def _is_disagreement(a: Dict[str, Any], b: Dict[str, Any]) -> bool:
    da, db_ = effective_direction(a), effective_direction(b)
    ca, cb = _canon(a), _canon(b)
    if frozenset((ca["family"], cb["family"])) in RELATED_OPPOSED:
        # viability UP with apoptosis UP (or both DOWN) is the opposed pattern
        return da == db_ and da in ("increase", "decrease")
    return da in DIRECTIONAL and db_ in DIRECTIONAL and da != db_


def detect_and_store(db: Any, research_question_id: Optional[int], router: Any = None, **kwargs) -> Dict[str, Any]:
    claims = db.primary_claims(research_question_id) if hasattr(db, "primary_claims") else db.verified_claims(research_question_id)
    groups = group_claims(claims)
    db.clear_contradictions(research_question_id)
    paper_text: Dict[Any, str] = {}

    def ptext(pid):
        if pid not in paper_text:
            p = db.get_paper(pid) or {}
            paper_text[pid] = f"{p.get('title') or ''} {p.get('abstract') or ''}"
        return paper_text[pid]

    n_pairs, conflict_groups, out_groups, n_mismatch = 0, 0, [], 0
    verdict_counts = {v: 0 for v in VERDICTS}
    seen_findings = set()

    def finding_key(c):
        # one independent finding = paper x model x canonical endpoint x direction (#9: no double counting)
        return (c.get("paper_id"), c.get("claim_model"), _canon(c)["family"], effective_direction(c))
    for members in groups:
        by_dir: Dict[str, List[int]] = defaultdict(list)
        for m in members:
            by_dir[effective_direction(m)].append(m["id"])
        pairs = []
        for a, b in combinations(members, 2):
            if a.get("paper_id") == b.get("paper_id"):
                continue                      # cross-study only
            if not same_topic(a, b) or not _is_disagreement(a, b):
                continue                      # pairwise, not merely transitively grouped
            fk = tuple(sorted((finding_key(a), finding_key(b)), key=str))
            if fk in seen_findings:
                continue                      # same two findings restated by other claims
            seen_findings.add(fk)
            res = adjudicate(a, b, ptext(a.get("paper_id")), ptext(b.get("paper_id")))
            db.add_contradiction(research_question_id, a["id"], b["id"], res["verdict"], res, res["why"])
            pairs.append({"claim_a": a["id"], "claim_b": b["id"], "verdict": res["verdict"], "kind": res["kind"], "reasoning": res["why"]})
            verdict_counts[res["verdict"]] += 1
            n_pairs += 1
            n_mismatch += int(res.get("kind") == "model_mismatch")
        has_conflict = bool(pairs)
        conflict_groups += int(has_conflict)
        interv = members[0].get("intervention") or members[0].get("subject") or "Biological Factor"
        outc = members[0].get("measurement") or members[0].get("outcome") or "Endpoint"
        key = f"{interv}::{outc}"
        summary = {"n_claims": len(members), "directions": dict(by_dir), "has_conflict": has_conflict,
                   "member_ids": [m["id"] for m in members]}
        db.save_claim_group(research_question_id, key, interv, outc, [m["id"] for m in members], summary, [])
        out_groups.append({"key": key, "intervention": interv, "outcome": outc, "summary": summary, "pairs": pairs})
    return {"groups": out_groups, "n_conflict_groups": conflict_groups, "n_conflicting_pairs": n_pairs,
            "verdict_counts": verdict_counts,
            # headline numbers: cross-system pairs (patients vs cells) are listed, but are NOT disagreements
            "n_different_system_pairs": n_mismatch, "n_same_system_disagreements": n_pairs - n_mismatch}


def extract_conflicts_for_geo(analysis_result: Dict[str, Any], claims: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    claims_by_id = {c["id"]: c for c in claims}
    targets = []
    for g in analysis_result.get("groups", []):
        if not g.get("summary", {}).get("has_conflict"):
            continue
        members = [claims_by_id[cid] for cid in g.get("summary", {}).get("member_ids", []) if cid in claims_by_id]
        organisms = {m.get("organism") for m in members if m.get("organism")}
        org = organisms.pop() if organisms else None
        targets.append({
            "group_key": g["key"], "intervention": g["intervention"], "outcome": g["outcome"], "organism": org,
            "conflicting_directions": list(g["summary"]["directions"].keys()),
        })
    return targets


# --------------------------------------------------------------------------- reporting vocabulary
# '0 true contradictions' was read as 'the literature is consistent'. The report states three separate numbers:
#   Direct contradictions        same system, same endpoint, comparable on every dimension, opposite results
#   Cross-context disagreements  opposite / null results that differ in system, population, dose or endpoint
#   Evidence-level conflict      laboratory (cell / animal) effects next to null or opposite human results
_LAB = ("cell_culture", "organoid_ex_vivo", "animal", "experimental_disease_model")
_HUMAN = ("human_clinical", "human_observational", "case_report")


def evidence_level_conflict(claims: List[Dict[str, Any]]) -> Optional[str]:
    """'preclinical vs clinical' when laboratory studies report an effect and human studies of the same agent report
    none (or the opposite) - on any outcome the question is about. None when there is no such pattern."""
    def d(c):
        return c.get("canonical_direction") or c.get("direction")
    # a review's report is not a study in the sample, and a combination regimen does not test the agent alone
    claims = [c for c in claims if (c.get("semantic_audit") or {}).get("evidence_role") not in ("secondary_report", "combination")]
    # the same ENDPOINT only: tumour growth in mice and survival in patients are different questions, not a conflict
    grp = lambda c: {"proliferation": "growth", "tumour_growth": "growth", "viability": "growth"}.get(  # noqa: E731
        c.get("canonical_outcome") or "", c.get("canonical_outcome") or "")
    lab_c = [c for c in claims if c.get("claim_model") in _LAB and d(c) in ("increase", "decrease")]
    lab = {d(c) for c in lab_c}
    lab_ends = {grp(c) for c in lab_c}
    all_human = [c for c in claims if c.get("claim_model") in _HUMAN]
    human = [c for c in all_human if grp(c) in lab_ends or not lab_ends - {""}]
    if lab and all_human and not [c for c in human if d(c) == "no_effect"]:
        # no human null on the SAME endpoint: a null on a clinical outcome (survival) is reported as what it is
        other = [c for c in all_human if d(c) == "no_effect" and c.get("canonical_outcome") in
                 ("patient_survival", "recurrence_incidence", "clinical_response")]
        if other:
            lab_names = sorted({(c.get("canonical_outcome") or "?").replace("_", " ") for c in lab_c})
            hum_names = sorted({(c.get("canonical_outcome") or "?").replace("_", " ") for c in other})
            rct = any(c.get("claim_model") == "human_clinical" for c in other)
            return (f"preclinical vs clinical on DIFFERENT endpoints - laboratory studies report an effect on "
                    f"{', '.join(lab_names)}; {'a randomised human trial' if rct else 'human studies'} found no significant "
                    f"effect on {', '.join(hum_names)} (not a direct test of the laboratory result)")
    if not lab or not human:
        return None
    nulls = [c for c in human if d(c) == "no_effect"]
    rct_null = [c for c in nulls if c.get("claim_model") == "human_clinical"]
    if rct_null:
        return ("preclinical vs clinical - laboratory studies report an effect, a randomised human trial found no "
                "significant effect" + (" (observational human studies in the sample report an association)"
                                        if any(c.get("claim_model") == "human_observational" and d(c) in ("increase", "decrease")
                                               for c in human) else ""))
    if nulls:
        return "preclinical vs clinical - laboratory studies report an effect, human studies found no significant effect"
    return None


def contradiction_counts(contradictions: List[Dict[str, Any]], claims: List[Dict[str, Any]]) -> Dict[str, Any]:
    direct = sum(1 for c in contradictions or [] if c.get("verdict") == "TRUE_CONTRADICTION")
    return {"direct_contradictions": direct, "cross_context_disagreements": len(contradictions or []) - direct,
            "evidence_level_conflict": evidence_level_conflict(claims or [])}


def contradiction_vocabulary(contradictions: List[Dict[str, Any]], claims: List[Dict[str, Any]]) -> str:
    k = contradiction_counts(contradictions, claims)
    return (f"Direct contradictions: {k['direct_contradictions']} · Cross-context disagreements: "
            f"{k['cross_context_disagreements']} · Evidence-level conflict: {k['evidence_level_conflict'] or 'none detected'}")
