"""
evidence_matrix.py
------------------
1. Automatic evidence buckets (deterministic, no AI):
       human    - clinical / epidemiological / human participants
       animal   - in vivo animals: lab models (mice, rats, zebrafish, ...) and wild / farm animals
       in_vitro - cell lines, organoids, cultured cells (human-derived noted as subtype)
       plant    - plants, algae, crops (Arabidopsis, rice, maize, ...)
       microbial- bacteria, archaea, fungi/yeast, viruses, microbiomes
       other    - in silico, reviews, unclear
   so the engine works for ALL of biology, not only biomedicine.
2. A Markdown Evidence Matrix table (one row per included paper, best claim shown),
   rendered per bucket so Human and Animal evidence are never blended silently.
"""

from __future__ import annotations

import csv
import io
import re
from collections import Counter
from typing import Any, Dict, List, Optional

BUCKET_LABELS = {
    "human": "Human",
    "animal": "Animal (in vivo)",
    "in_vitro": "In vitro / cell models",
    "plant": "Plant",
    "microbial": "Microbial",
    "other": "Other / unclear",
}
BUCKET_ORDER = ["human", "animal", "in_vitro", "plant", "microbial", "other"]
# claim-level model type -> display bucket (Paper -> Experiment -> Finding: a finding lives where it was measured)
_MODEL_BUCKET = {
    "human_clinical": "human", "human_observational": "human", "case_report": "human",
    "animal": "animal", "cell_culture": "in_vitro", "organoid_ex_vivo": "in_vitro",
    "plant": "plant", "microbial": "microbial", "in_silico": "other", "review_meta": "other",
}
MODEL_WORD = {
    "human_clinical": "clinical", "human_observational": "human observational", "case_report": "case report",
    "animal": "animal", "cell_culture": "cell", "organoid_ex_vivo": "organoid / ex vivo",
    "plant": "plant", "microbial": "microbial", "in_silico": "computational", "review_meta": "review",
}


# ------------------------------------------------------------------ one trial, several publications
# Several papers often report ONE trial (the main result, a conference abstract, a subgroup or biomarker analysis).
# Counting each as independent support exaggerates the evidence, and two analyses of one trial are not a
# 'contradiction between studies'. Publications are linked by a registry ID / trial name, by identical reported
# statistics, or (probable link) by the same sample size + design + intervention setting.
_TRIAL_ID = re.compile(r"\b(NCT\d{8}|ISRCTN\d{6,}|EudraCT[\s:-]*\d{4}-\d{6}-\d{2}|ACTRN\d{14}|CTRI/\d{4}/\d+/\d+|"
                       r"(?:CCTG |NCIC CTG |NCIC )?MA\.\d{2}|ALTTO|APHINITY|KATHERINE|NSABP [A-Z]-\d+|TEXT|SOFT|ATAC|IBIS-?II)\b")
_TRIAL_SETTING = re.compile(r"\b(presurgical|pre-?operative|window[- ]of[- ]opportunity|neoadjuvant|adjuvant|"
                            r"biopsy[- ]to[- ]surgery|window trial)\b", re.I)


def trial_identifiers(text: str) -> List[str]:
    return sorted({re.sub(r"^(?:CCTG |NCIC CTG |NCIC )", "", m.group(1)).replace(" ", "") for m in _TRIAL_ID.finditer(text or "")})


def stat_signatures(text: str) -> List[str]:
    """Distinctive reported numbers ('4.0% (95% CI -5.6 to 14.4)', 'HR 1.01'): two papers quoting the same ones report
    the same analysis of the same trial."""
    from semantic_audit import _CI
    sig = set()
    for m in _CI.finditer(text or ""):
        lo, hi = m.group(2), m.group(3)
        if lo != hi:
            sig.add(f"CI {float(lo):g}..{float(hi):g}")
    for m in re.finditer(r"\b(?:HR|OR|RR|hazard ratio(?: for [\w\s\-–]{0,40}?)?)\s*(?:=|of|was|:)?\s*(\d\.\d{2})\b", text or "", re.I):
        sig.add(f"ratio {m.group(1)}")
    return sorted(sig)


def _n_compatible(a: Any, b: Any) -> bool:
    try:
        a, b = int(a), int(b)
    except (TypeError, ValueError):
        return True                      # unknown sample size: the statistics decide
    return a > 0 and b > 0 and max(a, b) / min(a, b) <= 2.0


def group_trials(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Union of publications that report the same trial; every row gets 'trial_group' (T1, T2 ...) and the basis."""
    prim = [r for r in rows if r.get("evidence_type") in ("human_clinical", "human_observational")]
    parent = {r["ref"]: r["ref"] for r in prim}
    basis: Dict[frozenset, str] = {}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def link(a, b, why):
        ra, rb = find(a["ref"]), find(b["ref"])
        if ra != rb:
            parent[rb] = ra
        basis.setdefault(frozenset((a["ref"], b["ref"])), why)

    for i, a in enumerate(prim):
        for b in prim[i + 1:]:
            ids = set(a.get("trial_ids") or []) & set(b.get("trial_ids") or [])
            sig = set(a.get("stat_signatures") or []) & set(b.get("stat_signatures") or [])
            if ids:
                link(a, b, f"same trial identifier ({', '.join(sorted(ids))})")
            elif (any(x.startswith("CI ") for x in sig) or len(sig) >= 2) and _n_compatible(a.get("n"), b.get("n")):
                # one shared 'HR 1.01' is not enough; and the same analysis has the same patients - 20 vs 760 patients
                # with an identical CI is a coincidence (or a template), not one trial
                link(a, b, f"identical reported statistics ({', '.join(sorted(sig))[:60]})")
            elif a.get("lead_author") and a.get("lead_author") == b.get("lead_author") and a.get("evidence_type") == b.get("evidence_type") \
                    and a.get("n") and b.get("n") and (a.get("n") or 0) >= 100 and a.get("n") == b.get("n"):
                # same first author and the very same sample size (>= 100): two publications of one trial (v24.2)
                link(a, b, f"same lead author ({a['lead_author'].title()}) and same sample size (n = {a['n']})")
            elif a.get("evidence_type") == b.get("evidence_type") == "human_clinical" and a.get("n") and a.get("n") == b.get("n") \
                    and (a.get("n") or 0) >= 30 and set(a.get("trial_design_words") or []) & set(b.get("trial_design_words") or []):
                link(a, b, f"probably the same trial (same n = {a['n']}, same "
                           f"{'/'.join(sorted(set(a['trial_design_words']) & set(b['trial_design_words'])))} design)")
    groups: Dict[str, List[str]] = {}
    for r in prim:
        groups.setdefault(find(r["ref"]), []).append(r["ref"])
    out, k = [], 0
    for root, refs in groups.items():
        if len(refs) < 2:
            continue
        k += 1
        tid = f"T{k}"
        why = sorted({v for pair, v in basis.items() if pair <= set(refs)})
        out.append({"id": tid, "refs": sorted(refs, key=lambda x: int(x[1:])), "basis": why})
        for r in prim:
            if r["ref"] in refs:
                r["trial_group"], r["trial_basis"] = tid, "; ".join(why)
    n_pubs = len(prim)
    independent = n_pubs - sum(len(g["refs"]) - 1 for g in out)
    return {"groups": out, "human_publications": n_pubs, "independent_human_studies": independent,
            "statement": (f"{n_pubs} human publication(s) = {independent} independent trial(s) / cohort(s)"
                          + (": " + "; ".join(f"{g['id']} = {', '.join(g['refs'])} ({g['basis'][0]})" for g in out)
                             if out else " (no publication was linked to another)"))}


def plausible_n(n: Any, etype: Optional[str]) -> tuple:
    """One sample size for the whole report (matrix, quality profile and QC use the same value).
    A count parsed from a cell paper (cells, wells, reads), a review's pooled total or an impossible count is not this
    study's n - it is shown as 'not applicable' / 'not reported' instead of a misleading number."""
    try:
        n = int(n) if n is not None else None
    except (TypeError, ValueError):
        n = None
    if etype == "review_meta":
        return None, "not applicable (review - pools other studies)"
    if etype in ("cell_culture", "organoid_ex_vivo", "in_silico"):
        return None, "not applicable (laboratory experiments)"
    if n is not None and (n < 1 or (etype in ("animal", "experimental_disease_model") and n > 600) or n > 5_000_000):
        return None, "not reported (implausible count discarded)"
    return n, (n if n is not None else "not reported")


def bucket_of_model(model_type: Optional[str], fallback: str) -> str:
    """Bucket of one finding from its own model type; unknown / missing model types keep the paper's bucket."""
    return _MODEL_BUCKET.get(model_type or "", fallback)
# genetic / pharmacological perturbation wording (needed before a mechanism step can be called causal)
PERTURB_RE = re.compile(r"\b(knock-?down|knock-?out|siRNA|shRNA|CRISPR|deletion|deficien\w+|-/-|inhibitor|antagonist|"
                        r"blocked by|abolished|abrogated|rescued|reversed|dominant[- ]negative|was required|were required|"
                        r"is required|dependent on|depleti\w+|silenc\w+)\b", re.I)

HUMAN_PATTERNS = [
    r"\bpatients?\b", r"\bparticipants?\b", r"\bvolunteers?\b", r"\bclinical trial", r"\brandomi[sz]ed controlled",
    r"\bcohort\b", r"\bcase[- ]control\b", r"\bcross[- ]sectional\b", r"\bepidemiolog", r"\bwomen\b", r"\bmen\b",
    r"\bchildren\b", r"\badults?\b", r"\bhuman subjects?\b", r"\bin humans\b", r"\bhealthy (?:individuals|subjects)",
    r"\bphase (?:i{1,3}|[1-3])\b", r"\bplacebo\b", r"\bpopulation[- ]based\b", r"\bpostmenopausal\b", r"\binpatients?\b",
]
ANIMAL_PATTERNS = [
    r"\bmice\b", r"\bmouse\b", r"\bmurine\b", r"\brats?\b", r"\brodents?\b", r"\bzebrafish\b", r"\bdrosophila\b",
    r"\bc\. ?elegans\b", r"\bcaenorhabditis\b", r"\bporcine\b", r"\bpigs?\b", r"\bpiglets?\b", r"\bcanine\b", r"\bdogs?\b",
    r"\bprimates?\b", r"\bmacaques?\b", r"\brabbits?\b", r"\bxenograft", r"\bknockout mice\b", r"\btransgenic mice\b",
    r"\bsheep\b", r"\bbovine\b", r"\bhamsters?\b", r"\bguinea pigs?\b", r"\bin vivo\b",
    # wild, farm and non-model animals (ecology, evolution, zoology, veterinary)
    r"\bbirds?\b", r"\bavian\b", r"\bchickens?\b", r"\bpoultry\b", r"\bcattle\b", r"\bcows?\b", r"\bfish(?:es)?\b",
    r"\bsalmon\b", r"\bteleosts?\b", r"\bamphibians?\b", r"\bfrogs?\b", r"\bxenopus\b", r"\breptiles?\b",
    r"\blizards?\b", r"\binsects?\b", r"\bbees?\b", r"\bhoneybees?\b", r"\bmosquito(?:es)?\b", r"\bbeetles?\b",
    r"\bbutterfl(?:y|ies)\b", r"\bants?\b", r"\bnematodes?\b", r"\bcorals?\b", r"\bsea urchins?\b",
    r"\bmammals?\b", r"\bbats?\b", r"\bwhales?\b", r"\bwild populations?\b", r"\bwildlife\b", r"\blivestock\b",
]
IN_VITRO_PATTERNS = [
    r"\bin vitro\b", r"\bcell lines?\b", r"\bcultured\b", r"\bcell culture", r"\borganoids?\b", r"\bprimary cells?\b",
    r"\bMCF-?7\b", r"\bMDA-MB-\d+", r"\bHeLa\b", r"\bHEK-?293", r"\bA549\b", r"\bHepG2\b", r"\bCaco-?2\b", r"\bSH-SY5Y\b",
    r"\bU-?87\b", r"\biPSC", r"\bfibroblasts\b", r"\bTHP-1\b", r"\bJurkat\b", r"\bHCT-?116\b", r"\bPC-?3\b", r"\bLNCaP\b",
]
PLANT_PATTERNS = [
    r"\bplants?\b", r"\barabidopsis\b", r"\bmaize\b", r"\bzea mays\b", r"\brice\b", r"\boryza\b", r"\bwheat\b",
    r"\bbarley\b", r"\bsoybeans?\b", r"\btomato(?:es)?\b", r"\btobacco\b", r"\bnicotiana\b", r"\bseedlings?\b",
    r"\bleaves\b", r"\bleaf\b", r"\broots?\b", r"\bshoots?\b", r"\bchloroplasts?\b", r"\bphotosynthe", r"\bcrops?\b",
    r"\bcultivars?\b", r"\bpollen\b", r"\bflowering\b", r"\balgae\b", r"\bchlamydomonas\b", r"\bmoss\b",
    r"\bphyscomitrella\b", r"\bpotato(?:es)?\b", r"\bgrapevine\b", r"\bpoplar\b",
]
MICROBIAL_PATTERNS = [
    r"\bbacteri(?:a|um|al)\b", r"\bescherichia\b", r"\be\. ?coli\b", r"\bbacillus\b", r"\bpseudomonas\b",
    r"\bstaphylococc", r"\bstreptococc", r"\bmycobacteri", r"\bsalmonella\b", r"\blactobacill", r"\barchaea",
    r"\byeasts?\b", r"\bsaccharomyces\b", r"\bcandida\b", r"\bfung(?:i|us|al)\b", r"\baspergillus\b",
    r"\bmicrob(?:e|es|ial|iome|iota)\b", r"\bgut flora\b", r"\bviruse?s?\b", r"\bviral\b", r"\bbacteriophages?\b",
    r"\bphages?\b", r"\bstrains?\b", r"\bcyanobacteri", r"\bbiofilms?\b", r"\bpathogens?\b",
]
OTHER_PATTERNS = [
    r"\bin silico\b", r"\bmolecular docking\b", r"\bsimulation\b", r"\bmolecular dynamics\b", r"\bmodell?ing study\b",
    r"\bbioinformatic", r"\bphylogenetic analysis\b",
]
REVIEW_RE = re.compile(r"\b(review|meta-analysis|systematic review|narrative review|perspective)\b", re.I)
HUMAN_CELL_RE = re.compile(r"\bhuman\b.{0,40}\b(cells?|cell lines?)\b|\b(MCF-?7|MDA-MB-\d+|HeLa|HEK-?293|A549|HepG2|HCT-?116)\b", re.I)

STUDY_TYPE_BUCKET = {
    "clinical_trial": "human", "cohort": "human", "case_control": "human", "cross_sectional": "human",
    "in_vivo_animal_study": "animal", "in_vitro_cell_assay": "in_vitro", "computational": "other",
}

_ANIMAL_ORGS = ("mus ", "mus musculus", "mouse", "mice", "rat", "rattus", "zebrafish", "danio", "drosophila", "elegans",
                "pig", "sus scrofa", "dog", "canis", "macaque", "rabbit", "sheep", "bovine", "hamster",
                "chicken", "gallus", "xenopus", "frog", "fish", "bird", "bee", "apis", "mosquito", "anopheles",
                "aedes", "insect", "cattle", "bos taurus", "salmon", "bat")
_PLANT_ORGS = ("arabidopsis", "oryza", "rice", "zea mays", "maize", "wheat", "triticum", "hordeum", "barley",
               "glycine max", "soybean", "solanum", "tomato", "potato", "nicotiana", "tobacco", "chlamydomonas",
               "physcomitrella", "populus", "vitis", "plant", "alga")
_MICROBE_ORGS = ("escherichia", "e. coli", "bacillus", "pseudomonas", "staphylococcus", "streptococcus",
                 "mycobacterium", "salmonella", "lactobacillus", "saccharomyces", "yeast", "candida", "aspergillus",
                 "bacteri", "virus", "phage", "fung", "archae", "cyanobacter", "synechocystis", "vibrio",
                 "listeria", "clostridi", "helicobacter", "neurospora", "schizosaccharomyces")


def _count(patterns: List[str], text: str) -> int:
    return sum(len(re.findall(p, text, flags=re.I)) for p in patterns)


def classify_paper(paper: Dict[str, Any], extraction: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Display bucket for one paper. Delegates to the ordered rule list in study_classifier (title first, this
    study's own methods next, background / previous-work sentences ignored) - the old keyword-count scorer,
    where a cell-line paper mentioning 'previously tested in mice' could out-score 'in vitro', is gone.
    Returns {"bucket", "label", "subtype", "confidence", "mixed_with", "signals", "organism", "evidence_type", "rule"}."""
    from study_classifier import bucket_for, classify_study
    c = classify_study(paper, extraction or {})
    bucket = bucket_for(c["evidence_type"], c.get("organism_group"))
    text = f"{paper.get('title') or ''} {paper.get('abstract') or ''}"
    subtype = None
    if bucket == "in_vitro" and HUMAN_CELL_RE.search(text):
        subtype = "human-derived cells"
    elif c["evidence_type"] in ("human_clinical", "human_observational"):
        subtype = "clinical trial" if c["evidence_type"] == "human_clinical" else "observational"
    elif c["evidence_type"] == "review_meta":
        subtype = "secondary literature"
    mixed = next((d for d in c.get("secondary_designs") or [] if bucket_for(d, None) != bucket
                  and bucket_for(d, None) in ("human", "animal", "plant", "microbial")), None)
    return {
        "bucket": bucket,
        "label": BUCKET_LABELS[bucket],
        "subtype": subtype,
        "confidence": 1.0 if c["evidence_type"] != "unclear" else 0.0,
        "mixed_with": bucket_for(mixed, None) if mixed else None,
        "signals": [f"{k}:{v}" for k, v in (c.get("signals") or {}).items()],
        "organism": c.get("organism"),
        "evidence_type": c["evidence_type"],
        "rule": c["rule"],
    }


# -----------------------------------------------------------------------------
# Matrix
# -----------------------------------------------------------------------------

def _first_author(authors: Optional[str]) -> str:
    if not authors:
        return "Anon."
    first = authors.split(",")[0].strip()
    parts = first.split()
    if not parts:
        return "Anon."
    # "Smith J" (PubMed) vs "John Smith" (JATS) -> pick the surname-looking token
    surname = parts[0] if (len(parts) > 1 and len(parts[-1]) <= 3 and parts[-1].isupper()) else parts[-1]
    return f"{surname} et al." if "," in authors else surname


def _cell(v: Any, limit: int = 90) -> str:
    s = "—" if v in (None, "", [], {}) else str(v)
    s = re.sub(r"\s+", " ", s).replace("|", "\\|").strip()
    return s if len(s) <= limit else s[: limit - 1] + "…"


def _p(v: Any) -> str:
    if v is None:
        return "—"
    try:
        f = float(v)
    except (TypeError, ValueError):
        return "—"
    return "<0.001" if f < 0.001 else f"{f:.3g}"


def _best_claim(claims: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not claims:
        return None

    def key(c):
        p = c.get("p_value")
        return (bool(c.get("grounded")), c.get("effect_direction") in ("increase", "decrease", "no_effect"),
                p is not None, -(float(p) if p is not None else 1.0), c.get("grounding_score") or 0)

    return sorted(claims, key=key, reverse=True)[0]


_REASON_MAP = [("textual_grounding", "quote not found in the paper (extraction failure)"),
               ("this_study", "quotes describe prior work, not this study's result"),
               ("intervention", "unclear intervention"), ("endpoint", "unclear outcome"),
               ("direction", "outcome direction not stated or contradicted by the source"),
               ("statement_consistency", "generated statement contradicts the extracted fields"),
               ("population_model", "population/model of the claim does not match the source"),
               ("attribution", "combination regimen - individual components not isolated"),
               ("outcome_strength", "outcome overstated (e.g. 'cure' where the source reports recovery)"),
               ("magnitude", "numbers not supported by the source"), ("significance", "significance not supported by the source"),
               ("relevance_gate", "off-question: disease / model / exposure does not match the research question"),
               ("claim has no source paper", "AI inference without a source")]


def verification_summary(row: Dict[str, Any], claims: List[Dict[str, Any]], ctx: Dict[str, Any]) -> Dict[str, Any]:
    """Why a paper is (or is not) verified - never a black box (point 9)."""
    reasons: Dict[str, int] = {}
    for c in claims:
        if c.get("verification_status") != "quarantined":
            continue
        for r in (c.get("semantic_audit") or {}).get("reasons") or []:
            label = next((v for k, v in _REASON_MAP if r.startswith(k)), r.split(":")[0])
            reasons[label] = reasons.get(label, 0) + 1
    paper_level = []
    basis = row.get("text_basis") or ctx.get("text_basis")
    if basis in ("abstract", None):
        paper_level.append("full text not accessible - abstract only")
    if not ctx.get("source_text"):
        paper_level.append("no abstract or text available")
    if not claims:
        paper_level.append("no extractable result statement (insufficient abstract or extraction failure)")
    n_ok = sum(1 for c in claims if c.get("verification_status") == "verified")
    if n_ok and basis not in ("abstract", None):
        status = "verified (full text)"
    elif n_ok:
        status = "verified (abstract only)"
    else:
        status = "unverified"
    if n_ok and reasons:
        status += " - some claims rejected"
    return {"status": status, "verified_claims": n_ok, "rejected_claims": len(claims) - n_ok,
            "reasons": paper_level + [f"{k} ({v})" for k, v in sorted(reasons.items(), key=lambda x: -x[1])]}


def question_endpoints(question: Optional[str]) -> List[str]:
    """Canonical endpoint families named in the question (e.g. 'proliferation')."""
    from semantic_audit import FAMILY_RX
    q = question or ""
    return [f for f, rx, _ in FAMILY_RX if rx.search(q)]


def build_matrix(db: Any, research_question_id: int, paper_ids: List[int],
                 fulltext_sources: Optional[Dict[int, str]] = None, question: Optional[str] = None,
                 screening: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
    """One row per included paper, built ONLY from verified claims, the paper's frozen
    classification, its independent findings, separate quality dimensions and dose relevance."""
    from clinical_context import (INFERENCE_RANK, comparator, disease_context, follow_up_days, normalize_intervention,
                                  outcome_level, parse_question, sample_size)
    from deep_reasoner import question_focus
    from evidence_quality import apply_replication, dose_relevance, qc_checklist, quality_profile
    from semantic_audit import build_findings, paper_context
    from study_classifier import bucket_for, evidence_level, get_or_classify, level_label, model_profile

    fulltext_sources = fulltext_sources or {}
    focus = question_focus(question or "") if question else "human"
    q_endpoints = question_endpoints(question)
    screen_score = {r.get("paper_id"): r.get("score") for r in (screening or []) if r.get("paper_id")}
    q_info = parse_question(question or "")
    from evidence_identity import paper_on_question, question_frame, subtype_coverage
    frame = question_frame(question or "")
    verified = db.verified_claims(research_question_id)
    all_claims = db.claims_with_papers(research_question_id)
    all_by_paper: Dict[int, List[Dict[str, Any]]] = {}
    for c in all_claims:
        all_by_paper.setdefault(c.get("paper_id"), []).append(c)
    by_paper: Dict[int, List[Dict[str, Any]]] = {}
    for c in verified:
        by_paper.setdefault(c.get("paper_id"), []).append(c)
    total_by_paper = Counter(c.get("paper_id") for c in all_claims)

    rows, off_question = [], []
    for pid in paper_ids:
        p = db.get_paper(pid)
        if not p:
            # never dropped silently: the study registry must account for every deep-read paper
            off_question.append({"ref": f"X{len(off_question) + 1}", "paper_id": pid, "title": f"paper #{pid}",
                                 "study": "n.d.", "evidence_type_label": "unknown",
                                 "reason": "record could not be loaded from the local database", "claims_excluded": 0})
            continue
        # HARD RULE at paper level: a paper about another disease / organ system with no admitted claim
        # is not evidence for this question and is never counted in the evidence buckets
        relv = paper_on_question(p.get("title") or "", p.get("abstract") or "", frame)
        from semantic_audit import is_retracted
        from metadata_authority import is_retracted_meta
        if is_retracted(p.get("title"), p.get("abstract")) or is_retracted_meta(db, p):
            relv = {"on_question": False, "detail": "retracted / withdrawn publication - its results are not evidence"}
            by_paper.pop(pid, None)
        if not relv["on_question"] and not by_paper.get(pid):
            from prisma_screening import is_non_english
            if is_non_english(p.get("title") or "", p.get("abstract") or "") and not relv["detail"].startswith("retracted"):
                # a Chinese-language title is not 'about another disease': it could not be read
                relv = {**relv, "detail": "not in English - the audit reads English text only, so its relevance could not be assessed"}
            cls_off = get_or_classify(db, pid)
            off_question.append({"ref": f"X{len(off_question) + 1}", "paper_id": pid, "title": p.get("title"),
                                 "study": f"{_first_author(p.get('authors'))} ({p.get('year') or 'n.d.'})",
                                 "evidence_type_label": cls_off["label"], "reason": relv["detail"],
                                 "claims_excluded": total_by_paper.get(pid, 0)})
            continue
        n = len(rows) + 1
        ext = db.get_extraction_raw(pid) or {}
        cls = get_or_classify(db, pid)                      # frozen - never re-scored here
        etype = cls["evidence_type"]
        org_grp = cls.get("organism_group")
        pc = by_paper.get(pid, [])
        findings = build_findings(pc)
        # RECONCILIATION (one taxonomy for the whole report): a paper filed as human / animal whose verified findings
        # ALL come from another system (a cell-line paper whose abstract mentions patients; a mouse paper whose only
        # verified results are cell assays) is counted - in every table and in the headline - where its evidence was
        # measured. The frozen classification stays on record and the reason is shown.
        cls_rule = cls["rule"]
        fsys = [f.get("model_type") for f in findings if f.get("model_type") in _MODEL_BUCKET
                and f.get("model_type") not in ("review_meta",)]
        own_b = bucket_for(etype, org_grp)
        if etype not in ("review_meta",) and fsys and all(_MODEL_BUCKET[m] != own_b for m in fsys):
            new_type = Counter(fsys).most_common(1)[0][0]
            cls = {**cls, "evidence_type": new_type, "label": __import__("study_classifier").EVIDENCE_TYPES.get(new_type, new_type)}
            cls_rule = (f"R6: reconciled with its findings - every verified finding comes from "
                        f"{MODEL_WORD.get(new_type, new_type)} experiments (filed first as "
                        f"{__import__('study_classifier').EVIDENCE_TYPES.get(etype, etype)}: {cls['rule']})")
            etype = new_type
            if new_type in ("cell_culture", "organoid_ex_vivo") and org_grp == "human":
                org_grp = "human"                    # human cells: still not human evidence (bucket = in vitro)
        level = evidence_level(etype, focus, org_grp)
        for f in findings:
            f["id"] = f"S{n}.{f['finding']}"
            f["level"] = evidence_level(f.get("model_type") or etype, focus, org_grp)
        bucket0 = bucket_for(etype, org_grp)
        # representative finding: one from the paper's own system first (an animal paper is shown by its animal
        # experiment, not by its cell assay), then one that answers the question, most direct model, smallest p
        def fkey(f):
            return (bucket_of_model(f.get("model_type"), bucket0) == bucket0, bool(f.get("answers_question")),
                    -(f["level"] or 9), f["direction"] in ("increase", "decrease", "no_effect"),
                    f["p_value"] is not None, -(f["p_value"] if f["p_value"] is not None else 1.0))
        best_f = sorted(findings, key=fkey, reverse=True)[0] if findings else None
        best = next((c for c in pc if best_f and c["id"] == best_f["claim_ids"][0]), None)
        study = ext.get("study") or {}
        n_size = study.get("sample_size") or (best or {}).get("sample_size")
        if not n_size:
            sizes = ((ext.get("deterministic_parse") or {}).get("sample_sizes") or [])
            n_size = max(sizes) if sizes else None
        ident = (f"PMID {p['pmid']}" if p.get("pmid") else (p.get("pmcid") or (f"doi:{p['doi']}" if p.get("doi") else f"#{pid}")))
        link = (f"https://pubmed.ncbi.nlm.nih.gov/{p['pmid']}/" if p.get("pmid") else
                f"https://www.ncbi.nlm.nih.gov/pmc/articles/{p['pmcid']}/" if p.get("pmcid") else
                f"https://doi.org/{p['doi']}" if p.get("doi") else None)
        directions = Counter(f["direction"] or "unclear" for f in findings)
        n_total = total_by_paper.get(pid, 0)
        bucket = bucket_for(etype, org_grp)
        model = (best_f or {}).get("model_detail") or cls.get("organism") or "—"
        if best_f and bucket_of_model(best_f.get("model_type"), bucket) != bucket:
            model = f"{model} ({MODEL_WORD.get(best_f.get('model_type'), 'other')} experiment)"
        intervention = (best or {}).get("intervention") or ext.get("intervention")
        ctx = paper_context(db, pid)
        species_text = f"{cls.get('organism') or ''} {p.get('title') or ''} {p.get('abstract') or ''}"
        dose = dose_relevance(intervention or (question or ""),
                              " ".join([c.get("quote") or "" for c in pc] + [c.get("source_context") or "" for c in pc]
                                       + [ctx["source_text"][:6000]]),
                              etype if etype == "review_meta" else ((best_f or {}).get("model_type") or etype), species_text)
        row = {
            "ref": f"S{n}",
            "paper_id": pid,
            "study": f"{_first_author(p.get('authors'))} ({p.get('year') or 'n.d.'})",
            "title": p.get("title"),
            "journal": p.get("journal"),
            "evidence_type": etype,
            "evidence_type_label": cls["label"],
            "organism_group": org_grp,
            "model_profile": model_profile(etype, org_grp, cls.get("organism"), focus),
            "classification_rule": cls_rule,
            "secondary_designs": cls.get("secondary_designs") or [],
            "level": level,
            "level_label": level_label(level, focus, etype),
            "bucket": bucket,
            "bucket_label": BUCKET_LABELS.get(bucket, bucket),
            # Paper -> Experiment -> Finding: the systems this paper's verified findings actually come from
            "finding_buckets": sorted({bucket_of_model(f.get("model_type"), bucket) for f in findings}),
            "subtype": (best_f or {}).get("model_detail") or None,
            "mixed_with": None,
            "bucket_confidence": 1.0,
            "model": model,
            "design": cls["label"],
            "n": n_size,
            "intervention": intervention,
            "outcome": (best or {}).get("measurement"),
            "direction": (best_f or {}).get("direction"),
            "trend": (best_f or {}).get("trend"),           # non-significant numerical trend (shown, never counted)
            "p_value": (best_f or {}).get("p_value"),
            "key_finding": (best or {}).get("statement") if best else
                           (f"No verified finding ({n_total} claim(s) quarantined)" if n_total else "No claims extracted"),
            "quote": (best or {}).get("quote"),
            # v25 principle 6: how reliable the READING of this quote is (text basis, verbatim match, second reader)
            "extraction_confidence": (best or {}).get("extraction_confidence")
                                     or ("low" if (fulltext_sources.get(pid) or ext.get("basis") or "abstract") == "abstract" else None),
            "second_reader": ((best or {}).get("extraction") or {}).get("second_reader"),
            "source_section": (best or {}).get("source_section"),
            "grounded": bool(best),
            "n_claims": len(pc),
            "claim_statuses": dict(Counter(c.get("claim_status") or "?" for c in pc)),
            "authors": p.get("authors"),
            "n_quarantined": n_total - len(pc),
            "findings": findings,
            "n_findings": len(findings),
            "n_experiments": len({f["experiment"] for f in findings}),
            "direction_counts": dict(directions),
            "text_basis": fulltext_sources.get(pid) or ext.get("basis") or "abstract",
            "identifier": ident,
            "link": link,
            "dose_relevance": dose,
            "perturbation": sorted({m.group(0).lower() for c in pc
                                    for m in PERTURB_RE.finditer(f"{c.get('quote') or ''} {c.get('source_context') or ''}")}),
        }
        # ---- evidence identity: relevance, roles, subtypes, primary vs secondary
        row["paper_relevance"] = relv
        row["evidence_roles"] = dict(Counter(f.get("evidence_role") or "direct" for f in findings))
        row["answers_question"] = any(f.get("answers_question") for f in findings)
        subs = sorted({s for f in findings for s in f.get("subtypes") or []} | set(relv.get("subtypes") or []))
        row["disease_subtypes"] = subs or (["not specified"] if frame.get("disease") else [])
        row["secondary_report"] = etype == "review_meta"
        # ---- context the finding depends on (points 7, 9, 13, 26-30, 33)
        quotes_txt = " ".join(f"{c.get('quote') or ''} {c.get('source_context') or ''}" for c in pc)
        paper_txt = f"{p.get('title') or ''}. {p.get('abstract') or ''}"
        row["disease_context"] = disease_context(f"{paper_txt} {quotes_txt}", q_info["disease_terms"])
        row["comparator"] = comparator(f"{quotes_txt} {paper_txt} {ctx['source_text'][:3000]}", etype)
        row["n"] = sample_size(f"{paper_txt} {quotes_txt}", n_size, ctx["source_text"][:20000])
        row["n"], row["n_display"] = plausible_n(row["n"], etype)
        fus = [f.get("follow_up_days") for f in findings if f.get("follow_up_days")]
        row["follow_up_days"] = max(fus) if fus else follow_up_days(f"{paper_txt} {quotes_txt}")
        lv = [(outcome_level(f"{c.get('quote') or ''}")) for c in pc]
        lv = [x for x in lv if x["level"] is not None]
        row["best_outcome"] = max(lv, key=lambda x: x["level"]) if lv else {"level": None, "key": None, "label": "not stated"}
        row["intervention_profile"] = normalize_intervention(intervention or "", f"{quotes_txt} {paper_txt}")
        infs = [f.get("inference_type") for f in findings if f.get("inference_type")]
        row["inference_type"] = max(infs, key=INFERENCE_RANK.index) if infs else None
        row["verification"] = verification_summary(row, all_by_paper.get(pid, []), ctx)
        row["quality"] = quality_profile(row, ctx["source_text"], n_total, len(pc), screen_score.get(pid),
                                         q_endpoints, focus, dose)
        row["qc"] = qc_checklist(row, f"{quotes_txt} {ctx['source_text'][:8000]}")
        # v25: signalling-question risk of bias (RoB 2 / ROBINS-I / SYRCLE), beside the summary rating
        try:
            import risk_of_bias as _rob
            row["risk_of_bias_detail"] = _rob.assess(ctx["source_text"], etype, full_text=row["text_basis"] != "abstract")
        except Exception:
            row["risk_of_bias_detail"] = None
        trial_text = f"{p.get('title') or ''} {p.get('abstract') or ''} {quotes_txt}"
        from metadata_authority import lead_author as _lead, lookup as _meta_lookup
        row["trial_ids"] = sorted(set(trial_identifiers(trial_text)) | set(_meta_lookup(db, p).get("trial_ids") or []))
        row["lead_author"] = _lead(p.get("authors"))
        row["stat_signatures"] = stat_signatures(trial_text)
        row["trial_design_words"] = sorted({m.group(0).lower() for m in _TRIAL_SETTING.finditer(trial_text)})
        # known methodological traps: a cell line that is not what the paper calls it, a compound whose laboratory
        # signals are usually assay artefacts
        from methodology_flags import study_flags
        row["method_flags"] = study_flags(row, f"{paper_txt} {quotes_txt} {ctx['source_text'][:8000]}",
                                          frame.get("disease"))
        rows.append(row)
    apply_replication(rows)
    trials = group_trials(rows)

    buckets = Counter(r["bucket"] for r in rows)
    primary = [r for r in rows if r["level"] is not None]
    human_rows = [r for r in rows if r.get("organism_group") == "human" and r["evidence_type"] in
                  ("human_clinical", "human_observational", "case_report")]
    roles = Counter(f.get("evidence_role") or "direct" for r in rows for f in r.get("findings") or [])
    return {
        "rows": rows,
        "off_question": off_question,
        "question_frame": {k: frame[k] for k in ("question", "intent", "disease", "heterogeneous", "question_subtypes",
                                                  "exposures", "endpoints", "organism_restriction")},
        "question_normalisation": frame["normalisation"],
        "human_split": {"human_clinical": sum(1 for r in human_rows if r["evidence_type"] == "human_clinical"),
                        "human_observational": sum(1 for r in human_rows if r["evidence_type"] == "human_observational"),
                        "human_case_reports": sum(1 for r in human_rows if r["evidence_type"] == "case_report"),
                        "refs": {"human_clinical": [r["ref"] for r in human_rows if r["evidence_type"] == "human_clinical"],
                                 "human_observational": [r["ref"] for r in human_rows if r["evidence_type"] == "human_observational"],
                                 "human_case_reports": [r["ref"] for r in human_rows if r["evidence_type"] == "case_report"]}},
        "subtype_coverage": subtype_coverage(rows, frame),
        "role_counts": dict(roles),
        "trials": trials,
        "bucket_counts": {b: buckets.get(b, 0) for b in BUCKET_ORDER},
        "type_counts": dict(Counter(r["evidence_type"] for r in rows)),
        "level_counts": {f"L{lv}": sum(1 for r in rows if r["level"] == lv) for lv in range(1, 7)},
        "focus": focus,
        "question_endpoints": q_endpoints,
        "totals": {
            "papers": len(rows),
            "primary_studies": len(primary),
            "secondary_or_unclear": len(rows) - len(primary),
            "experiments": sum(r["n_experiments"] for r in rows),
            "independent_findings": sum(r["n_findings"] for r in rows),
            "verified_claims": sum(r["n_claims"] for r in rows),
            "verified_claims_primary": sum(r["n_claims"] for r in rows if not r.get("secondary_report")),
            "verified_claims_secondary": sum(r["n_claims"] for r in rows if r.get("secondary_report")),
            "findings_answering_question": sum(1 for r in rows for f in r.get("findings") or [] if f.get("answers_question")),
            "findings_context_only": sum(1 for r in rows for f in r.get("findings") or [] if not f.get("answers_question")),
            "quarantined_claims": sum(r["n_quarantined"] for r in rows) + sum(x["claims_excluded"] for x in off_question),
            "off_question_papers": len(off_question),
            # only claims the relevance gate itself REJECTED (every audit record mentions the gate - 'passed' ones too)
            "off_question_claims": sum(1 for c in all_claims if c.get("paper_id") in set(paper_ids) and
                                       c.get("verification_status") == "quarantined" and
                                       (((c.get("semantic_audit") or {}).get("checks") or {}).get("relevance_gate") or {})
                                       .get("ok") is False),
        },
    }


def matrix_markdown(matrix: Dict[str, Any], split_by_bucket: bool = True) -> str:
    rows = matrix.get("rows") or []
    if not rows:
        return "_No included studies — evidence matrix is empty._\n"
    header = ("| Ref | Study | Evidence type (level) | Model | n | Intervention → Outcome | Effect | p | "
              "Findings / claims | Full text | Verified quote |\n|---|---|---|---|---|---|---|---|---|---|---|")
    arrow = {"increase": "↑ increase", "decrease": "↓ decrease", "no_effect": "∅ no significant difference",
             "mixed": "↕ mixed", "unclear": "? unclear"}
    basis = {"full_text": "JATS XML", "jats_xml": "JATS XML", "unpaywall_pdf": "OA PDF (Unpaywall)", "oa_pdf": "OA PDF (OpenAlex/S2)",
             "abstract": "Abstract only"}

    def line(r):
        io_ = f"{_cell(r['intervention'], 40)} → {_cell(r['outcome'], 40)}"
        study = f"[{_cell(r['study'], 40)}]({r['link']})" if r.get("link") else _cell(r["study"], 40)
        model = _cell(r["model"], 40)
        etype = f"{_cell(r.get('evidence_type_label') or r.get('design'), 30)} ({_cell((r.get('level_label') or '').split(' ')[0], 6)})"
        quote = f"“{_cell(r['quote'], 110)}”" if r.get("quote") and r.get("grounded") else "—"
        if quote != "—" and r.get("extraction_confidence"):
            quote += f" _(reading confidence: {r['extraction_confidence']}" + \
                     (", 2nd reader agrees" if r.get("second_reader") == "agree" else
                      ", 2nd reader DISAGREES" if r.get("second_reader") == "disagree" else "") + ")_"
        fc = f"{r.get('n_findings', 0)} / {r.get('n_claims', 0)}" + (f" (+{r['n_quarantined']} quarantined)" if r.get("n_quarantined") else "")
        return (f"| {r['ref']} | {study} | {etype} | {model} | {_cell(r['n'] if r.get('n') is not None else ('n/a' if str(r.get('n_display', '')).startswith('not applicable') else '—'), 8)} | {io_} | "
                f"{arrow.get(r.get('direction') or 'unclear', '? unclear')}"
                f"{' (trend ' + ('↓' if r.get('trend') == 'decrease' else '↑') + ', not significant)' if r.get('trend') else ''}"
                f" | {_p(r.get('p_value'))} | {fc} | "
                f"{basis.get(r['text_basis'], r['text_basis'])} | {quote} |")

    if not split_by_bucket:
        return "\n".join([header] + [line(r) for r in rows]) + "\n"

    out = []
    for b in BUCKET_ORDER:
        sub = [r for r in rows if r["bucket"] == b]
        if not sub:
            continue
        out.append(f"### {BUCKET_LABELS[b]} evidence ({len(sub)} {'study' if len(sub) == 1 else 'studies'})\n")
        out.append(header)
        out.extend(line(r) for r in sub)
        out.append("")
    return "\n".join(out) + "\n"


def matrix_csv(matrix: Dict[str, Any]) -> str:
    cols = ["ref", "study", "title", "journal", "evidence_type_label", "level_label", "classification_rule", "bucket_label",
            "model", "n", "intervention", "outcome", "direction", "p_value", "key_finding", "quote", "source_section",
            "n_experiments", "n_findings", "n_claims", "n_quarantined",
            "paper_quality", "evidence_relevance", "claim_grounding", "experimental_quality", "risk_of_bias",
            "translational_relevance", "dose_relevance", "text_basis", "extraction_confidence", "second_reader", "identifier", "link"]
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=cols, extrasaction="ignore")
    w.writeheader()
    for r in matrix.get("rows") or []:
        flat = dict(r)
        for k, v in (r.get("quality") or {}).items():
            flat[k] = f"{v.get('rating')} - {'; '.join(v.get('reasons') or [])}"
        flat["dose_relevance"] = f"{(r.get('dose_relevance') or {}).get('category', '')}: {(r.get('dose_relevance') or {}).get('statement', '')}"
        w.writerow(flat)
    return buf.getvalue()


def bucket_direction_summary(matrix: Dict[str, Any]) -> Dict[str, Dict[str, int]]:
    """Per bucket, how many PRIMARY studies point each way (reviews and unclear designs are not counted;
    one paper counts once, by its representative verified finding)."""
    out: Dict[str, Dict[str, int]] = {}
    for r in matrix.get("rows") or []:
        if r.get("level", 0) is None or not r.get("direction"):
            continue
        d = out.setdefault(r["bucket"], {})
        k = r.get("direction") or "unclear"
        d[k] = d.get(k, 0) + 1
    return out
