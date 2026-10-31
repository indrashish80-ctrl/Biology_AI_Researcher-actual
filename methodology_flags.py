"""
methodology_flags.py - known methodological traps, checked locally
=================================================================

Three problems that make published results unreliable in ways no amount of statistics can repair. All three are
checked against lists kept in this file, so nothing is downloaded and nothing is sent anywhere.

1. MISIDENTIFIED CELL LINES (the ICLAC register)
   Some widely used cell lines are not what their name says. MDA-MB-435 was studied as a breast cancer line in
   hundreds of papers until genetic testing showed it is the melanoma line M14; many other lines were overgrown by
   HeLa. A paper that studies 'breast cancer' in such a line is measuring something else.

2. ASSAY-INTERFERENCE COMPOUNDS (PAINS)
   Some compounds appear active against almost every target because of how they behave in the assay rather than in
   biology: they fluoresce, aggregate, oxidise, or disturb membranes. Curcumin, resveratrol, quercetin and EGCG are
   the best-known. Their laboratory results need a caveat; their clinical results do not.

3. OUTCOME SWITCHING
   A trial registers the outcome it will measure before it starts. When the registered outcome fails, some papers
   report a secondary outcome instead and present it as the point of the trial. Comparing what a paper reports with
   what its registry entry promised catches this. The comparison needs the registry, so it runs only when the
   network is available and is skipped silently otherwise.

The lists are deliberately short and verifiable: well-documented cases only, with the reason stated for each, so a
reader can check any flag against the public register rather than taking this file's word for it.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

FLAGS_VERSION = "1.0"

# --------------------------------------------------------------------------- 1. misidentified cell lines
# name -> (what it actually is, how that was established). From the ICLAC Register of Misidentified Cell Lines;
# these are the long-established, frequently cited cases.
MISIDENTIFIED_LINES: Dict[str, Dict[str, str]] = {
    "MDA-MB-435": {"actually": "the M14 melanoma line", "claimed": "breast cancer",
                   "basis": "DNA profiling matched M14 (melanoma); the line is not of breast origin"},
    "MDA-N": {"actually": "a derivative of MDA-MB-435, i.e. M14 melanoma", "claimed": "breast cancer",
              "basis": "derived from MDA-MB-435, which is M14 melanoma"},
    "KB": {"actually": "HeLa (cervical adenocarcinoma)", "claimed": "oral / nasopharyngeal carcinoma",
           "basis": "HeLa marker profile; one of the earliest documented cross-contaminations"},
    "Hep-2": {"actually": "HeLa", "claimed": "laryngeal carcinoma", "basis": "HeLa marker profile"},
    "HEp-2": {"actually": "HeLa", "claimed": "laryngeal carcinoma", "basis": "HeLa marker profile"},
    "Int-407": {"actually": "HeLa", "claimed": "normal intestinal epithelium", "basis": "HeLa marker profile"},
    "INT 407": {"actually": "HeLa", "claimed": "normal intestinal epithelium", "basis": "HeLa marker profile"},
    "Chang Liver": {"actually": "HeLa", "claimed": "normal liver", "basis": "HeLa marker profile"},
    "WISH": {"actually": "HeLa", "claimed": "amnion", "basis": "HeLa marker profile"},
    "Girardi Heart": {"actually": "HeLa", "claimed": "heart", "basis": "HeLa marker profile"},
    "ECV304": {"actually": "T24 bladder carcinoma", "claimed": "normal endothelium",
               "basis": "DNA profiling matched T24; widely used as an endothelial model"},
    "EA.hy926": {"actually": "a hybrid with A549 lung carcinoma", "claimed": "endothelium",
                 "basis": "fusion partner is the A549 lung carcinoma line"},
    "U-937": {"actually": "reported as cross-contaminated in some stocks", "claimed": "histiocytic lymphoma",
              "basis": "stock-dependent; authentication of the specific stock is needed"},
    "NCI-H292": {"actually": "reported as cross-contaminated in some stocks", "claimed": "lung mucoepidermoid carcinoma",
                 "basis": "stock-dependent; authentication of the specific stock is needed"},
    "SW-13": {"actually": "reported as misidentified", "claimed": "adrenal cortex small-cell carcinoma",
              "basis": "ICLAC register entry"},
    "TE671": {"actually": "the RD rhabdomyosarcoma line", "claimed": "medulloblastoma",
              "basis": "DNA profiling matched RD (rhabdomyosarcoma)"},
    "HBL-100": {"actually": "not a normal breast line (contains SV40 sequences)", "claimed": "normal breast epithelium",
                "basis": "SV40-transformed; not a normal breast control"},
    "MCF-10A": {"actually": "a spontaneously immortalised line, not primary normal tissue", "claimed": "normal breast",
                "basis": "immortalised with genomic changes; a reasonable control but not normal primary tissue"},
}
# the disease / tissue a flag matters for: a melanoma line used AS melanoma is not a problem
_LINE_RX = {name: re.compile(r"\b" + re.escape(name).replace(r"\-", r"[-\s]?").replace(r"\ ", r"[-\s]?") + r"\b", re.I)
            for name in MISIDENTIFIED_LINES}


def misidentified_lines(text: str) -> List[Dict[str, str]]:
    """Every misidentified line named in the text, with what it actually is."""
    out = []
    for name, info in MISIDENTIFIED_LINES.items():
        if _LINE_RX[name].search(text or ""):
            out.append({"line": name, **info})
    return out


def cell_line_flags(text: str, disease: Optional[str] = None, model_type: Optional[str] = None) -> List[Dict[str, Any]]:
    """Flags for a study: a misidentified line used as a model of the disease the question is about.

    The flag is raised whichever disease is asked about (the line is still not what the paper calls it), but it is
    marked `critical` only when the paper's own claimed tissue is what the line is NOT - that is when the result
    cannot mean what the paper says.
    """
    flags = []
    for hit in misidentified_lines(text):
        claimed_here = bool(disease and re.search(r"\b" + re.escape(str(disease).split()[-1][:6]), hit["claimed"], re.I))
        actually_here = bool(disease and re.search(r"\b" + re.escape(str(disease).split()[-1][:6]), hit["actually"], re.I))
        critical = claimed_here and not actually_here
        flags.append({
            "kind": "misidentified_cell_line", "line": hit["line"], "severity": "critical" if critical else "note",
            "model_type": model_type,
            "statement": (f"{hit['line']} is {hit['actually']}, not {hit['claimed']} ({hit['basis']}). "
                          + (f"A result from this line is not evidence about {disease}." if critical
                             else "Check which tissue this result is evidence about.")),
        })
    return flags


# --------------------------------------------------------------------------- 2. assay-interference compounds (PAINS)
# compound -> why its in-vitro signals are unreliable. These are the classes named in the PAINS literature.
INTERFERENCE_COMPOUNDS: Dict[str, Dict[str, str]] = {
    "curcumin": {"why": "fluoresces, forms colloidal aggregates, is redox-active and degrades quickly in culture medium",
                 "note": "and reaches under about 1 micromolar in plasma, far below typical assay concentrations"},
    "resveratrol": {"why": "redox-active, aggregates, and inhibits many unrelated targets in assays",
                    "note": "and reaches only about 2 micromolar in plasma after gram doses"},
    "quercetin": {"why": "a redox-active flavonoid that oxidises in medium and interferes with optical read-outs",
                  "note": "poorly absorbed, so plasma levels are far below assay concentrations"},
    "epigallocatechin gallate": {"why": "oxidises in culture medium to produce hydrogen peroxide, which itself kills cells",
                                 "note": "the apparent activity is often the peroxide, not the compound"},
    "egcg": {"why": "oxidises in culture medium to produce hydrogen peroxide, which itself kills cells",
             "note": "the apparent activity is often the peroxide, not the compound"},
    "genistein": {"why": "promiscuous kinase inhibition and optical interference at assay concentrations", "note": ""},
    "emodin": {"why": "redox-active anthraquinone; a frequent screening false positive", "note": ""},
    "capsaicin": {"why": "membrane-perturbing at the concentrations used in cell assays", "note": ""},
    "berberine": {"why": "accumulates in membranes and fluoresces, disturbing optical assays", "note": ""},
    "rottlerin": {"why": "an uncoupler of oxidative phosphorylation, so it affects nearly every cellular read-out",
                  "note": ""},
    "tannic acid": {"why": "a polyphenol that precipitates proteins non-specifically", "note": ""},
    "gossypol": {"why": "a promiscuous binder flagged as a pan-assay interference compound", "note": ""},
    "sanguinarine": {"why": "a reactive alkaloid that interferes with many assays", "note": ""},
    "quinacrine": {"why": "intercalates DNA and fluoresces strongly", "note": ""},
    "methylene blue": {"why": "a redox dye that interferes directly with colorimetric assays", "note": ""},
}
_PAINS_RX = {name: re.compile(r"\b" + name.replace(" ", r"\s+") + r"s?\b", re.I) for name in INTERFERENCE_COMPOUNDS}
_LAB_MODELS = ("cell_culture", "organoid_ex_vivo", "in_silico", "microbial")


def interference_flags(intervention: str, model_type: Optional[str] = None,
                       text: str = "") -> List[Dict[str, Any]]:
    """A caveat when the studied compound is a known assay-interference compound AND the result is a laboratory one.

    A clinical result with such a compound needs no caveat: a trial measures the compound in people, where assay
    artefacts do not apply.
    """
    hay = f"{intervention or ''} {text or ''}"
    out = []
    for name, info in INTERFERENCE_COMPOUNDS.items():
        if not _PAINS_RX[name].search(hay):
            continue
        lab = (model_type or "").strip() in _LAB_MODELS
        out.append({
            "kind": "assay_interference", "compound": name, "severity": "note" if lab else "info",
            "applies_to_lab_result": lab,
            "statement": (f"{name.capitalize()} is a known assay-interference compound: it {info['why']}"
                          + (f", {info['note']}" if info["note"] else "") + ". "
                          + ("A laboratory signal from it may reflect the assay rather than an effect on the target; "
                             "a result in the laboratory alone is weak evidence of a real mechanism."
                             if lab else
                             "This affects laboratory assays only - a result measured in people is not affected.")),
        })
    return out


# --------------------------------------------------------------------------- 3. outcome switching
NCT_RE = re.compile(r"\bNCT\s?0?(\d{7,8})\b", re.I)
_PRIMARY_IN_PAPER = re.compile(r"\bprimary (?:outcome|endpoint|end[- ]point)s?\b[^.;]{0,160}", re.I)
_REG_API = "https://clinicaltrials.gov/api/v2/studies/{nct}"
REG_TIMEOUT_S = 6.0             # one attempt; an optional check never waits 25 s like an evidence source
MAX_REGISTRY_LOOKUPS = 8        # the trials most likely to matter, not every identifier in the sample
REGISTRY_BUDGET_S = 20.0        # the whole check, start to finish
_REG_CACHE: Dict[str, Optional[List[str]]] = {}
# what the last registry check actually did, so a skipped check is reported rather than looking like a clean pass
LAST_REGISTRY_RUN: Dict[str, Any] = {"looked": 0, "answered": 0, "stopped": None, "seconds": 0.0}


def trial_ids(text: str) -> List[str]:
    """Registered trial identifiers in the text, normalised to NCT________."""
    return sorted({f"NCT{m.group(1).zfill(8)}" for m in NCT_RE.finditer(text or "")})


# words that name no particular outcome on their own - almost every trial measures one of these
_GENERIC_OUTCOME = {"survival", "response", "responses", "outcome", "outcomes", "endpoint", "endpoints", "score",
                    "scores", "index", "level", "levels", "incidence", "event", "events", "status", "function",
                    "disease", "progression", "free", "overall"} - {"progression", "free", "overall"}


def _norm(s: str) -> set:
    stop = {"the", "of", "in", "to", "and", "or", "a", "an", "at", "from", "with", "by", "for", "change", "mean",
            "median", "rate", "time", "number", "percentage", "proportion", "months", "weeks", "days", "baseline",
            "was", "were", "is", "are", "its", "this", "that", "primary", "secondary", "measure", "measured",
            "assessed", "defined", "point", "end"}
    # a marker's name carries its number: 'Ki-67' is one token, not the word 'index'
    toks = re.findall(r"[a-z][a-z0-9]*(?:[-\u2011][a-z0-9]+)*", (s or "").lower())
    return {w for w in toks if len(w) >= 3 and w not in stop}


def compare_outcomes(registered: List[str], paper_text: str) -> Optional[Dict[str, Any]]:
    """Does the paper's stated primary outcome match the registered one?

    Returns None when either side is unavailable (nothing is inferred from missing data). Otherwise
    {'match': bool, 'registered', 'paper', 'statement'}.
    """
    if not registered:
        return None
    m = _PRIMARY_IN_PAPER.search(paper_text or "")
    if not m:
        return None
    said = m.group(0)
    reg_sets = [(_norm(r), r) for r in registered if _norm(r)]
    if not reg_sets:
        return None
    said_set = _norm(said)
    best, best_share = None, 0.0
    for toks, raw in reg_sets:
        # 'overall survival' vs 'progression-free survival' share only the generic word 'survival': what makes an
        # outcome itself is the rest ('overall', 'progression', 'free')
        dist = toks - _GENERIC_OUTCOME
        pool_ = said_set - _GENERIC_OUTCOME if dist else said_set
        ref = dist or toks
        share = len(ref & pool_) / max(1, len(ref))
        if share > best_share:
            best, best_share = raw, share
    match = best_share >= 0.5
    return {
        "match": match, "registered": registered, "paper": said.strip(), "overlap": round(best_share, 2),
        "statement": (f"The paper's stated primary outcome matches the registered one ({best})." if match else
                      f"The registry lists the primary outcome as '{registered[0]}', but the paper describes its "
                      f"primary outcome as '{said.strip()[:120]}'. A paper that reports a different outcome as its "
                      f"main result may be presenting a secondary finding as the point of the trial - read the "
                      f"registry entry before treating this as the trial's conclusion."),
    }


def registry_outcomes(nct: str, http: Any = None) -> Optional[List[str]]:
    """The registered primary outcome(s) of a trial from the public ClinicalTrials.gov interface.

    Strictly bounded: ONE attempt, a short timeout, and the answer cached for the run. This check is a nicety, not
    evidence - it must never slow a run down. Returns None offline, when disabled, or on any error, and the check is
    then skipped rather than guessed.
    """
    import os
    # v26: a registry entry already in the local library is used first (no network, works offline)
    try:
        from library_loader import registry_for
        reg = registry_for(nct)
        if reg and reg.get("primary_outcomes"):
            return list(reg["primary_outcomes"])
    except Exception:
        pass
    if os.getenv("BIOSYN_OFFLINE") or os.getenv("BIOSYN_NO_REGISTRY"):
        return None                      # offline or switched off: the check is skipped, never guessed
    if nct in _REG_CACHE:
        return _REG_CACHE[nct]           # the same trial is often named by several papers of one trial
    out = None
    try:
        if http is None:
            from http_client import HttpClient
            # one try, short timeout: an unreachable registry costs seconds, not minutes
            http = HttpClient(timeout=REG_TIMEOUT_S, retries=0)
        data = http.get_json(_REG_API.format(nct=nct))
        if isinstance(data, dict):
            prim = ((data.get("protocolSection") or {}).get("outcomesModule") or {}).get("primaryOutcomes") or []
            names = [str(o.get("measure") or "").strip() for o in prim if isinstance(o, dict)]
            out = [n for n in names if n] or None
    except Exception:
        out = None
    _REG_CACHE[nct] = out
    return out


def outcome_switch_flags(rows: List[Dict[str, Any]], http: Any = None, max_lookups: int = MAX_REGISTRY_LOOKUPS,
                         budget_s: float = REGISTRY_BUDGET_S) -> List[Dict[str, Any]]:
    """One flag per included trial whose paper names a primary outcome different from the registered one.

    Bounded three ways so it can never dominate a run: at most `max_lookups` trials, a total time budget, and a stop
    after two failures in a row (an unreachable registry is noticed once, not twelve times). Skipped silently when
    the registry cannot be reached - a run without a network is never penalised.
    """
    import time
    out: List[Dict[str, Any]] = []
    t0 = time.perf_counter()
    looked, misses, answered = 0, 0, 0
    stopped = None
    seen: set = set()
    for r in rows or []:
        text = " ".join([str(r.get("title") or ""), str(r.get("abstract") or "")]
                        + [str(f.get("quote") or "") for f in r.get("findings") or []])
        if not _PRIMARY_IN_PAPER.search(text):
            continue                     # the paper never states a primary outcome: nothing to compare, no lookup
        for nct in trial_ids(text):
            if nct in seen:
                continue
            seen.add(nct)
            stopped = ("reached the limit of trials checked" if looked >= max_lookups else
                       "ran out of time" if (time.perf_counter() - t0) > budget_s else
                       "the registry could not be reached" if misses >= 2 else None)
            if stopped:
                _finish(looked, answered, stopped, time.perf_counter() - t0)
                return out               # out of budget: report what was checked, skip the rest
            looked += 1
            reg = registry_outcomes(nct, http=http)
            if not reg:
                misses += 1
                continue
            misses = 0
            answered += 1
            cmp_ = compare_outcomes(reg, text)
            if cmp_ and not cmp_["match"]:
                out.append({"kind": "outcome_switching", "ref": r.get("ref"), "nct": nct, "severity": "major", **cmp_})
    _finish(looked, answered, stopped, time.perf_counter() - t0)
    return out


def _finish(looked: int, answered: int, stopped: Optional[str], seconds: float) -> None:
    LAST_REGISTRY_RUN.update({"looked": looked, "answered": answered, "stopped": stopped,
                              "seconds": round(seconds, 1)})


# --------------------------------------------------------------------------- collecting them for a study
def study_flags(row: Dict[str, Any], text: str, disease: Optional[str] = None) -> List[Dict[str, Any]]:
    """Every methodological flag for one study of the evidence matrix."""
    models = [f.get("model_type") for f in row.get("findings") or [] if f.get("model_type")] or [row.get("evidence_type")]
    model = models[0]
    flags = cell_line_flags(text, disease, model)
    iv = ((row.get("intervention_profile") or {}).get("canonical") or "")
    flags += interference_flags(iv, model, text)
    for f in flags:
        f["ref"] = row.get("ref")
    return flags
