"""
metadata_authority.py  (v24.2)
------------------------------
Use what INDEXERS already know before guessing from prose.

A text classifier reading "Metformin and breast cancer: pooled estimates ... (n = 4900)" can call it a clinical trial.
PubMed and Europe PMC already say it is a "Meta-Analysis" - a person at the National Library of Medicine decided that.
The same holds for case reports, retractions and the identity of a cell line. This module:

  1. keeps the publication types the literature sources return (side table `paper_meta`, no schema change);
  2. lets an indexed publication type OVERRIDE the text-based evidence type where it is the stronger signal
     (a meta-analysis is never a clinical trial; a retracted paper is never evidence; a paper that is only
     cell lines and culture reagents is not human observational research);
  3. holds a small table of common cell lines (tissue, subtype, species), so "MDA-MB-231" is triple-negative from
     the table and not from whichever subtype words happen to appear elsewhere in the abstract;
  4. extracts trial registry numbers (NCT ...) from every text it is given, so two publications of one trial link.

Everything here is offline. Network lookups happen where the records are first fetched (retrieval/*), which now keep
the publication types in raw_metadata. A record without publication types behaves exactly as before.
"""
from __future__ import annotations

import json
import re
from typing import Any, Dict, Iterable, List, Optional

_I = re.I

# ------------------------------------------------------------------------------------------ publication types
# PubMed / Europe PMC "publication type" strings, lower-cased. Order of precedence is in classify_by_pubtype().
_RETRACTED = {"retracted publication", "retraction of publication", "retraction notice", "withdrawn"}
_SECONDARY = {"meta-analysis", "systematic review", "review", "network meta-analysis", "scoping review",
              "umbrella review", "meta analysis", "pooled analysis", "editorial", "comment", "letter",
              "news", "published erratum", "erratum", "interview", "biography", "historical article",
              "practice guideline", "guideline", "consensus development conference", "technical report",
              "congress", "lecture", "patient education handout"}
_CASE = {"case reports", "case report"}
_RCT = {"randomized controlled trial", "randomised controlled trial", "controlled clinical trial",
        "pragmatic clinical trial", "clinical trial", "clinical trial, phase i", "clinical trial, phase ii",
        "clinical trial, phase iii", "clinical trial, phase iv", "equivalence trial", "adaptive clinical trial"}
_OBS = {"observational study", "cohort studies", "case-control studies", "comparative study", "twin study",
        "validation study", "evaluation study"}
_VETERINARY = {"veterinary", "clinical trial, veterinary", "randomized controlled trial, veterinary"}


def normalize_pub_types(raw: Any) -> List[str]:
    """Accept a list, a comma / semicolon separated string, or Europe PMC's {'pubType': [...]} and return a clean list."""
    if raw is None:
        return []
    if isinstance(raw, dict):
        raw = raw.get("pubType") or raw.get("pubtype") or raw.get("PublicationType") or []
    if isinstance(raw, str):
        raw = re.split(r"\s*[;|]\s*|\s*,\s*(?=[A-Z])", raw)
    out, seen = [], set()
    for x in raw if isinstance(raw, (list, tuple, set)) else []:
        t = x.get("name") if isinstance(x, dict) else x
        t = re.sub(r"\s+", " ", str(t or "")).strip().lower()
        if t and t not in seen:
            seen.add(t)
            out.append(t)
    return out


def classify_by_pubtype(pub_types: Iterable[str]) -> Dict[str, Any]:
    """What the indexed publication types say about the paper.

    Returns {'retracted', 'secondary', 'case_report', 'rct', 'observational', 'veterinary', 'basis'}.
    A paper can carry several (a 'Review' that is also 'Systematic Review'); precedence is applied by the caller."""
    pts = normalize_pub_types(list(pub_types or []))
    hit = lambda s: next((p for p in pts if p in s), None)          # noqa: E731
    info = {
        "retracted": hit(_RETRACTED), "secondary": hit(_SECONDARY), "case_report": hit(_CASE),
        "rct": hit(_RCT), "observational": hit(_OBS), "veterinary": hit(_VETERINARY), "pub_types": pts,
    }
    # a 'Comment' or 'Letter' on a trial is not a meta-analysis, but it is also not a primary study
    info["basis"] = next((f"publication type '{v}'" for k, v in info.items()
                          if k in ("retracted", "secondary", "case_report", "rct", "observational") and v), None)
    return info


# ------------------------------------------------------------------------------------------ side table
def _ensure(db: Any) -> None:
    with db._lock:
        db._conn.execute("CREATE TABLE IF NOT EXISTS paper_meta (key TEXT PRIMARY KEY, json TEXT NOT NULL)")
        db._conn.commit()


def _keys(pmid: Any = None, doi: Any = None, pmcid: Any = None) -> List[str]:
    ks = []
    if pmid:
        ks.append(f"pmid:{re.sub(r'[^0-9]', '', str(pmid))}")
    if doi:
        ks.append("doi:" + re.sub(r"^https?://(?:dx\.)?doi\.org/", "", str(doi).strip().lower()))
    if pmcid:
        m = re.search(r"(\d+)", str(pmcid))
        if m:
            ks.append(f"pmcid:PMC{m.group(1)}")
    return ks


def candidate_pub_types(cand: Any) -> List[str]:
    """Publication types carried by a retrieval candidate (any source)."""
    raw = getattr(cand, "raw_metadata", None) or {}
    for k in ("pub_types", "pubTypeList", "pubtype", "pub_type", "publication_types", "publicationType"):
        if raw.get(k):
            return normalize_pub_types(raw[k])
    return []


def record_candidate(db: Any, cand: Any) -> None:
    """Remember the indexer metadata of a candidate that is being ingested. Never raises."""
    try:
        pts = candidate_pub_types(cand)
        ks = _keys(getattr(cand, "pmid", None), getattr(cand, "doi", None), getattr(cand, "pmcid", None))
        if not ks:
            return
        txt = f"{getattr(cand, 'title', '')} {getattr(cand, 'abstract', '') or ''}"
        meta = {"pub_types": pts, "trial_ids": registry_ids(txt)}
        if not pts and not meta["trial_ids"]:
            return
        _ensure(db)
        with db._lock:
            for k in ks:
                db._conn.execute("INSERT OR REPLACE INTO paper_meta (key, json) VALUES (?, ?)", (k, json.dumps(meta)))
            db._conn.commit()
    except Exception:
        return


def lookup(db: Any, paper: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Stored indexer metadata for a paper row ({} when none)."""
    try:
        paper = paper or {}
        ks = _keys(paper.get("pmid"), paper.get("doi"), paper.get("pmcid"))
        if not ks:
            return {}
        _ensure(db)
        with db._lock:
            for k in ks:
                row = db._conn.execute("SELECT json FROM paper_meta WHERE key = ?", (k,)).fetchone()
                if row:
                    return json.loads(row[0])
    except Exception:
        return {}
    return {}


def is_retracted_meta(db: Any, paper: Optional[Dict[str, Any]]) -> bool:
    return bool(classify_by_pubtype(lookup(db, paper).get("pub_types")).get("retracted"))


# ------------------------------------------------------------------------------------------ registry IDs
_REG = re.compile(r"\b(NCT\d{8}|ISRCTN\d{8}|ACTRN\d{14}|ChiCTR[-A-Za-z]*\d{6,}|CTRI/\d{4}/\d{2,3}/\d{5,6}|"
                  r"EudraCT[\s:#-]*\d{4}-\d{6}-\d{2}|UMIN\d{9}|jRCT\w?\d{9,10}|DRKS\d{8})\b", _I)


def registry_ids(text: str) -> List[str]:
    return sorted({re.sub(r"[\s:#-]+", "", m.group(1)).upper() if m.group(1).upper().startswith("EUDRACT")
                   else m.group(1).upper() for m in _REG.finditer(text or "")})


# ------------------------------------------------------------------------------------------ cell lines
# name pattern -> (tissue / disease, subtype tags using the engine's own subtype names, species, note)
# Subtype tags are only those that are defined for the line in the literature (ATCC / Cellosaurus entries).
_L = [
    (r"MCF-?7", "breast", ["ER-positive"], "human", "luminal A; ER+ PR+ HER2-"),
    (r"T-?47D", "breast", ["ER-positive"], "human", "luminal A; ER+ PR+"),
    (r"ZR-?75-?1", "breast", ["ER-positive"], "human", "luminal; ER+"),
    (r"CAMA-?1", "breast", ["ER-positive"], "human", "luminal; ER+"),
    (r"BT-?474", "breast", ["HER2-positive", "ER-positive"], "human", "HER2 amplified, ER+"),
    (r"SK-?BR-?3", "breast", ["HER2-positive"], "human", "HER2 amplified, ER-"),
    (r"MDA-?MB-?453", "breast", ["HER2-positive"], "human", "HER2+, AR+"),
    (r"MDA-?MB-?231", "breast", ["triple-negative"], "human", "claudin-low; ER- PR- HER2-"),
    (r"MDA-?MB-?468", "breast", ["triple-negative"], "human", "basal-like; ER- PR- HER2-"),
    (r"BT-?549", "breast", ["triple-negative"], "human", "ER- PR- HER2-"),
    (r"HCC-?1937", "breast", ["triple-negative"], "human", "BRCA1-mutated TNBC"),
    (r"Hs-?578T", "breast", ["triple-negative"], "human", "claudin-low TNBC"),
    (r"4T1", "breast", ["triple-negative"], "mouse", "murine TNBC-like"),
    (r"MCF-?10A", "breast", [], "human", "non-tumorigenic breast epithelial"),
    (r"HeLa", "cervix", [], "human", "cervical adenocarcinoma, HPV18"),
    (r"A549", "lung", ["NSCLC"], "human", "lung adenocarcinoma"),
    (r"HepG2", "liver", [], "human", "hepatocellular carcinoma line"),
    (r"HCT-?116", "colon", [], "human", "colorectal carcinoma, MSI"),
    (r"HT-?29", "colon", [], "human", "colorectal adenocarcinoma"),
    (r"Caco-?2", "colon", [], "human", "colorectal adenocarcinoma"),
    (r"LNCaP", "prostate", [], "human", "prostate carcinoma, AR+"),
    (r"PC-?3", "prostate", [], "human", "prostate carcinoma, AR-"),
    (r"DU-?145", "prostate", [], "human", "prostate carcinoma, AR-"),
    (r"U-?87(?:-?MG)?", "brain", [], "human", "glioblastoma"),
    (r"A375", "skin", [], "human", "melanoma, BRAF V600E"),
    (r"SK-?MEL-?28", "skin", [], "human", "melanoma, BRAF V600E"),
    (r"B16(?:-?F10)?", "skin", [], "mouse", "murine melanoma"),
    (r"HaCaT", "skin", [], "human", "immortalised keratinocytes"),
    (r"PIG-?1|PIG-?3V", "skin", [], "human", "immortalised melanocyte lines"),
    (r"HEK-?293T?", "kidney", [], "human", "embryonic kidney (transfection host)"),
    (r"Jurkat", "blood", [], "human", "T-cell leukemia"),
    (r"K-?562", "blood", [], "human", "chronic myeloid leukemia"),
    (r"HL-?60", "blood", [], "human", "acute promyelocytic leukemia"),
    (r"THP-?1", "blood", [], "human", "acute monocytic leukemia"),
    (r"RAW ?264\.7", "blood", [], "mouse", "murine macrophage"),
    (r"NIH-?3T3", "connective", [], "mouse", "murine fibroblast"),
]
CELL_LINES = [(re.compile(r"(?<![\w-])(?:" + p + r")(?![\w])", _I), t, s, sp, n) for p, t, s, sp, n in _L]
_GENERIC_CELL = re.compile(r"\bcell lines?\b|\bcultured\b|\bin vitro\b|\bknock-?(?:down|out)\b|\btransfect\w*|\bsi-?RNA\b|"
                           r"\bshRNA\b|\bCRISPR\b|\bcell culture\b|\bprimary (?:human )?\w+ cells\b", _I)
_PARTICIPANTS = re.compile(r"\b(?:patients?|participants?|volunteers?|cohort|enrolled|recruited|women|men|children|"
                           r"adults?|individuals|people|donors?|subjects|probands?|case-control|trial|"
                           r"we (?:retrospectively )?(?:analy[sz]ed|reviewed|enrolled))\b", _I)
_ANIMAL = re.compile(r"\b(?:mice|mouse|murine|rats?|rodents?|xenograft\w*|zebrafish|in vivo|animals?)\b", _I)


def cell_lines_in(text: str) -> List[Dict[str, Any]]:
    out, seen = [], set()
    for rx, tissue, subtypes, species, note in CELL_LINES:
        m = rx.search(text or "")
        if m and m.group(0).lower() not in seen:
            seen.add(m.group(0).lower())
            out.append({"name": m.group(0), "tissue": tissue, "subtypes": list(subtypes), "species": species, "note": note})
    return out


def constrain_subtype(claim_model: Optional[str], text: str, paper_level: List[str]) -> List[str]:
    """A cell or animal result carries the subtype of ITS cell line, not whichever subtype words the abstract uses
    elsewhere (a TNBC line studied in a paper that also mentions ER-positive disease is triple-negative)."""
    if claim_model not in ("cell_culture", "animal", "organoid_ex_vivo"):
        return paper_level
    lines = [c for c in cell_lines_in(text) if c["species"] == "human" or claim_model == "animal"]
    tags = sorted({t for c in lines for t in c["subtypes"]})
    if not lines or not tags:
        return paper_level
    # keep only tags the paper-level reading also found OR that the table defines; the table wins for lines it knows
    return tags


# ------------------------------------------------------------------------------------------ reconcile
def reconcile_type(c: Dict[str, Any], paper: Dict[str, Any], meta: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Apply indexer metadata and one structural rule to the classifier's decision `c` (modified copy returned).

    Order: retracted > secondary (meta-analysis / review / comment) > case report > RCT fills 'unclear' >
    an all-cell paper filed under people becomes cell culture."""
    from study_classifier import organism_group
    c = dict(c)
    sig = dict(c.get("signals") or {})
    meta = meta or {}
    info = classify_by_pubtype(meta.get("pub_types"))
    title, abstract = paper.get("title") or "", paper.get("abstract") or ""
    body = f"{title}. {abstract}"

    def change(new: str, why: str) -> None:
        if new == c.get("evidence_type"):
            return
        from study_classifier import EVIDENCE_TYPES
        sig["overridden_from"] = c.get("evidence_type")
        c.update({"evidence_type": new, "label": EVIDENCE_TYPES[new], "rule": why,
                  "organism_group": organism_group(new, c.get("organism"), body)})

    if info["retracted"]:
        sig["retracted_pubtype"] = info["retracted"]
    if info["secondary"] and c.get("evidence_type") != "review_meta":
        # a 'Comment'/'Letter' about a trial is also secondary, but a primary report that happens to be tagged
        # 'Letter' is rare; trust the index
        change("review_meta", f"R-meta: indexed as '{info['secondary']}' (secondary literature, not a primary study)")
    elif info["case_report"] and c.get("evidence_type") in ("human_clinical", "human_observational", "unclear"):
        change("case_report", f"R-meta: indexed as '{info['case_report']}'")
    elif info["rct"] and c.get("evidence_type") == "unclear" and not info["veterinary"]:
        change("human_clinical", f"R-meta: indexed as '{info['rct']}'")
    elif info["rct"] and c.get("evidence_type") == "human_observational":
        sig["pubtype_conflict"] = f"indexed as '{info['rct']}' but classified observational"
    elif c.get("evidence_type") in ("human_clinical", "human_observational") and not (info["rct"] or info["observational"]) \
            and not _PARTICIPANTS.search(body) and (cell_lines_in(body) or re.search(r"\bcell lines?\b", body, _I)) \
            and _GENERIC_CELL.search(body) and not _ANIMAL.search(body):
        change("cell_culture", "R-cell: the title and abstract describe only cell lines and culture methods, "
                               "with no patients, participants or cohort")
    c["signals"] = {k: v for k, v in sig.items() if v}
    return c


def lead_author(authors: Any) -> str:
    """Lower-case surname of the first author ('Goodwin PJ, Chen B' -> 'goodwin'; 'Pamela J Goodwin' -> 'goodwin')."""
    first = re.split(r"[;]|,\s*(?=[A-Z][a-z]+\s+[A-Z])|\band\b", str(authors or ""))[0].strip()
    first = re.sub(r"\bet al\.?.*$", "", first, flags=re.I).strip()
    if "," in first:
        first = first.split(",")[0]
    else:
        toks = first.split()
        # 'Goodwin PJ' (surname + initials) vs 'Pamela J Goodwin' (given names + surname)
        first = toks[0] if len(toks) >= 2 and re.fullmatch(r"[A-Z]{1,3}", toks[-1]) else (toks[-1] if toks else "")
    return re.sub(r"[^a-z]", "", first.lower())
