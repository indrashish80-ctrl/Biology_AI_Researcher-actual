"""
study_funding.py  (v26)
=======================
Funding sources and conflict-of-interest (COI) disclosures, read by CODE from the paper (no AI):

    JATS XML     <funding-group> / <award-group> / <funding-source> / <funding-statement>,
                 <fn fn-type="COI-statement">, <fn fn-type="conflict">, sections titled Funding / Competing interests
    PubMed       <GrantList> agencies, <CoiStatement>
    plain text   'Funding: ...', 'This work was supported by ...', 'Conflicts of interest: ...' paragraphs (PDFs)

    profile(funding, coi_text) -> {
        'funders': [...], 'industry_funded': True | False | None (not reported),
        'coi': 'none declared' | 'declared' | 'not reported', 'coi_text': ..., 'statement': ...,
        'concern': None | text }   # e.g. 'industry funding of a trial of the funder's product'
Reported as a separate quality dimension. It is never a reason to drop a study: industry funding is a risk to
judge, not proof of bias.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

FUNDING_VERSION = "1.0"
_INDUSTRY = re.compile(
    r"\b(?:Inc\.?|Ltd\.?|LLC|GmbH|S\.?A\.?|AG|plc|Co\.,? Ltd|Corporation|Corp\.?|Pharmaceuticals?|Pharma|Biotech|"
    r"Therapeutics|Laboratories|Novartis|Pfizer|Roche|Genentech|Merck|MSD|AstraZeneca|GlaxoSmithKline|GSK|Sanofi|Bayer|"
    r"Novo Nordisk|Eli Lilly|Lilly|Boehringer|AbbVie|Abbott|Amgen|Gilead|Bristol[- ]Myers|Janssen|Johnson & Johnson|"
    r"Takeda|Astellas|Daiichi|Eisai|Otsuka|Teva|Servier|Ipsen|Lundbeck|Biogen|Regeneron|Vertex|Moderna|BioNTech)\b")
_PUBLIC = re.compile(r"\b(?:National Institutes? of Health|NIH|NCI|NHLBI|NIDDK|Medical Research Council|MRC|Wellcome|"
                     r"National Science Foundation|NSF|European (?:Research Council|Commission|Union)|Horizon 2020|"
                     r"Ministry of|Department of|Government|University|Foundation|Cancer Research UK|DFG|ANR|NHMRC|CIHR|"
                     r"NIHR|Indian Council|ICMR|DBT|National Natural Science Foundation)\b", re.I)
_NONE_COI = re.compile(r"\b(?:no|none|nothing)\b[^.]{0,40}\b(?:conflicts?|competing|financial|disclos|declare)|"
                       r"\b(?:declare|report|have)\b[^.]{0,30}\bno\b[^.]{0,30}\b(?:conflicts?|competing interests?)|"
                       r"^\s*none(?: declared)?\.?\s*$", re.I)
_FUND_HEAD = re.compile(r"^\s*(?:funding(?: sources?| information| statement)?|financial support|sources? of funding|"
                        r"grant support|acknowledg(?:e)?ments?)\s*[:.]?\s*", re.I)
_COI_HEAD = re.compile(r"^\s*(?:conflicts? of interests?|competing (?:financial )?interests?|declarations? of interests?|"
                       r"disclosures?|duality of interest)\s*(?:statement)?\s*[:.]?\s*", re.I)
_SUPPORTED = re.compile(r"(?:this (?:work|study|trial|research)|the (?:study|trial)) (?:was|were|is) (?:supported|funded|"
                        r"sponsored)(?: in part)? by ([^.]{3,300})", re.I)


def _l(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _txt(el: Any) -> str:
    return re.sub(r"\s+", " ", "".join(el.itertext())).strip() if el is not None else ""


def from_jats(art: Any) -> Dict[str, Any]:
    funders: List[str] = []
    statements: List[str] = []
    for e in art.iter():
        t = _l(e.tag)
        if t == "funding-source":
            v = _txt(e)
            if v and v not in funders:
                funders.append(v)
        elif t == "funding-statement":
            statements.append(_txt(e))
    for sec in art.iter():
        if _l(sec.tag) in ("sec", "ack", "fn"):
            ttl = next((c for c in sec if _l(c.tag) == "title"), None)
            head = _txt(ttl)
            if head and re.match(r"(?i)funding|financial support|sources? of funding|grant support", head):
                body = " ".join(_txt(p) for p in sec if _l(p.tag) == "p")
                if body:
                    statements.append(body)
    if not funders:
        for st in statements:
            funders += _funders_from_text(st)
    return {"funders": _uniq(funders), "statement": " ".join(_uniq(statements))[:2000] or None} if (funders or statements) else {}


def coi_from_jats(art: Any) -> Optional[str]:
    for e in art.iter():
        t = _l(e.tag)
        if t == "fn" and (e.get("fn-type") or "").lower() in ("coi-statement", "conflict", "competing-interests", "coi"):
            return _txt(e)[:2000] or None
        if t in ("sec", "fn", "notes"):
            ttl = next((c for c in e if _l(c.tag) == "title"), None)
            if ttl is not None and _COI_HEAD.match(_txt(ttl) + ":"):
                body = " ".join(_txt(p) for p in e if _l(p.tag) == "p")
                if body:
                    return body[:2000]
    return None


def _uniq(xs: List[str]) -> List[str]:
    out: List[str] = []
    for x in xs:
        x = x.strip(" ;,.")
        if x and x not in out:
            out.append(x)
    return out


def _funders_from_text(text: str) -> List[str]:
    m = _SUPPORTED.search(text or "")
    body = m.group(1) if m else re.sub(_FUND_HEAD, "", text or "")
    parts = re.split(r";|,? and (?=[A-Z])|, (?=[A-Z][a-z]+ (?:of|for|Institute|Foundation|Council))", body)
    return [re.sub(r"\s*\((?:grant|award)[^)]*\)|\s*(?:grant|award)s?\s*(?:no\.?|number)?\s*[\w\-/]+", "", p,
                   flags=re.I).strip() for p in parts if len(p.strip()) > 2][:12]


def from_text(text: str) -> Dict[str, Any]:
    """Funding + COI paragraphs from plain text (a PDF, or a full text without JATS markup)."""
    funding: List[str] = []
    coi: Optional[str] = None
    for para in re.split(r"\n\s*\n|\n(?=[A-Z][A-Za-z ]{3,40}:)", text or ""):
        p = para.strip()
        if not p:
            continue
        if _FUND_HEAD.match(p) and len(p) < 2500 and not p.lower().startswith("acknowledg") or _SUPPORTED.search(p):
            funding.append(p)
        elif _COI_HEAD.match(p) and coi is None:
            coi = _COI_HEAD.sub("", p)[:2000]
    funders = _uniq([f for st in funding for f in _funders_from_text(st)])
    return {"funding": {"funders": funders, "statement": " ".join(funding)[:2000]} if funding else {}, "coi": coi}


def profile(funding: Optional[Dict[str, Any]], coi_text: Optional[str], intervention: Optional[str] = None,
            design: Optional[str] = None) -> Dict[str, Any]:
    f = funding or {}
    funders = list(f.get("funders") or [])
    if f.get("lead_sponsor"):
        funders = _uniq([f["lead_sponsor"]] + funders + list(f.get("collaborators") or []))
    for g in f.get("grants") or []:
        if g.get("agency"):
            funders = _uniq(funders + [g["agency"]])
    text = " ".join(funders + [f.get("statement") or ""])
    if not text.strip():
        industry: Optional[bool] = None
    else:
        industry = bool(_INDUSTRY.search(text)) or (f.get("sponsor_class") == "INDUSTRY")
    if coi_text is None or not str(coi_text).strip():
        coi = "not reported"
    elif _NONE_COI.search(str(coi_text)):
        coi = "none declared"
    else:
        coi = "declared"
    concern = None
    if industry and design in ("human_clinical", "rct") and intervention:
        concern = "industry-funded trial - check whether the funder makes the intervention (sponsor bias risk)"
    elif coi == "declared":
        concern = "authors declare financial relationships - judge how they relate to the intervention"
    return {"funders": funders[:12], "industry_funded": industry, "public_funding": bool(_PUBLIC.search(text)) if text else None,
            "coi": coi, "coi_text": (str(coi_text)[:600] if coi_text else None), "statement": f.get("statement"),
            "concern": concern, "version": FUNDING_VERSION}
