"""
library_loader.py  (v26)
========================
Fills and updates the local library (library_db.py) from PERMITTED bulk files, so questions are answered from the
local collection instead of one API call per search.

Sources
    PubMed baseline + daily update files   https://ftp.ncbi.nlm.nih.gov/pubmed/baseline/  and  .../updatefiles/
                                           (pubmedNNnNNNN.xml.gz; <DeleteCitation> in update files is applied)
    PMC Open Access bulk packages          https://ftp.ncbi.nlm.nih.gov/pub/pmc/oa_bulk/  (.tar.gz of JATS XML)
                                           or any folder of JATS .xml / .nxml files (Europe PMC open-access too)
    ClinicalTrials.gov                     API v2 (registered and ongoing studies, sponsors, registered outcomes,
                                           whether results were posted) - also a JSON file saved from that API
    Your own files                         user_papers.py (kept in a separate collection)
Each record keeps its published date, its indexed date and its last revision date apart, its licence, its source
link and the time it was retrieved; every change is versioned (library_db.record_history).

Usage
    python library_loader.py scope --mesh "Breast Neoplasms; Diabetes Mellitus" --keywords "metformin" --year-from 2000
    python library_loader.py download pubmed-baseline [--max-files 5]     # downloads, checks md5, loads (in scope)
    python library_loader.py download pubmed-updates                      # every update file not applied yet
    python library_loader.py download pmc-oa --subset oa_comm [--max-files 2]
    python library_loader.py pubmed FILE.xml.gz [...]                     # load files you already downloaded
    python library_loader.py pmc PACKAGE.tar.gz | FOLDER [...]
    python library_loader.py trials --query "metformin breast cancer" [--max 500]   |   trials FILE.json
    python library_loader.py embed [--rebuild] [--hnsw]                   # semantic index of new / changed records
    python library_loader.py preread [--limit 200] [--designs rct,meta]   # AI reads each paper ONCE (needs a key)
    python library_loader.py update                                       # the daily job: updates + embed + statuses
    python library_loader.py status
The daily job also runs inside the web app (ingestion worker) when BIOSYN_LIBRARY_AUTOUPDATE is not 0.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import os
import re
import sys
import tarfile
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import library_db as L

LOADER_VERSION = "1.0"
PUBMED_BASELINE = "https://ftp.ncbi.nlm.nih.gov/pubmed/baseline/"
PUBMED_UPDATES = "https://ftp.ncbi.nlm.nih.gov/pubmed/updatefiles/"
PMC_OA_BULK = "https://ftp.ncbi.nlm.nih.gov/pub/pmc/oa_bulk/{subset}/xml/"
CTGOV_API = "https://clinicaltrials.gov/api/v2/studies"
_MONTHS = {m: i for i, m in enumerate(("jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"), 1)}


# ================================================================================================ scope
class Scope:
    """Which records belong in the library. Empty scope = everything in the files you load."""

    def __init__(self, mesh: Sequence[str] = (), keywords: Sequence[str] = (), journals: Sequence[str] = (),
                 year_from: Optional[int] = None, year_to: Optional[int] = None, pub_types: Sequence[str] = ()):
        self.mesh = [m.strip().lower() for m in mesh if m.strip()]
        self.keywords = [k.strip().lower() for k in keywords if k.strip()]
        self.journals = [j.strip().lower() for j in journals if j.strip()]
        self.year_from, self.year_to = year_from, year_to
        self.pub_types = [p.strip().lower() for p in pub_types if p.strip()]

    @classmethod
    def load(cls, lib: L.Library) -> "Scope":
        return cls(**(lib.get_sync("scope") or {}))

    def as_dict(self) -> Dict[str, Any]:
        return {"mesh": self.mesh, "keywords": self.keywords, "journals": self.journals, "year_from": self.year_from,
                "year_to": self.year_to, "pub_types": self.pub_types}

    def empty(self) -> bool:
        return not (self.mesh or self.keywords or self.journals or self.year_from or self.year_to or self.pub_types)

    def matches(self, rec: Dict[str, Any]) -> bool:
        y = rec.get("year")
        if self.year_from and y and int(y) < self.year_from:
            return False
        if self.year_to and y and int(y) > self.year_to:
            return False
        if self.journals and (rec.get("journal") or "").lower() not in self.journals:
            return False
        if self.pub_types and not ({p.lower() for p in rec.get("pub_types") or []} & set(self.pub_types)):
            return False
        if self.mesh or self.keywords:
            mesh = [m.lower() for m in rec.get("mesh") or []]
            hit_mesh = any(any(m == s or m.startswith(s + ",") or s in m for m in mesh) for s in self.mesh)
            text = f"{rec.get('title') or ''} {rec.get('abstract') or ''}".lower()
            hit_kw = any(re.search(r"\b" + re.escape(k), text) for k in self.keywords)
            return hit_mesh or hit_kw
        return True


# ================================================================================================ PubMed XML
def _txt(el: Optional[ET.Element]) -> str:
    return re.sub(r"\s+", " ", "".join(el.itertext())).strip() if el is not None else ""


def _date(el: Optional[ET.Element]) -> Optional[str]:
    """YYYY[-MM[-DD]] from a PubMed / JATS date element (Year / Month / Day, or MedlineDate '2014 Jan-Feb')."""
    if el is None:
        return None
    y = _txt(el.find("Year")) or _txt(el.find("year"))
    m = _txt(el.find("Month")) or _txt(el.find("month"))
    d = _txt(el.find("Day")) or _txt(el.find("day"))
    if not y:
        md = _txt(el.find("MedlineDate"))
        mm = re.match(r"(\d{4})(?:\s+([A-Za-z]{3}))?", md)
        if not mm:
            return None
        y, m = mm.group(1), mm.group(2) or ""
    if m and not m.isdigit():
        m = str(_MONTHS.get(m[:3].lower(), "") or "")
    out = y
    if m:
        out += f"-{int(m):02d}"
        if d and d.isdigit():
            out += f"-{int(d):02d}"
    return out


def _pubmed_record(art: ET.Element) -> Dict[str, Any]:
    mc = art.find("MedlineCitation")
    a = mc.find("Article") if mc is not None else None
    pmid = _txt(mc.find("PMID")) if mc is not None else ""
    title = _txt(a.find("ArticleTitle")) if a is not None else ""
    if not title and a is not None:                 # some records only have a title in the original language
        title = _txt(a.find("VernacularTitle"))
    title = title or (f"[no title] PMID {pmid}" if pmid else "[no title]")
    parts = []
    for t in (a.findall("Abstract/AbstractText") if a is not None else []):
        lab = t.get("Label")
        parts.append((f"{lab.title()}: " if lab else "") + _txt(t))
    authors = []
    for au in (a.findall("AuthorList/Author") if a is not None else []):
        if au.find("CollectiveName") is not None:
            authors.append(_txt(au.find("CollectiveName")))
        else:
            ln, ini = _txt(au.find("LastName")), _txt(au.find("Initials"))
            if ln:
                authors.append(f"{ln} {ini}".strip())
    journal = _txt(a.find("Journal/Title")) if a is not None else ""
    pub = _date(a.find("Journal/JournalIssue/PubDate")) if a is not None else None
    epub = None
    for ad in (a.findall("ArticleDate") if a is not None else []):
        epub = _date(ad) or epub
    published = min([d for d in (pub, epub) if d], default=None, key=lambda s: s.ljust(10, "9")) if (pub or epub) else None
    year = int(published[:4]) if published else None
    pub_types = [_txt(p) for p in (a.findall("PublicationTypeList/PublicationType") if a is not None else [])]
    mesh = [_txt(m.find("DescriptorName")) for m in (mc.findall("MeshHeadingList/MeshHeading") if mc is not None else [])]
    kws = [_txt(k) for k in (mc.findall("KeywordList/Keyword") if mc is not None else [])]
    revised = _date(mc.find("DateRevised")) if mc is not None else None
    indexed = None
    for pd in art.findall("PubmedData/History/PubMedPubDate"):
        if pd.get("PubStatus") in ("pubmed", "entrez", "medline"):
            dd = _date(pd)
            indexed = dd if (indexed is None or (dd and dd < indexed)) else indexed
    ids = {i.get("IdType"): _txt(i) for i in art.findall("PubmedData/ArticleIdList/ArticleId")}
    corrections = [{"type": c.get("RefType"), "reference": _txt(c.find("RefSource"))}
                   for c in (mc.findall("CommentsCorrectionsList/CommentsCorrections") if mc is not None else [])]
    grants = [{"agency": _txt(g.find("Agency")), "id": _txt(g.find("GrantID")), "country": _txt(g.find("Country"))}
              for g in (a.findall("GrantList/Grant") if a is not None else [])]
    coi = _txt(mc.find("CoiStatement")) if mc is not None else ""
    try:
        import publication_status as ps
        notices = ps.from_pub_types(pub_types) + ps.from_comment_corrections(corrections) + ps.from_text(title, "")
        status = ps.worst([n["status"] for n in notices]) if notices else "current"
    except Exception:
        notices, status = [], "current"
    return {"pmid": pmid or None, "doi": ids.get("doi") or None, "pmcid": ids.get("pmc") or None, "title": title,
            "abstract": " ".join(parts) or None, "authors": authors, "journal": journal or None, "year": year,
            "published_date": published, "indexed_date": indexed, "revised_date": revised, "pub_types": pub_types,
            "mesh": mesh, "keywords": kws, "status": status, "notices": notices,
            "funding": {"grants": grants} if grants else None, "coi": coi or None}
    # no licence from a PubMed record: a licence belongs to a full text (PMC / publisher), set when one is loaded


def _open_maybe_gz(path: Path) -> io.BufferedIOBase:
    with open(path, "rb") as fh:
        magic = fh.read(2)
    return gzip.open(path, "rb") if magic == b"\x1f\x8b" else open(path, "rb")


def iter_pubmed(path: Path) -> Iterator[Tuple[str, Any]]:
    """('upsert', record) for every PubmedArticle, ('delete', pmid) for every DeleteCitation/PMID. Streams the file."""
    with _open_maybe_gz(path) as fh:
        in_delete = False
        root = None
        for ev, el in ET.iterparse(fh, events=("start", "end")):
            tag = el.tag
            if ev == "start":
                if root is None:
                    root = el
                if tag == "DeleteCitation":
                    in_delete = True
                continue
            if tag == "PubmedArticle":
                try:
                    yield "upsert", _pubmed_record(el)
                except Exception as exc:
                    yield "error", f"{type(exc).__name__}: {exc}"
                el.clear()
                if root is not None:
                    root.clear()                     # drop finished articles: memory stays flat on a 30,000-record file
            elif tag == "PMID" and in_delete:
                yield "delete", (el.text or "").strip()
            elif tag == "DeleteCitation":
                in_delete = False
                el.clear()


# ================================================================================================ JATS (PMC / Europe PMC)
def _l(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def parse_jats(xml: str) -> Dict[str, Any]:
    """Record + sections + licence + funding + conflict-of-interest statement from one JATS article."""
    root = ET.fromstring(xml.encode("utf-8") if isinstance(xml, str) else xml)
    art = root if _l(root.tag) == "article" else next((e for e in root.iter() if _l(e.tag) == "article"), root)

    def first(path_tags: Sequence[str], base: ET.Element = art) -> Optional[ET.Element]:
        cur = [base]
        for t in path_tags:
            nxt = [c for e in cur for c in e if _l(c.tag) == t]
            if not nxt:
                return None
            cur = nxt
        return cur[0]

    meta = first(["front", "article-meta"])
    ids = {}
    for e in (meta if meta is not None else []):
        if _l(e.tag) == "article-id":
            ids[e.get("pub-id-type")] = (e.text or "").strip()
    title = _txt(first(["title-group", "article-title"], meta)) if meta is not None else ""
    abstract = ""
    for e in (meta if meta is not None else []):
        if _l(e.tag) == "abstract" and not e.get("abstract-type"):
            abstract = _txt(e)
            break
    authors = []
    for e in (meta.iter() if meta is not None else []):
        if _l(e.tag) == "contrib" and e.get("contrib-type", "author") == "author":
            sn = next((c for c in e.iter() if _l(c.tag) == "surname"), None)
            gn = next((c for c in e.iter() if _l(c.tag) == "given-names"), None)
            if sn is not None:
                authors.append(f"{_txt(sn)} {''.join(w[0] for w in _txt(gn).split() if w)}".strip())
    journal = _txt(next((e for e in art.iter() if _l(e.tag) == "journal-title"), None))
    dates = {}
    for e in (meta if meta is not None else []):
        if _l(e.tag) == "pub-date":
            kind = e.get("pub-type") or e.get("date-type") or "pub"
            y = _txt(next((c for c in e if _l(c.tag) == "year"), None))
            m = _txt(next((c for c in e if _l(c.tag) == "month"), None))
            d = _txt(next((c for c in e if _l(c.tag) == "day"), None))
            if y:
                dates[kind] = y + (f"-{int(m):02d}" if m.isdigit() else "") + (f"-{int(d):02d}" if m.isdigit() and d.isdigit() else "")
    published = min(dates.values(), key=lambda s: s.ljust(10, "9")) if dates else None
    kws = [_txt(e) for e in (meta.iter() if meta is not None else []) if _l(e.tag) == "kwd"]
    import fulltext_archive as fa
    lic = fa.licence_from_jats(xml if isinstance(xml, str) else xml.decode("utf-8", "replace"))
    sections: List[Tuple[str, str]] = []
    if abstract:
        sections.append(("abstract", abstract))
    body = first(["body"])
    from jats_ingest import classify_section_title

    def walk(el: ET.Element, cur: str) -> None:
        for ch in el:
            t = _l(ch.tag)
            if t == "sec":
                ttl = _txt(next((c for c in ch if _l(c.tag) == "title"), None))
                walk(ch, classify_section_title(ttl, cur) if ttl else cur)
            elif t == "p":
                txt = _txt(ch)
                if txt:
                    sections.append((cur, txt))
    if body is not None:
        walk(body, "other")
    for tw in art.iter():
        if _l(tw.tag) == "table-wrap":
            cap = _txt(next((c for c in tw.iter() if _l(c.tag) == "caption"), None))
            rows = []
            for tr in (c for c in tw.iter() if _l(c.tag) == "tr"):
                rows.append(" | ".join(_txt(td) for td in tr if _l(td.tag) in ("td", "th")))
            if rows:
                sections.append(("table", (cap + "\n" if cap else "") + "\n".join(rows)))
    import study_funding as sf
    funding, coi = sf.from_jats(art), sf.coi_from_jats(art)
    # paragraphs of the same section are joined with blank lines so split_passages keeps paragraph borders
    merged: List[Tuple[str, str]] = []
    for sec, txt in sections:
        if merged and merged[-1][0] == sec and sec != "table":
            merged[-1] = (sec, merged[-1][1] + "\n\n" + txt)
        else:
            merged.append((sec, txt))
    pmcid = ids.get("pmc") or ids.get("pmcid")
    if pmcid and not str(pmcid).upper().startswith("PMC"):
        pmcid = f"PMC{pmcid}"
    return {"record": {"pmid": ids.get("pmid") or None, "doi": (ids.get("doi") or "").lower() or None,
                       "pmcid": pmcid or None, "title": title, "abstract": abstract or None, "authors": authors,
                       "journal": journal or None, "year": int(published[:4]) if published else None,
                       "published_date": published, "keywords": kws, "licence": lic.get("licence"),
                       "licence_url": lic.get("licence_url"), "reuse": fa.reuse(" ".join(x for x in (lic.get("licence"), lic.get("licence_url")) if x)),
                       "funding": funding or None, "coi": coi or None},
            "sections": merged}


def iter_jats_files(path: Path) -> Iterator[Tuple[str, bytes]]:
    """(name, xml bytes) from a .tar.gz package, a folder or a single file."""
    if path.is_dir():
        for p in sorted(list(path.rglob("*.xml")) + list(path.rglob("*.nxml"))):
            yield str(p), p.read_bytes()
    elif str(path).endswith((".tar.gz", ".tgz", ".tar")):
        with tarfile.open(path, "r:*") as tf:
            for m in tf:
                if m.isfile() and m.name.endswith((".xml", ".nxml")):
                    f = tf.extractfile(m)
                    if f is not None:
                        yield m.name, f.read()
    else:
        yield str(path), path.read_bytes()


# ================================================================================================ ClinicalTrials.gov
def parse_ctgov(study: Dict[str, Any]) -> Dict[str, Any]:
    p = study.get("protocolSection") or {}
    if not isinstance(p, dict):
        return {}
    ident, desc, status = p.get("identificationModule") or {}, p.get("descriptionModule") or {}, p.get("statusModule") or {}
    design, cond = p.get("designModule") or {}, p.get("conditionsModule") or {}
    arms, outc = p.get("armsInterventionsModule") or {}, p.get("outcomesModule") or {}
    spons, refs = p.get("sponsorCollaboratorsModule") or {}, p.get("referencesModule") or {}
    nct = ident.get("nctId")
    lead = spons.get("leadSponsor") or {}
    interventions = [i.get("name") for i in arms.get("interventions") or [] if i.get("name")]
    primary = [o.get("measure") for o in outc.get("primaryOutcomes") or [] if o.get("measure")]
    secondary = [o.get("measure") for o in outc.get("secondaryOutcomes") or [] if o.get("measure")]
    first_post = (status.get("studyFirstPostDateStruct") or {}).get("date")
    abstract = " ".join(x for x in [
        desc.get("briefSummary"),
        f"Status: {status.get('overallStatus')}." if status.get("overallStatus") else "",
        f"Study type: {design.get('studyType')}; phase: {', '.join(design.get('phases') or []) or 'n/a'}.",
        f"Conditions: {', '.join(cond.get('conditions') or [])}." if cond.get("conditions") else "",
        f"Interventions: {', '.join(interventions)}." if interventions else "",
        f"Registered primary outcome(s): {'; '.join(primary)}." if primary else "",
        "Results posted on ClinicalTrials.gov." if study.get("hasResults") else "No results posted on ClinicalTrials.gov."]
        if x)
    pubs = [r.get("pmid") for r in refs.get("references") or [] if r.get("pmid") and r.get("type") in ("RESULT", "DERIVED")]
    return {"nct": nct, "title": ident.get("officialTitle") or ident.get("briefTitle") or nct, "abstract": abstract,
            "authors": [lead.get("name")] if lead.get("name") else [], "journal": "ClinicalTrials.gov registry",
            "year": int(first_post[:4]) if first_post else None, "published_date": first_post,
            "indexed_date": first_post, "revised_date": (status.get("lastUpdatePostDateStruct") or {}).get("date"),
            "pub_types": ["Clinical Trial Registration"] + ([design["studyType"].title()] if design.get("studyType") else []),
            "mesh": list(cond.get("conditions") or []), "keywords": interventions,
            "status": "current", "licence": "ClinicalTrials.gov (public domain, U.S. government work)",
            "reuse": "public domain",
            "funding": {"lead_sponsor": lead.get("name"), "sponsor_class": lead.get("class"),
                        "collaborators": [c.get("name") for c in spons.get("collaborators") or []]},
            "registry": {"overall_status": status.get("overallStatus"), "primary_outcomes": primary,
                         "secondary_outcomes": secondary, "has_results": bool(study.get("hasResults")),
                         "result_pmids": pubs, "enrollment": (design.get("enrollmentInfo") or {}).get("count")}}


def fetch_ctgov(query: str, max_studies: int = 200, http: Any = None) -> List[Dict[str, Any]]:
    """Studies matching `query` from the ClinicalTrials.gov API v2 (paged)."""
    import requests
    out: List[Dict[str, Any]] = []
    token = None
    while len(out) < max_studies:
        params = {"query.term": query, "pageSize": min(100, max_studies - len(out)), "format": "json"}
        if token:
            params["pageToken"] = token
        r = requests.get(CTGOV_API, params=params, timeout=30)
        r.raise_for_status()
        js = r.json()
        out += js.get("studies") or []
        token = js.get("nextPageToken")
        if not token:
            break
        time.sleep(0.4)
    return out


# ================================================================================================ downloads
def list_remote(url: str, pattern: str) -> List[str]:
    """File names in an NCBI FTP-over-HTTPS directory listing that match `pattern` (sorted)."""
    import requests
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    return sorted(set(re.findall(r'href="(' + pattern + r')"', r.text)))


def download(url: str, dest: Path, md5_url: Optional[str] = None, retries: int = 4) -> Path:
    """Resumable download with an optional md5 check (NCBI publishes FILE.md5 next to each PubMed file)."""
    import requests
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")
    for attempt in range(retries):
        try:
            have = part.stat().st_size if part.exists() else 0
            headers = {"Range": f"bytes={have}-"} if have else {}
            with requests.get(url, stream=True, timeout=120, headers=headers) as r:
                if r.status_code == 416:
                    break
                r.raise_for_status()
                mode = "ab" if have and r.status_code == 206 else "wb"
                with open(part, mode) as fh:
                    for chunk in r.iter_content(1 << 20):
                        fh.write(chunk)
            break
        except Exception:
            if attempt == retries - 1:
                raise
            time.sleep(2 ** attempt * 5)
    if md5_url:
        want = requests.get(md5_url, timeout=60).text.strip().split()[-1].lower()
        got = hashlib.md5(part.read_bytes()).hexdigest()
        if want and want != got:
            part.unlink(missing_ok=True)
            raise ValueError(f"md5 mismatch for {dest.name}: expected {want}, got {got}")
    os.replace(part, dest)
    return dest


# ================================================================================================ loading
def _save_fulltext(lib: L.Library, name: str, data: bytes, pmcid: Optional[str]) -> Path:
    d = lib.dir / "fulltext"
    d.mkdir(parents=True, exist_ok=True)
    safe = re.sub(r"[^A-Za-z0-9_.-]", "_", pmcid or Path(name).stem)[:100]
    p = d / f"{safe}.xml"
    if not p.exists() or p.read_bytes() != data:
        p.write_bytes(data)
    return p


def load_pubmed(lib: L.Library, paths: Sequence[Path], scope: Optional[Scope] = None,
                log: Any = print) -> Dict[str, int]:
    scope = scope or Scope.load(lib)
    counts = {"new": 0, "updated": 0, "unchanged": 0, "deleted": 0, "out_of_scope": 0, "errors": 0}
    done = set(lib.get_sync("pubmed_files_done", []) or [])
    for path in paths:
        path = Path(path)
        n = 0
        try:
            items = list(_safe_iter(path, counts, lib))
        except Exception as exc:                     # a truncated / corrupt file: reported, NOT marked as applied
            counts["errors"] += 1
            counts.setdefault("bad_files", []).append(f"{path.name}: {type(exc).__name__}: {str(exc)[:120]}")
            lib.log(None, "load_pubmed", "error", LOADER_VERSION, f"{path.name}: {exc}")
            lib.rollback()
            continue
        for kind, val in items:
            if kind == "error":
                counts["errors"] += 1
                lib.log(None, "parse_pubmed", "error", LOADER_VERSION, f"{path.name}: {val}")
                continue
            if kind == "delete":
                counts["deleted"] += int(lib.mark_deleted(val, source=f"pubmed:{path.name}"))
                continue
            if not scope.matches(val):
                counts["out_of_scope"] += 1
                continue
            try:
                rid, how = lib.upsert(val, source="pubmed", source_url=f"https://pubmed.ncbi.nlm.nih.gov/{val.get('pmid')}/")
            except Exception as exc:                 # one bad record never stops the file
                counts["errors"] += 1
                lib.log(None, "upsert", "error", LOADER_VERSION, f"PMID {val.get('pmid')}: {type(exc).__name__}: {exc}")
                continue
            counts[how] = counts.get(how, 0) + 1
            if how in ("new", "updated") and val.get("abstract") and not lib.get(rid).get("fulltext_path"):
                lib.set_passages(rid, [("abstract", val["abstract"])])     # full-text passages are never replaced
            n += 1
            if n % 2000 == 0:
                lib.commit()
                log(f"  [Library] {path.name}: {n} records")
        done.add(path.name)
        lib.set_sync("pubmed_files_done", sorted(done))
        lib.log(None, "load_pubmed", "ok", LOADER_VERSION, f"{path.name}: {n} records in scope")
        lib.commit()
    lib.set_sync("last_update", datetime.now(timezone.utc).isoformat(timespec="seconds"))
    return counts


def _safe_iter(path: Path, counts: Dict[str, Any], lib: L.Library) -> Iterator[Tuple[str, Any]]:
    """Parse the WHOLE file before applying it, so a file that is cut off half-way changes nothing (it is retried)."""
    return iter(list(iter_pubmed(path)))


def load_jats(lib: L.Library, paths: Sequence[Path], scope: Optional[Scope] = None, log: Any = print) -> Dict[str, int]:
    scope = scope or Scope.load(lib)
    counts = {"new": 0, "updated": 0, "unchanged": 0, "out_of_scope": 0, "errors": 0}
    for path in paths:
        n = 0
        try:
            files = list(iter_jats_files(Path(path)))
        except Exception as exc:
            counts["errors"] += 1
            counts.setdefault("bad_files", []).append(f"{Path(path).name}: {type(exc).__name__}: {str(exc)[:120]}")
            continue
        for name, data in files:
            try:
                parsed = parse_jats(data.decode("utf-8", "replace"))
                if not (parsed["record"].get("title") or "").strip():
                    raise ValueError("no article title - not a JATS article")
            except Exception as exc:
                counts["errors"] += 1
                lib.log(None, "parse_jats", "error", LOADER_VERSION, f"{name}: {type(exc).__name__}: {exc}")
                continue
            rec = parsed["record"]
            known = lib.find(rec)
            if not known and not scope.matches(rec):
                counts["out_of_scope"] += 1
                continue
            fp = _save_fulltext(lib, name, data, rec.get("pmcid"))
            rec.update({"fulltext_path": str(fp), "fulltext_kind": "jats_xml"})
            try:
                rid, how = lib.upsert(rec, source="pmc", source_url=f"https://pmc.ncbi.nlm.nih.gov/articles/{rec.get('pmcid')}/"
                                      if rec.get("pmcid") else "")
            except Exception as exc:
                counts["errors"] += 1
                lib.log(None, "upsert", "error", LOADER_VERSION, f"{name}: {type(exc).__name__}: {exc}")
                continue
            counts[how] = counts.get(how, 0) + 1
            if how in ("new", "updated") or not lib.passages(rid):
                lib.set_passages(rid, parsed["sections"])
                lib.log(rid, "passages", "ok", LOADER_VERSION, f"{len(parsed['sections'])} sections")
            n += 1
            if n % 500 == 0:
                lib.commit()
                log(f"  [Library] {Path(path).name}: {n} full texts")
        lib.commit()
    lib.set_sync("last_update", datetime.now(timezone.utc).isoformat(timespec="seconds"))
    return counts


def load_trials(lib: L.Library, studies: Iterable[Dict[str, Any]]) -> Dict[str, int]:
    counts = {"new": 0, "updated": 0, "unchanged": 0, "errors": 0}
    for st in studies:
        try:
            rec = parse_ctgov(st) if isinstance(st, dict) else {}
        except Exception:
            rec = {}
        if not rec:
            counts["errors"] += 1
            continue
        if not rec.get("nct") or not rec.get("title"):
            counts["errors"] += 1                    # not a usable registry entry: counted, never silently skipped
            continue
        registry = rec.pop("registry")
        rid, how = lib.upsert(rec, source="clinicaltrials", source_url=f"https://clinicaltrials.gov/study/{rec['nct']}")
        counts[how] = counts.get(how, 0) + 1
        lib.set_sync(f"registry:{rec['nct']}", registry)
        lib.set_passages(rid, [("abstract", rec.get("abstract") or "")])
    lib.commit()
    return counts


def registry_for(nct: str) -> Optional[Dict[str, Any]]:
    """Registered outcomes / status / results flag of a trial from the library (no network)."""
    if not L.exists():
        return None
    with L.Library() as lib:
        return lib.get_sync(f"registry:{nct.upper()}")


# ================================================================================================ embeddings
def embed_pending(lib: L.Library, rebuild: bool = False, batch: int = 256, log: Any = print,
                  embedder: Any = None) -> Dict[str, Any]:
    import library_vectors as LV
    vec = LV.LibraryVectors(lib.dir / "vectors", embedder=embedder)
    if rebuild:
        for f in vec.dir.glob("*"):
            if f.name != "index.db":
                f.unlink()
        vec.c.execute("DELETE FROM vec")
        vec.c.execute("DELETE FROM meta")
        vec.c.commit()
    added = 0
    last = 0
    while True:
        rows = lib.c.execute("SELECT id, title, abstract, content_hash FROM records WHERE deleted=0 AND id > ? ORDER BY id "
                             "LIMIT 5000", (last,)).fetchall()
        if not rows:
            break
        last = rows[-1]["id"]
        items = [{"record_id": r["id"], "text_hash": r["content_hash"],
                  "text": f"{r['title'] or ''}. {r['abstract'] or ''}"} for r in rows]
        todo = vec.missing(items)
        if todo:
            added += vec.add(todo, batch=batch)
            log(f"  [Library] embedded {added} records")
    for r in lib.c.execute("SELECT id FROM records WHERE deleted=1"):
        vec.remove(r["id"])
    info = vec.info()
    vec.close()
    return {"embedded": added, **info}


# ================================================================================================ the daily job
def update(lib: Optional[L.Library] = None, max_files: int = 0, fetch: bool = True, log: Any = print) -> Dict[str, Any]:
    """Apply every PubMed update file not applied yet, then embed new / changed records and refresh the run
    database's publication statuses for the papers that changed. Safe to run as often as you like."""
    own = lib is None
    lib = lib or L.Library()
    res: Dict[str, Any] = {"files": [], "counts": {}}
    try:
        if fetch:
            done = set(lib.get_sync("pubmed_files_done", []) or [])
            names = [n for n in list_remote(PUBMED_UPDATES, r"pubmed\d+n\d+\.xml\.gz") if n not in done]
            if max_files:
                names = names[:max_files]
            dl = lib.dir / "downloads"
            for n in names:
                p = download(PUBMED_UPDATES + n, dl / n, md5_url=PUBMED_UPDATES + n + ".md5")
                c = load_pubmed(lib, [p], log=log)
                res["files"].append(n)
                for k, v in c.items():
                    res["counts"][k] = res["counts"].get(k, 0) + v
                p.unlink(missing_ok=True)          # applied: the record is in the library, the file is not needed
        res["embed"] = embed_pending(lib, log=log)
        res["status_sync"] = sync_status_to_runs(lib)
        lib.set_sync("last_update", datetime.now(timezone.utc).isoformat(timespec="seconds"))
        lib.log(None, "update", "ok", LOADER_VERSION, json.dumps(res["counts"]))
        lib.commit()
        return res
    finally:
        if own:
            lib.close()


def sync_status_to_runs(lib: L.Library) -> Dict[str, int]:
    """A paper retracted / corrected in the library is flagged in the run database too (with its dependent claims)."""
    try:
        from database import ResearchDB
        import publication_status as ps
    except Exception:
        return {"checked": 0}
    since = lib.get_sync("status_synced_at") or "0"
    rows = lib.c.execute("SELECT r.id, r.pmid, r.doi, r.status, r.notices_json FROM records r JOIN record_history h ON "
                         "h.record_id = r.id WHERE h.changed_at > ? AND r.status != 'current' GROUP BY r.id", (since,)).fetchall()
    n = 0
    if rows:
        with ResearchDB() as db:
            ps.ensure(db)
            for r in rows:
                pid = db._conn.execute("SELECT id FROM papers WHERE (pmid IS NOT NULL AND pmid=?) OR (doi IS NOT NULL AND "
                                       "lower(doi)=?)", (r["pmid"], (r["doi"] or "").lower())).fetchone()
                if pid:
                    ps.check_paper(db, pid[0], fetch=False)            # reads the library's notices (no network)
                    n += 1
    lib.set_sync("status_synced_at", datetime.now(timezone.utc).isoformat(timespec="seconds"))
    return {"run_papers_flagged": n}


# ================================================================================================ CLI
def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Build and update the local BioSynthetica literature library")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("scope")
    s.add_argument("--mesh", default="")
    s.add_argument("--keywords", default="")
    s.add_argument("--journals", default="")
    s.add_argument("--pub-types", default="")
    s.add_argument("--year-from", type=int)
    s.add_argument("--year-to", type=int)
    d = sub.add_parser("download")
    d.add_argument("what", choices=["pubmed-baseline", "pubmed-updates", "pmc-oa"])
    d.add_argument("--max-files", type=int, default=0)
    d.add_argument("--subset", default="oa_comm", choices=["oa_comm", "oa_noncomm", "oa_other"])
    d.add_argument("--keep", action="store_true", help="keep the downloaded files after loading")
    for name in ("pubmed", "pmc"):
        x = sub.add_parser(name)
        x.add_argument("paths", nargs="+")
    t = sub.add_parser("trials")
    t.add_argument("file", nargs="?")
    t.add_argument("--query")
    t.add_argument("--max", type=int, default=500)
    e = sub.add_parser("embed")
    e.add_argument("--rebuild", action="store_true")
    e.add_argument("--hnsw", action="store_true")
    p = sub.add_parser("preread")
    p.add_argument("--limit", type=int, default=200)
    p.add_argument("--designs", default="rct,meta")
    sub.add_parser("update")
    sub.add_parser("status")
    a = ap.parse_args(argv)
    with L.Library() as lib:
        if a.cmd == "scope":
            sp = lambda v: [x for x in re.split(r"[;\n]", v or "") if x.strip()]   # noqa: E731
            sc = Scope(sp(a.mesh), sp(a.keywords), sp(a.journals), a.year_from, a.year_to, sp(a.pub_types))
            lib.set_sync("scope", sc.as_dict())
            print(json.dumps(sc.as_dict(), indent=1))
        elif a.cmd == "download":
            dl = lib.dir / "downloads"
            if a.what.startswith("pubmed"):
                base = PUBMED_BASELINE if a.what == "pubmed-baseline" else PUBMED_UPDATES
                done = set(lib.get_sync("pubmed_files_done", []) or [])
                names = [n for n in list_remote(base, r"pubmed\d+n\d+\.xml\.gz") if n not in done]
                for n in names[:a.max_files or None]:
                    print(f"  downloading {n}")
                    f = download(base + n, dl / n, md5_url=base + n + ".md5")
                    print(json.dumps(load_pubmed(lib, [f])))
                    if not a.keep:
                        f.unlink(missing_ok=True)
            else:
                base = PMC_OA_BULK.format(subset=a.subset)
                names = list_remote(base, r"[\w.\-]+\.tar\.gz")
                for n in names[:a.max_files or None]:
                    print(f"  downloading {n}")
                    f = download(base + n, dl / n)
                    print(json.dumps(load_jats(lib, [f])))
                    if not a.keep:
                        f.unlink(missing_ok=True)
        elif a.cmd == "pubmed":
            print(json.dumps(load_pubmed(lib, [Path(x) for x in a.paths])))
        elif a.cmd == "pmc":
            print(json.dumps(load_jats(lib, [Path(x) for x in a.paths])))
        elif a.cmd == "trials":
            studies = json.loads(Path(a.file).read_text(encoding="utf-8")) if a.file else fetch_ctgov(a.query or "", a.max)
            if isinstance(studies, dict):
                studies = studies.get("studies") or [studies]
            print(json.dumps(load_trials(lib, studies)))
        elif a.cmd == "embed":
            print(json.dumps(embed_pending(lib, rebuild=a.rebuild)))
            if a.hnsw:
                import library_vectors as LV
                print(json.dumps(LV.LibraryVectors(lib.dir / "vectors").build_hnsw()))
        elif a.cmd == "preread":
            import preread
            print(json.dumps(preread.run(lib, limit=a.limit, designs=[x.strip() for x in a.designs.split(",") if x.strip()])))
        elif a.cmd == "update":
            print(json.dumps(update(lib), default=str))
        elif a.cmd == "status":
            st = lib.stats()
            try:
                import library_vectors as LV
                st["vectors"] = LV.LibraryVectors(lib.dir / "vectors").info()
            except Exception as exc:
                st["vectors"] = {"error": str(exc)}
            print(json.dumps(st, indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
