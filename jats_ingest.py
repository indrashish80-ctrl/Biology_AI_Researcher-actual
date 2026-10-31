"""
jats_ingest.py
--------------
JATS/XML paper ingestion into ResearchDB with robust section classification,
structured 2D table grid extraction (including colspan flattening), 
and memory-clearing DOM disposal.
"""

from __future__ import annotations

import gc
import re
import xml.etree.ElementTree as ET
from typing import Any, List


def lname(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def text_of(el: ET.Element) -> str:
    return re.sub(r"\s+", " ", " ".join(el.itertext())).strip()


def classify_section_title(title: str, current_key: str = "") -> str:
    low = title.lower()
    if any(w in low for w in ("method", "material", "experimental", "procedure", "protocol", "assay", "culture")):
        return "methods"
    if any(w in low for w in ("result", "finding", "effect", "response", "growth", "biomass", "expression", "accumulation", "analysis", "observation", "characterization")):
        return "results"
    if any(w in low for w in ("discussion", "perspective", "implication")):
        return "discussion"
    if any(w in low for w in ("intro", "background")):
        return "introduction"
    if any(w in low for w in ("conclusion", "summary", "concluding")):
        return "conclusion"
    if "limitation" in low:
        return "limitations"
    return current_key or "other"


def ingest_jats(db: Any, xml_path: str, research_question_id: int | None = None, source_query: str = "") -> int:
    tree = ET.parse(xml_path)
    root = tree.getroot()
    title = ""
    abstract = ""
    pmid = pmcid = doi = None
    authors = []
    year = None
    journal = None

    # Metadata comes from <front> only: names inside <ref-list> are CITED authors, and
    # article-ids inside references belong to other papers.
    front = next((x for x in root.iter() if lname(x.tag) == "front"), root)
    for el in front.iter():
        t = lname(el.tag)
        if t == "article-title" and not title:
            title = text_of(el)
        elif t == "abstract" and not abstract and el.attrib.get("abstract-type") not in ("graphical", "teaser"):
            abstract = text_of(el)
        elif t == "journal-title" and not journal:
            journal = text_of(el)
        elif t == "pub-date":
            # the EARLIEST real publication date
            if (el.attrib.get("pub-type") or el.attrib.get("date-type") or "") in ("pmc-release", "nihms-submitted"):
                continue
            from retrieval.models import plausible_year
            y = plausible_year(next((text_of(x) for x in el if lname(x.tag) == "year"), ""))
            if y and (year is None or y < year):
                year = y
        elif t == "article-id":
            typ = el.attrib.get("pub-id-type")
            val = text_of(el)
            if typ == "pmid" and not pmid:
                pmid = val
            elif typ in ("pmcid", "pmc") and not pmcid:
                pmcid = val if val.upper().startswith("PMC") else f"PMC{val}"
            elif typ == "doi" and not doi:
                doi = val
        elif t == "contrib" and el.attrib.get("contrib-type", "author") == "author":
            name = next((x for x in el.iter() if lname(x.tag) == "name"), None)
            if name is not None:
                sn = next((text_of(x) for x in name if lname(x.tag) == "surname"), "")
                gn = next((text_of(x) for x in name if lname(x.tag) == "given-names"), "")
                if sn:
                    authors.append((gn + " " + sn).strip())

    paper = {
        "title": title,
        "authors": ", ".join(authors),
        "abstract": abstract,
        "pmid": pmid,
        "pmcid": pmcid,
        "doi": doi,
        "is_open_access": True,
        "full_text": None,
        "year": year,
        "journal": journal,
    }
    pid = db.upsert_paper(research_question_id, paper, source_query=source_query)
    db.clear_paper_sections(pid)
    db.clear_paper_tables(pid)

    art = next((x for x in root.iter() if lname(x.tag) == "article"), root)
    atype = (art.attrib.get("article-type") or "").strip().lower()
    if atype:
        db.save_paper_section(pid, "article_meta", "article-type", atype)

    section_map = {
        "intro": "introduction", "introduction": "introduction",
        "methods": "methods", "materials|methods": "methods", "materials": "methods",
        "results": "results", "discussion": "discussion",
        "conclusions": "conclusion", "conclusion": "conclusion",
        "limitations": "limitations", "results_discussion": "results_discussion",
    }

    def extract_structured_rows(tbl: ET.Element) -> List[List[str]]:
        """Extracts table rows, handling colspan by duplicating cells across the span."""
        rows = []
        for tr in tbl.iter():
            if lname(tr.tag) == "tr":
                cell_vals = []
                for c in tr:
                    if lname(c.tag) in ("th", "td"):
                        text = text_of(c)
                        # Address table misalignment from merged headers
                        try:
                            colspan = int(c.attrib.get("colspan", "1"))
                        except ValueError:
                            colspan = 1
                        cell_vals.extend([text] * max(1, colspan))
                if cell_vals:
                    rows.append(cell_vals)
        return rows

    def save_table(db: Any, pid: int, tbl_wrap: ET.Element):
        label = next((text_of(x) for x in tbl_wrap if lname(x.tag) == "label"), "")
        cap = next((text_of(x) for x in tbl_wrap if lname(x.tag) == "caption"), "")
        table = next((x for x in tbl_wrap.iter() if lname(x.tag) == "table"), None)
        if table is not None:
            rows = extract_structured_rows(table)
            if rows:
                db.save_paper_table(pid, label, cap, rows)

    def walk_secs(parent, current=""):
        for sec in parent:
            if lname(sec.tag) != "sec":
                continue
            title_el = next((x for x in sec if lname(x.tag) == "title"), None)
            stitle = text_of(title_el) if title_el is not None else ""
            sec_type = sec.attrib.get("sec-type", "").lower()

            key = section_map.get(sec_type) or section_map.get(re.sub(r"\s+", "_", stitle.lower()))
            if not key:
                key = classify_section_title(stitle, current)

            direct = []
            for child in sec:
                ct = lname(child.tag)
                if ct in ("p", "disp-formula", "statement"):
                    txt = text_of(child)
                    if txt:
                        direct.append(txt)

            if direct:
                db.save_paper_section(pid, key, stitle, "\n".join(direct))

            for tbl in sec:
                if lname(tbl.tag) == "table-wrap":
                    save_table(db, pid, tbl)

            walk_secs(sec, key)

    body = next((x for x in root.iter() if lname(x.tag) == "body"), None)
    if body is not None:
        walk_secs(body)
        loose = [text_of(p) for p in body if lname(p.tag) == "p"]
        loose = [t for t in loose if t]
        if loose:
            db.save_paper_section(pid, "other", "Body", "\n".join(loose))
        for tbl in body:
            if lname(tbl.tag) == "table-wrap":
                save_table(db, pid, tbl)

    for grp in root.iter():
        if lname(grp.tag) in ("floats-group", "back"):
            for tbl in grp.iter():
                if lname(tbl.tag) == "table-wrap":
                    save_table(db, pid, tbl)

    if abstract:
        db.save_paper_section(pid, "abstract", "Abstract", abstract)

    # MEMORY OPTIMIZATION: Clear and dereference DOM tree
    root.clear()
    del root, tree
    gc.collect()

    return pid