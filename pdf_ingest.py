"""
pdf_ingest.py
-------------
Ingest an open-access PDF (from Unpaywall) into ResearchDB using the same
section model as jats_ingest.py, so PaperExtractor / ClaimExtractor treat it
exactly like full-text XML (basis = "full_text").
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from jats_ingest import classify_section_title

HEADING_RE = re.compile(
    r"^\s*(?:\d{1,2}(?:\.\d{1,2})*\.?\s+)?"
    r"(abstract|summary|introduction|background|materials?\s+and\s+methods|methods?|methodology|"
    r"experimental(?:\s+procedures|\s+section)?|patients\s+and\s+methods|results?(?:\s+and\s+discussion)?|"
    r"findings|discussion|conclusions?|limitations?|references(?:\s+and\s+notes)?|literature\s+cited|bibliography|works\s+cited|acknowledg(?:e)?ments?)\s*:?\s*$",
    re.IGNORECASE,
)


def extract_pdf_text(pdf_path: str | Path, max_pages: int = 40) -> str:
    from pypdf import PdfReader

    reader = PdfReader(str(pdf_path))
    pages = []
    for page in reader.pages[:max_pages]:
        try:
            pages.append(page.extract_text() or "")
        except Exception:
            continue
    text = "\n".join(pages)
    text = re.sub(r"-\n(?=[a-z])", "", text)        # de-hyphenate line breaks
    text = re.sub(r"[ \t]+", " ", text)
    return text


def split_sections(text: str) -> List[Tuple[str, str, str]]:
    """Return [(section_type, title, body)] using common scientific headings."""
    sections: List[Tuple[str, str, List[str]]] = []
    cur_type, cur_title, buf = "other", "Body", []
    for line in text.splitlines():
        m = HEADING_RE.match(line)
        if m:
            if buf:
                sections.append((cur_type, cur_title, buf))
            title = m.group(1).strip().title()
            low = title.lower()
            if low.startswith("result") and "discussion" in low:
                cur_type = "results_discussion"
            elif low in ("abstract", "summary"):
                cur_type = "abstract"
            elif low.startswith(("reference", "acknowledg", "literature cited", "bibliograph", "works cited")):
                cur_type = "references"
            else:
                cur_type = classify_section_title(title, cur_type)
            cur_title, buf = title, []
        else:
            if line.strip():
                buf.append(line.strip())
    if buf:
        sections.append((cur_type, cur_title, buf))

    out = []
    for stype, title, lines in sections:
        if stype == "references":
            continue
        body = " ".join(lines)
        # paragraphs roughly every ~6 sentences so ClaimExtractor gets chunks
        sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z])", body)
        paras = [" ".join(sentences[i:i + 6]) for i in range(0, len(sentences), 6)]
        out.append((stype, title, "\n\n".join(p for p in paras if p.strip())))
    return out


def ingest_pdf(
    db: Any,
    pdf_path: str | Path,
    paper: Dict[str, Any],
    research_question_id: Optional[int] = None,
    source_query: str = "",
) -> Optional[int]:
    """Store paper metadata + parsed PDF sections. Returns paper id, or None if no usable text."""
    text = extract_pdf_text(pdf_path)
    if len(text) < 1500:   # scanned image PDF or failed extraction
        return None
    sections = split_sections(text)
    if not any(s[0] in ("results", "results_discussion", "methods", "discussion", "other") for s in sections):
        return None

    pid = db.upsert_paper(research_question_id, paper, source_query=source_query)
    db.clear_paper_sections(pid)
    db.clear_paper_tables(pid)
    has_abstract = False
    for stype, title, body in sections:
        if not body:
            continue
        if stype == "abstract":
            has_abstract = True
        db.save_paper_section(pid, stype, title, body[:60000])
    if not has_abstract and paper.get("abstract"):
        db.save_paper_section(pid, "abstract", "Abstract", paper["abstract"])
    return pid
