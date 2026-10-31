"""
passage_select.py  (v26)
========================
Selective passage retrieval for extraction: instead of the abstract + the FIRST four results paragraphs (v25), every
results / discussion / results-and-discussion / table paragraph of the paper is scored against the question and the
best ones are sent to the AI, in their original order.

    select(question, parts, max_paras=6) -> [(chunk_type, text)]

Rules (accuracy before brevity):
  * the unit is a whole PARAGRAPH (never a clipped sentence), so negations and qualifiers ('... but not in women
    older than 65', 'did not reach significance') stay with the result they qualify;
  * the abstract is always kept;
  * score = BM25-style term overlap with the question's PICO concepts (synonyms included) + a bonus for paragraphs that
    REPORT results (p-values, intervals, percentages, 'increased / did not differ') + a bonus for tables;
  * near-duplicate paragraphs (word-set Jaccard >= 0.8, e.g. a results sentence repeated in the discussion) are kept
    once - the one that reports more numbers;
  * a short limitations paragraph, when the paper has one, is appended as context (it is not a finding, and the
    extraction prompt keeps it out of the findings);
  * when the question gives no usable terms, the v25 behaviour (first paragraphs) is kept.
"""
from __future__ import annotations

import math
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

SELECT_VERSION = "1.0"
_RESULT = re.compile(r"\bp\s*[<=>≤≥]\s*0?\.\d|\b(?:95|90|99)\s*%\s*(?:CI|confidence)|\b\d+(?:\.\d+)?\s*%|"
                     r"\b(?:increased|decreased|reduced|higher|lower|did not|no significant|not significantly|"
                     r"differ(?:ed|ence)|odds ratio|hazard ratio|relative risk|mean difference)\b", re.I)
_WORD = re.compile(r"[a-z0-9][a-z0-9\-]+")


def _paras(text: str) -> List[str]:
    out = []
    for p in re.split(r"\n\s*\n", text or ""):
        p = re.sub(r"^\s*\[[^\]\n]{1,200}\]\s*\n", "", p.strip())
        if len(p) > 80:
            out.append(p)
    return out


def _words(t: str) -> set:
    return set(_WORD.findall((t or "").lower()))


def _terms(question: str) -> List[set]:
    """Concept word groups; the population groups are marked so they never make a paragraph relevant on their own."""
    try:
        import query_parser as qp
        concepts = qp.search_concepts(qp.parse(question))
        groups = []
        for c in concepts:
            g = {w for term in c["terms"] for w in _WORD.findall(term.lower()) if len(w) > 2}
            if g:
                groups.append(_Group(g, c.get("role") in ("intervention", "exposure", "outcome")))
        return groups
    except Exception:
        return [_Group({w for w in _WORD.findall((question or "").lower()) if len(w) > 3}, True)]


class _Group(set):
    def __init__(self, words, core: bool):
        super().__init__(words)
        self.core = core


def relevant(paragraph: str, groups: Sequence[set]) -> bool:
    """A paragraph is a candidate only if it names the intervention / exposure / outcome, or reports a result -
    a paragraph that only mentions the population ('women', 'patients') is not chosen."""
    w = _words(paragraph)
    return any(getattr(g, "core", True) and g & w for g in groups) or bool(_RESULT.search(paragraph))


def score(paragraph: str, groups: Sequence[set], is_table: bool = False) -> float:
    w = _words(paragraph)
    concept = sum(1.0 for g in groups if g & w) / max(1, len(groups))
    hits = sum(len(g & w) for g in groups)
    lexical = math.log1p(hits)
    result = min(1.0, len(_RESULT.findall(paragraph)) / 3.0)
    return 1.5 * concept + 0.5 * lexical + 1.0 * result + (0.3 if is_table else 0.0)


def select(question: Optional[str], parts: Dict[str, Any], max_paras: int = 6) -> List[Tuple[str, str]]:
    chunks: List[Tuple[str, str]] = []
    if parts.get("abstract"):
        chunks.append(("Abstract", parts["abstract"]))
    pool: List[Tuple[int, str, str]] = []           # (order, type, text)
    order = 0
    for key, label in (("results", "Results Paragraph"), ("results_discussion", "Results Paragraph"),
                       ("discussion", "Discussion Paragraph"), ("other", "Results Paragraph")):
        for p in _paras(parts.get(key) or ""):
            pool.append((order, label, p))
            order += 1
    for t in re.split(r"\n\s*\n(?=Table|\S+:)", parts.get("tables") or ""):
        if len(t.strip()) > 40:
            pool.append((order, "Table", t.strip()[:3000]))
            order += 1
    groups = _terms(question or "") if question else []
    if not groups:
        body = [x for x in pool if x[1] != "Table" and x[1] != "Discussion Paragraph"] or pool
        return chunks + [(lbl, txt) for _, lbl, txt in body[:4]]
    scored = sorted([x for x in pool if relevant(x[2], groups)], key=lambda x: -score(x[2], groups, x[1] == "Table"))
    picked: List[Tuple[int, str, str]] = []
    for cand in scored:
        if len(picked) >= max_paras:
            break
        cw = _words(cand[2])
        dup = next((i for i, p in enumerate(picked)
                    if len(cw & _words(p[2])) / max(1, len(cw | _words(p[2]))) >= 0.8), None)
        if dup is not None:
            if len(_RESULT.findall(cand[2])) > len(_RESULT.findall(picked[dup][2])):
                picked[dup] = cand
            continue
        picked.append(cand)
    picked.sort(key=lambda x: x[0])
    chunks += [(lbl, txt) for _, lbl, txt in picked]
    lim = _paras(parts.get("limitations") or "")
    if lim:
        chunks.append(("Limitations (context only - not a finding)", lim[0][:1200]))
    return chunks
