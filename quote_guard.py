"""
quote_guard.py  (v25)
=====================
The strongest single guard against a hallucinated result: every extracted finding must carry the EXACT sentence it
rests on, and code - not the model - checks it.

    locate(quote, source)          -> where the quote is in the paper, verbatim (or None)
    numbers_check(claim, quote)    -> every number the finding states is in its quote (or the paper)
    guard_claim(claim, source)     -> one decision per finding: keep (with the verbatim quote), fix numbers, or drop
    source_confidence(basis, ...)  -> high / moderate / low from the kind of text the finding was read from

Rules (all deterministic, no AI call):

1. A quote is accepted when, after harmless normalisation (case, curly quotes, dashes, ligatures, white space, soft
   hyphens, 'P<0.05' vs 'P < 0.05'), it is a substring of the paper. The stored quote is then replaced by the paper's
   own characters at that position, so what the reader sees is what the paper says.
2. A quote that is not verbatim but is within one or two small edits of exactly ONE sentence of the paper (the model
   dropped a bracket, changed 'tumor' to 'tumour') is SNAPPED to that sentence, provided every number in it is the same.
   Paraphrases, merged sentences and quotes whose numbers differ are never snapped.
3. 'part one ... part two' quotes are accepted when every part is verbatim, in order, inside one paragraph.
4. A finding whose quote cannot be located is DROPPED (not stored, counted in the run's guard statistics).
5. Numbers: the p-value, effect size and confidence interval a finding carries must be the ones its quote reports;
   when they differ the quote's numbers (parsed by code) replace the model's and the finding is flagged. Any number
   in the finding's own sentence that is in neither the quote nor the paper is a made-up number: the finding is dropped.
6. Confidence is lower for abstract-only reading (the least accurate mode) and higher for full-text XML, where
   sections and tables are clean. It is a label shown next to the finding, never a reason to hide it.
"""
from __future__ import annotations

import difflib
import re
import unicodedata
from typing import Any, Dict, List, Optional, Tuple

GUARD_VERSION = "1.1"   # 1.1: leading minus, mid-dot decimals, scientific notation

# ----------------------------------------------------------------------------------------------- normalisation
_CHAR_MAP = {
    "‘": "'", "’": "'", "‚": "'", "‛": "'", "′": "'", "´": "'", "`": "'",
    "“": '"', "”": '"', "„": '"', "″": '"',
    "‐": "-", "‑": "-", "‒": "-", "–": "-", "—": "-", "―": "-", "−": "-",
    "﹣": "-", "－": "-",
    " ": " ", " ": " ", " ": " ", " ": " ", " ": " ", " ": " ", " ": " ",
    "×": "x", "≤": "<=", "≥": ">=", "≦": "<=", "≧": ">=", "±": "+/-",
    "µ": "u", "μ": "u",
}
_DROP = {"­", "​", "‌", "‍", "﻿"}


def _norm_char(ch: str) -> str:
    if ch in _DROP:
        return ""
    if ch in _CHAR_MAP:
        return _CHAR_MAP[ch]
    if ord(ch) < 128:
        return ch.lower()
    k = unicodedata.normalize("NFKD", ch)
    k = "".join(c for c in k if not unicodedata.combining(c))
    return (k or ch).lower()


def normalise_with_map(text: str) -> Tuple[str, List[int]]:
    """The normalised text and, for every normalised character, the index of the original character it came from.
    White space runs become one space; spaces next to < > = are removed ('P < 0.05' == 'P<0.05')."""
    out: List[str] = []
    idx: List[int] = []
    prev_space = True
    text = text or ""
    for i, ch in enumerate(text):
        if ch in "\u00b7\u2219\u22c5" and 0 < i < len(text) - 1 and text[i - 1].isdigit() and text[i + 1].isdigit():
            n = "."                                   # Lancet mid-dot decimal: '0\u00b719' is 0.19, not 0 and 19
        else:
            n = _norm_char(ch)
        for c in n:
            if c.isspace():
                if prev_space:
                    continue
                out.append(" ")
                idx.append(i)
                prev_space = True
            else:
                out.append(c)
                idx.append(i)
                prev_space = False
    s = "".join(out)
    # remove spaces around comparison signs and inside '( ' / ' )', keeping the index map aligned
    keep = [True] * len(s)
    for j, c in enumerate(s):
        if c == " ":
            left = s[j - 1] if j > 0 else ""
            right = s[j + 1] if j + 1 < len(s) else ""
            if left in "<>=(/[" or right in "<>=),;:.%/]":
                keep[j] = False
    s2 = "".join(c for c, k in zip(s, keep) if k)
    idx2 = [i for i, k in zip(idx, keep) if k]
    if s2.endswith(" "):
        s2, idx2 = s2[:-1], idx2[:-1]
    return s2, idx2


def normalise(text: str) -> str:
    return normalise_with_map(text)[0]


def _strip_quote(q: str) -> str:
    q = (q or "").strip()
    q = re.sub(r"^[\s\"'“”‘’\[(]+|[\s\"'“”‘’\])]+$", "", q)
    return q.strip()


# ----------------------------------------------------------------------------------------------- numbers
# a number that stands on its own: '40%', '0.05', '1,700 mg', '-5.6', both ends of '0.65-0.93' - never part of a name
# ('Ki-67', 'MCF-7', 'p53', 'IL-6', 'CD8', 'H3K27', 'COVID-19', '4T1', 'H1N1', '3D')
_CAND = re.compile(r"\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?|(?<!\d)\.\d+")
_UNIT_AFTER = re.compile(r"(?:%|mg|g|kg|mcg|ug|ng|pg|ml|l|dl|ul|mm|cm|m|nm|um|µm|mmol|umol|nmol|pmol|mol|mm|h|hr|hrs|min|"
                         r"s|sec|d|wk|wks|mo|y|yr|yrs|x|fold|iu|u|kda|da|bp|kb|mb|gy|mgy|kcal|cal|mmhg|bpm|ppm|ppb|"
                         r"m2|kg/m2|°c|c|k|v|mv|hz|g/l|mg/dl|mg/kg|nm/l)\b", re.I)


_SCI = re.compile(r"(?<![\w.])(-?\d+(?:\.\d+)?)\s*(?:x|\*)\s*10\s*\^?\s*([-+]?\d+)(?!\d)|"
                  r"(?<![\w.])(-?\d+(?:\.\d+)?)e([-+]?\d+)(?![\w])")


def numbers_of(text: str) -> List[float]:
    vals: List[float] = []
    t = normalise(text or "")             # 'Ki‑67' (non-breaking hyphen) must read like 'Ki-67'
    # scientific notation first ('1.2x10^5', '3e-4'), then masked so its pieces are not read as more numbers
    def _sci(m: "re.Match[str]") -> str:
        mant, exp = (m.group(1), m.group(2)) if m.group(1) is not None else (m.group(3), m.group(4))
        try:
            vals.append(float(f"{mant}e{int(exp)}"))          # exact: 3e-4 is 0.0003, not 0.00030000000000000003
        except (ValueError, OverflowError):
            pass
        return " " * len(m.group(0))
    t = _SCI.sub(_sci, t)
    for m in _CAND.finditer(t):
        a, b = m.start(), m.end()
        before = t[a - 1] if a > 0 else ""
        before2 = t[a - 2] if a > 1 else ""   # NB "" in "-_" is True in Python: test before2 is non-empty
        if before.isalpha() or before == "_" or (before == "." and before2.isalpha()):
            continue                                  # p53, CD8, v2.0
        if before == "-" and before2 and (before2.isalpha() or before2 in "-_"):
            continue                                  # Ki-67, MCF-7, IL-6, COVID-19
        after = t[b:b + 12]
        if text and after[:1] == "d":
            # '3D' (three-dimensional) is a name, '3 d' / '3d' are days: decide on the ORIGINAL letter case
            orig = re.search(re.escape(m.group(0)) + r"\s*([Dd])(?![A-Za-z])", text)
            if orig and orig.group(1) == "D":
                continue
        if after[:1].isalpha():
            if re.match(r"[A-Za-z]+\d", after) or not _UNIT_AFTER.match(after):
                continue                              # 4T1, 3D, 5th - unless a unit is glued on ('40mg')
        if after[:1] == "-" and after[1:2].isalpha() and not _UNIT_AFTER.match(after[1:]):
            continue                                  # '2-AG', '5-FU' are names
        raw = m.group(0).replace(",", "")
        neg = before == "-" and (before2 in (" ", "(", "[", "") or a == 1)
        try:
            v = float(raw)
        except ValueError:
            continue
        vals.append(-v if neg else v)
    return vals


def _same_number(a: float, b: float) -> bool:
    if abs(a - b) <= max(1e-9, 1e-6 * max(abs(a), abs(b))):
        return True
    # 40% written as 0.40 (or the other way round); a rounded value (0.031 -> 0.03) is NOT the same
    for f in (100.0, 0.01):
        if abs(a * f - b) <= max(1e-9, 1e-6 * abs(b)):
            return True
    return False


def number_in(value: float, numbers: List[float]) -> bool:
    return any(_same_number(value, n) for n in numbers)


# ----------------------------------------------------------------------------------------------- sentences
_ABBR = r"(?:e\.g|i\.e|et al|vs|fig|figs|ref|refs|no|approx|ca|cf|resp|incl|al|dr|mr|ms|st|wk|wks|yr|yrs|mo)"
_SENT_END = re.compile(r"(?<=[.!?])[\"')\]]*\s+(?=[A-Z0-9(\[\"'])")


def sentences(text: str) -> List[Tuple[int, int]]:
    """(start, end) spans of the sentences of `text` in ORIGINAL character positions."""
    t = text or ""
    spans: List[Tuple[int, int]] = []
    start = 0
    for m in _SENT_END.finditer(t):
        before = t[max(start, m.start() - 12):m.start()]
        if re.search(rf"\b{_ABBR}\.?[\"')\]]*$", before, re.I) or re.search(r"\b[A-Z]\.$", before) \
                or re.search(r"\d\.$", before) and re.match(r"\s+\d", t[m.start():]):
            continue
        spans.append((start, m.start()))
        start = m.end()
    if start < len(t):
        spans.append((start, len(t)))
    out = []
    for a, b in spans:
        seg = t[a:b]
        lead = len(seg) - len(seg.lstrip())
        lab = _LABEL.match(seg[lead:])            # 'Results: Metformin ...' -> the sentence starts at 'Metformin'
        if lab:
            lead += lab.end()
        trail = len(seg.rstrip())
        if trail > lead:
            out.append((a + lead, a + trail))
    return out


_LABEL = re.compile(r"(?:background|introduction|objectives?|aims?|purpose|methods?|materials and methods|design|setting|"
                    r"participants|patients|interventions?|main outcome measures?|measurements|results?|findings|"
                    r"conclusions?|interpretation|significance|context|importance|limitations)\s*[:.]\s+", re.I)


# ----------------------------------------------------------------------------------------------- locate
def locate(quote: str, source: str) -> Optional[Dict[str, Any]]:
    """Where `quote` is in `source`: {'mode': 'exact'|'ellipsis', 'start', 'end', 'verbatim'} or None."""
    q = _strip_quote(quote)
    if len(q) < 12 or not source:
        return None
    ns, imap = normalise_with_map(source)
    parts = [p.strip(" .;,") for p in re.split(r"\s*(?:\.\.\.|…|\[\s*\.\.\.\s*\])\s*", q) if p.strip(" .;,")]
    if not parts:
        return None
    if len(parts) == 1:
        nq = normalise(parts[0]).rstrip(" .;,")
        if len(nq) < 12:
            return None
        k = ns.find(nq)
        if k < 0:
            return None
        a, b = imap[k], imap[k + len(nq) - 1] + 1
        return {"mode": "exact", "start": a, "end": b, "verbatim": source[a:b]}
    # ellipsis: every part verbatim, in order, within one paragraph (<= 1,500 characters apart)
    pos, first, last = 0, None, None
    for p in parts:
        nq = normalise(p)
        if len(nq) < 6:
            return None
        k = ns.find(nq, pos)
        if k < 0 or (first is not None and k - pos > 1500):
            return None
        first = k if first is None else first
        last = k + len(nq)
        pos = last
    a, b = imap[first], imap[last - 1] + 1
    return {"mode": "ellipsis", "start": a, "end": b, "verbatim": source[a:b]}


def _ratio(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, a, b, autojunk=False).ratio()


def snap(quote: str, source: str, min_ratio: float = 0.93) -> Optional[Dict[str, Any]]:
    """The ONE sentence of the paper the quote is a near-verbatim copy of, or None.

    Accepted only when (a) the best sentence is >= min_ratio similar, (b) it is clearly better than the runner-up
    (no ambiguity between two similar sentences), (c) the numbers are identical, (d) the lengths are close
    (a quote of half a sentence, or of two sentences merged, is not a near copy)."""
    q = normalise(_strip_quote(quote))
    if len(q) < 25 or not source:
        return None
    qn = numbers_of(q)
    q_words = set(re.findall(r"[a-z]{4,}", q))
    best: List[Tuple[float, int, int]] = []
    for a, b in sentences(source):
        s = normalise(source[a:b])
        if not s or not (0.8 <= len(s) / len(q) <= 1.25):
            continue
        s_words = set(re.findall(r"[a-z]{4,}", s))
        if q_words and len(q_words & s_words) / len(q_words) < 0.6:
            continue                          # cheap pre-filter before the character comparison
        best.append((_ratio(q, s.rstrip(" .")), a, b))
    if not best:
        return None
    best.sort(reverse=True)
    r, a, b = best[0]
    if r < min_ratio or (len(best) > 1 and best[1][0] >= r - 0.02):
        return None
    verb = source[a:b]
    if sorted(numbers_of(verb)) != sorted(qn):
        return None
    return {"mode": "snapped", "start": a, "end": b, "verbatim": verb, "similarity": round(r, 3)}


# ----------------------------------------------------------------------------------------------- numbers check
_P_RX = re.compile(r"\b[pP]\s*(?:-?\s*values?)?\s*(?:[<>=≤≥]|<=|>=|=<|less than|of)\s*(0?\.\d+|\d+(?:\.\d+)?\s*[x×]\s*10\s*[-−]\s*\d+|[0-9.]+e-?\d+)", re.I)


def _p_values(text: str) -> List[float]:
    out = []
    for m in _P_RX.finditer(text or ""):
        raw = m.group(1).replace(" ", "")
        try:
            if re.search(r"[x×]10", raw):
                mant, ex = re.split(r"[x×]10", raw)
                out.append(float(mant) * 10 ** float(ex.replace("−", "-")))
            else:
                out.append(float(raw))
        except ValueError:
            continue
    return out


def _effect_numbers(claim: Dict[str, Any]) -> List[Tuple[str, float]]:
    out: List[Tuple[str, float]] = []
    for key in ("effect_size",):
        v = claim.get(key)
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            out.append((key, float(v)))
    ci = claim.get("confidence_interval")
    if isinstance(ci, (list, tuple)):
        for x in ci:
            if isinstance(x, (int, float)):
                out.append(("confidence_interval", float(x)))
    elif isinstance(ci, str):
        for x in numbers_of(ci):
            if x not in (90.0, 95.0, 99.0):
                out.append(("confidence_interval", x))
    return out


_STATEMENT_SKIP = re.compile(r"\b(?:19|20)\d{2}\b")      # a year in a statement ('the 2019 cohort') is not a result


def numbers_check(claim: Dict[str, Any], quote: str, source: str = "") -> Dict[str, Any]:
    """Every number the finding states must be in its quote. Returns
    {'ok', 'corrected': {field: (old, new)}, 'unsupported': [(field, value)], 'cleared': [field]}."""
    qnums = numbers_of(quote)
    snums = numbers_of(source) if source else []
    corrected: Dict[str, Tuple[Any, Any]] = {}
    cleared: List[str] = []
    unsupported: List[Tuple[str, Any]] = []

    # p-value: the quote's own p wins (code parses it); a p that is nowhere in the quote is removed
    p = claim.get("p_value")
    qp = _p_values(quote)
    try:
        pf = float(p) if p is not None and p != "" else None
    except (TypeError, ValueError):
        pf = None
    if pf is not None:
        if qp and not any(_same_number(pf, x) for x in qp):
            corrected["p_value"] = (pf, qp[0])
        elif not qp and not number_in(pf, qnums):
            cleared.append("p_value")
    elif qp:
        corrected["p_value"] = (None, qp[0])

    # effect size / CI: must be in the quote; if not, cleared (never invented)
    for field, v in _effect_numbers(claim):
        if not number_in(v, qnums):
            if field not in cleared:
                cleared.append(field)

    # sample size: in the quote or anywhere in the paper (it is often in the Methods); otherwise cleared
    n = claim.get("sample_size")
    try:
        nf = float(re.sub(r"[^\d.]", "", str(n))) if n not in (None, "") else None
    except ValueError:
        nf = None
    if nf is not None and not (number_in(nf, qnums) or number_in(nf, snums)):
        cleared.append("sample_size")

    # the finding's own sentence: a number that is in neither the quote nor the paper is made up
    stmt = str(claim.get("statement") or "")
    for x in numbers_of(_STATEMENT_SKIP.sub(" ", stmt)):
        if x in (0.0, 1.0, 95.0) and not number_in(x, qnums):
            continue                          # '95% CI', 'one', 'zero' style tokens carry no result
        if not number_in(x, qnums) and not number_in(x, snums):
            unsupported.append(("statement", x))
    # dose numbers must exist somewhere in the paper
    for x in numbers_of(str(claim.get("dose") or "")):
        if not number_in(x, qnums) and not number_in(x, snums):
            unsupported.append(("dose", x))
    return {"ok": not unsupported, "corrected": corrected, "cleared": cleared, "unsupported": unsupported}


# ----------------------------------------------------------------------------------------------- confidence
BASIS_RANK = {"jats_xml": 3, "full_text_xml": 3, "xml": 3, "europepmc_xml": 3,
              "unpaywall_pdf": 2, "pdf": 2, "full_text_pdf": 2, "full_text": 2, "html": 2,
              "abstract": 1, "abstract_only": 1}
SECTION_BONUS = {"results": 0, "table": 0, "results_discussion": 0, "abstract": 0, "discussion": -1,
                 "introduction": -1, "background": -1}


def source_confidence(basis: Optional[str], section_type: Optional[str] = None, mode: str = "exact",
                      second_reader: Optional[str] = None, numbers_fixed: bool = False) -> Dict[str, Any]:
    """'high' | 'moderate' | 'low' with the reasons, shown next to a finding (never used to hide it)."""
    b = (basis or "abstract").lower()
    score = BASIS_RANK.get(b, 2 if "pdf" in b or "full" in b else 1)
    why = [{3: "full-text XML (clean sections and tables)", 2: "full text from PDF",
            1: "abstract only (least accurate reading mode)"}[score]]
    if section_type and SECTION_BONUS.get(section_type, 0) < 0:
        score += SECTION_BONUS[section_type]
        why.append(f"quoted from the {section_type} (not the results)")
    if mode == "snapped":
        why.append("quote matched to the paper's sentence after small differences")
    if mode == "ellipsis":
        why.append("quote joins parts of one paragraph")
    if numbers_fixed:
        score -= 1
        why.append("numbers corrected from the quote")
    if second_reader == "agree":
        score += 1
        why.append("a second, independent reader extracted the same finding")
    elif second_reader == "not_found":
        why.append("the second reader did not extract this finding")
    elif second_reader == "disagree":
        score = 0
        why.append("the second reader disagreed")
    if b in ("abstract", "abstract_only"):
        score = min(score, 2)          # abstract-only is never 'high', whatever else agrees
    label = "high" if score >= 3 else "moderate" if score == 2 else "low"
    return {"label": label, "score": score, "basis": b, "why": why}


# ----------------------------------------------------------------------------------------------- one decision
def guard_claim(claim: Dict[str, Any], source: str, basis: Optional[str] = None,
                section_lookup: Any = None) -> Dict[str, Any]:
    """Decide one finding. Returns {'keep': bool, 'claim': updated claim, 'reason': str|None, 'mode': ...}.

    The claim's 'quote' is replaced by the paper's verbatim text; numbers are corrected or cleared; flags record what
    code changed. `section_lookup(start) -> section_type` is optional (for the confidence label)."""
    c = dict(claim)
    flags = list(c.get("flags") or [])
    quote = c.get("quote") or ""
    if c.get("deterministic_fallback") and locate(quote, source):
        loc = locate(quote, source)
    else:
        loc = locate(quote, source) or snap(quote, source)
    if not loc:
        return {"keep": False, "claim": c, "reason": "quote_not_in_paper", "mode": None}
    if loc["mode"] == "snapped":
        flags.append(f"quote_snapped:{loc.get('similarity')}")
        c["quote_as_extracted"] = quote
    c["quote"] = loc["verbatim"].strip()
    c["quote_span"] = [loc["start"], loc["end"]]
    c["quote_mode"] = loc["mode"]
    num = numbers_check(c, c["quote"], source)
    if not num["ok"]:
        return {"keep": False, "claim": c, "reason": "number_not_in_paper:" + ",".join(f"{f}={v:g}" for f, v in num["unsupported"]),
                "mode": loc["mode"]}
    for field, (old, new) in num["corrected"].items():
        c[field] = new
        if old is not None:
            flags.append(f"numbers_corrected:{field}:{old}->{new}")
    for field in num["cleared"]:
        if c.get(field) not in (None, ""):
            flags.append(f"numbers_cleared:{field}:{c.get(field)}")
        c[field] = None
    section = None
    if callable(section_lookup):
        try:
            section = section_lookup(loc["start"])
        except Exception:
            section = None
    fixed = any(str(f).startswith("numbers_corrected") for f in flags)
    c["source_basis"] = basis or "abstract"
    c["extraction_confidence"] = source_confidence(basis, section, loc["mode"], c.get("second_reader"), fixed)
    c["flags"] = flags
    c["grounded"], c["grounding_score"] = True, 1.0
    return {"keep": True, "claim": c, "reason": None, "mode": loc["mode"]}


def section_locator(parts_in_order: List[Tuple[str, str]]) -> Tuple[str, Any]:
    """Join (section_type, text) parts into one source and return (source, lookup(position) -> section_type)."""
    bounds: List[Tuple[int, int, str]] = []
    chunks: List[str] = []
    pos = 0
    for stype, text in parts_in_order:
        if not text:
            continue
        chunks.append(text)
        bounds.append((pos, pos + len(text), stype))
        pos += len(text) + 2
    src = "\n\n".join(chunks)

    def lookup(i: int) -> Optional[str]:
        for a, b, s in bounds:
            if a <= i < b:
                return s
        return None

    return src, lookup
