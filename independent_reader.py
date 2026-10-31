"""
independent_reader.py  (v25)
============================
A second, INDEPENDENT reading of the same paper text. Agreement between two independent readers is more reliable
than one reader's confidence (principle 3).

Why a new module: v24.2's claim_verifier.py re-ran the engine's own deterministic direction reader on the first
reader's quote - the same code that had already read it - so it could never disagree. This reader is independent
of the first one in the two ways that matter:

  * a different MODEL when the user has more than one AI key (the provider that is NOT leading extraction), and
  * a different PROMPT always: it is asked for the results as a reviewer would tabulate them (endpoint, comparator,
    direction against the comparator, the verbatim sentence), not for 'claims' - so it does not share the first
    prompt's wording, examples or failure modes.

Its findings go through the same code guard (quote_guard: verbatim quote, numbers in the quote) before they are
compared, so a hallucinated second reading cannot overrule a real first one.

Comparison per first-reader finding:
    agree       the second reader extracted a finding from the same sentence(s) with the same endpoint family and
                a compatible direction                                     -> confidence goes up
    disagree    same sentence, same endpoint family, a CONFLICTING direction (increase vs decrease, or an effect vs
                no effect)                                                 -> the finding is quarantined for review
    not_found   the second reader did not extract it                       -> kept, labelled
Findings ONLY the second reader extracted are returned as 'missed_by_first' for the review queue (never added
silently - a person decides).

Switch: BIOSYN_SECOND_READER = auto (default: on for full runs, off for quick scans) | 1 | 0
Policy: BIOSYN_DISAGREEMENT  = quarantine (default) | drop | keep
"""
from __future__ import annotations

import os
import re
from typing import Any, Dict, List, Optional, Tuple

READER_VERSION = "1.0"


def enabled(quick: bool = False) -> bool:
    v = (os.getenv("BIOSYN_SECOND_READER") or "auto").strip().lower()
    if v in ("0", "off", "false", "no"):
        return False
    if v in ("1", "on", "true", "yes"):
        return True
    return not quick


def policy() -> str:
    v = (os.getenv("BIOSYN_DISAGREEMENT") or "quarantine").strip().lower()
    return v if v in ("quarantine", "drop", "keep") else "quarantine"


def prompt_second_reader(text: str, question: Optional[str], n_max: int = 6) -> str:
    focus = f'The review question is: "{question}".\n' if question else ""
    # v26: the varying parts (question, text) are at the END so a provider's prompt-prefix cache can reuse the rest
    return f"""You are the SECOND, independent reviewer tabulating the results of one study for a systematic review.
Work only from the text below. Tabulate up to {n_max} results that THIS study measured (include null results).
For each result give:
  "endpoint":   the quantity that was measured (e.g. "Ki-67", "tumour volume", "HbA1c", "cell viability")
  "comparator": what it was compared with ("placebo", "vehicle", "baseline", "untreated controls", or null)
  "direction":  what happened to the endpoint in the treated / exposed group RELATIVE TO THE COMPARATOR:
                "increase", "decrease", "no_effect" (no statistically significant difference) or "unclear"
  "agent":      the treatment, exposure or manipulation the study applied
  "sentence":   the sentence of the text that reports it, copied character for character
Do not include results the authors cite from other studies, background statements or hypotheses.
Return ONLY JSON: {{"results": [{{"endpoint": "...", "comparator": "...", "direction": "...", "agent": "...", "sentence": "..."}}]}}

{focus}TEXT:
{text}"""


_DIR = {"increase": "increase", "increased": "increase", "up": "increase", "higher": "increase",
        "decrease": "decrease", "decreased": "decrease", "down": "decrease", "lower": "decrease", "reduced": "decrease",
        "no_effect": "no_effect", "no effect": "no_effect", "no change": "no_effect", "none": "no_effect",
        "no significant difference": "no_effect", "unchanged": "no_effect", "null": "no_effect",
        "unclear": "unclear", "mixed": "unclear"}


def parse_second_reader(response: str) -> List[Dict[str, Any]]:
    from llm_json import parse_json_response
    try:
        obj = parse_json_response(response)
    except Exception:
        return []
    rows = obj.get("results") if isinstance(obj, dict) else obj
    out = []
    for r in rows or []:
        if not isinstance(r, dict):
            continue
        sent = str(r.get("sentence") or r.get("quote") or "").strip()
        ep = str(r.get("endpoint") or r.get("outcome") or "").strip()
        if not sent or not ep:
            continue
        out.append({"endpoint": ep, "comparator": r.get("comparator"), "agent": r.get("agent"),
                    "direction": _DIR.get(str(r.get("direction") or "unclear").strip().lower(), "unclear"),
                    "quote": sent})
    return out


def _family(endpoint: str) -> str:
    try:
        from semantic_audit import canonical_outcome
        return canonical_outcome(endpoint).get("family") or endpoint.lower()
    except Exception:
        return (endpoint or "").lower()


def _polarity(endpoint: str) -> int:
    try:
        from semantic_audit import canonical_outcome
        return int(canonical_outcome(endpoint).get("polarity", 1) or 1)
    except Exception:
        return 1


def _overlap(a: Tuple[int, int], b: Tuple[int, int]) -> float:
    lo, hi = max(a[0], b[0]), min(a[1], b[1])
    if hi <= lo:
        return 0.0
    return (hi - lo) / max(1, min(a[1] - a[0], b[1] - b[0]))


def _tokens(s: str) -> set:
    return set(re.findall(r"[a-z0-9][a-z0-9\-]{2,}", (s or "").lower()))


def _same_endpoint(e1: str, e2: str) -> bool:
    f1, f2 = _family(e1), _family(e2)
    if f1 and f1 == f2 and f1 not in ("other", "unclear", "expression"):
        return True
    t1, t2 = _tokens(e1), _tokens(e2)
    return bool(t1 and t2 and len(t1 & t2) / min(len(t1), len(t2)) >= 0.5)


def _canon(direction: str, endpoint: str) -> str:
    """Direction on the family's canonical scale ('lesion area decreased' == 'clinical response increased')."""
    d = (direction or "unclear").lower()
    if d not in ("increase", "decrease"):
        return d
    return d if _polarity(endpoint) > 0 else {"increase": "decrease", "decrease": "increase"}[d]


def compatible(d1: str, d2: str) -> Optional[bool]:
    """True = same reading, False = conflicting, None = cannot tell (one is unclear)."""
    if "unclear" in (d1, d2) or not d1 or not d2:
        return None
    return d1 == d2


def compare(first: List[Dict[str, Any]], second: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Annotate first-reader claims with second_reader = agree | disagree | not_found. Both lists must already be
    guarded (verbatim quotes with 'quote_span')."""
    used = set()
    verdicts = []
    for c in first:
        span = tuple(c.get("quote_span") or (0, 0))
        ep1 = c.get("outcome") or c.get("measurement") or ""
        d1 = _canon(c.get("direction") or c.get("effect_direction") or "unclear", ep1)
        best, best_score = None, 0.0
        for j, s in enumerate(second):
            sspan = tuple(s.get("quote_span") or (0, 0))
            ov = _overlap(span, sspan) if span != (0, 0) and sspan != (0, 0) else 0.0
            if ov <= 0.3 or not _same_endpoint(ep1, s.get("endpoint") or ""):
                continue
            if ov > best_score:
                best, best_score = j, ov
        if best is None:
            c["second_reader"] = "not_found"
            verdicts.append("not_found")
            continue
        used.add(best)
        s = second[best]
        d2 = _canon(s.get("direction") or "unclear", s.get("endpoint") or "")
        ok = compatible(d1, d2)
        c["second_reader_reading"] = {"endpoint": s.get("endpoint"), "direction": s.get("direction"),
                                      "comparator": s.get("comparator"), "quote": s.get("quote")}
        if ok is False:
            c["second_reader"] = "disagree"
            c["flags"] = list(c.get("flags") or []) + [
                f"reader_disagreement: first reader '{c.get('direction')}', second reader '{s.get('direction')}' "
                f"for {s.get('endpoint')}"]
        else:
            c["second_reader"] = "agree" if ok else "agree_unclear"
        verdicts.append(c["second_reader"])
    missed = [s for j, s in enumerate(second) if j not in used]
    n = len(first) or 1
    return {"agree": verdicts.count("agree"), "agree_unclear": verdicts.count("agree_unclear"),
            "disagree": verdicts.count("disagree"), "not_found": verdicts.count("not_found"),
            "missed_by_first": missed, "agreement_rate": round(verdicts.count("agree") / n, 3),
            "version": READER_VERSION}


def second_read(router: Any, text_chunks: List[str], question: Optional[str], source: str,
                basis: Optional[str] = None, first_provider: Optional[str] = None) -> List[Dict[str, Any]]:
    """Ask the second reader (a different provider when there is one) and guard every result with quote_guard."""
    import quote_guard as qg
    if router is None or not text_chunks:
        return []
    order = []
    try:
        order = list(router.order_for("extraction"))
    except Exception:
        pass
    # the provider that did NOT lead extraction; with one key, the same model under a different prompt
    prov = next((p for p in reversed(order) if p != first_provider), order[-1] if order else None)
    prompts = [prompt_second_reader(t, question) for t in text_chunks if t]
    answers: List[Optional[str]] = []
    for p in prompts:
        try:
            answers.append(router.ask(p, provider=prov, fallback=True, task="extraction_check"))
        except TypeError:
            answers.append(router.ask(p))
        except Exception:
            answers.append(None)
    out: List[Dict[str, Any]] = []
    for a in answers:
        if not a:
            continue
        for f in parse_second_reader(a):
            g = qg.guard_claim({"quote": f["quote"], "statement": "", "outcome": f["endpoint"]}, source, basis)
            if g["keep"]:
                f["quote"], f["quote_span"] = g["claim"]["quote"], g["claim"]["quote_span"]
                f["provider"] = prov
                out.append(f)
    return out
