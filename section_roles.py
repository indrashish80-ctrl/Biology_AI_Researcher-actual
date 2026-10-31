"""
section_roles.py  (v24.2)
-------------------------
What ROLE does a sentence play in its paper: background, aim, method, result or conclusion?

A structured abstract says so itself ("BACKGROUND: Metformin is associated with lower breast cancer risk in
epidemiologic studies ... RESULTS: ..."). A claim extracted from the BACKGROUND line is prior work, not this study's
result, however well it is grounded in the text. Unstructured abstracts carry the same signal in phrases
("has previously been reported", "in epidemiologic studies") unless the sentence also claims the work as the
authors' own ("we found", "in this trial").
"""
from __future__ import annotations

import re
from typing import Optional

_LABELS = [
    ("background", r"BACKGROUND|INTRODUCTION|CONTEXT|RATIONALE|IMPORTANCE"),
    ("aim", r"OBJECTIVES?|AIMS?|PURPOSE|HYPOTHESIS|QUESTION"),
    ("method", r"METHODS?|MATERIALS? AND METHODS|DESIGN|SETTING|PARTICIPANTS|PATIENTS|INTERVENTIONS?|MEASUREMENTS?|"
               r"EXPOSURES?|MAIN OUTCOMES? AND MEASURES|DESIGN, SETTING,? AND PARTICIPANTS"),
    ("result", r"RESULTS?|FINDINGS|OUTCOMES?"),
    ("conclusion", r"CONCLUSIONS?|INTERPRETATION|DISCUSSION|SIGNIFICANCE|IMPLICATIONS?"),
]
_LABEL_RX = re.compile(r"(?:(?<=[\s.])|^)(" + "|".join(f"(?P<{n}>{p})" for n, p in _LABELS) + r")\s*[:—\-](?=\s)")
_OWN = re.compile(r"\b(?:we|our|in this (?:study|trial|analysis|cohort|work|paper)|the present (?:study|trial)|"
                  r"this (?:study|trial|analysis) (?:found|showed|shows|demonstrates?|revealed|reports?)|"
                  r"were randomi[sz]ed|participants were|patients were|compared with placebo)\b", re.I)
_PRIOR = re.compile(r"\b(?:(?:has|have) (?:previously |already |also )?(?:been )?(?:reported|shown|demonstrated|observed|"
                    r"described|suggested)|previously (?:reported|shown|described)|in (?:several |many |previous |prior |"
                    r"recent )?(?:epidemiolog\w*|observational|retrospective|preclinical|clinical) (?:studies|trials|data)|"
                    r"according to (?:previous|prior|earlier) (?:studies|reports)|is well (?:known|established|documented)|"
                    r"it is (?:well )?known that|accumulating evidence|growing (?:body of )?evidence|studies have)\b", re.I)


def abstract_role(quote: str, source_text: str) -> Optional[str]:
    """Role of the structured-abstract section the quote sits in, or None (not found / not structured)."""
    if not quote or not source_text:
        return None
    probe = re.sub(r"\s+", " ", quote.strip())[:70]
    flat = re.sub(r"\s+", " ", source_text)
    pos = flat.lower().find(probe.lower())
    if pos < 0:
        return None
    role = None
    for m in _LABEL_RX.finditer(flat[:pos]):
        for n, _ in _LABELS:
            if m.group(n):
                role = n
    return role


def is_prior_work_phrase(quote: str) -> bool:
    """Generic 'it has been reported' background that does not claim the work as the authors' own."""
    return bool(_PRIOR.search(quote or "")) and not _OWN.search(quote or "")
