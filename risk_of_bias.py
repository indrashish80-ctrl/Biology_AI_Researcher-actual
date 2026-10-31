"""
risk_of_bias.py  (v25)
======================
Fuller quality assessment: RoB 2 (randomised trials), ROBINS-I (non-randomised studies of interventions) and a
SYRCLE-style checklist (animal studies), each as SIGNALLING QUESTIONS answered from the paper's text, domain
judgements by the tools' own algorithms, and an overall judgement.

Answers: Y (yes) / PY (probably yes) / PN (probably no) / N (no) / NI (no information). Code only answers what the
text states; everything else is NI - and NI is never silently counted as 'low risk'. Abstract-only papers are
'not assessable'. A person should confirm any judgement that changes a conclusion; every answer carries the
sentence it was read from.

    assess(text, design)  -> {'tool', 'domains': [{'domain', 'questions', 'judgement', 'why'}], 'overall', 'version'}

RoB 2 domains: D1 randomisation process · D2 deviations from intended interventions · D3 missing outcome data ·
D4 measurement of the outcome · D5 selection of the reported result.   Overall: high if any domain is high, some
concerns if any domain has some concerns, low only if all five are low (Sterne et al., BMJ 2019).
ROBINS-I domains: confounding · selection of participants · classification of interventions · deviations ·
missing data · measurement of outcomes · selection of the reported result. Overall: the most severe domain
(low / moderate / serious / critical).
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

ROB_VERSION = "1.0"


def _find(rx: str, text: str) -> Optional[str]:
    m = re.search(rx, text, re.I)
    if not m:
        return None
    a = text.rfind(". ", 0, m.start()) + 2
    b = text.find(". ", m.end())
    return text[max(0, a):(b if b > 0 else m.end() + 120)].strip()[:260]


def _q(qid: str, question: str, answer: str, evidence: Optional[str]) -> Dict[str, Any]:
    return {"id": qid, "question": question, "answer": answer, "evidence": evidence}


def _yn(text: str, yes: str, no: Optional[str] = None, py: Optional[str] = None) -> Tuple[str, Optional[str]]:
    if no:
        e = _find(no, text)
        if e:
            return "N", e
    e = _find(yes, text)
    if e:
        return "Y", e
    if py:
        e = _find(py, text)
        if e:
            return "PY", e
    return "NI", None


# ----------------------------------------------------------------------------------------------- RoB 2
def rob2(text: str) -> Dict[str, Any]:
    t = text or ""
    D: List[Dict[str, Any]] = []
    # D1 randomisation process
    a11, e11 = _yn(t, r"computer[- ]generated|random number (?:table|generator)|block randomi[sz]|stratified randomi[sz]|"
                      r"permuted blocks?|minimi[sz]ation|randomi[sz]ation (?:list|sequence|schedule)",
                   r"alternat(?:e|ing) allocation|by (?:date of birth|hospital number|day of the week)|quasi-random",
                   r"\brandomi[sz]ed\b|\brandomly (?:assigned|allocated)")
    a12, e12 = _yn(t, r"allocation conceal|sealed (?:opaque )?envelopes?|central(?:ised|ized)? (?:randomi[sz]ation|allocation)|"
                      r"interactive (?:web|voice) response|pharmacy[- ]controlled|sequentially numbered",
                   r"open allocation|allocation was not concealed")
    a13, e13 = _yn(t, r"baseline characteristics (?:were )?(?:similar|comparable|balanced|well[- ]balanced)|"
                      r"no (?:significant )?differences? (?:in|at) baseline",
                   r"imbalance at baseline|baseline (?:differences?|imbalances?) (?:in|between)")
    a13 = {"Y": "N", "N": "Y", "NI": "NI", "PY": "PN"}[a13]      # 1.3 asks whether imbalances SUGGEST a problem
    if a11 in ("N",) or a12 == "N":
        j1 = "high"
    elif a11 in ("Y", "PY") and a12 in ("Y", "PY") and a13 in ("N", "PN", "NI"):
        j1 = "low"
    else:
        j1 = "some concerns"
    D.append({"domain": "D1 Randomisation process", "judgement": j1, "questions": [
        _q("1.1", "Was the allocation sequence random?", a11, e11),
        _q("1.2", "Was the allocation sequence concealed until assignment?", a12, e12),
        _q("1.3", "Did baseline differences suggest a problem with randomisation?", a13, e13)]})
    # D2 deviations from intended interventions
    a21, e21 = _yn(t, r"double[- ]blind|triple[- ]blind|participants? (?:were|was) blinded|masked to (?:treatment|allocation)|"
                      r"placebo[- ]controlled", r"open[- ]label|unblinded|not blinded|single[- ]blind(?:ed)? \(investigator")
    a21 = {"Y": "N", "N": "Y", "NI": "NI", "PY": "PN"}[a21]      # 2.1 asks whether participants were AWARE
    a26, e26 = _yn(t, r"intention[- ]to[- ]treat|\bITT\b|modified intention|all randomi[sz]ed (?:participants|patients)",
                   r"per[- ]protocol (?:analysis )?only|completers? analysis", r"full analysis set")
    if a26 == "N":
        j2 = "high" if a21 == "Y" else "some concerns"
    elif a21 in ("N", "PN") and a26 in ("Y", "PY"):
        j2 = "low"
    else:
        j2 = "some concerns"
    D.append({"domain": "D2 Deviations from intended interventions", "judgement": j2, "questions": [
        _q("2.1", "Were participants aware of their assigned intervention?", a21, e21),
        _q("2.6", "Was an appropriate analysis used (intention to treat)?", a26, e26)]})
    # D3 missing outcome data
    lost = re.findall(r"(\d+(?:\.\d+)?)\s*%\s*(?:of (?:patients|participants) )?(?:were )?(?:lost to follow[- ]?up|dropped out|"
                      r"withdrew|discontinued)", t, re.I)
    a31, e31 = "NI", None
    if lost:
        worst = max(float(x) for x in lost)
        a31, e31 = ("Y" if worst <= 10 else "PN" if worst <= 20 else "N"), _find(r"lost to follow|dropped out|withdrew|discontinued", t)
    elif re.search(r"\b(?:all|every) (?:randomi[sz]ed )?(?:patients?|participants?) completed|no (?:patients?|participants?) "
                   r"(?:were )?lost|complete follow[- ]?up", t, re.I):
        a31, e31 = "Y", _find(r"completed|no (?:patients?|participants?) (?:were )?lost|complete follow", t)
    a32, e32 = _yn(t, r"multiple imputation|sensitivity analys[ie]s .{0,40}missing|missing data (?:were|was) (?:imputed|handled)")
    j3 = "low" if a31 == "Y" or a32 in ("Y", "PY") else ("high" if a31 == "N" else "some concerns")
    D.append({"domain": "D3 Missing outcome data", "judgement": j3, "questions": [
        _q("3.1", "Were outcome data available for (nearly) all participants?", a31, e31),
        _q("3.2", "Is there evidence the result was not biased by missing data?", a32, e32)]})
    # D4 measurement of the outcome
    a43, e43 = _yn(t, r"(?:outcome )?assessors? (?:were |was )?(?:blinded|masked|unaware)|blinded (?:outcome )?assess|"
                      r"double[- ]blind|central(?:ly)? (?:read|assessed|adjudicated)|independent (?:blinded )?adjudication",
                   r"assessors? (?:were )?(?:not blinded|aware)|unblinded assess")
    a43 = {"Y": "N", "N": "Y", "NI": "NI", "PY": "PN"}[a43]      # 4.3 asks whether assessors were AWARE
    objective = bool(re.search(r"\b(?:mortality|death|all-cause|serum|plasma|laboratory|hba1c|biopsy|histolog|imaging|mri|ct scan|"
                               r"ki-?67|blood pressure)\b", t, re.I))
    j4 = "low" if a43 in ("N", "PN") or (a43 == "NI" and objective) else ("high" if a43 == "Y" and not objective else "some concerns")
    D.append({"domain": "D4 Measurement of the outcome", "judgement": j4, "questions": [
        _q("4.3", "Were outcome assessors aware of the intervention received?", a43, e43),
        _q("4.x", "Is the outcome objective (laboratory / mortality / imaging)?", "Y" if objective else "NI", None)]})
    # D5 selection of the reported result
    a51, e51 = _yn(t, r"\bNCT\d{8}\b|\bISRCTN\d+|\bACTRN\d+|\bEudraCT\b|\bChiCTR|\bCTRI/|pre-?registered|prospectively registered|"
                      r"registered (?:at|with|on) (?:clinicaltrials|the)|statistical analysis plan|published protocol")
    a52, e52 = _yn(t, r"post[- ]hoc|exploratory analys[ie]s|subgroup analys[ie]s (?:showed|revealed|suggested)")
    a52 = {"Y": "PY", "N": "N", "NI": "NI", "PY": "PY"}[a52]     # results selected from several analyses?
    j5 = "low" if a51 in ("Y", "PY") and a52 in ("NI", "N", "PN") else "some concerns"
    D.append({"domain": "D5 Selection of the reported result", "judgement": j5, "questions": [
        _q("5.1", "Was the analysis pre-specified (registered protocol / analysis plan)?", a51, e51),
        _q("5.2", "Could the result have been selected from multiple analyses (post hoc / subgroups)?", a52, e52)]})
    js = [d["judgement"] for d in D]
    overall = "high" if "high" in js else "some concerns" if "some concerns" in js else "low"
    return {"tool": "RoB 2", "domains": D, "overall": overall}


# ----------------------------------------------------------------------------------------------- ROBINS-I
def robins_i(text: str) -> Dict[str, Any]:
    t = text or ""
    D = []

    def dom(name, rx_yes, rx_no=None, low="low", missing="serious"):
        a, e = _yn(t, rx_yes, rx_no)
        j = low if a in ("Y", "PY") else ("critical" if a == "N" else missing if missing != "moderate" else "moderate")
        D.append({"domain": name, "judgement": j, "questions": [_q(name[:2], name, a, e)]})

    dom("Confounding", r"adjust\w* for|multivariab|propensity score|inverse probability|matched (?:on|for)|instrumental variable",
        r"unadjusted|crude (?:analysis|estimates?) only", low="moderate", missing="serious")
    dom("Selection of participants", r"population-based|consecutive(?:ly)?|all (?:eligible )?patients|registry|inception cohort",
        r"convenience sample|self-selected", low="low", missing="moderate")
    dom("Classification of interventions", r"prescription records?|pharmacy (?:records|dispensing)|electronic (?:health|medical) records?|"
        r"registry|recorded at baseline", r"self-reported (?:use|exposure)|recall", low="low", missing="moderate")
    dom("Deviations from intended interventions", r"adherence|compliance|persistence|new[- ]user design|active comparator",
        None, low="low", missing="moderate")
    dom("Missing data", r"complete[- ]case|multiple imputation|missing data|lost to follow", None, low="low", missing="moderate")
    dom("Measurement of outcomes", r"blinded|registry|death certificates?|icd-?(?:9|10)|validated|adjudicat|laboratory",
        r"self-reported outcome", low="low", missing="moderate")
    dom("Selection of the reported result", r"pre-?registered|protocol|statistical analysis plan|NCT\d{8}", r"post[- ]hoc",
        low="low", missing="moderate")
    rank = {"low": 0, "moderate": 1, "serious": 2, "critical": 3}
    overall = max((d["judgement"] for d in D), key=lambda j: rank.get(j, 0))
    return {"tool": "ROBINS-I", "domains": D, "overall": overall}


# ----------------------------------------------------------------------------------------------- SYRCLE (animal)
def syrcle(text: str) -> Dict[str, Any]:
    t = text or ""
    items = [("Sequence generation", r"randomly (?:assigned|allocated|divided)|randomi[sz]ed (?:into|to)"),
             ("Allocation concealment", r"allocation conceal|coded (?:cages|treatments)"),
             ("Random housing", r"randomly housed|random(?:ised|ized)? cage"),
             ("Blinding of caregivers / investigators", r"blind\w* (?:to|investigators?|experimenters?)|investigators? were blinded"),
             ("Blinded outcome assessment", r"(?:assessed|scored|analy[sz]ed) (?:in a )?blind\w*|blinded (?:observer|assessor)"),
             ("Incomplete outcome data addressed", r"excluded from (?:the )?analysis because|all animals (?:were )?included|no animals? (?:were )?excluded"),
             ("Sample size calculation", r"power (?:analysis|calculation)|sample size (?:was )?(?:calculated|determined)")]
    D = []
    for name, rx in items:
        a, e = _yn(t, rx)
        D.append({"domain": name, "judgement": "low" if a in ("Y", "PY") else "unclear", "questions": [_q(name[:3], name, a, e)]})
    n_low = sum(d["judgement"] == "low" for d in D)
    overall = "low" if n_low >= 5 else "some concerns" if n_low >= 2 else "high"
    return {"tool": "SYRCLE (animal)", "domains": D, "overall": overall}


def assess(text: str, design: str, full_text: bool = True) -> Dict[str, Any]:
    d = (design or "").lower()
    if not full_text:
        return {"tool": None, "domains": [], "overall": "not assessable", "why": "abstract only - methods not available",
                "version": ROB_VERSION}
    if d in ("human_clinical", "rct") and re.search(r"\brandomi[sz]|\brandomly (?:assigned|allocated)", text or "", re.I):
        out = rob2(text)
    elif d in ("human_clinical", "human_observational", "cohort", "case_control"):
        out = robins_i(text)
    elif d in ("animal", "experimental_disease_model"):
        out = syrcle(text)
    else:
        return {"tool": None, "domains": [], "overall": "not assessable",
                "why": f"no risk-of-bias tool for design '{design}'", "version": ROB_VERSION}
    out["version"] = ROB_VERSION
    out["n_no_information"] = sum(q["answer"] == "NI" for dm in out["domains"] for q in dm["questions"])
    return out


def label(res: Dict[str, Any]) -> str:
    """Map to the matrix's 3-level vocabulary: Low / Some concerns / High / Not assessable."""
    o = (res or {}).get("overall") or "not assessable"
    return {"low": "Low", "some concerns": "Some concerns", "moderate": "Some concerns", "high": "High",
            "serious": "High", "critical": "High"}.get(o, "Not assessable")


def markdown(rows: List[Dict[str, Any]]) -> str:
    rows = [r for r in rows if (r.get("risk_of_bias_detail") or {}).get("tool")]
    if not rows:
        return ""
    L = ["## Risk of bias (RoB 2 / ROBINS-I / SYRCLE signalling questions)", "",
         "| Study | Tool | " + " | ".join(f"D{i}" for i in range(1, 8)) + " | Overall |", "|---|---|" + "---|" * 8]
    sym = {"low": "+", "some concerns": "?", "moderate": "?", "unclear": "?", "high": "-", "serious": "-", "critical": "!"}
    for r in rows:
        d = r["risk_of_bias_detail"]
        cells = [sym.get(x["judgement"], "?") for x in d["domains"]] + [""] * (7 - len(d["domains"]))
        L.append(f"| {r.get('ref')} | {d['tool']} | " + " | ".join(cells[:7]) + f" | **{d['overall']}** |")
    L += ["", "_+ low · ? some concerns / unclear · - high / serious · ! critical. Answers come only from what the text "
          "states (NI = no information); confirm any judgement that changes a conclusion._", ""]
    return "\n".join(L) + "\n"
