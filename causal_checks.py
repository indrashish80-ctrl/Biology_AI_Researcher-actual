"""
causal_checks.py  (v26)
=======================
Association versus causation, per study and for the answer, by code (no AI).

    assess(row, text) -> {
        'design_support':   'randomised experiment' | 'controlled experiment' | 'observational, adjusted' |
                            'observational, unadjusted' | 'mechanistic only' | 'case report / series' | 'unclear',
        'confounders':      adjusted variables found ('age', 'BMI', 'smoking' ...), method (multivariable, propensity,
                            matching, instrumental variable, Mendelian randomisation),
        'risks':            [{'type': 'reverse causation' | 'selection bias' | 'confounding by indication' |
                              'immortal time bias' | 'healthy-user effect' | 'mediator adjustment' | 'residual
                              confounding', 'why': ...}],
        'mechanism_tested': True | False (a mechanism is claimed but the study did not manipulate it),
        'causal_language_ok': can this study's result be stated causally?
    }
    dag(question, rows) -> Mermaid causal diagram: exposure -> outcome, measured confounders -> both, mediators on the
                           path, a dashed reverse arrow when reverse causation is a risk. The diagram shows ASSUMED
                           relationships taken from the studies' own adjustment sets - it is not evidence.
    language(report_text, rows) -> report sentences that state causation while every supporting study is
                           observational (or only mechanistic / in cells or animals for a claim about people)
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

CAUSAL_VERSION = "1.0"
_CONF = ["age", "sex", "gender", "body mass index", "BMI", "smoking", "alcohol", "diabetes", "hypertension", "income",
         "education", "socioeconomic", "race", "ethnicity", "comorbidit", "physical activity", "diet", "stage", "grade",
         "HbA1c", "insulin", "statin", "aspirin", "family history", "menopaus", "parity", "region", "calendar year"]
_METHODS = {"multivariable": r"multivariab|multivariat|adjusted for|adjusting for|after adjustment",
            "propensity score": r"propensity", "matching": r"\bmatched\b|matching",
            "instrumental variable": r"instrumental variable", "Mendelian randomisation": r"mendelian randomi",
            "difference-in-differences": r"difference[- ]in[- ]differences", "active comparator": r"active comparator",
            "new-user design": r"new[- ]user|incident user"}
_CAUSAL_WORDS = re.compile(r"\b(?:causes?|caused|causing|leads? to|led to|results? in|prevents?|prevented|protects? against|"
                           r"reduces? the risk|lowers? the risk|increases? the risk|is responsible for|drives?|"
                           r"therefore (?:reduces|increases|prevents))\b", re.I)
_HEDGED = re.compile(r"\b(?:associat|correlat|linked|may|might|could|suggest|possibl|potential|observational)\w*", re.I)


def _found(text: str, words: List[str]) -> List[str]:
    out = []
    for w in words:
        if re.search(r"\b" + re.escape(w) if w[0].isalpha() and w.islower() else re.escape(w), text, re.I):
            out.append(w.rstrip())
    return out


def assess(row: Dict[str, Any], text: str) -> Dict[str, Any]:
    t = text or ""
    et = row.get("evidence_type") or "unclear"
    inf = row.get("inference_type")
    adj_window = " ".join(m.group(0) for m in re.finditer(r"(?:adjust\w*|controll\w* for|account\w* for)[^.]{0,260}", t, re.I))
    confounders = _found(adj_window, _CONF)
    methods = [k for k, rx in _METHODS.items() if re.search(rx, t, re.I)]
    risks: List[Dict[str, str]] = []
    if et == "human_clinical" and (inf == "randomised_controlled" or re.search(r"\brandomi[sz]", t, re.I)):
        support = "randomised experiment"
    elif et in ("animal", "cell_culture", "organoid_ex_vivo", "plant", "microbial", "experimental_disease_model"):
        support = "controlled experiment" if re.search(r"\bcontrol|vehicle|sham|untreated|wild[- ]type|scrambled", t, re.I) \
            else "mechanistic only"
    elif et == "human_observational":
        support = "observational, adjusted" if (confounders or methods) else "observational, unadjusted"
        if re.search(r"cross[- ]sectional", t, re.I):
            risks.append({"type": "reverse causation", "why": "cross-sectional design: exposure and outcome measured at the "
                                                              "same time, so the direction cannot be established"})
        if re.search(r"prevalent users?|current users?|ever[- ]users?", t, re.I) and not re.search(r"new[- ]user|incident user", t, re.I):
            risks.append({"type": "selection bias", "why": "prevalent users: people who tolerate and benefit from the "
                                                           "drug remain on it (survivor / healthy-adherer effect)"})
        if re.search(r"hospital(?:-based)? controls|case[- ]control", t, re.I):
            risks.append({"type": "selection bias", "why": "case-control design: control selection and recall can bias the association"})
        if re.search(r"time[- ]to|survival|follow[- ]up", t, re.I) and re.search(r"ever (?:used|exposed)|any use", t, re.I) and \
                not re.search(r"time[- ]varying|landmark|immortal", t, re.I):
            risks.append({"type": "immortal time bias", "why": "exposure defined as 'ever use' during follow-up without a "
                                                                "time-varying or landmark analysis"})
        if re.search(r"(?:users|treated patients) (?:were|are) (?:younger|healthier|more educated)|healthy[- ]user", t, re.I):
            risks.append({"type": "healthy-user effect", "why": "treated people differ in health behaviour from untreated people"})
        if re.search(r"\bindication\b", t, re.I) and not re.search(r"active comparator", t, re.I):
            risks.append({"type": "confounding by indication", "why": "the reason for treatment may itself affect the outcome"})
        if not methods and not confounders:
            risks.append({"type": "residual confounding", "why": "no adjustment for confounders reported"})
        elif confounders and len(confounders) < 3:
            risks.append({"type": "residual confounding", "why": f"adjusted only for {', '.join(confounders)}"})
    elif et in ("case_report",) or (row.get("n") == 1):
        support = "case report / series"
    else:
        support = "unclear"
    mediators = []
    for m in re.finditer(r"adjust\w* for[^.]{0,200}", t, re.I):
        for med in ("weight loss", "insulin", "HbA1c", "glucose", "blood pressure", "LDL", "inflammation", "CRP"):
            if med.lower() in m.group(0).lower() and re.search(re.escape(med), row.get("outcome") or "", re.I) is None:
                mediators.append(med)
    if mediators and et == "human_observational":
        risks.append({"type": "mediator adjustment", "why": f"adjusting for {', '.join(sorted(set(mediators)))}, which may lie on "
                                                            f"the causal path, can hide part of a real effect"})
    claims_mech = bool(re.search(r"\b(?:via|through|mediated by|mechanism)\b", " ".join(
        str(f.get("statement") or "") for f in row.get("findings") or []), re.I))
    manipulated = bool(re.search(r"knock[- ]?down|knock[- ]?out|siRNA|shRNA|CRISPR|inhibitor of|antagonist|blockade|"
                                 r"overexpress", t, re.I))
    mech_tested = None if not claims_mech else manipulated
    return {"design_support": support, "confounders": confounders, "adjustment_methods": methods, "risks": risks,
            "mediators_adjusted": sorted(set(mediators)), "mechanism_claimed": claims_mech, "mechanism_tested": mech_tested,
            "causal_language_ok": support in ("randomised experiment", "controlled experiment"),
            "version": CAUSAL_VERSION}


def dag(question: str, rows: List[Dict[str, Any]]) -> Optional[str]:
    from evidence_identity import question_frame
    fr = question_frame(question or "")
    exposure = (fr.get("exposures") or [None])[0] or next((r.get("intervention") for r in rows if r.get("intervention")), None)
    from evidence_identity import question_outcome
    outcome = question_outcome(fr)
    outcome = outcome or next((f.get("endpoint") for r in rows for f in r.get("findings") or [] if f.get("endpoint")), None)
    if not exposure or not outcome:
        return None
    conf, meds, reverse = {}, set(), False
    for r in rows:
        c = r.get("causal") or {}
        for x in c.get("confounders") or []:
            conf[x] = conf.get(x, 0) + 1
        meds |= set(c.get("mediators_adjusted") or [])
        reverse |= any(k["type"] == "reverse causation" for k in c.get("risks") or [])
    def nid(s: str) -> str:
        return re.sub(r"[^A-Za-z0-9]", "_", s)[:30] or "x"
    L = ["```mermaid", "flowchart LR", f'  E["{exposure}"] --> O["{str(outcome).replace("_", " ")}"]']
    for x in sorted(conf, key=lambda k: -conf[k])[:8]:
        L.append(f'  C_{nid(x)}(["{x}"]) --> E')
        L.append(f'  C_{nid(x)} --> O')
    for m in sorted(meds)[:4]:
        L.append(f'  E --> M_{nid(m)}["{m} (mediator?)"] --> O')
    if reverse:
        L.append("  O -. reverse causation? .-> E")
    L.append("```")
    return "\n".join(L)


def language(report_text: str, rows: List[Dict[str, Any]], human_question: bool = True) -> List[Dict[str, str]]:
    """Causal sentences in the report that the evidence cannot carry."""
    supports = [((r.get("causal") or {}).get("design_support") or "unclear", r.get("evidence_type")) for r in rows
                if not r.get("secondary_report")]
    if not supports:
        return []
    any_rct = any(s == "randomised experiment" for s, _ in supports)
    human_exp = any(s == "randomised experiment" for s, et in supports if et == "human_clinical")
    out = []
    from evidence_reviewer import _prose
    for sent in _prose(report_text):              # bullet points included; tables, quotes and code excluded
        if len(sent) > 400:
            continue
        if _CAUSAL_WORDS.search(sent) and not _HEDGED.search(sent):
            if not any_rct and all(s.startswith("observational") or s == "case report / series" for s, _ in supports):
                out.append({"sentence": sent.strip()[:300], "why": "causal wording, but every supporting study is observational"})
            elif human_question and not human_exp and re.search(r"\b(?:patients|people|humans|women|men|adults|children)\b",
                                                                 sent, re.I):
                out.append({"sentence": sent.strip()[:300],
                            "why": "causal claim about people, but no randomised human study supports it"})
    return out


def markdown(rows: List[Dict[str, Any]], diagram: Optional[str], lang_issues: List[Dict[str, str]]) -> str:
    L = ["## Causal inference checks", "",
         "| study | design support | adjusted for | risks |", "|---|---|---|---|"]
    for r in rows:
        c = r.get("causal")
        if not c:
            continue
        L.append(f"| {r.get('ref')} | {c['design_support']} | {', '.join(c['confounders'][:6]) or ('; '.join(c['adjustment_methods']) or '—')} | "
                 f"{'; '.join(k['type'] for k in c['risks']) or '—'} |")
    if diagram:
        L += ["", "Assumed causal structure (from the studies' own adjustment sets - a picture of assumptions, not evidence):",
              "", diagram]
    if lang_issues:
        L += ["", "**Causal wording the evidence cannot carry:**"] + [f"- \"{x['sentence']}\" - {x['why']}" for x in lang_issues[:8]]
    return "\n".join(L) + "\n"
