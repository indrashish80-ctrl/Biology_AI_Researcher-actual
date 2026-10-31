"""
reader_report.py  (v27)
=======================
The report a reader wants: the ANSWER to the question first, then the explanation, in plain language - and nothing
technical in the way. Built only from the verified evidence in the dossier (no new claims).

    build(dossier, router=None) -> markdown

    # <question>
    ## Short answer                one or two sentences
    ## What the research shows     the human studies, then laboratory / animal work, how they fit together
    ## Key studies                 a small table in plain words, linked
    ## How sure can we be?         certainty and the reasons, in plain language
    ## Tested in the virtual lab   hypotheses / claims checked in the simulator (if any)
    ## What this means             practical meaning and the open questions
    ## Sources                     numbered, with links

Citations are (First author et al., year) - never internal codes such as S12, C49, T3 or E1.F1. Run messages, rate
limits, audits, screening logs and indexes go to the optional technical appendix (research_engine appends it).

With an AI key, ONE extra call may rewrite 'What the research shows' more fluently. The code then checks the rewrite:
every author it cites must be in the source list and every number must occur in the verified facts - otherwise the
plain deterministic text is kept.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

READER_VERSION = "1.0"
HUMAN = ("human_clinical", "human_observational")
DIR_VERB = {"decrease": "lowered", "increase": "raised", "no_effect": "did not clearly change", "mixed": "had mixed effects on",
            "unclear": "had an unclear effect on"}
DESIGN_WORD = {"human_clinical": "clinical trial", "human_observational": "observational study", "animal": "animal study",
               "cell_culture": "cell study", "organoid_ex_vivo": "tissue study", "in_silico": "computer study",
               "review_meta": "review", "case_report": "case report", "plant": "plant study", "microbial": "microbial study"}


def _cite(row: Dict[str, Any]) -> str:
    s = str(row.get("study") or "").strip()
    m = re.match(r"(.+?)\s*\((\d{4})\)\s*$", s)
    return f"{m.group(1)}, {m.group(2)}" if m else (s or "unnamed study")


GLOSSARY = {"ki-67": "Ki-67 (a marker of how fast cells divide)", "ki67": "Ki-67 (a marker of how fast cells divide)",
            "hba1c": "HbA1c (average blood sugar over about 3 months)", "crp": "CRP (a marker of inflammation)",
            "ldl": "LDL cholesterol", "egfr": "eGFR (kidney function)", "tunel": "TUNEL (a measure of cell death)",
            "pfs": "progression-free survival", "os": "overall survival", "dfs": "disease-free survival",
            "mace": "major heart and blood-vessel events", "bmi": "body mass index", "homa-ir": "HOMA-IR (insulin resistance)"}


def _plain_endpoint(f: Dict[str, Any]) -> str:
    meas = (f.get("measurement") or "").strip()
    ep = (f.get("endpoint") or "").replace("_", " ").strip()
    if ep.startswith("other:"):
        ep = ep[6:].strip()
    g = GLOSSARY.get(meas.lower()) or GLOSSARY.get(ep.lower())
    if g:
        return g
    if meas and ep and meas.lower() not in ep.lower() and ep.lower() not in meas.lower():
        return f"{meas} ({ep})"
    return meas or ep or "the outcome"


def _cap(t: str) -> str:
    return t[:1].upper() + t[1:]


def _a(word: str) -> str:
    return ("an " if word[:1].lower() in "aeiou" else "a ") + word


def _strip_codes(t: str) -> str:
    """Remove internal references from a sentence: (… observed in C39 …), [S12], T3, E1.F1."""
    t = re.sub(r"\s*\([^()]*\b(?:C\d+|observed in|also verified in)\b[^()]*\)", "", str(t or ""))
    t = re.sub(r"\s*\[?\b(?:S|C|T|X)\d+(?:\.E\d+\.F\d+)?\b\]?", "", t)
    return re.sub(r"\s{2,}", " ", t).strip()


def _finding_sentence(f: Dict[str, Any], intervention: str) -> str:
    d = f.get("direction") or "unclear"
    txt = f"{intervention} {DIR_VERB.get(d, 'affected')} {_plain_endpoint(f)}"
    sub = f.get("subgroup") or (", ".join(f.get("subgroups_separate") or []) if f.get("subgroups_separate") else None)
    if sub:
        txt += f", but only in {sub}"
    p = f.get("p_value")
    if isinstance(p, (int, float)):
        txt += f" (p = {p:g})"
    return txt


def _rows(d: Dict[str, Any]) -> List[Dict[str, Any]]:
    return [r for r in (d.get("evidence_matrix") or {}).get("rows") or [] if r.get("evidence_type") != "review_meta"]


def _answering(rows: List[Dict[str, Any]], direct_only: bool = False) -> List[Tuple[Dict[str, Any], Dict[str, Any]]]:
    """Findings that answer the question; direct_only keeps those that measure the ASKED outcome itself."""
    out = []
    for r in rows:
        if r.get("excluded") or r.get("methodology_excluded"):
            continue
        for f in r.get("findings") or []:
            if f.get("answers_question") and f.get("direction"):
                if direct_only and (f.get("evidence_role") or "direct") != "direct":
                    continue
                if direct_only and not _measures_asked(f, rows):
                    continue
                out.append((r, f))
    return out


_ASKED: List[str] = []
SYNONYMS = {"proliferation": ["ki-67", "ki67", "tumour growth", "tumor growth", "cell growth", "mitotic"],
            "tumour growth": ["tumor growth", "tumour volume", "tumor volume"], "glycaemic": ["hba1c", "glucose"],
            "mortality": ["survival", "death"], "survival": ["mortality", "death"]}


def _measures_asked(f: Dict[str, Any], rows: List[Dict[str, Any]]) -> bool:
    asked = list(_ASKED)
    if not asked:
        return True
    txt = f"{f.get('endpoint') or ''} {f.get('measurement') or ''}".lower()
    return any(a in txt for a in asked)


def _level(d: Dict[str, Any]) -> str:
    es = d.get("executive_synthesis") or {}
    v = d.get("calibrated_verdict") or {}
    lv = (es.get("certainty") or "").lower().strip()
    if lv in ("high", "moderate", "low", "very low"):
        return lv
    return {"strongly_supported": "moderate", "supported_with_limits": "moderate", "mixed": "low",
            "preliminary": "low", "insufficient": "very low"}.get(v.get("category"), "low")


def short_answer(d: Dict[str, Any]) -> str:
    v = d.get("calibrated_verdict") or {}
    cat, direction = v.get("category"), v.get("direction")
    rows = _rows(d)
    ans = _answering(rows, direct_only=True)
    human = [(r, f) for r, f in ans if r.get("evidence_type") in HUMAN]
    trials = [(r, f) for r, f in human if r.get("evidence_type") == "human_clinical"]
    level = _level(d)
    subgroup_only = human and all(f.get("subgroup") or f.get("subgroups_separate") for _, f in human
                                  if f.get("direction") in ("decrease", "increase"))
    effect = {"decrease": "a reduction", "increase": "an increase"}.get(direction or "", "an effect")
    iv = next((f.get("intervention") for _, f in ans if f.get("intervention")), "") or "it"
    outc = _plain_endpoint(ans[0][1]) if ans else "the outcome"
    same = [(r, f) for r, f in trials if f.get("direction") == direction]
    people = sum((r.get("n") or 0) for r, _ in same if isinstance(r.get("n"), int))
    detail = (f" {len(same)} randomised trial{'s' if len(same) != 1 else ''}"
              + (f" ({people:,} people in total)" if people else "")
              + f" found that {iv} {DIR_VERB.get(direction, 'changed')} {outc}.") if same and direction in DIR_VERB else ""
    hedge = "" if level in ("high", "moderate") else (" The evidence is not strong yet, so this could change "
                                                       "(see 'How sure can we be?').")
    if cat == "strongly_supported" and direction in ("decrease", "increase"):
        head = "**Yes.**" if level in ("high", "moderate") else "**Probably yes.**"
        return f"{head}{detail or ' Several human studies agree on ' + effect + '.'}{hedge}"
    if cat == "strongly_supported" and direction == "no_effect":
        return f"**Probably not.** Good-quality human studies consistently found no clear effect on {outc}.{hedge}"
    if cat == "supported_with_limits":
        return f"**Probably, with limits.**{detail or ' The human evidence points to ' + effect + '.'} It has gaps (see 'How sure can we be?')."
    if cat == "mixed":
        s = "**Uncertain - the human studies disagree.**"
        if subgroup_only:
            s += f" {iv[:1].upper() + iv[1:]} appears to help only some subgroups of people, not the study populations as a whole."
        elif human:
            ups = sum(1 for _, f in human if f.get("direction") in ("decrease", "increase"))
            s += f" {ups} of {len(human)} human results show {effect} in {outc}; the others show no clear difference."
        return s
    if cat == "preliminary":
        return ("**Possibly, but it is not shown in people yet.** The signal comes from laboratory, animal or small "
                "early studies.")
    return "**Not known yet.** Too few of the studies read measure what the question asks."


def explanation(d: Dict[str, Any]) -> str:
    rows = _rows(d)
    ans = _answering(rows, direct_only=True)
    iv = ""
    for r, f in ans:
        iv = f.get("intervention") or r.get("intervention") or ""
        if iv:
            break
    iv = iv or "the intervention"
    human = [(r, f) for r, f in ans if r.get("evidence_type") in HUMAN]
    pre = [(r, f) for r, f in ans if r.get("evidence_type") not in HUMAN]
    P: List[str] = []
    if human:
        trials = [x for x in human if x[0].get("evidence_type") == "human_clinical"]
        obs = [x for x in human if x[0].get("evidence_type") == "human_observational"]
        for group, word in ((trials, "clinical trial"), (obs, "observational study")):
            if not group:
                continue
            sents = []
            for r, f in group[:6]:
                n = r.get("n_display") or r.get("n")
                who = f" of {n} people" if isinstance(n, int) and n > 0 else ""
                sents.append(f"{_a(word)}{who} ({_cite(r)}) found that {_finding_sentence(f, iv)}")
            more = f" {len(group) - 6} more {word}(s) are listed under Key studies." if len(group) > 6 else ""
            P.append("In people: " + "; ".join(sents) + "." + more if word == "clinical trial" or not trials else
                     "Observational data: " + "; ".join(sents) + "." + more)
        if obs and trials:
            P.append("Observational studies can show an association but cannot prove cause and effect, so the trials "
                     "weigh more.")
    else:
        P.append("None of the human studies read measured what the question asks.")
    if pre:
        dirs = {}
        for r, f in pre:
            dirs.setdefault(f.get("direction"), []).append(_cite(r))
        parts = [f"{len(set(v))} {'study' if len(set(v)) == 1 else 'studies'} found that {iv} "
                 f"{DIR_VERB.get(k, 'affected')} it ({'; '.join(sorted(set(v))[:4])})" for k, v in dirs.items()]
        P.append("In the laboratory and in animals: " + "; ".join(parts) + ". Laboratory results show a possible "
                 "mechanism, not a benefit in people - doses in such studies are often far higher than people reach.")
    direct_ids = {id(f) for _, f in ans}
    other = [(r, f) for r in rows for f in (r.get("findings") or []) if id(f) not in direct_ids and f.get("direction")
             and r.get("evidence_type") in HUMAN]
    if other:
        eps = sorted({_plain_endpoint(f) for _, f in other})[:5]
        ups = sorted({_cite(r) for r, f in other})
        P.append(f"Related outcomes (not the one asked about) were also studied - {', '.join(eps)} - in "
                 f"{len(ups)} human stud{'y' if len(ups) == 1 else 'ies'}; they are listed in the technical appendix.")
    excl = [r for r in rows if r.get("excluded") or r.get("methodology_excluded")]
    if excl:
        P.append(f"{len(excl)} study(ies) were set aside because of a serious methodological problem (for example a "
                 f"misidentified cell line).")
    return "\n\n".join(P)


def key_studies(d: Dict[str, Any], limit: int = 12) -> Tuple[str, List[Dict[str, Any]]]:
    rows = _rows(d)
    ans = _answering(rows, direct_only=True) or _answering(rows)
    order = {"human_clinical": 0, "human_observational": 1}
    seen, picked = set(), []
    for r, f in sorted(ans, key=lambda x: (order.get(x[0].get("evidence_type"), 2), -(x[0].get("n") or 0))):
        if r.get("ref") in seen:
            continue
        seen.add(r.get("ref"))
        picked.append((r, f))
    if not picked:
        return "", []
    L = ["| Study | Type | Who / what was studied | What was found |", "|---|---|---|---|"]
    sources = []
    for r, f in picked[:limit]:
        n = r.get("n_display") or r.get("n")
        who = re.sub(r"^[—\-\s]+$", "", (r.get("model") or r.get("subtype") or "").replace("|", "/")).strip()
        if who.lower() in ("humans", "human", "not specified"):
            who = ""
        if isinstance(n, int) and n > 0:
            unit = "people" if r.get("evidence_type") in HUMAN else "animals" if r.get("evidence_type") == "animal" else ""
            who = f"{n:,} {unit}".strip() + (f" - {who}" if who else "")
        who = who or "-"
        name = _cite(r)
        name = f"[{name}]({r['link']})" if r.get("link") else name
        L.append(f"| {name} | {DESIGN_WORD.get(r.get('evidence_type'), r.get('evidence_type_label') or 'study')} | "
                 f"{who} | {_cap(_finding_sentence(f, f.get('intervention') or r.get('intervention') or 'it'))} |")
        sources.append(r)
    return "\n".join(L), sources


def certainty(d: Dict[str, Any]) -> str:
    v = d.get("calibrated_verdict") or {}
    level = _level(d)
    reasons = []
    for x in (v.get("criteria_met") or [])[:3]:
        reasons.append(f"in favour: {x}")
    for x in (v.get("limits") or v.get("criteria_missed") or [])[:4]:
        reasons.append(f"against: {x}")
    rows = _rows(d)
    abstract_only = sum(1 for r in rows if (r.get("text_basis") or "abstract") == "abstract" and r.get("answers_question"))
    if abstract_only:
        reasons.append(f"{abstract_only} of the studies that answer the question could only be read as an abstract")
    p = (d.get("prisma") or {})
    read, elig = p.get("deep_read") or p.get("included") or len(rows), p.get("eligible")
    if isinstance(elig, int) and isinstance(read, int) and elig > read:
        reasons.append(f"{read} of {elig} relevant studies were read - unread studies could change the picture")
    comp = d.get("completeness") or {}
    if comp.get("complete") is False:
        reasons.append("some papers could not be read by the AI during this run (the AI service's limits were reached), "
                       "so evidence may be missing - running it again later can fill the gaps")
    L = [f"**Confidence: {level}.**"] + [f"- {r[0].upper() + r[1:]}." for r in reasons]
    return "\n".join(L)


TOOL_PLAIN = {"custom_model": "a purpose-built computer model", "library_model": "a documented published model",
              "bacteria_antibiotic": "a simulated bacterial infection under the drug", "virus_antiviral": "a simulated "
              "virus infection with the drug", "epidemic": "an epidemic simulation", "drug_exposure": "a simulation of the "
              "drug level in the blood", "dose_response": "a dose-response calculation", "dna_analysis": "DNA sequence "
              "analysis", "dna_mutation": "a mutation analysis", "crispr_guides": "CRISPR guide design", "primers": "PCR "
              "primer design", "restriction_sites": "restriction-site mapping", "protein_analysis": "protein analysis",
              "protein_mutation": "a protein-change analysis", "protein_structure": "3D structure",
              "pathway": "a signalling-pathway calculation", "power": "a study-size calculation"}


def lab_section(d: Dict[str, Any]) -> str:
    ver = d.get("claim_verification") or (d.get("executive_synthesis") or {}).get("claim_verification") or []
    hyp = d.get("hypothesis_tests") or []
    if not ver and not hyp:
        return ""
    L = []
    for h in hyp:
        tools = sorted({x.get("tool") for x in h.get("experiments") or [] if x.get("tool") and not x.get("error")})
        how = ", ".join(TOOL_PLAIN.get(t, t.replace("_", " ")) for t in tools)
        who = "Your hypothesis" if h.get("user_hypothesis") else "Hypothesis"
        L.append(f"- **{h.get('verdict_plain') or h.get('verdict')}** - {who}: {_strip_codes(h.get('statement', ''))[:220]}"
                 + (f" *(tested with: {how})*" if how else ""))
    for v in ver:
        if any(v.get("claim") == h.get("statement") for h in hyp):
            continue
        L.append(f"- **{v.get('verdict')}** - {_strip_codes(v.get('claim', ''))[:220]}")
    L.append("\n*These are computer simulations: they show whether an idea is consistent with a model of the biology, "
             "not that it is true in people.*")
    return "\n".join(L)


def meaning(d: Dict[str, Any]) -> str:
    v = d.get("calibrated_verdict") or {}
    q = (d.get("research_question") or "").lower()
    medical = bool(re.search(r"\b(patients?|people|humans?|women|men|children|treat|therapy|dose|drug|cancer|disease|"
                             r"diabetes|infection)\b", q))
    L = []
    cat = v.get("category")
    if cat in ("mixed", "preliminary", "insufficient"):
        L.append("The question is still open. A larger randomised trial in the people the question is about, measuring "
                 "this outcome directly, would settle it.")
    elif cat == "supported_with_limits":
        L.append("The evidence leans one way, but the gaps listed above mean the answer could still change.")
    elif _level(d) in ("high", "moderate"):
        L.append("The evidence is consistent; new large trials would be needed to overturn it.")
    else:
        L.append("The studies found so far agree, but there are few of them and several could only be read in part - "
                 "treat this as a likely answer, not a settled one.")
    gaps = (d.get("research_value") or {}).get("gaps") or []
    for g in gaps[:3]:
        if g.get("search") or g.get("detail"):
            L.append(f"- Open question: {g.get('detail') or g.get('type')}.")
    if medical:
        L.append("\n*This is a summary of published research, not medical advice. Talk to a doctor before changing any "
                 "treatment.*")
    return "\n".join(L)


def _sources_md(sources: List[Dict[str, Any]], rows: List[Dict[str, Any]]) -> str:
    seen, out = set(), []
    for r in sources + [r for r, _ in _answering(rows)]:
        if r.get("ref") in seen:
            continue
        seen.add(r.get("ref"))
        t = re.sub(r"^\[SYNTHETIC\]\s*", "", str(r.get("title") or "")).strip()
        j = r.get("journal")
        link = f" {r['link']}" if r.get("link") else ""
        out.append(f"{len(out) + 1}. {_cite(r)}. {t}{'. ' + j if j else ''}.{link}")
    return "\n".join(out)


_NUM = re.compile(r"(?<![\w.])\d+(?:\.\d+)?")


def polish(router: Any, question: str, plain: str, allowed_cites: List[str]) -> Optional[str]:
    """One AI call rewrites the explanation more readably; accepted only if it cites nothing new and adds no number."""
    if router is None:
        return None
    prompt = ("Rewrite the explanation below for an educated reader who is not a scientist. Keep every fact, add NO new "
              "fact, number or study, keep the citations exactly as written in brackets (Author et al., year), use short "
              "paragraphs, no bullet points, no headings, no internal codes. Return only the rewritten text.\n\n"
              f"QUESTION: {question}\n\nEXPLANATION:\n{plain}")
    try:
        out = (router.ask(prompt, task="synthesis") or "").strip()
    except Exception:
        return None
    if len(out) < 80 or len(out) > 3 * len(plain) + 400:
        return None
    cites = set(re.findall(r"\(([^()]*?\d{4})\)", out))
    allowed = set(allowed_cites)
    for c in cites:
        for part in re.split(r";\s*", c):
            if part.strip() and part.strip() not in allowed:
                return None                     # a citation the facts do not contain
    if set(_NUM.findall(out)) - set(_NUM.findall(plain + " " + question)):
        return None                             # a number the facts do not contain
    if re.search(r"\b[SCTX]\d+\b|E\d+\.F\d+", out):
        return None
    return out


def build(d: Dict[str, Any], router: Any = None) -> str:
    q = d.get("research_question") or "Research question"
    rows = _rows(d)
    m = d.get("evidence_matrix") or {}
    asked = [str(x).lower().replace("_", " ") for x in (m.get("question_endpoints") or []) if x]
    _ASKED.clear()
    for a in asked:
        _ASKED.extend([a] + SYNONYMS.get(a, []))
    table, sources = key_studies(d)
    plain = explanation(d)
    better = polish(router, q, plain, [_cite(r) for r in rows]) if router is not None else None
    L = [f"# {q}", ""]
    if (d.get("speed") or "") == "quick":
        L += ["> **Quick scan** - this answer is based on the abstracts of the top papers only. Run a *Research dossier* "
              "for a fuller answer.", ""]
    L += ["## Short answer", "", short_answer(d), "", "## What the research shows", "", better or plain, ""]
    if table:
        L += ["## Key studies", "", table, ""]
    L += ["## How sure can we be?", "", certainty(d), ""]
    lab = lab_section(d)
    if lab:
        L += ["## Tested in the virtual lab", "", lab, ""]
    L += ["## What this means", "", meaning(d), ""]
    src = _sources_md(sources, rows)
    if src:
        L += ["## Sources", "", src, ""]
    L += ["---", "", "*Technical details - how the studies were found and screened, every verified quote, the quality "
                     "checks, and the run log - are in the technical appendix below (optional reading).*", ""]
    text = "\n".join(L)
    return re.sub(r"\[SYNTHETIC\]\s*", "", text)
