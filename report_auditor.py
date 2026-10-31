"""
report_auditor.py
-----------------
Auto-healer: reads what a run produced (dossier + report text), detects the failure patterns found in real
reports, and tells the engine how to REPAIR THE RUN. It never edits the program's source code - a program that
rewrites itself from its own reports cannot check its own edits, and one bad rewrite can silently break every
later run. Instead:

  * repairable problems  -> the engine re-runs the investigation with a fix (broader concept queries, a larger
                            screening / deep-read budget, forced re-extraction), up to MAX_ROUNDS extra rounds,
                            and keeps a round only if it is better than the previous one;
  * in-place repairs     -> applied to the dossier (e.g. template hypotheses with no evidence are removed);
  * anything left        -> listed in the report ("Automatic quality check & repairs") and written to
                            BUG_REPORT.md so it can be sent to the developer.

Checks (each one comes from a real report):
  Q1 question words used as an intervention / subject ("Can regulates ...", "What Are demonstrated ...")
  Q2 too few papers deep-read while eligible records were left unread
  Q3 no verified finding answers the question
  Q4 off-question / non-English / computational-method papers among the included studies
  Q5 implausible sample sizes (cell counts reported as n)
  Q6 template hypotheses with no verified support
  Q7 broken report text (raw JSON, question words as subjects, withheld markers in deterministic text)
  Q8 PRISMA arithmetic inconsistent
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import Any, Dict, List, Optional

MAX_ROUNDS = 1          # one repair round at most (a second rarely added evidence and doubled the run time)
SCREEN_CAP, DEEP_CAP = 300, 40          # the same ceilings as the website allows
_QWORD = re.compile(r"^(?:can|could|does|do|did|is|are|was|were|what|which|how|why|when|where|who|should|would|will|"
                    r"what are|what is|what are the|the|a|an)$", re.I)
_QWORD_SUBJECT_TEXT = re.compile(r"\b(?:Can|Could|Does|Do|Is|Are|What Are|What Is|How Does|Why Does)\s+(?:regulates?|mediates?|"
                                 r"demonstrated|increased|decreased|reduced|modulat\w*|engagement|effects of)\b")
_TEMPLATE = re.compile(r"concentration-dependent signal(?:l)?ing threshold", re.I)


def _issue(code: str, severity: str, detail: str, repair: str, action: Optional[str] = None) -> Dict[str, Any]:
    return {"code": code, "severity": severity, "detail": detail, "repair": repair, "action": action}


def audit_run(dossier: Dict[str, Any], report_text: str, question: str,
              budgets: Optional[Dict[str, int]] = None) -> List[Dict[str, Any]]:
    """Inspect one finished run. Returns a list of issues, each with the repair the engine can apply."""
    from evidence_identity import is_question_word_subject, question_frame
    budgets = budgets or {}
    issues: List[Dict[str, Any]] = []
    matrix = dossier.get("evidence_matrix") or {}
    rows = matrix.get("rows") or []
    totals = matrix.get("totals") or {}
    prisma = dossier.get("prisma") or {}
    counts = prisma.get("counts") or {}
    frame = question_frame(question)

    # Q1 question words as a subject - in claims, hypotheses or the report
    bad_claims = [f.get("intervention") for r in rows for f in r.get("findings") or []
                  if f.get("intervention") and (is_question_word_subject(f["intervention"]) or _QWORD.match(f["intervention"].strip()))]
    bad_h = [h.get("statement") for h in dossier.get("hypotheses") or []
             if _QWORD_SUBJECT_TEXT.search(h.get("statement") or "") or _QWORD.match(((h.get("statement") or "").split() or [""])[0])
             and re.search(r"\bregulates?\b", h.get("statement") or "")]
    if bad_claims or bad_h or _QWORD_SUBJECT_TEXT.search(report_text or ""):
        issues.append(_issue("Q1", "critical", "question words used as an intervention / subject: "
                             + "; ".join(str(x) for x in (bad_claims + bad_h)[:3] or ["(in the report text)"]),
                             "re-extract claims and drop template hypotheses", "reextract"))

    # Q2 too few deep-read while more eligible records exist
    deep_k = budgets.get("deep_read_k") or 25
    included = counts.get("included", len(rows))
    left = (counts.get("eligible_not_deep_read") or 0) + (counts.get("not_screened") or 0)
    if counts and included < deep_k and left > 0:            # any unused budget with unread records -> read more
        issues.append(_issue("Q2", "major", f"only {included} paper(s) deep-read (budget {deep_k}) although {left} retrieved "
                             f"record(s) were not read", "re-run with a larger screening / deep-read budget", "expand"))

    # Q3 nothing answers the question
    if matrix and not totals.get("findings_answering_question"):
        issues.append(_issue("Q3", "major", "no verified finding measures what the question asks"
                             + (f" ({', '.join(frame.get('endpoints') or [])})" if frame.get("endpoints") else ""),
                             "re-run with broader concept queries (disease subtypes, outcome synonyms) and a larger budget",
                             "broaden"))

    # Q4 off-question / non-English / computational papers in the included set
    try:
        from prisma_screening import eligibility
        from retrieval.models import PaperCandidate
        off = []
        for r in rows:
            why = eligibility(PaperCandidate(title=r.get("title") or r.get("study") or "", abstract=r.get("abstract") or ""), frame)
            if why and ("English" in why or "computational" in why or why.startswith("does not concern")):
                off.append(f"{r.get('ref')}: {why}")
        if off:
            issues.append(_issue("Q4", "major", "included paper(s) that should have been excluded: " + "; ".join(off[:4]),
                                 "excluded from the evidence on the next round", "broaden"))
    except Exception:
        pass

    # Q5 implausible n
    weird = [f"{r.get('ref')} n={r.get('n')}" for r in rows if isinstance(r.get("n"), int)
             and ((r.get("evidence_type") in ("cell_culture", "organoid_ex_vivo", "animal") and r["n"] > 2000) or r["n"] > 5_000_000)]
    if weird:
        issues.append(_issue("Q5", "minor", "implausible sample size(s): " + ", ".join(weird), "flagged for the developer"))

    # Q6 template hypotheses without verified support
    tmpl = [h for h in dossier.get("hypotheses") or [] if _TEMPLATE.search(h.get("statement") or "")
            and not h.get("derived_from") and not ((h.get("support") or {}).get("score"))]
    if tmpl:
        issues.append(_issue("Q6", "major", f"{len(tmpl)} template hypothesis(es) with no verified support",
                             "removed from the report", "drop_templates"))

    # Q7 broken report text
    broken = []
    if '{"experiment_title"' in (report_text or ""):
        broken.append("raw JSON experiment")
    syn = dossier.get("executive_synthesis") or {}
    if syn.get("mode") == "deterministic" and "⛔" in (syn.get("executive_summary") or ""):
        broken.append("withheld-statement marker inside the deterministic summary")
    if broken:
        issues.append(_issue("Q7", "minor", "report text problems: " + ", ".join(broken), "flagged for the developer"))

    # Q8 PRISMA arithmetic
    cons = prisma.get("consistency") or {}
    if cons and not cons.get("ok", True):
        issues.append(_issue("Q8", "critical", "PRISMA counts inconsistent: " + "; ".join(cons.get("errors") or [])[:300],
                             "flagged for the developer"))
    return issues


def repair_plan(issues: List[Dict[str, Any]], budgets: Dict[str, int], question: str, round_no: int,
                use_prisma: bool = True) -> Optional[Dict[str, Any]]:
    """What to change for the next round, or None when re-running cannot help."""
    actions = {i["action"] for i in issues if i.get("action")}
    if round_no >= MAX_ROUNDS or not (actions & {"expand", "broaden", "reextract"}):
        return None
    from research_planner import build_keyword_queries
    screen_n = budgets.get("screen_n") or 150
    deep_k = budgets.get("deep_read_k") or 25
    max_papers = budgets.get("max_papers") or 6
    new = {"screen_n": min(SCREEN_CAP, int(screen_n * 1.5) + 10), "deep_read_k": min(DEEP_CAP, deep_k + 5),
           "max_papers": min(12, max_papers + 3), "force_reextract": "reextract" in actions, "extra_queries": []}
    if "broaden" in actions or "expand" in actions:
        qs = build_keyword_queries(question, max_queries=10)
        new["extra_queries"] = qs[3:] if len(qs) > 3 else []
        # outcome-free variant: the disease / exposure core alone finds studies that use other outcome words
        from evidence_identity import question_frame
        fr = question_frame(question)
        core = " ".join(x for x in ([fr["disease"]] if fr.get("disease") else []) + (fr.get("exposures") or [])[:2] if x)
        if core:
            new["extra_queries"].append(core)
    if use_prisma and new["screen_n"] == screen_n and new["deep_read_k"] == deep_k and not new["extra_queries"] \
            and not new["force_reextract"]:
        return None                       # budgets already at the maximum and nothing new to try
    return new


def run_quality(dossier: Dict[str, Any]) -> tuple:
    """Higher is better: (verified findings answering the question, independent findings, papers deep-read)."""
    t = (dossier.get("evidence_matrix") or {}).get("totals") or {}
    return (t.get("findings_answering_question") or 0, t.get("independent_findings") or 0,
            ((dossier.get("prisma") or {}).get("counts") or {}).get("included") or len((dossier.get("evidence_matrix") or {}).get("rows") or []))


def apply_inplace(dossier: Dict[str, Any], issues: List[Dict[str, Any]]) -> List[str]:
    """Repairs that need no re-run. Returns what was done."""
    done = []
    if any(i["action"] == "drop_templates" for i in issues):
        before = len(dossier.get("hypotheses") or [])
        dossier["hypotheses"] = [h for h in dossier.get("hypotheses") or [] if not (
            _TEMPLATE.search(h.get("statement") or "") and not h.get("derived_from") and not ((h.get("support") or {}).get("score")))]
        done.append(f"removed {before - len(dossier['hypotheses'])} template hypothesis(es) with no verified support")
    return done


def heal_markdown(heal: Dict[str, Any]) -> str:
    """Report section describing what the auto-healer checked, repaired and could not repair."""
    lines = ["\n## Automatic quality check & repairs\n",
             "_After every run the engine checks its own report for known failure patterns and repairs the RUN "
             "(new search, larger budget, re-extraction). It never edits its own program code._\n"]
    for h in heal.get("history") or []:
        found = "; ".join(f"{i['code']} {i['detail']}" for i in h.get("issues") or []) or "no problems found"
        lines.append(f"- **Round {h['round']}:** {found}" + (f" → **repair:** {h['repair']}" if h.get("repair") else "")
                     + (f" ({h['note']})" if h.get("note") else ""))
    if heal.get("kept_round") is not None:
        lines.append(f"- Kept round {heal['kept_round']} (the round with the most verified evidence answering the question).")
    for d in heal.get("inplace") or []:
        lines.append(f"- In-place repair: {d}")
    rem = heal.get("remaining") or []
    if rem:
        lines.append("\n**Not repaired automatically (written to BUG_REPORT.md):**")
        lines += [f"- {i['code']} ({i['severity']}): {i['detail']}" for i in rem]
    else:
        lines.append("\n_No unresolved problems._")
    return "\n".join(lines) + "\n"


def write_bug_report(path: str, question: str, heal: Dict[str, Any]) -> Optional[str]:
    rem = heal.get("remaining") or []
    if not rem:
        return None
    body = [f"# BioSynthetica bug report ({datetime.now().strftime('%Y-%m-%d %H:%M')})", "",
            f"Question: {question}", "", "Problems the auto-healer could not repair (send this file to the developer):", ""]
    body += [f"- **{i['code']} ({i['severity']})** {i['detail']}" for i in rem]
    body += ["", "Repair history:", ""] + [f"- round {h['round']}: " + "; ".join(i["code"] for i in h.get("issues") or [])
                                          for h in heal.get("history") or []]
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(body) + "\n")
    return path
