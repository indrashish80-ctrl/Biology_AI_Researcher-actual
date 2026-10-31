"""
enhanced_report.py
------------------
Wraps the original ResearchReportGenerator output with:
  Executive Summary (deep reasoning) + Markdown Evidence Matrix (Human / Animal / In vitro)
  + PRISMA flow + STRING-db interactions, followed by the unchanged original report.
"""

from __future__ import annotations

import json
import re

from core_utils import ref_key
from pathlib import Path
from typing import Any, Dict, List, Optional

from evidence_matrix import BUCKET_LABELS, matrix_markdown
from reasoning_guards import traceability_markdown

_BASIS = {"jats_xml": "JATS XML (PMC / Europe PMC)", "unpaywall_pdf": "Open-access PDF via Unpaywall",
          "oa_pdf": "Open-access PDF via OpenAlex / Semantic Scholar",
          "abstract": "Abstract only"}


def prisma_markdown(prisma: Optional[Dict[str, Any]], matrix: Optional[Dict[str, Any]] = None) -> str:
    if not prisma:
        return "_PRISMA screening was not used for this run (legacy retrieval)._\n"
    c = prisma["counts"]
    cons = prisma.get("consistency") or {}
    errors = list(cons.get("errors", [])) if cons and not cons.get("ok", True) else []
    if prisma.get("ledger") is not None:            # recomputed at publication - a stored 'ok' is not trusted
        from prisma_screening import validate_prisma
        pids = prisma.get("paper_ids") or [x.get("paper_id") for x in prisma["ledger"] if x.get("included")]
        errors += [e for e in validate_prisma(c, prisma["ledger"], pids) if e not in errors]
    calc = c.get("identified_databases", c["identified"]) + c.get("identified_citation_chasing", 0) - c["duplicates_removed"]
    if calc != c["after_dedup"]:
        errors.append(f"{c.get('identified_databases', c['identified'])} + {c.get('identified_citation_chasing', 0)} - "
                      f"{c['duplicates_removed']} = {calc}, but {c['after_dedup']} unique records are reported")
    if errors:
        return ("> **PRISMA CONSISTENCY ERROR** - the record counts do not reconcile, so no flow diagram is published.\n>\n"
                + "\n".join(f"> - {e}" for e in errors) + "\n")
    off = (matrix or {}).get("off_question") or []
    arithmetic = (f"- Arithmetic (recomputed): {c.get('identified_databases', c['identified'])} database + "
                  f"{c.get('identified_citation_chasing', 0)} citation-chasing − {c['duplicates_removed']} duplicates = "
                  f"**{c['after_dedup']} unique records** ✓\n")
    off_line = (f"- Of the {c['included']} included papers, **{len(off)} were removed at the evidence-identity check** "
                f"(about a different disease / organ system): " + "; ".join(f"{x['ref']} {_c(x['title'], 60)} - {x['reason']}" for x in off)
                + "\n") if off else ""
    src = ", ".join(f"{k.replace('_', ' ').title()}: {v}" for k, v in c["by_source"].items())
    reasons = "\n".join(f"  - {k}: {v}" for k, v in c["exclusion_reasons"].items()) or "  - none"
    ft_reasons = "\n".join(f"  - {k}: {v}" for k, v in (c.get("fulltext_exclusion_reasons") or {}).items()) or "  - none"
    basis = ", ".join(f"{_BASIS.get(k, k)}: {v}" for k, v in c["full_text_basis"].items()) or "—"
    return (
        "| Stage | Records |\n|---|---|\n"
        f"| Identified in databases (all queries × sources) | {c.get('identified_databases', c['identified'])} |\n"
        f"| + Identified by citation chasing (Semantic Scholar) | {c.get('identified_citation_chasing', 0)} |\n"
        f"| = Total identified | {c['identified']} |\n"
        f"| − Duplicates removed | {c['duplicates_removed']} |\n"
        f"| = Unique records | {c['after_dedup']} |\n"
        f"| Not screened (screening budget - lower-ranked) | {c['not_screened']} |\n"
        f"| Title/abstract screened | {c['screened']} |\n"
        f"| Screened and excluded | {c['excluded_screening']} |\n"
        f"| Eligible | {c['eligible']} |\n"
        f"| Eligible but NOT deep-read (deep-read budget) | {c['eligible_not_deep_read']} |\n"
        f"| Full text sought | {c.get('fulltext_assessed', c['deep_read'])} |\n"
        f"| Eligible but inaccessible / excluded at full text | {c.get('excluded_fulltext', 0)} |\n"
        f"| Deep-read (full text where legal & open) | {c['deep_read']} |\n"
        f"| **Included in synthesis** | **{c['included']}** |\n\n"
        f"- Sources: {src}\n- Full-text basis of included studies: {basis}\n"
        + arithmetic + off_line +
        f"- Exclusion reasons at screening:\n{reasons}\n- Exclusion reasons at full text:\n{ft_reasons}\n"
        + (f"- Consistency: all {cons.get('checked_identities', 14)} PRISMA identities verified "
           f"(every record has an ID R0001-R{c['after_dedup']:04d}; see dossier ledger).\n" if cons else "")
    )


def _c(v: Any, n: int = 160) -> str:
    s = re.sub(r"\s+", " ", str(v if v not in (None, "") else "—")).replace("|", "\\|").strip()
    return s if len(s) <= n else s[: n - 1] + "…"


def classification_markdown(matrix: Dict[str, Any]) -> str:
    rows = matrix.get("rows") or []
    if not rows:
        return "_No studies._\n"
    out = ["| Ref | Evidence type | Level | Rule that decided it (frozen) | Other designs present |",
           "|---|---|---|---|---|"]
    for r in rows:
        out.append(f"| {r['ref']} | {_c(r.get('evidence_type_label'))} | {_c(r.get('level_label'))} | "
                   f"{_c(r.get('classification_rule'))} | {_c(', '.join(x.replace('_', ' ') for x in r.get('secondary_designs') or []) or '—')} |")
    return "\n".join(out) + ("\n\n_Each paper is classified once by an ordered rule list and stored; later steps cannot "
                             "move it between categories. Reviews (S) are not counted as independent findings._\n")


def quality_markdown(matrix: Dict[str, Any]) -> str:
    rows = [r for r in matrix.get("rows") or [] if r.get("quality")]
    if not rows:
        return "_No quality profile available._\n"
    # v26: methodological quality and relevance are kept apart - relevance is the LAST column, after a separator,
    # and never combined with the quality dimensions
    dims = ("claim_grounding", "paper_quality", "causal_strength", "replication",
            "experimental_quality", "risk_of_bias", "translational_relevance", "evidence_relevance")
    out = ["| Ref | Grounding (did the source say it?) | Study quality | Causal strength | Replication | "
           "Experimental quality | Risk of bias | Translational relevance | ‖ Relevance to THIS question (not quality) |",
           "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        out.append(f"| {r['ref']} | " + " | ".join(_c((r["quality"].get(d) or {}).get("rating", "—")) for d in dims) + " |")
    out.append("\n<details><summary>Reasons per dimension</summary>\n")
    for r in rows:
        out.append(f"- **{r['ref']}** " + "; ".join(f"*{d.replace('_', ' ')}*: {', '.join(r['quality'][d]['reasons'])}" for d in dims if d in r["quality"]))
    out.append("\n</details>\n\n_Separate dimensions - never combined into one score; relevance to the question is shown "
               "apart from quality (a rigorous study can still be irrelevant). The formal RoB 2 / ROBINS-I / SYRCLE "
               "assessment and the full quality profile follow in their own sections._\n")
    return "\n".join(out)


def dose_markdown(matrix: Dict[str, Any]) -> str:
    rows = [r for r in matrix.get("rows") or [] if (r.get("dose_relevance") or {}).get("statement")]
    if not rows:
        return "_No doses found._\n"
    lab = {"physiological": "Physiological", "borderline": "Borderline", "supra-physiological": "Supra-physiological",
           "not assessed": "Not assessed", "not reported": "Not reported", "not applicable": "Not applicable"}
    return "\n".join(f"- **{r['ref']}** · {lab.get(r['dose_relevance']['category'], r['dose_relevance']['category'])} — "
                     f"{r['dose_relevance']['statement']}" for r in rows) + \
        "\n\n_Human exposure references are approximate published plasma ranges; animal doses are converted to " \
        "human-equivalent doses with FDA body-surface-area factors._\n"


def _finding_claims(matrix: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Primary verified findings as {claim_model, canonical_direction} (for the evidence-level conflict line)."""
    out = []
    for r in (matrix or {}).get("rows") or []:
        for f in r.get("findings") or []:
            if f.get("evidence_role") == "secondary_report" or r.get("evidence_type") == "review_meta":
                continue
            out.append({"claim_model": f.get("model_type"), "canonical_direction": f.get("direction"),
                        "canonical_outcome": f.get("endpoint")})
    return out


def method_flags_markdown(matrix: Optional[Dict[str, Any]], synthesis: Optional[Dict[str, Any]] = None) -> str:
    """Cell lines that are not what the paper calls them, compounds whose laboratory signals are usually artefacts,
    and trials whose published primary outcome differs from the one they registered."""
    rows = (matrix or {}).get("rows") or []
    flags = [f for r in rows for f in (r.get("method_flags") or [])]
    switches = ((synthesis or {}).get("outcome_switching") or [])
    reg = (synthesis or {}).get("registry_check") or {}
    reg_note = ""
    if reg.get("stopped"):
        reg_note = (f" The registered-outcome check was **not completed** ({reg['stopped']} after {reg.get('looked', 0)} "
                    f"lookup(s)), so no conclusion is drawn about outcome switching in this run.")
    elif reg.get("answered"):
        reg_note = f" Registered outcomes were checked for {reg['answered']} trial(s)."
    if not flags and not switches:
        return ("_No misidentified cell line, assay-interference compound or registered-outcome mismatch was found in "
                "the studies read." + reg_note + "_\n")
    out = []
    crit = [f for f in flags if f.get("severity") == "critical"]
    if crit:
        out.append("| Ref | Flag | What it means |\n|---|---|---|")
        out += [f"| {f.get('ref', '?')} | **{f['kind'].replace('_', ' ')}** ({f.get('line') or f.get('compound')}) "
                f"| {_c(f['statement'], 300)} |" for f in crit]
        out.append("")
    rest = [f for f in flags if f.get("severity") != "critical"]
    if rest:
        out.append("| Ref | Flag | Note |\n|---|---|---|")
        out += [f"| {f.get('ref', '?')} | {f['kind'].replace('_', ' ')} ({f.get('line') or f.get('compound')}) "
                f"| {_c(f['statement'], 300)} |" for f in rest]
        out.append("")
    if switches:
        out.append("| Ref | Trial | Registered primary outcome | What the paper reports |\n|---|---|---|---|")
        out += [f"| {x.get('ref', '?')} | {x.get('nct', '?')} | {_c('; '.join(x.get('registered') or []), 90)} "
                f"| {_c(x.get('paper') or '', 120)} |" for x in switches]
        out.append("\n_A paper that reports a different main outcome from the one it registered may be presenting a "
                   "secondary finding as the trial's conclusion. The registry entry is public - the link above each "
                   "trial identifier leads to it._\n")
    if reg_note:
        out.append("_" + reg_note.strip() + "_\n")
    out.append("_Cell-line identity comes from the ICLAC Register of Misidentified Cell Lines; interference compounds "
               "are the classes named in the pan-assay interference (PAINS) literature; registered outcomes come from "
               "ClinicalTrials.gov. Each flag names its basis so it can be checked._\n")
    return "\n".join(out) + "\n"


def meta_markdown(matrix: Optional[Dict[str, Any]]) -> str:
    """Pooled effect sizes with a forest plot - what the studies say together, weighted by their own precision.

    This is what a systematic review reports instead of counting studies for and against. It appears only when at
    least two independent studies report an effect size with a confidence interval on the same endpoint.
    """
    from meta_analysis import analyse, duplicate_effects, forest_plot, not_poolable
    from semantic_audit import FAMILY_LABEL
    _BUCKET = {"human": "human studies", "animal": "animal studies", "cell": "cell / organoid studies"}
    rows = (matrix or {}).get("rows") or []
    results = analyse(rows)
    missing = not_poolable(rows)
    if not results:
        why = ("No two independent studies report an effect size with a confidence interval on the same endpoint, so "
               "nothing can be pooled. The findings are reported individually above.")
        if missing:
            why += (f" Studies whose verified passage reports no interval: "
                    f"{', '.join(sorted({m['ref'] for m in missing if m.get('ref')}, key=ref_key))}.")
        dups = duplicate_effects(rows)
        if dups:
            why += (" Counted once because they report an identical estimate and interval (identical numbers cannot "
                    "come from two independent samples): "
                    + "; ".join(f"{d['ref']} = {d['same_as']} ({d['effect']})" for d in dups[:8]) + ".")
        return "_" + why + "_\n"
    out = ["_Each study is weighted by the precision of its own estimate (inverse-variance, DerSimonian-Laird "
           "random-effects), so a small trial no longer counts the same as a large one. Ratios (HR / OR / RR) and "
           "absolute differences are pooled separately, never together; publications of one trial count once; human, "
           "animal and cell studies are pooled separately. I-squared is how much of the variation is disagreement "
           "between studies rather than chance._\n"]
    for r in results:
        label = FAMILY_LABEL.get(r["endpoint"], str(r["endpoint"] or "?").replace("other:", "").replace("_", " "))
        unit = "ratio" if r["kind"] == "ratio" else "absolute difference"
        out.append(f"**{label} - {_BUCKET.get(r['bucket'], r['bucket'])} ({r['k']} independent studies, {unit})**\n")
        out.append(f"Pooled {'ratio' if r['kind'] == 'ratio' else 'difference'}: **{_num(r['estimate'], r['kind'])}** "
                   f"(95% CI {_num(r['lo'], r['kind'])} to {_num(r['hi'], r['kind'])}), "
                   f"{'statistically significant' if r['significant'] else 'not statistically significant'} "
                   f"(P = {r['p']:.3g}). Heterogeneity: I-squared {r['i2']:.0f}% - {r['heterogeneity']}; "
                   f"Q = {r['q']:.2f} on {r['df']} degrees of freedom.\n")
        out.append("```\n" + "\n".join(forest_plot(r)) + "\n```\n")
        out.append("| Ref | Effect (95% CI) | Weight |\n|---|---|---|")
        out += [f"| {x['ref']} | {_num(x['estimate'], r['kind'])} ({_num(x['lo'], r['kind'])} to "
                f"{_num(x['hi'], r['kind'])}) | {x['weight_pct']:.0f}% |" for x in r["studies"]]
        out.append("")
        if r["i2"] >= 75:
            out.append("> The studies disagree too much for this pooled value to summarise them; it is shown for "
                       "completeness, and the individual results above are the honest answer.\n")
        if r.get("duplicates"):
            out.append("_Counted once: " + "; ".join(f"{d['ref']} reports the same estimate as {d['same_as']} "
                                                      f"({d['effect']})" for d in r["duplicates"][:6])
                       + " - identical numbers cannot come from two independent samples, so pooling them twice would "
                         "invent precision._\n")
    if missing:
        refs = ", ".join(sorted({m["ref"] for m in missing if m.get("ref")}, key=ref_key))
        out.append(f"_Not pooled (the verified passage reports no confidence interval, so the study cannot be "
                   f"weighted): {refs}. These studies are not evidence against the pooled value - they simply "
                   f"cannot enter it._\n")
    return "\n".join(out) + "\n"


def _num(v: float, kind: str) -> str:
    return f"{v:.2f}" if kind == "ratio" else f"{v:.3g}"


def contradictions_markdown(items: List[Dict[str, Any]], matrix: Optional[Dict[str, Any]] = None) -> str:
    from contradiction_detection import contradiction_vocabulary
    vocab = f"**{contradiction_vocabulary(items or [], _finding_claims(matrix))}**\n"
    if not items:
        return vocab + "\n_No two verified findings disagree in direction for the same intervention and endpoint._\n"
    order = {"TRUE_CONTRADICTION": 0, "UNRESOLVED_DISAGREEMENT": 1, "CONTEXT_DIFFERENCE": 2}
    kind = lambda x: x.get("kind") or (x.get("comparison") or {}).get("kind") or "same_system"  # noqa: E731
    same = [c for c in items if kind(c) != "model_mismatch"]
    cross = [c for c in items if kind(c) == "model_mismatch"]
    out = [vocab, f"_{len(same)} disagreement(s) within the same kind of experimental system · {len(cross)} across different "
                  f"systems (e.g. patients vs cells). A direct contradiction needs the same system, endpoint and context; the "
                  f"others are real disagreements explained (or not) by differences in system, population, dose or design._\n"]
    ordered = sorted(same, key=lambda x: order.get(x.get("verdict"), 3)) + cross
    cref, tri = _claim_ref(matrix), _trial_of(matrix)
    for i, c in enumerate(ordered, 1):
        if i == len(same) + 1 and cross:
            out.append("#### Cross-context disagreements - different experimental systems\n")
        label = ("CROSS-CONTEXT (different systems)" if kind(c) == "model_mismatch"
                 else "DIRECT CONTRADICTION" if c.get("verdict") == "TRUE_CONTRADICTION"
                 else "CROSS-CONTEXT DISAGREEMENT" if c.get("verdict") == "CONTEXT_DIFFERENCE" else "UNRESOLVED DISAGREEMENT")
        ra, rb = cref.get(c.get("claim_a_id")), cref.get(c.get("claim_b_id"))
        same_trial = ra and rb and (ra == rb or (tri.get(ra) and tri.get(ra) == tri.get(rb)))
        if same_trial:
            label = "SAME TRIAL - two analyses / subgroups, not two studies"
        out.append(f"**D{i} · {label}** — claim C{c.get('claim_a_id')} ({ra or '?'}) vs claim C{c.get('claim_b_id')} ({rb or '?'})  ")
        if same_trial:
            out.append(f"_Both results come from {'the same paper' if ra == rb else 'publications of the same trial (' + tri[ra] + ')'}: "
                       f"they describe different subgroups or analyses of one population and are not independent evidence "
                       f"against each other._\n")
        out.append(f"{c.get('reasoning') or ''}\n")
        comp = (c.get("comparison") or {}).get("comparison") or []
        if comp:
            out.append("| Dimension | Study A | Study B | Status |\n|---|---|---|---|")
            out.extend(f"| {r['dimension'].replace('_', ' ')} | {_c(r['a'], 60)} | {_c(r['b'], 60)} | {r['status']} |" for r in comp)
            out.append("")
    return "\n".join(out) + "\n"


_EXP_LABELS = [("experiment_title", ""), ("model", "Model"), ("independent_variable", "Vary"), ("dependent_variable", "Measure"),
               ("controls", "Controls"), ("power_notes", "Sample size / power"), ("readout", "Read-out"), ("duration", "Duration")]


def experiment_text(ne: Any) -> str:
    """A proposed experiment as readable text (it is stored as JSON) - never a raw {"experiment_title": ...} dump."""
    if isinstance(ne, str):
        try:
            ne = json.loads(ne)
        except (ValueError, TypeError):
            return ne
    if not isinstance(ne, dict):
        return str(ne)
    parts = []
    for k, label in _EXP_LABELS:
        if ne.get(k):
            parts.append(f"{label}: {ne[k]}" if label else str(ne[k]))
    parts += [f"{k.replace('_', ' ').capitalize()}: {v}" for k, v in ne.items() if v and k not in dict(_EXP_LABELS)]
    return " — ".join(parts[:1]) + (" · " + "; ".join(parts[1:]) if len(parts) > 1 else "")


def user_experiment_markdown(ux: Dict[str, Any]) -> str:
    """The user's own data: its statistics, how it compares with the literature, and the hypothesis test - kept
    out of the evidence matrix, contradictions, bottom line and certainty (it is not peer-reviewed literature)."""
    a = ux.get("analysis") or {}
    tg = a.get("two_group") or {}
    out = [f"_Dataset: {ux.get('dataset') or '?'} · grouping column: {a.get('group_column') or '?'} · outcome column: "
           f"{a.get('outcome_column') or '?'}. These results are computed from YOUR file; they are not part of the literature "
           f"evidence, the bottom line or the certainty rating above._\n"]
    if a.get("error"):
        out.append(f"**The dataset could not be analysed:** {a['error']}\n")
        return "\n".join(out)
    if tg:
        out.append(f"- **{tg.get('group_a')}** (n = {tg.get('n_a')}, mean {tg.get('mean_a', 0):.4g}) vs **{tg.get('group_b')}** "
                   f"(n = {tg.get('n_b')}, mean {tg.get('mean_b', 0):.4g}); difference {tg.get('mean_difference_a_minus_b', 0):.4g} "
                   f"(95% CI {', '.join(f'{x:.4g}' for x in tg.get('mean_difference_95_ci') or [])}); Welch p = "
                   f"{tg.get('welch_p_value', 1):.3g}; Cohen's d = {tg.get('cohens_d', 0):.2f}; Mann-Whitney p = "
                   f"{tg.get('mann_whitney_p_value', 1):.3g}.")
    out.append(f"- **Direction in your data:** {str(a.get('empirical_direction') or 'unclear').replace('_', ' ')} "
               f"({a.get('outcome_column')}, endpoint family: {str(a.get('endpoint_family') or '?').replace('_', ' ')}).")
    comps = a.get("comparisons") or []
    if comps:
        out.append(f"- **Compared with {len(comps)} published finding(s) on the same endpoint and agent:** "
                   f"{a.get('replications', 0)} agree, {a.get('divergences', 0)} disagree.")
        for c in comps[:8]:
            out.append(f"  - C{c.get('claim_id')} ({str(c.get('model') or '?').replace('_', ' ')}): published "
                       f"{c.get('published_direction')} → your data {c.get('empirical_direction')} → **{c.get('status', '').replace('_', ' ').lower()}**")
    else:
        out.append("- No verified published finding measured the same endpoint with the same agent, so your data could not "
                   "be compared with the literature.")
    ht = ux.get("hypothesis_test") or {}
    if ht:
        out.append(f"- **Hypothesis tested on your data:** H#{ht.get('hypothesis_id')} \"{(ht.get('hypothesis') or '')[:160]}\" → "
                   f"{str(ht.get('new_status') or '').replace('_', ' ')} on your data ({ht.get('verdict')}); {ht.get('match')}. "
                   f"Its literature-based ranking is unchanged.")
    return "\n".join(out) + "\n"


_KIND_HEAD = [
    ("replication", "Replication / translation hypotheses",
     "Each asks whether a VERIFIED finding holds in another system or population. They do not explain why the effect "
     "occurs - no mechanism is proposed."),
    ("mechanistic", "Mechanistic hypotheses (evidence-based)",
     "Each step is named in a verified source passage (M2 association or M3 perturbation). They propose HOW the effect "
     "arises; the full chain still has to be tested."),
    ("network_candidate", "Mechanism candidates from protein networks",
     "A verified effect plus a protein-network association (M1). The network shows the proteins can interact - not that "
     "the agent acts through them."),
    ("mechanistic_ai", "Mechanistic hypotheses proposed by the AI (not established)",
     "One or more steps are not in any verified source. Ideas to test, not findings."),
]


def _trial_of(matrix: Optional[Dict[str, Any]]) -> Dict[str, str]:
    """ref -> trial group (T1 ...) for publications that report the same trial."""
    return {r["ref"]: r["trial_group"] for r in (matrix or {}).get("rows") or [] if r.get("trial_group")}


def _claim_ref(matrix: Optional[Dict[str, Any]]) -> Dict[int, str]:
    return {cid: r["ref"] for r in (matrix or {}).get("rows") or [] for f in r.get("findings") or []
            for cid in f.get("claim_ids") or []}


def design_warnings(h: Dict[str, Any]) -> List[str]:
    """What is wrong with the EXPERIMENT a hypothesis proposes, judged separately from the hypothesis itself.

    A concentration far above anything a patient reaches tests a different question; a plan with no stated controls
    or replication cannot support its own conclusion. These are design faults, not reasons to doubt the biology.
    """
    out = []
    lab = h.get("virtual_lab") or {}
    for c in lab.get("checks") or []:
        r = c.get("result") or {}
        if str(c.get("check", "")).startswith("dose") and r.get("fold_above_plasma"):
            fold = r["fold_above_plasma"]
            if fold >= 10:
                conc = r.get("in_vitro_um")
                shown = f"{conc / 1000:g} mM" if conc and conc >= 1000 else f"{conc:g} µM"
                out.append(f"the proposed concentration ({shown}) is **{fold:g} times** the highest level reached in "
                           f"a patient's blood (about {r.get('human_cmax_um')} µM) - at that concentration the "
                           f"experiment tests a different question from the clinical one")
        if str(c.get("check", "")).startswith("statistical") and r.get("n_per_group"):
            pw = r.get("power_medium_effect")
            if pw is not None and pw < 0.8:
                out.append(f"n = {r['n_per_group']} per group gives only {pw:.0%} power for a medium effect "
                           f"(about {r.get('n_needed_for_medium_effect')} per group would be needed)")
    exp = h.get("next_experiment") or {}
    if isinstance(exp, dict):
        if not str(exp.get("controls") or "").strip():
            out.append("no control group is stated, so the result could not be attributed to the treatment")
        if not re.search(r"\b(?:replicate|replication|independent experiments?|n\s*=|per group|per arm|triplicate)\b",
                         " ".join(str(v) for v in exp.values()), re.I):
            out.append("no replication or sample size is stated, so the experiment's own precision is unknown")
    return out


def hypotheses_markdown(hyps: List[Dict[str, Any]], matrix: Optional[Dict[str, Any]] = None) -> str:
    if not hyps:
        return "_No hypotheses: no verified finding answered the question, and no mechanism is invented without one._\n"
    from reasoning_guards import fix_falsification, hypothesis_kind
    for h in hyps or []:
        fix_falsification(h)
    for h in hyps:
        if not h.get("kind"):
            h["kind"], h["kind_label"] = hypothesis_kind(h)
    out = []
    for kind, head, note in _KIND_HEAD:
        group = [h for h in hyps if h.get("kind") == kind]
        if not group:
            continue
        out.append(f"### {head}\n")
        out.append(f"_{note}_\n")
        for h in group:
            out.extend(_one_hypothesis(h, _trial_of(matrix)))
    return "\n".join(out) + "_Ranked by verified support weighted by evidence level (L1=6 … L6=1; one vote per paper). " \
        "Publications of the same trial count once. The ranking score orders hypotheses; it is not a probability or a confidence " \
        "level. Numbers marked [AI-PROPOSED EXPERIMENTAL PARAMETER] have no verified source._\n"


def _one_hypothesis(h: Dict[str, Any], trial_of: Optional[Dict[str, str]] = None) -> List[str]:
    out: List[str] = []
    if True:
        sup = h.get("support") or {}
        refs = ", ".join(f"{s['ref']} ({s['level']})" for s in sup.get("supporting") or []) or "none"
        out.append(f"#### {h.get('title') or h.get('statement')}\n")
        out.append(f"- **Statement:** {h.get('statement')}")
        out.append(f"- **Type:** {h.get('kind_label') or '—'}")
        sup_refs = [x["ref"] for x in sup.get("supporting") or []]
        trials = {(trial_of or {}).get(r, r) for r in sup_refs}
        out.append(f"- **Evidence status:** {h.get('evidence_status') or '—'} · ranking score {sup.get('score', 0)} "
                   f"from {sup.get('n_papers', 0)} paper(s){f' = {len(trials)} independent trial(s)' if len(trials) != len(sup_refs) else ''}: "
                   f"{refs} _(an internal ranking of how much verified evidence the hypothesis rests on - not a measure "
                   f"of scientific confidence)_")
        out.append(f"- **Derivation:** {h.get('derivation') or '—'}")
        if h.get("merged_from"):
            out.append("- **Merged overlapping hypotheses:** " + " | ".join(_c(m, 120) for m in h["merged_from"]))
        ms = h.get("mechanism_status") or {}
        out.append(f"- **Mechanism ({ms.get('status', 'status unknown')}):** {h.get('mechanism') or '—'}"
                   + (f" _({ms.get('detail')})_" if ms.get("detail") else ""))
        sc = h.get("scope") or {}
        if sc:
            out.append(f"- **Scope:** subtype(s) {', '.join(sc.get('subtypes') or ['not specified'])}"
                       + (f" · {sc['note']}" if sc.get("note") else ""))
        ne = h.get("next_experiment")
        if ne:
            out.append(f"- **Proposed experiment:** {experiment_text(ne)}")
        lab = h.get("in_silico") or {}
        if lab and not lab.get("checks"):
            out.append("- **Virtual lab (in silico):** no check applicable - the mechanism is not written as a signed chain "
                       "(A -> B -| C) and no dose or sample size is stated.")
        if lab.get("checks"):
            parts = []
            for c in lab["checks"]:
                r = c.get("result") or {}
                if c["check"].startswith("mechanism"):
                    parts.append(f"mechanism signs predict {c.get('predicted_direction') or '?'} "
                                 f"(hypothesis says {c.get('stated_direction') or 'not stated'}) - {c['verdict']}")
                elif c["check"].startswith("dose"):
                    parts.append(f"dose {r.get('in_vitro_um')} µM is {r.get('fold_above_plasma')}x the human plasma level "
                                 f"({r.get('verdict')})")
                elif c["check"].startswith("statistical"):
                    parts.append(f"n = {r.get('n_per_group')} per group gives {r.get('power_medium_effect', 0):.0%} power for a "
                                 f"medium effect (needs {r.get('n_needed_for_medium_effect')})")
            out.append(f"- **Virtual lab (in silico, not evidence):** {lab.get('verdict')} - " + "; ".join(parts))
        _edges = (h.get("mechanism_status") or {}).get("edges") or []
        if len(_edges) > 1:
            out.append("- **Each step of the mechanism, judged on its own:**")
            out += [f"    - {'✓ observed in a cited source' if e['status'] == 'observed in source' else '✗ UNVERIFIED'}: "
                    f"{_c(e['step'], 110)} — _{_c(e['why'], 150)}_" for e in _edges]
        if h.get("ai_assumptions"):
            out.append("- > **AI-proposed design assumptions - NOT derived from the literature:** "
                       + "; ".join(h["ai_assumptions"]))
        # A plausible biological idea does not make its PROPOSED EXPERIMENT sound. The design is judged separately,
        # and its problems are stated plainly rather than left inside the in-silico sentence.
        dw = design_warnings(h)
        if dw:
            out.append("- > **Problems with the proposed experiment (the design, not the idea):** "
                       + " · ".join(dw))
        preds = [p for p in (h.get("predictions") or []) if isinstance(p, str)]
        fals = [p for p in (h.get("falsification_conditions") or []) if isinstance(p, str)]
        if preds:
            out.append("- **Predictions:** " + " · ".join(preds))
        if fals:
            out.append("- **Falsified if:** " + " · ".join(fals))
        out.append("")
    return out


def _registry(syn: Dict[str, Any], matrix: Dict[str, Any], prisma: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    reg = (syn.get("report_validation") or {}).get("registry")
    if reg:
        return reg
    from report_validator import study_registry
    return study_registry(matrix, prisma)


def validation_markdown(syn: Dict[str, Any]) -> str:
    """Every pre-publication check with its own status - passed / FAILED / not applicable - never one blanket line."""
    val = syn.get("report_validation") or {}
    checks = val.get("checks") or []
    if not checks:
        return ""
    mark = {"passed": "passed", "failed": "**FAILED**", "skipped": "not applicable", "noted": "noted (disclosure)"}
    out = ["## Pre-publication validation (every check)\n", f"_{val.get('tally') or val.get('summary') or ''}_\n"]
    if val.get("layers"):
        out.append(f"> **What these checks do and do not show:** {val['layers']}\n")
    out += ["| Check | Kind | Severity | Result | Detail |", "|---|---|---|---|---|"]
    kind = {"technical": "technical consistency", "evidence": "evidence vs source"}
    for c in checks:
        st = c.get("status") or ("passed" if c.get("ok") else "failed")
        out.append(f"| {c.get('check')} | {kind.get(c.get('layer') or 'technical', c.get('layer'))} | {c.get('severity')} | "
                   f"{mark.get(st, st)} | {_c(str(c.get('detail') or ''), 220)} |")
    out.append("\n_Audit flags are a different thing: most of them are claims the audit REJECTED (quarantined) - wrong "
               "direction, wrong exposure or disease, a quote that does not support the claim. Rejected claims never reach "
               "the answer; they are listed below so every exclusion can be checked._\n")
    return "\n".join(out) + "\n"


def uncertainty_decomposition(syn: Dict[str, Any], matrix: Dict[str, Any]) -> List[str]:
    """Biological, statistical, coverage and extraction uncertainty - different problems with different fixes."""
    rows = [r for r in matrix.get("rows") or [] if r.get("evidence_type") != "review_meta"]
    if not rows:
        return []
    out = []
    subs = sorted({s for r in rows for s in (r.get("disease_subtypes") or []) if s != "not specified"})
    trend = [r["ref"] for r in rows for f in r.get("findings") or [] if f.get("trend")]
    nulls = [r["ref"] for r in rows for f in r.get("findings") or [] if f.get("direction") == "no_effect"]
    out.append("**Biological** (the effect may differ by population / condition): "
               + (f"findings come from different subtypes ({', '.join(subs)})" if len(subs) > 1 else "no subtype heterogeneity recorded")
               + "; subgroup effects are candidate explanations, not established ones.")
    out.append("**Statistical** (imprecise estimates): "
               + (f"{len(set(nulls))} study(ies) report no statistically significant difference ({', '.join(sorted(set(nulls), key=ref_key))})"
                  if nulls else "no null result recorded")
               + (f"; non-significant numerical trends in {', '.join(sorted(set(trend)))} are shown but never counted as effects"
                  if trend else "") + ".")
    b = (syn.get("budget_limitation") or {})
    reg = ((syn.get("report_validation") or {}).get("registry") or {})
    out.append("**Coverage** (relevant studies not yet assessed): "
               + (f"{reg.get('deep_read')} of {reg.get('eligible')} eligible records were read" if reg.get("eligible") else
                  (b.get("statement") or "coverage not recorded"))
               + "; unread evidence types are listed under 'What was and was not examined'.")
    ab = [r["ref"] for r in rows if r.get("text_basis") in (None, "abstract")]
    q = (matrix.get("totals") or {}).get("quarantined_claims")
    out.append("**Extraction** (the text read may not contain enough to verify a claim): "
               + (f"{len(ab)} of {len(rows)} studies were read from the abstract only" if ab else "every study was read in full text")
               + (f"; {q} extracted claim(s) failed verification and were excluded" if q else "") + ".")
    return out


def reliability_markdown(syn: Dict[str, Any], matrix: Dict[str, Any]) -> str:
    """Shown at the very top: the report must never look more certain than its evidence."""
    rows = matrix.get("rows") or []
    val = syn.get("report_validation") or {}
    lines = ["> **Reliability & limitations**", ">"]
    if val:
        lines.append(f"> - Pre-publication validation: {val.get('summary')}")
        if val.get("layers"):
            lines.append(f"> - {val['layers']}")
    unc = uncertainty_decomposition(syn, matrix)
    if unc:
        lines.append("> - **Where the uncertainty comes from** (four different problems, reported separately):")
        lines.extend(f">   - {u}" for u in unc)
        if (val.get("registry") or {}).get("statement"):
            lines.append(f"> - Study registry: {val['registry']['statement']}")
    b = syn.get("budget_limitation") or {}
    if b.get("statement"):
        lines.append(f"> - {b['statement']}")
    ab = [r["ref"] for r in rows if r.get("text_basis") in (None, "abstract")]
    if ab:
        lines.append(f"> - Abstract-only evidence (methods, dose, statistics and adverse effects not checkable): {', '.join(ab)}")
    if not any(r.get("bucket") == "human" for r in rows):
        lines.append("> - No human study was deep-read: no human efficacy claim is made.")
    off = matrix.get("off_question") or []
    if off:
        lines.append(f"> - {len(off)} retrieved paper(s) concern a different disease / organ system and were excluded from the "
                     f"evidence: " + ", ".join(x["ref"] for x in off))
    t = matrix.get("totals") or {}
    if t.get("verified_claims_secondary"):
        lines.append(f"> - {t['verified_claims_secondary']} verified claim(s) are results reported by reviews (secondary literature), "
                     f"not experiments BioSynthetica verified.")
    fr = matrix.get("question_frame") or {}
    if fr.get("intent") == "etiology" and not any(f.get("evidence_role") == "disease_risk" for r in rows for f in r.get("findings") or []):
        lines.append(f"> - No deep-read study measured {fr.get('disease') or 'disease'} occurrence or risk: mechanisms and intermediate "
                     f"effects are reported, but no cause is established.")
    if syn.get("blocked_statements"):
        lines.append(f"> - {len(syn['blocked_statements'])} generated statement(s) were withheld or rewritten by evidence rules (see below).")
    cap = syn.get("certainty_cap") or {}
    if cap.get("reasons"):
        lines.append(f"> - Certainty ceiling {cap.get('max', '').upper()}: " + "; ".join(cap["reasons"]))
    return "\n".join(lines) + "\n"


def availability_markdown(syn: Dict[str, Any]) -> str:
    av = syn.get("evidence_availability") or {}
    if not av:
        return "_Not assessed._\n"
    out = [f"_{av.get('search', '')}. {av.get('note', '')}_\n", "| Evidence type | State in this search |", "|---|---|"]
    out += [f"| {v['label']} | {v['state']} |" for v in (av.get("types") or {}).values()]
    return "\n".join(out) + "\n\n_\"Not found in the searched sources\" is absence of evidence in this search - not evidence that such studies do not exist._\n"


def definitions_markdown(syn: Dict[str, Any], matrix: Dict[str, Any]) -> str:
    out = []
    cd = syn.get("cure_definition")
    if cd:
        out.append(f"**Operational definition of cure (fixed before synthesis):** {cd['definition']}")
        out.append("**Outcome ladder (not interchangeable):** " + " < ".join(cd["ladder"]) + "\n")
        out.append("| Ref | Strongest reported outcome | Follow-up | Meets cure definition |\n|---|---|---|---|")
        from clinical_context import meets_cure
        for r in matrix.get("rows") or []:
            bo = r.get("best_outcome") or {}
            fu = r.get("follow_up_days")
            _clinical = (r.get("bucket") in ("human", "human_clinical", "human_observational")
                         or r.get("evidence_type") in ("human_clinical", "human_observational", "case_report"))
            if not _clinical:
                fu_txt = "not applicable (not a clinical follow-up)"
                fu = None
            else:
                fu_txt = f"{fu / 30.44:.1f} months" if fu else "not reported"
            out.append(f"| {r['ref']} | {bo.get('label', 'not stated')} | {fu_txt} | "
                       f"{'yes' if meets_cure(bo.get('level'), fu) else 'no'} |")
        out.append("")
    dd = syn.get("disease_definition")
    if dd:
        out.append(f"**Disease definition check:** {dd['statement']}\n")
        out.append("| Ref | What the study actually studied | Species | Induced model |\n|---|---|---|---|")
        out += [f"| {p['ref']} | {_c(p['studied'], 80)} | {p.get('species') or '—'} | {'yes' if p['induced'] else 'no'} |" for p in dd["per_study"]]
        out.append("")
    rows = matrix.get("rows") or []
    if rows:
        out.append("| Ref | Model (normalised) | Translational relation to the question | Intervention (normalised) | Comparator | n |")
        out.append("|---|---|---|---|---|---|")
        for r in rows:
            mp, ip = r.get("model_profile") or {}, r.get("intervention_profile") or {}
            iv = ip.get("canonical") or "—"
            if ip.get("combination_components"):
                iv = "regimen: " + " + ".join(ip["combination_components"])
            out.append(f"| {r['ref']} | {mp.get('system', '—')} · {mp.get('species', '—')} | distance {mp.get('translational_distance', '?')}: "
                       f"{_c(mp.get('relation', '—'), 90)} | {_c(iv, 60)} ({ip.get('type', '?')}; route {ip.get('route', '?')}; dose {ip.get('dose', '?')}) | "
                       f"{r.get('comparator', 'not reported')} | {r.get('n_display', 'not reported')} |")
    return "\n".join(out) + "\n" if out else "_No definitions needed._\n"


def verification_markdown(matrix: Dict[str, Any]) -> str:
    rows = matrix.get("rows") or []
    if not rows:
        return "_No studies._\n"
    out = ["| Ref | Verification | Claim statuses | Why not (fully) verified |", "|---|---|---|---|"]
    for r in rows:
        v = r.get("verification") or {}
        st = "; ".join(f"{k} ×{n}" for k, n in (r.get("claim_statuses") or {}).items()) or "—"
        out.append(f"| {r['ref']} | {v.get('status', '—')} | {_c(st, 120)} | {_c('; '.join(v.get('reasons') or []) or '—', 200)} |")
    return "\n".join(out) + "\n"


def identity_markdown(matrix: Dict[str, Any]) -> str:
    """Evidence identity: which studies answer the question, in which role, for which subtype - and which
    retrieved papers were excluded because they concern another disease / organ system."""
    rows = matrix.get("rows") or []
    fr = matrix.get("question_frame") or {}
    out = []
    if fr:
        out.append(f"**Question frame:** intent = {fr.get('intent')} · disease = {fr.get('disease') or 'none named'}"
                   + (f" · named exposure/intervention = {', '.join(fr.get('exposures') or [])}" if fr.get("exposures") else "")
                   + (f" · endpoints = {', '.join(e.replace('_', ' ') for e in fr.get('endpoints') or [])}" if fr.get("endpoints") else "") + "\n")
    out.append("_Hard rule: a claim enters the evidence pool only if its disease, model (cell/tissue system), exposure and "
               "outcome match the question. Textual grounding (the quote occurs in the paper) is NOT scientific relevance._\n")
    if rows:
        out.append("| Ref | Answers the question? | Evidence roles | Disease subtype(s) | Relevance |\n|---|---|---|---|---|")
        for r in rows:
            roles = ", ".join(f"{k.replace('_', ' ')} ×{v}" for k, v in (r.get("evidence_roles") or {}).items()) or "—"
            out.append(f"| {r['ref']} | {'yes' if r.get('answers_question') else 'no - context only' if r.get('n_findings') else 'no verified finding'} | "
                       f"{_c(roles, 80)} | {_c(', '.join(r.get('disease_subtypes') or []) or '—', 60)} | "
                       f"{_c((r.get('paper_relevance') or {}).get('detail'), 80)} |")
        out.append("")
    off = matrix.get("off_question") or []
    if off:
        out.append(f"**Excluded as off-question ({len(off)} paper(s) - not counted in any evidence bucket):**\n")
        out.append("| Ref | Paper | Design | Why excluded | Claims excluded |\n|---|---|---|---|---|")
        out += [f"| {x['ref']} | {_c(x['title'], 90)} | {x.get('evidence_type_label')} | {_c(x['reason'], 80)} | {x['claims_excluded']} |" for x in off]
        out.append("")
    t = matrix.get("totals") or {}
    if t.get("off_question_claims"):
        out.append(f"_{t['off_question_claims']} claim(s) from included papers were excluded by the relevance gate (listed under "
                   f"Quarantined Claims in the detailed report)._\n")
    out.append("_Roles: **disease occurrence/risk** = measures the disease itself; **intermediate** = e.g. chromosome damage - the link "
               "to the disease is not shown; **mechanistic** = a process in disease cells (ROS, apoptosis, expression) - not a "
               "demonstrated cause; **secondary report** = a review reporting someone else's experiment._\n")
    return "\n".join(out) + "\n"


def subtype_markdown(matrix: Dict[str, Any]) -> str:
    sc = matrix.get("subtype_coverage") or {}
    if not sc or not sc.get("coverage"):
        return "_Not applicable (no disease subtypes for this question)._\n"
    out = [f"{sc['statement']}\n" if sc.get("statement") else "",
           "| Subtype | Studies |\n|---|---|"]
    out += [f"| {k} | {', '.join(v)} |" for k, v in sc["coverage"].items()]
    missing = [s for s in sc.get("known_subtypes") or [] if s not in sc["coverage"]]
    if missing:
        out.append(f"\n_No deep-read study addressed: {', '.join(missing)} (absence in this sample, not evidence of absence)._")
    return "\n".join(out) + "\n"


def qc_markdown(matrix: Dict[str, Any]) -> str:
    rows = [r for r in matrix.get("rows") or [] if r.get("qc")]
    if not rows:
        return "_No QC data._\n"
    keys = ("experiment_origin", "statistical_information", "statistical_power", "dose_validity", "sample_size", "comparator",
            "follow_up", "text_basis")
    out = ["| Ref | Experiment origin | Statistics reported | Statistical power | Dose validity | n | Comparator | Follow-up | Text |",
           "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        out.append(f"| {r['ref']} | " + " | ".join(_c((r["qc"].get(k) or {}).get("status", "—"), 40) for k in keys) + " |")
    return "\n".join(out) + ("\n\n_Nothing is marked as validated unless the data to check it exists: power needs sample size, effect size, "
                             "variance and alpha (a p-value alone does not establish power); dose validity is 'unknown' when no dose is reported._\n")


def gaps_markdown(gaps: List[Dict[str, Any]]) -> str:
    if not gaps:
        return "_No research gaps._\n"
    out = []
    for i, g in enumerate(gaps, 1):
        out.append(f"{i}. **{g.get('description')}**" + (f"  \n   _{g['evidence_status']}_" if g.get("evidence_status") else "")
                   + (f"  \n   Supporting observation: {g['supporting_observation']}" if g.get("supporting_observation") else "")
                   + (f"  \n   ⚠ {g['chain_note']}" if g.get("chain_note") else "")
                   + (f"  \n   Dose check: {g['dose_note']}" if g.get("dose_note") else "")
                   + (f"  \n   (rewritten from: \"{_c(g['rewritten_from'], 140)}\")" if g.get("rewritten_from") else ""))
    return "\n".join(out) + "\n"


def blocked_markdown(syn: Dict[str, Any]) -> str:
    items = syn.get("blocked_statements") or []
    if not items:
        return "_None - every generated statement passed the evidence rules._\n"
    out = []
    for b in items:
        out.append(f"- **{b['field'].replace('_', ' ')}** - rule: {'; '.join(b['rules'])}  \n  original: ~~{_c(b['original'], 220)}~~"
                   + (f"  \n  published instead: {b['replacement']}" if b.get("replacement") else "  \n  published instead: (withheld)"))
    ann = syn.get("scope_annotations") or []
    if ann:
        out.append(f"\n_{len(ann)} sentence(s) received a subtype-scope note (heterogeneous disease)._")
    return "\n".join(out) + "\n"


def pathways_markdown(chains: List[Dict[str, Any]]) -> str:
    if not chains:
        return "_No cross-study chains._\n"
    return "\n".join(f"- [{c.get('evidence_level', 'M1 candidate (cross-study inference)')}] {c.get('pathway')}"
                     for c in chains[:10]) + \
        "\n\n_Chains link findings from DIFFERENT papers; no single experiment demonstrated the full sequence, " \
        "so they are candidate mechanisms, not causal evidence._\n"


def structures_markdown(structs: Optional[List[Dict[str, Any]]]) -> str:
    """3D models of the run's proteins (AlphaFold DB): confidence and the largest pockets. Structural context only."""
    if not structs:
        return ""
    out = ["## 3D Protein Structures (in silico context, not evidence)\n",
           "_Predicted models from the AlphaFold Protein Structure Database for proteins named in the evidence. pLDDT is the "
           "model's own confidence (0-100). Pockets are geometric (where a molecule could fit), not predicted binding "
           "strength; they are not evidence for the research question._\n"]
    for st in structs:
        if st.get("error"):
            out.append(f"- **{st.get('gene')}:** no model ({_c(st['error'], 120)})")
            continue
        a = st.get("analysis") or {}
        conf = a.get("confidence") or {}
        ss = a.get("secondary_structure") or {}
        pk = "; ".join(f"pocket {p.get('rank')}: {p.get('volume_A3')} A^3, lined by {', '.join(p.get('lining_residues', [])[:6])}"
                       + (f" (pocket pLDDT {p['mean_plddt']}: {p.get('confidence_note')})" if p.get("mean_plddt") is not None else "")
                       for p in st.get("top_pockets") or []) or "none found"
        out.append(f"- **{st.get('gene')}** ([AlphaFold {st.get('id')}]({st.get('url')})): {a.get('n_residues')} residues, "
                   f"mean pLDDT {conf.get('mean_plddt')} ({conf.get('reading')}); helix {ss.get('helix', 0):.0%}, "
                   f"strand {ss.get('strand', 0):.0%}. Pockets: {pk}."
                   + (f" _Why shown: {st['why']}._" if st.get("why") else "")
                   + " _Level S - structural hypothesis only: a pocket is where a molecule COULD fit; it does not show that "
                     "any compound binds or acts here._")
    return "\n".join(out) + "\n"


EVIDENCE_HIERARCHY = (
    "| Level | Kind of evidence | What it can support |\n|---|---|---|\n"
    "| 1 | Direct evidence (a study measured the effect) | the effect, in the system that was studied |\n"
    "| 2 | Mechanistic evidence (perturbation: changing X changed Y - M3 / M4) | that X can cause Y in that model |\n"
    "| 3 | Molecular association (co-change M2; STRING database link M1) | that X and Y are related - not cause |\n"
    "| 4 | Structural hypothesis (AlphaFold model, predicted pocket - S) | something worth testing - nothing more |\n")


def graphs_markdown(matrix: Dict[str, Any], synthesis: Dict[str, Any], string_net: Optional[Dict[str, Any]],
                    structures: Optional[List[Dict[str, Any]]]) -> str:
    """Three separate graphs so 'A is associated with B' is never read as 'A causes B':
    (1) evidence graph - verified study results only; (2) mechanistic candidate graph - every arrow with its OWN level;
    (3) structural investigation graph - computational only."""
    arrow = {"increase": "↑", "decrease": "↓", "no_effect": "∅"}
    ev: Dict[tuple, List[str]] = {}
    for r in matrix.get("rows") or []:
        if r.get("evidence_type") == "review_meta":
            continue
        for f in r.get("findings") or []:
            if f.get("evidence_role") == "secondary_report" or not f.get("intervention") or not f.get("endpoint"):
                continue
            if f.get("evidence_role") == "combination" and "combination" not in str(f["intervention"]).lower():
                iv_ = f"{f['intervention']} (combination)"
            else:
                iv_ = str(f["intervention"])
            ep_ = str(f["endpoint"]).replace("_", " ").replace("other:", "")
            if f.get("target"):
                ep_ = f"{f['target']} {ep_}"          # 'METTL3 expression', never a bare 'expression'
            k = (iv_.lower(), ep_, f.get("direction"))
            ev.setdefault(k, []).append(((f.get('model_type') or '?').replace('_', ' '), r['ref']))
    out = ["### Evidence hierarchy\n", EVIDENCE_HIERARCHY,
           "### Graph 1 - Evidence graph (verified study results only)\n"]
    if ev:
        for (iv, ep, d), where in sorted(ev.items(), key=lambda x: -len(x[1]))[:12]:
            by_model: Dict[str, List[str]] = {}
            for mdl, ref in where:
                if ref not in by_model.setdefault(mdl, []):
                    by_model[mdl].append(ref)
            # one label per system: 'cell culture: S2, S3, S9; animal: S10'
            out.append(f"- {iv} → {arrow.get(d, '?')} {ep}  [{'; '.join(f'{m}: ' + ', '.join(rs) for m, rs in by_model.items())}]")
    else:
        out.append("_No verified primary finding._")
    out.append("\n### Graph 2 - Mechanistic candidate graph (each arrow has its own level)\n")
    edges = [e for x in (synthesis.get("mechanistic_chain_levels") or []) for e in (x.get("edges") or [])]
    ev_genes = set((string_net or {}).get("query_genes") or [])
    st_edges = [e for e in ((string_net or {}).get("edges") or []) if e.get("a") in ev_genes or e.get("b") in ev_genes]
    if edges:
        out.append("| From | To | Level | Basis |\n|---|---|---|---|")
        out.extend(f"| {e['from']} | {e['to']} | {e['level']} | {_c(e['basis'], 140)} |" for e in edges)
        if synthesis.get("mechanistic_chain_summary"):
            out.append(f"\n**Reading:** {synthesis['mechanistic_chain_summary']}\n")
    if st_edges:
        out.append("\nProtein-network links touching proteins of the verified evidence (M1 - association only):")
        out.extend(f"- {e['a']} — {e['b']} (STRING {e['score']:.3f}, {e.get('top_evidence') or 'combined'})" for e in st_edges[:8])
        out.append("\n_What is NOT established: that the agent acts through any of these proteins. A network link means the "
                   "proteins can interact; it is not a step of the observed effect._")
    if not edges and not st_edges:
        out.append("_No mechanistic link was proposed or retrieved._")
    out.append("\n### Graph 3 - Structural investigation graph (computational only)\n")
    shown = [st for st in (structures or []) if not st.get("error")]
    if shown:
        for st in shown:
            for p in (st.get("top_pockets") or [])[:2]:
                out.append(f"- {st['gene']} → pocket {p.get('rank')} ({p.get('volume_A3')} A^3"
                           + (f", pocket pLDDT {p['mean_plddt']}" if p.get("mean_plddt") is not None else "")
                           + ") → open question: does any compound bind here? (untested - level S)")
    else:
        out.append("_No 3D model in this run._")
    return "\n".join(out) + "\n"


def string_markdown(net: Optional[Dict[str, Any]]) -> str:
    lv = (net or {}).get("interpretation_levels") or {}
    levels = ("\n| Level | What this report can say |\n|---|---|\n" + "\n".join(f"| {k.replace('_', ' ')} | {v} |" for k, v in lv.items()) + "\n") if lv else ""
    if not net or not net.get("edges"):
        genes = ", ".join((net or {}).get("query_genes") or []) or "none resolved"
        return f"_No physical interactions retrieved from STRING-db (genes queried: {genes}). {(net or {}).get('evidence_link_note', '')}_\n" + levels
    lines = ["| Protein A | Protein B | STRING score | Strongest evidence channel |", "|---|---|---|---|"]
    for e in net["edges"][:25]:
        lines.append(f"| {e['a']} | {e['b']} | {e['score']:.3f} | {e.get('top_evidence') or '—'} |")
    link = f"\n[Open interactive network on string-db.org]({net['link']})\n" if net.get("link") else ""
    link += ("\n> **Interpretation:** STRING scores indicate that two proteins are physically or functionally "
             "associated. They are not evidence of a directional causal cascade in vivo, and not proof that the "
             "intervention acts through this network in the organism studied.\n")
    return (f"**Network associations:** {net.get('evidence_link_note', '')}.\n\n"
            f"Species taxon {net['species']} · query genes: {', '.join(net['query_genes'])} · "
            f"mode: {net['mode']}\n\n" + "\n".join(lines) + "\n" + link + levels)


def level_answer_obj(question: str, matrix: Dict[str, Any]) -> Dict[str, Any]:
    from report_generator import level_answer
    claims = []
    for r in matrix.get("rows") or []:
        if (r.get("paper_relevance") or {}).get("ok") is False:
            continue
        for f in r.get("findings") or []:
            claims.append({"id": (f.get("claim_ids") or ["?"])[0], "statement": f.get("statement"),
                           "paper_id": r.get("paper_id") or r.get("ref"), "trial": r.get("trial_group"),
                           "claim_model": f.get("model_type"), "canonical_direction": f.get("direction"),
                           "canonical_outcome": f.get("endpoint"),
                           "semantic_audit": {"evidence_role": "secondary_report" if r.get("secondary_report")
                                              else "mechanistic" if f.get("answers_question") is False
                                              else f.get("evidence_role") or "direct"}})
    if not claims:
        return {"text": "", "verdict": "Not answered by the papers read.", "discordance": ""}
    out = level_answer(question, claims)
    note = trial_count_note(matrix, {c["id"] for c in claims})
    if note and out.get("text"):
        out["text"] = out["text"].replace("This summarises the papers read", note + " This summarises the papers read", 1)
    pooled = pooled_sentence(matrix, question)
    if pooled and out.get("text"):
        # the answer stops at counting studies for and against: the pooled estimate says how large the effect is
        out["text"] = out["text"].replace("This summarises the papers read", pooled + " This summarises the papers read", 1)
        out["pooled"] = pooled
    return out


def pooled_sentence(matrix: Dict[str, Any], question: str = "") -> str:
    """'Pooled across 3 independent human studies: ratio 0.80 (95% CI 0.68 to 0.92) ...' for the asked endpoint.

    Only the human pooling on the question's own endpoint goes into the answer: that is the quantity the question
    asks about. The other poolings stay in the evidence layer.
    """
    from meta_analysis import analyse
    from semantic_audit import FAMILY_LABEL
    rows = matrix.get("rows") or []
    want = set((matrix.get("question_frame") or {}).get("endpoints") or [])
    res = [r for r in analyse(rows) if r["bucket"] == "human"]
    if want:
        on_q = [r for r in res if r["endpoint"] in want]
        res = on_q or []
    if not res:
        return ""
    r = max(res, key=lambda x: x["k"])
    label = FAMILY_LABEL.get(r["endpoint"], str(r["endpoint"] or "?").replace("other:", "").replace("_", " "))
    word = "ratio" if r["kind"] == "ratio" else "difference"
    body = (f"**Pooled across {r['k']} independent human studies** (weighted by each study's precision, not one vote "
            f"each): {label} {word} {r['estimate']:.2f} (95% CI {r['lo']:.2f} to {r['hi']:.2f}), "
            f"{'statistically significant' if r['significant'] else 'not statistically significant'}; "
            f"heterogeneity I-squared {r['i2']:.0f}%")
    if r["i2"] >= 75:
        body += " - the studies disagree too much for this pooled value to be read as one answer"
    return body + "."


def trial_count_note(matrix: Dict[str, Any], claim_ids: set) -> str:
    """'S5 and S8 report the same trial (T2): their findings are not independent' - for the counted findings only."""
    by_trial: Dict[str, List[str]] = {}
    for r in matrix.get("rows") or []:
        if r.get("trial_group") and any(set(f.get("claim_ids") or []) & claim_ids for f in r.get("findings") or []):
            by_trial.setdefault(r["trial_group"], []).append(r["ref"])
    groups = {t: refs for t, refs in by_trial.items() if len(refs) > 1}
    if not groups:
        return ""
    return ("**Independent trials:** " + "; ".join(f"{', '.join(refs)} are publications of one trial ({t})"
                                                   for t, refs in groups.items())
            + " - their findings are not independent confirmations of each other.")


def level_answer_markdown(question: str, matrix: Dict[str, Any]) -> str:
    """The answer level by level (people / animals / cells), from the verified findings only - so a null clinical trial
    is never hidden behind positive cell or mouse results, and a mouse result is never read as an effect in people."""
    claims = []
    for r in matrix.get("rows") or []:
        if (r.get("paper_relevance") or {}).get("ok") is False:
            continue
        for f in r.get("findings") or []:
            claims.append({"id": (f.get("claim_ids") or ["?"])[0], "statement": f.get("statement"),
                           "paper_id": r.get("paper_id") or r.get("ref"), "trial": r.get("trial_group"),
                           "claim_model": f.get("model_type"), "canonical_direction": f.get("direction"),
                           "canonical_outcome": f.get("endpoint"),
                           "semantic_audit": {"evidence_role": "secondary_report" if r.get("secondary_report")
                                              else "mechanistic" if f.get("answers_question") is False
                                              else f.get("evidence_role") or "direct"}})
    if not claims:
        return ""
    ans = level_answer_obj(question, matrix)             # same text as Layer 1 (incl. the independent-trial note)
    return (f"**Evidence by level:** {ans['text']}\n" + (f"\n**Across levels:** {ans['discordance']}\n" if ans["discordance"] else ""))


def compose(question: str, synthesis: Dict[str, Any], matrix: Dict[str, Any],
            prisma: Optional[Dict[str, Any]], string_net: Optional[Dict[str, Any]],
            original_report: str, output_path: Optional[str] = None,
            hypotheses: Optional[List[Dict[str, Any]]] = None,
            contradictions: Optional[List[Dict[str, Any]]] = None,
            pathways: Optional[List[Dict[str, Any]]] = None,
            gaps: Optional[List[Dict[str, Any]]] = None,
            structures: Optional[List[Dict[str, Any]]] = None) -> str:
    counts = matrix.get("bucket_counts") or {}
    totals = matrix.get("totals") or {}
    bucket_line = " · ".join(f"{BUCKET_LABELS[b]}: **{n}**" for b, n in counts.items())
    model = (f"{synthesis.get('provider')} / {synthesis.get('model')}" if synthesis.get("model")
             else "deterministic (no AI key)")
    norm = matrix.get("question_normalisation") or synthesis.get("question_normalisation") or {}
    if not norm.get("corrections"):
        from evidence_identity import normalize_question
        norm = normalize_question(question)            # a misspelled question is always shown with its correction
    if norm.get("corrections"):
        question = norm["question"]
    hs = matrix.get("human_split") or {}
    ans = level_answer_obj(question, matrix)
    esc_q = question.replace('<', '&lt;').replace('>', '&gt;')
    # ================================================================ LAYER 1 - the answer (read this first)
    md = [
        "# BioSynthetica Evidence Dossier\n",
        f"**Research question:** {esc_q}\n",
        (f"_{norm['note']} (as typed: \"{str(norm.get('original') or '').replace('<', '&lt;').replace('>', '&gt;')}\")._\n" if norm.get("corrections") else ""),
        "_Three layers: **1 Answer** (this part) · **2 Evidence** (studies, tables, doses, limitations, hypotheses) · "
        "**3 Audit** (every claim, quote, classification and rejection)._\n",
        "# Layer 1 — Answer\n",
        f"**Verdict:** {ans.get('verdict') or '—'}\n",
        (f"{ans['text']}\n" if ans.get("text") else ""),
        (f"**Across levels:** {ans['discordance']}\n" if ans.get("discordance") else ""),
        f"**Bottom line:** {synthesis.get('bottom_line') or '—'}\n",
        f"**BioSynthetica certainty (internal framework - GRADE-inspired, not a formal GRADE assessment):** "
        f"{str(synthesis.get('certainty') or '—').upper()} — {synthesis.get('certainty_rationale') or ''}\n",
        f"**Scope (study registry):** {_registry(synthesis, matrix, prisma)['statement']}. Conclusions are limited to these "
        f"studies; they are a sample of the literature, not all of it.\n",
        "---\n",
        # ============================================================ LAYER 2 - the evidence
        "# Layer 2 — Evidence\n",
        reliability_markdown(synthesis, matrix),
        "## Executive Summary\n",
        synthesis.get("executive_summary") or "",
        "",
        f"_Synthesis engine: {model}_\n",
        ("_AI task assignment: " + "; ".join(f"{k} → {v}" for k, v in (synthesis.get("ai_assignments") or {}).items()) + "_\n")
        if synthesis.get("ai_assignments") else "",
        "### Evidence by bucket\n",
        f"{bucket_line}\n",
        (f"Human evidence split: clinical **{hs.get('human_clinical', 0)}** · observational / epidemiological "
         f"**{hs.get('human_observational', 0)}** · case reports **{hs.get('human_case_reports', 0)}** "
         f"(off-question papers are not counted in any bucket).\n" if hs else ""),
    ]
    focus = synthesis.get("question_focus")
    if focus:
        md.append(f"_Question focus: {focus}_\n")
    for b, key, label in (("human", "human_evidence", "Human"), ("animal", "animal_evidence", "Animal"),
                          ("in_vitro", "in_vitro_evidence", "In vitro"), ("plant", "plant_evidence", "Plant"),
                          ("microbial", "microbial_evidence", "Microbial")):
        # a bucket is shown when it has studies or is what the question is about
        # (human-health / general questions always show the classic three)
        classic = b in ("human", "animal", "in_vitro") and focus in (None, "human", "general")
        if classic or counts.get(b) or focus == b:
            md.append(f"- **{label}:** {synthesis.get(key) or '—'}")
    md.append(f"- **Translational gap:** {synthesis.get('translational_gap') or '—'}\n")
    for title, key in (("Conflicts explained", "conflicts_explained"), ("Key uncertainties", "key_uncertainties")):
        items = synthesis.get(key) or []
        if items:
            md.append(f"### {title}\n")
            md.extend(f"- {x}" for x in items)
            md.append("")
    md += [
        "## Evidence Matrix (verified evidence only)\n",
        matrix_markdown(matrix, split_by_bucket=True),
        "## Disease Subtype Coverage\n",
        subtype_markdown(matrix),
        "## Dose / Exposure Relevance\n",
        dose_markdown(matrix),
        "## Quality Profile (scored separately)\n",
        quality_markdown(matrix),
        "## Methodological Flags (known traps, checked against public registers)\n",
        method_flags_markdown(matrix, synthesis),
        "## Pooled Effect (meta-analysis of the reported effect sizes)\n",
        meta_markdown(matrix),
        "## Cross-Study Contradictions (context comparator)\n",
        contradictions_markdown(contradictions or [], matrix),
        # Question -> Evidence -> Verified claims -> Gaps -> Replication hypotheses -> Mechanistic hypotheses
        "## Research Gaps (validated against the evidence)\n",
        gaps_markdown(gaps or []),
        "## Hypotheses (replication tests first, then mechanisms)\n",
        hypotheses_markdown(hypotheses or [], matrix),
    ]
    chain_levels = synthesis.get("mechanistic_chain_levels") or []
    chain = synthesis.get("mechanistic_chain") or []
    if chain_levels or chain:
        md.append("## Hypothetical mechanistic pathway (each arrow judged on its own)\n")
        md.append("_A hypothesis, not a demonstrated pathway: physical protein interactions (STRING) show that proteins can "
                  "interact, not that the intervention causes this sequence in the organism studied. "
                  "A step is only as strong as its weakest arrow: causal evidence "
                  "for a downstream arrow (e.g. CDK4 inhibition → Rb) says nothing about whether the agent reaches the "
                  "upstream protein. M1 database association → M2 experimental association → M3 causal evidence in a model "
                  "→ M4 causal evidence in the target organism · L7 not verified._\n")
        if chain_levels:
            md.append("| # | Step | Level (weakest arrow) | Arrows and their evidence |\n|---|---|---|---|")
            md.extend(f"| {i} | {_c(x['step'])} | {x['label']} | {_c(x['basis'], 400)} |" for i, x in enumerate(chain_levels, 1))
        else:
            md.extend(f"{i}. {s_}" for i, s_ in enumerate(chain, 1))
        if synthesis.get("mechanistic_chain_summary"):
            md.append(f"\n**Reading:** {synthesis['mechanistic_chain_summary']}\n")
        md.append("")
    md += [
        "## Three graphs: evidence · mechanism candidates · structure\n",
        graphs_markdown(matrix, synthesis, string_net, structures),
        ("## Physical Protein Interactions (STRING-db)\n" if (string_net or {}).get("evidence_link") == "linked"
         else "## Appendix - Background Protein Associations (STRING-db, not linked to the evidence)\n"),
        string_markdown(string_net),
        structures_markdown(structures),
        ("## Your Experiment (uploaded data) - analysed separately from the literature\n" if synthesis.get("user_experiment") else ""),
        (user_experiment_markdown(synthesis["user_experiment"]) if synthesis.get("user_experiment") else ""),
        "## What was and was not examined (evidence availability)\n",
        availability_markdown(synthesis),
        "## Outcome & Disease Definitions\n",
        definitions_markdown(synthesis, matrix),
        "## PRISMA Flow\n",
        prisma_markdown(prisma, matrix),
        "---\n",
        # ============================================================ LAYER 3 - the audit
        "# Layer 3 — Audit\n",
        "_Everything the engine decided, with the reason: accounting, identity gate, verification, QC, classification, "
        "traceability, withheld statements and the full engine report._\n",
        validation_markdown(synthesis),
        (f"**Evidence accounting:** {totals.get('papers')} papers ({totals.get('primary_studies')} primary studies) · "
         f"{totals.get('experiments')} experiments · **{totals.get('independent_findings')} independent findings** · "
         f"{totals.get('verified_claims')} verified claims ({totals.get('verified_claims_primary', totals.get('verified_claims'))} from "
         f"primary studies, {totals.get('verified_claims_secondary', 0)} reported by reviews = secondary, not independent "
         f"verification) · {totals.get('findings_answering_question', '?')} finding(s) measure the question's outcome, "
         f"{totals.get('findings_context_only', 0)} are mechanistic/context only · {totals.get('quarantined_claims')} quarantined "
         f"claims, of which {totals.get('off_question_claims', 0)} off-question · {totals.get('off_question_papers', 0)} included "
         f"paper(s) excluded as off-question (excluded from all synthesis).\n" if totals else ""),
        "_The synthesis is generated only from verified claims, the frozen study classifications, validated "
        "contradiction comparisons and verified numbers. AI hypotheses are labelled L7; mechanism steps carry M1-M4 levels._\n",
        ("> **Grounding audit:** " + " ".join(synthesis["notes"]) + "\n" if synthesis.get("notes") else ""),
        "## Evidence Identity & Relevance Gate\n",
        identity_markdown(matrix),
        "## Verification Status per Study (why each study is or is not verified)\n",
        verification_markdown(matrix),
        "## Quality-Control Checklist (what could and could NOT be validated)\n",
        qc_markdown(matrix),
        "## Study Classification (frozen) & Evidence Levels\n",
        classification_markdown(matrix),
        "## Traceability: Conclusion → Claim → Experiment → Paper → Source passage\n",
        traceability_markdown(synthesis.get("traceability") or []),
        "## Statements Withheld or Rewritten by Evidence Rules\n",
        blocked_markdown(synthesis),
        "## Cross-Study Pathway Chains\n",
        pathways_markdown(pathways or []),
        "---\n",
        "# Detailed Engine Report\n",
        original_report or "",
    ]
    text = "\n".join(md)
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_text(text, encoding="utf-8")
    return text
