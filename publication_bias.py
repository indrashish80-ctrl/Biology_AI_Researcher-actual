"""
publication_bias.py  (v25)
==========================
Is the published evidence a biased sample? Three checks, each with its limits stated:

    funnel plot            effect (x; log scale for ratios) against standard error (y, inverted) with the pseudo 95%
                           limits around the pooled estimate - drawn as SVG for the report
    Egger's test           regression of the standardised effect (theta / SE) on precision (1 / SE); an intercept far
                           from 0 means small studies report systematically different effects (Egger et al., BMJ 1997).
                           Cochrane advises at least 10 studies: with fewer the test is computed but labelled
                           'underpowered' and never used to change a verdict
    trim-and-fill          Duval & Tweedie's L0 estimator: how many studies seem to be missing on one side, and the
                           pooled estimate with them imputed (a sensitivity analysis, not a correction)
    registry vs published  trials whose registered primary outcome differs from the published one (outcome
                           switching, from methodology_flags) and registered trials with no publication found

    analyse(pooled_result)            -> {'k', 'egger', 'trim_fill', 'funnel_svg', 'notes'}
    registry_section(exec_synthesis)  -> summary of outcome switching / unpublished registered trials
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

BIAS_VERSION = "1.0"
MIN_K_EGGER = 10


def _t_two_sided(x: float, df: int) -> float:
    try:
        from scipy.stats import t as _t
        return float(2 * _t.sf(abs(x), df))
    except Exception:
        return math.erfc(abs(x) / math.sqrt(2))


def egger(theta: List[float], se: List[float]) -> Optional[Dict[str, Any]]:
    pairs = [(t, s) for t, s in zip(theta, se) if math.isfinite(t) and math.isfinite(s) and s > 0]
    theta, se = [p[0] for p in pairs], [p[1] for p in pairs]   # a zero / missing standard error has no precision
    k = len(theta)
    if k < 3:
        return None
    x = [1.0 / s for s in se]               # precision
    y = [t / s for t, s in zip(theta, se)]  # standardised effect
    mx, my = sum(x) / k, sum(y) / k
    sxx = sum((xi - mx) ** 2 for xi in x)
    if sxx <= 0:
        return None
    slope = sum((xi - mx) * (yi - my) for xi, yi in zip(x, y)) / sxx
    intercept = my - slope * mx
    resid = [yi - (intercept + slope * xi) for xi, yi in zip(x, y)]
    s2 = sum(r * r for r in resid) / (k - 2) if k > 2 else 0.0
    se_int = math.sqrt(s2 * (1.0 / k + mx * mx / sxx)) if s2 > 0 else 0.0
    # an exact fit leaves only rounding noise (~1e-16): t would be ~1e15 and p = 0, a false 'asymmetry'. Treat a
    # residual spread below 1e-9 of the data's own spread as zero - the test is not defined then.
    syy = sum((yi - my) ** 2 for yi in y)
    if s2 <= 1e-18 * max(1.0, syy):
        return {"intercept": round(intercept, 4), "se": 0.0, "t": None, "df": k - 2, "p": None, "asymmetry": False,
                "underpowered": k < MIN_K_EGGER,
                "interpretation": "not testable - the points lie exactly on a line (no residual variance)"}
    tval = intercept / se_int if se_int > 0 else 0.0
    p = _t_two_sided(tval, k - 2) if se_int > 0 else 1.0
    return {"intercept": round(intercept, 4), "se": round(se_int, 4), "t": round(tval, 3), "df": k - 2, "p": round(p, 4),
            "asymmetry": p < 0.10, "underpowered": k < MIN_K_EGGER,
            "interpretation": ("small-study effects suggested (p < 0.10)" if p < 0.10 else "no small-study asymmetry detected")
            + (f" - UNDERPOWERED with {k} studies (Cochrane: >= {MIN_K_EGGER})" if k < MIN_K_EGGER else "")}


def _fe(theta, v):
    w = [1 / vi for vi in v]
    return sum(wi * t for wi, t in zip(w, theta)) / sum(w)


def trim_and_fill(theta: List[float], se: List[float], max_iter: int = 20) -> Optional[Dict[str, Any]]:
    """Duval & Tweedie L0 estimator, fixed-effect centre; returns the number imputed and the adjusted estimate."""
    pairs = [(t, s) for t, s in zip(theta, se) if math.isfinite(t) and math.isfinite(s) and s > 0]
    theta, se = [p[0] for p in pairs], [p[1] for p in pairs]
    k = len(theta)
    if k < 3:
        return None
    v = [s * s for s in se]
    mu = _fe(theta, v)
    # which side is missing: the side opposite to the asymmetry of the larger effects
    side = 1 if sum(1 for t in theta if t > mu) >= sum(1 for t in theta if t < mu) else -1
    k0 = 0
    for _ in range(max_iter):
        order = sorted(range(k), key=lambda i: side * (theta[i] - mu))
        trimmed = order[:k - k0] if k0 else order
        mu = _fe([theta[i] for i in trimmed], [v[i] for i in trimmed])
        dev = [theta[i] - mu for i in range(k)]
        ranks = sorted(range(k), key=lambda i: abs(dev[i]))
        rank_of = {i: r + 1 for r, i in enumerate(ranks)}
        tn = sum(rank_of[i] for i in range(k) if side * dev[i] > 0)
        new = max(0, int(round((4 * tn - k * (k + 1)) / (2 * k - 1))))
        if new == k0:
            break
        k0 = min(new, k - 1)
    filled_t = theta + [2 * mu - theta[i] for i in sorted(range(k), key=lambda i: -side * (theta[i] - mu))[:k0]]
    filled_v = v + [v[i] for i in sorted(range(k), key=lambda i: -side * (theta[i] - mu))[:k0]]
    adj = _fe(filled_t, filled_v)
    se_adj = math.sqrt(1 / sum(1 / x for x in filled_v))
    return {"imputed": k0, "side": "left" if side > 0 else "right", "adjusted_theta": adj,
            "adjusted_ci_theta": (adj - 1.96 * se_adj, adj + 1.96 * se_adj), "fixed_effect_theta": _fe(theta, v)}


def funnel_svg(theta: List[float], se: List[float], centre: float, ratio: bool, w: int = 360, h: int = 260) -> str:
    if not theta:
        return ""
    smax = max(se) * 1.1 or 1.0
    span = max(max(abs(t - centre) for t in theta), 1.96 * smax) * 1.1 or 1.0
    xmin, xmax = centre - span, centre + span
    X = lambda t: 40 + (t - xmin) / (xmax - xmin) * (w - 60)          # noqa: E731
    Y = lambda s: 20 + s / smax * (h - 50)                             # noqa: E731
    lab = (lambda t: f"{math.exp(t):.2f}") if ratio else (lambda t: f"{t:.2f}")
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}" font-family="sans-serif" font-size="10">',
             f'<rect x="0" y="0" width="{w}" height="{h}" fill="white"/>',
             f'<polyline fill="none" stroke="#999" stroke-dasharray="4 3" points="{X(centre - 1.96 * smax):.1f},{Y(smax):.1f} '
             f'{X(centre):.1f},{Y(0):.1f} {X(centre + 1.96 * smax):.1f},{Y(smax):.1f}"/>',
             f'<line x1="{X(centre):.1f}" y1="{Y(0):.1f}" x2="{X(centre):.1f}" y2="{Y(smax):.1f}" stroke="#555"/>']
    for t, s in zip(theta, se):
        parts.append(f'<circle cx="{X(t):.1f}" cy="{Y(s):.1f}" r="4" fill="#2b6cb0" fill-opacity="0.8"/>')
    parts += [f'<line x1="40" y1="{h - 30}" x2="{w - 20}" y2="{h - 30}" stroke="#333"/>',
              f'<text x="40" y="{h - 15}">{lab(xmin)}</text>', f'<text x="{w - 50}" y="{h - 15}">{lab(xmax)}</text>',
              f'<text x="{X(centre) - 12:.1f}" y="{h - 15}">{lab(centre)}</text>',
              f'<text x="5" y="15">SE 0</text><text x="5" y="{Y(smax):.0f}">SE {smax:.2f}</text>',
              f'<text x="{w / 2 - 40:.0f}" y="{h - 2}">{"effect (log scale)" if ratio else "effect"}</text></svg>']
    return "".join(parts)


def analyse(pooled: Dict[str, Any]) -> Dict[str, Any]:
    """pooled: a result of meta_analysis.pool / pool_advanced (needs per-study estimate + interval)."""
    import meta_analysis as ma
    if not pooled or pooled.get("estimate") is None:
        return {"endpoint": (pooled or {}).get("endpoint"), "k": 0, "egger": None, "trim_fill": None, "funnel_svg": "",
                "notes": ["no pooled estimate - publication bias cannot be assessed"], "version": BIAS_VERSION}
    studies = pooled.get("studies") or []
    theta, se = [], []
    for s in studies:
        if "theta" in s and "se" in s:
            t, e = s["theta"], s["se"]
        else:
            t, e = ma._to_scale({"kind": pooled.get("kind"), "estimate": s.get("estimate"), "lo": s.get("lo"),
                                 "hi": s.get("hi"), "level": s.get("level") or 95})
        if t is not None and e is not None and math.isfinite(t) and math.isfinite(e) and e > 0:
            theta.append(t)
            se.append(e)
    ratio = pooled.get("kind") == "ratio"
    if ratio and pooled["estimate"] <= 0:
        ratio = False
    centre = math.log(pooled["estimate"]) if ratio else pooled["estimate"]
    eg = egger(theta, se)
    tf = trim_and_fill(theta, se)
    if tf:
        tr = math.exp if ratio else (lambda x: x)
        tf["adjusted_estimate"] = tr(tf["adjusted_theta"])
        tf["adjusted_ci"] = (tr(tf["adjusted_ci_theta"][0]), tr(tf["adjusted_ci_theta"][1]))
    notes = []
    if len(theta) < MIN_K_EGGER:
        notes.append(f"only {len(theta)} studies: funnel asymmetry cannot be judged reliably (>= {MIN_K_EGGER} advised)")
    return {"endpoint": pooled.get("endpoint"), "k": len(theta), "egger": eg, "trim_fill": tf,
            "funnel_svg": funnel_svg(theta, se, centre, ratio), "notes": notes, "version": BIAS_VERSION}


def registry_section(exec_synthesis: Dict[str, Any]) -> Dict[str, Any]:
    sw = (exec_synthesis or {}).get("outcome_switching") or []
    reg = (exec_synthesis or {}).get("registry_check") or {}
    return {"outcome_switching": sw, "registry_lookups": reg.get("looked"), "registry_stopped": reg.get("stopped"),
            "statement": (f"{len(sw)} trial(s) published a different primary outcome than registered" if sw else
                          "no registered-vs-published primary outcome difference found" if reg.get("looked") else
                          "registry not checked (offline or no registry ids)")}


def markdown(results: List[Dict[str, Any]], registry: Optional[Dict[str, Any]] = None) -> str:
    L = ["## Publication bias", ""]
    if not results:
        L.append("Not assessed: no endpoint has at least 3 poolable independent studies.")
    for r in results:
        L.append(f"**{r.get('endpoint')}** ({r['k']} studies)")
        if r.get("egger"):
            e = r["egger"]
            L.append(f"- Egger's test: intercept {e['intercept']} (SE {e['se']}), p = {e['p'] if e['p'] is not None else 'n/a'} - {e['interpretation']}")
        if r.get("trim_fill"):
            t = r["trim_fill"]
            L.append(f"- Trim-and-fill: {t['imputed']} stud(ies) imputed on the {t['side']}; adjusted estimate "
                     f"{t['adjusted_estimate']:.2f} (95% CI {t['adjusted_ci'][0]:.2f} to {t['adjusted_ci'][1]:.2f})")
        for n in r.get("notes") or []:
            L.append(f"- Note: {n}")
        if r.get("funnel_svg"):
            L.append("")
            L.append(r["funnel_svg"])
        L.append("")
    if registry:
        L.append(f"- Registered vs published outcomes: {registry['statement']}")
        for s in (registry.get("outcome_switching") or [])[:10]:
            L.append(f"  - {s.get('ref') or s.get('trial') or ''}: {s.get('detail') or s}")
    return "\n".join(L) + "\n"


# ================================================================================================ v26 registry gap
_DONE = {"COMPLETED", "TERMINATED", "SUSPENDED", "WITHDRAWN"}
_ONGOING = {"RECRUITING", "ACTIVE_NOT_RECRUITING", "NOT_YET_RECRUITING", "ENROLLING_BY_INVITATION"}


def registry_gap(registered: List[Dict[str, Any]], rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Registered studies on the question that have no published or posted result among the included evidence.
    It is an UNCERTAINTY to report: a missing result is not evidence of a negative result."""
    import re as _re
    texts = " ".join(" ".join(str(x or "") for x in (r.get("title"), r.get("identifier"), r.get("trial_group"),
                                                     " ".join(f.get("quote") or "" for f in r.get("findings") or [])))
                     for r in rows).upper()
    included_pmids = {str(r.get("pmid")) for r in rows if r.get("pmid")}
    completed_missing, ongoing, matched = [], [], []
    for st in registered or []:
        nct = str(st.get("nct") or "").upper()
        reg = st.get("registry") or {}
        status = str(reg.get("overall_status") or "").upper()
        if not status:
            m = _re.search(r"Status:\s*([A-Z_]+)", st.get("summary") or "")
            status = m.group(1) if m else ""
        found = nct and nct in texts or bool(set(map(str, reg.get("result_pmids") or [])) & included_pmids)
        item = {"nct": nct, "title": st.get("title"), "status": status or "unknown",
                "results_posted": reg.get("has_results"), "enrollment": reg.get("enrollment")}
        if found:
            matched.append(item)
        elif status in _ONGOING:
            ongoing.append(item)
        elif status in _DONE and not reg.get("has_results"):
            completed_missing.append(item)
    return {"registered": len(registered or []), "matched_to_included": len(matched),
            "completed_without_result": completed_missing, "ongoing": ongoing,
            "statement": (f"{len(completed_missing)} registered stud(ies) completed or stopped with no posted or included "
                          f"result; {len(ongoing)} ongoing. Missing results are an uncertainty about the evidence, not "
                          f"proof that they were negative." if registered else
                          "no registry search for this run (add 'clinicaltrials' to the sources, or load trials into the library)")}


def registry_gap_markdown(g: Dict[str, Any]) -> str:
    L = ["## Registered studies and missing evidence", "", f"- {g['statement']}"]
    for it in (g.get("completed_without_result") or [])[:12]:
        L.append(f"  - {it['nct']} ({it['status'].lower()}{', n=' + str(it['enrollment']) if it.get('enrollment') else ''}): "
                 f"{(it.get('title') or '')[:140]}")
    if g.get("ongoing"):
        L.append("- ongoing: " + ", ".join(it["nct"] for it in g["ongoing"][:15]))
    return "\n".join(L) + "\n"
