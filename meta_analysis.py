"""
meta_analysis.py - pooling effect sizes instead of counting studies (local, no API cost)
=======================================================================================

Counting votes ('2 studies show a reduction, 4 show no difference') gives a trial of 15 patients the same weight as a
trial of 3,500 and ignores how precise each estimate is. This module pools the effect sizes the papers report, with
each study weighted by its own precision, using the DerSimonian-Laird random-effects model - the standard method of
systematic reviews. Everything is computed here with the standard library; nothing is sent anywhere.

What it needs from a study: a ratio (HR / OR / RR) or a mean difference WITH a 95% confidence interval. Papers that
report only a p-value cannot be pooled; they are listed as such, never silently dropped.

Rules it keeps (the same ones the rest of the engine keeps):
  - one endpoint at a time (survival is never pooled with proliferation), and ratios are never pooled with differences
  - publications of one trial count once (the engine's trial grouping decides which)
  - a pooled estimate from fewer than two independent studies is not reported
  - heterogeneity (I-squared) is always reported beside the pooled value, and a high value is stated as a warning
"""
from __future__ import annotations

import math
import re
from typing import Any, Dict, List, Optional, Tuple

ANALYSIS_VERSION = "1.0"

# a ratio is null at 1 and pooled on the log scale; a difference is null at 0 and pooled as reported
RATIO_CUE = re.compile(r"\b(?:HR|OR|RR|IRR|aHR|aOR|SHR|hazard ratios?|odds ratios?|risk ratios?|rate ratios?|"
                       r"incidence rate ratios?|relative risks?)\b", re.I)
DIFF_CUE = re.compile(r"\b(?:mean differences?|MDs?|difference in means?|absolute differences?|net changes?|"
                      r"(?:between|treatment|group|adjusted|least[- ]squares?)[- ](?:group )?(?:mean )?differences?|"
                      r"differences? (?:of|was|were)|changes? (?:with|in) [\w\- ]{1,40}? was|"
                      r"standardi[sz]ed mean differences?|SMDs?|betas?|regression coefficients?)\b", re.I)
_CI = re.compile(r"(?:(\d{2})\s*%\s*)?(?:CI|confidence intervals?(?:\s*\(CI\))?)s?\s*[:=,]?\s*[\[(]?\s*(-?\d+(?:\.\d+)?)\s*%?\s*"
                 r"(?:to|-|–|,|and)\s*(-?\d+(?:\.\d+)?)", re.I)
_EST = re.compile(r"(-?\d+(?:\.\d+)?)")
Z95 = 1.959963984540054          # the 97.5th percentile of the standard normal distribution


def _z_for(level: Optional[int]) -> float:
    """The multiplier for the interval's own level: 95% -> 1.96, 90% -> 1.645, 99% -> 2.576."""
    return {90: 1.6448536269514722, 95: Z95, 99: 2.5758293035489004}.get(level or 95, Z95)


def extract_effect(text: str) -> Optional[Dict[str, Any]]:
    """The effect size and interval a sentence reports, as {'kind','estimate','lo','hi','level','raw'}.

    'HR = 0.78, 95% CI 0.65-0.93' -> ratio 0.78 (0.65, 0.93); 'mean difference 193 mL, 95% CI 100 to 285' ->
    difference 193 (100, 285). None when no interval is reported - a bare number cannot be weighted.
    """
    from stats_parsing import normalize_numeric_text
    t = normalize_numeric_text(text or "")          # v25: '−2.6 (95% CI −3.3 to −1.8)', '0·97' were not parsed
    m = _CI.search(t)
    if not m:
        return None
    level = int(m.group(1)) if m.group(1) else 95
    lo, hi = float(m.group(2)), float(m.group(3))
    if lo > hi:
        lo, hi = hi, lo
    before = t[max(0, m.start() - 90):m.start()]
    kind = "ratio" if RATIO_CUE.search(before) else "difference" if DIFF_CUE.search(before) else None
    if kind is None:
        # no cue: an interval whose bounds are both positive and straddle-able around 1 is treated as a ratio only
        # when the text names one elsewhere; otherwise the shape is unknown and the study is not pooled
        kind = "ratio" if RATIO_CUE.search(t) else "difference" if DIFF_CUE.search(t) else None
    if kind is None:
        return None
    # the estimate is the last number before the interval ('HR = 0.78, 95% CI ...')
    nums = _EST.findall(re.sub(r"\b\d{2}\s*%\s*(?:CI|confidence)", " ", before))
    est = float(nums[-1]) if nums else None
    if est is None or (kind == "ratio" and est <= 0):
        return None
    if not (lo <= est <= hi):
        return None                      # an estimate outside its own interval means the sentence was misread
    if kind == "ratio" and lo <= 0:
        return None                      # a ratio interval cannot reach zero; the numbers are not a ratio
    return {"kind": kind, "estimate": est, "lo": lo, "hi": hi, "level": level, "raw": t[max(0, m.start() - 90):m.end()].strip()}


def _to_scale(eff: Dict[str, Any]) -> Tuple[float, float]:
    """(value, standard error) on the scale the pooling happens on: log for a ratio, as reported for a difference."""
    z = _z_for(eff.get("level"))
    # the same checks as extract_effect, for effects that come from anywhere else (a stored dossier, the API):
    # an estimate outside its own interval or a ratio interval reaching zero is a misreading -> (nan, nan), skipped
    try:
        est, lo, hi = float(eff["estimate"]), float(eff["lo"]), float(eff["hi"])
    except (KeyError, TypeError, ValueError):
        return float("nan"), float("nan")
    if not all(math.isfinite(x) for x in (est, lo, hi)) or not (lo <= est <= hi) or lo == hi or \
            (eff["kind"] == "ratio" and lo <= 0):
        return float("nan"), float("nan")
    if eff["kind"] == "ratio":
        theta = math.log(eff["estimate"])
        se = (math.log(eff["hi"]) - math.log(eff["lo"])) / (2 * z)
    else:
        theta = float(eff["estimate"])
        se = (eff["hi"] - eff["lo"]) / (2 * z)
    return theta, se


def pool(studies: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """DerSimonian-Laird random-effects pooling of studies that report the same KIND of effect.

    studies: [{'ref','effect': {...from extract_effect...}}, ...]
    Returns the pooled estimate with its interval, the heterogeneity statistics, and each study's weight - or None
    when fewer than two studies can be pooled.
    """
    rows = []
    for s in studies:
        eff = s.get("effect") or {}
        if not eff.get("kind"):
            continue
        theta, se = _to_scale(eff)
        if not (se > 0) or not math.isfinite(se) or not math.isfinite(theta):
            continue
        rows.append({"ref": s.get("ref"), "theta": theta, "se": se, "v": se * se, "effect": eff})
    kinds = {r["effect"]["kind"] for r in rows}
    if len(rows) < 2 or len(kinds) != 1:
        return None                      # ratios and differences are different quantities - never pooled together
    kind = kinds.pop()
    # fixed-effect (inverse-variance) weights first: they give Q, the heterogeneity statistic
    w_fe = [1.0 / r["v"] for r in rows]
    sum_w = sum(w_fe)
    theta_fe = sum(w * r["theta"] for w, r in zip(w_fe, rows)) / sum_w
    q = sum(w * (r["theta"] - theta_fe) ** 2 for w, r in zip(w_fe, rows))
    df = len(rows) - 1
    c = sum_w - sum(w * w for w in w_fe) / sum_w
    tau2 = max(0.0, (q - df) / c) if c > 0 else 0.0
    i2 = max(0.0, (q - df) / q) * 100.0 if q > 0 else 0.0
    # random-effects weights add the between-study variance
    w_re = [1.0 / (r["v"] + tau2) for r in rows]
    sum_wre = sum(w_re)
    theta_re = sum(w * r["theta"] for w, r in zip(w_re, rows)) / sum_wre
    se_re = math.sqrt(1.0 / sum_wre)
    lo_s, hi_s = theta_re - Z95 * se_re, theta_re + Z95 * se_re
    if kind == "ratio":
        est, lo, hi, null = math.exp(theta_re), math.exp(lo_s), math.exp(hi_s), 1.0
    else:
        est, lo, hi, null = theta_re, lo_s, hi_s, 0.0
    z = theta_re / se_re if se_re > 0 else 0.0
    for r, w in zip(rows, w_re):
        r["weight_pct"] = 100.0 * w / sum_wre
    return {
        "kind": kind, "k": len(rows), "estimate": est, "lo": lo, "hi": hi, "null": null,
        "significant": not (lo <= null <= hi),
        "z": z, "p": _normal_two_sided(abs(z)),
        "q": q, "df": df, "i2": i2, "tau2": tau2,
        "heterogeneity": _i2_label(i2),
        "studies": [{"ref": r["ref"], "estimate": r["effect"]["estimate"], "lo": r["effect"]["lo"], "hi": r["effect"]["hi"],
                     "weight_pct": r["weight_pct"], "kind": r["effect"]["kind"], "level": r["effect"].get("level") or 95,
                     "theta": r["theta"], "se": r["se"]} for r in rows],
        "version": ANALYSIS_VERSION,
    }


def _i2_label(i2: float) -> str:
    """Cochrane's rough bands, stated as the rough guide they are."""
    if i2 < 30:
        return "low (the studies agree closely)"
    if i2 < 50:
        return "moderate"
    if i2 < 75:
        return "substantial - the pooled value should be read with caution"
    return "considerable - the studies disagree too much for one pooled value to summarise them"


def _normal_two_sided(z: float) -> float:
    """Two-sided p-value from a z-score (the error function, so no SciPy is needed)."""
    return max(0.0, min(1.0, math.erfc(z / math.sqrt(2.0))))


def _fmt(v: float, kind: str) -> str:
    return f"{v:.2f}" if kind == "ratio" else f"{v:.3g}"


def forest_plot(result: Dict[str, Any], width: int = 44) -> List[str]:
    """A forest plot in plain text: one line per study, its interval drawn to scale, the pooled value last.

    Drawn with text so it reads the same in the terminal, in the report and in a file the user keeps.
    """
    if not result:
        return []
    kind = result["kind"]
    rows = result["studies"]
    log = kind == "ratio"

    def pos(v: float) -> float:
        return math.log(max(v, 1e-9)) if log else v

    lo_all = min([pos(r["lo"]) for r in rows] + [pos(result["lo"])])
    hi_all = max([pos(r["hi"]) for r in rows] + [pos(result["hi"])])
    span = (hi_all - lo_all) or 1.0
    lo_all, hi_all = lo_all - 0.08 * span, hi_all + 0.08 * span
    span = hi_all - lo_all

    def col(v: float) -> int:
        return max(0, min(width - 1, int(round((pos(v) - lo_all) / span * (width - 1)))))

    null_col = col(result["null"])
    out = []
    for r in rows:
        line = [" "] * width
        line[null_col] = "|"
        a, b = col(r["lo"]), col(r["hi"])
        for i in range(a, b + 1):
            if line[i] == " ":
                line[i] = "-"
        line[col(r["estimate"])] = "#"
        out.append(f"{str(r['ref'] or '?'):>6}  {''.join(line)}  {_fmt(r['estimate'], kind)} "
                   f"({_fmt(r['lo'], kind)}, {_fmt(r['hi'], kind)})  {r['weight_pct']:.0f}%")
    line = [" "] * width
    line[null_col] = "|"
    a, b = col(result["lo"]), col(result["hi"])
    for i in range(a, b + 1):
        if line[i] == " ":
            line[i] = "="
    line[col(result["estimate"])] = "@"
    out.append(f"{'POOLED':>6}  {''.join(line)}  {_fmt(result['estimate'], kind)} "
               f"({_fmt(result['lo'], kind)}, {_fmt(result['hi'], kind)})  100%")
    axis = [" "] * width
    axis[null_col] = "+"
    out.append(f"{'':>6}  {''.join(axis)}  no difference = {_fmt(result['null'], kind)}")
    return out


def group_for_pooling(rows: List[Dict[str, Any]], endpoint: Optional[str] = None) -> Dict[str, List[Dict[str, Any]]]:
    """Collect poolable studies from the evidence matrix, keyed by '<endpoint>|<kind>'.

    One study contributes once per endpoint (its first poolable finding), publications of one trial contribute once
    (the row's trial_group decides), and reviews never contribute - they pool other people's studies already.
    """
    groups: Dict[str, List[Dict[str, Any]]] = {}
    seen_unit: set = set()
    seen_fp: Dict[tuple, Any] = {}
    duplicates: Dict[str, List[Dict[str, Any]]] = {}
    for r in rows or []:
        if r.get("evidence_type") == "review_meta" or r.get("secondary_report"):
            continue
        unit = r.get("trial_group") or r.get("ref")
        for f in r.get("findings") or []:
            ep = f.get("endpoint")
            if endpoint and ep != endpoint:
                continue
            if f.get("answers_question") is False:
                continue
            text = f"{f.get('quote') or ''} {f.get('statement') or ''}"
            eff = extract_effect(text)
            if not eff:
                continue
            key = f"{ep}|{eff['kind']}"
            if (key, unit) in seen_unit:
                continue
            # the SAME estimate and interval to two decimals cannot come from two independent samples: it is one
            # analysis published twice (or quoted from the other paper). Pooling it twice would invent precision.
            fp = (key, round(eff["estimate"], 4), round(eff["lo"], 4), round(eff["hi"], 4))
            if fp in seen_fp:
                duplicates.setdefault(key, []).append({"ref": r.get("ref"), "same_as": seen_fp[fp],
                                                        "effect": f"{eff['estimate']:g} ({eff['lo']:g}, {eff['hi']:g})"})
                continue
            seen_fp[fp] = r.get("ref")
            seen_unit.add((key, unit))
            groups.setdefault(key, []).append({"ref": r.get("ref"), "effect": eff, "endpoint": ep,
                                               "model_type": f.get("model_type") or r.get("evidence_type")})
    groups["__duplicates__"] = [d for v in duplicates.values() for d in v]
    return groups


def analyse(rows: List[Dict[str, Any]], endpoints: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    """Every endpoint with at least two poolable independent studies, strongest evidence first.

    Human studies are pooled separately from animal and cell studies: pooling a mouse result with a trial would
    produce a number that describes nothing.
    """
    out = []
    grouped = group_for_pooling(rows)
    dups = grouped.pop("__duplicates__", [])
    for key, studies in grouped.items():
        ep, kind = key.split("|", 1)
        if endpoints and ep not in endpoints:
            continue
        for bucket, want in (("human", ("human_clinical", "human_observational", "case_report")),
                             ("animal", ("animal", "experimental_disease_model")),
                             ("cell", ("cell_culture", "organoid_ex_vivo"))):
            part = [s for s in studies if (s.get("model_type") or "") in want]
            res = pool(part)
            if res:
                res["endpoint"] = ep
                res["bucket"] = bucket
                res["designs"] = sorted({s.get("model_type") for s in part if s.get("model_type")})
                res["duplicates"] = [d for d in dups if d.get("same_as") in {x["ref"] for x in res["studies"]}]
                out.append(res)
    return sorted(out, key=lambda r: ({"human": 0, "animal": 1, "cell": 2}.get(r["bucket"], 3), -r["k"]))


def duplicate_effects(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Studies that report an effect size identical to another study's - counted once, and said so."""
    return group_for_pooling(rows).get("__duplicates__") or []


def not_poolable(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Studies whose findings answer the question but report no interval - they cannot be weighted.

    Listed in the report so the pooled estimate is never mistaken for the whole evidence base.
    """
    out = []
    for r in rows or []:
        if r.get("evidence_type") == "review_meta" or r.get("secondary_report"):
            continue
        for f in r.get("findings") or []:
            if f.get("answers_question") is False:
                continue
            text = f"{f.get('quote') or ''} {f.get('statement') or ''}"
            if extract_effect(text):
                continue
            out.append({"ref": r.get("ref"), "endpoint": f.get("endpoint"), "direction": f.get("direction"),
                        "why": "no confidence interval reported in the verified passage"})
            break
    return out


# =====================================================================================================================
# v25 - quantitative meta-analysis for several randomised trials
# =====================================================================================================================
# DerSimonian-Laird (above) underestimates the between-study variance when there are few trials, and its normal-based
# interval is too narrow then. When at least two independent RANDOMISED trials report the same effect with an interval,
# the run also reports the methods recommended for few studies:
#   tau^2 by restricted maximum likelihood (REML; Paule-Mandel when REML does not converge)
#   the Hartung-Knapp-Sidik-Jonkman (HKSJ) interval with a t distribution on k-1 degrees of freedom
#   a 95% prediction interval (where the effect of a NEW trial would lie; needs k >= 3)
#   leave-one-out sensitivity (does any single trial carry the result?)
ANALYSIS_VERSION_V25 = "2.0"


def _t_quantile(p: float, df: int) -> float:
    try:
        from scipy.stats import t as _t
        return float(_t.ppf(p, df))
    except Exception:          # no scipy: the normal quantile (slightly too narrow for very small k)
        return Z95 if abs(p - 0.975) < 1e-9 else 1.6448536269514722


def _t_two_sided(x: float, df: int) -> float:
    try:
        from scipy.stats import t as _t
        return float(2 * _t.sf(abs(x), df))
    except Exception:
        return _normal_two_sided(abs(x))


def _tau2_reml(y: List[float], v: List[float], tol: float = 1e-8, max_iter: int = 200) -> Tuple[float, bool]:
    """REML estimate of the between-study variance (Fisher scoring)."""
    tau2 = max(0.0, (sum((yy - sum(y) / len(y)) ** 2 for yy in y) / max(1, len(y) - 1)) - sum(v) / len(v))
    for _ in range(max_iter):
        w = [1.0 / (vi + tau2) for vi in v]
        sw = sum(w)
        mu = sum(wi * yi for wi, yi in zip(w, y)) / sw
        # REML fixed-point iteration (Viechtbauer 2005, eq. 12)
        new = max(0.0, sum(wi * wi * ((yi - mu) ** 2 - vi) for wi, yi, vi in zip(w, y, v)) / sum(wi * wi for wi in w)
                  + 1.0 / sw)
        if abs(new - tau2) < tol:
            return new, True
        tau2 = new
    return tau2, False


def _tau2_pm(y: List[float], v: List[float]) -> float:
    """Paule-Mandel: tau^2 such that the generalised Q equals k - 1 (bisection)."""
    k = len(y)

    def q(t2):
        w = [1.0 / (vi + t2) for vi in v]
        mu = sum(wi * yi for wi, yi in zip(w, y)) / sum(w)
        return sum(wi * (yi - mu) ** 2 for wi, yi in zip(w, y))
    if q(0.0) <= k - 1:
        return 0.0
    lo, hi = 0.0, max(1e-6, max(abs(yi) for yi in y) ** 2 + max(v) * 10)
    while q(hi) > k - 1:
        hi *= 2
    for _ in range(200):
        mid = (lo + hi) / 2
        if q(mid) > k - 1:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def pool_advanced(studies: List[Dict[str, Any]], method: str = "REML", hksj: bool = True) -> Optional[Dict[str, Any]]:
    rows = []
    for s in studies:
        eff = s.get("effect") or {}
        if not eff.get("kind"):
            continue
        theta, se = _to_scale(eff)
        if se > 0 and math.isfinite(se) and math.isfinite(theta):
            rows.append({"ref": s.get("ref"), "theta": theta, "v": se * se, "effect": eff})
    kinds = {r["effect"]["kind"] for r in rows}
    if len(rows) < 2 or len(kinds) != 1:
        return None
    kind = kinds.pop()
    y, v = [r["theta"] for r in rows], [r["v"] for r in rows]
    k = len(y)
    used = method.upper()
    if used == "REML":
        tau2, ok = _tau2_reml(y, v)
        if not ok:
            tau2, used = _tau2_pm(y, v), "PM"
    elif used == "PM":
        tau2 = _tau2_pm(y, v)
    else:
        base = pool(studies)
        tau2, used = (base or {}).get("tau2", 0.0), "DL"
    w = [1.0 / (vi + tau2) for vi in v]
    sw = sum(w)
    mu = sum(wi * yi for wi, yi in zip(w, y)) / sw
    se = math.sqrt(1.0 / sw)
    if hksj and k >= 2:
        q_hk = sum(wi * (yi - mu) ** 2 for wi, yi in zip(w, y)) / (k - 1)
        se_hk = math.sqrt(max(q_hk, 1.0) / sw)          # the 'ad hoc' floor (never narrower than the standard interval)
        tq = _t_quantile(0.975, k - 1)
        lo_s, hi_s, se_used = mu - tq * se_hk, mu + tq * se_hk, se_hk
        p = _t_two_sided(mu / se_hk, k - 1) if se_hk > 0 else 1.0
        ci_method = f"Hartung-Knapp-Sidik-Jonkman (t, {k - 1} df)"
    else:
        lo_s, hi_s, se_used = mu - Z95 * se, mu + Z95 * se, se
        p = _normal_two_sided(abs(mu / se)) if se > 0 else 1.0
        ci_method = "normal (Wald)"
    pi = None
    if k >= 3:
        tq2 = _t_quantile(0.975, k - 2)
        half = tq2 * math.sqrt(tau2 + se_used ** 2)
        pi = (mu - half, mu + half)
    w_fe = [1.0 / vi for vi in v]
    mu_fe = sum(wi * yi for wi, yi in zip(w_fe, y)) / sum(w_fe)
    qstat = sum(wi * (yi - mu_fe) ** 2 for wi, yi in zip(w_fe, y))
    i2 = max(0.0, (qstat - (k - 1)) / qstat) * 100.0 if qstat > 0 else 0.0
    tr = (lambda x: math.exp(x)) if kind == "ratio" else (lambda x: x)
    null = 1.0 if kind == "ratio" else 0.0
    est, lo, hi = tr(mu), tr(lo_s), tr(hi_s)
    out = {"kind": kind, "k": k, "estimate": est, "lo": lo, "hi": hi, "null": null, "significant": not (lo <= null <= hi),
           "p": p, "tau2": tau2, "tau2_method": used, "ci_method": ci_method, "i2": i2, "q": qstat, "df": k - 1,
           "heterogeneity": _i2_label(i2),
           "prediction_interval": (tr(pi[0]), tr(pi[1])) if pi else None,
           "studies": [{"ref": r["ref"], "estimate": r["effect"]["estimate"], "lo": r["effect"]["lo"], "hi": r["effect"]["hi"],
                        "weight_pct": 100.0 * wi / sw, "theta": r["theta"], "se": math.sqrt(r["v"]),
                        "level": r["effect"].get("level") or 95} for r, wi in zip(rows, w)],
           "version": ANALYSIS_VERSION_V25}
    if k >= 3:
        loo = []
        for i in range(k):
            sub = [s for j, s in enumerate(studies) if j != i and (s.get("effect") or {}).get("kind")]
            r = pool_advanced(sub, method=method, hksj=hksj) if len(sub) >= 2 else None
            if r:
                loo.append({"omitted": rows[i]["ref"], "estimate": r["estimate"], "lo": r["lo"], "hi": r["hi"],
                            "significant": r["significant"]})
        out["leave_one_out"] = loo
        out["robust"] = all(x["significant"] == out["significant"] for x in loo) if loo else None
    return out


def rct_meta(rows: List[Dict[str, Any]], endpoints: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    """Pooling restricted to independent human RANDOMISED trials (publications of one trial count once), per endpoint
    and effect kind, with REML + HKSJ + prediction interval + leave-one-out. Empty when fewer than 2 trials."""
    rct_rows = [r for r in rows or [] if r.get("evidence_type") == "human_clinical" and not r.get("secondary_report")
                and re.search(r"random", f"{r.get('design') or ''} {r.get('title') or ''} {r.get('evidence_type_label') or ''} "
                                         f"{' '.join(f.get('quote') or '' for f in r.get('findings') or [])}", re.I)]
    grouped = group_for_pooling(rct_rows)
    grouped.pop("__duplicates__", None)
    out = []
    for key, studies in grouped.items():
        ep, kind = key.split("|", 1)
        if endpoints and ep not in endpoints:
            continue
        res = pool_advanced(studies)
        if res:
            res["endpoint"], res["bucket"], res["design"] = ep, "human RCTs", "randomised controlled trials only"
            out.append(res)
    return sorted(out, key=lambda r: -r["k"])


def advanced_markdown(results: List[Dict[str, Any]]) -> str:
    if not results:
        return ""
    L = ["## Meta-analysis of randomised trials (REML, Hartung-Knapp, prediction interval)", ""]
    for r in results:
        f = (lambda x: f"{x:.2f}")
        L.append(f"**{r['endpoint']}** - {r['k']} independent trials: pooled {'ratio' if r['kind'] == 'ratio' else 'difference'} "
                 f"**{f(r['estimate'])}** (95% CI {f(r['lo'])} to {f(r['hi'])}; {r['ci_method']}; p = {r['p']:.3g}); "
                 f"tau² {r['tau2']:.3g} ({r['tau2_method']}), I² {r['i2']:.0f}% ({r['heterogeneity']})"
                 + (f"; 95% prediction interval {f(r['prediction_interval'][0])} to {f(r['prediction_interval'][1])}"
                    if r.get("prediction_interval") else "; prediction interval needs >= 3 trials")
                 + (f"; leave-one-out: {'robust' if r.get('robust') else 'NOT robust - one trial carries the result'}"
                    if r.get("leave_one_out") else ""))
        L.append("")
        L.append("| trial | estimate | 95% CI | weight |")
        L.append("|---|---|---|---|")
        for s in r["studies"]:
            L.append(f"| {s['ref']} | {s['estimate']:g} | {s['lo']:g} to {s['hi']:g} | {s['weight_pct']:.0f}% |")
        L.append("")
    return "\n".join(L) + "\n"


# ================================================================================================ v26 subgroup analysis
def _chi2_sf(x: float, df: int) -> float:
    try:
        from scipy.stats import chi2
        return float(chi2.sf(x, df))
    except Exception:                                # Wilson-Hilferty approximation
        if df <= 0:
            return 1.0
        z = ((x / df) ** (1 / 3) - (1 - 2 / (9 * df))) / math.sqrt(2 / (9 * df))
        return 0.5 * math.erfc(z / math.sqrt(2))


SUBGROUP_FACTORS = ("risk of bias", "dose relevance", "follow-up length", "population subgroup")


def _factor(row: Dict[str, Any], factor: str) -> Optional[str]:
    if factor == "risk of bias":
        o = (row.get("risk_of_bias_detail") or {}).get("overall")
        return None if not o or o == "not assessable" else ("low" if o == "low" else "some concerns / high")
    if factor == "dose relevance":
        c = (row.get("dose_relevance") or {}).get("category")
        return c if c in ("physiological", "borderline", "supra-physiological") else None
    if factor == "follow-up length":
        d = ((row.get("quality_profile") or {}).get("duration") or {}).get("days")
        return None if d is None else ("< 6 months" if d < 182 else ">= 6 months")
    if factor == "population subgroup":
        subs = {str(f.get("subgroup")).strip().lower() for f in row.get("findings") or [] if f.get("subgroup")}
        return "subgroup only" if subs else "whole population"
    return None


def subgroup_analysis(pooled: Dict[str, Any], rows: List[Dict[str, Any]], min_k: int = 4) -> List[Dict[str, Any]]:
    """Pre-specified subgroups (risk of bias, dose relevance, follow-up length, subgroup-only reports), each pooled with
    REML; Q-between test for a difference between subgroups. Run only when the pooled result has >= min_k studies and a
    factor splits them into >= 2 groups of >= 2 studies. Exploratory by nature: reported, never used to change the verdict."""
    if not pooled or pooled.get("k", 0) < min_k:
        return []
    by_ref = {r.get("ref"): r for r in rows}
    studies = [{"ref": st["ref"], "effect": {"kind": pooled["kind"], "estimate": st["estimate"], "lo": st["lo"],
                                              "hi": st["hi"], "level": st.get("level") or 95}} for st in pooled.get("studies") or []]
    out = []
    for factor in SUBGROUP_FACTORS:
        groups: Dict[str, List[Dict[str, Any]]] = {}
        for st in studies:
            lab = _factor(by_ref.get(st["ref"]) or {}, factor)
            if lab:
                groups.setdefault(lab, []).append(st)
        groups = {k: v for k, v in groups.items() if len(v) >= 2}
        if len(groups) < 2:
            continue
        res = {}
        for lab, sts in groups.items():
            pg = pool_advanced(sts, hksj=False)
            if pg:
                res[lab] = pg
        if len(res) < 2:
            continue
        log = pooled["kind"] == "ratio"
        mus = {k: (math.log(v["estimate"]) if log else v["estimate"]) for k, v in res.items()}
        ses = {k: ((math.log(v["hi"]) - math.log(v["lo"])) if log else (v["hi"] - v["lo"])) / (2 * 1.959964) for k, v in res.items()}
        w = {k: 1 / ses[k] ** 2 for k in res if ses[k] > 0}
        if len(w) < 2:
            continue
        mu = sum(w[k] * mus[k] for k in w) / sum(w.values())
        q = sum(w[k] * (mus[k] - mu) ** 2 for k in w)
        p = _chi2_sf(q, len(w) - 1)
        out.append({"endpoint": pooled.get("endpoint"), "factor": factor,
                    "groups": {k: {"k": v["k"], "estimate": round(v["estimate"], 4), "lo": round(v["lo"], 4),
                                   "hi": round(v["hi"], 4)} for k, v in res.items()},
                    "q_between": round(q, 3), "df": len(w) - 1, "p": round(p, 4),
                    "interpretation": ("subgroups differ (p < 0.10) - exploratory, not a confirmed effect modifier"
                                       if p < 0.10 else "no evidence that the subgroups differ"),
                    "excluded": [st["ref"] for st in studies if not _factor(by_ref.get(st["ref"]) or {}, factor)]})
    return out


def subgroup_markdown(results: List[Dict[str, Any]]) -> str:
    if not results:
        return ""
    L = ["## Subgroup analyses (pre-specified factors, exploratory)", ""]
    for r in results:
        L.append(f"**{r['endpoint']}** by {r['factor']}: Q = {r['q_between']} (df {r['df']}), p = {r['p']} - {r['interpretation']}")
        for g, v in r["groups"].items():
            L.append(f"- {g}: {v['k']} studies, pooled {v['estimate']:.2f} (95% CI {v['lo']:.2f} to {v['hi']:.2f})")
        if r["excluded"]:
            L.append(f"- not classifiable for this factor (left out): {', '.join(map(str, r['excluded']))}")
        L.append("")
    return "\n".join(L) + "\n"
