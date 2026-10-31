"""
model_uncertainty.py  (v26)
===========================
A simulator gives precise numbers from a possibly wrong model. This module reports uncertainty at every level the
virtual lab can measure:

    numerical(sim, params, observable, metric)       solver error: the run repeated with 100x tighter and 100x looser
                                                     tolerances and with twice the output points
    parametric(sim, params, observable, uncertain)   parameter / measurement uncertainty propagated by Monte Carlo
                                                     (log-normal with a CV per parameter, or a fitted covariance from
                                                     model_validation) -> median and 95% prediction band over time,
                                                     interval of the metric
    structural(library, model_id, params, ...)       the SAME experiment on every alternative model structure of the
                                                     family (e.g. 1- vs 2-compartment PK, Gompertz vs logistic growth):
                                                     spread of the metric across structures; with data, AIC of each fit
    assumptions(sim, params, observable)             one-at-a-time sensitivity of the result to each parameter (+/-20%)
    report(...)                                      all of the above in one dict + markdown
Repeating a simulation does not make its biology more correct: the report always says which uncertainty was NOT
assessed (e.g. no independent data -> no external validation).
"""
from __future__ import annotations

import math
import random
from typing import Any, Dict, List, Optional, Sequence

UNCERTAINTY_VERSION = "1.0"


def _metric(run: Dict[str, Any], observable: str, metric: str) -> Optional[float]:
    return (run.get("summary") or {}).get(observable, {}).get(metric)


def numerical(sim: Any, params: Dict[str, float], observable: str, metric: str = "max", t_end: float = 100,
              points: int = 201) -> Dict[str, Any]:
    base = sim.run(params, t_end=t_end, points=points, observables=[observable])
    tight = sim.run(params, t_end=t_end, points=points, observables=[observable], rtol=1e-10, atol=1e-12)
    loose = sim.run(params, t_end=t_end, points=points, observables=[observable], rtol=1e-6, atol=1e-8)
    fine = sim.run(params, t_end=t_end, points=min(5000, 2 * points - 1), observables=[observable])
    v0, vt, vl, vf = (_metric(r, observable, metric) for r in (base, tight, loose, fine))
    def rel(a, b):
        return None if a is None or b is None else abs(a - b) / max(abs(b), 1e-300)
    err = rel(v0, vt)
    return {"value": v0, "tight_tolerance": vt, "loose_tolerance": vl, "finer_output": vf,
            "relative_solver_error": err, "relative_output_grid_effect": rel(v0, vf),
            "ok": err is not None and err < 1e-4,
            "note": ("the output grid changes the metric - 'max' / 'time_to_max' depend on the sampling; use more points"
                     if (rel(v0, vf) or 0) > 1e-3 else None)}


def parametric(sim: Any, params: Dict[str, float], observable: str, uncertain: Dict[str, float],
               metric: str = "max", n: int = 200, t_end: float = 100, points: int = 101, seed: int = 1,
               covariance: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """uncertain: {parameter: coefficient of variation} (log-normal), clipped to the card's range.
    covariance: {'names': [...], 'mean': [...], 'cov': [[...]]} from a fit (multivariate normal)."""
    rng = random.Random(seed)
    card = sim.card
    import numpy as np
    draws: List[Dict[str, float]] = []
    if covariance:
        nr = np.random.default_rng(seed)
        z = nr.multivariate_normal(np.array(covariance["mean"], float), np.array(covariance["cov"], float), size=n)
        for row in z:
            draws.append({k: float(v) for k, v in zip(covariance["names"], row)})
    else:
        for _ in range(n):
            d = {}
            for k, cv in uncertain.items():
                mu = float(params.get(k, card["parameters"][k]["default"]))
                if mu <= 0:
                    d[k] = mu
                    continue
                s = math.sqrt(math.log(1 + cv * cv))
                d[k] = mu * math.exp(rng.gauss(-s * s / 2, s))
            draws.append(d)
    vals, curves, failed = [], [], 0
    t = None
    for d in draws:
        p = dict(params)
        for k, v in d.items():
            lo, hi = card["parameters"][k]["min"], card["parameters"][k]["max"]
            p[k] = min(max(v, lo), hi)
        try:
            r = sim.run(p, t_end=t_end, points=points, observables=[observable])
        except Exception:
            failed += 1
            continue
        if not r["numerically_ok"]:
            failed += 1
            continue
        t = r["time"]
        curves.append(r["series"][observable])
        m = _metric(r, observable, metric)
        if m is not None:
            vals.append(m)
    if not vals:
        return {"n": 0, "failed": failed}
    arr = np.array(curves)
    q = lambda a, x: float(np.quantile(a, x))   # noqa: E731
    return {"n": len(vals), "failed": failed, "metric": metric, "median": q(vals, 0.5), "p2_5": q(vals, 0.025),
            "p97_5": q(vals, 0.975), "relative_width": (q(vals, 0.975) - q(vals, 0.025)) / max(abs(q(vals, 0.5)), 1e-300),
            "band": {"time": t[::max(1, len(t) // 50)], "p2_5": [float(x) for x in np.quantile(arr, 0.025, axis=0)][::max(1, len(t) // 50)],
                     "median": [float(x) for x in np.quantile(arr, 0.5, axis=0)][::max(1, len(t) // 50)],
                     "p97_5": [float(x) for x in np.quantile(arr, 0.975, axis=0)][::max(1, len(t) // 50)]},
            "uncertain": uncertain if not covariance else {"from_fit": covariance["names"]}, "seed": seed}


def structural(library: Any, model_id: str, params: Dict[str, float], observable: str, metric: str = "max",
               t_end: float = 100, points: int = 201, data: Optional[Dict[str, Any]] = None,
               fit: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    import simulator as S
    card = library.get(model_id)
    fam = card.get("family")
    alts = [c for c in library.cards.values() if fam and c.get("family") == fam and observable in c["observables"]]
    if len(alts) < 2:
        return {"family": fam, "structures": 1, "note": "no alternative model structure in the library for this system - "
                                                         "structural uncertainty NOT assessed"}
    out = []
    for c in alts:
        sim = S.Simulator(c)
        p = {k: v for k, v in params.items() if k in c["parameters"]}
        r = sim.run(p, t_end=t_end, points=points, observables=[observable])
        row = {"model": c["id"], "value": _metric(r, observable, metric), "ok": r["numerically_ok"]}
        if data and fit:
            try:
                import model_validation as mv
                f = [x for x in fit if x in c["parameters"]]
                v = mv.validate(sim, {**data, "observable": observable}, f)
                # structures are compared on the HELD-OUT points they did not see (a fit on its own data favours
                # the more flexible structure)
                row.update({"heldout_r2": v.get("r2"), "heldout_rmse": v.get("rmse"), "validation": v.get("verdict"),
                            "parameters_fitted": len(f)})
            except Exception as exc:
                row["fit_error"] = str(exc)[:120]
        out.append(row)
    vals = [o["value"] for o in out if o["value"] is not None]
    spread = (max(vals) - min(vals)) / max(abs(sum(vals) / len(vals)), 1e-300) if len(vals) >= 2 else None
    best = None
    if any(o.get("heldout_rmse") is not None for o in out):
        best = min((o for o in out if o.get("heldout_rmse") is not None), key=lambda o: o["heldout_rmse"])["model"]
    return {"family": fam, "structures": len(out), "results": out, "relative_spread": spread, "best_on_heldout": best,
            "note": ("the model structure changes the answer by more than 20% - the conclusion depends on the assumed structure"
                     if spread is not None and spread > 0.2 else "the answer is stable across the model structures in the library")}


def assumptions(sim: Any, params: Dict[str, float], observable: str, metric: str = "max", t_end: float = 100,
                points: int = 101, delta: float = 0.2) -> List[Dict[str, Any]]:
    import sensitivity as se
    names = [k for k, v in sim.card["parameters"].items() if (params.get(k, v["default"]) or 0) != 0]
    loc = se.local(sim, params, names, observable, metric, t_end=t_end, points=points, rel_step=delta)
    return [{"parameter": k, "relative_sensitivity": loc["sensitivities"][k]["S"]} for k in loc.get("ranking") or []]


def report(library: Any, model_id: str, params: Dict[str, float], observable: str, metric: str = "max",
           t_end: float = 100, uncertain: Optional[Dict[str, float]] = None, n_mc: int = 200,
           data: Optional[Dict[str, Any]] = None, fit: Optional[Sequence[str]] = None, seed: int = 1) -> Dict[str, Any]:
    import simulator as S
    card = library.get(model_id)                    # KeyError: unknown model
    if observable not in card["observables"]:
        raise ValueError(f"{model_id} has no observable '{observable}' ({', '.join(card['observables'])})")
    bad = [k for k in list(params or {}) + list(uncertain or {}) if k not in card["parameters"]]
    if bad:
        raise ValueError(f"not parameters of {model_id}: {', '.join(bad)}")
    from experiment_spec import METRICS
    if metric not in METRICS:
        raise ValueError(f"metric must be one of {METRICS}")
    sim = S.Simulator(card)
    unc = uncertain or {k: 0.2 for k in list(sim.card["parameters"])[:4]}
    res = {"model": model_id, "model_version": sim.version, "observable": observable, "metric": metric,
           "numerical": numerical(sim, params, observable, metric, t_end),
           "parametric": parametric(sim, params, observable, unc, metric, n=n_mc, t_end=t_end, seed=seed),
           "structural": structural(library, model_id, params, observable, metric, t_end, data=data, fit=fit),
           "external_validation": ("held-out data supplied" if data else
                                   "NOT assessed - no independent data supplied (model_validation needs it)"),
           "version": UNCERTAINTY_VERSION}
    try:
        res["sensitivity_to_parameters"] = assumptions(sim, params, observable, metric, t_end)
    except Exception as exc:
        res["sensitivity_to_parameters"] = {"error": str(exc)[:120]}
    return res


def markdown(r: Dict[str, Any]) -> str:
    n, p, s = r["numerical"], r["parametric"], r["structural"]
    L = [f"### Uncertainty of {r['observable']} {r['metric']} ({r['model']})", "",
         f"- numerical: relative solver error {n['relative_solver_error']:.1e}" + (" (ok)" if n["ok"] else " - TOO LARGE")
         + (f"; {n['note']}" if n.get("note") else "")]
    if p.get("n"):
        L.append(f"- parameter uncertainty ({p['n']} Monte Carlo runs): median {p['median']:.4g}, 95% interval "
                 f"{p['p2_5']:.4g} to {p['p97_5']:.4g}")
    L.append(f"- model structure: {s.get('note')}" + (f" (spread {s['relative_spread']:.0%} over {s['structures']} structures)"
                                                     if s.get("relative_spread") is not None else ""))
    L.append(f"- external validation: {r['external_validation']}")
    return "\n".join(L) + "\n"
