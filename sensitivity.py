"""
sensitivity.py  (v25)
=====================
Sensitivity analysis, parameter sweeps and active learning for library models.

    local(sim, params, observable, metric)      normalised local sensitivity S = (dY/Y) / (dP/P) by central differences
                                                (S = 1: a 10% change of P changes the output by 10%)
    morris(sim, names, observable, metric, r)   Morris elementary effects (global screening over the card's ranges, in
                                                log space for parameters spanning > 1 decade): mu* = importance,
                                                sigma = non-linearity / interaction. Reproducible with the seed.
    sweep(sim, name, values, observable, metric)            1-D sweep;  grid(sim, {p: values}, ...) 2-D
    threshold(sim, name, observable, metric, target, lo, hi) ACTIVE LEARNING by bisection: the parameter value at which
                                                the output crosses the target (e.g. the dose giving Cmax 10 µM), with
                                                every probe kept
    adaptive_sweep(sim, name, lo, hi, observable, metric, budget)
                                                ACTIVE LEARNING: starts with a coarse sweep, then spends the remaining
                                                budget where neighbouring outputs differ most (finds the steep region
                                                of a dose-response with few simulations)
"""
from __future__ import annotations

import math
import random
from typing import Any, Dict, List, Optional, Sequence

SENS_VERSION = "1.0"


def _metric(sim: Any, params: Dict[str, float], observable: str, metric: str, t_end: float, points: int) -> float:
    out = sim.run(params, t_end=t_end, points=points, observables=[observable])
    return float(out["summary"][observable][metric])


def _range(card: Dict[str, Any], name: str):
    p = card["parameters"][name]
    lo, hi = float(p["min"]), float(p["max"])
    logscale = lo > 0 and hi / lo > 10
    return lo, hi, logscale


def local(sim: Any, params: Dict[str, float], names: Sequence[str], observable: str, metric: str = "max",
          t_end: float = 100, points: int = 201, rel_step: float = 0.05) -> Dict[str, Any]:
    card = sim.card
    base_p = {**{k: v["default"] for k, v in card["parameters"].items()}, **(params or {})}
    y0 = _metric(sim, base_p, observable, metric, t_end, points)
    out = {}
    for n in names:
        p0 = base_p[n]
        if p0 == 0:
            out[n] = {"S": None, "note": "parameter is 0 - relative sensitivity undefined"}
            continue
        lo, hi, _ = _range(card, n)
        up, dn = min(hi, p0 * (1 + rel_step)), max(lo, p0 * (1 - rel_step))
        yu = _metric(sim, {**base_p, n: up}, observable, metric, t_end, points)
        yd = _metric(sim, {**base_p, n: dn}, observable, metric, t_end, points)
        S = ((yu - yd) / y0) / ((up - dn) / p0) if y0 and up != dn else None
        out[n] = {"S": None if S is None else round(S, 4), "y_up": yu, "y_down": yd}
    ranked = sorted([k for k in out if out[k]["S"] is not None], key=lambda k: -abs(out[k]["S"]))
    return {"method": "local (central differences, ±5%)", "output": f"{observable} {metric}", "baseline": y0,
            "sensitivities": out, "ranking": ranked, "simulations": 1 + 2 * len(names), "version": SENS_VERSION}


def morris(sim: Any, names: Sequence[str], observable: str, metric: str = "max", r: int = 10, levels: int = 4,
           seed: int = 1, t_end: float = 100, points: int = 201, params: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
    card = sim.card
    rnd = random.Random(seed)
    k = len(names)
    delta = levels / (2 * (levels - 1))
    base = {**{q: v["default"] for q, v in card["parameters"].items()}, **(params or {})}
    rng = {n: _range(card, n) for n in names}

    def to_value(n, u):
        lo, hi, lg = rng[n]
        return math.exp(math.log(lo) + u * (math.log(hi) - math.log(lo))) if lg else lo + u * (hi - lo)

    ee: Dict[str, List[float]] = {n: [] for n in names}
    sims = 0
    grid = [i / (levels - 1) for i in range(levels)]
    for _ in range(r):
        x = [rnd.choice([g for g in grid if g + delta <= 1 + 1e-12]) for _ in range(k)]
        order = list(range(k))
        rnd.shuffle(order)
        y_prev = _metric(sim, {**base, **{n: to_value(n, x[i]) for i, n in enumerate(names)}}, observable, metric, t_end, points)
        sims += 1
        for j in order:
            x2 = list(x)
            x2[j] = x[j] + delta
            y_new = _metric(sim, {**base, **{n: to_value(n, x2[i]) for i, n in enumerate(names)}}, observable, metric, t_end, points)
            sims += 1
            ee[names[j]].append((y_new - y_prev) / delta)
            x, y_prev = x2, y_new
    res = {}
    for n, e in ee.items():
        mu_star = sum(abs(v) for v in e) / len(e)
        mu = sum(e) / len(e)
        sigma = math.sqrt(sum((v - mu) ** 2 for v in e) / (len(e) - 1)) if len(e) > 1 else 0.0
        res[n] = {"mu_star": mu_star, "mu": mu, "sigma": sigma}
    top = max((v["mu_star"] for v in res.values()), default=0) or 1.0
    for v in res.values():
        v["relative_importance"] = round(v["mu_star"] / top, 3)
        v["nonlinear_or_interacting"] = v["sigma"] > 0.5 * v["mu_star"] if v["mu_star"] else False
    return {"method": f"Morris elementary effects (r={r}, {levels} levels, seed {seed})", "output": f"{observable} {metric}",
            "effects": res, "ranking": sorted(res, key=lambda n: -res[n]["mu_star"]), "simulations": sims, "version": SENS_VERSION}


def sweep(sim: Any, name: str, values: Sequence[float], observable: str, metric: str = "max", t_end: float = 100,
          points: int = 201, params: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
    ys = [_metric(sim, {**(params or {}), name: v}, observable, metric, t_end, points) for v in values]
    return {"parameter": name, "values": list(values), "output": f"{observable} {metric}", "results": ys,
            "monotonic": all(a <= b for a, b in zip(ys, ys[1:])) or all(a >= b for a, b in zip(ys, ys[1:])),
            "simulations": len(values)}


def grid(sim: Any, axes: Dict[str, Sequence[float]], observable: str, metric: str = "max", t_end: float = 100,
         points: int = 201, params: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
    import itertools
    keys = list(axes)
    cells = []
    for combo in itertools.product(*[axes[k] for k in keys]):
        p = dict(zip(keys, combo))
        cells.append({**p, "result": _metric(sim, {**(params or {}), **p}, observable, metric, t_end, points)})
    return {"axes": {k: list(v) for k, v in axes.items()}, "output": f"{observable} {metric}", "cells": cells,
            "simulations": len(cells)}


def threshold(sim: Any, name: str, observable: str, metric: str, target: float, low: float, high: float,
              t_end: float = 100, points: int = 201, tol: float = 0.01, max_iter: int = 25,
              params: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
    f = lambda v: _metric(sim, {**(params or {}), name: v}, observable, metric, t_end, points) - target   # noqa: E731
    probes = []
    flo, fhi = f(low), f(high)
    probes += [(low, flo + target), (high, fhi + target)]
    if flo == 0 or fhi == 0:
        return {"found": True, "value": low if flo == 0 else high, "probes": probes, "simulations": 2}
    if (flo > 0) == (fhi > 0):
        return {"found": False, "why": f"the output does not cross {target} between {low:g} and {high:g} "
                                       f"({flo + target:.4g} to {fhi + target:.4g})", "probes": probes, "simulations": 2}
    lo, hi = low, high
    geometric = low > 0 and high / low > 10
    for _ in range(max_iter):
        mid = math.sqrt(lo * hi) if geometric else (lo + hi) / 2
        fm = f(mid)
        probes.append((mid, fm + target))
        if abs(fm) <= tol * max(1.0, abs(target)) or abs(hi - lo) <= tol * abs(mid):
            return {"found": True, "value": mid, "output_at_value": fm + target, "probes": probes, "simulations": len(probes),
                    "method": "bisection (active learning: each probe chosen from the previous ones)"}
        if (fm > 0) == (flo > 0):
            lo, flo = mid, fm
        else:
            hi = mid
    return {"found": True, "value": mid, "output_at_value": fm + target, "probes": probes, "simulations": len(probes),
            "note": "stopped at the iteration limit"}


def adaptive_sweep(sim: Any, name: str, low: float, high: float, observable: str, metric: str = "max",
                   budget: int = 20, n_init: int = 5, t_end: float = 100, points: int = 201,
                   params: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
    geometric = low > 0 and high / low > 10
    if geometric:
        xs = [math.exp(math.log(low) + i * (math.log(high) - math.log(low)) / (n_init - 1)) for i in range(n_init)]
    else:
        xs = [low + i * (high - low) / (n_init - 1) for i in range(n_init)]
    pts = {x: _metric(sim, {**(params or {}), name: x}, observable, metric, t_end, points) for x in xs}
    while len(pts) < budget:
        s = sorted(pts)
        gaps = [(abs(pts[b] - pts[a]), a, b) for a, b in zip(s, s[1:])]
        _, a, b = max(gaps)
        mid = math.sqrt(a * b) if geometric and a > 0 else (a + b) / 2
        if mid in pts:
            break
        pts[mid] = _metric(sim, {**(params or {}), name: mid}, observable, metric, t_end, points)
    s = sorted(pts)
    steep = max(((abs(pts[b] - pts[a]) / (b - a), a, b) for a, b in zip(s, s[1:])), default=(0, None, None))
    return {"parameter": name, "values": s, "results": [pts[x] for x in s], "simulations": len(pts),
            "steepest_interval": [steep[1], steep[2]], "method": "adaptive refinement where neighbouring outputs differ most"}
