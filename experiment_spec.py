"""
experiment_spec.py  (v25)
=========================
Structured experiment spec + validator: the AI PROPOSES, code VALIDATES, and only valid jobs run.

A spec (JSON):
    {
      "model": "pk_oral_1cmt",                      # an id from the model library
      "hypothesis": "1000 mg twice daily keeps plasma metformin above 10 µM",
      "parameters": {"dose": 1000, "tau": 12},       # overrides of the card's defaults (must be card parameters)
      "t_end": 168, "points": 337,                   # in the model's own time unit
      "observables": ["C_uM"],
      "sweep": {"parameter": "dose", "values": [250, 500, 1000, 2000]}      # optional, OR
      "grid":  {"dose": [500, 1000], "tau": [8, 12, 24]}                    # optional 2-D grid
      "sensitivity": {"method": "morris", "parameters": ["ka", "ke", "Vd"], "trajectories": 10, "metric": "max"},
      "threshold": {"parameter": "dose", "observable": "C_uM", "metric": "max", "target": 10, "low": 100, "high": 5000},
      "expected": {"observable": "C_uM", "metric": "min", "comparison": ">=", "value": 10},
      "solver": "cvode" | "gillespie", "seed": 1
    }

The validator refuses (with the reason) anything the model cannot answer or the machine should not run:
    unknown model / parameter / observable; non-numeric values; values outside the card's range; t_end above the
    card's max_time; more than 5,000 output points; a sweep of more than 200 values or a grid of more than 400 cells;
    a Morris design of more than 50 trajectories; more than 2,000 simulations in total; a stochastic run without a seed
    (a seed is assigned and reported); an 'expected' test on an observable or metric that does not exist.
"""
from __future__ import annotations

import copy
import itertools
import math
import re
from typing import Any, Dict, List, Optional

SPEC_VERSION = "1.0"
LIMITS = {"points": 5000, "sweep_values": 200, "grid_cells": 400, "morris_trajectories": 50, "total_simulations": 2000}
METRICS = ("final", "max", "min", "auc", "time_to_max", "mean", "min_late", "mean_late", "max_late")
COMPARISONS = (">", ">=", "<", "<=", "==", "between")


def _num(x: Any) -> Optional[float]:
    if isinstance(x, bool):
        return None
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def validate(spec: Dict[str, Any], library: Any) -> Dict[str, Any]:
    errors: List[str] = []
    warnings: List[str] = []
    if not isinstance(spec, dict):
        return {"valid": False, "errors": ["the spec must be a JSON object"], "warnings": [], "spec": spec, "n_simulations": 0}
    s = copy.deepcopy(spec)
    mid = s.get("model")
    if not isinstance(mid, str) or mid not in library.cards:
        return {"valid": False, "errors": [f"unknown model '{mid}' - choose one of {sorted(library.cards)}"], "warnings": [],
                "spec": s, "n_simulations": 0}
    card = library.get(mid)
    P = card["parameters"]
    # shape first: a wrong type is a reported error, never an exception (the API would answer 500, a job would crash)
    shapes = {"parameters": dict, "sweep": dict, "grid": dict, "sensitivity": dict, "threshold": dict, "expected": dict,
              "observables": list}
    for key, typ in shapes.items():
        if s.get(key) is not None and not isinstance(s[key], typ):
            errors.append(f"'{key}' must be a JSON {'object' if typ is dict else 'list'}, not {type(s[key]).__name__}")
            s.pop(key)
    if isinstance(s.get("sweep"), dict) and not isinstance(s["sweep"].get("values", []), list):
        errors.append("sweep.values must be a list of numbers")
        s.pop("sweep")
    if isinstance(s.get("grid"), dict) and not all(isinstance(v, list) for v in s["grid"].values()):
        errors.append("every grid entry must be a list of numbers")
        s.pop("grid")
    if isinstance(s.get("sensitivity"), dict) and not isinstance(s["sensitivity"].get("parameters", []), list):
        errors.append("sensitivity.parameters must be a list of parameter names")
        s.pop("sensitivity")

    def check_param(name: Any, value: Any, where: str) -> Optional[float]:
        if not isinstance(name, str) or name not in P:
            errors.append(f"{where}: '{name}' is not a parameter of {mid} ({', '.join(sorted(P))})")
            return None
        v = _num(value)
        if v is None:
            errors.append(f"{where}: {name} = {value!r} is not a number")
            return None
        if not (P[name]["min"] <= v <= P[name]["max"]):
            errors.append(f"{where}: {name} = {v:g} {P[name]['unit']} is outside the model's range "
                          f"[{P[name]['min']:g}, {P[name]['max']:g}]")
        return v

    params = {}
    for k, v in (s.get("parameters") or {}).items():
        x = check_param(k, v, "parameters")
        if x is not None:
            params[k] = x
    s["parameters"] = params
    t_end = _num(s.get("t_end", min(card["max_time"], 100)))
    if t_end is None or t_end <= 0:
        errors.append(f"t_end must be a positive number of {card['time_unit']}")
    elif t_end > card["max_time"]:
        errors.append(f"t_end {t_end:g} {card['time_unit']} exceeds this model's limit of {card['max_time']} {card['time_unit']}")
    s["t_end"] = t_end
    pts = s.get("points", 201)
    if not isinstance(pts, int) or pts < 3 or pts > LIMITS["points"]:
        errors.append(f"points must be an integer between 3 and {LIMITS['points']}")
    obs = s.get("observables") or list(card["observables"])
    bad = [o for o in obs if not isinstance(o, str) or o not in card["observables"]]
    if bad:
        errors.append(f"unknown observable(s) {bad} - {mid} reports {sorted(card['observables'])}")
    s["observables"] = obs
    n = 1
    if s.get("sweep"):
        sw = s["sweep"]
        vals = sw.get("values") or []
        if len(vals) > LIMITS["sweep_values"]:
            errors.append(f"sweep of {len(vals)} values exceeds {LIMITS['sweep_values']}")
        sw["values"] = [x for x in (check_param(sw.get("parameter"), v, "sweep") for v in vals) if x is not None]
        n = max(n, len(sw["values"]))
    if s.get("grid"):
        g = s["grid"]
        cells = 1
        for k, vals in g.items():
            g[k] = [x for x in (check_param(k, v, "grid") for v in vals) if x is not None]
            cells *= max(1, len(g[k]))
        if cells > LIMITS["grid_cells"]:
            errors.append(f"grid of {cells} cells exceeds {LIMITS['grid_cells']}")
        n = max(n, cells)
    if s.get("sensitivity"):
        se = s["sensitivity"]
        if se.get("method") not in ("local", "morris"):
            errors.append("sensitivity.method must be 'local' or 'morris'")
        for k in se.get("parameters") or []:
            if k not in P:
                errors.append(f"sensitivity: '{k}' is not a parameter of {mid}")
        if se.get("metric", "max") not in METRICS:
            errors.append(f"sensitivity.metric must be one of {METRICS}")
        r_ = _num(se.get("trajectories", 10))
        r = int(r_) if r_ is not None else 0
        if r_ is None or r_ != r or r < 1:
            errors.append("sensitivity.trajectories must be a positive integer")
            r = 1
        se["trajectories"] = r
        if r > LIMITS["morris_trajectories"]:
            errors.append(f"Morris design of {r} trajectories exceeds {LIMITS['morris_trajectories']}")
        k = len(se.get("parameters") or [])
        n += (r * (k + 1)) if se.get("method") == "morris" else (2 * k + 1)
    if s.get("threshold"):
        th = s["threshold"]
        for key in ("parameter", "observable", "target", "low", "high"):
            if th.get(key) is None:
                errors.append(f"threshold needs '{key}'")
        if th.get("target") is not None:
            if _num(th["target"]) is None:
                errors.append("threshold.target must be a number")
            else:
                th["target"] = _num(th["target"])
        if th.get("parameter"):
            lo = check_param(th["parameter"], th.get("low"), "threshold.low")
            hi = check_param(th["parameter"], th.get("high"), "threshold.high")
            if lo is not None and hi is not None:
                th["low"], th["high"] = lo, hi
                if lo >= hi:
                    errors.append("threshold.low must be below threshold.high")
        if th.get("metric", "max") not in METRICS:
            errors.append(f"threshold.metric must be one of {METRICS}")
        if th.get("observable") and th["observable"] not in card["observables"]:
            errors.append(f"threshold: unknown observable '{th['observable']}'")
        n += 25
    if s.get("expected"):
        e = s["expected"]
        if e.get("observable") not in card["observables"]:
            errors.append(f"expected: unknown observable '{e.get('observable')}'")
        if e.get("metric", "final") not in METRICS:
            errors.append(f"expected.metric must be one of {METRICS}")
        if e.get("comparison") not in COMPARISONS:
            errors.append(f"expected.comparison must be one of {COMPARISONS}")
        if e.get("comparison") == "between":
            vals = e.get("value")
            if not (isinstance(vals, (list, tuple)) and len(vals) == 2 and all(_num(x) is not None for x in vals)):
                errors.append("expected.value must be [low, high] (two numbers) for 'between'")
            else:
                e["value"] = sorted(_num(x) for x in vals)
        elif _num(e.get("value")) is None:
            errors.append("expected.value must be a number")
        else:
            e["value"] = _num(e["value"])           # "5" would otherwise fail later as 13.1 > "5"
    if s.get("solver", "cvode") not in ("cvode", "gillespie"):
        errors.append("solver must be 'cvode' or 'gillespie'")
    if s.get("seed") is None:
        s["seed"] = 1
        if s.get("solver") == "gillespie":
            warnings.append("no seed given for a stochastic run - seed 1 assigned (reported in the log)")
    if n > LIMITS["total_simulations"]:
        errors.append(f"the spec needs {n} simulations - more than the limit of {LIMITS['total_simulations']}")
    for a in card.get("limits") or []:
        if re.search(r"tissue|tumou?r exposure", a, re.I) and re.search(r"tumou?r|tissue", s.get("hypothesis") or "", re.I) \
                and mid.startswith("pk"):
            warnings.append(f"model limit relevant to this hypothesis: {a}")
    return {"valid": not errors, "errors": errors, "warnings": warnings, "spec": s, "n_simulations": n,
            "model_version": None, "version": SPEC_VERSION}


def evaluate_expected(expected: Dict[str, Any], summary: Dict[str, Dict[str, float]]) -> Dict[str, Any]:
    v = (summary.get(expected["observable"]) or {}).get(expected.get("metric", "final"))
    if v is None:
        return {"verdict": "not evaluable", "value": None}
    c, target = expected["comparison"], expected["value"]
    ok = {">": lambda: v > target, ">=": lambda: v >= target, "<": lambda: v < target, "<=": lambda: v <= target,
          "==": lambda: abs(v - target) <= 1e-9 * max(1.0, abs(target)),
          "between": lambda: target[0] <= v <= target[1]}[c]()
    return {"verdict": "consistent with the hypothesis" if ok else "inconsistent with the hypothesis",
            "value": v, "test": f"{expected['observable']} {expected.get('metric', 'final')} {c} {target}",
            "basis": "in silico - shows the hypothesis is (in)consistent with the MODEL, not that it is true"}


def prompt_for_spec(hypothesis: str, library: Any) -> str:
    """The AI is shown the library (ids, parameters with ranges, observables) and asked for ONE spec as JSON."""
    lines = []
    for c in library.catalogue():
        ps = ", ".join(f"{k} [{v['min']:g}-{v['max']:g} {v['unit']}]" for k, v in c["parameters"].items())
        lines.append(f"- {c['id']} ({c['title']}; time in {c['time_unit']}, max {c['max_time']}): parameters {ps}; "
                     f"observables {', '.join(c['observables'])}")
    return ("Design ONE in-silico experiment that tests the hypothesis below with a model from this library. "
            "Use only the listed model ids, parameters (within their ranges) and observables. Return ONLY JSON with keys "
            "model, hypothesis, parameters, t_end, points, observables, and optionally sweep / grid / sensitivity / "
            "threshold / expected (observable, metric in final|max|min|auc|time_to_max|mean|min_late|mean_late|max_late - "
            "'min_late' is the steady-state trough over the last quarter of the run - comparison, value).\n\n"
            "LIBRARY:\n" + "\n".join(lines) + f"\n\nHYPOTHESIS: {hypothesis}")


def propose(hypothesis: str, library: Any, router: Any = None) -> Dict[str, Any]:
    """AI proposal -> validation. Without a router, returns the prompt so a person can write the spec."""
    if router is None:
        return {"proposed": None, "validation": None, "prompt": prompt_for_spec(hypothesis, library)}
    from llm_json import parse_json_response
    raw = router.ask(prompt_for_spec(hypothesis, library), task="hypotheses")
    try:
        spec = parse_json_response(raw)
    except Exception as exc:
        return {"proposed": None, "validation": {"valid": False, "errors": [f"the AI's spec is not JSON: {exc}"]}}
    return {"proposed": spec, "validation": validate(spec, library)}


def expand(spec: Dict[str, Any]) -> List[Dict[str, Any]]:
    """The list of parameter sets a validated spec runs (base, sweep values or grid cells)."""
    base = dict(spec.get("parameters") or {})
    if spec.get("sweep"):
        p = spec["sweep"]["parameter"]
        return [{**base, p: v} for v in spec["sweep"]["values"]]
    if spec.get("grid"):
        keys = list(spec["grid"])
        return [{**base, **dict(zip(keys, combo))} for combo in itertools.product(*[spec["grid"][k] for k in keys])]
    return [base]
