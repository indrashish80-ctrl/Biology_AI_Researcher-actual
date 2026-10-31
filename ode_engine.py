"""
ode_engine.py  (v27)
====================
BioSynthetica's OWN simulation engine: models written as equations or reactions, solved with SciPy - no Tellurium,
no external simulator. The AI (or a person) can build a new model for any system - a gene circuit, a signalling
pathway, a bacterium under an antibiotic, a virus and the immune response - and run it.

    spec = {
      "name": "bacteria under an antibiotic",
      "species":    {"S": 1e6, "R": 10},                      # initial amounts
      "parameters": {"r": 0.8, "K": 1e9, "kmax": 2.0, "MIC": 1.0, "C": 4.0, "mu": 1e-7},
      "equations":  {"S": "r*S*(1-(S+R)/K) - kmax*C/(MIC+C)*S - mu*S",           # d(species)/dt
                     "R": "r*0.9*R*(1-(S+R)/K) + mu*S"},
      # or "reactions": [{"rate": "k*A*B", "change": {"A": -1, "B": -1, "C": 1}}, ...]
      "events": [{"at": 24, "set": {"C": 0}}],                # optional: change a parameter / species at a time
      "t_end": 72, "points": 241, "time_unit": "h",
      "stochastic": false                                     # true: exact Gillespie simulation of the reactions
    }
    run(spec) -> {"time", "series", "summary", "numerically_ok", ...}   (the same shape as simulator.Simulator.run)

Safety: an expression may only use numbers, the model's own names, + - * / ** and the functions in FUNCS - anything
else (attributes, calls to other functions, comprehensions, names it does not declare) is refused before it runs.
Limits: 40 species, 80 parameters, 200 reactions, 20,000 output points, 2 million Gillespie events.
"""
from __future__ import annotations

import ast
import math
import time
from typing import Any, Callable, Dict, List, Optional

import numpy as np

ENGINE_VERSION = "1.0"
FUNCS = {"exp": np.exp, "log": np.log, "log10": np.log10, "sqrt": np.sqrt, "sin": np.sin, "cos": np.cos, "tanh": np.tanh,
         "abs": np.abs, "min": np.minimum, "max": np.maximum, "hill": lambda x, k, n: x ** n / (k ** n + x ** n),
         "pos": lambda x: np.maximum(x, 0.0), "step": lambda x: np.where(x >= 0, 1.0, 0.0)}
_OK_NODES = (ast.Expression, ast.BinOp, ast.UnaryOp, ast.Constant, ast.Name, ast.Load, ast.Call, ast.Add, ast.Sub,
             ast.Mult, ast.Div, ast.Pow, ast.USub, ast.UAdd, ast.Mod, ast.Compare, ast.Gt, ast.Lt, ast.GtE, ast.LtE,
             ast.IfExp, ast.BoolOp, ast.And, ast.Or)
MAX_SPECIES, MAX_PARAMS, MAX_REACTIONS, MAX_POINTS, MAX_EVENTS = 40, 80, 200, 20000, 2_000_000


class ModelError(ValueError):
    pass


def _compile(expr: str, names: set) -> Callable[[Dict[str, Any]], Any]:
    if not isinstance(expr, str) or not expr.strip() or len(expr) > 2000:
        raise ModelError(f"an equation must be a short text expression, not {expr!r}")
    try:
        tree = ast.parse(expr.replace("^", "**"), mode="eval")
    except SyntaxError as exc:
        raise ModelError(f"cannot read the expression '{expr}': {exc.msg}")
    for node in ast.walk(tree):
        if not isinstance(node, _OK_NODES):
            raise ModelError(f"'{expr}': '{type(node).__name__}' is not allowed in a model expression")
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name) or node.func.id not in FUNCS or node.keywords:
                raise ModelError(f"'{expr}': only the functions {', '.join(sorted(FUNCS))} may be used")
        if isinstance(node, ast.Name) and node.id not in names and node.id not in FUNCS and node.id != "t":
            raise ModelError(f"'{expr}': '{node.id}' is not a species or parameter of this model")
        if isinstance(node, ast.Constant) and not isinstance(node.value, (int, float)):
            raise ModelError(f"'{expr}': only numbers may appear as constants")
    code = compile(tree, "<model>", "eval")

    def f(env: Dict[str, Any]) -> Any:
        return eval(code, {"__builtins__": {}}, {**FUNCS, **env})   # noqa: S307 - the tree was whitelisted above
    return f


def validate(spec: Dict[str, Any]) -> Dict[str, Any]:
    """Every problem with a model before it runs. Returns the normalised spec or raises ModelError."""
    if not isinstance(spec, dict):
        raise ModelError("the model must be a JSON object")
    sp = spec.get("species") or {}
    pa = spec.get("parameters") or {}
    if not isinstance(sp, dict) or not sp:
        raise ModelError("the model needs 'species' with initial amounts, e.g. {\"X\": 100}")
    if not isinstance(pa, dict):
        raise ModelError("'parameters' must be an object of name: number")
    if len(sp) > MAX_SPECIES or len(pa) > MAX_PARAMS:
        raise ModelError(f"at most {MAX_SPECIES} species and {MAX_PARAMS} parameters")
    names = set()
    for k, v in list(sp.items()) + list(pa.items()):
        if not isinstance(k, str) or not k.isidentifier() or k in FUNCS or k == "t":
            raise ModelError(f"'{k}' is not a usable name (letters, digits and _; not a function name or 't')")
        if k in names:
            raise ModelError(f"'{k}' is both a species and a parameter")
        names.add(k)
        try:
            fv = float(v)
        except (TypeError, ValueError):
            raise ModelError(f"'{k}' must be a number, not {v!r}")
        if not math.isfinite(fv):
            raise ModelError(f"'{k}' must be a finite number")
    eqs, rx = spec.get("equations"), spec.get("reactions")
    if not eqs and not rx:
        raise ModelError("the model needs 'equations' (d species / dt) or 'reactions'")
    if eqs is not None and not isinstance(eqs, dict):
        raise ModelError("'equations' must be an object: species -> expression")
    if eqs:
        for k in eqs:
            if k not in sp:
                raise ModelError(f"equation for '{k}', which is not a species")
    if rx is not None:
        if not isinstance(rx, list) or len(rx) > MAX_REACTIONS:
            raise ModelError(f"'reactions' must be a list of at most {MAX_REACTIONS}")
        for r in rx:
            if not isinstance(r, dict) or "rate" not in r or not isinstance(r.get("change"), dict):
                raise ModelError("each reaction needs 'rate' (expression) and 'change' ({species: +n / -n})")
            for k, v in r["change"].items():
                if k not in sp or not isinstance(v, (int, float)):
                    raise ModelError(f"reaction changes '{k}', which is not a species (or by a non-number)")
    t_end = float(spec.get("t_end") or 100)
    points = int(spec.get("points") or 201)
    if not (0 < t_end <= 1e7):
        raise ModelError("t_end must be positive")
    if not (2 <= points <= MAX_POINTS):
        raise ModelError(f"points must be between 2 and {MAX_POINTS}")
    for e in spec.get("events") or []:
        if not isinstance(e, dict) or not isinstance(e.get("set"), dict):
            raise ModelError("each event needs 'at' (time) and 'set' ({name: value})")
        float(e.get("at"))
        for k in e["set"]:
            if k not in names:
                raise ModelError(f"event sets '{k}', which is not in the model")
    return {**spec, "species": sp, "parameters": pa, "t_end": t_end, "points": points}


def _summ(t: np.ndarray, y: np.ndarray) -> Dict[str, Optional[float]]:
    from simulator import _summ as s       # the same metrics as the model library (max, min_late, auc, ...)
    return s(list(map(float, t)), list(map(float, y)))


def _derivs(spec: Dict[str, Any]):
    sp, pa = list(spec["species"]), dict(spec["parameters"])
    names = set(sp) | set(pa)
    if spec.get("equations"):
        fs = {k: _compile(spec["equations"][k], names) for k in spec["equations"]}

        def rhs(t: float, y: np.ndarray, params: Dict[str, float]) -> np.ndarray:
            env = {**params, **dict(zip(sp, y)), "t": t}
            return np.array([float(fs[k](env)) if k in fs else 0.0 for k in sp])
        return rhs
    rates = [(_compile(r["rate"], names), r["change"]) for r in spec["reactions"]]

    def rhs(t: float, y: np.ndarray, params: Dict[str, float]) -> np.ndarray:  # noqa: F811
        env = {**params, **dict(zip(sp, y)), "t": t}
        dy = np.zeros(len(sp))
        for f, ch in rates:
            v = float(f(env))
            for k, n in ch.items():
                dy[sp.index(k)] += n * v
        return dy
    return rhs


def run(spec: Dict[str, Any], overrides: Optional[Dict[str, float]] = None, seed: Optional[int] = None) -> Dict[str, Any]:
    """Simulate. `overrides` changes parameters / initial amounts for this run (sweeps, Monte Carlo)."""
    s = validate(spec)
    sp = list(s["species"])
    params = {k: float(v) for k, v in s["parameters"].items()}
    y0 = np.array([float(s["species"][k]) for k in sp])
    for k, v in (overrides or {}).items():
        if k in params:
            params[k] = float(v)
        elif k in sp:
            y0[sp.index(k)] = float(v)
        else:
            raise ModelError(f"'{k}' is not in the model")
    t_end, points = s["t_end"], s["points"]
    t_eval = np.linspace(0.0, t_end, points)
    started = time.perf_counter()
    if s.get("stochastic"):
        if not s.get("reactions"):
            raise ModelError("a stochastic run needs 'reactions'")
        T, Y = _gillespie(s, y0, params, t_eval, seed)
    else:
        T, Y = _integrate(s, y0, params, t_eval)
    series = {k: [float(v) for v in Y[:, i]] for i, k in enumerate(sp)}
    bad = [k for k, v in series.items() if any(not math.isfinite(x) for x in v)]
    return {"model": s.get("name") or "custom model", "engine": f"BioSynthetica ode_engine {ENGINE_VERSION}",
            "time": [float(x) for x in T], "time_unit": s.get("time_unit") or "", "series": series,
            "summary": {k: _summ(T, Y[:, i]) for i, k in enumerate(sp)}, "parameters": params,
            "initial": dict(zip(sp, map(float, y0))), "numerically_ok": not bad, "nonfinite": bad,
            "stochastic": bool(s.get("stochastic")), "seed": seed, "seconds": round(time.perf_counter() - started, 4),
            "basis": "in silico - a model, not evidence"}


def _integrate(s: Dict[str, Any], y0: np.ndarray, params: Dict[str, float], t_eval: np.ndarray):
    from scipy.integrate import solve_ivp
    rhs = _derivs(s)
    events = sorted(s.get("events") or [], key=lambda e: float(e["at"]))
    sp = list(s["species"])
    t0, y, Ts, Ys = 0.0, y0.copy(), [], []
    p = dict(params)
    cuts = [float(e["at"]) for e in events if 0 < float(e["at"]) < t_eval[-1]] + [float(t_eval[-1])]
    ei = 0
    for t1 in cuts:
        mask = (t_eval >= t0) & (t_eval <= t1)
        te = t_eval[mask]
        if t1 > t0:
            sol = solve_ivp(lambda t, yy: rhs(t, yy, p), (t0, t1), y, method="LSODA", t_eval=te if len(te) else None,
                            rtol=1e-7, atol=1e-9)
            if not sol.success:
                raise ModelError(f"the solver failed: {sol.message}")
            if len(te):
                Ts.append(sol.t)
                Ys.append(sol.y.T)
            y = sol.y[:, -1].copy() if sol.y.size else y
        while ei < len(events) and float(events[ei]["at"]) <= t1 + 1e-12:
            for k, v in events[ei]["set"].items():
                if k in p:
                    p[k] = float(v)
                else:
                    y[sp.index(k)] = float(v)
            ei += 1
        t0 = t1
    T = np.concatenate(Ts) if Ts else t_eval
    Y = np.vstack(Ys) if Ys else np.tile(y0, (len(t_eval), 1))
    _, idx = np.unique(T, return_index=True)
    return T[idx], Y[idx]


def _gillespie(s: Dict[str, Any], y0: np.ndarray, params: Dict[str, float], t_eval: np.ndarray, seed: Optional[int]):
    """Exact stochastic simulation (Gillespie's direct method) - for small populations, where chance matters (the
    first resistant mutant, extinction of an infection)."""
    rng = np.random.default_rng(seed)
    sp = list(s["species"])
    names = set(sp) | set(params)
    rx = [(_compile(r["rate"], names), np.array([r["change"].get(k, 0) for k in sp], float)) for r in s["reactions"]]
    events = sorted(s.get("events") or [], key=lambda e: float(e["at"]))
    p = dict(params)
    y = np.round(y0).astype(float)
    t, i_out, n_ev, ei = 0.0, 0, 0, 0
    Y = np.zeros((len(t_eval), len(sp)))
    while i_out < len(t_eval):
        env = {**p, **dict(zip(sp, y)), "t": t}
        a = np.array([max(0.0, float(f(env))) for f, _ in rx])
        a0 = a.sum()
        dt = rng.exponential(1.0 / a0) if a0 > 0 else math.inf
        t_next = t + dt
        nxt_event = float(events[ei]["at"]) if ei < len(events) else math.inf
        while i_out < len(t_eval) and t_eval[i_out] <= min(t_next, nxt_event):
            Y[i_out] = y
            i_out += 1
        if nxt_event <= t_next and nxt_event <= t_eval[-1]:
            for k, v in events[ei]["set"].items():
                if k in p:
                    p[k] = float(v)
                else:
                    y[sp.index(k)] = float(v)
            ei += 1
            t = nxt_event
            continue
        if not math.isfinite(t_next):
            Y[i_out:] = y
            break
        j = int(np.searchsorted(np.cumsum(a), rng.random() * a0))
        y = np.maximum(y + rx[min(j, len(rx) - 1)][1], 0)
        t = t_next
        n_ev += 1
        if n_ev > MAX_EVENTS:
            raise ModelError("the stochastic run exceeded 2 million events - use the deterministic run (stochastic: "
                             "false) for large populations")
    return t_eval, Y
