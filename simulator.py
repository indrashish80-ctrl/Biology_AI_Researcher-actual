"""
simulator.py  (v25)
===================
A real simulator for the virtual lab: SBML models run by libRoadRunner (through Tellurium / Antimony), loaded from
the model library (model_library/*.json).

    lib = ModelLibrary()                      # every card, validated
    sim = Simulator(lib.get("pk_oral_1cmt"))
    out = sim.run({"dose": 1000}, t_end=168, points=337, observables=["C_uM"], seed=1)
    out["series"]["C_uM"], out["summary"]["C_uM"] -> {'final','max','min','auc','time_to_max'}
    sim.sbml()                                # SBML export of the exact model that ran

Solver: CVODE (stiff, adaptive) with rtol 1e-8 / atol 1e-10 by default; 'gillespie' runs the stochastic version with
the given seed. Every run returns the model version (SHA-256 of the card + Antimony source), the seed, the solver
and its tolerances, so experiment_log can store it and a run can be repeated exactly.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

SIM_VERSION = "1.0"
LIB_DIR = Path(__file__).resolve().parent / "model_library"
_LOCK = threading.Lock()


class SimulatorUnavailable(RuntimeError):
    pass


def backend() -> Dict[str, Any]:
    try:
        import tellurium as te  # noqa: F401
        import roadrunner
        return {"available": True, "engine": "tellurium/libroadrunner", "roadrunner": roadrunner.__version__}
    except Exception as exc:
        return {"available": False, "engine": None, "error": f"{type(exc).__name__}: {exc}",
                "install": "pip install tellurium"}


class ModelLibrary:
    def __init__(self, folder: Optional[Path] = None):
        self.folder = Path(folder or LIB_DIR)
        self.cards: Dict[str, Dict[str, Any]] = {}
        self.problems: Dict[str, List[str]] = {}
        for p in sorted(self.folder.glob("*.json")):
            try:
                c = json.loads(p.read_text(encoding="utf-8"))
            except ValueError as exc:
                self.problems[p.name] = [f"invalid JSON: {exc}"]
                continue
            errs = validate_card(c)
            if errs:
                self.problems[c.get("id") or p.name] = errs
            self.cards[c["id"]] = c

    def get(self, model_id: str) -> Dict[str, Any]:
        if model_id not in self.cards:
            raise KeyError(f"unknown model '{model_id}' (library: {', '.join(sorted(self.cards))})")
        return self.cards[model_id]

    def catalogue(self) -> List[Dict[str, Any]]:
        return [{"id": c["id"], "title": c["title"], "domain": c.get("domain"), "version": c.get("version"),
                 "observables": c.get("observables"), "time_unit": c.get("time_unit"), "max_time": c.get("max_time"),
                 "parameters": {k: {kk: v[kk] for kk in ("default", "min", "max", "unit", "meaning")} for k, v in c["parameters"].items()},
                 "assumptions": c.get("assumptions"), "limits": c.get("limits"), "references": c.get("references")}
                for c in self.cards.values()]


def validate_card(c: Dict[str, Any]) -> List[str]:
    errs = []
    for k in ("id", "title", "version", "antimony", "parameters", "observables", "assumptions", "limits", "time_unit", "max_time"):
        if not c.get(k):
            errs.append(f"missing '{k}'")
    for name, p in (c.get("parameters") or {}).items():
        for k in ("default", "min", "max", "unit", "meaning"):
            if k not in p:
                errs.append(f"parameter {name}: missing '{k}'")
        if all(k in p for k in ("default", "min", "max")) and not (p["min"] <= p["default"] <= p["max"]):
            errs.append(f"parameter {name}: default {p['default']} outside [{p['min']}, {p['max']}]")
        if c.get("antimony") and not re.search(rf"(?<![\w]){re.escape(name)}\s*=", c["antimony"]):
            errs.append(f"parameter {name} is not assigned in the Antimony model")
    return errs


def model_hash(card: Dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(card, sort_keys=True).encode()).hexdigest()[:16]


def _summ(t: List[float], y: List[float]) -> Dict[str, Optional[float]]:
    if not y:
        return {}
    auc = 0.0
    for i in range(1, len(t)):
        auc += (t[i] - t[i - 1]) * (y[i] + y[i - 1]) / 2
    j = max(range(len(y)), key=lambda i: y[i])
    late = [v for tt, v in zip(t, y) if tt >= t[0] + 0.75 * (t[-1] - t[0])] or y
    # 'min' covers the whole run (it includes the start, e.g. C = 0 before the first dose); 'min_late' / 'mean_late'
    # cover the last quarter - the steady-state trough / level a dosing question usually means
    return {"final": y[-1], "max": y[j], "min": min(y), "auc": auc, "time_to_max": t[j], "mean": sum(y) / len(y),
            "min_late": min(late), "mean_late": sum(late) / len(late), "max_late": max(late)}


class Simulator:
    def __init__(self, card: Dict[str, Any]):
        b = backend()
        if not b["available"]:
            raise SimulatorUnavailable(f"the simulator needs Tellurium ({b['error']}); install it with: {b['install']}")
        import tellurium as te
        self.card = card
        self.version = model_hash(card)
        with _LOCK:                          # antimony's parser is not thread-safe
            self.rr = te.loada(card["antimony"])
        self.global_params = set(self.rr.model.getGlobalParameterIds())
        self.species = set(self.rr.model.getFloatingSpeciesIds())

    def sbml(self) -> str:
        return self.rr.getSBML()

    def run(self, params: Optional[Dict[str, Any]] = None, t_end: Optional[float] = None, points: int = 201,
            observables: Optional[Sequence[str]] = None, seed: Optional[int] = None, solver: str = "cvode",
            rtol: float = 1e-8, atol: float = 1e-10) -> Dict[str, Any]:
        card = self.card
        t_end = float(t_end if t_end is not None else min(card["max_time"], 100))
        obs = list(observables or list(card["observables"]))
        with _LOCK:
            # parameters first, then a reset that re-applies initial assignments with them
            self.rr.resetToOrigin()
            for k, v in (params or {}).items():
                if k not in self.global_params and k not in self.species:
                    raise KeyError(f"'{k}' is not a parameter of {card['id']}")
                self.rr.setValue(k, float(v)) if k in self.global_params else self.rr.setValue(f"init([{k}])", float(v))
            self.rr.reset()
            for k, v in (params or {}).items():
                if k in self.global_params:
                    self.rr.setValue(k, float(v))
            if solver == "gillespie":
                self.rr.setIntegrator("gillespie")
                if seed is not None:
                    self.rr.integrator.setValue("seed", int(seed))
            else:
                self.rr.setIntegrator("cvode")
                self.rr.integrator.setValue("relative_tolerance", rtol)
                self.rr.integrator.setValue("absolute_tolerance", atol)
            sel = ["time"] + obs
            res = self.rr.simulate(0, t_end, int(points), selections=sel)
            cols = list(res.colnames)
            data = {c: [float(x) for x in res[:, i]] for i, c in enumerate(cols)}
            self.rr.setIntegrator("cvode")
        t = data.pop("time")
        series = {}
        for o in obs:
            key = o if o in data else f"[{o}]" if f"[{o}]" in data else None
            series[o] = data.get(key) if key else data.get(o, [])
        bad = [o for o, y in series.items() if any(not math.isfinite(v) for v in y)]
        return {"model": card["id"], "model_version": self.version, "card_version": card.get("version"),
                "time": t, "time_unit": card.get("time_unit"), "series": series,
                "summary": {o: _summ(t, y) for o, y in series.items()},
                "parameters": {**{k: v["default"] for k, v in card["parameters"].items()}, **(params or {})},
                "solver": {"name": solver, "rtol": rtol, "atol": atol, "points": int(points), "t_end": t_end},
                "seed": seed, "numerically_ok": not bad, "nonfinite": bad, "basis": "in silico - a model, not evidence"}
