"""
sim_verifier.py  (v26.2)
========================
Independent in-silico verification of a claim or hypothesis, with the AI kept out of the judgement.

    verify(claim, library=None, router=None, spec=None) -> {...verdict, steps, runs...}

1. TRANSLATE  - an AI turns the claim into ONE experiment on a documented model (experiment_spec.propose); the
                code validator refuses unknown models / parameters, out-of-range values and over-long runs.
                (A person can pass `spec` instead - then no AI is needed.)
2. READ AGAIN - a SECOND, independent AI reading (another provider when the user has several keys) writes what the
                claim predicts - observable, metric, comparison, value - from the claim and the model card ONLY,
                without seeing the first reading. If the two readings differ, the claim is ambiguous and BOTH are
                tested and reported.
3. SIMULATE   - the code (not the AI) runs the model and evaluates each reading:
                  * base run at the claim's conditions;
                  * robustness: Monte Carlo over the biological parameters the claim does not fix (log-normal,
                    CV 30%, clipped to the card's ranges) - the share of plausible worlds in which the claim holds;
                  * structure: the same test on every alternative model of the same system (`family`);
                  * the quick virtual-lab checks (mechanism signs, dose vs human exposure, power).
4. VERDICT    - 'supported by the model (robust)' | 'consistent, but depends on assumptions' |
                'contradicted by the model' | 'ambiguous claim' | 'not testable with the available models'.
                Always labelled IN SILICO: it shows whether the claim is consistent with a documented model, never
                that it is true.
"""
from __future__ import annotations

import math
import random
import re
import time
from typing import Any, Dict, List, Optional

VERIFIER_VERSION = "1.0"
SAME_DIR = {">": ">", ">=": ">", "<": "<", "<=": "<", "==": "==", "between": "between"}


def _norm_test(t: Any, card: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    from experiment_spec import COMPARISONS, METRICS
    if not isinstance(t, dict):
        return None
    obs, met, comp, val = t.get("observable"), t.get("metric", "final"), t.get("comparison"), t.get("value")
    if obs not in card["observables"] or met not in METRICS or comp not in COMPARISONS:
        return None
    try:
        val = [float(val[0]), float(val[1])] if comp == "between" else float(val)
    except (TypeError, ValueError, IndexError, KeyError):
        return None
    return {"observable": obs, "metric": met, "comparison": comp, "value": val}


def same_reading(a: Dict[str, Any], b: Dict[str, Any]) -> bool:
    """Two readings of a claim agree when they test the same output, in the same direction, against a similar
    threshold (within 25%)."""
    if a["observable"] != b["observable"] or a["metric"] != b["metric"] \
            or SAME_DIR[a["comparison"]] != SAME_DIR[b["comparison"]]:
        return False                       # 'peak above 5' and 'trough above 5' are different claims
    va, vb = a["value"], b["value"]
    if isinstance(va, list) or isinstance(vb, list):
        return va == vb
    return abs(va - vb) <= 0.25 * max(abs(va), abs(vb), 1e-12)


def _second_reading_prompt(claim: str, card: Dict[str, Any]) -> str:
    from experiment_spec import METRICS
    obs = ", ".join(f"{o} ({card['observables'][o]})" if isinstance(card.get("observables"), dict) else o
                    for o in card["observables"])
    return ("You are checking what a scientific claim PREDICTS, so that a computer model can test it. Read the claim "
            "and state ONE measurable prediction on the model below. Do not judge whether the claim is true.\n"
            f"MODEL: {card['id']} - {card['title']} (time unit: {card.get('time_unit')})\n"
            f"OUTPUTS: {obs}\nMETRICS: {', '.join(METRICS)} ('min_late' = steady-state trough over the last quarter of "
            "the run, 'max_late' = steady-state peak)\nCOMPARISONS: >, >=, <, <=, between\n"
            'Return ONLY JSON: {"observable": ..., "metric": ..., "comparison": ..., "value": number or [low, high], '
            '"reading": "one sentence: what the claim predicts"}\n\n'
            f"CLAIM: {claim}")


def _other_provider(router: Any) -> Optional[str]:
    try:
        provs = router.get_available_providers()
        first = (router.order_for("hypotheses") or [None])[0]
        others = [p for p in provs if p != first]
        return others[0] if others else None
    except Exception:
        return None


def _evaluate(sim: Any, params: Dict[str, float], test: Dict[str, Any], t_end: float, points: int) -> Optional[bool]:
    from experiment_spec import evaluate_expected
    try:
        r = sim.run(params, t_end=t_end, points=points, observables=[test["observable"]])
    except Exception:
        return None
    if not r.get("numerically_ok"):
        return None
    ev = evaluate_expected(test, r["summary"])
    if ev["value"] is None:
        return None
    return ev["verdict"].startswith("consistent"), ev["value"]


DESIGN_NAMES = {"dose", "tau", "n_doses", "mw", "c_drug", "t_inf", "infusion", "interval"}
DESIGN_UNITS = {"count", "mg", "g/mol", "mg/kg"}


def _is_design(name: str, p: Dict[str, Any]) -> bool:
    """A choice of the experiment (dose, interval, number of doses, drug concentration) or a constant of the drug
    (molecular weight) - not uncertain biology, so it is not varied in the robustness check."""
    return name.lower() in DESIGN_NAMES or str(p.get("unit") or "").lower() in DESIGN_UNITS


def _robustness(sim: Any, card: Dict[str, Any], params: Dict[str, float], test: Dict[str, Any], t_end: float,
                points: int, n: int, seed: int, fixed: set) -> Dict[str, Any]:
    rng = random.Random(seed)
    free = [k for k, v in card["parameters"].items()
            if k not in fixed and (v.get("default") or 0) > 0 and not _is_design(k, v)]
    held, total, vals = 0, 0, []
    for _ in range(n):
        p = dict(params)
        for k in free:
            mu = float(params.get(k, card["parameters"][k]["default"]))
            s = math.sqrt(math.log(1 + 0.3 ** 2))
            v = mu * math.exp(rng.gauss(-s * s / 2, s))
            p[k] = min(max(v, card["parameters"][k]["min"]), card["parameters"][k]["max"])
        out = _evaluate(sim, p, test, t_end, points)
        if out is None:
            continue
        total += 1
        held += int(out[0])
        vals.append(out[1])
    vals.sort()
    return {"n": total, "holds": held, "share": round(held / total, 3) if total else None, "varied": free,
            "range": [vals[int(0.025 * (len(vals) - 1))], vals[int(0.975 * (len(vals) - 1))]] if vals else None}


def _structures(library: Any, card: Dict[str, Any], params: Dict[str, float], test: Dict[str, Any], t_end: float,
                points: int) -> List[Dict[str, Any]]:
    import simulator as S
    fam = card.get("family")
    out = []
    if not fam:
        return out
    for mid, other in library.cards.items():
        if mid == card["id"] or other.get("family") != fam or test["observable"] not in other["observables"]:
            continue
        p = {k: v for k, v in params.items() if k in other["parameters"]}
        res = _evaluate(S.Simulator(other), p, test, min(t_end, other["max_time"]), points)
        out.append({"model": mid, "holds": None if res is None else res[0], "value": None if res is None else res[1]})
    return out


def _grade(base: Optional[bool], rob: Dict[str, Any], structs: List[Dict[str, Any]]) -> str:
    share = rob.get("share")
    struct_ok = [s["holds"] for s in structs if s["holds"] is not None]
    if base is None:
        return "not testable with the available models"
    if base and (share is None or share >= 0.9) and all(struct_ok):
        return "supported by the model (robust)"
    if not base and (share is None or share <= 0.1) and not any(struct_ok):
        return "contradicted by the model"
    return "consistent, but depends on assumptions" if base else "contradicted at the stated conditions, but depends on assumptions"


def verify(claim: str, library: Any = None, router: Any = None, spec: Optional[Dict[str, Any]] = None,
           n_mc: int = 40, seed: int = 1, time_budget_s: float = 60.0) -> Dict[str, Any]:
    import experiment_spec as es
    import simulator as S
    t0 = time.time()
    claim = re.sub(r"\s+", " ", str(claim or "")).strip()
    res: Dict[str, Any] = {"claim": claim, "version": VERIFIER_VERSION, "steps": [], "basis":
                           "in silico - shows whether the claim is consistent with a documented model, not that it is true"}
    if not claim and not spec:
        res["verdict"] = "not testable with the available models"
        res["steps"].append("no claim given")
        return res
    # quick checks (no simulator, no AI): mechanism signs, dose vs human exposure, power
    try:
        import virtual_lab as vl
        res["quick_checks"] = vl.check_hypothesis({"statement": claim}, human_cmax_um=vl.human_cmax_for(claim))
    except Exception as exc:
        res["quick_checks"] = {"verdict": "not run", "error": str(exc)[:160]}
    if not S.backend()["available"]:
        res["verdict"] = res["quick_checks"].get("verdict") or "not testable with the available models"
        res["steps"].append("simulator not installed (pip install -r requirements-lab.txt): quick checks only")
        return res
    library = library or S.ModelLibrary()
    # 1. translate
    if spec is None:
        if router is None:
            res["verdict"] = "not testable with the available models"
            res["steps"].append("translating a claim into an experiment needs an AI key (or pass a spec)")
            return res
        try:
            prop = es.propose(claim, library, router)
        except Exception as exc:              # the AI could not be reached: reported, never a crash of the run
            res["verdict"] = "not testable with the available models"
            res["steps"].append(f"the AI could not translate the claim ({type(exc).__name__}: {str(exc)[:120]})")
            return res
        spec = prop.get("proposed")
        res["steps"].append("experiment proposed by the AI (reader 1)")
    val = es.validate(spec, library) if spec else {"valid": False, "errors": ["no experiment could be proposed"]}
    res["spec"] = val.get("spec")
    if not val.get("valid"):
        res["verdict"] = "not testable with the available models"
        res["errors"] = val.get("errors")
        res["steps"].append("the validator refused the experiment: " + "; ".join(val.get("errors") or [])[:300])
        return res
    s = val["spec"]
    card = library.get(s["model"])
    sim = S.Simulator(card)
    params = dict(s.get("parameters") or {})
    t_end, points = float(s.get("t_end") or min(card["max_time"], 100)), int(s.get("points") or 201)
    readings: List[Dict[str, Any]] = []
    t1 = _norm_test(s.get("expected"), card)
    if t1:
        readings.append({**t1, "reader": "AI reader 1 (with the experiment)"})
    # 2. independent second reading
    if router is not None:
        try:
            from llm_json import parse_json_response
            other = _other_provider(router)
            import contextlib
            ctx = router.prefer(other) if other and hasattr(router, "prefer") else contextlib.nullcontext()
            with ctx:
                raw = router.ask(_second_reading_prompt(claim, card), task="hypotheses")
            t2 = _norm_test(parse_json_response(raw), card)
            if t2:
                readings.append({**t2, "reader": f"AI reader 2 (independent{', ' + other if other else ''})"})
                res["steps"].append("an independent second AI reading stated what the claim predicts")
        except Exception as exc:
            res["steps"].append(f"second reading failed ({type(exc).__name__}) - one reading only")
    if not readings:
        res["verdict"] = "not testable with the available models"
        res["steps"].append("no measurable prediction could be stated for this model")
        return res
    agree = len(readings) < 2 or same_reading(readings[0], readings[1])
    res["readings_agree"] = agree
    tests = readings[:1] if agree else readings
    # 3. simulate (the code decides)
    out = []
    for t in tests:
        base = _evaluate(sim, params, t, t_end, points)
        budget_left = time_budget_s - (time.time() - t0)
        rob = _robustness(sim, card, params, t, t_end, points, n_mc if budget_left > 10 else 10, seed,
                          set(params)) if base is not None else {}
        structs = _structures(library, card, params, t, t_end, points) if base is not None else []
        test_txt = f"{t['observable']} {t['metric']} {t['comparison']} {t['value']}"
        out.append({"reader": t["reader"], "test": test_txt, "base_holds": None if base is None else base[0],
                    "base_value": None if base is None else round(base[1], 6), "robustness": rob,
                    "structures": structs, "verdict": _grade(None if base is None else base[0], rob, structs)})
    res["tests"] = out
    res["model"] = {"id": card["id"], "title": card["title"], "version": sim.version,
                    "assumptions": card.get("assumptions"), "limits": card.get("limits")}
    if not agree:
        res["verdict"] = "ambiguous claim"
        res["steps"].append("the two readings of the claim differ - both were tested; make the claim precise")
    else:
        res["verdict"] = out[0]["verdict"]
    res["seconds"] = round(time.time() - t0, 1)
    return res


def markdown(results: List[Dict[str, Any]]) -> str:
    if not results:
        return ""
    L = ["## Simulation checks of the hypotheses (in silico - a model, not evidence)", "",
         "Each claim was turned into an experiment on a documented model by one AI, read again independently by a "
         "second AI, and judged by the code: a base run, robustness over the biology the claim does not fix (Monte "
         "Carlo), and alternative model structures.", ""]
    for r in results:
        L.append(f"- **{r.get('verdict')}** - {r.get('claim', '')[:220]}")
        for t in r.get("tests") or []:
            rob = t.get("robustness") or {}
            share = f"{rob['share']*100:.0f}% of {rob['n']} plausible parameter sets" if rob.get("share") is not None else "n/a"
            st = ", ".join(f"{s['model']}: {'holds' if s['holds'] else 'fails' if s['holds'] is False else 'n/a'}"
                           for s in t.get("structures") or [])
            L.append(f"  - {t['reader']}: test `{t['test']}` on {r.get('model', {}).get('id')} -> base "
                     f"{'holds' if t['base_holds'] else 'fails' if t['base_holds'] is False else 'n/a'} "
                     f"(value {t.get('base_value')}); holds in {share}" + (f"; other structures: {st}" if st else ""))
        if r.get("errors"):
            L.append("  - not testable: " + "; ".join(r["errors"])[:240])
    L.append("")
    return "\n".join(L)


WANTS_VERIFY = re.compile(r"\b(verif\w*|validat\w*|simulat\w*|in[\s-]?silico|test (?:the|my|this) (?:hypothes\w*|claim)|"
                          r"check (?:the|my|this) (?:hypothes\w*|claim))\b", re.I)


def wants_verification(question: str) -> bool:
    """The user asked for the virtual lab to check a hypothesis ('verify ...', 'simulate ...', 'in silico ...')."""
    return bool(WANTS_VERIFY.search(question or ""))
