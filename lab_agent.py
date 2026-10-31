"""
lab_agent.py  (v26)
===================
The AUTONOMOUS EXPERIMENT LOOP of the virtual lab:

    formulate hypothesis -> design a controlled experiment -> validate and execute -> analyse the numbers
    -> compare predictions with outcomes -> choose the next experiment or stop

The AI (when a key is given) may only PROPOSE: a goal, a model and competing hypotheses. Every experiment is a spec that
experiment_spec.validate() accepts, and every run is logged (experiment_log), including failed and rejected ones.

Two loop types:

1 DISCRIMINATE (Bayesian experimental design). Competing hypotheses H1..Hk, each a parameter set and/or a model
  structure from the library, start with prior probabilities. A design is a setting of one controllable parameter
  (e.g. dose, drug concentration) and a measurement (observable + metric + time). For every candidate design the
  agent simulates what each hypothesis predicts and computes the EXPECTED INFORMATION GAIN about which hypothesis is
  true, under Gaussian measurement noise (relative SD `noise`, Monte Carlo over the predictive mixture). It runs the
  most informative design, obtains the outcome, updates the posterior (Bayes' rule) and repeats.
  Where the outcome comes from:
    * `observations`: measured data you supply ({design key: value}) - the loop then tells you which REAL experiment
      to run next whenever a needed observation is missing (it stops with 'needs data' and the recommended design);
    * `reference_model`: an in-silico ground truth (for testing the method, labelled as such in the log).
2 EXPLORE. One model, one question ('does parameter X drive observable Y?'): base run, Morris sensitivity, the
  expected-result test, then - if the result is near the decision boundary - a threshold search by bisection.

Stopping rules (every one is logged with its reason):
    the posterior of one hypothesis >= `stop_posterior` (default 0.95);
    no remaining design has expected information gain >= `min_gain` (default 0.02 bits) - further experiments are
    unlikely to be informative;
    the budget of experiments / simulations / wall time is spent;
    the model is invalid for the question (validator rejection, non-finite results) - the loop stops rather than
    report a number from a broken model.
"""
from __future__ import annotations

import math
import random
import time
from typing import Any, Dict, List, Optional, Sequence

AGENT_VERSION = "1.0"


def _log2(x: float) -> float:
    return math.log(x, 2) if x > 0 else 0.0


def _norm_pdf(x: float, mu: float, sd: float) -> float:
    return math.exp(-0.5 * ((x - mu) / sd) ** 2) / (sd * math.sqrt(2 * math.pi))


def expected_information_gain(preds: Sequence[float], prior: Sequence[float], noise: float, n_mc: int = 400,
                              seed: int = 1, floor: float = 1e-9) -> float:
    """EIG (bits) of observing y ~ N(pred_true, (noise*|pred|)^2) about which hypothesis is true."""
    rng = random.Random(seed)
    sds = [max(abs(p) * noise, floor) for p in preds]
    h0 = -sum(p * _log2(p) for p in prior if p > 0)
    acc = 0.0
    for _ in range(n_mc):
        i = rng.choices(range(len(preds)), weights=prior)[0]
        y = rng.gauss(preds[i], sds[i])
        like = [_norm_pdf(y, mu, sd) for mu, sd in zip(preds, sds)]
        z = sum(pr * li for pr, li in zip(prior, like))
        if z <= 0:
            continue
        post = [pr * li / z for pr, li in zip(prior, like)]
        acc += -sum(p * _log2(p) for p in post if p > 0)
    return max(0.0, h0 - acc / n_mc)


def posterior(prior: Sequence[float], preds: Sequence[float], y: float, noise: float, floor: float = 1e-9) -> List[float]:
    like = [_norm_pdf(y, mu, max(abs(mu) * noise, floor)) for mu in preds]
    z = sum(p * l for p, l in zip(prior, like))
    if z <= 0:
        return list(prior)
    return [p * l / z for p, l in zip(prior, like)]


def _norm_obs(observations: Dict[str, Any], parameter: str) -> Dict[str, float]:
    """'dose=2000', 'dose = 2000.0' and '2000' all name the same design."""
    out: Dict[str, float] = {}
    for k, v in (observations or {}).items():
        try:
            val = float(str(k).split("=", 1)[-1])
            out[f"{parameter}={val:g}"] = float(v)
        except (TypeError, ValueError):
            continue
    return out


def validate_goal(goal: Any, library: Any) -> List[str]:
    """Every problem with a goal, BEFORE anything runs (types, names, ranges, budgets)."""
    from experiment_spec import METRICS
    if not isinstance(goal, dict):
        return ["the goal must be a JSON object"]
    errs: List[str] = []
    for k in ("max_experiments", "max_simulations", "max_seconds", "noise", "stop_posterior", "min_gain", "t_end", "points"):
        if k in goal:
            try:
                v = float(goal[k])
                if not math.isfinite(v) or v < 0:
                    raise ValueError
            except (TypeError, ValueError):
                errs.append(f"'{k}' must be a non-negative number")
    if "noise" in goal and not errs and float(goal["noise"]) <= 0:
        errs.append("'noise' must be > 0 (a measurement without noise cannot be compared probabilistically)")
    if goal.get("metric") is not None and goal.get("metric") not in METRICS:
        errs.append(f"metric must be one of {METRICS}")
    kind = goal.get("type")
    if kind not in ("discriminate", "explore"):
        return ["goal.type must be 'discriminate' or 'explore'"]
    if kind == "discriminate":
        hs = goal.get("hypotheses") or []
        if not isinstance(hs, list) or len(hs) < 2:
            errs.append("discriminate needs at least two hypotheses")
        names = set()
        for h in hs if isinstance(hs, list) else []:
            if not isinstance(h, dict) or not h.get("name"):
                errs.append("every hypothesis must be an object with a 'name'")
                continue
            if h["name"] in names:
                errs.append(f"hypothesis name '{h['name']}' is used twice")
            names.add(h["name"])
            m = h.get("model") or goal.get("model")
            if not isinstance(m, str) or m not in library.cards:
                errs.append(f"hypothesis {h.get('name')}: unknown model '{m}'")
                continue
            if goal.get("observable") and goal["observable"] not in library.get(m)["observables"]:
                errs.append(f"hypothesis {h.get('name')}: {m} has no observable '{goal['observable']}'")
            if h.get("prior") is not None and not isinstance(h.get("prior"), (int, float)):
                errs.append(f"hypothesis {h.get('name')}: prior must be a number")
            if h.get("parameters") is not None and not isinstance(h.get("parameters"), dict):
                errs.append(f"hypothesis {h.get('name')}: parameters must be an object")
                continue
            for k in (h.get("parameters") or {}):
                if k not in library.get(m)["parameters"]:
                    errs.append(f"hypothesis {h.get('name')}: '{k}' is not a parameter of {m}")
        d = goal.get("design") if isinstance(goal.get("design"), dict) else {}
        if isinstance(d.get("values"), list) and not all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in d["values"]):
            errs.append("design values must be numbers")
        if not d.get("parameter") or not isinstance(d.get("values"), list) or not d["values"]:
            errs.append("design needs 'parameter' and a non-empty list of 'values'")
        if not goal.get("observable") or not goal.get("metric"):
            errs.append("goal needs 'observable' and 'metric'")
        if len(d.get("values") or []) > 60:
            errs.append("at most 60 candidate designs")
    else:
        m = goal.get("model")
        if not isinstance(m, str) or m not in library.cards:
            errs.append(f"unknown model '{m}'")
        elif not goal.get("observable") or goal["observable"] not in library.get(m)["observables"]:
            errs.append(f"goal needs an 'observable' of {m}: {sorted(library.get(m)['observables'])}")
        elif goal.get("parameter") and goal["parameter"] not in library.get(m)["parameters"]:
            errs.append(f"'{goal['parameter']}' is not a parameter of {m}")
    return errs


def run(goal: Dict[str, Any], library: Any = None, observations: Optional[Dict[str, float]] = None,
        log_runs: bool = True) -> Dict[str, Any]:
    import experiment_spec as es
    import simulator as S
    library = library or S.ModelLibrary()
    errs = validate_goal(goal, library)
    if errs:
        _log({"status": "rejected", "error": "; ".join(errs), "spec": {"agent": goal}}, log_runs)
        return {"status": "rejected", "errors": errs}
    if goal["type"] == "explore":
        return _explore(goal, library, log_runs)
    budget = {"experiments": int(goal.get("max_experiments", 8)), "simulations": int(goal.get("max_simulations", 1500)),
              "seconds": float(goal.get("max_seconds", 120))}
    noise = float(goal.get("noise", 0.1))
    stop_p = float(goal.get("stop_posterior", 0.95))
    min_gain = float(goal.get("min_gain", 0.02))
    obs, metric = goal["observable"], goal["metric"]
    t_end, points = float(goal.get("t_end", 100)), int(goal.get("points", 201))
    d = goal["design"]
    hyps = goal["hypotheses"]
    prior = [float(h.get("prior", 1.0)) for h in hyps]
    s = sum(prior) or 1.0
    post = [p / s for p in prior]
    sims = {}
    for h in hyps:
        m = h.get("model") or goal.get("model")
        sims.setdefault(m, S.Simulator(library.get(m)))
    t0 = time.time()
    n_sim = 0
    preds: Dict[float, List[float]] = {}
    log: List[Dict[str, Any]] = []

    def predict(value: float) -> Optional[List[float]]:
        nonlocal n_sim
        if value in preds:
            return preds[value]
        row = []
        for h in hyps:
            m = h.get("model") or goal.get("model")
            card = library.get(m)
            p = {**(goal.get("fixed") or {}), **(h.get("parameters") or {}), d["parameter"]: value}
            p = {k: v for k, v in p.items() if k in card["parameters"]}
            spec = {"model": m, "parameters": p, "t_end": t_end, "points": points, "observables": [obs]}
            v = es.validate(spec, library)
            if not v["valid"]:
                raise ValueError(f"design {d['parameter']}={value} rejected for {h.get('name')}: {'; '.join(v['errors'])}")
            r = sims[m].run(v["spec"]["parameters"], t_end=t_end, points=points, observables=[obs])
            n_sim += 1
            if not r["numerically_ok"]:
                raise ValueError(f"non-finite simulation for {h.get('name')} at {d['parameter']}={value}")
            row.append(float(r["summary"][obs][metric]))
        preds[value] = row
        return row

    done: List[float] = []
    reason = None
    status = "ok"
    ref_sim = None
    if goal.get("reference_model"):
        rm = goal["reference_model"]
        ref_sim = (S.Simulator(library.get(rm.get("model") or goal.get("model"))), rm)
    for step in range(budget["experiments"]):
        if time.time() - t0 > budget["seconds"]:
            reason = "time budget spent"
            break
        if max(post) >= stop_p:
            reason = f"hypothesis '{hyps[post.index(max(post))]['name']}' reached posterior {max(post):.3f} (>= {stop_p})"
            break
        cands = [v for v in d["values"] if v not in done]
        if not cands:
            reason = "every candidate design has been run"
            break
        gains, invalid = [], []
        for v in cands:
            if n_sim + len(hyps) > budget["simulations"]:
                break
            try:
                gains.append((expected_information_gain(predict(v), post, noise, seed=step + 1), v))
            except ValueError as exc:                 # this design is outside the model's validity: skip it, say so
                invalid.append(str(exc))
                done.append(v)
        if invalid:
            log.append({"step": step + 1, "skipped_designs": invalid[:10]})
        if not gains:
            if invalid and len(invalid) == len(cands):
                status, reason = "model_invalid", "no candidate design is valid for the model: " + invalid[0]
            else:
                reason = "simulation budget spent"
            break
        g, best = max(gains)
        if g < min_gain:
            reason = f"no remaining design is informative (best expected gain {g:.3f} bits < {min_gain})"
            break
        key = f"{d['parameter']}={best:g}"
        if observations is not None:
            observations = _norm_obs(observations, d["parameter"])
            if key not in observations:
                status = "needs_data"
                reason = (f"the most informative next REAL experiment is {key}: measure {obs} ({metric}); expected gain "
                          f"{g:.2f} bits")
                log.append({"step": step + 1, "design": key, "expected_gain_bits": round(g, 4), "predictions":
                            dict(zip([h["name"] for h in hyps], preds[best])), "outcome": None})
                break
            y, source = float(observations[key]), "measured data"
        elif ref_sim is not None:
            sim_r, rm = ref_sim
            card = sim_r.card
            p = {k: v for k, v in {**(goal.get("fixed") or {}), **(rm.get("parameters") or {}), d["parameter"]: best}.items()
                 if k in card["parameters"]}
            r = sim_r.run(p, t_end=t_end, points=points, observables=[obs])
            n_sim += 1
            rng = random.Random(1000 + step)
            y0 = float(r["summary"][obs][metric])
            y = y0 + rng.gauss(0, abs(y0) * noise) if goal.get("noisy_reference", True) else y0
            source = "in-silico reference model (method test, not evidence)"
        else:
            status = "needs_data"
            reason = f"no observations or reference model: the recommended experiment is {key} (expected gain {g:.2f} bits)"
            break
        post = posterior(post, preds[best], y, noise)
        done.append(best)
        entry = {"step": step + 1, "design": key, "expected_gain_bits": round(g, 4), "outcome": y, "outcome_source": source,
                 "predictions": dict(zip([h["name"] for h in hyps], preds[best])),
                 "posterior": dict(zip([h["name"] for h in hyps], [round(p, 4) for p in post]))}
        log.append(entry)
        _log({"status": "ok", "model_id": "|".join(sorted(sims)), "parameters": {d["parameter"]: best}, "spec": {"agent": goal.get("name"), "design": key},
              "summary": entry, "n_simulations": len(hyps), "finished": time.time()}, log_runs)
    else:
        reason = reason or "experiment budget spent"
    lead = post.index(max(post))
    return {"status": status, "type": "discriminate", "stop_reason": reason, "steps": log,
            "posterior": dict(zip([h["name"] for h in hyps], [round(p, 4) for p in post])),
            "leading_hypothesis": hyps[lead]["name"], "simulations": n_sim, "seconds": round(time.time() - t0, 2),
            "noise_model": f"Gaussian, relative SD {noise}", "basis": "in silico - which hypothesis the MODEL outcomes favour; "
            "a posterior from simulated outcomes is a test of the method, not evidence about biology",
            "version": AGENT_VERSION}


def _explore(goal: Dict[str, Any], library: Any, log_runs: bool) -> Dict[str, Any]:
    import experiment_spec as es
    import sensitivity as se
    import simulator as S
    card = library.get(goal["model"])
    sim = S.Simulator(card)
    obs, metric = goal["observable"], goal.get("metric", "max")
    t_end, points = float(goal.get("t_end", min(card["max_time"], 100))), int(goal.get("points", 201))
    params = goal.get("parameters") or {}
    steps: List[Dict[str, Any]] = []
    v = es.validate({"model": card["id"], "parameters": params, "t_end": t_end, "points": points, "observables": [obs],
                     **({"expected": goal["expected"]} if goal.get("expected") else {})}, library)
    if not v["valid"]:
        _log({"status": "rejected", "error": "; ".join(v["errors"]), "spec": {"agent": goal}}, log_runs)
        return {"status": "rejected", "errors": v["errors"]}
    base = sim.run(v["spec"]["parameters"], t_end=t_end, points=points, observables=[obs])
    if not base["numerically_ok"]:
        return {"status": "model_invalid", "stop_reason": "non-finite base simulation"}
    steps.append({"step": "base run", "value": base["summary"][obs][metric]})
    names = goal.get("sensitivity_parameters") or [k for k in card["parameters"] if (params.get(k, card["parameters"][k]["default"]) or 0) != 0][:6]
    mo = se.morris(sim, names, obs, metric, r=int(goal.get("trajectories", 6)), seed=int(goal.get("seed", 1)),
                   t_end=t_end, points=min(points, 201), params=params)
    steps.append({"step": "Morris sensitivity", "ranking": mo["ranking"]})
    target_p = goal.get("parameter")
    verdict = None
    if target_p:
        rank = mo["ranking"].index(target_p) + 1 if target_p in mo["ranking"] else None
        verdict = (f"{target_p} is the #{rank} driver of {obs} {metric} out of {len(mo['ranking'])}" if rank else
                   f"{target_p} has no measurable effect on {obs} {metric}")
        steps.append({"step": "hypothesis check", "result": verdict})
    if goal.get("expected"):
        ev = es.evaluate_expected(v["spec"]["expected"], base["summary"])
        steps.append({"step": "expected-result test", **ev})
        e = v["spec"]["expected"]
        val = ev.get("value")
        near = val is not None and isinstance(e.get("value"), (int, float)) and \
            abs(val - e["value"]) <= 0.2 * max(abs(e["value"]), 1e-12)
        if near and target_p:
            lo, hi = card["parameters"][target_p]["min"], card["parameters"][target_p]["max"]
            th = se.threshold(sim, target_p, obs, metric, float(e["value"]), lo, hi, t_end=t_end, points=min(points, 201),
                              params=params)
            steps.append({"step": "threshold search (near the decision boundary)", **{k: th.get(k) for k in ("found", "value", "simulations")}})
    _log({"status": "ok", "model_id": card["id"], "model_version": sim.version, "parameters": params,
          "spec": {"agent": goal.get("name"), "type": "explore"}, "summary": {"steps": steps},
          "n_simulations": 1 + mo["simulations"], "finished": time.time()}, log_runs)
    return {"status": "ok", "type": "explore", "steps": steps, "verdict": verdict,
            "stop_reason": "exploration plan complete", "basis": "in silico - consistency with a model, not evidence",
            "version": AGENT_VERSION}


def propose_goal(hypothesis: str, library: Any, router: Any = None) -> Dict[str, Any]:
    """The AI proposes a goal (model, competing hypotheses, candidate designs); the validator checks it. Nothing runs."""
    import json
    from llm_json import parse_json_response
    cat = [{"id": c["id"], "title": c["title"], "observables": list(c["observables"]),
            "parameters": {k: [v["min"], v["max"], v["unit"]] for k, v in c["parameters"].items()}} for c in library.catalogue()]
    prompt = ("Design an in-silico experiment plan to test the hypothesis below with ONE model of this library. Return ONLY "
              "JSON: {\"type\": \"discriminate\", \"model\": id, \"observable\": ..., \"metric\": \"max|min_late|final|auc|mean_late\", "
              "\"t_end\": number, \"design\": {\"parameter\": name, \"values\": [numbers within range]}, "
              "\"hypotheses\": [{\"name\": \"H1\", \"parameters\": {...}}, {\"name\": \"H2\", \"parameters\": {...}}]}. "
              "Use only listed ids, parameters (within ranges) and observables.\n"
              f"LIBRARY: {json.dumps(cat)[:6000]}\nHYPOTHESIS: {hypothesis}")
    if router is None:
        return {"proposed": None, "prompt": prompt, "note": "no AI key: write the goal yourself or add a key"}
    try:
        goal = parse_json_response(router.ask(prompt, task="planning"))
    except Exception as exc:
        return {"proposed": None, "error": f"{type(exc).__name__}: {exc}"}
    errs = validate_goal(goal or {}, library)
    return {"proposed": goal, "valid": not errs, "errors": errs}


def _log(rec: Dict[str, Any], enabled: bool) -> None:
    if not enabled:
        return
    try:
        import experiment_log as el
        el.log(rec)
    except Exception:
        pass
