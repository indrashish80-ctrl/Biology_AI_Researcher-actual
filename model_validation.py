"""
model_validation.py  (v25)
==========================
Uncertainty and validation: compare a model against HELD-OUT experimental data.

    validate(sim, data, fit=["ke", "ka"], holdout=0.3, seed=1)

    data = {"observable": "C_uM", "time": [...], "value": [...], "sd": [...] (optional), "parameters": {...} (the
            conditions of the experiment, e.g. the dose)}

Steps
  1. split the time points into a TRAINING part and a HELD-OUT part (every k-th point held out, reproducible)
  2. fit the chosen parameters to the training points only (bounded least squares inside the card's ranges, on a log
     scale for positive parameters)
  3. parameter uncertainty from the fit's Jacobian (covariance = s^2 (J'J)^-1), 200 parameter draws, a 95% band
  4. predict the held-out points: RMSE, MAE, R^2 and the share of held-out points inside the 95% band (coverage)
  5. verdict: 'validated' (coverage >= 80% and R^2 >= 0.8), 'partly validated', or 'not validated'

The model is never 'validated' on the points it was fitted to.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Sequence

VALIDATION_VERSION = "1.0"


def _interp(t: List[float], y: List[float], x: float) -> float:
    if x <= t[0]:
        return y[0]
    for i in range(1, len(t)):
        if x <= t[i]:
            f = (x - t[i - 1]) / (t[i] - t[i - 1]) if t[i] > t[i - 1] else 0
            return y[i - 1] + f * (y[i] - y[i - 1])
    return y[-1]


def _predict(sim: Any, params: Dict[str, float], obs: str, times: Sequence[float]) -> List[float]:
    t_end = max(times) * 1.02 if times else 1.0
    out = sim.run(params, t_end=t_end, points=max(101, int(len(times) * 4)), observables=[obs])
    return [_interp(out["time"], out["series"][obs], x) for x in times]


def validate(sim: Any, data: Dict[str, Any], fit: Sequence[str], holdout: float = 0.3, seed: int = 1,
             draws: int = 200) -> Dict[str, Any]:
    import numpy as np
    from scipy.optimize import least_squares
    card = sim.card
    obs = data["observable"]
    T, Y = list(map(float, data["time"])), list(map(float, data["value"]))
    if len(T) != len(Y) or len(T) < 5:
        raise ValueError("need at least 5 (time, value) points of equal length")
    cond = dict(data.get("parameters") or {})
    step = max(2, round(1 / holdout)) if holdout > 0 else 0
    idx = list(range(len(T)))
    test = [i for i in idx if step and i % step == step - 1]
    train = [i for i in idx if i not in test]
    names = list(fit)
    for n in names:
        if n not in card["parameters"]:
            raise KeyError(f"'{n}' is not a parameter of {card['id']}")
    lo = [card["parameters"][n]["min"] for n in names]
    hi = [card["parameters"][n]["max"] for n in names]
    x0 = [card["parameters"][n]["default"] for n in names]
    logs = [l > 0 for l in lo]
    enc = lambda x: [math.log(v) if lg else v for v, lg in zip(x, logs)]            # noqa: E731
    dec = lambda z: [math.exp(v) if lg else v for v, lg in zip(z, logs)]            # noqa: E731
    tt, yt = [T[i] for i in train], [Y[i] for i in train]
    scale = max(1e-12, max(abs(v) for v in yt))

    def resid(z):
        p = {**cond, **dict(zip(names, dec(z)))}
        pred = _predict(sim, p, obs, tt)
        return [(a - b) / scale for a, b in zip(pred, yt)]

    r = least_squares(resid, enc(x0), bounds=(enc(lo), enc(hi)), max_nfev=200)
    best = dict(zip(names, dec(r.x)))
    dof = max(1, len(tt) - len(names))
    s2 = float(np.sum(r.fun ** 2)) / dof
    try:
        cov = np.linalg.inv(r.jac.T @ r.jac) * s2
        ok_cov = bool(np.all(np.isfinite(cov)))
    except np.linalg.LinAlgError:
        cov, ok_cov = None, False
    test_t, test_y = [T[i] for i in test], [Y[i] for i in test]
    pred_test = _predict(sim, {**cond, **best}, obs, test_t)
    band_lo, band_hi = None, None
    if ok_cov and test_t:
        rnd = np.random.default_rng(seed)
        z = rnd.multivariate_normal(r.x, cov, size=draws)
        z = np.clip(z, enc(lo), enc(hi))
        preds = np.array([_predict(sim, {**cond, **dict(zip(names, dec(list(zz))))}, obs, test_t) for zz in z])
        resid_sd = math.sqrt(s2) * scale
        band_lo = list(np.percentile(preds, 2.5, axis=0) - 1.96 * resid_sd)
        band_hi = list(np.percentile(preds, 97.5, axis=0) + 1.96 * resid_sd)
    err = [p - y for p, y in zip(pred_test, test_y)]
    rmse = math.sqrt(sum(e * e for e in err) / len(err)) if err else None
    mae = sum(abs(e) for e in err) / len(err) if err else None
    my = sum(test_y) / len(test_y) if test_y else 0
    ss_tot = sum((y - my) ** 2 for y in test_y)
    r2 = 1 - sum(e * e for e in err) / ss_tot if ss_tot > 0 else None
    coverage = (sum(1 for y, a, b in zip(test_y, band_lo, band_hi) if a <= y <= b) / len(test_y)) if band_lo else None
    if coverage is not None and r2 is not None and coverage >= 0.8 and r2 >= 0.8:
        verdict = "validated on held-out data"
    elif r2 is not None and r2 >= 0.5:
        verdict = "partly validated (fit acceptable, uncertainty band misses held-out points)" if (coverage or 0) < 0.8 else "partly validated"
    else:
        verdict = "not validated - the model does not predict the held-out data"
    return {"model": card["id"], "model_version": sim.version, "observable": obs, "fitted": best,
            "fitted_se_log": dict(zip(names, [math.sqrt(cov[i, i]) for i in range(len(names))])) if ok_cov else None,
            "train_points": len(train), "heldout_points": len(test), "rmse": rmse, "mae": mae, "r2": r2,
            "coverage_95": coverage, "verdict": verdict, "heldout": [{"time": t, "observed": y, "predicted": p}
                                                                    for t, y, p in zip(test_t, test_y, pred_test)],
            "seed": seed, "version": VALIDATION_VERSION,
            "note": "validated means: predicts THESE held-out measurements within its uncertainty - not that the biology is right"}
