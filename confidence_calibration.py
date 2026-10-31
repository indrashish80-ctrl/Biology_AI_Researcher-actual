"""
confidence_calibration.py  (v26)
================================
CALIBRATED confidence for the engine's reading of an effect's direction - measured, not stated by a model.

A reading's features (was the endpoint located in the sentence, was the sentence-level fallback used, was a
statistical test read and was it significant, which direction was read) are mapped to the accuracy that readings with
the same features ACTUALLY had on the doctor-labelled development set (gold/papers). The map is then checked on the
UNTOUCHED held-out set (gold/heldout): expected calibration error (ECE), Brier score and a reliability table are
stored next to the map in gold/calibration.json - so a reported "0.85" can be checked: of the held-out readings that got
about 0.85, about 85% were right.

    python confidence_calibration.py          # fit on dev, evaluate on held-out, write gold/calibration.json
    probability(reading) -> float | None      # for one reading (gold_bench.read_direction output)
    for_finding(finding) -> {...}             # re-reads a finding's quote and returns its calibrated confidence

Back-off: the most specific feature group with >= 8 development readings is used, else a coarser one; every
probability is Beta(1,1)-smoothed. A reading of 'none' (the engine abstained) has no probability.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

CAL_VERSION = "1.0"
ROOT = Path(__file__).resolve().parent
CAL_PATH = ROOT / "gold" / "calibration.json"
MIN_N = 8


def _sig(r: Dict[str, Any]) -> str:
    s = r.get("significant")
    return "sig" if s is True else "ns" if s is False else "untested"


def keys(r: Dict[str, Any]) -> List[str]:
    lab = r.get("direction") or "none"
    loc = "located" if r.get("endpoint_located") else "not_located"
    fb = "fallback" if r.get("fallback") else "direct"
    return [f"{loc}|{fb}|{_sig(r)}|{lab}", f"{loc}|{_sig(r)}|{lab}", f"{_sig(r)}|{lab}", lab]


def fit(pairs: Sequence[Tuple[Dict[str, Any], bool]]) -> Dict[str, List[int]]:
    table: Dict[str, List[int]] = {}
    for r, ok in pairs:
        if (r.get("direction") or "none") == "none":
            continue
        for k in keys(r):
            t = table.setdefault(k, [0, 0])
            t[0] += int(ok)
            t[1] += 1
    return table


def predict(r: Dict[str, Any], table: Dict[str, List[int]]) -> Optional[float]:
    if (r.get("direction") or "none") == "none":
        return None
    for k in keys(r):
        ok_n = table.get(k)
        if ok_n and ok_n[1] >= MIN_N:
            return (ok_n[0] + 1) / (ok_n[1] + 2)
    ok_n = table.get(keys(r)[-1]) or [0, 0]
    return (ok_n[0] + 1) / (ok_n[1] + 2)


def evaluate(pairs: Sequence[Tuple[Dict[str, Any], bool]], table: Dict[str, List[int]], bins: int = 10) -> Dict[str, Any]:
    pts = [(predict(r, table), ok) for r, ok in pairs]
    pts = [(p, ok) for p, ok in pts if p is not None]
    if not pts:
        return {"n": 0}
    rel = []
    ece = 0.0
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        inb = [(p, ok) for p, ok in pts if lo <= p < hi or (b == bins - 1 and p == 1.0)]
        if not inb:
            continue
        conf = sum(p for p, _ in inb) / len(inb)
        acc = sum(ok for _, ok in inb) / len(inb)
        ece += len(inb) / len(pts) * abs(conf - acc)
        rel.append({"bin": f"{lo:.1f}-{hi:.1f}", "n": len(inb), "mean_confidence": round(conf, 3), "accuracy": round(acc, 3)})
    brier = sum((p - ok) ** 2 for p, ok in pts) / len(pts)
    base = sum(ok for _, ok in pts) / len(pts)
    return {"n": len(pts), "ece": round(ece, 4), "brier": round(brier, 4),
            "brier_constant": round(sum((base - ok) ** 2 for _, ok in pts) / len(pts), 4),
            "accuracy": round(base, 4), "reliability": rel}


def _pairs(papers: List[Dict[str, Any]]) -> List[Tuple[Dict[str, Any], bool]]:
    import gold_bench as gb
    out = []
    for g in papers:
        for f in g.get("findings") or []:
            r = gb.read_direction(f["quote"], f["endpoint"], f.get("intervention", ""), f.get("comparator", ""),
                                  g.get("source_text") or "")
            out.append((r, r["direction"] == f["direction"]))
    return out


def load_calibration_set() -> List[Dict[str, Any]]:
    p = ROOT / "gold" / "calibration" / "ei_calibration.jsonl"
    if not p.exists():
        return []
    return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]


def build(write: bool = True) -> Dict[str, Any]:
    """Fit on the CALIBRATION set (doctor-labelled, never used to tune the engine), check on the held-out set.
    The development set is reported too, but it is NOT used for fitting: the engine was tuned on it, so readings there
    are more often right than on new papers (fitting there made the confidence over-confident: 0.92 said, 0.84 true)."""
    import gold_bench as gb
    import gold_eval as ge
    cal_papers = load_calibration_set()
    cal = _pairs(cal_papers)
    dev = _pairs(ge.load_gold())
    held = _pairs(gb.load_heldout())
    fit_on, name = (cal, f"calibration set ({len(cal)} readings, never tuned on)") if cal else \
        (dev, f"development set ({len(dev)} readings - tuned on, expect over-confidence)")
    table = fit(fit_on)
    res = {"version": CAL_VERSION, "code_versions": gb.versions(), "fitted_on": name,
           "table": table, "calibration": evaluate(cal, table) if cal else None, "development": evaluate(dev, table),
           "heldout": evaluate(held, table),
           "abstained_heldout": sum(1 for r, _ in held if (r.get("direction") or "none") == "none")}
    if write:
        CAL_PATH.parent.mkdir(parents=True, exist_ok=True)
        CAL_PATH.write_text(json.dumps(res, indent=1), encoding="utf-8")
    return res


_TABLE: Optional[Dict[str, List[int]]] = None


def _table() -> Dict[str, List[int]]:
    global _TABLE
    if _TABLE is None:
        try:
            _TABLE = json.loads(CAL_PATH.read_text(encoding="utf-8"))["table"]
        except Exception:
            _TABLE = {}
    return _TABLE


def probability(reading: Dict[str, Any]) -> Optional[float]:
    t = _table()
    return round(predict(reading, t), 3) if t else None


def for_finding(f: Dict[str, Any], intervention: str = "", comparator: str = "") -> Dict[str, Any]:
    """The calibrated confidence that the code reads this finding's quote in the direction the finding states."""
    import gold_bench as gb
    q, ep = f.get("quote") or "", f.get("endpoint") or f.get("outcome") or ""
    if not q or not ep:
        return {"confidence": None, "basis": "no quote / endpoint"}
    r = gb.read_direction(q, str(ep).replace("_", " "), intervention or f.get("intervention") or "",
                          comparator or f.get("comparator") or "")
    p = probability(r)
    agrees = r["direction"] == f.get("direction") if r["direction"] != "none" else None
    return {"confidence": p if agrees else (round(1 - p, 3) if p is not None and agrees is False else None),
            "code_reading": r["direction"], "agrees_with_finding": agrees,
            "basis": "calibrated on the gold development set, checked on the held-out set (gold/calibration.json)"}


def markdown(res: Dict[str, Any]) -> str:
    h = res.get("heldout") or {}
    L = ["## Confidence calibration (direction reading)", "",
         f"- fitted on {res.get('fitted_on')}; checked on the untouched held-out set ({h.get('n')} readings, "
         f"{res.get('abstained_heldout')} abstentions)",
         f"- held-out expected calibration error **{h.get('ece')}**, Brier {h.get('brier')} "
         f"(a constant guess: {h.get('brier_constant')})", "",
         "| confidence bin | readings | mean confidence | actual accuracy |", "|---|---|---|---|"]
    for b in h.get("reliability") or []:
        L.append(f"| {b['bin']} | {b['n']} | {b['mean_confidence']:.2f} | {b['accuracy']:.2f} |")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    r = build()
    print(markdown(r))
    sys.exit(0)
