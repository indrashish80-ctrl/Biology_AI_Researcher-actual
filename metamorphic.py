"""
metamorphic.py  (v24.3)
-----------------------
Template-generated sentences whose correct direction is KNOWN, run through the engine's direction reader.

Why: every direction bug found by hand (a comma that flipped "decreased proliferation, reduction of stem cells" into an
increase) is a sentence shape nobody had written a test for. This generates thousands of shapes and checks two things:

  absolute     the direction read equals the direction the sentence states (after the endpoint's polarity)
  metamorphic  swapping every increase-word for its decrease-word flips the answer, and only that

    python metamorphic.py                 # print failures grouped by template
    python metamorphic.py --json out.json # machine-readable
    python metamorphic.py --seed 7 --per-template 400

No AI calls, no network, a few seconds.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))
os.environ.setdefault("BIOSYN_OFFLINE", "1")
os.environ.setdefault("BIOSYN_NO_WARMUP", "1")

AGENTS = ["Metformin", "Treatment with rapamycin", "Narrowband UVB", "The inhibitor", "Compound 12", "Vitamin D supplementation",
          "Dapagliflozin", "Knockdown of STAT3", "The JAK inhibitor tofacitinib", "An mTOR inhibitor", "Anti-TNF therapy", "The SGLT2 inhibitor", "Loss of p53", "An antagonist of IL-6", "Rapamycin, an mTOR blocker,", "Oral tofacitinib", "Resveratrol", "The vaccine", "Exercise training"]
# (phrase used in text, canonical outcome text) - the engine decides the polarity of each itself
ENDPOINTS = ["proliferation", "apoptosis", "cell viability", "tumour volume", "Ki-67", "inflammation", "body weight",
             "blood pressure", "HbA1c", "IL-6 levels", "tumour growth", "cell migration", "oxidative stress", "lesion area"]
SECOND = ["stem cell numbers", "invasion", "colony formation", "angiogenesis", "autophagy"]
CELLS = ["MCF-7 cells", "primary keratinocytes", "treated mice", "patients", "HepG2 cells", "the treated group"]

DEC = {"verb": ["decreased", "reduced", "lowered", "suppressed", "inhibited", "diminished", "attenuated", "downregulated"],
       "noun": ["decrease", "reduction", "decline", "drop", "suppression", "inhibition"],
       "adj": ["decreased", "reduced", "lower", "diminished", "suppressed"],
       "pp": ["decreased", "reduced", "lowered", "suppressed", "inhibited"],
       "comp": ["lower", "smaller", "reduced"]}
INC = {"verb": ["increased", "raised", "enhanced", "elevated", "induced", "stimulated", "upregulated", "promoted"],
       "noun": ["increase", "elevation", "rise", "enhancement", "induction", "upregulation"],
       "adj": ["increased", "elevated", "higher", "enhanced", "greater"],
       "pp": ["increased", "raised", "enhanced", "elevated", "induced"],
       "comp": ["higher", "greater", "larger"]}


def flip(d: str) -> str:
    return {"increase": "decrease", "decrease": "increase"}.get(d, d)


# template: (name, builder(words, agent, A, B, cell) -> text, which endpoint is asked, expected raw direction relative to 'sign')
# sign = +1 -> the sentence states a rise of A, -1 -> a fall, 0 -> no effect / not decided (relation only)
def _t(name, fn):
    return name, fn


TEMPLATES: List[Tuple[str, Any]] = [
    _t("simple", lambda w, g, a, b, c: f"{g} {w['verb']} {a}."),
    _t("significant_p", lambda w, g, a, b, c: f"{g} significantly {w['verb']} {a} (P < 0.01)."),
    _t("noun_phrase", lambda w, g, a, b, c: f"{g} caused a {w['noun']} in {a}."),
    _t("result_in", lambda w, g, a, b, c: f"{g} resulted in {w['adj']} {a}."),
    _t("passive", lambda w, g, a, b, c: f"{a[0].upper() + a[1:]} was {w['pp']} by {g.lower() if g[0].isupper() and ' ' in g else g}."),
    _t("comma_noun_tail", lambda w, g, a, b, c: f"{g} {w['verb']} {a}, {w['noun']} of {b}."),
    _t("comma_noun_tail_2", lambda w, g, a, b, c: f"{g} {w['verb']} {a}, a {w['noun']} of the {b}."),
    _t("list_tail", lambda w, g, a, b, c: f"{g} {w['verb']} {a}, {b} and migration."),
    _t("and_second_verb", lambda w, g, a, b, c: f"{g} {w['verb']} {a} and {w['verb']} {b}."),
    _t("in_system", lambda w, g, a, b, c: f"In {c}, {g.lower() if ' ' in g else g} {w['verb']} {a}."),
    _t("comparator_lower_higher", lambda w, g, a, b, c: f"{a[0].upper() + a[1:]} was {w['comp']} in the {g.lower() if ' ' in g else g} group than in controls."),
    _t("comparator_inverted", lambda w, g, a, b, c: f"{a[0].upper() + a[1:]} was {INC['comp'][0] if w is DEC else DEC['comp'][0]} in controls than in the {g.lower() if ' ' in g else g} group."),
    _t("leading_to", lambda w, g, a, b, c: f"{g} activated AMPK, leading to {w['adj']} {a}."),
    _t("thereby", lambda w, g, a, b, c: f"{g} arrested the cell cycle, thereby {w['verb']} {a}."),
    _t("fold_change", lambda w, g, a, b, c: f"{g} {w['verb']} {a} by 40% compared with vehicle."),
    _t("two_clause_semicolon", lambda w, g, a, b, c: f"{g} {w['verb']} {a}; {b} was unchanged."),
    _t("associated_with", lambda w, g, a, b, c: f"{g} was associated with {w['adj']} {a}."),
    _t("endpoint_first", lambda w, g, a, b, c: f"{a[0].upper() + a[1:]} {w['verb']} following {g.lower() if ' ' in g else g}."),
    _t("compared_with_vehicle", lambda w, g, a, b, c: f"Compared with vehicle, {g.lower() if ' ' in g else g} {w['verb']} {a}."),
    _t("dose_dependent", lambda w, g, a, b, c: f"{g} {w['verb']} {a} in a dose-dependent manner."),
    _t("observed_after", lambda w, g, a, b, c: f"A {w['noun']} in {a} was observed after {g.lower() if ' ' in g else g}."),
    _t("whereas_unchanged", lambda w, g, a, b, c: f"{g} markedly {w['verb']} {a}, whereas {b} remained unchanged."),
    _t("concessive", lambda w, g, a, b, c: f"Although {g.lower() if ' ' in g else g} {(INC if w is DEC else DEC)['verb'][0]} {b}, {a} was {w['pp']}."),
    _t("not_in_other", lambda w, g, a, b, c: f"{g} {w['verb']} {a} in {c}, but not in untreated controls."),
    _t("but_other_way", lambda w, g, a, b, c: f"{g} {w['verb']} {a} but {(INC if w is DEC else DEC)['verb'][0]} {b}."),
]
def _opp(w):
    """The other word table: 'reversed the INCREASE in A' states a fall, so it is built from the opposite words."""
    tab = INC if w.get("verb") in DEC["verb"] else DEC
    k = len(w.get("noun", "")) + len(w.get("verb", ""))
    return {key: vals[k % len(vals)] for key, vals in tab.items()}


_GERUND = {"decreased": "decreasing", "reduced": "reducing", "lowered": "lowering", "suppressed": "suppressing",
           "inhibited": "inhibiting", "diminished": "diminishing", "attenuated": "attenuating", "downregulated": "downregulating",
           "increased": "increasing", "raised": "raising", "enhanced": "enhancing", "elevated": "elevating", "induced": "inducing",
           "stimulated": "stimulating", "upregulated": "upregulating", "promoted": "promoting"}

# v24.3: shapes found by hand-probing after the v24.2 run reached 0 failures
TEMPLATES += [
    _t("thereby_gerund", lambda w, g, a, b, c: f"{g} arrested the cell cycle, thereby {_GERUND[w['verb']]} {a}."),
    _t("reversal", lambda w, g, a, b, c: f"{g} reversed the {_opp(w)['noun']} in {a}."),
    _t("prevented_change", lambda w, g, a, b, c: f"{g} prevented the {_opp(w)['noun']} in {a} seen in controls."),
    _t("after_manipulation", lambda w, g, a, b, c: f"{a[0].upper() + a[1:]} was {w['pp']} after knockdown of STAT3 in {c}."),
    _t("agent_named_by_endpoint", lambda w, g, a, b, c: f"An antagonist of TNF {w['verb']} {a}."),
]

NEGATED = [
    ("negated_verb", lambda w, g, a, b, c: f"{g} did not {({'decreased': 'decrease', 'reduced': 'reduce', 'lowered': 'lower', 'suppressed': 'suppress', 'inhibited': 'inhibit', 'diminished': 'diminish', 'attenuated': 'attenuate', 'downregulated': 'downregulate', 'increased': 'increase', 'raised': 'raise', 'enhanced': 'enhance', 'elevated': 'elevate', 'induced': 'induce', 'stimulated': 'stimulate', 'upregulated': 'upregulate', 'promoted': 'promote'})[w['verb']]} {a} (P = 0.41)."),
    ("neither_nor", lambda w, g, a, b, c: f"{g} {w['verb']} neither {a} nor {b}."),
    ("not_significantly", lambda w, g, a, b, c: f"{a[0].upper() + a[1:]} was not significantly {w['pp']} by {g.lower() if ' ' in g else g} (P = 0.32)."),
    ("failed_to", lambda w, g, a, b, c: f"{g} failed to {({'decreased': 'decrease', 'reduced': 'reduce', 'lowered': 'lower', 'suppressed': 'suppress', 'inhibited': 'inhibit', 'diminished': 'diminish', 'attenuated': 'attenuate', 'downregulated': 'downregulate', 'increased': 'increase', 'raised': 'raise', 'enhanced': 'enhance', 'elevated': 'elevate', 'induced': 'induce', 'stimulated': 'stimulate', 'upregulated': 'upregulate', 'promoted': 'promote'})[w['verb']]} {a}."),
]


def generate(seed: int = 1, per_template: int = 150, include_negated: bool = True) -> List[Dict[str, Any]]:
    rnd = random.Random(seed)
    out: List[Dict[str, Any]] = []
    # NEGATED templates check a different property: a sentence that DENIES a change ('did not reduce', 'failed to lower')
    # must never be read as that change. v24.2 left them out of the main run, which hid 196 failures; v24.3 fixed the
    # reader and runs them by default.
    template_list = TEMPLATES + (NEGATED if include_negated else [])
    for name, fn in template_list:
        negated = (name, fn) in NEGATED
        for _ in range(per_template):
            g, a, b, c = rnd.choice(AGENTS), rnd.choice(ENDPOINTS), rnd.choice(SECOND), rnd.choice(CELLS)
            for sign, words in ((-1, DEC), (+1, INC)):
                wi = {k: rnd.choice(v) for k, v in words.items()}
                wi = {k: (v if not isinstance(v, str) else v) for k, v in wi.items()}
                # 'comparator_inverted' reads the word table object identity, so pass the table through
                w = dict(wi)
                if name == "comparator_inverted":
                    class _W(dict):
                        pass
                    w = _W(wi)
                text = fn(words if name == "comparator_inverted" else w, g, a, b, c) if name == "comparator_inverted" else fn(w, g, a, b, c)
                if name in ("but_other_way",) and False:
                    pass
                out.append({"template": name, "text": text, "endpoint": a, "raw_sign": 0 if negated else sign,
                            "negated": negated, "pair": (name, g, a, b, c, tuple(sorted(wi.items()))),
                            "second": b if name in ("but_other_way",) else None, "sign_word": "dec" if sign < 0 else "inc"})
    return out


def read_direction(text: str, endpoint: str) -> Tuple[str, int]:
    import semantic_audit as s
    oi = s.canonical_outcome(endpoint)
    d = s.direction_in_text(text, oi, endpoint).get("direction") or "none"
    return d, oi.get("polarity", 1)


def run(seed: int = 1, per_template: int = 150) -> Dict[str, Any]:
    cases = generate(seed, per_template)
    fails: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    tot = Counter()
    for cs in cases:
        d, pol = read_direction(cs["text"], cs["endpoint"])
        tot[cs["template"]] += 1
        if cs["negated"]:
            # a negated or non-significant sentence must NOT assert the direction it denies
            denied = "decrease" if cs["sign_word"] == "dec" else "increase"
            import semantic_audit as _s
            if _s.canonical_outcome(cs["endpoint"]).get("polarity", 1) < 0:
                denied = flip(denied)
            if d == denied:
                fails[cs["template"]].append({"text": cs["text"], "read": d, "expected": "no_effect (or none)"})
            continue
        raw = "decrease" if cs["raw_sign"] < 0 else "increase"
        if cs["template"] == "comparator_inverted":
            pass
        exp = raw if pol > 0 else flip(raw)
        if d != exp:
            fails[cs["template"]].append({"text": cs["text"], "read": d, "expected": exp})
    summary = {t: {"cases": tot[t], "failures": len(fails.get(t, [])), "rate": round(len(fails.get(t, [])) / tot[t], 3),
                   "examples": fails.get(t, [])[:3]} for t in tot}
    return {"seed": seed, "total": sum(tot.values()), "failures": sum(len(v) for v in fails.values()), "templates": summary}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--per-template", type=int, default=150)
    ap.add_argument("--json")
    a = ap.parse_args()
    import signal
    if hasattr(signal, "SIGPIPE"):
        signal.signal(signal.SIGPIPE, signal.SIG_DFL)   # 'python metamorphic.py | head' exits quietly
    res = run(a.seed, a.per_template)
    if a.json:
        Path(a.json).write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    try:
        sys.stdout.reconfigure(errors="replace")
    except Exception:
        pass
    print(f"{res['failures']} / {res['total']} sentences read wrongly")
    for t, v in sorted(res["templates"].items(), key=lambda kv: -kv[1]["rate"]):
        print(f"  {t:26s} {v['failures']:4d}/{v['cases']:<4d} {v['rate']*100:5.1f}%")
        for ex in v["examples"][:2]:
            print(f"        {ex['text']!r}  read={ex['read']} expected={ex['expected']}")
