"""
ai_lab.py  (v27)
================
The AUTONOMOUS laboratory. Given a claim, hypothesis or question, BioSynthetica designs its own experiments, runs them
on its own engine (biolab: custom models, genes, proteins, pathogens, drugs, pathways, statistics) and judges them -
with the AI kept out of the judgement.

    investigate(claim, router=None, max_experiments=4) -> {
        "verdict", "verdict_plain", "experiments": [...], "views": [...], "readings_agree", "steps", "basis", ...}

1. DESIGN    the AI (or, without a key, a rule-based planner) chooses up to 4 experiments from the tool catalogue and,
             BEFORE anything runs, writes down what the claim predicts for each one ("prediction"): a metric, a
             comparison and a value - or a ratio to a control experiment ("at least ten times lower than without the
             drug"). Invalid designs get one repair round with the validator's reasons.
2. RE-READ   a second, independent AI reading (another provider when there are several keys) states the prediction of
             the main experiment again WITHOUT seeing the first one. If the two differ, the claim is ambiguous and both
             readings are tested.
3. RUN       the code runs every experiment and compares each result with its prediction.
4. STRESS    each main experiment is repeated over plausible values of the uncertain biology (Monte Carlo, CV 30 %,
             inside the tool's ranges) - the share of plausible worlds in which the claim holds.
5. VERDICT   'supported by the simulation' | 'supported, but depends on assumptions' | 'contradicted by the
             simulation' | 'ambiguous claim' | 'explored (no testable prediction)' | 'not testable in the lab'.
             A control experiment that does not behave as predicted marks the result as unreliable.
Every result says it is in silico: consistent with a model, not proof.
"""
from __future__ import annotations

import json
import math
import random
import re
import time
from typing import Any, Dict, List, Optional

AI_LAB_VERSION = "1.0"
COMPARE = {">": lambda a, b: a > b, ">=": lambda a, b: a >= b, "<": lambda a, b: a < b, "<=": lambda a, b: a <= b,
           "==": lambda a, b: abs(a - b) <= 1e-9 * max(1.0, abs(b))}
PLAIN = {"supported by the simulation": "Supported by the simulation",
         "supported, but depends on assumptions": "Supported, but only under some assumptions",
         "contradicted by the simulation": "Contradicted by the simulation",
         "contradicted, but depends on assumptions": "Fails as stated, but could hold under other assumptions",
         "ambiguous claim": "The claim can be read in two ways",
         "explored (no testable prediction)": "Explored - no testable prediction",
         "not testable in the lab": "Not testable in the lab"}


# ------------------------------------------------------------------------------------------------ design
def _catalogue_text() -> str:
    import biolab as B
    L = []
    for t in B.catalogue():
        args = ", ".join(f"{k}" + (f"={a['default']}" if a.get("default") is not None else "") +
                         (f" [{a['min']}-{a['max']}]" if a.get("min") is not None else "") + (" (uncertain)" if a.get("uncertain") else "")
                         for k, a in t["args"].items())
        L.append(f"- {t['name']}: {t['description']}\n    args: {args}")
    return "\n".join(L)


def design_prompt(claim: str, context: str = "") -> str:
    return ("You are designing computer experiments to TEST a scientific claim with BioSynthetica's own laboratory "
            "tools. Choose 1 to 4 experiments. For EACH experiment state, BEFORE it runs, what the claim predicts.\n"
            "Rules:\n- Use only the tools and arguments listed; keep values inside the ranges; use realistic values for the "
            "organism, drug or gene (typical published values).\n- A prediction is {\"metric\": <a metric of that tool>, "
            "\"comparison\": one of > >= < <=, \"value\": number} or a comparison with a control experiment: "
            "{\"metric\": m, \"comparison\": \"<=\", \"relative_to\": <index of the control experiment>, \"factor\": 0.1} "
            "(= the metric is at most 0.1 times the control's).\n- Add a control experiment (e.g. no drug) when the claim "
            "compares; give the control its own prediction.\n- custom_model lets you write ANY system as equations, e.g. "
            "{\"species\": {\"X\": 1}, \"parameters\": {\"k\": 0.1}, \"equations\": {\"X\": \"-k*X\"}, \"t_end\": 50}.\n"
            "- If no tool can test the claim, set testable to false and say why.\n"
            "Return ONLY JSON: {\"testable\": true, \"experiments\": [{\"tool\": ..., \"args\": {...}, \"role\": \"main\" | "
            "\"control\", \"prediction\": {...}, \"why\": \"one sentence\"}], \"reason\": \"...\"}\n\n"
            f"TOOLS:\n{_catalogue_text()}\n\n" + (f"CONTEXT: {context[:1500]}\n\n" if context else "") + f"CLAIM: {claim}")


def _ask_json(router: Any, prompt: str, provider: Optional[str] = None) -> Optional[Dict[str, Any]]:
    from llm_json import parse_json_response
    import contextlib
    ctx = router.prefer(provider) if provider and hasattr(router, "prefer") else contextlib.nullcontext()
    with ctx:
        raw = router.ask(prompt, task="hypotheses")
    try:
        d = parse_json_response(raw)
        return d if isinstance(d, dict) else None
    except Exception:
        return None


def _check_design(plan: Dict[str, Any]) -> List[str]:
    import biolab as B
    errs = []
    exps = plan.get("experiments")
    if not isinstance(exps, list) or not exps:
        return ["no experiments"]
    for i, e in enumerate(exps[:4]):
        if not isinstance(e, dict):
            errs.append(f"experiment {i}: not an object")
            continue
        try:
            B.validate_args(e.get("tool"), e.get("args") or {})
        except Exception as exc:
            errs.append(f"experiment {i} ({e.get('tool')}): {exc}")
        p = e.get("prediction")
        if p is not None and (not isinstance(p, dict) or p.get("comparison") not in COMPARE or not p.get("metric")):
            errs.append(f"experiment {i}: prediction needs metric and comparison (> >= < <=)")
    return errs


# rule-based planner (no AI key): enough to make common claims testable
_SEQ_DNA = re.compile(r"\b[ACGTU]{30,}\b", re.I)
_SEQ_PROT = re.compile(r"\b[ACDEFGHIKLMNPQRSTVWY]{20,}\b")
_GENE = re.compile(r"\b([A-Z][A-Z0-9]{1,7}\d?[A-Z0-9]*)\b")


def _num(pattern: str, text: str) -> Optional[float]:
    m = re.search(pattern, text, re.I)
    return float(m.group(1).replace(",", "")) if m else None


def rule_plan(claim: str) -> Dict[str, Any]:
    c = claim or ""
    low = c.lower()
    exps: List[Dict[str, Any]] = []
    dna = _SEQ_DNA.search(c)
    if dna and not re.search(r"[DEFHIKLMNPQRSVWY]{3,}", dna.group(0).upper()):
        seq = dna.group(0)
        pos, base = _num(r"position\s*(\d+)", c), re.search(r"\bto\s+([ACGT])\b", c, re.I)
        if pos and base:
            exps.append({"tool": "dna_mutation", "args": {"sequence": seq, "position": int(pos), "new_base": base.group(1)},
                         "role": "main", "prediction": ({"metric": "nonsense", "comparison": ">=", "value": 1}
                                                        if re.search(r"stop|nonsense|truncat", low) else
                                                        {"metric": "synonymous", "comparison": ">=", "value": 1}
                                                        if re.search(r"silent|synonymous|no effect", low) else None)})
        elif "crispr" in low or "guide" in low:
            exps.append({"tool": "crispr_guides", "args": {"sequence": seq}, "role": "main",
                         "prediction": {"metric": "n_good_guides", "comparison": ">=", "value": 1}})
        elif "primer" in low or "pcr" in low:
            exps.append({"tool": "primers", "args": {"sequence": seq}, "role": "main",
                         "prediction": {"metric": "tm_difference", "comparison": "<=", "value": 5}})
        else:
            gc = _num(r"gc[^0-9]{0,20}(\d+(?:\.\d+)?)\s*%", c)
            exps.append({"tool": "dna_analysis", "args": {"sequence": seq, "min_orf_aa": 20}, "role": "main",
                         "prediction": ({"metric": "gc_percent", "comparison": ">=" if re.search(r"above|over|more|high", low) else "<=",
                                         "value": gc} if gc else
                                        {"metric": "n_orfs", "comparison": ">=", "value": 1} if "open reading frame" in low or "encodes" in low
                                        else None)})
        return {"testable": True, "experiments": exps, "planner": "rules"}
    prot = _SEQ_PROT.search(c)
    if prot and re.search(r"[DEFHIKLMNPQRSVWY]", prot.group(0)):
        pred = None
        if re.search(r"membrane|transmembrane", low):
            pred = {"metric": "transmembrane_segments", "comparison": ">=", "value": 1}
        elif re.search(r"hydrophobic", low):
            pred = {"metric": "gravy", "comparison": ">", "value": 0}
        elif re.search(r"hydrophilic|soluble", low):
            pred = {"metric": "gravy", "comparison": "<", "value": 0}
        return {"testable": True, "planner": "rules",
                "experiments": [{"tool": "protein_analysis", "args": {"sequence": prot.group(0)}, "role": "main", "prediction": pred}]}
    if re.search(r"epidemic|outbreak|r0|transmission|vaccin|lockdown|herd", low):
        r0 = _num(r"r0\D{0,6}(\d+(?:\.\d+)?)", c) or 2.5
        cut = _num(r"(\d+(?:\.\d+)?)\s*%", c)
        args = {"R0": r0, "intervention": (cut or 50) / 100}
        return {"testable": True, "planner": "rules", "experiments": [
            {"tool": "epidemic", "args": args, "role": "main",
             "prediction": {"metric": "peak_infected", "comparison": "<", "relative_to": 1, "factor": 1.0}},
            {"tool": "epidemic", "args": {**args, "intervention": 0.0}, "role": "control", "prediction": None}]}
    if re.search(r"antibiot|bacteri|\bmic\b|e\.? ?coli|staph|strepto|tuberculos|klebsiella|pseudomonas", low):
        conc, mic = _num(r"(\d+(?:\.\d+)?)\s*(?:mg/l|µg/ml|ug/ml|mg/ml)", c), _num(r"mic\D{0,12}(\d+(?:\.\d+)?)", c)
        args = {"conc_mg_l": conc or 4.0, "mic_mg_l": mic or 1.0, "hours": 48}
        org = re.search(r"\b(e\.? ?coli|staphylococcus aureus|s\.? aureus|klebsiella\w*|pseudomonas\w*|m\.? tuberculosis)\b", low)
        if org:
            args["organism"] = org.group(1)
        pred = ({"metric": "resistant_fraction_final", "comparison": ">", "value": 0.5} if "resist" in low else
                {"metric": "cleared", "comparison": ">=", "value": 1} if re.search(r"clear|kill|eradicat|steril", low) else
                {"metric": "log10_change", "comparison": "<", "value": 0} if re.search(r"reduc|suppress|lower|inhibit", low) else None)
        return {"testable": True, "planner": "rules", "experiments": [
            {"tool": "bacteria_antibiotic", "args": args, "role": "main", "prediction": pred},
            {"tool": "bacteria_antibiotic", "args": {**args, "conc_mg_l": 0.0}, "role": "control",
             "prediction": {"metric": "log10_change", "comparison": ">", "value": 0}}]}
    if re.search(r"virus|viral|antiviral|influenza|hiv|hepatitis|sars|covid", low):
        eff = _num(r"(\d+(?:\.\d+)?)\s*%", c)
        args = {"efficacy": (eff or 90) / 100, "target": "entry" if "entry" in low else "production"}
        fold = _num(r"(\d+(?:\.\d+)?)\s*-?\s*fold", c) or (10 if "tenfold" in low or "ten-fold" in low else None)
        pred = ({"metric": "peak_viral_load", "comparison": "<=", "relative_to": 1, "factor": 1 / fold} if fold else
                {"metric": "peak_viral_load", "comparison": "<", "relative_to": 1, "factor": 1.0}
                if re.search(r"lower|reduc|suppress|block|stop|prevent", low) else None)
        return {"testable": True, "planner": "rules", "experiments": [
            {"tool": "virus_antiviral", "args": args, "role": "main", "prediction": pred},
            {"tool": "virus_antiviral", "args": {**args, "efficacy": 0.0}, "role": "control",
             "prediction": {"metric": "R0_within_host", "comparison": ">", "value": 1}}]}
    mg = _num(r"(\d+(?:\.\d+)?)\s*mg\b", c)
    um = _num(r"(\d+(?:\.\d+)?)\s*(?:µm|um|micromolar)", c)
    if mg:
        args = {"dose_mg": mg, "n_doses": 1 if re.search(r"single|one dose|once", low) else 14}
        pred = None
        if um:
            pred = {"metric": "trough_um" if re.search(r"trough|keeps|stays|above .* all|throughout", low) and args["n_doses"] > 1
                    else "cmax_um", "comparison": "<" if re.search(r"below|under|less than", low) else ">=", "value": um}
        return {"testable": True, "planner": "rules", "experiments": [{"tool": "drug_exposure", "args": args, "role": "main",
                                                                      "prediction": pred}]}
    if re.search(r"\s(?:->|-\|)\s", c):
        src = re.split(r"\s*(?:->|-\|)\s*", c.strip())[0].split()[-1]
        tgt = re.split(r"\s*(?:->|-\|)\s*", c.strip())[-1].split()[0]
        verb = re.search(r"\b(reduc|lower|inhibit|suppress|decreas|block)\w*|\b(increas|rais|activat|promot|enhanc)\w*", low)
        pred = None
        if verb:
            pred = ({"metric": "effect_on_target", "comparison": "<=", "value": -1} if verb.group(1) else
                    {"metric": "effect_on_target", "comparison": ">=", "value": 1})
        edges = re.sub(r"^.*?(\w+\s*(?:->|-\|))", r"\1", c.split(":")[-1]).strip()
        return {"testable": True, "planner": "rules", "experiments": [
            {"tool": "pathway", "args": {"edges": edges, "source": src, "target": tgt}, "role": "main", "prediction": pred}]}
    gene = next((g for g in _GENE.findall(c) if re.search(r"\d", g) or len(g) >= 3 and g.isupper()), None)
    if gene and re.search(r"protein|structure|fold|membrane|gene", low):
        return {"testable": True, "planner": "rules", "experiments": [
            {"tool": "protein_analysis", "args": {"gene": gene}, "role": "main", "prediction": None}]}
    return {"testable": False, "experiments": [], "planner": "rules",
            "reason": "no laboratory tool matches this claim without an AI key to design a custom experiment"}


# ------------------------------------------------------------------------------------------------ run and judge
def _evaluate(pred: Optional[Dict[str, Any]], result: Dict[str, Any], results: List[Optional[Dict[str, Any]]]) -> Dict[str, Any]:
    if not pred:
        return {"holds": None, "text": "no prediction"}
    m = pred.get("metric")
    v = (result.get("metrics") or {}).get(m)
    if v is None:
        return {"holds": None, "text": f"the tool reported no '{m}'"}
    comp = pred.get("comparison")
    if pred.get("relative_to") is not None:
        j = int(pred["relative_to"])
        ref = results[j] if 0 <= j < len(results) else None
        rv = ((ref or {}).get("metrics") or {}).get(m)
        if rv is None:
            return {"holds": None, "text": f"control experiment {j} has no '{m}'"}
        target = float(pred.get("factor", 1.0)) * rv
        txt = f"{m} = {v:.4g} vs {pred.get('factor', 1.0):g} x control ({rv:.4g})"
    else:
        try:
            target = float(pred.get("value"))
        except (TypeError, ValueError):
            return {"holds": None, "text": "the prediction has no number"}
        txt = f"{m} = {v:.4g} (predicted {comp} {target:g})"
    ok = COMPARE[comp](v, target)
    return {"holds": bool(ok), "value": v, "target": target, "text": txt}


def _stress(exp: Dict[str, Any], results: List[Optional[Dict[str, Any]]], n: int, seed: int) -> Dict[str, Any]:
    import biolab as B
    if not exp.get("prediction"):
        return {}
    custom = exp["tool"] == "custom_model"
    if custom:              # a purpose-built model: its own parameters are the uncertain biology (except any marked fixed)
        model = exp["args"].get("model") or {}
        fixed = set(model.get("fixed_parameters") or [])
        unc = {k: float(v) for k, v in (model.get("parameters") or {}).items()
               if isinstance(v, (int, float)) and v > 0 and k not in fixed}
    else:
        unc = B.uncertain_args(exp["tool"], exp["args"])
    if not unc:
        return {}
    rng = random.Random(seed)
    decl = B.TOOLS[exp["tool"]]["args"]
    held = total = 0
    for _ in range(n):
        a = dict(exp["args"])
        draw = {}
        for k, mu in unc.items():
            s = math.sqrt(math.log(1 + 0.3 ** 2))
            draw[k] = mu * math.exp(rng.gauss(-s * s / 2, s))
        if custom:
            a["overrides"] = draw
        else:
            for k, v in draw.items():
                lo, hi = decl[k]["min"], decl[k]["max"]
                a[k] = min(max(v, lo if lo is not None else v), hi if hi is not None else v)
        try:
            r = B.run_tool(exp["tool"], a)
        except Exception:
            continue
        ev = _evaluate(exp["prediction"], r, results)
        if ev["holds"] is None:
            continue
        total += 1
        held += int(ev["holds"])
    return {"n": total, "holds": held, "share": round(held / total, 3) if total else None, "varied": sorted(unc)}


def _grade(base: Optional[bool], share: Optional[float]) -> str:
    if base is None:
        return "explored (no testable prediction)"
    if base and (share is None or share >= 0.9):
        return "supported by the simulation"
    if not base and (share is None or share <= 0.1):
        return "contradicted by the simulation"
    return "supported, but depends on assumptions" if base else "contradicted, but depends on assumptions"


def investigate(claim: str, router: Any = None, context: str = "", max_experiments: int = 4, n_mc: int = 30,
                seed: int = 1, time_budget_s: float = 90.0) -> Dict[str, Any]:
    import biolab as B
    t0 = time.time()
    claim = re.sub(r"\s+", " ", str(claim or "")).strip()
    out: Dict[str, Any] = {"claim": claim, "version": AI_LAB_VERSION, "steps": [], "experiments": [], "views": [],
                           "basis": "in silico - the claim is consistent (or not) with BioSynthetica's models and "
                                    "calculations; that is not proof that it is true"}
    if len(claim) < 8:
        out.update(verdict="not testable in the lab", verdict_plain=PLAIN["not testable in the lab"])
        out["steps"].append("no claim")
        return out
    plan = None
    if router is not None:
        try:
            plan = _ask_json(router, design_prompt(claim, context))
            if plan is not None:
                out["steps"].append("the AI designed the experiments and wrote down its predictions BEFORE they ran")
                errs = _check_design(plan) if plan.get("testable", True) else []
                if errs:
                    fix = _ask_json(router, design_prompt(claim, context) + "\n\nYOUR PREVIOUS DESIGN WAS REFUSED:\n- "
                                    + "\n- ".join(errs[:8]) + "\nReturn a corrected design.")
                    if fix and not _check_design(fix):
                        plan = fix
                        out["steps"].append("one repair round fixed the design")
                    else:
                        out["steps"].append("the AI's design was refused: " + "; ".join(errs[:3]))
                        plan = None
        except Exception as exc:
            out["steps"].append(f"the AI could not design experiments ({type(exc).__name__}) - rule-based design used")
            plan = None
    if plan is None:
        plan = rule_plan(claim)
        out["steps"].append("experiments chosen by the rule-based planner" if plan.get("testable") else
                            "the rule-based planner found no matching tool")
    out["planner"] = "ai" if plan.get("planner") != "rules" else "rules"
    if not plan.get("testable", True) or not plan.get("experiments"):
        out.update(verdict="not testable in the lab", verdict_plain=PLAIN["not testable in the lab"],
                   reason=plan.get("reason"))
        return out
    exps = [e for e in plan["experiments"][:max_experiments] if isinstance(e, dict)]
    for e in exps:
        e.setdefault("role", "main")
        e["args"] = B.validate_args(e["tool"], e.get("args") or {})
    # 2. independent second reading of the main experiment's prediction
    mains = [i for i, e in enumerate(exps) if e.get("role") != "control"] or [0]
    m0 = exps[mains[0]]
    agree = None
    if router is not None and m0.get("prediction"):
        try:
            other = None
            try:
                provs = router.get_available_providers()
                first = (router.order_for("hypotheses") or [None])[0]
                other = next((p for p in provs if p != first), None)
            except Exception:
                pass
            metrics = re.search(r"Metrics: (.*)$", B.TOOLS[m0["tool"]]["description"])
            p2 = _ask_json(router, "A computer experiment will test the claim below. Without judging whether the claim is "
                                   "true, state what the claim PREDICTS for this experiment. Return ONLY JSON {\"metric\": "
                                   "..., \"comparison\": \">|>=|<|<=\", \"value\": number}" +
                                   (" or, relative to the control experiment, {\"metric\": ..., \"comparison\": ..., "
                                    "\"relative_to\": <index>, \"factor\": number}" if len(exps) > 1 else "") +
                                   f".\nEXPERIMENT: tool {m0['tool']}, arguments {json.dumps(m0['args'])[:800]}"
                                   f"\nMETRICS AVAILABLE: {metrics.group(1) if metrics else 'see tool'}\nCLAIM: {claim}", other)
            if p2 and p2.get("metric") and p2.get("comparison") in COMPARE:
                p1 = m0["prediction"]
                same = p1.get("metric") == p2.get("metric") and \
                    (p1.get("comparison") or "")[:1] == (p2.get("comparison") or "")[:1]
                if same and p1.get("relative_to") is None and p2.get("relative_to") is None:
                    try:
                        a, b = float(p1["value"]), float(p2["value"])
                        same = abs(a - b) <= 0.25 * max(abs(a), abs(b), 1e-12)
                    except (TypeError, ValueError, KeyError):
                        same = False
                agree = bool(same)
                if not agree:
                    exps.append({**m0, "prediction": p2, "role": "main", "reader": 2})
                out["steps"].append("an independent second AI reading of the claim " + ("agreed" if agree else "DISAGREED - both readings are tested"))
        except Exception:
            pass
    # 3. run, 4. stress
    results: List[Optional[Dict[str, Any]]] = []
    for e in exps:
        try:
            results.append(B.run_tool(e["tool"], e["args"]))
        except Exception as exc:
            results.append(None)
            e["error"] = f"{type(exc).__name__}: {str(exc)[:200]}"
    for i, (e, r) in enumerate(zip(exps, results)):
        rec = {"tool": e["tool"], "role": e.get("role"), "why": e.get("why"), "args": e["args"],
               "prediction": e.get("prediction"), "reader": e.get("reader", 1)}
        if r is None:
            rec["error"] = e.get("error")
            out["experiments"].append(rec)
            continue
        ev = _evaluate(e.get("prediction"), r, results)
        rec.update(result=ev, metrics={k: (round(v, 6) if isinstance(v, float) else v) for k, v in r["metrics"].items()},
                   notes=r.get("notes"))
        if e.get("role") != "control" and time.time() - t0 < time_budget_s:
            rec["robustness"] = _stress(e, results, n_mc, seed + i)
        rec["verdict"] = _grade(ev["holds"], (rec.get("robustness") or {}).get("share")) if e.get("role") != "control" else \
            ("control behaved as expected" if ev["holds"] else "CONTROL FAILED" if ev["holds"] is False else "control run")
        out["experiments"].append(rec)
        for v in r.get("views") or []:
            out["views"].append({**v, "experiment": i, "role": e.get("role")})
    # 5. verdict
    main_recs = [x for x in out["experiments"] if x.get("role") != "control" and "verdict" in x]
    controls_failed = [x for x in out["experiments"] if x.get("verdict") == "CONTROL FAILED"]
    if agree is False:
        verdict = "ambiguous claim"
    elif not main_recs:
        verdict = "not testable in the lab"
    else:
        vs = {x["verdict"] for x in main_recs}
        verdict = vs.pop() if len(vs) == 1 else ("supported, but depends on assumptions"
                                                if any(v.startswith("supported") for v in vs) else
                                                "contradicted, but depends on assumptions")
    out.update(verdict=verdict, verdict_plain=PLAIN.get(verdict, verdict), readings_agree=agree,
               controls_failed=len(controls_failed), seconds=round(time.time() - t0, 2))
    if controls_failed:
        out["steps"].append("a control experiment did not behave as predicted - the model may not fit this system; "
                            "treat the result with caution")
    return out


def markdown(results: List[Dict[str, Any]]) -> str:
    if not results:
        return ""
    L = ["## Laboratory tests (in silico)", "",
         "Each claim was turned into computer experiments with predictions written BEFORE they ran; the code ran them, "
         "repeated them over plausible values of the uncertain biology, and judged them.", ""]
    for r in results:
        L.append(f"- **{r.get('verdict_plain') or r.get('verdict')}** - {str(r.get('claim', ''))[:220]}")
        for x in r.get("experiments") or []:
            if x.get("error"):
                L.append(f"  - {x['tool']}: could not run ({x['error'][:120]})")
                continue
            res = x.get("result") or {}
            rob = x.get("robustness") or {}
            share = f"; holds in {rob['share'] * 100:.0f}% of {rob['n']} plausible variations" if rob.get("share") is not None else ""
            L.append(f"  - {x.get('role')}: {x['tool']} - {res.get('text', '')}{share} -> {x.get('verdict')}")
    L.append("")
    return "\n".join(L)
