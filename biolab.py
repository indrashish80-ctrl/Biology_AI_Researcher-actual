"""
biolab.py  (v27)
================
The tool catalogue of BioSynthetica's own laboratory. Every tool is computed HERE (BioSynthetica's ode_engine, its
sequence and protein code, the virtual lab) - no external simulator is needed; the documented Tellurium model library
is one more tool when it is installed.

    TOOLS[name] = {"description", "args": {name: {type, default, min, max, uncertain, help}}, "metrics": [...], "run"}
    run_tool(name, args) -> {"tool", "args", "metrics": {name: number}, "views": [...], "notes": [...]}

`metrics` are the named numbers a prediction can be tested against ("peak_viral_load", "S.final", "gc_percent", ...).
`views` are what the website draws: plot, dna, protein, structure, pathogen, pathway, table.
Arguments marked `uncertain` are biology nobody has pinned down (rates, potencies, volumes); the robustness check
varies them, never the experiment's own choices (dose, sequence, time).

Tools: custom_model, library_model, dna_analysis, dna_mutation, crispr_guides, primers, restriction_sites,
protein_analysis, protein_mutation, protein_structure, bacteria_antibiotic, virus_antiviral, epidemic,
drug_exposure, dose_response, pathway, power.
"""
from __future__ import annotations

import math
import re
from typing import Any, Callable, Dict, List, Optional

BIOLAB_VERSION = "1.0"
BASIS = "in silico - a computer model or calculation, not experimental evidence"


def _arg(t: str, default: Any = None, lo: Optional[float] = None, hi: Optional[float] = None, uncertain: bool = False,
         help: str = "") -> Dict[str, Any]:
    return {"type": t, "default": default, "min": lo, "max": hi, "uncertain": uncertain, "help": help}


def _plot(title: str, t: List[float], series: Dict[str, List[float]], xlabel: str, ylabel: str = "",
          log: bool = False) -> Dict[str, Any]:
    step = max(1, len(t) // 300)
    return {"type": "plot", "title": title, "x": {"label": xlabel, "values": [round(v, 6) for v in t[::step]]},
            "series": [{"name": k, "values": [float(f"{x:.6g}") for x in v[::step]]} for k, v in series.items()],
            "ylabel": ylabel, "log": log}


def _summ_metrics(prefix: str, summ: Dict[str, Any]) -> Dict[str, float]:
    return {f"{prefix}.{m}": float(v) for m, v in (summ or {}).items() if isinstance(v, (int, float)) and math.isfinite(v)}


# ------------------------------------------------------------------------------------------------ simulation tools
def custom_model(model: Dict[str, Any], overrides: Optional[Dict[str, float]] = None, seed: Optional[int] = None) -> Dict[str, Any]:
    import ode_engine as E
    r = E.run(model, overrides=overrides, seed=seed)
    metrics: Dict[str, float] = {}
    for k, s in r["summary"].items():
        metrics.update(_summ_metrics(k, s))
    return {"metrics": metrics, "views": [_plot(model.get("name") or "Custom model", r["time"], r["series"],
                                                f"time ({r.get('time_unit') or 'model units'})")],
            "notes": [f"{r['engine']}; {'stochastic (Gillespie)' if r['stochastic'] else 'deterministic (LSODA)'}"],
            "raw": {k: v for k, v in r.items() if k not in ("series", "time")}}


def library_model(model: str, parameters: Optional[Dict[str, float]] = None, t_end: float = 100,
                  observable: Optional[str] = None) -> Dict[str, Any]:
    import simulator as S
    lib = S.ModelLibrary()
    card = lib.get(model)
    sim = S.Simulator(card)
    obs = [observable] if observable else list(card["observables"])
    r = sim.run(parameters or {}, t_end=min(float(t_end), card["max_time"]), points=301, observables=obs)
    metrics: Dict[str, float] = {}
    for k, s in r["summary"].items():
        metrics.update(_summ_metrics(k, s))
    return {"metrics": metrics, "views": [_plot(card["title"], r["time"], r["series"], f"time ({card.get('time_unit')})")],
            "notes": [f"documented model {model} (Tellurium); assumes: " + "; ".join(card.get("assumptions") or [])[:300]]}


def bacteria_antibiotic(conc_mg_l: float = 4.0, mic_mg_l: float = 1.0, growth_per_h: float = 0.8, kmax_per_h: float = 2.0,
                        hill: float = 1.0, n0: float = 1e6, capacity: float = 1e9, mutation_rate: float = 1e-8,
                        resistant_mic_fold: float = 16.0, resistant_fitness: float = 0.9, hours: float = 72,
                        stop_drug_at_h: float = 0.0, organism: str = "bacterium", drug: str = "antibiotic") -> Dict[str, Any]:
    """Susceptible (S) and resistant (R) bacteria under a constant antibiotic level: logistic growth, Emax / Hill kill
    above the MIC, resistant mutants arising at `mutation_rate` per division."""
    import ode_engine as E
    model = {"name": f"{organism} under {drug}",
             "species": {"S": n0, "R": 0.0},
             "parameters": {"r": growth_per_h, "K": capacity, "kmax": kmax_per_h, "C": conc_mg_l, "MIC": mic_mg_l,
                            "MICr": mic_mg_l * resistant_mic_fold, "h": hill, "mu": mutation_rate, "f": resistant_fitness},
             "equations": {"S": "r*S*(1-(S+R)/K) - kmax*hill(C,MIC,h)*S - mu*r*S",
                           "R": "f*r*R*(1-(S+R)/K) - kmax*hill(C,MICr,h)*R + mu*r*S"},
             "t_end": hours, "points": 241, "time_unit": "h"}
    if stop_drug_at_h and 0 < stop_drug_at_h < hours:
        model["events"] = [{"at": stop_drug_at_h, "set": {"C": 0.0}}]
    r = E.run(model)
    tot = [a + b for a, b in zip(r["series"]["S"], r["series"]["R"])]
    final, mn = tot[-1], min(tot)
    cleared = final < 1.0
    res_frac = r["series"]["R"][-1] / final if final >= 1.0 else 0.0     # meaningless once everything is cleared
    status = ("cleared" if cleared else "resistant strain took over" if res_frac > 0.5 else
              "suppressed" if final < n0 else "keeps growing")
    regrowth = next((r["time"][i] for i in range(1, len(tot)) if tot[i] > tot[i - 1] and tot[i - 1] <= mn * 1.0001
                     and r["time"][i] > 0), None)
    metrics = {"total.final": final, "total.min": mn, "log10_change": math.log10(max(final, 1e-12) / n0), "cleared": float(cleared),
               "resistant_fraction_final": res_frac, "conc_over_mic": conc_mg_l / mic_mg_l}
    if regrowth is not None:
        metrics["time_to_regrowth_h"] = regrowth
    return {"metrics": metrics,
            "views": [_plot(f"{organism}: susceptible and resistant cells", r["time"],
                            {"susceptible": r["series"]["S"], "resistant": r["series"]["R"]}, "hours", "cells", log=True),
                      {"type": "pathogen", "kind": "bacterium", "title": organism,
                       "facts": {"doubling time": f"{math.log(2) / growth_per_h * 60:.0f} min", "drug": drug,
                                 "level / MIC": f"{conc_mg_l / mic_mg_l:.2g}x", "resistant mutants": f"{mutation_rate:g} per division",
                                 "after": f"{hours:g} h: {final:.3g} cells ({status})"},
                       "status": status}],
            "notes": ["Logistic growth with Hill-type drug kill and resistant mutants (BioSynthetica ode_engine)."]}


def virus_antiviral(efficacy: float = 0.0, target: str = "production", start_day: float = 0.0, beta: float = 2.7e-5,
                    delta: float = 4.0, p: float = 0.012, c: float = 3.0, T0: float = 4e8, V0: float = 0.093,
                    days: float = 14.0, virus: str = "virus") -> Dict[str, Any]:
    """Target-cell-limited infection (T target cells, I infected, V virus) with an antiviral that blocks entry or
    production from `start_day`."""
    import ode_engine as E
    eps_e = efficacy if target == "entry" else 0.0
    eps_p = efficacy if target == "production" else 0.0
    model = {"name": f"{virus} infection", "species": {"T": T0, "I": 0.0, "V": V0},
             "parameters": {"b": beta, "d": delta, "p": p, "c": c, "ee": 0.0, "ep": 0.0},
             "equations": {"T": "-(1-ee)*b*T*V", "I": "(1-ee)*b*T*V - d*I", "V": "(1-ep)*p*I - c*V"},
             "t_end": days, "points": 281, "time_unit": "day"}
    if efficacy > 0:
        if start_day <= 0:
            model["parameters"].update(ee=eps_e, ep=eps_p)
        else:
            model["events"] = [{"at": start_day, "set": {"ee": eps_e, "ep": eps_p}}]
    r = E.run(model)
    V = r["series"]["V"]
    peak = max(V)
    t_peak = r["time"][V.index(peak)]
    R0 = beta * p * T0 / (delta * c)
    metrics = {"peak_viral_load": peak, "log10_peak": math.log10(max(peak, 1e-12)), "time_to_peak_day": t_peak,
               "viral_load_final": V[-1], "target_cells_left_percent": 100 * r["series"]["T"][-1] / T0,
               "R0_within_host": R0, "R0_with_drug": R0 * (1 - eps_e) * (1 - eps_p),
               "viral_auc": r["summary"]["V"].get("auc") or 0.0}
    status = "infection does not take hold" if peak <= V0 * 1.0001 else "infection takes hold"
    logs = [math.log10(max(v, 1e-6)) for v in V]
    return {"metrics": metrics,
            "views": [_plot(f"{virus} in the body", r["time"], {"virus (log10 copies/mL)": logs}, "days", "log10 virus"),
                      {"type": "pathogen", "kind": "virus", "title": virus,
                       "facts": {"within-host R0": f"{R0:.3g}", "with the drug": f"{metrics['R0_with_drug']:.3g}",
                                 "antiviral": f"{efficacy * 100:.0f}% block of {target} from day {start_day:g}",
                                 "peak": f"{peak:.3g} copies/mL on day {t_peak:.1f}"},
                       "status": status}],
            "notes": ["Target-cell-limited viral dynamics (Perelson-type), BioSynthetica ode_engine."]}


def epidemic(R0: float = 2.5, infectious_days: float = 7.0, latent_days: float = 0.0, population: float = 1e6,
             initial_infected: float = 10, intervention: float = 0.0, start_day: float = 0.0, days: float = 365,
             pathogen: str = "pathogen") -> Dict[str, Any]:
    import ode_engine as E
    g = 1.0 / infectious_days
    b = R0 * g
    sp = {"S": population - initial_infected, "I": initial_infected, "R": 0.0}
    eq = {"S": "-(1-x)*b*S*I/N", "I": "(1-x)*b*S*I/N - g*I", "R": "g*I"}
    if latent_days > 0:
        sp["E"] = 0.0
        eq = {"S": "-(1-x)*b*S*I/N", "E": "(1-x)*b*S*I/N - s*E", "I": "s*E - g*I", "R": "g*I"}
    model = {"name": f"{pathogen} epidemic", "species": sp,
             "parameters": {"b": b, "g": g, "N": population, "x": 0.0, "s": 1.0 / latent_days if latent_days > 0 else 1.0},
             "equations": eq, "t_end": days, "points": 366, "time_unit": "day"}
    if intervention > 0:
        if start_day <= 0:
            model["parameters"]["x"] = intervention
        else:
            model["events"] = [{"at": start_day, "set": {"x": intervention}}]
    r = E.run(model)
    I = r["series"]["I"]
    peak = max(I)
    metrics = {"peak_infected": peak, "peak_infected_percent": 100 * peak / population,
               "peak_day": r["time"][I.index(peak)], "attack_rate_percent": 100 * r["series"]["R"][-1] / population,
               "R_effective": R0 * (1 - intervention)}
    return {"metrics": metrics,
            "views": [_plot(f"{pathogen} epidemic", r["time"], {k: v for k, v in r["series"].items()}, "days", "people"),
                      {"type": "pathogen", "kind": "epidemic", "title": pathogen,
                       "facts": {"R0": f"{R0:g}", "with intervention": f"{metrics['R_effective']:.3g}",
                                 "peak": f"{metrics['peak_infected_percent']:.1f}% infected on day {metrics['peak_day']:.0f}",
                                 "ever infected": f"{metrics['attack_rate_percent']:.1f}%"},
                       "status": "epidemic" if metrics["R_effective"] > 1 else "dies out"}],
            "notes": ["SIR / SEIR compartment model, BioSynthetica ode_engine."]}


def drug_exposure(dose_mg: float = 850, interval_h: float = 12, n_doses: int = 14, ka_per_h: float = 1.0,
                  ke_per_h: float = 0.14, vd_l: float = 300.0, bioavailability: float = 0.55, mw_g_mol: float = 129.16,
                  lab_conc_um: Optional[float] = None, drug: str = "drug") -> Dict[str, Any]:
    """Plasma level in people (oral, repeated dosing) - and how a laboratory concentration compares with it."""
    import ode_engine as E
    model = {"name": f"{drug} plasma level", "species": {"G": 0.0, "C": 0.0},
             "parameters": {"ka": ka_per_h, "ke": ke_per_h, "V": vd_l},
             "equations": {"G": "-ka*G", "C": "ka*G/V - ke*C"}, "time_unit": "h"}
    # dose events add to the gut amount: emulate by integrating piecewise
    import numpy as np
    t_all, c_all, g, c = [], [], dose_mg * bioavailability, 0.0
    for i in range(int(n_doses)):
        seg = E.run({**{k: v for k, v in model.items() if k != "events"}, "species": {"G": g, "C": c},
                     "t_end": interval_h, "points": 61})
        t_all += [i * interval_h + x for x in seg["time"]]
        c_all += seg["series"]["C"]
        g, c = seg["series"]["G"][-1] + dose_mg * bioavailability, seg["series"]["C"][-1]
    um = [x / mw_g_mol * 1000 for x in c_all]            # mg/L -> µM
    last = um[-61:]
    metrics = {"cmax_um": max(um), "trough_um": min(last), "mean_um_last_interval": float(np.mean(last)),
               "half_life_h": math.log(2) / ke_per_h}
    if lab_conc_um:
        metrics["lab_over_cmax"] = lab_conc_um / max(um)
    series = {"plasma (µM)": um}
    if lab_conc_um:
        series["laboratory concentration"] = [lab_conc_um] * len(um)
    return {"metrics": metrics, "views": [_plot(f"{drug} in plasma", t_all, series, "hours", "µM")],
            "notes": ["Oral one-compartment pharmacokinetics, repeated dosing (BioSynthetica ode_engine)."]}


def dose_response(conc: float = 10.0, ic50: float = 10.0, hill: float = 1.0, emax: float = 100.0) -> Dict[str, Any]:
    xs = [ic50 * 10 ** (k / 20) for k in range(-60, 61)]
    ys = [emax * x ** hill / (ic50 ** hill + x ** hill) for x in xs]
    eff = emax * conc ** hill / (ic50 ** hill + conc ** hill)
    return {"metrics": {"effect_percent": eff, "conc_over_ic50": conc / ic50},
            "views": [_plot("Dose-response", [math.log10(x) for x in xs], {"effect (%)": ys}, "log10 concentration", "effect %")],
            "notes": ["Hill / Emax model."]}


# ------------------------------------------------------------------------------------------------ genes / proteins
_DNA = re.compile(r"[^ACGTN]")


def _dna(seq: str) -> str:
    s = re.sub(r"\s+|\d+", "", str(seq or "")).upper().replace("U", "T")
    if not s or _DNA.search(s):
        raise ValueError("a DNA sequence may contain only A, C, G, T (or U, N)")
    if len(s) > 200000:
        raise ValueError("at most 200,000 bases")
    return s


def dna_analysis(sequence: str, min_orf_aa: int = 30) -> Dict[str, Any]:
    import virtual_lab as V
    s = _dna(sequence)
    r = V.analyze_dna(s, min_orf_aa=min(int(min_orf_aa), max(3, len(s) // 3 - 1)))
    orfs = r.get("orfs") or []
    feats = [{"start": o["start_nt"], "end": o["start_nt"] + 3 * o["length_aa"] + 2, "label": f"ORF {o['length_aa']} aa",
              "kind": "orf", "strand": o["strand"]} for o in orfs[:20] if o.get("strand") == "+"]
    longest = max(orfs, key=lambda o: o["length_aa"])["protein"] if orfs else ""
    views = [{"type": "dna", "title": "DNA sequence", "sequence": s[:3000], "features": feats,
              "facts": {"length": f"{len(s):,} nt", "GC": f"{r['gc_percent']}%", "Tm": f"{r['melting_temp_c']} °C",
                        "open reading frames": len(orfs)}}]
    if longest:
        views.append({"type": "protein", "title": "Protein of the longest open reading frame", "sequence": longest[:2000]})
    return {"metrics": {"length_nt": float(len(s)), "gc_percent": r["gc_percent"], "n_orfs": float(len(orfs)),
                        "longest_orf_aa": float(r.get("longest_orf_aa") or 0), "melting_temp_c": r["melting_temp_c"]},
            "views": views, "notes": []}


def dna_mutation(sequence: str, position: int, new_base: str, frame: int = 0) -> Dict[str, Any]:
    import virtual_lab as V
    s = _dna(sequence)
    r = V.mutate_dna(s, int(position), str(new_base).upper()[:1], int(frame))
    eff = r.get("effect")
    p = int(position)
    return {"metrics": {"synonymous": float(eff == "synonymous"), "missense": float(eff == "missense"),
                        "nonsense": float(eff == "nonsense"), "codon_number": float(r.get("codon_number") or 0)},
            "views": [{"type": "dna", "title": f"Mutation at position {p}", "sequence": s[max(0, p - 61):p + 60],
                       "offset": max(0, p - 61), "highlight": [p], "features": [],
                       "facts": {"codon": r.get("codon"), "amino acid": r.get("amino_acid") or "unchanged", "effect": eff}}],
            "notes": [f"{r.get('codon')}: {eff}" + (f" ({r.get('amino_acid')})" if r.get("amino_acid") else "")]}


def crispr_guides(sequence: str, max_guides: int = 10) -> Dict[str, Any]:
    """20-nt SpCas9 guides followed by an NGG PAM on either strand; ranked by GC 40-70% and no TTTT run."""
    import virtual_lab as V
    s = _dna(sequence)
    out = []
    for strand, seq in (("+", s), ("-", V.revcomp(s))):
        for m in re.finditer(r"(?=([ACGT]{20})[ACGT]GG)", seq):
            g = m.group(1)
            gc = 100 * (g.count("G") + g.count("C")) / 20
            ok = 40 <= gc <= 70 and "TTTT" not in g
            pos = m.start() + 1 if strand == "+" else len(s) - m.start() - 22
            out.append({"guide": g, "strand": strand, "position": pos, "gc": round(gc, 1), "good": ok})
    out.sort(key=lambda x: (not x["good"], abs(x["gc"] - 55)))
    top = out[:int(max_guides)]
    return {"metrics": {"n_guides": float(len(out)), "n_good_guides": float(sum(o["good"] for o in out))},
            "views": [{"type": "dna", "title": "CRISPR guide sites (NGG PAM)", "sequence": s[:3000],
                       "features": [{"start": o["position"], "end": o["position"] + 19, "label": f"guide {i + 1} ({o['strand']})",
                                     "kind": "guide"} for i, o in enumerate(top)]},
                      {"type": "table", "title": "Best guides", "columns": ["guide (5'-3')", "strand", "position", "GC %", "usable"],
                       "rows": [[o["guide"], o["strand"], o["position"], o["gc"], "yes" if o["good"] else "no"] for o in top]}],
            "notes": ["Off-target effects are not checked (that needs the whole genome)."]}


def _tm(p: str) -> float:
    gc = p.count("G") + p.count("C")
    return 64.9 + 41 * (gc - 16.4) / len(p) if len(p) > 13 else 2 * (len(p) - gc) + 4 * gc


def primers(sequence: str, length: int = 20) -> Dict[str, Any]:
    import virtual_lab as V
    s = _dna(sequence)
    L = max(16, min(int(length), 30))
    if len(s) < 2 * L + 20:
        raise ValueError("the sequence is too short for a primer pair")

    def best(seq: str) -> Dict[str, Any]:
        cands = []
        for i in range(0, min(60, len(seq) - L)):
            p = seq[i:i + L]
            gc = 100 * (p.count("G") + p.count("C")) / L
            cands.append((abs(_tm(p) - 60) + (0 if 40 <= gc <= 60 else 10) + (0 if p[-1] in "GC" else 2), i, p, gc))
        _, i, p, gc = min(cands)
        return {"seq": p, "pos": i, "tm": round(_tm(p), 1), "gc": round(gc, 1)}
    f = best(s)
    rv = best(V.revcomp(s))
    product = len(s) - f["pos"] - rv["pos"]
    return {"metrics": {"tm_forward": f["tm"], "tm_reverse": rv["tm"], "tm_difference": abs(f["tm"] - rv["tm"]),
                        "product_bp": float(product)},
            "views": [{"type": "dna", "title": "PCR primers", "sequence": s[:3000],
                       "features": [{"start": f["pos"] + 1, "end": f["pos"] + L, "label": "forward primer", "kind": "primer"},
                                    {"start": len(s) - rv["pos"] - L + 1, "end": len(s) - rv["pos"], "label": "reverse primer",
                                     "kind": "primer"}]},
                      {"type": "table", "title": "Primer pair", "columns": ["primer", "sequence (5'-3')", "Tm °C", "GC %"],
                       "rows": [["forward", f["seq"], f["tm"], f["gc"]], ["reverse", rv["seq"], rv["tm"], rv["gc"]]]}],
            "notes": [f"PCR product {product} bp; Tm by the GC rule (approximate)."]}


ENZYMES = {"EcoRI": "GAATTC", "BamHI": "GGATCC", "HindIII": "AAGCTT", "NotI": "GCGGCCGC", "XhoI": "CTCGAG",
           "NdeI": "CATATG", "XbaI": "TCTAGA", "PstI": "CTGCAG", "SmaI": "CCCGGG", "KpnI": "GGTACC", "SalI": "GTCGAC",
           "NcoI": "CCATGG", "BglII": "AGATCT", "SacI": "GAGCTC", "SpeI": "ACTAGT"}


def restriction_sites(sequence: str) -> Dict[str, Any]:
    s = _dna(sequence)
    feats, counts = [], {}
    for name, site in ENZYMES.items():
        hits = [m.start() + 1 for m in re.finditer(f"(?={site})", s)]
        counts[name] = len(hits)
        feats += [{"start": h, "end": h + len(site) - 1, "label": name, "kind": "site"} for h in hits[:10]]
    return {"metrics": {f"sites.{k}": float(v) for k, v in counts.items()} | {"enzymes_cutting": float(sum(1 for v in counts.values() if v))},
            "views": [{"type": "dna", "title": "Restriction sites", "sequence": s[:3000], "features": feats[:60]},
                      {"type": "table", "title": "Cut sites", "columns": ["enzyme", "site", "cuts"],
                       "rows": [[k, ENZYMES[k], v] for k, v in counts.items() if v]}],
            "notes": []}


KD = {"A": 1.8, "R": -4.5, "N": -3.5, "D": -3.5, "C": 2.5, "Q": -3.5, "E": -3.5, "G": -0.4, "H": -3.2, "I": 4.5, "L": 3.8,
      "K": -3.9, "M": 1.9, "F": 2.8, "P": -1.6, "S": -0.8, "T": -0.7, "W": -0.9, "Y": -1.3, "V": 4.2}
GROUPS = [set("AVLIMC"), set("FWY"), set("KRH"), set("DE"), set("STNQ"), set("GP")]


def _prot(seq: str) -> str:
    s = re.sub(r"\s+|\d+|\*", "", str(seq or "")).upper()
    if not s or re.search(r"[^ACDEFGHIKLMNPQRSTVWYX]", s):
        raise ValueError("a protein sequence may contain only the 20 amino-acid letters (and X)")
    return s[:20000]


def _profile(s: str, w: int = 9) -> List[float]:
    h = [KD.get(a, 0.0) for a in s]
    half = w // 2
    return [round(sum(h[max(0, i - half):i + half + 1]) / len(h[max(0, i - half):i + half + 1]), 2) for i in range(len(s))]


def protein_analysis(sequence: Optional[str] = None, gene: Optional[str] = None) -> Dict[str, Any]:
    import virtual_lab as V
    notes, views = [], []
    if not sequence and gene:
        try:
            fetched = V.fetch_protein_sequence(gene)
            sequence = fetched.get("sequence")
            notes.append(f"sequence of {gene} from UniProt ({fetched.get('accession')})")
        except Exception as exc:
            raise ValueError(f"could not fetch the protein for {gene} ({type(exc).__name__}) - paste its sequence")
    s = _prot(sequence or "")
    r = V.protein_properties(s)
    tm = r.get("predicted_transmembrane_segments") or []
    feats = [{"start": t.get("start"), "end": t.get("end"), "label": "membrane segment", "kind": "tm"} for t in tm if isinstance(t, dict)]
    views.append({"type": "protein", "title": f"Protein{' ' + gene if gene else ''}", "sequence": s, "features": feats,
                  "profile": {"label": "hydropathy (Kyte-Doolittle)", "values": _profile(s)},
                  "facts": {"length": f"{r['length_aa']} aa", "mass": f"{r['molecular_weight_kda']} kDa",
                            "pI": r["isoelectric_point"], "charge at pH 7.4": r["net_charge_ph7_4"],
                            "GRAVY": r["gravy_hydropathy"], "membrane segments": len(tm)}})
    if gene:
        views.append({"type": "structure", "title": f"{gene} predicted 3D structure (AlphaFold)", "source": "alphafold",
                      "gene": gene})
    return {"metrics": {"length_aa": float(r["length_aa"]), "mass_kda": r["molecular_weight_kda"],
                        "isoelectric_point": r["isoelectric_point"], "net_charge": r["net_charge_ph7_4"],
                        "gravy": r["gravy_hydropathy"], "transmembrane_segments": float(len(tm))},
            "views": views, "notes": notes}


def protein_mutation(sequence: str, position: int, new_aa: str) -> Dict[str, Any]:
    s = _prot(sequence)
    p = int(position)
    if not 1 <= p <= len(s):
        raise ValueError(f"position must be 1-{len(s)}")
    old, new = s[p - 1], str(new_aa).upper()[:1]
    if new not in KD:
        raise ValueError("new_aa must be one amino-acid letter")
    mut = s[:p - 1] + new + s[p:]
    charge = {"K": 1, "R": 1, "H": 0.1, "D": -1, "E": -1}
    conservative = any(old in g and new in g for g in GROUPS)
    w0, w1 = _profile(s)[p - 1], _profile(mut)[p - 1]
    return {"metrics": {"delta_hydropathy_local": w1 - w0, "delta_charge": charge.get(new, 0) - charge.get(old, 0),
                        "conservative": float(conservative), "to_proline": float(new == "P"), "from_glycine": float(old == "G")},
            "views": [{"type": "protein", "title": f"{old}{p}{new}", "sequence": mut, "highlight": [p],
                       "profile": {"label": "hydropathy after the change", "values": _profile(mut)},
                       "facts": {"change": f"{old}{p}{new}", "kind": "conservative" if conservative else "non-conservative",
                                 "charge change": charge.get(new, 0) - charge.get(old, 0)}}],
            "notes": ["Sequence-level estimate; whether the change harms the protein depends on where it sits in the 3D "
                      "structure."]}


def protein_structure(gene: Optional[str] = None, pdb_id: Optional[str] = None) -> Dict[str, Any]:
    if not gene and not pdb_id:
        raise ValueError("give a gene name or a PDB code")
    return {"metrics": {}, "views": [{"type": "structure", "title": f"{pdb_id or gene} 3D structure",
                                      "source": "pdb" if pdb_id else "alphafold", "pdb_id": pdb_id, "gene": gene}],
            "notes": ["The structure is downloaded by your browser from RCSB PDB / AlphaFold DB."]}


def pathway(edges: str, source: str, change: str = "increase", target: Optional[str] = None) -> Dict[str, Any]:
    import virtual_lab as V
    r = V.perturb_pathway(edges, source, change)
    pred = r.get("predictions") or {}
    sign = {"increase": 1.0, "decrease": -1.0}
    metrics = {f"effect.{k}": sign.get(v, 0.0) for k, v in pred.items()}
    if target:
        metrics["effect_on_target"] = sign.get(pred.get(target, ""), 0.0)
    es = V.parse_edges(edges)
    return {"metrics": metrics,
            "views": [{"type": "pathway", "title": f"{change} {source}", "edges": [[a, b, s] for a, b, s in es],
                       "source": source, "effects": pred}],
            "notes": ["Signed graph propagation (activation ->, inhibition -|); no quantities."]}


def power(effect_size: float = 0.5, n_per_group: Optional[int] = None, alpha: float = 0.05) -> Dict[str, Any]:
    import virtual_lab as V
    r = V.power_two_groups(effect_size=effect_size, n_per_group=n_per_group, alpha=alpha)
    m = {k: float(v) for k, v in r.items() if isinstance(v, (int, float)) and not isinstance(v, bool)}
    return {"metrics": m, "views": [{"type": "table", "title": "Study size", "columns": ["quantity", "value"],
                                     "rows": [[k.replace("_", " "), v] for k, v in m.items()]}], "notes": []}


TOOLS: Dict[str, Dict[str, Any]] = {
    "custom_model": {"run": custom_model, "description": "Build and simulate ANY system as equations or reactions (gene "
                     "circuit, signalling, metabolism, cells, pathogens, immune response). Metrics: <species>.max, .min, "
                     ".final, .auc, .time_to_max, .min_late, .mean_late",
                     "args": {"model": _arg("model", None, help="ode_engine spec: species, parameters, equations or reactions, "
                                                                "events, t_end, points, stochastic"),
                              "overrides": _arg("dict", None)}},
    "library_model": {"run": library_model, "description": "A documented, referenced model from the library (needs "
                      "Tellurium). Metrics: <observable>.max/.min/.final/...",
                      "args": {"model": _arg("str"), "parameters": _arg("dict", None), "t_end": _arg("float", 100, 0.1, 1e5),
                               "observable": _arg("str", None)}},
    "bacteria_antibiotic": {"run": bacteria_antibiotic, "description": "Bacteria (susceptible + resistant mutants) under an "
                            "antibiotic. Metrics: total.final, total.min, log10_change, cleared (1/0), "
                            "resistant_fraction_final, time_to_regrowth_h, conc_over_mic",
                            "args": {"conc_mg_l": _arg("float", 4.0, 0, 1e4), "mic_mg_l": _arg("float", 1.0, 1e-4, 1e4, True),
                                     "growth_per_h": _arg("float", 0.8, 0.01, 5, True), "kmax_per_h": _arg("float", 2.0, 0, 20, True),
                                     "hill": _arg("float", 1.0, 0.3, 6, True), "n0": _arg("float", 1e6, 1, 1e12),
                                     "capacity": _arg("float", 1e9, 1e3, 1e13), "mutation_rate": _arg("float", 1e-8, 0, 1e-3, True),
                                     "resistant_mic_fold": _arg("float", 16, 1, 1e4, True),
                                     "resistant_fitness": _arg("float", 0.9, 0.1, 1.5, True), "hours": _arg("float", 72, 1, 2000),
                                     "stop_drug_at_h": _arg("float", 0, 0, 2000), "organism": _arg("str", "bacterium"),
                                     "drug": _arg("str", "antibiotic")}},
    "virus_antiviral": {"run": virus_antiviral, "description": "A virus infection in the body with an antiviral. Metrics: "
                        "peak_viral_load, log10_peak, time_to_peak_day, viral_load_final, target_cells_left_percent, "
                        "R0_within_host, R0_with_drug, viral_auc",
                        "args": {"efficacy": _arg("float", 0.0, 0, 1), "target": _arg("str", "production"),
                                 "start_day": _arg("float", 0, 0, 60), "beta": _arg("float", 2.7e-5, 1e-12, 1, True),
                                 "delta": _arg("float", 4.0, 0.01, 50, True), "p": _arg("float", 0.012, 1e-6, 1e5, True),
                                 "c": _arg("float", 3.0, 0.01, 100, True), "T0": _arg("float", 4e8, 1, 1e13),
                                 "V0": _arg("float", 0.093, 1e-9, 1e9), "days": _arg("float", 14, 0.5, 365),
                                 "virus": _arg("str", "virus")}},
    "epidemic": {"run": epidemic, "description": "An epidemic in a population (SIR / SEIR) with an intervention. Metrics: "
                 "peak_infected, peak_infected_percent, peak_day, attack_rate_percent, R_effective",
                 "args": {"R0": _arg("float", 2.5, 0.1, 30, True), "infectious_days": _arg("float", 7, 0.5, 365, True),
                          "latent_days": _arg("float", 0, 0, 60, True), "population": _arg("float", 1e6, 10, 1e10),
                          "initial_infected": _arg("float", 10, 1, 1e8), "intervention": _arg("float", 0, 0, 1),
                          "start_day": _arg("float", 0, 0, 3650), "days": _arg("float", 365, 1, 7300),
                          "pathogen": _arg("str", "pathogen")}},
    "drug_exposure": {"run": drug_exposure, "description": "Plasma level of an oral drug in people (repeated doses) and how a "
                      "lab concentration compares. Metrics: cmax_um, trough_um, mean_um_last_interval, half_life_h, "
                      "lab_over_cmax",
                      "args": {"dose_mg": _arg("float", 850, 0.01, 1e5), "interval_h": _arg("float", 12, 1, 336),
                               "n_doses": _arg("int", 14, 1, 200), "ka_per_h": _arg("float", 1.0, 0.01, 20, True),
                               "ke_per_h": _arg("float", 0.14, 0.001, 10, True), "vd_l": _arg("float", 300, 1, 1e5, True),
                               "bioavailability": _arg("float", 0.55, 0.01, 1, True), "mw_g_mol": _arg("float", 129.16, 10, 1e6),
                               "lab_conc_um": _arg("float", None, 0, 1e9), "drug": _arg("str", "drug")}},
    "dose_response": {"run": dose_response, "description": "Effect at a concentration from IC50 / Hill. Metrics: "
                      "effect_percent, conc_over_ic50",
                      "args": {"conc": _arg("float", 10, 0, 1e9), "ic50": _arg("float", 10, 1e-9, 1e9, True),
                               "hill": _arg("float", 1.0, 0.1, 10, True), "emax": _arg("float", 100, 0, 1000, True)}},
    "dna_analysis": {"run": dna_analysis, "description": "A DNA / gene sequence: GC, open reading frames, translation. "
                     "Metrics: length_nt, gc_percent, n_orfs, longest_orf_aa, melting_temp_c",
                     "args": {"sequence": _arg("str"), "min_orf_aa": _arg("int", 30, 3, 10000)}},
    "dna_mutation": {"run": dna_mutation, "description": "What a single-base change does to the protein. Metrics: "
                     "synonymous, missense, nonsense (1/0), codon_number",
                     "args": {"sequence": _arg("str"), "position": _arg("int", None, 1, 200000), "new_base": _arg("str"),
                              "frame": _arg("int", 0, 0, 2)}},
    "crispr_guides": {"run": crispr_guides, "description": "CRISPR-Cas9 guide sites in a gene. Metrics: n_guides, n_good_guides",
                      "args": {"sequence": _arg("str"), "max_guides": _arg("int", 10, 1, 100)}},
    "primers": {"run": primers, "description": "A PCR primer pair for a sequence. Metrics: tm_forward, tm_reverse, "
                "tm_difference, product_bp", "args": {"sequence": _arg("str"), "length": _arg("int", 20, 16, 30)}},
    "restriction_sites": {"run": restriction_sites, "description": "Restriction-enzyme cut sites. Metrics: sites.<enzyme>, "
                          "enzymes_cutting", "args": {"sequence": _arg("str")}},
    "protein_analysis": {"run": protein_analysis, "description": "A protein (sequence, or a human gene name fetched from "
                         "UniProt): mass, pI, charge, hydropathy, membrane segments, 3D structure. Metrics: length_aa, "
                         "mass_kda, isoelectric_point, net_charge, gravy, transmembrane_segments",
                         "args": {"sequence": _arg("str", None), "gene": _arg("str", None)}},
    "protein_mutation": {"run": protein_mutation, "description": "An amino-acid change. Metrics: delta_hydropathy_local, "
                         "delta_charge, conservative, to_proline, from_glycine",
                         "args": {"sequence": _arg("str"), "position": _arg("int", None, 1, 20000), "new_aa": _arg("str")}},
    "protein_structure": {"run": protein_structure, "description": "Show a protein's 3D structure (gene -> AlphaFold, or a "
                          "PDB code)", "args": {"gene": _arg("str", None), "pdb_id": _arg("str", None)}},
    "pathway": {"run": pathway, "description": "Direction of effects through a signed pathway ('A -> B -| C'). Metrics: "
                "effect.<node> (+1 / -1 / 0), effect_on_target",
                "args": {"edges": _arg("str"), "source": _arg("str"), "change": _arg("str", "increase"),
                         "target": _arg("str", None)}},
    "power": {"run": power, "description": "Statistical power / study size for two groups. Metrics: n_per_group_required, power",
              "args": {"effect_size": _arg("float", 0.5, 0.01, 5), "n_per_group": _arg("int", None, 2, 1e7),
                       "alpha": _arg("float", 0.05, 1e-6, 0.5)}},
}


def catalogue() -> List[Dict[str, Any]]:
    out = []
    for name, t in TOOLS.items():
        if name == "library_model":
            try:
                import simulator as S
                if not S.backend()["available"]:
                    continue
            except Exception:
                continue
        out.append({"name": name, "description": t["description"],
                    "args": {k: {kk: vv for kk, vv in a.items() if vv is not None and kk != "uncertain" or kk == "uncertain" and vv}
                             for k, a in t["args"].items()}})
    return out


def validate_args(name: str, args: Dict[str, Any]) -> Dict[str, Any]:
    """Type and range check against the tool's declaration; returns the cleaned arguments or raises ValueError."""
    if name not in TOOLS:
        raise ValueError(f"unknown tool '{name}' (tools: {', '.join(TOOLS)})")
    if not isinstance(args, dict):
        raise ValueError("args must be an object")
    decl = TOOLS[name]["args"]
    clean = {}
    for k, v in args.items():
        if k not in decl:
            raise ValueError(f"{name}: unknown argument '{k}' (allowed: {', '.join(decl)})")
        a = decl[k]
        if v is None:
            continue
        if a["type"] in ("float", "int"):
            try:
                v = float(v) if a["type"] == "float" else int(round(float(v)))
            except (TypeError, ValueError):
                raise ValueError(f"{name}: '{k}' must be a number")
            if (a["min"] is not None and v < a["min"]) or (a["max"] is not None and v > a["max"]):
                raise ValueError(f"{name}: '{k}' = {v:g} is outside the allowed range [{a['min']:g}, {a['max']:g}]")
        elif a["type"] == "str":
            v = str(v)[:200000]
        elif a["type"] in ("dict", "model") and not isinstance(v, dict):
            raise ValueError(f"{name}: '{k}' must be an object")
        clean[k] = v
    for k, a in decl.items():
        if k not in clean and a["default"] is None and a["type"] in ("str", "model") and k in ("sequence", "model", "edges",
                                                                                              "source", "new_base", "new_aa"):
            if not (name == "protein_analysis"):
                raise ValueError(f"{name}: '{k}' is required")
    if name == "custom_model":
        import ode_engine as E
        E.validate(clean.get("model") or {})
    return clean


def run_tool(name: str, args: Dict[str, Any]) -> Dict[str, Any]:
    clean = validate_args(name, args)
    out = TOOLS[name]["run"](**clean)
    out.setdefault("views", [])
    out.setdefault("notes", [])
    return {"tool": name, "args": clean, **out, "basis": BASIS}


def uncertain_args(name: str, args: Dict[str, Any]) -> Dict[str, float]:
    """The numeric arguments of a tool that are uncertain biology (varied by the robustness check), with their values
    (the given value, else the default)."""
    decl = TOOLS[name]["args"]
    out = {}
    for k, a in decl.items():
        if a.get("uncertain"):
            v = args.get(k, a.get("default"))
            if isinstance(v, (int, float)) and v > 0:
                out[k] = float(v)
    return out
