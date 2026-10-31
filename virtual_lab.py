"""
virtual_lab.py
--------------
BioSynthetica's VIRTUAL LAB: in-silico tools and small simulation environments that let the AI (and the user)
test a hypothesis about a gene, protein, virus, drug or pathway BEFORE anyone touches a pipette.

Everything runs locally on the user's computer (numpy / scipy); only the optional sequence fetchers call public
databases (UniProt / NCBI). Every result is labelled "in silico" - a simulation can show that a hypothesis is
internally consistent, mathematically possible or untestable as designed, but it is never evidence that it is true.

Tools (all return plain dicts, JSON-safe):

  GENES / DNA / RNA      analyze_dna(seq)            GC %, length, reverse complement, ORFs in 6 frames, translation
                         mutate_dna(seq, pos, base)   effect of a point mutation (silent / missense / nonsense)
                         align(a, b)                  global alignment, identity %, gaps (Needleman-Wunsch)
  PROTEINS               protein_properties(seq)     mass, pI, net charge at pH 7.4, GRAVY, aromaticity,
                                                      extinction coefficient, Kyte-Doolittle transmembrane segments
  VIRUSES                simulate_virus(...)          target-cell-limited viral dynamics (T, I, V) with / without an
                                                      antiviral; R0, peak load, time to peak, viral AUC reduction
                         simulate_epidemic(...)       SIR epidemic: R0, peak infected, attack rate
  DRUGS                  dose_response(...)           Hill / Emax curve: effect at a dose, IC50 / EC50
                         fit_dose_response(doses, effects)   estimate IC50, Hill slope, Emax from data
                         pk_one_compartment(...)      plasma concentration-time, Cmax, AUC, time above a target
                         exposure_vs_potency(...)     is the human plasma level anywhere near the in-vitro IC50?
  CELLS / TUMOURS        simulate_growth(...)         exponential / logistic / Gompertz growth with drug kill
  PATHWAYS               perturb_pathway(edges, ...)  signed-network propagation: predicted direction of every node
                                                      after a perturbation (A -| B -> C)
  STATISTICS             power_two_groups(...)        power or required n for a two-group comparison
                         compare_groups(a, b)         Welch t-test, Mann-Whitney, Cohen's d
  HYPOTHESES             check_hypothesis(h, ...)     runs every applicable tool on a hypothesis and returns
                                                      consistent / inconsistent / not testable as designed
"""

from __future__ import annotations

import math
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

IN_SILICO = "in silico (simulation / calculation - not experimental evidence)"

# --------------------------------------------------------------------------- genetic code
_BASES = "TCAG"
_AA = "FFLLSSSSYY**CC*WLLLLPPPPHHQQRRRRIIIMTTTTNNKKSSRRVVVVAAAADDEEGGGG"
CODON = {a + b + c: _AA[16 * i + 4 * j + k] for i, a in enumerate(_BASES) for j, b in enumerate(_BASES) for k, c in enumerate(_BASES)}
AA3 = {"A": "Ala", "R": "Arg", "N": "Asn", "D": "Asp", "C": "Cys", "Q": "Gln", "E": "Glu", "G": "Gly", "H": "His",
       "I": "Ile", "L": "Leu", "K": "Lys", "M": "Met", "F": "Phe", "P": "Pro", "S": "Ser", "T": "Thr", "W": "Trp",
       "Y": "Tyr", "V": "Val", "*": "Stop"}


def _clean(seq: str, alphabet: str) -> str:
    s = re.sub(r"^>.*$", "", seq or "", flags=re.M)                 # FASTA header
    s = re.sub(r"[\s\d]", "", s).upper()
    bad = sorted(set(s) - set(alphabet))
    if bad:
        raise ValueError(f"unexpected characters {''.join(bad)[:10]} (allowed: {alphabet})")
    if not s:
        raise ValueError("empty sequence")
    return s


def revcomp(dna: str) -> str:
    return dna.translate(str.maketrans("ACGTN", "TGCAN"))[::-1]


def translate(dna: str, frame: int = 0, to_stop: bool = False) -> str:
    out = []
    for i in range(frame, len(dna) - 2, 3):
        aa = CODON.get(dna[i:i + 3], "X")
        if aa == "*" and to_stop:
            break
        out.append(aa)
    return "".join(out)


def analyze_dna(seq: str, min_orf_aa: int = 30) -> Dict[str, Any]:
    """Basic gene / transcript analysis. RNA (U) is accepted and converted."""
    s = _clean((seq or "").upper().replace("U", "T"), "ACGTN")
    gc = (s.count("G") + s.count("C")) / max(1, len(s) - s.count("N")) * 100
    orfs = []
    for strand, sq in (("+", s), ("-", revcomp(s))):
        for f in range(3):
            prot = translate(sq, f)
            pos = 0
            for chunk in prot.split("*"):
                start = chunk.find("M")
                if start >= 0 and len(chunk) - start >= min_orf_aa:
                    nt_start = f + 3 * (pos + start)
                    orfs.append({"strand": strand, "frame": f + 1, "start_nt": nt_start + 1,
                                 "length_aa": len(chunk) - start, "protein": chunk[start:][:60] + ("…" if len(chunk) - start > 60 else "")})
                pos += len(chunk) + 1
    orfs.sort(key=lambda o: -o["length_aa"])
    tm = (4 * (s.count("G") + s.count("C")) + 2 * (s.count("A") + s.count("T"))) if len(s) < 14 else \
        64.9 + 41 * (s.count("G") + s.count("C") - 16.4) / len(s)
    return {"tool": "analyze_dna", "basis": IN_SILICO, "length_nt": len(s), "gc_percent": round(gc, 1),
            "melting_temp_c": round(tm, 1), "reverse_complement": revcomp(s)[:120] + ("…" if len(s) > 120 else ""),
            "translation_frame1": translate(s)[:80], "orfs": orfs[:5],
            "longest_orf_aa": orfs[0]["length_aa"] if orfs else 0}


def mutate_dna(seq: str, position: int, new_base: str, frame: int = 0) -> Dict[str, Any]:
    """Effect of a point mutation at 1-based `position` in a coding sequence read in `frame`."""
    s = _clean((seq or "").replace("U", "T"), "ACGTN")
    nb = _clean(new_base, "ACGT")
    if not 1 <= position <= len(s):
        raise ValueError(f"position must be 1..{len(s)}")
    mut = s[:position - 1] + nb + s[position:]
    ci = (position - 1 - frame) // 3
    if ci < 0:
        return {"tool": "mutate_dna", "basis": IN_SILICO, "effect": "outside the coding frame"}
    c0, c1 = s[frame + 3 * ci: frame + 3 * ci + 3], mut[frame + 3 * ci: frame + 3 * ci + 3]
    a0, a1 = CODON.get(c0, "X"), CODON.get(c1, "X")
    effect = "silent (synonymous)" if a0 == a1 else "nonsense (premature stop)" if a1 == "*" else \
        "stop-loss" if a0 == "*" else "missense"
    return {"tool": "mutate_dna", "basis": IN_SILICO, "codon_number": ci + 1, "codon": f"{c0} -> {c1}",
            "amino_acid": f"{AA3.get(a0, a0)}{ci + 1}{AA3.get(a1, a1)}", "effect": effect,
            "protein_truncated_at": ci + 1 if a1 == "*" else None}


def align(a: str, b: str, match: int = 1, mismatch: int = -1, gap: int = -2) -> Dict[str, Any]:
    """Global (Needleman-Wunsch) alignment of two DNA or protein sequences (up to ~3000 residues each)."""
    a, b = re.sub(r"\s", "", a or "").upper(), re.sub(r"\s", "", b or "").upper()
    if not a or not b:
        raise ValueError("two sequences are needed")
    if len(a) * len(b) > 4_000_000:
        raise ValueError("sequences too long for this alignment (limit about 2000 x 2000 residues)")
    n, m = len(a), len(b)
    sc = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        sc[i][0] = i * gap
    for j in range(1, m + 1):
        sc[0][j] = j * gap
    for i in range(1, n + 1):
        ai, prev, cur = a[i - 1], sc[i - 1], sc[i]
        for j in range(1, m + 1):
            d = prev[j - 1] + (match if ai == b[j - 1] else mismatch)
            u = prev[j] + gap
            l = cur[j - 1] + gap
            cur[j] = d if d >= u and d >= l else (u if u >= l else l)
    sc = np.array(sc)
    i, j, al_a, al_b = n, m, [], []
    while i > 0 or j > 0:
        if i > 0 and j > 0 and sc[i, j] == sc[i - 1, j - 1] + (match if a[i - 1] == b[j - 1] else mismatch):
            al_a.append(a[i - 1]); al_b.append(b[j - 1]); i -= 1; j -= 1
        elif i > 0 and sc[i, j] == sc[i - 1, j] + gap:
            al_a.append(a[i - 1]); al_b.append("-"); i -= 1
        else:
            al_a.append("-"); al_b.append(b[j - 1]); j -= 1
    al_a, al_b = "".join(reversed(al_a)), "".join(reversed(al_b))
    ident = sum(1 for x, y in zip(al_a, al_b) if x == y and x != "-")
    return {"tool": "align", "basis": IN_SILICO, "score": int(sc[n, m]), "identity_percent": round(100 * ident / len(al_a), 1),
            "gaps": al_a.count("-") + al_b.count("-"), "aligned_length": len(al_a),
            "alignment_preview": [al_a[:80], "".join("|" if x == y else " " for x, y in zip(al_a[:80], al_b[:80])), al_b[:80]]}


# --------------------------------------------------------------------------- proteins
_MASS = {"A": 71.0788, "R": 156.1875, "N": 114.1038, "D": 115.0886, "C": 103.1388, "E": 129.1155, "Q": 128.1307,
         "G": 57.0519, "H": 137.1411, "I": 113.1594, "L": 113.1594, "K": 128.1741, "M": 131.1926, "F": 147.1766,
         "P": 97.1167, "S": 87.0782, "T": 101.1051, "W": 186.2132, "Y": 163.1760, "V": 99.1326}
_KD = {"A": 1.8, "R": -4.5, "N": -3.5, "D": -3.5, "C": 2.5, "Q": -3.5, "E": -3.5, "G": -0.4, "H": -3.2, "I": 4.5,
       "L": 3.8, "K": -3.9, "M": 1.9, "F": 2.8, "P": -1.6, "S": -0.8, "T": -0.7, "W": -0.9, "Y": -1.3, "V": 4.2}
_PKA = {"Nterm": 9.0, "Cterm": 2.0, "K": 10.5, "R": 12.4, "H": 6.0, "D": 3.65, "E": 4.25, "C": 8.3, "Y": 10.1}


def _charge(seq: str, ph: float) -> float:
    pos = 1 / (1 + 10 ** (ph - _PKA["Nterm"])) + sum(seq.count(a) / (1 + 10 ** (ph - _PKA[a])) for a in "KRH")
    neg = 1 / (1 + 10 ** (_PKA["Cterm"] - ph)) + sum(seq.count(a) / (1 + 10 ** (_PKA[a] - ph)) for a in "DECY")
    return pos - neg


def protein_properties(seq: str, window: int = 19) -> Dict[str, Any]:
    p = _clean(seq, "ACDEFGHIKLMNPQRSTVWYX*").replace("*", "")
    std = p.replace("X", "")
    mass = sum(_MASS.get(a, 110.0) for a in p) + 18.015
    lo, hi = 0.0, 14.0
    for _ in range(60):                                       # pI by bisection on net charge
        mid = (lo + hi) / 2
        if _charge(std, mid) > 0:
            lo = mid
        else:
            hi = mid
    gravy = sum(_KD.get(a, 0) for a in std) / max(1, len(std))
    tm = []
    if len(std) >= window:
        vals = np.convolve([_KD.get(a, 0) for a in std], np.ones(window) / window, mode="valid")
        i = 0
        while i < len(vals):
            if vals[i] >= 1.6:                                # Kyte-Doolittle transmembrane threshold
                j = i
                while j < len(vals) and vals[j] >= 1.6:
                    j += 1
                tm.append({"start": i + 1, "end": j + window - 1, "mean_hydropathy": round(float(vals[i:j].max()), 2)})
                i = j
            else:
                i += 1
    comp = {a: round(100 * std.count(a) / max(1, len(std)), 1) for a in sorted(set(std))}
    return {"tool": "protein_properties", "basis": IN_SILICO, "length_aa": len(p), "molecular_weight_kda": round(mass / 1000, 2),
            "isoelectric_point": round((lo + hi) / 2, 2), "net_charge_ph7_4": round(_charge(std, 7.4), 1),
            "gravy_hydropathy": round(gravy, 3), "aromaticity": round(sum(std.count(a) for a in "FWY") / max(1, len(std)), 3),
            "extinction_coefficient_280nm": std.count("W") * 5500 + std.count("Y") * 1490 + (std.count("C") // 2) * 125,
            "predicted_transmembrane_segments": tm[:10],
            "composition_percent": dict(sorted(comp.items(), key=lambda x: -x[1])[:8]),
            "note": "GRAVY > 0 = hydrophobic; transmembrane segments are a Kyte-Doolittle prediction only"}


# --------------------------------------------------------------------------- viruses
def simulate_virus(beta: float = 2.7e-5, delta: float = 4.0, p: float = 1.2e-2, c: float = 3.0,
                   T0: float = 4e8, V0: float = 9.3e-2, days: float = 12.0, drug_efficacy: float = 0.0,
                   drug_target: str = "entry", drug_start_day: float = 0.0) -> Dict[str, Any]:
    """Target-cell-limited model:  dT/dt = -bTV,  dI/dt = bTV - dI,  dV/dt = pI - cV.
    An antiviral with efficacy e (0-1) blocks entry (b -> (1-e)b) or production (p -> (1-e)p) from drug_start_day."""
    from scipy.integrate import solve_ivp

    def run(eff):
        def f(t, y):
            T, I, V = y
            on = eff if t >= drug_start_day else 0.0
            b = beta * (1 - on) if drug_target == "entry" else beta
            pp = p * (1 - on) if drug_target != "entry" else p
            return [-b * T * V, b * T * V - delta * I, pp * I - c * V]
        t = np.linspace(0, days, int(days * 48) + 1)
        sol = solve_ivp(f, (0, days), [T0, 0.0, V0], t_eval=t, method="LSODA", rtol=1e-6, atol=1e-9)
        V = np.maximum(sol.y[2], 1e-12)
        return t, V, sol.y[0]
    t, V, T = run(0.0)
    r0 = beta * p * T0 / (c * delta)
    out = {"tool": "simulate_virus", "basis": IN_SILICO, "R0_within_host": round(r0, 2),
           "peak_viral_load": float(f"{V.max():.3g}"), "time_to_peak_days": round(float(t[V.argmax()]), 2),
           "target_cells_remaining_percent": round(100 * float(T[-1] / T0), 1),
           "viral_auc_log10": round(float(np.log10(np.trapezoid(V, t) if hasattr(np, "trapezoid") else np.trapz(V, t))), 2),
           "infection_establishes": bool(r0 > 1)}
    if drug_efficacy:
        t2, V2, _ = run(drug_efficacy)
        auc0 = np.trapezoid(V, t) if hasattr(np, "trapezoid") else np.trapz(V, t)
        auc1 = np.trapezoid(V2, t2) if hasattr(np, "trapezoid") else np.trapz(V2, t2)
        r_eff = r0 * (1 - drug_efficacy)
        out.update({"with_drug": {"efficacy": drug_efficacy, "target": drug_target, "start_day": drug_start_day,
                                  "R0_effective": round(r_eff, 2), "peak_viral_load": float(f"{V2.max():.3g}"),
                                  "viral_auc_reduction_percent": round(float(100 * (1 - auc1 / auc0)), 1),
                                  "critical_efficacy_to_stop_spread": round(max(0.0, 1 - 1 / r0), 3) if r0 > 1 else 0.0}})
    out["curve"] = {"day": [round(float(x), 2) for x in t[::12]], "log10_V": [round(float(np.log10(x)), 2) for x in V[::12]]}
    if drug_efficacy:
        out["curve"]["log10_V_with_drug"] = [round(float(np.log10(x)), 2) for x in V2[::12]]
    return out


def simulate_epidemic(R0: float = 2.5, infectious_days: float = 7.0, population: float = 1e6, initial_infected: float = 10,
                      days: int = 365, intervention_reduction: float = 0.0) -> Dict[str, Any]:
    """SIR model; `intervention_reduction` lowers transmission by that fraction."""
    from scipy.integrate import solve_ivp
    gamma = 1 / infectious_days
    beta = R0 * gamma * (1 - intervention_reduction)
    N = population

    def f(t, y):
        S, I, R = y
        return [-beta * S * I / N, beta * S * I / N - gamma * I, gamma * I]
    t = np.linspace(0, days, days + 1)
    sol = solve_ivp(f, (0, days), [N - initial_infected, initial_infected, 0], t_eval=t)
    I = sol.y[1]
    return {"tool": "simulate_epidemic", "basis": IN_SILICO, "R_effective": round(R0 * (1 - intervention_reduction), 2),
            "peak_infected": int(I.max()), "peak_day": int(t[I.argmax()]),
            "attack_rate_percent": round(100 * float(sol.y[2][-1] / N), 1),
            "herd_immunity_threshold_percent": round(100 * max(0.0, 1 - 1 / R0), 1) if R0 > 0 else None}


# --------------------------------------------------------------------------- drugs
def dose_response(dose: float, ic50: float, hill: float = 1.0, emax: float = 100.0, baseline: float = 0.0) -> Dict[str, Any]:
    """Effect (% inhibition) at a dose for a Hill / Emax curve."""
    if dose < 0 or ic50 <= 0:
        raise ValueError("dose must be >= 0 and IC50 > 0")
    eff = baseline + (emax - baseline) * dose ** hill / (ic50 ** hill + dose ** hill) if dose > 0 else baseline
    return {"tool": "dose_response", "basis": IN_SILICO, "effect_percent": round(eff, 1),
            "fraction_of_ic50": round(dose / ic50, 4),
            "dose_for_90_percent": round(ic50 * 9 ** (1 / hill), 4), "dose_for_10_percent": round(ic50 / 9 ** (1 / hill), 4)}


def fit_dose_response(doses: Sequence[float], effects: Sequence[float]) -> Dict[str, Any]:
    """Fit IC50 / Hill / Emax to measured % effects (e.g. % inhibition of proliferation)."""
    from scipy.optimize import curve_fit
    d = np.asarray(doses, float)
    e = np.asarray(effects, float)
    if len(d) < 4 or len(d) != len(e):
        raise ValueError("need at least 4 dose / effect pairs")
    pos = d > 0

    def hill(x, top, ic50, h):
        return top * x ** h / (ic50 ** h + x ** h)
    p0 = [max(e.max(), 1), float(np.median(d[pos])) if pos.any() else 1.0, 1.0]
    (top, ic50, h), cov = curve_fit(hill, np.maximum(d, 1e-12), e, p0=p0, bounds=([0, 1e-12, 0.1], [200, d.max() * 1e3 + 1, 8]),
                                    maxfev=20000)
    pred = hill(np.maximum(d, 1e-12), top, ic50, h)
    ss_res, ss_tot = float(((e - pred) ** 2).sum()), float(((e - e.mean()) ** 2).sum()) or 1.0
    se = np.sqrt(np.diag(cov)) if np.all(np.isfinite(cov)) else [None] * 3
    return {"tool": "fit_dose_response", "basis": IN_SILICO, "ic50": float(f"{ic50:.4g}"), "hill_slope": round(float(h), 2),
            "emax_percent": round(float(top), 1), "r_squared": round(1 - ss_res / ss_tot, 3),
            "ic50_se": float(f"{se[1]:.3g}") if se[1] is not None else None,
            "warning": "IC50 lies outside the tested doses - extrapolated" if not (d[pos].min() <= ic50 <= d.max()) else None}


def pk_one_compartment(dose_mg: float, interval_h: float = 12, ka_per_h: float = 1.0, ke_per_h: float = 0.14,
                       vd_l: float = 300.0, bioavailability: float = 0.55, n_doses: int = 14, mw_g_mol: float = 129.16,
                       target_um: Optional[float] = None) -> Dict[str, Any]:
    """Oral one-compartment model with repeated dosing (defaults approximate metformin 850 mg twice daily)."""
    t = np.linspace(0, interval_h * n_doses, int(interval_h * n_doses * 10) + 1)
    c = np.zeros_like(t)
    for k in range(n_doses):
        tk = t - k * interval_h
        m = tk >= 0
        if abs(ka_per_h - ke_per_h) < 1e-9:
            ka_per_h += 1e-6
        c[m] += bioavailability * dose_mg * ka_per_h / (vd_l * (ka_per_h - ke_per_h)) * (np.exp(-ke_per_h * tk[m]) - np.exp(-ka_per_h * tk[m]))
    um = c / mw_g_mol * 1000                                   # mg/L -> µM
    last = t >= interval_h * (n_doses - 1)
    out = {"tool": "pk_one_compartment", "basis": IN_SILICO, "cmax_mg_l": round(float(c.max()), 3), "cmax_um": round(float(um.max()), 2),
           "steady_state_trough_um": round(float(um[last].min()), 2),
           "auc_last_interval_mg_h_l": round(float((np.trapezoid if hasattr(np, "trapezoid") else np.trapz)(c[last], t[last])), 2),
           "half_life_h": round(math.log(2) / ke_per_h, 2)}
    if target_um:
        out["time_above_target_percent"] = round(100 * float((um[last] >= target_um).mean()), 1)
    return out


def _to_um(value: float, unit: str, mw: Optional[float] = None) -> Optional[float]:
    u = (unit or "").lower().replace("μ", "µ").replace("u", "µ") if unit else ""
    table = {"m": 1e6, "mm": 1e3, "µm": 1.0, "nm": 1e-3, "pm": 1e-6}
    if u in table:
        return value * table[u]
    if u in ("mg/l", "µg/ml", "mg/ml", "µg/l", "ng/ml") and mw:
        mgl = value * {"mg/l": 1, "µg/ml": 1, "mg/ml": 1000, "µg/l": 1e-3, "ng/ml": 1e-3}[u]
        return mgl / mw * 1000
    return None


def exposure_vs_potency(in_vitro_value: float, in_vitro_unit: str, human_cmax_um: float, mw_g_mol: Optional[float] = None) -> Dict[str, Any]:
    """Could the concentration that worked in cells ever be reached in people? (ratio in vitro / plasma Cmax)"""
    conc = _to_um(in_vitro_value, in_vitro_unit, mw_g_mol)
    if conc is None:
        raise ValueError("unit not understood (use M, mM, µM, nM, or mg/L with a molecular weight)")
    ratio = conc / max(human_cmax_um, 1e-12)
    verdict = ("within the human plasma range" if ratio <= 1 else "up to 10x above human plasma levels" if ratio <= 10 else
               "10-100x above human plasma levels - relevance to patients uncertain" if ratio <= 100 else
               ">100x above human plasma levels - unlikely to be reached in patients")
    return {"tool": "exposure_vs_potency", "basis": IN_SILICO, "in_vitro_um": round(conc, 3), "human_cmax_um": human_cmax_um,
            "fold_above_plasma": round(ratio, 1), "verdict": verdict}


# --------------------------------------------------------------------------- cells / tumours
def simulate_growth(model: str = "gompertz", n0: float = 1e6, rate_per_day: float = 0.3, capacity: float = 1e9,
                    days: float = 30, drug_kill_per_day: float = 0.0, drug_start_day: float = 0.0) -> Dict[str, Any]:
    """Tumour / cell growth (exponential, logistic or Gompertz) with an optional drug kill rate."""
    from scipy.integrate import solve_ivp

    def rhs(kill):
        def f(t, y):
            n = max(y[0], 1e-9)
            k = kill if t >= drug_start_day else 0.0
            if model == "exponential":
                g = rate_per_day * n
            elif model == "logistic":
                g = rate_per_day * n * (1 - n / capacity)
            else:
                g = rate_per_day * n * math.log(capacity / n)
            return [g - k * n]
        return f
    t = np.linspace(0, days, int(days * 4) + 1)
    ctrl = solve_ivp(rhs(0.0), (0, days), [n0], t_eval=t).y[0]
    out = {"tool": "simulate_growth", "basis": IN_SILICO, "model": model, "final_size_control": float(f"{ctrl[-1]:.3g}"),
           "doubling_time_days_initial": round(math.log(2) / rate_per_day, 2) if model == "exponential" else None}
    if drug_kill_per_day:
        trt = solve_ivp(rhs(drug_kill_per_day), (0, days), [n0], t_eval=t).y[0]
        out.update({"final_size_treated": float(f"{trt[-1]:.3g}"),
                    "growth_inhibition_percent": round(100 * (1 - trt[-1] / ctrl[-1]), 1),
                    "tumour_shrinks": bool(trt[-1] < n0)})
    return out


# --------------------------------------------------------------------------- pathways
_EDGE = re.compile(r"\s*([^\s,;:>|\-][^,;]*?)\s*(--?>|->|→|-\||⊣|-x|inhibits|activates|increases|decreases|reduces|"
                   r"induces|suppresses|blocks)\s*([^,;]+?)\s*$", re.I)


def parse_edges(text: str) -> List[Tuple[str, str, int]]:
    """'metformin -> AMPK; AMPK -| mTOR; mTOR -> proliferation' -> signed edges (+1 activate, -1 inhibit).
    Chains in one line work too: 'metformin -> AMPK -| mTOR -> proliferation'."""
    edges = []
    for part in re.split(r"[;\n]", text or ""):
        part = re.split(r"(?<=[a-z0-9)])\.\s", part)[0]        # the chain, not the explanatory sentence after it
        toks = re.split(r"\s*(-\||⊣|-x|--?>|→|\binhibits\b|\bactivates\b|\bincreases\b|\bdecreases\b|\breduces\b|"
                        r"\binduces\b|\bsuppresses\b|\bblocks\b|\blowers\b|\braises\b|\bpromotes\b|\bstimulates\b|"
                        r"\benhances\b|\battenuates\b|\bdownregulates\b|\bupregulates\b)\s*", part.strip())
        if len(toks) < 3:
            continue
        for k in range(0, len(toks) - 2, 2):
            a, op, b = toks[k].strip(" ."), toks[k + 1].lower(), toks[k + 2].strip(" .")
            if a and b:
                sign = -1 if op in ("-|", "⊣", "-x", "inhibits", "decreases", "reduces", "suppresses", "blocks", "lowers",
                                    "attenuates", "downregulates") else 1
                # 'AMPK activation' and 'AMPK' are the same node
                norm = lambda x: re.sub(r"\s+(?:activation|activity|signall?ing|levels?|expression|phosphorylation)$", "", x, flags=re.I).strip()
                edges.append((norm(a), norm(b), sign))
    return edges


def perturb_pathway(edges: Any, source: str, change: str = "increase", targets: Optional[List[str]] = None) -> Dict[str, Any]:
    """Predicted direction of every downstream node after perturbing `source` (signed path propagation).
    Conflicting paths (one predicts up, another down) are reported as 'ambiguous'."""
    E = parse_edges(edges) if isinstance(edges, str) else [(a, b, int(s)) for a, b, s in edges]
    if not E:
        raise ValueError("no edges: write e.g. 'metformin -> AMPK -| mTOR -> proliferation'")
    key = {n.lower(): n for a, b, _ in E for n in (a, b)}
    src = key.get(source.lower())
    if not src:
        raise ValueError(f"'{source}' is not in the network ({', '.join(sorted(key.values()))})")
    s0 = 1 if change.lower().startswith(("inc", "up", "activ", "over")) else -1
    signs: Dict[str, set] = {src: {s0}}
    paths: Dict[str, List[str]] = {src: [src]}
    frontier = [(src, s0, [src])]
    while frontier:
        node, sg, path = frontier.pop()
        for a, b, s in E:
            if a == node and b not in path:
                ns = sg * s
                if ns not in signs.setdefault(b, set()):
                    signs[b].add(ns)
                    paths.setdefault(b, path + [b])
                    frontier.append((b, ns, path + [b]))
    word = {1: "increase", -1: "decrease"}
    pred = {n: ("ambiguous (opposing paths)" if len(v) > 1 else word[next(iter(v))]) for n, v in signs.items() if n != src}
    out = {"tool": "perturb_pathway", "basis": IN_SILICO, "perturbation": f"{src} {word[s0]}", "predictions": pred,
           "example_paths": {n: " → ".join(p) for n, p in paths.items() if n != src}}
    if targets:
        out["targets"] = {t: pred.get(key.get(t.lower(), t), "not reachable from the source") for t in targets}
    return out


# --------------------------------------------------------------------------- statistics
def power_two_groups(effect_size: Optional[float] = None, n_per_group: Optional[int] = None, alpha: float = 0.05,
                     power: Optional[float] = None) -> Dict[str, Any]:
    """Two-sided two-sample t-test (normal approximation). Give effect size (Cohen's d) and either n or power."""
    from scipy.stats import norm
    if not effect_size or effect_size <= 0:
        raise ValueError("effect_size (Cohen's d) > 0 is required")
    za = norm.ppf(1 - alpha / 2)
    if n_per_group:
        pw = float(norm.cdf(effect_size * math.sqrt(n_per_group / 2) - za))
        return {"tool": "power_two_groups", "basis": IN_SILICO, "effect_size_d": effect_size, "n_per_group": n_per_group,
                "power": round(pw, 3), "adequate": pw >= 0.8}
    pw = power or 0.8
    n = math.ceil(2 * ((za + norm.ppf(pw)) / effect_size) ** 2)
    return {"tool": "power_two_groups", "basis": IN_SILICO, "effect_size_d": effect_size, "target_power": pw,
            "n_per_group_required": n}


def compare_groups(a: Sequence[float], b: Sequence[float]) -> Dict[str, Any]:
    from scipy import stats
    x, y = np.asarray(a, float), np.asarray(b, float)
    if len(x) < 2 or len(y) < 2:
        raise ValueError("each group needs at least 2 values")
    t, p = stats.ttest_ind(x, y, equal_var=False)
    u, pu = stats.mannwhitneyu(x, y, alternative="two-sided")
    sd = math.sqrt(((len(x) - 1) * x.var(ddof=1) + (len(y) - 1) * y.var(ddof=1)) / (len(x) + len(y) - 2)) or 1e-12
    return {"tool": "compare_groups", "basis": IN_SILICO, "mean_a": round(float(x.mean()), 4), "mean_b": round(float(y.mean()), 4),
            "welch_t": round(float(t), 3), "welch_p": float(f"{p:.3g}"), "mann_whitney_p": float(f"{pu:.3g}"),
            "cohens_d": round(float((x.mean() - y.mean()) / sd), 3)}


# --------------------------------------------------------------------------- sequence fetch (optional, online)
def fetch_protein_sequence(gene: str, organism_taxon: int = 9606, http: Any = None) -> Dict[str, Any]:
    """Reviewed UniProt sequence for a gene symbol (needs internet; used only when the user asks)."""
    if http is None:
        from http_client import HttpClient
        http = HttpClient(cache_dir="cache/biodb")
    txt = http.get_text("https://rest.uniprot.org/uniprotkb/search",
                        params={"query": f"(gene_exact:{gene}) AND (organism_id:{organism_taxon}) AND (reviewed:true)",
                                "format": "fasta", "size": "1"})
    seq = "".join(l.strip() for l in (txt or "").splitlines() if l and not l.startswith(">"))
    if not seq:
        raise ValueError(f"no reviewed UniProt entry for {gene} (taxon {organism_taxon})")
    return {"tool": "fetch_protein_sequence", "gene": gene, "taxon": organism_taxon, "sequence": seq,
            "header": next((l for l in txt.splitlines() if l.startswith(">")), "")}


# --------------------------------------------------------------------------- 3D structures and omics (separate modules)
def protein_structure(source: str = "pdb", identifier: str = "", text: str = "", residue: Optional[int] = None,
                      chain: Optional[str] = None, taxon: int = 9606, pockets: bool = True,
                      flexibility: bool = True) -> Dict[str, Any]:
    """Fetch (PDB ID, AlphaFold by gene / UniProt, ESMFold from a sequence) or read an uploaded PDB / mmCIF file, then
    analyse it: secondary structure, confidence, pockets, ligand contacts, flexibility, one residue's context."""
    from structure import full_report
    return full_report(source, identifier, text, None, int(taxon or 9606), pockets=bool(pockets), flex=bool(flexibility),
                       residue=int(residue) if residue not in (None, "") else None, chain=chain or None)


def differential_expression(table: Any = None, samples_a: Optional[List[str]] = None, samples_b: Optional[List[str]] = None,
                            label_a: str = "A", label_b: str = "B", batch: Optional[Dict[str, str]] = None,
                            fdr: float = 0.05, min_log2fc: float = 0.0) -> Dict[str, Any]:
    """Genes x samples table (first column = gene IDs). Two groups of sample columns; optional batch per sample."""
    import omics
    if table is None:
        raise ValueError("upload a gene expression table (genes in rows, samples in columns)")
    t = omics.from_table(table)
    a = [x for x in (samples_a or []) if x in t["samples"]]
    b = [x for x in (samples_b or []) if x in t["samples"]]
    if len(a) < 2 or len(b) < 2:
        raise ValueError("choose at least 2 sample columns for each group")
    if set(a) & set(b):
        raise ValueError("a sample cannot be in both groups")
    idx = [t["samples"].index(x) for x in a + b]
    groups = [label_a] * len(a) + [label_b] * len(b)
    bt = None
    if batch:
        bt = [str(batch.get(x, "")) for x in a + b]
        if any(not x for x in bt):
            raise ValueError("give a batch for every selected sample, or none")
        if len(set(bt)) < 2:
            bt = None
    return omics.differential_expression(t["matrix"][:, idx], t["genes"], [t["samples"][i] for i in idx], groups,
                                         label_a, label_b, batch=bt, fdr=float(fdr), min_lfc=float(min_log2fc))


def single_cell_clusters(table: Any = None, cells_are_rows: bool = True, max_clusters: int = 10) -> Dict[str, Any]:
    """Count table of single cells (cells in rows and genes in columns, or the other way round)."""
    import omics
    if table is None:
        raise ValueError("upload a single-cell count table")
    import numpy as np
    import pandas as pd
    first = table.columns[0]
    if cells_are_rows:
        body = table.drop(columns=[first]) if not pd.api.types.is_numeric_dtype(table[first]) else table
        body = body.apply(pd.to_numeric, errors="coerce")
        body = body.loc[:, ~body.isna().all()].fillna(0)
        genes = [str(c) for c in body.columns]
        X = body.to_numpy(dtype=float)
    else:
        t = omics.from_table(table)
        genes, X = t["genes"], t["matrix"].T
    return omics.cluster_cells(np.asarray(X, dtype=float), genes, cells_are_rows=True,
                               k_range=range(2, max(3, min(int(max_clusters), 20)) + 1))


# --------------------------------------------------------------------------- tool registry (agent + website)
TOOLS = {
    "analyze_dna": (analyze_dna, "genes", "GC %, ORFs, translation of a DNA / RNA sequence"),
    "mutate_dna": (mutate_dna, "genes", "effect of a point mutation (silent / missense / nonsense)"),
    "align": (align, "genes", "global alignment and % identity of two sequences"),
    "protein_properties": (protein_properties, "proteins", "mass, pI, charge, hydropathy, transmembrane segments"),
    "simulate_virus": (simulate_virus, "viruses", "within-host viral dynamics with / without an antiviral"),
    "simulate_epidemic": (simulate_epidemic, "viruses", "SIR epidemic with an intervention"),
    "dose_response": (dose_response, "drugs", "Hill curve effect at a dose"),
    "fit_dose_response": (fit_dose_response, "drugs", "IC50 / Hill slope from dose-effect data"),
    "pk_one_compartment": (pk_one_compartment, "drugs", "plasma concentration over repeated oral dosing"),
    "exposure_vs_potency": (exposure_vs_potency, "drugs", "in-vitro concentration vs human plasma level"),
    "simulate_growth": (simulate_growth, "cells", "tumour / cell growth with a drug kill rate"),
    "perturb_pathway": (perturb_pathway, "pathways", "predicted direction of downstream nodes after a perturbation"),
    "power_two_groups": (power_two_groups, "statistics", "power / sample size for a two-group experiment"),
    "compare_groups": (compare_groups, "statistics", "Welch t-test, Mann-Whitney, Cohen's d"),
    "fetch_protein_sequence": (fetch_protein_sequence, "proteins", "reviewed UniProt sequence for a gene (online)"),
    "protein_structure": (protein_structure, "structures", "3D structure: fetch / read, analyse, pockets, ligand contacts, flexibility"),
    "differential_expression": (differential_expression, "omics", "RNA-seq / microarray: normalisation, moderated t, FDR, batch, PCA"),
    "single_cell_clusters": (single_cell_clusters, "omics", "single-cell counts: clusters and marker genes"),
}
NEEDS_TABLE = {"differential_expression", "single_cell_clusters"}
ONLINE_TOOLS = {"fetch_protein_sequence"}


def run_tool(name: str, **params) -> Dict[str, Any]:
    """Safe entry point for the agent and the website: unknown tools / bad parameters return an error dict."""
    if name not in TOOLS:
        return {"tool": name, "error": f"unknown tool; available: {', '.join(TOOLS)}"}
    fn = TOOLS[name][0]
    import inspect
    allowed = set(inspect.signature(fn).parameters)
    clean = {k: v for k, v in (params or {}).items() if k in allowed}
    try:
        return fn(**clean)
    except Exception as exc:
        return {"tool": name, "error": f"{type(exc).__name__}: {exc}", "basis": IN_SILICO}


def catalogue() -> List[Dict[str, Any]]:
    import inspect
    out = []
    for name, (fn, area, desc) in TOOLS.items():
        params = []
        for p in inspect.signature(fn).parameters.values():
            if p.name in ("http", "table", "text"):
                continue
            params.append({"name": p.name, "default": None if p.default is inspect.Parameter.empty else p.default,
                           "required": p.default is inspect.Parameter.empty})
        out.append({"name": name, "area": area, "description": desc, "params": params, "needs_table": name in NEEDS_TABLE})
    return out


# --------------------------------------------------------------------------- hypothesis testing
_DOSE = re.compile(r"(\d+(?:\.\d+)?)\s*(mM|µM|uM|μM|nM|M)\b")
_N_RE = re.compile(r"\bn\s*(?:=|>=|≥)\s*(\d+)|(\d+)\s*(?:per group|mice per|animals per|replicates)", re.I)


def check_hypothesis(h: Dict[str, Any], human_cmax_um: Optional[float] = None) -> Dict[str, Any]:
    """Run the applicable virtual-lab tools on one hypothesis.

    * mechanism written as a chain (A -> B -| C -> outcome): signed propagation must predict the hypothesis'
      stated direction for the final outcome - otherwise the mechanism is internally inconsistent;
    * a dose named in the hypothesis / experiment is compared with the human plasma level (if known);
    * the proposed experiment's n is checked for power to detect a large (d = 0.8) and medium (d = 0.5) effect.
    Verdicts: 'consistent', 'inconsistent', 'not testable as designed', or 'no in-silico check applicable'."""
    text = " ".join(str(h.get(k) or "") for k in ("statement", "mechanism", "predictions", "next_experiment"))
    checks, verdicts = [], []
    # the mechanism is the chain; the plain statement ('X reduces Y') is only used when no mechanism chain is given
    edges = parse_edges(str(h.get("mechanism") or "")) or parse_edges(str(h.get("statement") or ""))
    if len(edges) >= 2:
        src = edges[0][0]
        final = edges[-1][1]
        res = perturb_pathway(edges, src, "increase")
        predicted = res["predictions"].get(final)
        stated = "decrease" if re.search(r"\b(?:reduc|decreas|inhibit|suppress|lower|block)\w*\b[^.]{0,60}" + re.escape(final.split()[0]),
                                         str(h.get("statement") or ""), re.I) else \
            "increase" if re.search(r"\b(?:increas|promot|induc|enhanc|rais|activat)\w*\b[^.]{0,60}" + re.escape(final.split()[0]),
                                    str(h.get("statement") or ""), re.I) else None
        ok = None if not stated or not predicted or predicted.startswith("ambig") else (predicted == stated)
        checks.append({"check": "mechanism sign consistency", "result": res, "stated_direction": stated,
                       "predicted_direction": predicted,
                       "verdict": "consistent" if ok else "inconsistent" if ok is False else "not determinable"})
        verdicts.append("consistent" if ok else "inconsistent" if ok is False else None)
    doses = _DOSE.findall(text)
    if doses and human_cmax_um:
        v, u = doses[0]
        res = exposure_vs_potency(float(v), u, human_cmax_um)
        checks.append({"check": "dose vs human exposure", "result": res,
                       "verdict": "not testable as designed" if res["fold_above_plasma"] > 100 else "consistent"})
        verdicts.append(checks[-1]["verdict"])
    m = _N_RE.search(text)
    if m:
        n = int(m.group(1) or m.group(2))
        big, med = power_two_groups(0.8, n), power_two_groups(0.5, n)
        need = power_two_groups(0.5)["n_per_group_required"]
        checks.append({"check": "statistical power of the proposed experiment",
                       "result": {"n_per_group": n, "power_large_effect": big["power"], "power_medium_effect": med["power"],
                                  "n_needed_for_medium_effect": need},
                       "verdict": "consistent" if big["adequate"] else "not testable as designed"})
        verdicts.append(checks[-1]["verdict"])
    vs = [v for v in verdicts if v]
    overall = ("inconsistent" if "inconsistent" in vs else "not testable as designed" if "not testable as designed" in vs
               else "consistent" if vs else "no in-silico check applicable")
    return {"basis": IN_SILICO, "verdict": overall, "checks": checks}


def human_cmax_for(text: str) -> Optional[float]:
    """Upper human plasma level (µM) of a drug named in the text, from the dose-relevance reference table."""
    try:
        from evidence_quality import EXPOSURE_REFERENCE
    except Exception:
        return None
    for pat, ref in EXPOSURE_REFERENCE.items():
        if re.search(rf"\b(?:{pat})", text or "", re.I) and ref.get("upper_M"):
            return ref["upper_M"] * 1e6
    return None


def check_hypotheses(hyps: List[Dict[str, Any]], question: str = "") -> List[Dict[str, Any]]:
    """Attach an in-silico check to every hypothesis (in place) and return the summaries."""
    out = []
    for h in hyps or []:
        txt = " ".join(str(h.get(k) or "") for k in ("statement", "mechanism", "next_experiment"))
        res = check_hypothesis(h, human_cmax_um=human_cmax_for(f"{txt} {question}"))
        h["in_silico"] = res
        out.append({"statement": h.get("statement"), **res})
    return out
