"""
statistical_analysis.py
-----------------------
Scientific statistical analysis layer:
- Descriptive stats & normality (Shapiro-Wilk)
- Welch t-test & Cohen's d (2-group)
- One-way ANOVA & Kruskal-Wallis (3+ groups)
- Two-way Factorial ANOVA with interaction term (Factor A x Factor B)
- Benjamini-Hochberg False Discovery Rate (FDR) adjustment
- Linear regression & correlation
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

try:
    from scipy import stats
except ImportError:
    stats = None


import re

CONTROL_RE = re.compile(
    r"\b(control|ctrl|vehicle|placebo|untreated|sham|mock|baseline|wild[- ]?type|wt|dmso|saline|pbs|normal)\b",
    re.IGNORECASE,
)


def order_treatment_control(groups: List[Any]):
    """Return (treatment, control) so that 'increase' always means treatment > control.
    Falls back to file order only when no group looks like a control."""
    a, b = groups[0], groups[1]
    a_ctrl, b_ctrl = bool(CONTROL_RE.search(str(a))), bool(CONTROL_RE.search(str(b)))
    if a_ctrl and not b_ctrl:
        return b, a
    return a, b


def benjamini_hochberg(p_values: List[float]) -> List[float]:
    """Computes Benjamini-Hochberg False Discovery Rate (FDR) adjusted q-values."""
    m = len(p_values)
    if m == 0:
        return []
    indexed = sorted(enumerate(p_values), key=lambda x: x[1])
    q_vals = [1.0] * m
    cum_min = 1.0
    for rank, (orig_idx, p) in reversed(list(enumerate(indexed, start=1))):
        q = min(cum_min, (p * m) / rank)
        cum_min = q
        q_vals[orig_idx] = float(min(1.0, max(0.0, q)))
    return q_vals


class ScientificStatisticsEngine:
    def __init__(self, data: Optional[pd.DataFrame] = None):
        self.data = data.copy() if data is not None else None

    def load_dataframe(self, data: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(data, pd.DataFrame):
            raise TypeError("data must be a pandas DataFrame.")
        self.data = data.copy()
        return self.data

    def load_csv(self, path: str) -> pd.DataFrame:
        self.data = pd.read_csv(path)
        return self.data

    def _require_data(self) -> None:
        if self.data is None:
            raise RuntimeError("No dataset is loaded.")

    def _require_scipy(self) -> None:
        if stats is None:
            raise RuntimeError("SciPy is required: pip install scipy")

    def _validate_columns(self, *columns: str) -> None:
        self._require_data()
        for col in columns:
            if col not in self.data.columns:
                raise ValueError(f"Column '{col}' not found in dataset.")

    # --- Two-Group Comparison ---

    def two_group_comparison(
        self,
        group_column: str,
        value_column: str,
        group_a: Optional[Any] = None,
        group_b: Optional[Any] = None,
    ) -> Dict[str, Any]:
        self._require_scipy()
        self._validate_columns(group_column, value_column)

        clean = self.data[[group_column, value_column]].dropna()
        groups = list(clean[group_column].unique())

        if group_a is None or group_b is None:
            if len(groups) != 2:
                raise ValueError(f"Found {len(groups)} groups; supply group_a and group_b explicitly.")
            # group_a = treatment, group_b = control (so mean difference = treatment - control)
            group_a, group_b = order_treatment_control(groups)

        a = clean.loc[clean[group_column] == group_a, value_column].astype(float).to_numpy()
        b = clean.loc[clean[group_column] == group_b, value_column].astype(float).to_numpy()

        if len(a) < 2 or len(b) < 2:
            raise ValueError("Each group must contain at least 2 observations.")

        mean_a, mean_b = float(np.mean(a)), float(np.mean(b))
        diff = mean_a - mean_b

        # Welch's t-test
        t_stat, p_val = stats.ttest_ind(a, b, equal_var=False)

        var_a, var_b = np.var(a, ddof=1), np.var(b, ddof=1)
        na, nb = len(a), len(b)
        se = np.sqrt(var_a / na + var_b / nb)
        num = (var_a / na + var_b / nb) ** 2
        den = ((var_a / na) ** 2) / (na - 1) + ((var_b / nb) ** 2) / (nb - 1)
        df = num / den if den > 0 else 1.0
        crit = stats.t.ppf(0.975, df)

        pooled_sd = np.sqrt((((na - 1) * var_a) + ((nb - 1) * var_b)) / (na + nb - 2))
        cohens_d = diff / pooled_sd if pooled_sd > 0 else 0.0

        u_stat, mann_p = stats.mannwhitneyu(a, b, alternative="two-sided")

        return {
            "analysis": "two_group_comparison",
            "group_a": str(group_a),
            "group_b": str(group_b),
            "n_a": na,
            "n_b": nb,
            "mean_a": mean_a,
            "mean_b": mean_b,
            "mean_difference_a_minus_b": float(diff),
            "mean_difference_95_ci": [float(diff - crit * se), float(diff + crit * se)],
            "welch_t": float(t_stat),
            "welch_df": float(df),
            "welch_p_value": float(p_val),
            "cohens_d": float(cohens_d),
            "mann_whitney_u": float(u_stat),
            "mann_whitney_p_value": float(mann_p),
        }

    # --- Two-Way Factorial ANOVA ---

    def two_way_anova(
        self,
        factor_a: str,
        factor_b: str,
        outcome: str,
    ) -> Dict[str, Any]:
        """Calculates Type I / Balanced Two-Way ANOVA with Factor A x Factor B interaction."""
        self._require_scipy()
        self._validate_columns(factor_a, factor_b, outcome)

        df_clean = self.data[[factor_a, factor_b, outcome]].dropna()
        n_total = len(df_clean)
        y = df_clean[outcome].astype(float).to_numpy()
        grand_mean = np.mean(y)
        ss_total = np.sum((y - grand_mean) ** 2)

        # Factor A
        levels_a = df_clean[factor_a].unique()
        levels_b = df_clean[factor_b].unique()

        ss_a = sum(
            len(df_clean[df_clean[factor_a] == a]) * (df_clean[df_clean[factor_a] == a][outcome].mean() - grand_mean) ** 2
            for a in levels_a
        )
        df_a = len(levels_a) - 1

        # Factor B
        ss_b = sum(
            len(df_clean[df_clean[factor_b] == b]) * (df_clean[df_clean[factor_b] == b][outcome].mean() - grand_mean) ** 2
            for b in levels_b
        )
        df_b = len(levels_b) - 1

        # Subcells
        ss_cells = 0.0
        for a in levels_a:
            for b in levels_b:
                sub = df_clean[(df_clean[factor_a] == a) & (df_clean[factor_b] == b)][outcome]
                if len(sub) > 0:
                    ss_cells += len(sub) * (sub.mean() - grand_mean) ** 2

        ss_axb = max(0.0, ss_cells - ss_a - ss_b)
        df_axb = df_a * df_b

        # Error
        ss_error = max(1e-12, ss_total - ss_cells)
        df_error = n_total - (len(levels_a) * len(levels_b))

        ms_error = ss_error / max(1, df_error)
        ms_a = ss_a / max(1, df_a)
        ms_b = ss_b / max(1, df_b)
        ms_axb = ss_axb / max(1, df_axb)

        f_a = ms_a / ms_error
        f_b = ms_b / ms_error
        f_axb = ms_axb / ms_error

        p_a = 1.0 - stats.f.cdf(f_a, df_a, df_error)
        p_b = 1.0 - stats.f.cdf(f_b, df_b, df_error)
        p_axb = 1.0 - stats.f.cdf(f_axb, df_axb, df_error)

        return {
            "analysis": "two_way_factorial_anova",
            "factor_a": {"name": factor_a, "levels": [str(x) for x in levels_a], "f": float(f_a), "p_value": float(p_a)},
            "factor_b": {"name": factor_b, "levels": [str(x) for x in levels_b], "f": float(f_b), "p_value": float(p_b)},
            "interaction": {"name": f"{factor_a} x {factor_b}", "f": float(f_axb), "p_value": float(p_axb)},
            "model_fit": {"r_squared": float((ss_total - ss_error) / ss_total) if ss_total > 0 else 0.0},
        }

    # --- Experiment Dispatcher ---

    def analyze_experiment(self, group_column: str, outcome_column: str) -> Dict[str, Any]:
        self._validate_columns(group_column, outcome_column)
        clean = self.data[[group_column, outcome_column]].dropna()
        n_groups = clean[group_column].nunique()

        res: Dict[str, Any] = {
            "group_column": group_column,
            "outcome_column": outcome_column,
            "n_groups": n_groups,
        }

        if n_groups == 2:
            res["two_group_comparison"] = self.two_group_comparison(group_column, outcome_column)
        elif n_groups >= 3:
            groups_data = [sub[outcome_column].astype(float).values for _, sub in clean.groupby(group_column)]
            f_stat, anova_p = stats.f_oneway(*groups_data)
            h_stat, kruskal_p = stats.kruskal(*groups_data)
            res["multi_group_comparison"] = {
                "anova_f": float(f_stat),
                "anova_p_value": float(anova_p),
                "kruskal_h": float(h_stat),
                "kruskal_p_value": float(kruskal_p),
            }
        return res