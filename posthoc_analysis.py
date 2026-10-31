import json
from itertools import combinations

import numpy as np
import pandas as pd

try:
    from scipy import stats
except ImportError:
    stats = None


class PostHocAnalysisEngine:
    """
    Post-hoc analysis for multi-group biological experiments.

    Features:
      - pairwise Welch t-tests
      - pairwise Mann-Whitney U tests
      - Cohen's d
      - Hedges' g
      - mean differences
      - 95% confidence intervals
      - Holm multiple-comparison correction
      - Bonferroni correction
      - omnibus ANOVA / Kruskal-Wallis context

    This module reports statistical evidence. It does not infer
    biological causation.
    """

    def __init__(self, data=None):
        self.data = data

    def load_csv(self, path):
        self.data = pd.read_csv(path)
        return self.data

    def load_dataframe(self, data):
        if not isinstance(data, pd.DataFrame):
            raise TypeError("data must be a pandas DataFrame.")
        self.data = data.copy()
        return self.data

    def _require(self):
        if self.data is None:
            raise RuntimeError("No dataset is loaded.")

        if stats is None:
            raise RuntimeError(
                "SciPy is required. Install it with: pip install scipy"
            )

    def _validate(self, group_column, outcome_column):
        self._require()

        if group_column not in self.data.columns:
            raise ValueError(
                f"Grouping column not found: {group_column}"
            )

        if outcome_column not in self.data.columns:
            raise ValueError(
                f"Outcome column not found: {outcome_column}"
            )

        if not pd.api.types.is_numeric_dtype(
            self.data[outcome_column]
        ):
            raise ValueError(
                f"Outcome column must be numeric: {outcome_column}"
            )

    # ------------------------------------------------------
    # EFFECT SIZE
    # ------------------------------------------------------

    @staticmethod
    def _effect_sizes(a, b):
        n1 = len(a)
        n2 = len(b)

        mean1 = np.mean(a)
        mean2 = np.mean(b)

        var1 = np.var(a, ddof=1)
        var2 = np.var(b, ddof=1)

        pooled_variance = (
            ((n1 - 1) * var1)
            + ((n2 - 1) * var2)
        ) / (n1 + n2 - 2)

        pooled_sd = np.sqrt(pooled_variance)

        if pooled_sd == 0:
            return None, None

        d = (mean1 - mean2) / pooled_sd

        # Small-sample correction for Hedges' g.
        df = n1 + n2 - 2

        correction = (
            1 - (3 / (4 * df - 1))
        )

        g = d * correction

        return float(d), float(g)

    # ------------------------------------------------------
    # RAW PAIRWISE COMPARISON
    # ------------------------------------------------------

    def _pairwise_comparison(
        self,
        group_a,
        group_b,
        values_a,
        values_b,
    ):
        n_a = len(values_a)
        n_b = len(values_b)

        mean_a = float(np.mean(values_a))
        mean_b = float(np.mean(values_b))

        difference = mean_a - mean_b

        var_a = np.var(values_a, ddof=1)
        var_b = np.var(values_b, ddof=1)

        # Welch t-test.
        t_stat, p_value = stats.ttest_ind(
            values_a,
            values_b,
            equal_var=False
        )

        # Welch-Satterthwaite degrees of freedom.
        numerator = (
            var_a / n_a + var_b / n_b
        ) ** 2

        denominator = (
            ((var_a / n_a) ** 2) / (n_a - 1)
            + ((var_b / n_b) ** 2) / (n_b - 1)
        )

        degrees_of_freedom = (
            numerator / denominator
        )

        standard_error = np.sqrt(
            var_a / n_a
            + var_b / n_b
        )

        critical_value = stats.t.ppf(
            0.975,
            degrees_of_freedom
        )

        ci_low = (
            difference
            - critical_value * standard_error
        )

        ci_high = (
            difference
            + critical_value * standard_error
        )

        # Non-parametric comparison.
        u_stat, mann_p = stats.mannwhitneyu(
            values_a,
            values_b,
            alternative="two-sided"
        )

        cohens_d, hedges_g = self._effect_sizes(
            values_a,
            values_b
        )

        return {
            "group_a": str(group_a),
            "group_b": str(group_b),
            "n_a": int(n_a),
            "n_b": int(n_b),
            "mean_a": mean_a,
            "mean_b": mean_b,
            "mean_difference_a_minus_b": float(
                difference
            ),
            "welch_t": float(t_stat),
            "welch_df": float(
                degrees_of_freedom
            ),
            "raw_p_value": float(p_value),
            "mean_difference_95_ci": [
                float(ci_low),
                float(ci_high),
            ],
            "cohens_d": cohens_d,
            "hedges_g": hedges_g,
            "mann_whitney_u": float(u_stat),
            "mann_whitney_raw_p_value": float(
                mann_p
            ),
        }

    # ------------------------------------------------------
    # MULTIPLE-COMPARISON CORRECTION
    # ------------------------------------------------------

    @staticmethod
    def _bonferroni(p_values):
        m = len(p_values)

        return [
            min(float(p) * m, 1.0)
            for p in p_values
        ]

    @staticmethod
    def _holm(p_values):
        """
        Holm-Bonferroni step-down adjustment.
        """

        m = len(p_values)

        indexed = sorted(
            enumerate(p_values),
            key=lambda x: x[1]
        )

        adjusted = [0.0] * m

        running_max = 0.0

        for rank, (original_index, p_value) in enumerate(
            indexed
        ):
            multiplier = m - rank

            adjusted_value = min(
                float(p_value) * multiplier,
                1.0
            )

            running_max = max(
                running_max,
                adjusted_value
            )

            adjusted[original_index] = running_max

        return adjusted

    # ------------------------------------------------------
    # COMPLETE POST-HOC ANALYSIS
    # ------------------------------------------------------

    def analyze(
        self,
        group_column,
        outcome_column,
    ):
        self._validate(
            group_column,
            outcome_column
        )

        clean = self.data[
            [group_column, outcome_column]
        ].dropna()

        groups = list(
            clean[group_column].unique()
        )

        if len(groups) < 3:
            raise ValueError(
                "Post-hoc multi-group analysis requires "
                "at least 3 groups."
            )

        # --------------------------------------------------
        # Omnibus tests
        # --------------------------------------------------

        group_values = []

        for group in groups:

            values = clean.loc[
                clean[group_column] == group,
                outcome_column
            ].astype(float).to_numpy()

            if len(values) < 2:
                raise ValueError(
                    f"Group '{group}' has fewer than 2 observations."
                )

            group_values.append(values)

        anova_f, anova_p = stats.f_oneway(
            *group_values
        )

        kruskal_h, kruskal_p = stats.kruskal(
            *group_values
        )

        # --------------------------------------------------
        # Pairwise comparisons
        # --------------------------------------------------

        comparisons = []

        for group_a, group_b in combinations(
            groups,
            2
        ):

            values_a = clean.loc[
                clean[group_column] == group_a,
                outcome_column
            ].astype(float).to_numpy()

            values_b = clean.loc[
                clean[group_column] == group_b,
                outcome_column
            ].astype(float).to_numpy()

            comparison = self._pairwise_comparison(
                group_a,
                group_b,
                values_a,
                values_b
            )

            comparisons.append(
                comparison
            )

        # --------------------------------------------------
        # Adjust p-values
        # --------------------------------------------------

        welch_p = [
            item["raw_p_value"]
            for item in comparisons
        ]

        mann_p = [
            item["mann_whitney_raw_p_value"]
            for item in comparisons
        ]

        welch_holm = self._holm(
            welch_p
        )

        welch_bonferroni = self._bonferroni(
            welch_p
        )

        mann_holm = self._holm(
            mann_p
        )

        mann_bonferroni = self._bonferroni(
            mann_p
        )

        for i, comparison in enumerate(
            comparisons
        ):

            comparison[
                "holm_adjusted_p_value"
            ] = float(
                welch_holm[i]
            )

            comparison[
                "bonferroni_adjusted_p_value"
            ] = float(
                welch_bonferroni[i]
            )

            comparison[
                "mann_whitney_holm_adjusted_p_value"
            ] = float(
                mann_holm[i]
            )

            comparison[
                "mann_whitney_bonferroni_adjusted_p_value"
            ] = float(
                mann_bonferroni[i]
            )

            comparison[
                "significant_after_holm_alpha_0_05"
            ] = bool(
                welch_holm[i] < 0.05
            )

        return {
            "analysis": "post_hoc_multi_group_analysis",
            "group_column": group_column,
            "outcome_column": outcome_column,
            "groups": [str(g) for g in groups],
            "number_of_groups": len(groups),
            "number_of_pairwise_comparisons": len(
                comparisons
            ),
            "omnibus_tests": {
                "one_way_anova": {
                    "f_statistic": float(anova_f),
                    "p_value": float(anova_p),
                },
                "kruskal_wallis": {
                    "h_statistic": float(kruskal_h),
                    "p_value": float(kruskal_p),
                },
            },
            "pairwise_comparisons": comparisons,
            "multiple_testing": {
                "methods": [
                    "Holm-Bonferroni",
                    "Bonferroni"
                ],
                "alpha": 0.05
            },
            "scientific_note": (
                "Post-hoc statistical differences do not by themselves "
                "establish biological causation. Interpretation requires "
                "experimental design, controls, randomization, measurement "
                "quality, biological plausibility, and replication."
            )
        }

    # ------------------------------------------------------
    # SAVE
    # ------------------------------------------------------

    def save_results(
        self,
        results,
        output_file="posthoc_analysis.json"
    ):

        with open(
            output_file,
            "w",
            encoding="utf-8"
        ) as file:

            json.dump(
                results,
                file,
                indent=4,
                ensure_ascii=False
            )

        return output_file

    # ------------------------------------------------------
    # REPORT
    # ------------------------------------------------------

    def print_report(self, results):

        print()
        print("=" * 80)
        print("POST-HOC SCIENTIFIC ANALYSIS")
        print("=" * 80)

        print(
            f"Grouping variable: "
            f"{results['group_column']}"
        )

        print(
            f"Outcome variable: "
            f"{results['outcome_column']}"
        )

        print(
            "Groups: "
            + ", ".join(
                results["groups"]
            )
        )

        print()
        print("OMNIBUS TESTS")
        print("-" * 80)

        anova = results[
            "omnibus_tests"
        ]["one_way_anova"]

        kruskal = results[
            "omnibus_tests"
        ]["kruskal_wallis"]

        print(
            f"ANOVA: "
            f"F={anova['f_statistic']:.6f}, "
            f"p={anova['p_value']:.10g}"
        )

        print(
            f"Kruskal-Wallis: "
            f"H={kruskal['h_statistic']:.6f}, "
            f"p={kruskal['p_value']:.10g}"
        )

        print()
        print("PAIRWISE COMPARISONS")
        print("-" * 80)

        for item in results[
            "pairwise_comparisons"
        ]:

            print()
            print(
                f"{item['group_a']} vs "
                f"{item['group_b']}"
            )

            print(
                f"  Mean difference: "
                f"{item['mean_difference_a_minus_b']:.6f}"
            )

            ci = item[
                "mean_difference_95_ci"
            ]

            print(
                "  95% CI: "
                f"[{ci[0]:.6f}, {ci[1]:.6f}]"
            )

            print(
                f"  Welch t: "
                f"{item['welch_t']:.6f}"
            )

            print(
                f"  Raw p: "
                f"{item['raw_p_value']:.10g}"
            )

            print(
                f"  Holm-adjusted p: "
                f"{item['holm_adjusted_p_value']:.10g}"
            )

            print(
                f"  Bonferroni-adjusted p: "
                f"{item['bonferroni_adjusted_p_value']:.10g}"
            )

            print(
                f"  Cohen's d: "
                f"{item['cohens_d']:.6f}"
                if item["cohens_d"] is not None
                else "  Cohen's d: unavailable"
            )

            print(
                f"  Hedges' g: "
                f"{item['hedges_g']:.6f}"
                if item["hedges_g"] is not None
                else "  Hedges' g: unavailable"
            )

            print(
                "  Significant after Holm "
                f"(alpha=0.05): "
                f"{item['significant_after_holm_alpha_0_05']}"
            )

        print()
        print("SCIENTIFIC CAUTION")
        print("-" * 80)
        print(
            "These tests identify statistical differences under the "
            "assumptions of the selected methods. They do not by "
            "themselves establish causation or biological mechanism."
        )


# ==========================================================
# TEST PROGRAM
# ==========================================================

import sys  # noqa: E402

if __name__ == "__main__":
    # --help and closed / interrupted input exit cleanly instead of with a traceback
    import builtins as _bi
    if any(_a in ("-h", "--help") for _a in sys.argv[1:]):
        print("usage: python posthoc_analysis.py\n\nInteractive post-hoc statistics: asks for a CSV dataset path, the group column and the outcome column.\nPress Ctrl+C at any prompt to quit.")
        raise SystemExit(0)
    _real_input = _bi.input

    def _safe_input(prompt=""):
        try:
            return _real_input(prompt)
        except (EOFError, KeyboardInterrupt):
            print("\nNo input - exiting.")
            raise SystemExit(0)
    _bi.input = _safe_input

    print("=" * 80)
    print("POST-HOC STATISTICAL ENGINE")
    print("=" * 80)

    dataset_path = input(
        "\nEnter dataset path (CSV):\n> "
    ).strip()

    group_column = input(
        "Enter grouping column:\n> "
    ).strip()

    outcome_column = input(
        "Enter outcome column:\n> "
    ).strip()

    engine = PostHocAnalysisEngine()

    try:

        engine.load_csv(
            dataset_path
        )

        results = engine.analyze(
            group_column,
            outcome_column
        )

        engine.print_report(
            results
        )

        output = engine.save_results(
            results
        )

        print()
        print(
            f"Detailed results saved to: "
            f"{output}"
        )

    except Exception as error:

        print()
        print("=" * 80)
        print("POST-HOC ENGINE ERROR")
        print("=" * 80)
        print(error)
