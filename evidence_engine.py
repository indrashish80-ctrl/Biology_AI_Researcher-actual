import json
import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from statistical_analysis import ScientificStatisticsEngine
from posthoc_analysis import PostHocAnalysisEngine


class EvidenceEngine:
    """
    Unified evidence engine for biological datasets.

    Design principle:
        raw data -> deterministic computation -> structured evidence

    AI interpretation should happen AFTER this engine has produced
    the numerical evidence. The engine itself does not make causal
    or mechanistic claims.
    """

    VERSION = "0.1"

    def __init__(self, dataset_path=None):
        self.dataset_path = dataset_path
        self.data = None

    # ------------------------------------------------------
    # LOAD
    # ------------------------------------------------------

    def load_dataset(self, dataset_path):
        self.dataset_path = dataset_path

        extension = os.path.splitext(
            dataset_path
        )[1].lower()

        if extension in (".csv", ".tsv", ".txt", ".xlsx", ".xls", ".json"):
            from core_utils import read_table
            self.data = read_table(dataset_path)

        else:
            raise ValueError(
                "Supported formats: CSV, XLSX, XLS, JSON"
            )

        return self.data

    # ------------------------------------------------------
    # BASIC DATA EVIDENCE
    # ------------------------------------------------------

    def _data_quality(self):
        df = self.data

        missing = {
            str(column): int(
                df[column].isna().sum()
            )
            for column in df.columns
        }

        duplicate_rows = int(
            df.duplicated().sum()
        )

        numeric_columns = [
            str(column)
            for column in df.columns
            if pd.api.types.is_numeric_dtype(
                df[column]
            )
        ]

        categorical_columns = [
            str(column)
            for column in df.columns
            if column not in numeric_columns
        ]

        numeric_summary = {}

        for column in numeric_columns:

            values = pd.to_numeric(
                df[column],
                errors="coerce"
            ).dropna()

            if len(values) == 0:
                continue

            numeric_summary[column] = {
                "n": int(len(values)),
                "mean": float(values.mean()),
                "median": float(values.median()),
                "std": float(values.std()),
                "min": float(values.min()),
                "max": float(values.max()),
                "q1": float(values.quantile(0.25)),
                "q3": float(values.quantile(0.75)),
            }

        return {
            "rows": int(len(df)),
            "columns": int(len(df.columns)),
            "column_names": [
                str(column)
                for column in df.columns
            ],
            "numeric_columns": numeric_columns,
            "categorical_columns": categorical_columns,
            "missing_values": missing,
            "total_missing_values": int(
                sum(missing.values())
            ),
            "duplicate_rows": duplicate_rows,
            "numeric_summary": numeric_summary,
        }

    # ------------------------------------------------------
    # CORRELATIONS
    # ------------------------------------------------------

    def _correlations(self):
        numeric = self.data.select_dtypes(
            include=[np.number]
        )

        if numeric.shape[1] < 2:
            return {}

        matrix = numeric.corr(
            method="pearson"
        )

        result = {}

        for column_a in matrix.columns:

            for column_b in matrix.columns:

                if str(column_a) >= str(column_b):
                    continue

                value = matrix.loc[
                    column_a,
                    column_b
                ]

                if pd.isna(value):
                    continue

                result[
                    f"{column_a}__{column_b}"
                ] = float(value)

        return result

    # ------------------------------------------------------
    # STATISTICAL ANALYSIS
    # ------------------------------------------------------

    def analyze_experiment(
        self,
        group_column,
        outcome_column
    ):

        statistics_engine = (
            ScientificStatisticsEngine()
        )

        statistics_engine.load_dataframe(
            self.data
        )

        statistics = (
            statistics_engine.analyze_experiment(
                group_column,
                outcome_column
            )
        )

        posthoc = None

        # Post-hoc analysis is appropriate for
        # experiments with at least three groups.
        unique_groups = (
            self.data[group_column]
            .dropna()
            .unique()
        )

        if len(unique_groups) >= 3:

            posthoc_engine = (
                PostHocAnalysisEngine()
            )

            posthoc_engine.load_dataframe(
                self.data
            )

            try:
                posthoc = posthoc_engine.analyze(
                    group_column,
                    outcome_column
                )
            except ValueError as exc:
                posthoc = {"error": str(exc)}

        return {
            "group_column": group_column,
            "outcome_column": outcome_column,
            "number_of_groups": int(
                len(unique_groups)
            ),
            "statistics": statistics,
            "posthoc": posthoc,
        }

    # ------------------------------------------------------
    # EVIDENCE RECORD
    # ------------------------------------------------------

    def build_evidence_record(
        self,
        group_column=None,
        outcome_column=None
    ):

        if self.data is None:
            if not self.dataset_path:
                raise RuntimeError(
                    "No dataset loaded."
                )

            self.load_dataset(
                self.dataset_path
            )

        record = {
            "evidence_schema": "biology_ai_evidence_v0.1",
            "engine_version": self.VERSION,
            "generated_at_utc": datetime.now(
                timezone.utc
            ).isoformat(),
            "source": {
                "dataset_path": self.dataset_path,
                "file_name": os.path.basename(
                    self.dataset_path
                )
                if self.dataset_path
                else None,
            },

            # This section describes what was actually
            # observed in the supplied dataset.
            "raw_data": {
                "data_quality": self._data_quality(),
                "pearson_correlations": (
                    self._correlations()
                ),
            },

            # Deterministic computations.
            "computed_evidence": {},

            # Reserved explicitly for future AI
            # interpretation. It is intentionally empty.
            "ai_interpretation": {
                "status": "not_generated",
                "claims": [],
                "mechanisms": [],
                "causal_claims": [],
                "hypotheses": [],
            },

            "limitations": [
                "Evidence is limited to the supplied dataset.",
                "Statistical association does not establish causation.",
                "Effect sizes must be interpreted with sample size and experimental design.",
                "Post-hoc p-values are adjusted for multiple comparisons.",
                "Biological mechanism requires independent evidence beyond these statistical calculations.",
            ],
        }

        if group_column and outcome_column:

            record[
                "computed_evidence"
            ]["experiment"] = (
                self.analyze_experiment(
                    group_column,
                    outcome_column
                )
            )

        return record

    # ------------------------------------------------------
    # SAVE
    # ------------------------------------------------------

    def save(
        self,
        record,
        output_file="evidence_record.json"
    ):

        with open(
            output_file,
            "w",
            encoding="utf-8"
        ) as file:

            json.dump(
                record,
                file,
                indent=4,
                ensure_ascii=False
            )

        return output_file

    # ------------------------------------------------------
    # HUMAN REPORT
    # ------------------------------------------------------

    def print_report(self, record):

        quality = record[
            "raw_data"
        ]["data_quality"]

        print()
        print("=" * 80)
        print("UNIFIED BIOLOGICAL EVIDENCE ENGINE")
        print("=" * 80)

        print(
            f"Schema: "
            f"{record['evidence_schema']}"
        )

        print(
            f"Dataset: "
            f"{record['source']['file_name']}"
        )

        print()
        print("DATA")
        print("-" * 80)

        print(
            f"Rows: {quality['rows']}"
        )

        print(
            f"Columns: {quality['columns']}"
        )

        print(
            f"Missing values: "
            f"{quality['total_missing_values']}"
        )

        print(
            f"Duplicate rows: "
            f"{quality['duplicate_rows']}"
        )

        print()
        print("NUMERIC VARIABLES")
        print("-" * 80)

        for name, summary in quality[
            "numeric_summary"
        ].items():

            print(
                f"{name}: "
                f"n={summary['n']}, "
                f"mean={summary['mean']:.6f}, "
                f"median={summary['median']:.6f}, "
                f"std={summary['std']:.6f}"
            )

        correlations = record[
            "raw_data"
        ]["pearson_correlations"]

        if correlations:

            print()
            print("PEARSON CORRELATIONS")
            print("-" * 80)

            for pair, value in correlations.items():

                print(
                    f"{pair}: "
                    f"{value:.6f}"
                )

        experiment = (
            record[
                "computed_evidence"
            ].get("experiment")
        )

        if experiment:

            print()
            print("EXPERIMENTAL ANALYSIS")
            print("-" * 80)

            print(
                f"Grouping variable: "
                f"{experiment['group_column']}"
            )

            print(
                f"Outcome variable: "
                f"{experiment['outcome_column']}"
            )

            print()
            print(
                "Statistical analysis completed."
            )

            if experiment["posthoc"]:

                comparisons = experiment[
                    "posthoc"
                ]["pairwise_comparisons"]

                print(
                    f"Pairwise comparisons: "
                    f"{len(comparisons)}"
                )

                for comparison in comparisons:

                    print(
                        f"  "
                        f"{comparison['group_a']} vs "
                        f"{comparison['group_b']}: "
                        f"Holm p="
                        f"{comparison['holm_adjusted_p_value']:.10g}, "
                        f"d="
                        f"{comparison['cohens_d']:.4f}"
                    )

        print()
        print("EVIDENCE / INTERPRETATION SEPARATION")
        print("-" * 80)

        print(
            "Raw observations and deterministic computations are stored "
            "separately from AI interpretation."
        )

        print(
            "AI interpretation is currently empty by design."
        )

        print()
        print("LIMITATIONS")
        print("-" * 80)

        for limitation in record[
            "limitations"
        ]:

            print(
                f"- {limitation}"
            )

        print()
        print("=" * 80)


# ==========================================================
# COMMAND-LINE PROGRAM
# ==========================================================

import sys  # noqa: E402

if __name__ == "__main__":
    # --help and closed / interrupted input exit cleanly instead of with a traceback
    import builtins as _bi
    if any(_a in ("-h", "--help") for _a in sys.argv[1:]):
        print("usage: python evidence_engine.py\n\nInteractive unified evidence engine: asks for a dataset path and runs statistics + literature comparison.\nPress Ctrl+C at any prompt to quit.")
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
    print("UNIFIED BIOLOGICAL EVIDENCE ENGINE")
    print("=" * 80)

    dataset_path = input(
        "\nEnter dataset path:\n> "
    ).strip()

    group_column = input(
        "\nEnter grouping column "
        "(press Enter to skip experiment analysis):\n> "
    ).strip()

    outcome_column = input(
        "\nEnter outcome column "
        "(press Enter to skip experiment analysis):\n> "
    ).strip()

    if not group_column:
        group_column = None

    if not outcome_column:
        outcome_column = None

    engine = EvidenceEngine()

    try:

        engine.load_dataset(
            dataset_path
        )

        record = engine.build_evidence_record(
            group_column=group_column,
            outcome_column=outcome_column
        )

        engine.print_report(
            record
        )

        output = engine.save(
            record,
            "evidence_record.json"
        )

        print()
        print(
            f"Complete evidence record saved to: "
            f"{output}"
        )

    except Exception as error:

        print()
        print("=" * 80)
        print("EVIDENCE ENGINE ERROR")
        print("=" * 80)
        print(error)
