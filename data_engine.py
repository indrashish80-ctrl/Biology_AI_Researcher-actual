
import json
from pathlib import Path

import numpy as np
import pandas as pd


# ==========================================================
# DATA ENGINE
# ==========================================================

class BiologicalDataEngine:
    """
    Core data-processing engine for the biology research AI.

    Responsibilities:

    - Load CSV, Excel, and JSON datasets
    - Inspect dataset structure
    - Identify numeric and categorical variables
    - Calculate descriptive statistics
    - Calculate correlations
    - Detect missing values
    - Detect duplicate rows
    - Detect numerical outliers
    - Produce machine-readable analysis results

    Important:

    This class performs actual computation.

    It does NOT ask an LLM to calculate statistics.
    """

    SUPPORTED_EXTENSIONS = {
        ".csv",
        ".xlsx",
        ".xls",
        ".json"
    }

    def __init__(self):

        self.data = None
        self.file_path = None

    # ======================================================
    # LOAD DATA
    # ======================================================

    def load_file(self, file_path):
        """
        Load a supported dataset.

        Supported:
            CSV
            XLSX
            XLS
            JSON
        """

        path = Path(file_path)

        if not path.exists():

            raise FileNotFoundError(
                f"Dataset not found: {file_path}"
            )

        extension = path.suffix.lower()

        if extension not in self.SUPPORTED_EXTENSIONS:

            raise ValueError(
                f"Unsupported file type: {extension}\n"
                f"Supported types: "
                f"{', '.join(sorted(self.SUPPORTED_EXTENSIONS))}"
            )

        print()
        print("=" * 80)
        print("LOADING DATASET")
        print("=" * 80)

        print(
            f"File: {path.name}"
        )

        # --------------------------------------------------
        # CSV
        # --------------------------------------------------

        if extension == ".csv":

            self.data = pd.read_csv(
                path
            )

        # --------------------------------------------------
        # Excel
        # --------------------------------------------------

        elif extension in (
            ".xlsx",
            ".xls"
        ):

            self.data = pd.read_excel(
                path
            )

        # --------------------------------------------------
        # JSON
        # --------------------------------------------------

        elif extension == ".json":

            with open(
                path,
                "r",
                encoding="utf-8"
            ) as file:

                json_data = json.load(
                    file
                )

            # Handle normal list-of-records JSON.
            self.data = pd.DataFrame(
                json_data
            )

        self.file_path = str(
            path.resolve()
        )

        print(
            f"Rows: {len(self.data)}"
        )

        print(
            f"Columns: {len(self.data.columns)}"
        )

        return self.data

    # ======================================================
    # BASIC DATASET INFORMATION
    # ======================================================

    def get_basic_info(self):

        if self.data is None:

            raise RuntimeError(
                "No dataset is loaded."
            )

        numeric_columns = list(
            self.data.select_dtypes(
                include=np.number
            ).columns
        )

        categorical_columns = list(
            self.data.select_dtypes(
                exclude=np.number
            ).columns
        )

        return {
            "file": self.file_path,
            "rows": int(len(self.data)),
            "columns": int(len(self.data.columns)),
            "column_names": list(
                self.data.columns
            ),
            "numeric_columns": numeric_columns,
            "categorical_columns": categorical_columns,
            "memory_usage_bytes": int(
                self.data.memory_usage(
                    deep=True
                ).sum()
            )
        }

    # ======================================================
    # MISSING DATA
    # ======================================================

    def analyze_missing_data(self):

        if self.data is None:

            raise RuntimeError(
                "No dataset is loaded."
            )

        missing = self.data.isna().sum()

        result = {}

        for column, count in missing.items():

            if count > 0:

                percentage = (
                    count / len(self.data)
                ) * 100

                result[column] = {
                    "missing_count": int(count),
                    "missing_percentage": float(
                        percentage
                    )
                }

        return result

    # ======================================================
    # DUPLICATES
    # ======================================================

    def analyze_duplicates(self):

        if self.data is None:

            raise RuntimeError(
                "No dataset is loaded."
            )

        duplicate_count = int(
            self.data.duplicated().sum()
        )

        return {
            "duplicate_rows": duplicate_count,
            "duplicate_percentage": float(
                (
                    duplicate_count
                    / max(len(self.data), 1)
                ) * 100
            )
        }

    # ======================================================
    # NUMERIC SUMMARY
    # ======================================================

    def numeric_summary(self):

        if self.data is None:

            raise RuntimeError(
                "No dataset is loaded."
            )

        numeric_data = self.data.select_dtypes(
            include=np.number
        )

        if numeric_data.empty:

            return {}

        summary = {}

        for column in numeric_data.columns:

            series = numeric_data[column].dropna()

            if len(series) == 0:

                continue

            summary[column] = {
                "count": int(series.count()),
                "mean": float(series.mean()),
                "median": float(series.median()),
                "std": float(series.std()),
                "min": float(series.min()),
                "max": float(series.max()),
                "q1": float(series.quantile(0.25)),
                "q3": float(series.quantile(0.75))
            }

        return summary

    # ======================================================
    # CORRELATION ANALYSIS
    # ======================================================

    def correlation_analysis(self):

        if self.data is None:

            raise RuntimeError(
                "No dataset is loaded."
            )

        numeric_data = self.data.select_dtypes(
            include=np.number
        )

        if numeric_data.shape[1] < 2:

            return {}

        correlation_matrix = (
            numeric_data.corr(
                method="pearson"
            )
        )

        result = {}

        columns = list(
            correlation_matrix.columns
        )

        for i, column_a in enumerate(columns):

            for j, column_b in enumerate(columns):

                if j <= i:
                    continue

                value = correlation_matrix.loc[
                    column_a,
                    column_b
                ]

                if pd.isna(value):
                    continue

                result[
                    f"{column_a} vs {column_b}"
                ] = float(value)

        return result

    # ======================================================
    # OUTLIER ANALYSIS
    # ======================================================

    def outlier_analysis(self):

        if self.data is None:

            raise RuntimeError(
                "No dataset is loaded."
            )

        numeric_data = self.data.select_dtypes(
            include=np.number
        )

        result = {}

        for column in numeric_data.columns:

            series = numeric_data[column].dropna()

            if len(series) < 4:

                continue

            q1 = series.quantile(0.25)
            q3 = series.quantile(0.75)

            iqr = q3 - q1

            lower_bound = q1 - (
                1.5 * iqr
            )

            upper_bound = q3 + (
                1.5 * iqr
            )

            outliers = series[
                (series < lower_bound)
                | (series > upper_bound)
            ]

            result[column] = {
                "outlier_count": int(
                    len(outliers)
                ),
                "lower_bound": float(
                    lower_bound
                ),
                "upper_bound": float(
                    upper_bound
                ),
                "outlier_percentage": float(
                    (
                        len(outliers)
                        / len(series)
                    ) * 100
                )
            }

        return result

    # ======================================================
    # CATEGORICAL SUMMARY
    # ======================================================

    def categorical_summary(self):

        if self.data is None:

            raise RuntimeError(
                "No dataset is loaded."
            )

        categorical_data = self.data.select_dtypes(
            exclude=np.number
        )

        result = {}

        for column in categorical_data.columns:

            value_counts = (
                categorical_data[column]
                .value_counts(dropna=False)
            )

            result[column] = {
                str(key): int(value)
                for key, value
                in value_counts.items()
            }

        return result

    # ======================================================
    # FULL ANALYSIS
    # ======================================================

    def analyze(self):

        if self.data is None:

            raise RuntimeError(
                "No dataset is loaded."
            )

        print()
        print("=" * 80)
        print("PROCESSING DATA")
        print("=" * 80)

        result = {
            "dataset": self.get_basic_info(),
            "missing_data": self.analyze_missing_data(),
            "duplicates": self.analyze_duplicates(),
            "numeric_summary": self.numeric_summary(),
            "correlations": self.correlation_analysis(),
            "outliers": self.outlier_analysis(),
            "categorical_summary": self.categorical_summary()
        }

        return result

    # ======================================================
    # SAVE RESULTS
    # ======================================================

    def save_analysis(
        self,
        results,
        output_file="data_analysis.json"
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

        print()
        print(
            f"Analysis saved to: {output_file}"
        )

    # ======================================================
    # HUMAN-READABLE REPORT
    # ======================================================

    def print_report(self, results):

        print()
        print("=" * 80)
        print("DATA ANALYSIS REPORT")
        print("=" * 80)

        dataset = results["dataset"]

        print()
        print("DATASET")
        print("-" * 80)

        print(
            f"Rows: {dataset['rows']}"
        )

        print(
            f"Columns: {dataset['columns']}"
        )

        print(
            "Numeric columns:",
            ", ".join(
                dataset["numeric_columns"]
            )
            if dataset["numeric_columns"]
            else "None"
        )

        print(
            "Categorical columns:",
            ", ".join(
                dataset["categorical_columns"]
            )
            if dataset["categorical_columns"]
            else "None"
        )

        # --------------------------------------------------
        # Missing data
        # --------------------------------------------------

        print()
        print("MISSING DATA")
        print("-" * 80)

        missing = results["missing_data"]

        if missing:

            for column, info in missing.items():

                print(
                    f"{column}: "
                    f"{info['missing_count']} "
                    f"missing "
                    f"({info['missing_percentage']:.2f}%)"
                )

        else:

            print("No missing values detected.")

        # --------------------------------------------------
        # Duplicates
        # --------------------------------------------------

        print()
        print("DUPLICATES")
        print("-" * 80)

        duplicates = results["duplicates"]

        print(
            f"Duplicate rows: "
            f"{duplicates['duplicate_rows']}"
        )

        # --------------------------------------------------
        # Numeric summary
        # --------------------------------------------------

        print()
        print("NUMERIC VARIABLES")
        print("-" * 80)

        numeric_summary = results[
            "numeric_summary"
        ]

        if numeric_summary:

            for column, stats in (
                numeric_summary.items()
            ):

                print()
                print(column)

                print(
                    f"  Mean:   {stats['mean']:.4f}"
                )

                print(
                    f"  Median: {stats['median']:.4f}"
                )

                print(
                    f"  Std:    {stats['std']:.4f}"
                )

                print(
                    f"  Min:    {stats['min']:.4f}"
                )

                print(
                    f"  Max:    {stats['max']:.4f}"
                )

        else:

            print(
                "No numeric variables found."
            )

        # --------------------------------------------------
        # Correlations
        # --------------------------------------------------

        print()
        print("CORRELATIONS")
        print("-" * 80)

        correlations = results[
            "correlations"
        ]

        if correlations:

            for pair, value in (
                correlations.items()
            ):

                print(
                    f"{pair}: "
                    f"{value:.4f}"
                )

        else:

            print(
                "Not enough numeric variables "
                "for correlation analysis."
            )

        # --------------------------------------------------
        # Outliers
        # --------------------------------------------------

        print()
        print("OUTLIERS")
        print("-" * 80)

        outliers = results[
            "outliers"
        ]

        if outliers:

            for column, info in (
                outliers.items()
            ):

                print(
                    f"{column}: "
                    f"{info['outlier_count']} "
                    f"outliers "
                    f"({info['outlier_percentage']:.2f}%)"
                )

        else:

            print(
                "No numerical outlier analysis available."
            )


# ==========================================================
# TEST PROGRAM
# ==========================================================

import sys  # noqa: E402

if __name__ == "__main__":
    # --help and closed / interrupted input exit cleanly instead of with a traceback
    import builtins as _bi
    if any(_a in ("-h", "--help") for _a in sys.argv[1:]):
        print("usage: python data_engine.py\n\nInteractive biological data engine: asks for a dataset path (CSV / Excel / JSON) and the columns to analyse.\nPress Ctrl+C at any prompt to quit.")
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
    print("BIOLOGICAL DATA ENGINE")
    print("=" * 80)

    file_path = input(
        "\nEnter dataset path "
        "(CSV / Excel / JSON):\n> "
    ).strip()

    if not file_path:

        print(
            "\nERROR: No dataset path supplied."
        )

        raise SystemExit(1)

    engine = BiologicalDataEngine()

    try:

        engine.load_file(
            file_path
        )

        results = engine.analyze()

        engine.print_report(
            results
        )

        engine.save_analysis(
            results
        )

        print()
        print("=" * 80)
        print("DATA PROCESSING COMPLETE")
        print("=" * 80)

    except Exception as error:

        print()
        print("=" * 80)
        print("DATA ENGINE ERROR")
        print("=" * 80)

        print(error)

