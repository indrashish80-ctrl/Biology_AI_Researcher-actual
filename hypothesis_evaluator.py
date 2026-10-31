"""
hypothesis_evaluator.py
-----------------------
Closed-Loop Hypothesis Falsification & Empirical Testing Engine:
    1. Ingests experimental dataset (CSV/Excel).
    2. Runs deterministic Welch t-test & Cohen's d via ScientificStatisticsEngine.
    3. Evaluates predictions and explicit falsification conditions.
    4. Automatically updates hypothesis status in SQLite to 'supported', 'weakened', or 'refuted'.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, Optional

from database import ResearchDB
from statistical_analysis import ScientificStatisticsEngine


class HypothesisEvaluator:
    def __init__(self, db: ResearchDB):
        self.db = db
        self.stats_engine = ScientificStatisticsEngine()

    def evaluate_hypothesis_with_data(
        self,
        hypothesis_id: int,
        dataset_path: str,
        group_column: str,
        outcome_column: str,
        expected_direction: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Calculates empirical statistics and tests whether experimental data
        supports, weakens, or refutes a stored hypothesis.
        """
        if expected_direction is None:
            expected_direction = self.infer_expected_direction(hypothesis_id)
        path = Path(dataset_path)
        if not path.exists():
            raise FileNotFoundError(f"Dataset '{dataset_path}' not found.")

        # 1. Deterministic Calculation (CSV, Excel or JSON - the web app accepts all three)
        try:
            from core_utils import read_table
            df = read_table(path)
            self.stats_engine.load_dataframe(df)
            exp_res = self.stats_engine.analyze_experiment(group_column, outcome_column)
        except Exception as exc:
            return self._not_evaluated(hypothesis_id, f"Dataset could not be analysed: {exc}")

        two_group = exp_res.get("two_group_comparison")
        if not two_group:
            return self._not_evaluated(
                hypothesis_id,
                f"Falsification test needs exactly 2 groups (treatment vs control); "
                f"'{group_column}' has {exp_res.get('n_groups')}. Multi-group statistics are in the evidence record.")

        diff = two_group["mean_difference_a_minus_b"]
        p_val = two_group["welch_p_value"]
        cohens_d = two_group["cohens_d"]

        observed_dir = "no_effect" if p_val >= 0.05 else ("increase" if diff > 0 else "decrease")

        # 2. Decision Logic: Supported, Weakened, or Refuted
        if p_val < 0.05 and observed_dir == expected_direction:
            new_status = "supported"
            verdict_detail = f"Data statistically supports hypothesis: observed {observed_dir} (p={p_val:.4g}, d={cohens_d:.2f})."
        elif p_val < 0.05 and observed_dir != expected_direction and observed_dir != "no_effect":
            new_status = "refuted"
            verdict_detail = f"Falsification condition triggered: expected {expected_direction}, observed opposing {observed_dir} (p={p_val:.4g}, d={cohens_d:.2f})."
        else:
            new_status = "weakened"
            verdict_detail = f"Inconclusive / null effect observed (p={p_val:.4g} >= 0.05). Hypothesis not supported under current experimental conditions."

        # 3. Update SQLite Database Record
        self.db.update_hypothesis_status(hypothesis_id, new_status)
        self.db.add_provenance(
            entity_type="hypothesis",
            entity_id=hypothesis_id,
            provenance_type="CALCULATED",
            source_ref=f"dataset:{path.name}",
            details={"dataset": str(path), "group_col": group_column, "outcome_col": outcome_column, "p_val": p_val, "status": new_status},
        )

        return {
            "hypothesis_id": hypothesis_id,
            "new_status": new_status,
            "verdict": verdict_detail,
            "interpretation": ("Statistically consistent with the prediction. This is evidence of CAUSATION only if the "
                               "groups were assigned by an intervention (randomised / controlled manipulation); in an "
                               "observational dataset it is an association."),
            "empirical_statistics": {
                "observed_direction": observed_dir,
                "mean_difference": diff,
                "welch_p_value": p_val,
                "cohens_d": cohens_d,
            },
        }

    _DOWN = re.compile(r"\b(decreas\w*|reduc\w*|lower\w*|inhibit\w*|suppress\w*|attenuat\w*|impair\w*|"
                       r"block\w*|diminish\w*|downregulat\w*|loss of|less)\b", re.IGNORECASE)
    _UP = re.compile(r"\b(increas\w*|enhanc\w*|promot\w*|elevat\w*|rais\w*|stimulat\w*|induc\w*|"
                     r"activat\w*|upregulat\w*|improv\w*|more)\b", re.IGNORECASE)

    def infer_expected_direction(self, hypothesis_id: int) -> str:
        """Read the predicted direction from the hypothesis itself instead of assuming 'increase'."""
        row = next((h for h in self.db.list_hypotheses() if h["id"] == hypothesis_id), None) or {}
        text = " ".join([row.get("statement") or ""] + [str(p) for p in (row.get("predictions") or [])])
        down, up = len(self._DOWN.findall(text)), len(self._UP.findall(text))
        return "decrease" if down > up else "increase"

    @staticmethod
    def _not_evaluated(hypothesis_id: int, reason: str) -> Dict[str, Any]:
        """Status stays 'proposed'; the investigation continues instead of crashing."""
        return {
            "hypothesis_id": hypothesis_id,
            "new_status": "not_evaluated",
            "verdict": reason,
            "empirical_statistics": None,
        }
