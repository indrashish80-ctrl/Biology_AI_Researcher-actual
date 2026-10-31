"""
evidence_bridge.py
------------------
Bridges user experimental tabular datasets (CSV/Excel) with literature claims.
Dynamically infers factors and outcomes without hardcoded biological assumptions.
"""

from __future__ import annotations

import re
from typing import Any, Dict, Optional
from database import ResearchDB
from evidence_engine import EvidenceEngine


class EvidenceBridge:
    def __init__(self, db: ResearchDB):
        self.db = db
        self.evidence_engine = EvidenceEngine()

    def compare_experiment_to_literature(
        self,
        research_question_id: int,
        dataset_path: str,
        group_col: str,
        outcome_col: str,
        intervention_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        # every run analyses ITS dataset: earlier runs' experiment records are not mixed into this one
        if hasattr(self.db, "clear_experiment_evidence"):
            self.db.clear_experiment_evidence(research_question_id)
        # 1. Load dataset & calculate empirical statistics deterministically
        try:
            self.evidence_engine.load_dataset(dataset_path)
            record = self.evidence_engine.build_evidence_record(group_column=group_col, outcome_column=outcome_col)
        except Exception as exc:
            stmt = (f"Experimental dataset could not be analysed ({group_col} -> {outcome_col}): {exc}. "
                    "Check that the outcome column is numeric and every group has at least 2 rows.")
            eid = self.db.add_evidence(research_question_id=research_question_id, evidence_type="CALCULATED",
                                       statement=stmt, source_type="experiment", confidence="low",
                                       value={"dataset_path": dataset_path, "error": str(exc)})
            return {"evidence_id": eid, "statement": stmt, "empirical_direction": "unclear", "comparisons": [],
                    "replications": 0, "divergences": 0, "matches_found": 0, "error": str(exc)}
        exp_stats = record.get("computed_evidence", {}).get("experiment", {}).get("statistics", {})

        two_group = exp_stats.get("two_group_comparison")
        diff = two_group.get("mean_difference_a_minus_b", 0) if two_group else 0
        p_val = two_group.get("welch_p_value", 1.0) if two_group else 1.0
        d_val = two_group.get("cohens_d") if two_group else None

        empirical_dir = "no_effect" if p_val >= 0.05 else ("increase" if diff > 0 else "decrease")

        # 2. Match the experiment to published claims on the SAME endpoint (canonical family, e.g. 'ki67_index'
        #    -> proliferation) and the same agent (named, or the treated group's name). Only this run's verified
        #    claims are compared; the experiment itself never becomes a literature claim.
        from semantic_audit import canonical_outcome
        claims = self.db.verified_claims(research_question_id)
        comparisons = []
        replications, divergences, matches = 0, 0, 0
        out_fam = canonical_outcome(outcome_col.replace("_", " "))["family"]
        treated = (two_group or {}).get("group_a") or ""
        agent_tokens = {t for t in re.findall(r"[a-z][a-z0-9\-]{2,}", (intervention_name or treated).lower())
                        if t not in ("treated", "treatment", "group", "drug", "arm", "high", "low", "dose")}
        out_tokens = {t for t in re.findall(r"[a-z][a-z0-9\-]{2,}", outcome_col.lower().replace("_", " "))}

        for c in claims:
            c_fam = c.get("canonical_outcome") or canonical_outcome(c.get("measurement") or "")["family"]
            same_endpoint = (c_fam == out_fam) if not str(out_fam).startswith("other:") else \
                bool(out_tokens & set(re.findall(r"[a-z][a-z0-9\-]{2,}", (c.get("measurement") or "").lower())))
            same_agent = not agent_tokens or bool(agent_tokens & set(re.findall(r"[a-z][a-z0-9\-]{2,}",
                                                                                 (c.get("intervention") or "").lower())))
            if not (same_endpoint and same_agent):
                continue

            matches += 1
            pub_dir = c.get("canonical_direction") or c.get("effect_direction") or "unclear"
            status = "INCONCLUSIVE"

            if empirical_dir == pub_dir and empirical_dir != "unclear":
                status = "REPLICATED_PUBLISHED_EFFECT"
                replications += 1
            elif (
                empirical_dir in ("increase", "decrease")
                and pub_dir in ("increase", "decrease")
                and empirical_dir != pub_dir
            ):
                status = "DIVERGES_FROM_PUBLISHED_EFFECT"
                divergences += 1

            comparisons.append({
                "claim_id": c["id"],
                "paper": c.get("paper_title"),
                "model": c.get("claim_model"),
                "endpoint": c_fam,
                "published_direction": pub_dir,
                "empirical_direction": empirical_dir,
                "status": status,
                "empirical_p_value": p_val,
                "empirical_cohens_d": d_val,
            })

        stmt = (
            f"Experimental dataset '{dataset_path}' ({group_col} -> {outcome_col}): "
            f"observed {empirical_dir} (Welch p={p_val:.4g}, Cohen's d={d_val if d_val is not None else 0.0:.2f}). "
            f"Evaluated against {matches} literature claims: {replications} replicated, {divergences} diverged."
        )

        eid = self.db.add_evidence(
            research_question_id=research_question_id,
            evidence_type="CALCULATED",
            statement=stmt,
            source_type="experiment",
            confidence="high" if p_val < 0.05 else "medium",
            value={"comparisons": comparisons, "statistics": exp_stats, "dataset_path": dataset_path},
        )

        return {
            "evidence_id": eid,
            "statement": stmt,
            "dataset": dataset_path,
            "group_column": group_col,
            "outcome_column": outcome_col,
            "endpoint_family": out_fam,
            "two_group": two_group,
            "statistics": exp_stats,
            "empirical_direction": empirical_dir,
            "comparisons": comparisons,
            "replications": replications,
            "divergences": divergences,
            "matches_found": matches,
        }