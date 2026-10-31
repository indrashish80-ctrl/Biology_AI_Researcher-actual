"""
synthesis_engine.py
-------------------
Universal scientific synthesis engine. Synthesizes cross-study claims and produces
falsifiable hypotheses tailored dynamically to ANY research question.
"""

from __future__ import annotations

import json
from typing import Any, Dict, Optional
from database import ResearchDB
from llm_json import ask_json


class SynthesisEngine:
    def __init__(self, db: ResearchDB, router: Any = None):
        self.db = db
        self.router = router

    def synthesize(self, research_question_id: int, question: str) -> Dict[str, Any]:
        claims = self.db.verified_claims(research_question_id)
        contradictions = self.db.current_contradictions(research_question_id)

        ai_synthesis = None
        if self.router:
            ai_synthesis = self._ask_ai_synthesis(question, claims, contradictions)

        if not ai_synthesis or not ai_synthesis.get("hypotheses"):
            ai_synthesis = self._dynamic_fallback_synthesis(question, claims)

        for h in ai_synthesis["hypotheses"]:
            hid = self.db.add_hypothesis(
                research_question_id=research_question_id,
                hypothesis={
                    "statement": h["statement"],
                    "mechanism": h["mechanism"],
                    "predictions": h["predictions"],
                    "falsification_conditions": h["falsification_conditions"],
                    "supporting_evidence": [c.get("statement") for c in claims[:3]],
                    "next_experiment": json.dumps(h.get("next_experiment", {})),
                    "status": "proposed",
                },
            )
            if h.get("next_experiment"):
                self.db.add_experiment_design(
                    hypothesis_id=hid,
                    design=h["next_experiment"],
                    power_notes="Sample size >= 12 across 3 independent biological experimental blocks",
                )

        for g in ai_synthesis["research_gaps"]:
            self.db.add_research_gap(
                research_question_id=research_question_id,
                description=g["description"],
                category=g.get("category", "general"),
                supporting_observation=g.get("supporting_observation", ""),
            )

        return {
            "hypotheses": ai_synthesis["hypotheses"],
            "research_gaps": ai_synthesis["research_gaps"],
        }

    def _ask_ai_synthesis(self, question: str, claims: list, contradictions: list) -> Optional[Dict[str, Any]]:
        evidence_text = "\n".join(
            f"- [{(c.get('paper_title') or 'Study')[:45]}] {c.get('intervention')} -> {c.get('effect_direction')} "
            f"in {c.get('measurement')} (p={c.get('p_value')})"
            for c in claims[:12]
        )
        prompt = f"""You are a senior scientific research synthesis engine.
Research Question: {question}

Reported Evidence:
{evidence_text or 'Limited experimental evidence.'}

Generate:
1. Exactly 2 mechanistic, falsifiable hypotheses.
2. For each hypothesis, design a concrete validation experiment with explicit falsification conditions.
3. Exactly 2 unaddressed research gaps.

Return ONLY JSON:
{{
  "hypotheses": [
    {{
      "statement": "...",
      "mechanism": "...",
      "predictions": ["..."],
      "falsification_conditions": ["..."],
      "next_experiment": {{
        "experiment_title": "...",
        "controls": "...",
        "independent_variable": "...",
        "dependent_variable": "...",
        "power_notes": "sample size and replication requirements"
      }}
    }}
  ],
  "research_gaps": [
    {{
      "description": "...",
      "category": "dosage|mechanism|timing|generalization",
      "supporting_observation": "..."
    }}
  ]
}}"""

        def _as_list(v: Any) -> list:
            return v if isinstance(v, list) else ([v] if v else [])

        def validate(obj: Any) -> Dict[str, Any]:
            if not isinstance(obj, dict):
                raise ValueError("Expected JSON object.")
            hyps = [h for h in _as_list(obj.get("hypotheses")) if isinstance(h, dict) and h.get("statement")]
            gaps = []
            for g in _as_list(obj.get("research_gaps")):
                if isinstance(g, str) and g.strip():
                    g = {"description": g.strip()}
                if isinstance(g, dict) and g.get("description"):
                    gaps.append(g)
            for h in hyps:
                h["mechanism"] = h.get("mechanism") or "Unspecified mechanism"
                h["predictions"] = [str(x) for x in _as_list(h.get("predictions"))]
                h["falsification_conditions"] = [str(x) for x in _as_list(h.get("falsification_conditions"))]
                if h.get("next_experiment") is not None and not isinstance(h["next_experiment"], dict):
                    h["next_experiment"] = {"experiment_title": str(h["next_experiment"])}
            if not hyps:
                raise ValueError("No usable hypotheses.")
            return {"hypotheses": hyps, "research_gaps": gaps}

        try:
            return ask_json(self.router, prompt, validate=validate)
        except Exception:
            return None

    def _dynamic_fallback_synthesis(self, question: str, claims: list) -> Dict[str, Any]:
        """Evidence-derived replication hypotheses (same as the main engine) - never a canned template, and
        nothing at all when no verified claim exists."""
        if not claims:
            return {"hypotheses": [], "research_gaps": []}
        from research_engine import ResearchEngine
        return ResearchEngine._deterministic_hypothesis_fallback(None, question, claims, [])