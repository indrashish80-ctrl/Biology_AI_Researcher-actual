"""
strict_answer_layer.py  (v24.2)
--------------------------------
Strict answer layer: answer from RCT/meta-analysis only, rest as context.

When BIOSYN_STRICT_ANSWER=1, the report shows:
- Layer 1 answer from RCT/meta-analysis evidence only
- Other evidence (cell, animal, observational) marked as "Context & mechanism"

Usage:
  from strict_answer_layer import tier_evidence
  tiers = tier_evidence(claims, strict_mode=True)
"""
import os
from typing import Any, Dict, List


STRICT_MODE = os.getenv("BIOSYN_STRICT_ANSWER", "0") == "1"

# Evidence hierarchy: higher tier = higher strength
TIERS = {
    "rct": 5,
    "meta_analysis": 5,
    "observational_human": 3,
    "animal": 2,
    "cell_culture": 1,
    "computational": 0,
}


def tier_evidence(claims: List[Dict[str, Any]], strict_mode: bool = STRICT_MODE) -> Dict[str, List[Dict]]:
    """
    Organize claims by evidence tier.

    Args:
        claims: List of extracted claims with study_type
        strict_mode: If True, only tier 5 (RCT/meta) are answer; rest are context

    Returns:
        Dict with "answer_claims" and "context_claims" keys
    """
    answer_claims = []
    context_claims = []

    for claim in claims:
        study_type = claim.get("study_type", "").lower()
        tier = TIERS.get(study_type, 1)

        if strict_mode:
            # Strict mode: only RCT and meta-analysis count as "answer"
            if tier == 5:
                answer_claims.append({**claim, "tier": tier, "tier_name": "RCT/Meta-analysis"})
            else:
                context_claims.append({**claim, "tier": tier, "tier_name": _tier_name(tier)})
        else:
            # Normal mode: all are answer claims, but marked by tier
            answer_claims.append({**claim, "tier": tier, "tier_name": _tier_name(tier)})

    return {
        "answer_claims": answer_claims,
        "context_claims": context_claims,
        "strict_mode": strict_mode,
    }


def _tier_name(tier: int) -> str:
    """Get human-readable tier name."""
    names = {
        5: "RCT/Meta-analysis",
        3: "Observational (human)",
        2: "Animal studies",
        1: "Cell culture",
        0: "Computational",
    }
    return names.get(tier, "Unknown")
