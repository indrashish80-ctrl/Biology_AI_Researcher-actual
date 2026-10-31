"""
claim_verifier.py  (v24.2; superseded in v25 by independent_reader.py)
---------------------------
NOTE (v25): this is NOT an independent second reader - it re-runs the engine's own deterministic direction reader on
the first reader's quote, so it can only catch a stored direction that contradicts that code's reading. The real
second reader (a different provider / a different prompt, with its own guarded quotes) is independent_reader.py.
Kept for callers of v24.2.

If BIOSYN_VERIFY_PASS=1, unverified claims are sent to direction_in_text a second time.
Disagreement -> quarantine. Agreement -> accept as-is.

Usage:
  Enabled via env var BIOSYN_VERIFY_PASS=1 in audit_claim or during evidence matrix build.
"""
import os
from typing import Any, Dict
from semantic_audit import direction_in_text, canonical_outcome


VERIFY_PASS = os.getenv("BIOSYN_VERIFY_PASS", "0") == "1"


def verify_direction(
    claim: Dict[str, Any],
    context: Dict[str, Any],
    original_direction: str
) -> Dict[str, Any]:
    """
    Run a second independent pass on the claim's direction.

    Args:
        claim: The claim dict with quote and other fields
        context: Audit context (source_text, title, question, etc.)
        original_direction: The direction from the first pass

    Returns:
        Modified claim dict with verification result or quarantine flag.
    """
    if not VERIFY_PASS:
        return claim

    quote = claim.get("quote") or ""
    outcome = claim.get("outcome") or ""

    if not quote or not outcome:
        return claim

    # Run second pass
    outcome_info = canonical_outcome(outcome)
    second_result = direction_in_text(quote, outcome_info, outcome)
    second_direction = second_result.get("direction")

    # Compare
    if second_direction and second_direction != original_direction:
        # Disagreement: quarantine the claim
        return {
            **claim,
            "verifier_status": "quarantined",
            "verifier_reason": f"Direction disagreement: first={original_direction}, second={second_direction}",
            "second_direction": second_direction,
        }
    else:
        # Agreement: mark as verified by second pass
        return {
            **claim,
            "verifier_status": "verified_agreement",
            "second_direction": second_direction,
        }
