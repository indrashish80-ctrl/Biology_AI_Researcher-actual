"""
completeness_gate.py  (v24.2)
------------------------------
Completeness gate: adds "INCOMPLETE" banner if AI call success rate falls below threshold.

Checks provider_health summary (calls, failures, rate_limited) and compares to threshold.
If success rate < threshold, report gets banner and reduced certainty.

Usage:
  from completeness_gate import check_completeness
  status = check_completeness(provider_health, threshold=0.95)
"""
from typing import Any, Dict


DEFAULT_COMPLETENESS_THRESHOLD = 0.95  # 95% success rate required


def check_completeness(
    provider_health: Dict[str, Any],
    threshold: float = DEFAULT_COMPLETENESS_THRESHOLD
) -> Dict[str, Any]:
    """
    Check if AI call success rate is above threshold.

    Args:
        provider_health: Summary dict with 'calls', 'failures', 'rate_limited'
        threshold: Minimum success rate required (0.0-1.0)

    Returns:
        {
            "is_complete": bool,
            "success_rate": float,
            "calls": int,
            "failures": int,
            "banner": str (empty if complete, warning text if not)
        }
    """
    calls = provider_health.get("calls", 0)
    failures = provider_health.get("failures", 0)
    rate_limited = provider_health.get("rate_limited", 0)
    # v26.2: measured per REQUEST (a request that failed on Groq and then succeeded on Gemini is complete), not per
    # provider attempt - the old figure called a run '28% complete' when most of its fallbacks had succeeded
    requests = provider_health.get("requests", 0)
    if requests:
        req_failed = provider_health.get("requests_failed", 0)
        rate = (requests - req_failed) / requests
        ok = rate >= threshold
        banner = "" if ok else (
            f"⚠️ INCOMPLETE: only {rate*100:.0f}% of AI requests completed ({req_failed} of {requests} failed on every "
            f"provider; {failures} provider attempt(s) failed in total, {rate_limited} of them rate limits). "
            f"Results may be incomplete or biased - see 'AI provider problems in this run'.")
        return {"is_complete": ok, "success_rate": rate, "calls": requests, "failures": req_failed,
                "attempt_failures": failures, "banner": banner}

    if calls == 0:
        return {
            "is_complete": True,
            "success_rate": 1.0,
            "calls": 0,
            "failures": 0,
            "banner": ""
        }

    success_rate = (calls - failures) / calls if calls > 0 else 0.0
    is_complete = success_rate >= threshold

    if is_complete:
        banner = ""
    else:
        banner = (
            f"⚠️ INCOMPLETE: Only {success_rate*100:.0f}% of AI calls completed "
            f"({failures} failures, {rate_limited} rate-limited). "
            f"Results may be incomplete or biased."
        )

    return {
        "is_complete": is_complete,
        "success_rate": success_rate,
        "calls": calls,
        "failures": failures,
        "banner": banner,
    }
