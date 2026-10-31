"""
regression_tracker.py  (v24.2, fixed in v25)
-------------------------------
Fixed-question regression tracker: snapshots and diffs for verified claims.

Usage:
  from regression_tracker import snapshot_claims, compare_snapshots
  snapshot = snapshot_claims(question, verified_claims, question_version)
  diff = compare_snapshots(old_snapshot, new_snapshot)
"""
import json
from typing import Any, Dict, List
from pathlib import Path
from datetime import datetime, timezone


def snapshot_claims(
    question: str,
    verified_claims: List[Dict[str, Any]],
    question_version: str = "1.0"
) -> Dict[str, Any]:
    """
    Create a snapshot of verified claims for a fixed question.

    Args:
        question: The research question
        verified_claims: List of claims marked as verified
        question_version: Version identifier for the question

    Returns:
        Snapshot dict with timestamp, question, and claim summaries
    """
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "question": question,
        "question_version": question_version,
        "total_claims": len(verified_claims),
        "claims": [
            {
                "id": c.get("id"),
                "intervention": c.get("intervention"),
                "outcome": c.get("outcome") or c.get("measurement"),
                "direction": c.get("direction") or c.get("effect_direction"),
                "quote": c.get("quote", "")[:100],
                "paper": c.get("paper") or c.get("paper_title") or c.get("pmid") or "",
            }
            for c in verified_claims
        ]
    }


def compare_snapshots(
    old: Dict[str, Any],
    new: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Compare two snapshots and report differences.

    Returns:
        Diff report with added, removed, and changed claims
    """
    # v25: claims are matched by CONTENT (paper + outcome + quote), never by database id - ids are renumbered on every
    # run, so the v24.2 version reported every claim of a re-run as removed and added again
    def _key(c):
        import hashlib
        import re as _re
        n = lambda s: _re.sub(r"[^a-z0-9]+", " ", str(s or "").lower()).strip()   # noqa: E731
        return hashlib.sha1(f"{n(c.get('paper'))}|{n(c.get('outcome'))}|{n(c.get('quote'))[:100]}".encode()).hexdigest()[:16]
    old_claims = {_key(c): c for c in old.get("claims", [])}
    new_claims = {_key(c): c for c in new.get("claims", [])}

    added = [c for cid, c in new_claims.items() if cid not in old_claims]
    removed = [c for cid, c in old_claims.items() if cid not in new_claims]
    changed = [
        {"old": old_claims[cid], "new": new_claims[cid]}
        for cid in set(old_claims.keys()) & set(new_claims.keys())
        if old_claims[cid].get("direction") != new_claims[cid].get("direction")
        or old_claims[cid].get("intervention") != new_claims[cid].get("intervention")
    ]

    return {
        "old_timestamp": old.get("timestamp"),
        "new_timestamp": new.get("timestamp"),
        "question": new.get("question"),
        "added": len(added),
        "removed": len(removed),
        "changed": len(changed),
        "details": {
            "added_claims": added[:3],
            "removed_claims": removed[:3],
            "changed_claims": changed[:3],
        }
    }


def main(argv=None) -> int:
    """python regression_tracker.py OLD_DOSSIER.json NEW_DOSSIER.json  - the diff CLI the v24.2 notes promised."""
    import sys
    import dossier_repro
    args = list(argv if argv is not None else sys.argv[1:])
    if len(args) != 2:
        print("usage: python regression_tracker.py OLD_DOSSIER.json NEW_DOSSIER.json")
        return 2
    a, b = (json.loads(Path(x).read_text(encoding="utf-8")) for x in args)
    d = dossier_repro.diff(a, b)
    print(dossier_repro.markdown(d))
    return 0 if d["identical"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
