"""
read_results.py
---------------
Interactive terminal viewer for the research knowledge database.
Automatically displays your most recent investigation by default.
"""

from __future__ import annotations

import argparse
from database import ResearchDB


def main():
    parser = argparse.ArgumentParser(description="Inspect research knowledge database.")
    parser.add_argument("--id", type=int, default=None, help="Specific research question ID to inspect")
    parser.add_argument("--list", action="store_true", help="List all research questions in the database")
    args = parser.parse_args()

    db = ResearchDB()

    # 1. Option to list all questions in database
    all_rqs = db._rows(db._conn.execute("SELECT * FROM research_questions ORDER BY id DESC").fetchall())
    if not all_rqs:
        print("No research questions found in database.")
        return

    if args.list:
        print("=" * 80)
        print("ALL INVESTIGATIONS IN DATABASE")
        print("=" * 80)
        for r in all_rqs:
            summary = db.research_question_summary(r["id"])
            print(f"[ID: {r['id']}] \"{r['question']}\"")
            print(f"       -> Papers: {summary['papers']} | Claims: {summary['claims']} | Hypotheses: {summary['hypotheses']}")
        print("=" * 80)
        return

    # 2. Select target question (latest by default)
    if args.id:
        target_rq = db.get_research_question(args.id)
        if not target_rq:
            print(f"Error: Question ID {args.id} not found.")
            return
    else:
        # Default to the most recent investigation
        target_rq = all_rqs[0]

    rq_id = target_rq["id"]
    print("=" * 80)
    print(f"RESEARCH QUESTION [ID: {rq_id}]: {target_rq['question']}")
    print("=" * 80)

    # Summary Counts
    summary = db.research_question_summary(rq_id)
    print(f"\nPapers Ingested: {summary['papers']}")
    print(f"Verified Claims: {summary['claims']}")
    print(f"Hypotheses:      {summary['hypotheses']}")
    print(f"Research Gaps:   {summary['research_gaps']}")
    print(f"Audit Flags:     {summary['audit_flags']}")

    # Display Verified Claims
    claims = db.verified_claims(rq_id)
    quarantined = db.quarantined_claims(rq_id)
    print(f"Quarantined:     {len(quarantined)} claim(s) failed semantic verification (excluded from synthesis)")
    if claims:
        print("\n" + "=" * 80)
        print(f"VERIFIED CLAIMS ({len(claims)} total)")
        print("=" * 80)
        for i, c in enumerate(claims, 1):
            print(f"\n[{i}] {c.get('statement')}")
            print(f"    Paper:     {(c.get('paper_title') or 'Unknown')[:70]}...")
            print(f"    Effect:    {c.get('intervention')} -> {c.get('effect_direction')} in {c.get('measurement')}")
            print(f"    Quote:     \"{c.get('quote')}\"")

    # Display Hypotheses
    hypotheses = db.list_hypotheses(rq_id)
    if hypotheses:
        print("\n" + "=" * 80)
        print(f"FALSIFIABLE HYPOTHESES ({len(hypotheses)} total)")
        print("=" * 80)
        for i, h in enumerate(hypotheses, 1):
            print(f"\n[Hypothesis {i}] {h.get('statement')}")
            print(f"  Mechanism:     {h.get('mechanism')}")
            preds = h.get("predictions") or []
            if preds:
                print(f"  Predictions:   {', '.join(preds[:2])}")
            falsif = h.get("falsification_conditions") or ["N/A"]
            print(f"  Falsification: {falsif[0]}")

    # Display Research Gaps
    gaps = db.list_research_gaps(rq_id)
    if gaps:
        print("\n" + "=" * 80)
        print(f"UNADDRESSED RESEARCH GAPS ({len(gaps)} total)")
        print("=" * 80)
        for i, g in enumerate(gaps, 1):
            print(f"[{i}] [{(g.get('category') or 'General').upper()}] {g.get('description')}")

    # Display Audit Flags
    flags = db.list_audit_flags(rq_id)
    if flags:
        print("\n" + "=" * 80)
        print(f"AUDIT / QUALITY CONTROL FLAGS ({len(flags)} total)")
        print("=" * 80)
        for f in flags:
            print(f"- [{f.get('severity')}] {f.get('issue')} (Ref: {f.get('evidence_reference')})")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()