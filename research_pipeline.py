"""
research_pipeline.py
--------------------
End-to-end autonomous multi-paper biology research engine with GEO loop closure.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ai_router import AIRouter
from bio_databases import BioDatabases, EntityLinker
from claim_extraction import ClaimExtractor
from contradiction_detection import detect_and_store, extract_conflicts_for_geo
from database import ResearchDB
from dataset_discovery import DatasetDiscovery
from jats_ingest import ingest_jats
from paper_extraction import PaperExtractor
from research_planner import generate_model_plan
from retrieval.source_manager import SourceManager


def run_geo_loop_closure(
    db: ResearchDB,
    research_question_id: int,
    conflicts: list,
    router: any = None,
) -> list:
    """
    Loop Closure: Search NCBI GEO for empirical datasets matching contradictory findings.
    """
    print("\n" + "=" * 80)
    print("EMPIRICAL LOOP CLOSURE: SEARCHING NCBI GEO FOR DISAGREEMENT TOPICS")
    print("=" * 80)

    dd = DatasetDiscovery(db=db, router=router)
    geo_results = []

    for c in conflicts[:3]:
        query = f"{c['intervention']} {c['organism']}"
        print(f"\n[GEO Closure] Checking empirical datasets for: '{c['intervention']}' -> '{c['outcome']}' ({c['organism']})")

        datasets = dd.discover(
            question=query,
            organism=c["organism"],
            sources=("GEO",),
            per_query=4,
            min_relevance=0.20,
            research_question_id=research_question_id,
        )

        if datasets:
            top = datasets[0]
            print(f"  -> Found GEO Dataset: {top['accession']} (Score: {top['relevance_score']}) - {top['title'][:70]}")
            stmt = (
                f"Public dataset {top['accession']} found (not analysed here) that could help evaluate the disagreement on "
                f"'{c['intervention']}' and '{c['outcome']}' in {c['organism']}."
            )
            eid = db.add_evidence(
                research_question_id=research_question_id,
                evidence_type="DATASET_FOUND",
                statement=stmt,
                source_type="dataset",
                confidence="medium",
                value={"geo_accession": top["accession"], "relevance_reason": top.get("relevance_reason")},
            )
            geo_results.append({"conflict": c["group_key"], "dataset": top["accession"], "evidence_id": eid})
        else:
            print("  -> No public GEO series matrix found for this query.")

    return geo_results


def main():
    ap = argparse.ArgumentParser(description="Autonomous biological research pipeline.")
    ap.add_argument("--question", required=True, help="The core biological research question")
    ap.add_argument("--dir", default=None, help="Optional directory containing local JATS/XML files")
    ap.add_argument("--max-papers", type=int, default=3, help="Max papers to retrieve and analyze")
    ap.add_argument("--min-relevance", type=float, default=0.35, help="Relevance cutoff (0.0 to 1.0)")
    ap.add_argument("--db", default=None, help="Path to SQLite research database")
    ap.add_argument("--no-ai", action="store_true", help="Run deterministic components only")
    ap.add_argument("--no-bio", action="store_true", help="Skip biological entity linking")
    ap.add_argument("--no-geo", action="store_true", help="Skip automated GEO loop closure")
    args = ap.parse_args()

    print("=" * 80)
    print("BIOLOGY AI RESEARCH ENGINE - AUTONOMOUS MULTI-PAPER PIPELINE")
    print(f"Question: {args.question}")
    print("=" * 80)

    with ResearchDB(args.db) as db:
        rq = db.create_research_question(args.question)
        router = None if args.no_ai else AIRouter()
        if router is not None and not router.configured_providers():
            print("  [AI] No API keys found in .env - running deterministic mode (use --no-ai to silence).")
            router = None
        providers = ["groq", "nvidia", "gemini"] if router else None

        bio_client = None if args.no_bio else BioDatabases()
        linker = None if args.no_bio else EntityLinker(db, bio=bio_client, router=router)

        paper_ids = []

        # 1. Paper Acquisition
        if args.dir:
            xml_dir = Path(args.dir)
            if not xml_dir.is_dir():
                print(f"ERROR: Directory '{args.dir}' not found.")
                return
            xml_files = list(xml_dir.glob("*.xml"))
            if not xml_files:
                print(f"ERROR: No .xml files found in '{args.dir}'.")
                return

            print(f"\n[Mode: Local Directory] Processing {len(xml_files)} papers from {args.dir}")
            for xml_path in xml_files:
                pid = ingest_jats(db, str(xml_path), rq, source_query=f"local_dir:{xml_dir.name}")
                paper_ids.append(pid)
        else:
            print("\n[Mode: Autonomous Scientific Discovery] Querying Europe PMC, PMC OA, PubMed, and Crossref...")

            queries = [args.question]
            if router:
                try:
                    plan = generate_model_plan(router, args.question, provider="groq")
                    if plan and plan.get("search_queries"):
                        queries = plan["search_queries"][:4]
                except Exception as exc:
                    print(f"  [Planner fallback] {exc}")

            mgr = SourceManager()
            paper_ids = mgr.retrieve_and_ingest(
                db=db,
                queries=queries,
                question=args.question,
                research_question_id=rq,
                max_papers=args.max_papers,
                min_relevance=args.min_relevance,
            )

        if not paper_ids:
            print("\nNotice: No new papers ingested. Checking existing papers in database...")
            paper_ids = [p["id"] for p in db.list_papers(rq)]

        print(f"\nAnalyzing {len(paper_ids)} papers in active knowledge base...")

        # 2. Extract Study Design & Quote-Anchored Claims
        for i, pid in enumerate(paper_ids, 1):
            paper_rec = db.get_paper(pid)
            title = (paper_rec or {}).get("title") or f"Paper #{pid}"
            print(f"\n--- [{i}/{len(paper_ids)}] {title[:75]} ---")

            print("  [1/3] Extracting study design & experimental statistics...")
            ext_res = PaperExtractor(db, router, providers=providers).extract(pid, force=False)
            print(f"        Status: {ext_res.get('status')} | Stats found: {ext_res.get('n_statistics', 0)}")

            print("  [2/3] Extracting quote-grounded claims...")
            clm_res = ClaimExtractor(db, router, providers=providers).extract(pid, rq, args.question, force=False)
            print(f"        Status: {clm_res.get('status')} | Verified claims: {clm_res.get('claims', 0)}")

            if linker:
                print("  [3/3] Linking biological entities (NCBI Gene, UniProt, GO)...")
                try:
                    b_res = linker.link_paper(pid)
                    print(f"        Entities linked: {len(b_res.get('linked', []))}")
                except Exception as exc:
                    print(f"        Bio linking notice: {exc}")

        # 3. Cross-Study Synthesis & Contradiction Detection
        print("\n" + "=" * 80)
        print("RUNNING CROSS-STUDY CONTRADICTION DETECTION")
        print("=" * 80)

        contradictions = detect_and_store(
            db,
            rq,
            router=router,
            use_ai_merge=True,
            use_ai_explanations=True,
        )

        all_claims = db.verified_claims(rq)
        conflict_targets = extract_conflicts_for_geo(contradictions, all_claims)

        # 4. Empirical Loop Closure (NCBI GEO)
        geo_adjudications = []
        if not args.no_geo and conflict_targets:
            geo_adjudications = run_geo_loop_closure(db, rq, conflict_targets, router=router)

        # 5. Final Synthesis Output
        summary = db.research_question_summary(rq)

        out = {
            "research_question": args.question,
            "papers_analyzed": len(paper_ids),
            "total_claims": summary["claims"],
            "total_evidence_anchors": summary["evidence"],
            "contradiction_summary": {
                "claim_groups_analyzed": len(contradictions.get("groups", [])),
                "conflicting_groups_found": contradictions.get("n_conflict_groups", 0),
                "conflicting_pairs": contradictions.get("n_conflicting_pairs", 0),
            },
            "empirical_geo_loop_closure": geo_adjudications,
            "database_summary": summary,
        }

        out_path = Path("multi_paper_result.json")
        out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")

        print("\n" + "=" * 80)
        print("RESEARCH PIPELINE SYNTHESIS COMPLETE")
        print("=" * 80)
        print(json.dumps(out, indent=2, ensure_ascii=False))
        print(f"\nFinal synthesized research state saved to: {out_path.resolve()}")


if __name__ == "__main__":
    main()