"""
literature_search.py
--------------------
Command-line exploration tool using the multi-source scientific retrieval layer.
Queries Europe PMC, PubMed/PMC, and Crossref with full provenance and XML checks.
"""

from __future__ import annotations

import argparse
from retrieval.source_manager import SourceManager


def main():
    parser = argparse.ArgumentParser(description="Multi-source scientific literature search.")
    parser.add_argument("query", help="Biological research query or topic")
    parser.add_argument("--limit", type=int, default=8, help="Max results per database")
    parser.add_argument("--oa-only", action="store_true", help="Restrict to open-access papers")
    parser.add_argument("--download", action="store_true", help="Download available JATS/XML files")
    args = parser.parse_args()

    print("=" * 80)
    print("CONTROLLED SCIENTIFIC LITERATURE RETRIEVAL")
    print(f"Query: {args.query}")
    print("=" * 80)

    mgr = SourceManager()
    candidates = mgr.search(queries=[args.query], question=args.query, per_query_limit=args.limit)

    if not candidates:
        print("\nNo papers found.")
        return

    print(f"\nDiscovered {len(candidates)} unique scientific papers across sources:\n")

    for i, c in enumerate(candidates, 1):
        print(f"[{i}] {c.title}")
        print("-" * 80)
        print(f"Authors:     {', '.join(c.authors[:4]) if c.authors else 'Unknown'}")
        print(f"Journal:     {c.journal or 'Unknown'} ({c.year or 'n.d.'})")
        print(f"Identifiers: DOI={c.doi or '-'} | PMID={c.pmid or '-'} | PMCID={c.pmcid or '-'}")
        print(f"Open Access: {'YES' if c.is_open_access else 'NO'} | Has XML: {'YES' if c.has_full_text_xml else 'NO'}")
        print(f"Source DB:   {c.source_db.upper()} (Relevance: {c.relevance_score:.2f})")
        if c.url:
            print(f"URL:         {c.url}")

        if args.download and (c.has_full_text_xml or c.pmcid):
            xml_path = mgr.download_xml(c)
            if xml_path:
                print(f"-> XML Saved: {xml_path}")

        if c.abstract:
            print(f"\nAbstract fragment:\n{c.abstract[:300]}...")
        print()


if __name__ == "__main__":
    main()