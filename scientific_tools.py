"""
scientific_tools.py
-------------------
Controlled scientific tool interface for AI reasoning.
Gives the AI structured scientific API access (PubMed, Europe PMC, PMC XML,
NCBI Gene, UniProt, and GEO) with strict provenance boundaries.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from bio_databases import BioDatabases
from database import ResearchDB
from dataset_discovery import DatasetDiscovery
from http_client import HttpClient
from retrieval.source_manager import SourceManager


class ScientificToolbox:
    """
    Standardized, safe tool interface for scientific investigation.
    """

    def __init__(self, db: Optional[ResearchDB] = None):
        self.db = db or ResearchDB()
        self.http = HttpClient(cache_dir="cache/tools")
        self.source_manager = SourceManager(http=self.http)
        self.bio = BioDatabases(http=self.http)
        self.geo = DatasetDiscovery(db=self.db, http=self.http)

    def search_papers(self, query: str, max_results: int = 10) -> List[Dict[str, Any]]:
        """Search PubMed, Europe PMC, PMC, and Crossref with deduplication and relevance ranking."""
        candidates = self.source_manager.search(
            queries=[query],
            question=query,
            per_query_limit=max_results,
            min_relevance=0.25,
        )
        return [c.to_dict() for c in candidates[:max_results]]

    def get_paper_metadata(self, paper_id: int) -> Optional[Dict[str, Any]]:
        """Get study design, organism, sample size, and identifiers for an ingested paper."""
        paper = self.db.get_paper(paper_id)
        if not paper:
            return None
        extraction = self.db.get_paper_extraction(paper_id) or {}
        return {
            "id": paper["id"],
            "title": paper["title"],
            "journal": paper["journal"],
            "year": paper["year"],
            "doi": paper["doi"],
            "pmid": paper["pmid"],
            "pmcid": paper["pmcid"],
            "study_type": extraction.get("study_type"),
            "organism": extraction.get("organism"),
            "sample_size": extraction.get("sample_size"),
            "intervention": extraction.get("intervention"),
            "control": extraction.get("control"),
        }

    def open_paper(self, paper_id: int) -> Dict[str, Any]:
        """Read all parsed sections, abstract, and tables of an ingested paper."""
        paper = self.db.get_paper(paper_id)
        if not paper:
            raise KeyError(f"Paper {paper_id} not found in database.")
        sections = self.db.get_sections(paper_id)
        tables = self.db.get_tables(paper_id)
        return {
            "title": paper["title"],
            "abstract": paper["abstract"],
            "sections": [{"type": s["section_type"], "title": s["title"], "text": s["text"]} for s in sections],
            "tables": [{"label": t["label"], "caption": t["caption"], "rows": t["rows"]} for t in tables],
        }

    def get_full_text(self, paper_id: int, section_type: Optional[str] = None) -> str:
        """Get the full text or a specific section ('results', 'methods', 'discussion')."""
        sections = self.db.get_sections(paper_id)
        if section_type:
            matching = [s["text"] for s in sections if s["section_type"] == section_type.lower().strip()]
            if matching:
                return "\n\n".join(matching)
        return "\n\n".join(f"[{s['section_type'].upper()}: {s['title']}]\n{s['text']}" for s in sections)

    def search_within_paper(self, paper_id: int, keyword: str, window: int = 250) -> List[str]:
        """Search for a biological factor or measurement within a paper and return exact snippets."""
        full_text = self.get_full_text(paper_id)
        snippets = []
        pattern = re.compile(rf"\b{re.escape(keyword)}\b", re.IGNORECASE)
        for m in pattern.finditer(full_text):
            start = max(0, m.start() - window // 2)
            end = min(len(full_text), m.end() + window // 2)
            snippets.append(full_text[start:end].strip().replace("\n", " "))
            if len(snippets) >= 5:
                break
        return snippets

    def search_gene(self, symbol: str, organism: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Look up official gene accession, description, and organism in NCBI Gene."""
        from bio_databases import taxon_for
        tid = taxon_for(organism, self.http)
        return self.bio.ncbi_gene(symbol, tid)

    def search_protein(self, symbol: str, organism: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Look up protein function, GO terms, and KEGG pathways in UniProt."""
        from bio_databases import taxon_for
        tid = taxon_for(organism, self.http)
        return self.bio.uniprot(symbol, tid)

    def search_geo_datasets(self, query: str, organism: Optional[str] = None) -> List[Dict[str, Any]]:
        """Discover matching expression series matrices in NCBI GEO."""
        return self.geo.search_geo(query, organism=organism, max_results=5)