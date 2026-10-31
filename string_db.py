"""
string_db.py
------------
Physical protein-protein interactions from the free STRING-db REST API
(https://string-db.org/api). No key required.

    network(["TP53","MDM2"], species=9606, network_type="physical")
    interaction_partners("TP53", species=9606)

Scores are STRING combined scores (0-1). Evidence channels are kept so the
report can show WHY an edge exists (experimental, database, text-mining...).
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from http_client import HttpClient, HttpError

STRING_BASE = "https://string-db.org/api"
CALLER = "BioSynthetica"

SPECIES_BY_NAME = {
    "homo sapiens": 9606, "human": 9606, "mus musculus": 10090, "mouse": 10090,
    "rattus norvegicus": 10116, "rat": 10116, "danio rerio": 7955, "zebrafish": 7955,
    "drosophila melanogaster": 7227, "caenorhabditis elegans": 6239,
    "saccharomyces cerevisiae": 4932, "arabidopsis thaliana": 3702, "escherichia coli": 511145,
    # wider biology (STRING taxon ids)
    "arabidopsis": 3702, "oryza sativa": 39947, "rice": 39947, "zea mays": 4577, "maize": 4577,
    "solanum lycopersicum": 4081, "tomato": 4081, "glycine max": 3847, "soybean": 3847,
    "bacillus subtilis": 224308, "pseudomonas aeruginosa": 208964, "mycobacterium tuberculosis": 83332,
    "e. coli": 511145, "yeast": 4932, "saccharomyces": 4932, "gallus gallus": 9031, "chicken": 9031,
    "bos taurus": 9913, "cattle": 9913, "cow": 9913, "sus scrofa": 9823, "pig": 9823,
    "canis": 9615, "dog": 9615, "xenopus tropicalis": 8364, "xenopus": 8364,
    "drosophila": 7227, "fruit fly": 7227, "c. elegans": 6239, "elegans": 6239,
}

CHANNELS = {
    "escore": "experimental", "dscore": "curated database", "tscore": "text-mining",
    "ascore": "co-expression", "nscore": "neighborhood", "fscore": "gene fusion", "pscore": "co-occurrence",
}

_SYMBOL_RE = re.compile(r"\b([A-Z][A-Z0-9]{1,9}(?:-[A-Z0-9]{1,3})?)\b")
_NOT_GENES = {
    "DNA", "RNA", "ATP", "ADP", "AMP", "NAD", "PCR", "MRI", "BMI", "HIV", "USA", "WHO", "RCT", "ROS",
    "CI", "OR", "HR", "RR", "SD", "SE", "IC50", "EC50", "II", "III", "IV", "AND", "THE", "NO", "ICU",
    "FDA", "MRNA", "CT", "UK", "EU", "IL", "COVID", "SARS", "LDL", "HDL", "LPS",
    # abbreviations / English words that collide with real gene symbols but almost never mean the gene in text
    # ('SET domain', 'MFI', 'EMT markers', 'BC' = breast cancer, 'TNBC', 'ER-alpha' is written as ESR1 when meant)
    "SET", "EMT", "MFI", "BC", "TNBC", "ER", "PR", "GH", "TAM", "TAX", "PBCC", "PBCCS", "NKT", "PGE2", "COX", "MOK",
    "CAT", "MAX", "REST", "IMPACT", "CAMP", "WAS", "FAT", "SDS", "ELISA", "FACS", "QPCR", "RT-PCR", "WB", "IHC",
    "MTT", "CCK", "EDU", "BRDU", "TUNEL", "AMPK/MTOR", "NF", "PD-1", "PD1", "OS", "PFS", "DFS", "HBA1C", "IGF", "IFN",
}


_CELL_LINE_RE = re.compile(r"^(MCF|MDA|T47D|HEK|HCT|A549|HELA|U87|PC3|LNCAP|SH-SY5Y|HEPG2|CACO|JURKAT|THP|SKBR|BT474|SYNTHETIC)", re.I)


def find_species(text: Optional[str]) -> Optional[int]:
    """STRING taxon id for the first known organism named in the text, else None."""
    o = (text or "").lower()
    best = None
    for name, tid in SPECIES_BY_NAME.items():
        m = re.search(r"\b" + re.escape(name) + r"(?:s|es)?\b", o)
        if m and (best is None or m.start() < best[0] or (m.start() == best[0] and len(name) > best[2])):
            best = (m.start(), tid, len(name))
    return best[1] if best else None


def species_for(organism: Optional[str]) -> int:
    return find_species(organism) or 9606


def symbols_from_text(text: str, limit: int = 10) -> List[str]:
    """Cheap symbol guess from free text (used when no NCBI-linked genes exist)."""
    seen: List[str] = []
    for m in _SYMBOL_RE.finditer(text or ""):
        s = m.group(1)
        if s in _NOT_GENES or s.isdigit() or len(s) < 3 or not re.search(r"[A-Z]", s) or _CELL_LINE_RE.match(s):
            continue
        if s not in seen:
            seen.append(s)
        if len(seen) >= limit:
            break
    return seen


class StringDBClient:
    def __init__(self, http: Optional[HttpClient] = None):
        self.http = http or HttpClient(cache_dir="cache/string_db", timeout=25.0, retries=1)

    def _get(self, endpoint: str, params: Dict[str, Any]) -> Any:
        params = dict(params)
        params["caller_identity"] = CALLER
        return self.http.get_json(f"{STRING_BASE}/json/{endpoint}", params=params)

    def map_ids(self, symbols: List[str], species: int = 9606) -> Dict[str, str]:
        """symbol -> preferred STRING name (drops symbols STRING does not know)."""
        if not symbols:
            return {}
        try:
            rows = self._get("get_string_ids", {"identifiers": "\r".join(symbols), "species": species, "limit": 1})
        except HttpError as exc:
            print(f"  [STRING] id mapping failed: {str(exc)[:120]}")
            return {}
        out = {}
        for r in rows or []:
            if isinstance(r, dict) and r.get("queryItem"):
                out[str(r["queryItem"])] = r.get("preferredName") or r["queryItem"]
        return out

    def network(self, symbols: List[str], species: int = 9606, required_score: int = 400,
                network_type: str = "physical", add_nodes: int = 0) -> List[Dict[str, Any]]:
        if not symbols:
            return []
        params = {"identifiers": "\r".join(symbols), "species": species, "required_score": required_score,
                  "network_type": network_type}
        if add_nodes:
            params["add_nodes"] = add_nodes
        try:
            rows = self._get("network", params)
        except HttpError as exc:
            print(f"  [STRING] network query failed: {str(exc)[:120]}")
            return []
        return self._edges(rows)

    def interaction_partners(self, symbol: str, species: int = 9606, limit: int = 10,
                             required_score: int = 700, network_type: str = "physical") -> List[Dict[str, Any]]:
        try:
            rows = self._get("interaction_partners", {"identifiers": symbol, "species": species, "limit": limit,
                                                      "required_score": required_score, "network_type": network_type})
        except HttpError as exc:
            print(f"  [STRING] partner query failed for {symbol}: {str(exc)[:120]}")
            return []
        return self._edges(rows)

    @staticmethod
    def _edges(rows: Any) -> List[Dict[str, Any]]:
        edges, seen = [], set()
        for r in rows or []:
            if not isinstance(r, dict):
                continue
            a, b = r.get("preferredName_A"), r.get("preferredName_B")
            if not a or not b:
                continue
            key = tuple(sorted((a, b)))
            if key in seen:
                continue
            seen.add(key)
            channels = {CHANNELS[k]: round(float(r[k]), 3) for k in CHANNELS if r.get(k)}
            top = max(channels, key=channels.get) if channels else None
            edges.append({"a": a, "b": b, "score": round(float(r.get("score", 0)), 3),
                          "channels": channels, "top_evidence": top})
        return sorted(edges, key=lambda e: -e["score"])

    def build_for_genes(self, symbols: List[str], species: int = 9606, partners_if_single: bool = True,
                        max_genes: int = 15) -> Dict[str, Any]:
        """High-level helper used by the research engine."""
        mapped = self.map_ids(symbols[:max_genes], species)
        genes = list(dict.fromkeys(mapped.values()))
        edges: List[Dict[str, Any]] = []
        mode = "network"
        if len(genes) >= 2:
            edges = self.network(genes, species)
        if len(edges) == 0 and genes and partners_if_single:
            mode = "partners"
            for g in genes[:3]:
                edges.extend(self.interaction_partners(g, species, limit=8))
        nodes = sorted({e["a"] for e in edges} | {e["b"] for e in edges} | set(genes))
        return {
            "species": species, "query_genes": genes, "unmapped": [s for s in symbols if s not in mapped],
            "mode": mode, "nodes": nodes, "edges": edges[:60],
            "link": f"https://string-db.org/cgi/network?identifiers={'%0d'.join(genes)}&species={species}" if genes else None,
        }
