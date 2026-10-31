"""
bio_databases.py
----------------
Universal Biological Entity Resolution for General Biology & Biomedicine:
  - Genes: Human (TP53, EGFR), Mammals (Brca1, Il6), Microbes (recA), Plants (NRT1.1).
  - Organisms: Dynamic NCBI Taxonomy across all domains of life.
  - Chemicals: PubChem resolver for drugs, metabolites, ions, and hormones.
  - Proteins: UniProt GO terms and KEGG cellular pathways.
"""

from __future__ import annotations

import os
import re
from typing import Any, Dict, Optional, Tuple

from http_client import HttpClient
from ncbi_throttle import ncbi_wait

EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
UNIPROT = "https://rest.uniprot.org/uniprotkb/search"
QUICKGO = "https://www.ebi.ac.uk/QuickGO/services/ontology/go/terms"
PUBCHEM = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name"

# Common model organism prefixes in biology
SPECIES_PREFIXES = {
    "H": "homo sapiens", "M": "mus musculus", "R": "rattus norvegicus",
    "ZM": "zea mays", "AT": "arabidopsis thaliana", "OS": "oryza sativa",
    "EC": "escherichia coli", "SC": "saccharomyces cerevisiae", "DM": "drosophila melanogaster",
    "DR": "danio rerio", "CE": "caenorhabditis elegans",
}

TAXON_IDS = {
    "homo sapiens": 9606, "human": 9606, "mus musculus": 10090, "mouse": 10090,
    "rattus norvegicus": 10116, "rat": 10116, "arabidopsis thaliana": 3702, "arabidopsis": 3702,
    "drosophila melanogaster": 7227, "fruit fly": 7227, "danio rerio": 7955, "zebrafish": 7955,
    "caenorhabditis elegans": 6239, "escherichia coli": 562, "saccharomyces cerevisiae": 559292,
    "zea mays": 4577, "maize": 4577, "oryza sativa": 4530, "rice": 4530,
}

ENGLISH_STOPWORDS = {
    "THE", "IN", "IT", "OVERALL", "TOTAL", "AND", "OR", "OF", "TO", "A", "AN", "ON",
    "AT", "BY", "FOR", "WITH", "FROM", "AS", "IS", "ARE", "WAS", "WERE", "THIS", "THAT",
    "THESE", "THOSE", "WE", "OUR", "THEIR", "ITS", "ALL", "BOTH", "EACH", "FEW", "MORE",
    "MOST", "OTHER", "SOME", "SUCH", "NO", "NOR", "NOT", "ONLY", "OWN", "SAME", "SO",
    "THAN", "TOO", "VERY", "CAN", "WILL", "JUST", "SHOULD", "NOW", "COAL", "HIGH", "LOW",
    "NN", "NUE", "UP", "DOWN", "WELL", "ALSO", "BUT", "YET", "ONE", "TWO", "THREE",
    "BECAUSE", "SENTINEL", "HEALTH", "POWDERY", "SOLUBLE", "EXPRESSION", "APHA", "FAME",
    "LEACHATE", "GANGUE", "SCREENING", "OPPORTUNITIES", "ROLE", "ROLES", "UNDER", "DURING",
    "AFTER", "BEFORE", "ACQUISITION", "RESPONSES", "RESPONSE", "REGULATION", "EFFECTS", "EFFECT",
    "DATA", "RESULTS", "METHODS", "DISCUSSION", "CONCLUSION", "STUDY", "STUDIES", "ANALYSIS"
}

# Broad biological chemical dictionary: drugs, neurotransmitters, metabolites, and nutrients
CURATED_BIO_CHEMICALS = {
    "nitrate", "ammonium", "urea", "phosphate", "potassium", "calcium", "magnesium", "iron",
    "glucose", "lactate", "pyruvate", "atp", "adp", "amp", "nad", "nadh", "nadp", "nadph",
    "dopamine", "serotonin", "glutamate", "gaba", "acetylcholine", "epinephrine", "norepinephrine",
    "insulin", "glucagon", "cortisol", "estrogen", "testosterone", "melatonin", "thyroxine",
    "cholesterol", "palmitate", "oleate", "citrate", "succinate", "fumarate", "malate",
    "doxorubicin", "cisplatin", "metformin", "paclitaxel", "rapamycin", "aspirin", "staurosporine",
    "auxin", "cytokinin", "gibberellin", "abscisic acid", "ethylene", "sorbitol", "mannitol"
}

# Universal biological gene token regex (matches TP53, BRCA1, NRT1.1, recA, bHLH118)
GENE_RE = re.compile(r"\b(?:[A-Z][A-Za-z0-9]{1,8}(?:\.\d+)?|[a-z]{3,4}[A-Z0-9]?)\b")


_NOT_SYMBOLS = {"DNA", "RNA", "MRNA", "PCR", "QPCR", "HIV", "BMI", "USA", "UK", "WHO", "SD", "SEM", "CI", "HR", "RR", "OR",
                "ANOVA", "ELISA", "FDA", "ICU", "MRI", "CT", "UV", "UVB", "NB", "IV", "BID", "TID", "QD", "PO", "OK", "VS",
                "ROS", "ATP", "PBS", "DMSO", "PMID", "PMC", "DOI", "NIH", "NCBI", "COVID", "TCM", "CASE", "REPORT", "FIG",
                "TABLE", "AIM", "AIMS", "METHODS", "RESULTS", "CONCLUSION", "CONCLUSIONS", "BACKGROUND", "OBJECTIVE"}


def looks_like_symbol(tok: str, around: str = "") -> bool:
    """A gene / protein symbol, not an ordinary word: 'TP53', 'NFE2L2', 'AMPK', 'Slc7a11', 'Nqo1' - never 'sex', 'hair',
    'Out', 'Medicine', or a word of a title written in capitals ('A REMARKABLE CASE OF ... REMEDY')."""
    if tok.upper() in _NOT_SYMBOLS:
        return False
    letters = [c for c in around if c.isalpha()]
    if letters and sum(c.isupper() for c in letters) / len(letters) > 0.8 and not re.search(r"\d", tok):
        return False                                   # the whole passage is in capitals: words, not symbols
    if re.fullmatch(r"[A-Z][A-Z0-9\-]{1,9}", tok):
        return bool(re.search(r"\d", tok)) or len(tok) >= 3
    if re.fullmatch(r"[A-Z][a-z]{1,5}\d+[a-z]?\d*", tok):   # mouse / rat style: Slc7a11, Nqo1, Sox2
        return True
    return False


def is_probable_chemical(name: str) -> bool:
    s = name.lower().strip()
    if s in CURATED_BIO_CHEMICALS:
        return True
    if len(s) < 4 or name.upper() in ENGLISH_STOPWORDS:
        return False
    if re.fullmatch(r"[A-Z][a-z]?\d*(?:[A-Z][a-z]?\d*)+[+-]?", name):
        return True
    suffixes = ("ate", "ide", "ite", "ose", "ol", "ine", "one", "ic acid", "ium", "ib", "mab")
    return s.endswith(suffixes) and not s.endswith(("state", "provide", "outside", "inside", "alone", "combine"))


def taxon_for(organism: Optional[str], http: Optional[HttpClient] = None) -> Optional[int]:
    if not organism:
        return 9606  # Default fallback to human if unspecified in biomedical papers
    o = organism.lower().strip()
    if o in TAXON_IDS:
        return TAXON_IDS[o]
    for name, tid in TAXON_IDS.items():
        if name in o or o in name:
            return tid
    try:
        client = http or HttpClient(cache_dir="cache/taxonomy")
        clean_org = re.sub(r"[^a-zA-Z ]+", "", o).strip()
        data = client.get_json(
            f"{EUTILS}/esearch.fcgi",
            params={"db": "taxonomy", "term": clean_org, "retmode": "json", "retmax": "1"},
        )
        id_list = (data.get("esearchresult") or {}).get("idlist") or []
        if id_list:
            tid = int(id_list[0])
            TAXON_IDS[o] = tid
            return tid
    except Exception:
        pass
    return None


class BioDatabases:
    def __init__(self, http: Optional[HttpClient] = None):
        self.http = http or HttpClient(cache_dir="cache/biodb")

    @property
    def ncbi_key(self):                    # v27: the key from the API keys page, else NCBI_API_KEY
        from ncbi_throttle import api_key
        return api_key()

    def _ncbi(self, endpoint: str, **params) -> Any:
        ncbi_wait(bool(self.ncbi_key))
        params.setdefault("retmode", "json")
        params.setdefault("tool", "BiologyAIResearcher")
        params.setdefault("email", "researcher@biopipeline.org")
        if self.ncbi_key:
            params["api_key"] = self.ncbi_key
        return self.http.get_json(f"{EUTILS}/{endpoint}.fcgi", params=params)

    def ncbi_gene(self, symbol: str, taxon_id: Optional[int]) -> Optional[Dict[str, Any]]:
        term = f"{symbol}[sym]" + (f" AND {taxon_id}[taxid]" if taxon_id else "")
        ids = (self._ncbi("esearch", db="gene", term=term, retmax=3).get("esearchresult", {}) or {}).get("idlist", [])
        if not ids:
            return None
        summ = self._ncbi("esummary", db="gene", id=",".join(ids)).get("result", {})
        for gid in ids:
            rec = summ.get(gid)
            if rec and str(rec.get("name", "")).lower() == symbol.lower():
                return {
                    "gene_id": gid, "symbol": rec.get("name"), "description": rec.get("description"),
                    "organism": (rec.get("organism") or {}).get("scientificname"),
                    "taxon_id": (rec.get("organism") or {}).get("taxid"), "summary": (rec.get("summary") or "")[:500],
                }
        return None

    def uniprot(self, symbol: str, taxon_id: Optional[int]) -> Optional[Dict[str, Any]]:
        q = f"(gene_exact:{symbol})" + (f" AND (organism_id:{taxon_id})" if taxon_id else "") + " AND (reviewed:true)"
        data = self.http.get_json(UNIPROT, params={"query": q, "format": "json", "size": "1"})
        results = data.get("results", [])
        if not results:
            return None
        r = results[0]
        gene = ((r.get("genes") or [{}])[0].get("geneName") or {}).get("value")
        prot = (((r.get("proteinDescription") or {}).get("recommendedName") or {}).get("fullName") or {}).get("value")
        go, kegg, pdb = [], [], []
        for x in r.get("uniProtKBCrossReferences", []) or []:
            db, xid = x.get("database"), x.get("id")
            if db == "GO":
                name = next((p["value"] for p in x.get("properties", []) if p.get("key") == "GoTerm"), "")
                go.append({"id": xid, "name": name[2:] if name[:2] in ("P:", "F:", "C:") else name})
            elif db == "KEGG":
                kegg.append(xid)
            elif db == "PDB":
                pdb.append(xid)
        return {
            "accession": r.get("primaryAccession"), "gene": gene, "protein": prot,
            "organism": (r.get("organism") or {}).get("scientificName"), "go": go[:10], "kegg": kegg[:3], "pdb": pdb[:3],
        }

    def pubchem_compound(self, name: str) -> Optional[Dict[str, Any]]:
        if not is_probable_chemical(name):
            return None
        try:
            data = self.http.get_json(f"{PUBCHEM}/{name}/JSON")
            cid = data["PC_Compounds"][0]["id"]["id"]["cid"]
            return {"cid": cid, "name": name, "url": f"https://pubchem.ncbi.nlm.nih.gov/compound/{cid}"}
        except Exception:
            return None


class EntityLinker:
    def __init__(self, db: Any, bio: Optional[BioDatabases] = None, router: Any = None, max_cache_size: int = 1500):
        self.db, self.bio, self.router = db, bio or BioDatabases(), router
        self._cache: Dict[Tuple[str, Optional[int]], Optional[int]] = {}
        self.max_cache_size = max_cache_size

    def _cache_set(self, key: Tuple[str, Optional[int]], val: Optional[int]) -> None:
        if len(self._cache) >= self.max_cache_size:
            drop_keys = list(self._cache.keys())[: self.max_cache_size // 4]
            for k in drop_keys:
                self._cache.pop(k, None)
        self._cache[key] = val

    def resolve_symbol(self, raw_symbol: str, organism: Optional[str]) -> Tuple[Optional[int], float]:
        sym_up = raw_symbol.upper().strip()
        if sym_up in ENGLISH_STOPWORDS or len(sym_up) <= 2:
            return None, 0.0

        taxon = taxon_for(organism, self.bio.http)
        key = (sym_up, taxon)
        if key in self._cache:
            return self._cache[key], 1.0

        # Tier 1: Exact Gene Match in Target Organism (Confidence 1.0)
        g = self.bio.ncbi_gene(raw_symbol, taxon)
        if g:
            eid = self.db.upsert_entity("gene", "ncbi_gene", str(g["gene_id"]), g["symbol"], g.get("description") or "", g.get("organism") or "", g)
            self._cache_set(key, eid)
            return eid, 1.0

        # Tier 2: Strip Species Prefix (e.g. mIL6 -> IL6 in mouse; ZmbHLH -> bHLH in maize) (Confidence 0.90)
        for pfx, org_name in SPECIES_PREFIXES.items():
            if sym_up.startswith(pfx) and len(raw_symbol) > len(pfx) + 2:
                stripped = raw_symbol[len(pfx):]
                inferred_tid = taxon_for(org_name, self.bio.http)
                g_pfx = self.bio.ncbi_gene(stripped, inferred_tid)
                if g_pfx:
                    eid = self.db.upsert_entity(
                        "gene", "ncbi_gene", str(g_pfx["gene_id"]), g_pfx["symbol"],
                        f"[Resolved from {raw_symbol}] {g_pfx.get('description', '')}",
                        g_pfx.get("organism") or org_name, g_pfx
                    )
                    self._cache_set(key, eid)
                    return eid, 0.90

        # Tier 3: Chemical compound / drug lookup (Confidence 0.95)
        if is_probable_chemical(raw_symbol):
            chem = self.bio.pubchem_compound(raw_symbol)
            if chem:
                eid = self.db.upsert_entity("chemical", "pubchem", str(chem["cid"]), chem["name"], "PubChem Chemical Compound", "", chem)
                self._cache_set(key, eid)
                return eid, 0.95

        self._cache_set(key, None)
        return None, 0.0

    def link_paper(self, paper_id: int, organism: Optional[str] = None) -> Dict[str, Any]:
        p = self.db.get_paper(paper_id)
        text = " ".join(x for x in (p.get("title"), p.get("abstract")) if x)
        for s in self.db.get_sections(paper_id):
            if s["section_type"] in ("results", "abstract", "results_discussion", "conclusion", "other"):
                text += " " + s["text"]

        study = self.db.get_extraction_raw(paper_id).get("study") or {}
        org = organism or study.get("organism") or study.get("species")
        tokens = [m.group(0) for m in GENE_RE.finditer(text) if m.group(0).upper() not in ENGLISH_STOPWORDS
                  and looks_like_symbol(m.group(0), text[max(0, m.start() - 40): m.end() + 40])]
        seen, linked, unresolved = set(), [], []

        for sym in tokens[:30]:
            if sym in seen:
                continue
            seen.add(sym)
            eid, conf = self.resolve_symbol(sym, org)
            if eid and conf >= 0.85:
                self.db.link_entity(eid, "paper", paper_id, sym)
                linked.append(f"{sym} (conf={conf})")
            else:
                unresolved.append(sym)

        return {"linked": linked, "unresolved": unresolved}