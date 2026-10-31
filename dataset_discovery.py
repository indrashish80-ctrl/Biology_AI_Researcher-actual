"""
dataset_discovery.py
--------------------
NCBI GEO Empirical Dataset Discovery & Bioinformatic Re-analysis Engine:
  1. Searches NCBI GEO DataSets for public microarray and RNA-seq series.
  2. Parses series matrix expression tables with automated log2 scale detection.
  3. Evaluates differential expression (log2FC, Welch's t-test, Cohen's d).
  4. Stores discovered datasets directly into ResearchDB.
"""

from __future__ import annotations

import gzip
import io
import math
import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from database import ResearchDB
from http_client import HttpClient
from ncbi_throttle import ncbi_wait

EUTILS_SEARCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
EUTILS_SUMMARY = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"


def _with_key(params):
    from ncbi_throttle import api_key
    k = api_key()
    return {**params, "api_key": k} if k else params


def parse_series_matrix(content: bytes) -> Dict[str, Any]:
    """Parse GEO series matrix text and normalize log2 scaling."""
    text = (
        gzip.decompress(content).decode("utf-8", errors="replace")
        if content[:2] == b"\x1f\x8b"
        else content.decode("utf-8", errors="replace")
    )
    meta: Dict[str, List[List[str]]] = {}
    table_lines: List[str] = []
    in_table = False

    for line in text.splitlines():
        if line.startswith("!series_matrix_table_begin"):
            in_table = True
            continue
        if line.startswith("!series_matrix_table_end"):
            in_table = False
            continue
        if in_table:
            table_lines.append(line)
        elif line.startswith("!Sample_"):
            parts = [p.strip().strip('"') for p in line.rstrip("\n").split("\t")]
            meta.setdefault(parts[0][1:], []).append(parts[1:])

    if not table_lines:
        raise ValueError("No expression table found in series matrix.")

    expr = pd.read_csv(io.StringIO("\n".join(table_lines)), sep="\t", index_col=0, quotechar='"')
    expr.columns = [str(c).strip('"') for c in expr.columns]
    expr.index = [str(i).strip('"') for i in expr.index]
    expr = expr.apply(pd.to_numeric, errors="coerce")

    # Automated log2 scaling detection
    median_val = float(np.nanmedian(expr.values))
    if median_val >= 35.0:
        expr = np.log2(np.maximum(expr, 1.0))

    # Sample-median normalization
    sample_medians = expr.median(axis=0)
    global_median = float(sample_medians.median())
    expr = expr.subtract(sample_medians - global_median, axis=1)

    samples = pd.DataFrame(index=expr.columns)
    if "Sample_geo_accession" in meta:
        samples["geo_accession"] = meta["Sample_geo_accession"][0]
    if "Sample_title" in meta:
        samples["title"] = meta["Sample_title"][0]
    for key in ("Sample_source_name_ch1", "Sample_organism_ch1"):
        if key in meta:
            samples[key.replace("Sample_", "")] = meta[key][0]
    for row in meta.get("Sample_characteristics_ch1", []):
        keys = {c.split(":", 1)[0].strip().lower() for c in row if ":" in c}
        name = keys.pop() if len(keys) == 1 else "characteristic"
        if name in samples.columns:
            name = f"{name}_{len(samples.columns)}"
        samples[name] = [c.split(":", 1)[1].strip() if ":" in c else c for c in row]

    return {"samples": samples, "expr": expr, "log2_scaled": True}


def select_best_probe(expr: pd.DataFrame, probes: List[str]) -> Optional[pd.Series]:
    """Select the probe with highest inter-sample variance to avoid dead probes."""
    valid = [p for p in probes if p in expr.index]
    if not valid:
        return None
    if len(valid) == 1:
        return expr.loc[valid[0]]
    variances = expr.loc[valid].var(axis=1)
    best_probe = variances.idxmax()
    return expr.loc[best_probe]


class DatasetDiscovery:
    """NCBI GEO Series discovery and differential expression analyzer."""

    def __init__(
        self,
        db: Optional[ResearchDB] = None,
        router: Optional[Any] = None,
        http: Optional[HttpClient] = None,
    ):
        self.db = db
        self.router = router
        self.http = http or HttpClient(cache_dir="cache/geo")

    def search_geo(
        self,
        query: str,
        organism: Optional[str] = None,
        max_results: int = 5,
    ) -> List[Dict[str, Any]]:
        clean_q = re.sub(r"\b(?:None|null|unknown)\b", " ", query or "", flags=re.IGNORECASE)
        clean_q = re.sub(r"\s+", " ", clean_q).strip()
        if not clean_q:
            return []
        term = f"{clean_q} AND gse[ETYP]"
        if organism and str(organism).strip().lower() not in ("none", "null", "unknown", "unspecified"):
            term += f" AND {organism}[Organism]"

        params = {
            "db": "gds",
            "term": term,
            "retmax": str(max_results),
            "retmode": "json",
            "tool": "BiologyAIResearcher",
            "email": "researcher@biopipeline.org",
        }

        try:
            ncbi_wait()
            data = self.http.get_json(EUTILS_SEARCH, params=_with_key(params))
            if not isinstance(data, dict):
                return []
            id_list = (data.get("esearchresult") or {}).get("idlist") or []
            if not id_list:
                return []

            sum_params = {
                "db": "gds",
                "id": ",".join(id_list),
                "retmode": "json",
                "tool": "BiologyAIResearcher",
                "email": "researcher@biopipeline.org",
            }
            ncbi_wait()
            sum_data = self.http.get_json(EUTILS_SUMMARY, params=_with_key(sum_params))
            if not isinstance(sum_data, dict):
                return []

            results = sum_data.get("result") or {}
            datasets = []

            for gid in id_list:
                rec = results.get(gid)
                if not rec or not isinstance(rec, dict):
                    continue

                acc = rec.get("accession") or f"GSE{gid}"
                title = rec.get("title") or "NCBI GEO Series Dataset"
                summary = rec.get("summary") or ""
                org = rec.get("taxon") or organism or "Unspecified"

                datasets.append({
                    "accession": acc,
                    "title": title,
                    "description": summary,
                    "organism": org,
                    "source": "GEO",
                    "url": f"https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc={acc}",
                    "gds_id": gid,
                })

            return datasets

        except Exception:
            return []

    def discover(
        self,
        question: str,
        organism: Optional[str] = None,
        sources: Tuple[str, ...] = ("GEO",),
        per_query: int = 4,
        min_relevance: float = 0.20,
        research_question_id: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        datasets = self.search_geo(question, organism=organism, max_results=per_query)
        scored = []
        q_tokens = set(re.findall(r"\w{3,}", (question or "").lower())) - {"none", "null", "unknown"}

        for d in datasets:
            text = f"{d['title']} {d['description']}".lower()
            overlap = sum(1 for tok in q_tokens if tok in text)
            rel_score = overlap / max(1, len(q_tokens))

            if rel_score >= min_relevance or not q_tokens:
                d["relevance_score"] = round(max(0.5, rel_score), 2)
                d["relevance_reason"] = f"Empirical match on query tokens: {', '.join(list(q_tokens)[:3])}"

                if self.db is not None:
                    self.db.upsert_dataset(
                        research_question_id=research_question_id,
                        source=d["source"],
                        accession=d["accession"],
                        title=d["title"],
                        description=d["description"][:500],
                        organism=d["organism"],
                        url=d["url"],
                        metadata=d,
                    )
                scored.append(d)

        return sorted(scored, key=lambda x: x.get("relevance_score", 0.0), reverse=True)

    def evaluate_expression_difference(
        self,
        expr_df: pd.DataFrame,
        groups: pd.Series,
        group_a: str,
        group_b: str,
        probes: List[str],
    ) -> Dict[str, Any]:
        """Calculates log2FC, Welch's t-test, Cohen's d, and effect direction."""
        vals = select_best_probe(expr_df, probes)
        if vals is None:
            raise ValueError("No matching probes found on expression matrix.")

        a = vals[groups == group_a].dropna().astype(float)
        b = vals[groups == group_b].dropna().astype(float)
        if len(a) < 2 or len(b) < 2:
            raise ValueError(f"Insufficient replicates: group A (n={len(a)}), group B (n={len(b)}).")

        t_stat, p_val = stats.ttest_ind(a, b, equal_var=False)
        mean_a, mean_b = float(a.mean()), float(b.mean())
        log2_fc = mean_a - mean_b
        # empirical-Bayes moderated t over the WHOLE array (limma method): the gene's variance borrows strength from
        # all probes, and its p-value is FDR-adjusted against every probe tested - a single unadjusted t-test on one
        # gene of 20,000 is not a discovery
        moderated = {}
        try:
            from omics import moderated_t
            sel = [c for c in expr_df.columns if groups.get(c) in (group_a, group_b)]
            sub = expr_df[sel].apply(pd.to_numeric, errors="coerce").dropna()
            if len(sub) >= 50:
                res = moderated_t(sub.to_numpy(dtype=float), [groups[c] for c in sel], group_a, group_b)
                probe = vals.name if vals.name in sub.index else None
                if probe is not None:
                    k = list(sub.index).index(probe)
                    moderated = {"moderated_t": round(float(res["t"][k]), 3), "moderated_p_value": float(res["p"][k]),
                                 "fdr_across_array": float(res["fdr"][k]), "probes_tested": int(len(sub)),
                                 "prior_df": None if not np.isfinite(res["prior_df"]) or res["prior_df"] > 1e6
                                 else round(float(res["prior_df"]), 2)}
        except Exception:
            moderated = {}

        pooled_sd = math.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / (len(a) + len(b) - 2))
        cohens_d = log2_fc / pooled_sd if pooled_sd > 0 else 0.0
        p_dec = moderated.get("fdr_across_array", p_val)        # the FDR decides when it could be computed
        direction = "no_effect" if p_dec >= 0.05 else ("increase" if log2_fc > 0 else "decrease")

        return {
            "group_a": group_a,
            "group_b": group_b,
            "n_a": len(a),
            "n_b": len(b),
            "log2_fold_change": round(log2_fc, 3),
            "welch_t_stat": round(float(t_stat), 3),
            "raw_p_value": float(p_val),
            "cohens_d": round(cohens_d, 3),
            **moderated,
            "direction": direction,
            "decision_basis": "FDR across the array (moderated t)" if moderated else "unadjusted Welch p (array too small for FDR)",
            "bioinformatic_limitations": (
                "Normalized series matrix re-analysis (sample-median centered, max-variance probe). "
                "Moderated t (limma method) and Benjamini-Hochberg FDR across all probes; no correction for "
                "probe-specific hybridization affinity or unmodeled batch effects. "
                "Requires raw CEL/FASTQ reprocessing via DESeq2/limma-voom for regulatory validation."
            ),
        }