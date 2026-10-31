"""
paper_extraction.py
-------------------
Extracts study design, statistics, and findings from full-text JATS XML
with dedicated Abstract-Only Extraction Mode for paywalled literature.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from llm_json import LLMJSONError, ask_json
from stats_parsing import (parse_confidence_intervals, parse_effect_sizes, parse_p_values,
                           parse_sample_sizes, quote_grounding)

STUDY_TYPES = [
    "randomized_controlled_experiment", "controlled_experiment", "in_vitro_cell_assay",
    "in_vivo_animal_study", "clinical_trial", "cohort", "case_control", "cross_sectional",
    "correlational_omics", "meta_analysis", "systematic_review", "computational", "other", "unclear",
]
DIRECTIONS = ["increase", "decrease", "no_effect", "mixed", "unclear"]
MAX_PASS_CHARS = 14000


def _clip(text: str, limit: int = MAX_PASS_CHARS) -> Tuple[str, bool]:
    text = text or ""
    return (text, False) if len(text) <= limit else (text[:limit] + "\n[...truncated...]", True)


def gather_parts(db: Any, paper: Dict[str, Any]) -> Dict[str, Any]:
    secs = db.get_sections(paper["id"])
    parts: Dict[str, List[str]] = {
        k: [] for k in (
            "abstract", "introduction", "methods", "results", "discussion",
            "conclusion", "limitations", "results_discussion", "other"
        )
    }
    for s in secs:
        stype = s["section_type"]
        if stype == "article_meta":
            continue                      # JATS article-type marker (e.g. review-article), not paper text
        if stype in parts:
            parts[stype].append(f"[{s['title']}]\n{s['text']}")
        else:
            parts["other"].append(f"[{s['title']}]\n{s['text']}")

    tables = db.get_tables(paper["id"])
    tables_text = "\n\n".join(
        f"{t.get('label') or 'Table'}: {t.get('caption') or ''}\n" +
        "\n".join(" | ".join(r) for r in (t.get("rows") or [])[:40]) for t in tables
    )
    joined = {k: "\n\n".join(v) for k, v in parts.items()}
    if not joined["abstract"] and paper.get("abstract"):
        joined["abstract"] = paper["abstract"]

    has_full = bool(joined["methods"] or joined["results"] or joined["results_discussion"] or joined["other"] or joined["discussion"])
    joined["tables"] = tables_text
    joined["basis"] = "full_text" if has_full else ("abstract" if joined["abstract"] else "none")
    return joined


def prompt_abstract_extraction(abstract_text: str) -> str:
    return f"""Extract the core biological findings, study design, and organism from this scientific abstract.
Rules:
- Use ONLY the provided abstract. Do not invent details. Use null if not explicitly stated.
- Extract any reported percentage changes, fold changes, or p-values.

Return strictly valid JSON:
{{
  "organism": "Model organism or cell line, or null",
  "study_type": "in_vitro_cell_assay|in_vivo_animal_study|controlled_experiment|clinical_trial|other",
  "sample_size_summary": "Sample size mentioned or null",
  "intervention": "Primary drug, nutrient, mutation, or factor manipulated",
  "reported_findings": [
    {{
      "outcome": "Measured biological outcome",
      "direction": "increase|decrease|no_effect|unclear",
      "p_value": number or null,
      "effect_summary": "Summary of effect or percentage change",
      "quote": "Exact verbatim quote from abstract (max 30 words)"
    }}
  ],
  "author_conclusion": "One sentence summary of authors' conclusion"
}}

ABSTRACT:
{abstract_text}"""


def prompt_design(text: str) -> str:
    return f"""Extract study-design facts from this biological methods/results text.
Return strictly valid JSON:
{{
  "organism": "Organism or cell line",
  "study_type": "one of {STUDY_TYPES}",
  "sample_size_total": integer or null,
  "intervention": "Factor manipulated",
  "control": "Control condition",
  "dose": "Doses/concentrations or null",
  "duration": "Duration or null",
  "experimental_groups": [{{"name": "...", "n": integer or null}}],
  "measured_outcomes": [{{"name": "...", "unit": "..."}}],
  "statistical_methods": ["..."]
}}

TEXT:
{text}"""


def prompt_results(text: str) -> str:
    return f"""Extract reported statistical results from the RESULTS text of this biology paper.
Return strictly valid JSON:
{{
  "reported_statistics": [
    {{
      "outcome": "measured variable",
      "direction": "one of {DIRECTIONS}",
      "effect_size": number or null,
      "p_value": number or null,
      "n": integer or null,
      "quote": "exact verbatim sentence fragment (max 30 words)"
    }}
  ]
}}

TEXT:
{text}"""


DESIGN_VERSION = "design-1"


class PaperExtractor:
    def __init__(self, db: Any, router: Any = None, providers: Optional[List[str]] = None):
        self.db = db
        self.router = router
        self.providers = providers

    def _ask(self, prompt: str, validator):
        if self.router is None:
            return None
        # v25: per-paper extraction cache (design facts do not depend on the question)
        paper = getattr(self, "_paper", None)
        key = None
        if paper:
            import extraction_cache as xc
            key = xc.make_key(paper, "design", DESIGN_VERSION, prompt)
            hit = xc.get(key)
            if hit is not None:
                try:
                    return validator(hit)
                except Exception:
                    pass
        try:
            out = ask_json(self.router, prompt, validate=validator, providers=self.providers)
        except LLMJSONError:
            return None
        if key and out:
            import extraction_cache as xc
            xc.put(key, out)
        return out

    def extract(self, paper_id: int, force: bool = False) -> Dict[str, Any]:
        paper = self.db.get_paper(paper_id)
        if not paper:
            raise KeyError(f"Paper {paper_id} not found")
        if not force and self.db.get_paper_extraction(paper_id):
            return {"paper_id": paper_id, "skipped": True}

        self._paper = paper
        parts = gather_parts(self.db, paper)
        basis = parts["basis"]
        flags: List[str] = []

        if basis == "none":
            return {"paper_id": paper_id, "status": "no_text"}

        full_source = "\n\n".join(
            x for x in (
                parts["abstract"], parts["methods"], parts["results"],
                parts["results_discussion"], parts["other"], parts["tables"],
                parts["discussion"], parts["conclusion"]
            ) if x
        )
        det = {
            "p_values": parse_p_values(full_source),
            "confidence_intervals": parse_confidence_intervals(full_source),
            "sample_sizes": sorted(set(parse_sample_sizes(full_source))),
            "effect_sizes": parse_effect_sizes(full_source),
        }

        # DEDICATED ABSTRACT-ONLY EXTRACTION (Handles Paywalled Literature)
        if basis == "abstract":
            flags.append("Extraction based on abstract (full text paywalled/unavailable); methods and sample sizes preliminary.")
            res = self._ask(prompt_abstract_extraction(parts["abstract"][:3500]), lambda x: x if isinstance(x, dict) else {})
            stats_items = []
            findings = (res or {}).get("reported_findings")
            if isinstance(findings, list):
                for rf in (x for x in findings if isinstance(x, dict)):
                    quote = rf.get("quote") or ""
                    g = quote_grounding(quote, parts["abstract"])
                    stats_items.append({
                        "outcome": rf.get("outcome", "Biological phenotype"),
                        "direction": rf.get("direction", "unclear"),
                        "p_value": rf.get("p_value"),
                        "quote": quote if g >= 0.70 else parts["abstract"][:120],
                        "quote_grounding": round(g, 2),
                    })

            extraction = {
                "study": {
                    "organism": (res or {}).get("organism"),
                    "sample_size": None,
                    "study_type": (res or {}).get("study_type", "abstract_summary"),
                },
                "intervention": (res or {}).get("intervention"),
                "control": "Standard vehicle/untreated control",
                "outcomes": stats_items,
                "reported_statistics": stats_items,
                "conclusions": [(res or {}).get("author_conclusion", "")] if (res or {}).get("author_conclusion") else [],
                "basis": "abstract",
                "flags": flags,
                "deterministic_parse": det,
            }
            eid = self.db.save_paper_extraction_ext(paper_id, extraction, status="abstract_validated", basis="abstract")
            return {
                "paper_id": paper_id, "extraction_id": eid, "status": "abstract_validated",
                "basis": "abstract", "n_statistics": len(stats_items), "n_deterministic_statistics": len(det["p_values"])
            }

        # FULL-TEXT EXTRACTION PIPELINE
        results_body = parts["results"] or parts["other"] or parts["discussion"]
        results_src = "\n\n".join(x for x in (results_body, parts["results_discussion"], parts["tables"]) if x)
        methods_src = "\n\n".join(x for x in (parts["methods"], parts["other"], parts["abstract"]) if x)

        design = self._ask(prompt_design(_clip(methods_src, 9000)[0]), lambda x: x if isinstance(x, dict) else {}) or {}
        raw_stats = self._ask(prompt_results(_clip(results_src, 10000)[0]), lambda x: x if isinstance(x, dict) else {}) or {}

        verified_stats = []
        stats_list = raw_stats.get("reported_statistics") if isinstance(raw_stats.get("reported_statistics"), list) else []
        for s in (x for x in stats_list if isinstance(x, dict)):
            q = s.get("quote") or ""
            g = quote_grounding(q, full_source)
            if q and g >= 0.70:
                s["quote_grounding"] = round(g, 2)
                verified_stats.append(s)

        extraction = {
            "study": {
                "organism": design.get("organism"),
                "sample_size": design.get("sample_size_total"),
                "study_type": design.get("study_type", "controlled_experiment"),
            },
            "intervention": design.get("intervention"),
            "control": design.get("control"),
            "dose": design.get("dose"),
            "duration": design.get("duration"),
            "outcomes": verified_stats,
            "reported_statistics": verified_stats,
            "deterministic_parse": det,
            "basis": "full_text",
            "flags": flags,
        }
        eid = self.db.save_paper_extraction_ext(paper_id, extraction, status="validated", basis="full_text")
        return {
            "paper_id": paper_id, "extraction_id": eid, "status": "validated",
            "basis": "full_text", "n_statistics": len(verified_stats), "n_deterministic_statistics": len(det["p_values"])
        }