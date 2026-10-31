"""
research_engine.py
------------------
Master orchestration engine for the biology AI research stack.

Modes:
  1. Standard Scientific Investigation:
     python research_engine.py "How does metformin affect breast cancer cell proliferation and apoptosis?" --max-papers 6

  2. Closed-Loop Hypothesis Testing with Experimental CSV:
     python research_engine.py "How does metformin affect breast cancer cell proliferation and apoptosis?" \
         --data experiment.csv --group-col treatment --outcome-col viability

  3. Autonomous ReAct Agent Mode (Dynamic tool-calling loop):
     python research_engine.py "How does metformin affect breast cancer cell proliferation and apoptosis?" --agent

  4. PRISMA Evidence Dossier (screen 50 abstracts, deep-read top 10, Human vs Animal matrix,
     Unpaywall OA PDFs, STRING-db interactions, deep-reasoning executive summary):
     python research_engine.py "How does metformin affect breast cancer?" --prisma --email you@university.edu
"""

from __future__ import annotations

import concurrent.futures
import contextlib
import gc
import json
import os
import re
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from ai_router import AIRouter
from autonomous_agent import AutonomousResearchAgent
from bio_databases import BioDatabases, EntityLinker
from claim_extraction import ClaimExtractor
from contradiction_detection import detect_and_store, extract_conflicts_for_geo
from database import ResearchDB
from dataset_discovery import DatasetDiscovery
from evidence_bridge import EvidenceBridge
from hypothesis_evaluator import HypothesisEvaluator
from llm_json import ask_json
from paper_extraction import PaperExtractor
from pathway_synthesizer import PathwaySynthesizer
from report_generator import ResearchReportGenerator
from research_planner import endpoint_expansion_queries, generate_model_plan
from retrieval.source_manager import SourceManager
from scientific_tools import ScientificToolbox

# --- Additive feature modules (₹0 stack) ---
from deep_reasoner import DeepReasoner
from enhanced_report import compose as compose_enhanced_report
from evidence_matrix import build_matrix, matrix_csv, matrix_markdown
from prisma_screening import PrismaScreener, adopt_existing_claims
from string_db import StringDBClient, find_species, symbols_from_text
from unpaywall import UnpaywallClient


NO_KEY_MESSAGE = ("No API key found. BioSynthetica needs at least one API key (Groq, Gemini or NVIDIA - all free) to read "
                  "papers, extract claims and write the report; it does not run without one. Add a key (web: API keys page; "
                  "command line: GROQ_API_KEY / GEMINI_API_KEY / NVIDIA_NIM_API_KEY in .env) and start again.")


class NoAIKeyError(RuntimeError):
    """Raised when a research run is started without any AI key."""


def drop_non_genes(text: str, not_genes: set) -> str:
    """Remove words that are known NOT to be genes (a drug's abbreviation such as 'MET' = metformin, a defined
    non-gene abbreviation, the intervention itself) before gene symbols are read from verified claims. 'MET-R cells' and
    'MET-treated' are the drug too, so the abbreviation is removed wherever it stands as its own token."""
    out = " ".join(w for w in re.split(r"(\s+)", text or "") if w.strip(" ,.;:()").upper() not in not_genes)
    for a in sorted(x for x in not_genes if x):
        out = re.sub(rf"(?<![A-Za-z0-9]){re.escape(a)}(?![A-Za-z0-9])", " ", out)
    return out


EXTRACTION_WORKERS_ONE_KEY = 3


def structures_relevant(question: str) -> bool:
    """3D protein models can only help a MECHANISM question (one naming a protein, or asking how / through what)."""
    from string_db import symbols_from_text
    return bool(symbols_from_text(question or "", limit=3)) or bool(re.search(
        r"\b(?:how|mechanism\w*|pathway\w*|via|through|target\w*|bind\w*|structure\w*|protein\w*|receptor\w*|"
        r"enzyme\w*|molecular)\b", question or "", re.I))


def timings_markdown(timings: List[Dict[str, Any]]) -> str:
    """'Where the time went': one row per stage - clock time, seconds, AI calls / failures / retries / rate-limit
    hits / tokens, web requests / cache hits / retries. The slowest stage is named, so a long run shows its cause."""
    if not timings:
        return ""
    total = sum(t.get("seconds", 0) for t in timings) or 1
    out = ["## Where the time went (run instrumentation)\n",
           "| Stage | Start | End | Seconds | Share | AI calls | AI answers from cache | AI failures | AI retries (rate-limited) "
           "| AI tokens in / out | Web requests | From cache | Web retries |",
           "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for t in timings:
        out.append(f"| {t['stage'].replace('_', ' ')} | {t.get('start', '')} | {t.get('end', '')} | {t.get('seconds', 0):.1f} "
                   f"| {100 * t.get('seconds', 0) / total:.0f}% | {t.get('ai_calls', 0)} | {t.get('ai_cache_hits', 0)} | {t.get('ai_failures', 0)} "
                   f"| {t.get('ai_retries', 0)} ({t.get('ai_rate_limited', 0)}) "
                   f"| {t.get('ai_tokens_in', 0):,} / {t.get('ai_tokens_out', 0):,} | {t.get('requests', 0)} "
                   f"| {t.get('cache_hits', 0)} | {t.get('retries', 0)} |")
    sites = [f"{t['stage'].replace('_', ' ')}: {t['slowest_site']['host']} ({t['slowest_site']['seconds']:.0f} s for "
             f"{t['slowest_site']['requests']} request(s))" for t in timings if t.get("slowest_site")]
    if sites:
        out.append("\n_Slowest website per stage - " + "; ".join(sites) + "._")
    slow = max(timings, key=lambda t: t.get("seconds", 0))
    out.append(f"\n_Total {total:.0f} s. Slowest stage: **{slow['stage'].replace('_', ' ')}** ({slow.get('seconds', 0):.0f} s, "
               f"{100 * slow.get('seconds', 0) / total:.0f}% of the run). Tokens are counted when the provider reports them._\n")
    # why the AI stages took what they took: a free key accepts a fixed number of requests per minute
    try:
        from ai_pacing import summary as _pace
        ps = _pace()
        if ps.get("paced_waits") or ps.get("cooldowns"):
            rates = ", ".join(f"{k} {v:.0f}/min" for k, v in sorted((ps.get("rates_per_minute") or {}).items()))
            out.append(f"\n_AI request pacing: waited {ps['paced_seconds']:.0f} s in total across "
                       f"{ps['paced_waits']} request(s) to stay inside the key's accepted rate"
                       + (f"; {ps['cooldowns']} rate limit(s) hit, {ps['switches']} switch(es) to another provider"
                          if ps.get("cooldowns") else "")
                       + (f" (rates used: {rates})" if rates else "")
                       + ". More keys means more requests per minute: add a second free key to cut this time._\n")
    except Exception:
        pass
    return "\n".join(out)


class ResearchEngine:
    def __init__(
        self,
        db_path: Optional[str] = None,
        router: Optional[AIRouter] = None,
        raw_papers_dir: Optional[str] = None,          # None: BIOSYN_RAW_DIR, else papers/raw
        use_ai: bool = True,
        contact_email: Optional[str] = None,
        vector_store_path: Optional[str] = None,      # None: vector_store.DEFAULT_PATH (next to the program)
        progress: Optional[Callable[[str, str], None]] = None,
        openalex_key: Optional[str] = None,
        s2_key: Optional[str] = None,
    ):
        self.db = ResearchDB(db_path)
        self.router = router or (AIRouter() if use_ai else None)
        if self.router is not None and hasattr(self.router, "configured_providers") and not self.router.configured_providers():
            self.router = None
        # Without an AI key the engine does not run a research investigation (no silent 'deterministic mode').
        # use_ai=False is an explicit developer / test opt-out only.
        self.allow_without_ai = not use_ai
        self.providers = ["groq", "nvidia", "gemini"] if self.router else None

        self.source_manager = SourceManager(raw_papers_dir=raw_papers_dir, openalex_key=openalex_key, s2_key=s2_key)
        self.bio_client = BioDatabases()
        self.linker = EntityLinker(self.db, bio=self.bio_client, router=self.router)
        self.dataset_discovery = DatasetDiscovery(db=self.db, router=self.router)
        self.tools = ScientificToolbox(db=self.db)
        self.bridge = EvidenceBridge(db=self.db)
        self.reporter = ResearchReportGenerator(db=self.db)
        self.pathways = PathwaySynthesizer(db=self.db)
        self.evaluator = HypothesisEvaluator(db=self.db)

        # Additive components (no change to the pipeline above)
        self.contact_email = contact_email or os.getenv("UNPAYWALL_EMAIL")
        self.unpaywall = UnpaywallClient(self.contact_email)
        self.string_db = StringDBClient()
        self.vector_store_path = vector_store_path
        self._vector_store = None
        self.progress = progress

    @property
    def vector_store(self):
        """Lazy: loading the local embedding model takes a few seconds the first time."""
        if self._vector_store is None:
            from vector_store import LocalVectorStore
            self._vector_store = LocalVectorStore(self.vector_store_path) if self.vector_store_path else LocalVectorStore()
        return self._vector_store

    def _counters(self) -> Dict[str, int]:
        """AI calls / failures and web requests / cache hits / retries so far (process-wide counters)."""
        import http_client
        out = dict(http_client.STATS)
        out["_hosts"] = {h: list(v) for h, v in http_client.HOST_TIME.items()}
        out["cpu_ms"] = int(time.process_time() * 1000)          # v26: CPU time per stage (all threads of this process)
        tot = getattr(getattr(self.router, "health", None), "totals", None)
        if callable(tot):
            try:
                t = tot()
                out.update({"ai_calls": t.get("calls", 0), "ai_failures": t.get("failures", 0),
                            "ai_tokens_in": t.get("tokens_in", 0), "ai_tokens_out": t.get("tokens_out", 0),
                            "ai_tokens_cached": t.get("tokens_cached", 0),
                            "ai_retries": t.get("retries", 0), "ai_rate_limited": t.get("rate_limited", 0),
                            "ai_cache_hits": t.get("cache_hits", 0)})
            except Exception:
                pass
        return out

    def _close_stage(self) -> None:
        """Record how long the stage that just ended took, and how many AI calls / web requests it made (shown in the
        run log and the dossier) - so a slow run shows exactly where its time went."""
        cur = getattr(self, "_cur_stage", None)
        if cur:
            dt = time.time() - cur[1]
            now = self._counters()
            before = cur[2] if len(cur) > 2 else {}
            hosts_now, hosts_before = now.pop("_hosts", {}), before.get("_hosts", {})
            delta = {k: now.get(k, 0) - before.get(k, 0) for k in now if not k.startswith("_")}
            host_d = {h: (v[0] - hosts_before.get(h, [0, 0])[0], v[1] - hosts_before.get(h, [0, 0])[1])
                      for h, v in hosts_now.items()}
            host_d = {h: v for h, v in host_d.items() if v[1] > 0}
            entry = {"stage": cur[0], "seconds": round(dt, 1),
                     "start": time.strftime("%H:%M:%S", time.localtime(cur[1])),
                     "end": time.strftime("%H:%M:%S", time.localtime(cur[1] + dt)),
                     **{k: v for k, v in delta.items() if v}}
            if host_d:
                h, (sec, n) = max(host_d.items(), key=lambda kv: kv[1][0])
                entry["slowest_site"] = {"host": h, "seconds": round(sec, 1), "requests": n}
            self._timings = getattr(self, "_timings", []) + [entry]
            extra = []
            if delta.get("ai_calls"):
                extra.append(f"{delta['ai_calls']} AI call(s)" + (f", {delta['ai_failures']} failed" if delta.get("ai_failures") else "")
                             + (f", {delta['ai_retries']} retries" if delta.get("ai_retries") else "")
                             + (f" ({delta['ai_rate_limited']} rate-limited)" if delta.get("ai_rate_limited") else "")
                             + (f", {delta['ai_cache_hits']} answered from the AI cache" if delta.get("ai_cache_hits") else "")
                             + (f", ~{delta.get('ai_tokens_in', 0) + delta.get('ai_tokens_out', 0):,} tokens"
                                if delta.get("ai_tokens_in") or delta.get("ai_tokens_out") else ""))
            if delta.get("requests") or delta.get("cache_hits"):
                extra.append(f"{delta.get('requests', 0)} web request(s), {delta.get('cache_hits', 0)} from cache"
                             + (f", {delta['retries']} retries" if delta.get("retries") else ""))
            print(f"  [Time] {cur[0].replace('_', ' ')}: {dt:.1f} s" + (" · " + " · ".join(extra) if extra else ""))
        self._cur_stage = None

    def _stage(self, stage: str, message: str = "") -> None:
        """Progress hook for the web UI; prints in CLI mode exactly like other steps."""
        self._close_stage()
        self._cur_stage = (stage, time.time(), self._counters())
        print("\n" + "=" * 80)
        print(f"{stage.upper().replace('_', ' ')}" + (f": {message}" if message else ""))
        print("=" * 80)
        if self.progress:
            try:
                self.progress(stage, message)
            except Exception:
                pass

    # -------------------------------------------------------------------------
    # Literature Acquisition & Processing
    # -------------------------------------------------------------------------

    def acquire_literature(
        self,
        question: str,
        research_question_id: int,
        max_papers: int = 6,
        min_relevance: float = 0.30,
    ) -> List[int]:
        queries = self.plan_queries(question)      # concept queries (+ AI plan when a key is present)

        return self.source_manager.retrieve_and_ingest(
            db=self.db,
            queries=queries,
            question=question,
            research_question_id=research_question_id,
            max_papers=max_papers,
            min_relevance=min_relevance,
        )

    def process_papers(
        self,
        paper_ids: List[int],
        research_question_id: int,
        question: str,
        link_bio: bool = True,
        max_claims_per_paper: int = 7,
    ) -> None:
        ext_order = self.router.order_for("extraction") if self.router is not None and hasattr(self.router, "order_for") else None
        # papers are extracted in parallel: several AI keys -> one worker per provider, each leading with a different
        # provider (separate rate limits); one key -> 3 workers on that key (the router waits and retries on a rate
        # limit, so a burst costs a few seconds, while sequential reading cost minutes)
        n_prov = len(ext_order or [])
        from ai_pacing import workers_for
        workers = workers_for(n_prov, len(paper_ids), EXTRACTION_WORKERS_ONE_KEY) if self.router is not None else 1
        lead_order = ext_order if n_prov > 1 else None
        prov = None if lead_order else (ext_order or self.providers)      # None = the thread's preferred provider first
        paper_extractor = PaperExtractor(self.db, self.router, providers=prov)
        # short papers (an abstract each) are read several per AI call: a free key accepts a fixed number of requests
        # per minute, so fewer calls is the only way under that floor. Each claim is still verified against its own
        # paper, and any paper the batch did not answer is read the ordinary way.
        prefetched = {}
        if self.router is not None and len(paper_ids) > 1 and not os.getenv("BIOSYN_NO_BATCH"):
            try:
                from claim_extraction import prefetch_claims
                prefetched = prefetch_claims(self.db, self.router, list(paper_ids), question, log=print)
            except Exception as exc:
                print(f"  [Batch notice] reading papers one at a time ({type(exc).__name__}: {str(exc)[:100]})")
                prefetched = {}
        claim_extractor = ClaimExtractor(self.db, self.router, providers=prov, prefetched=prefetched)
        claim_extractor.quick = bool(getattr(self, "_quick", False))      # v25: no second reader in a quick scan

        try:                                  # v26.2: papers added / updated by the search on THIS thread are
            self.db.commit()                  # committed before the reader threads start, so they can see them
        except Exception:
            pass

        def _extract(i_pid):
            i, pid = i_pid
            paper_rec = self.db.get_paper(pid)
            title = (paper_rec or {}).get("title") or f"Paper #{pid}"
            lead = lead_order[(i - 1) % len(lead_order)] if lead_order else None
            ctx = self.router.prefer(lead) if lead else contextlib.nullcontext()
            try:
                with ctx:
                    # quick scan: the study design is read from title / abstract by the classifier rules, so the separate
                    # AI design-extraction call per paper is skipped (half the AI calls)
                    p_res = ({"status": "skipped (quick scan)", "n_statistics": 0} if getattr(self, "_quick", False)
                             else paper_extractor.extract(pid, force=False))
                    c_res = claim_extractor.extract(
                        pid, research_question_id, question, force=bool(getattr(self, "_force_reextract", False)),
                        max_claims=max_claims_per_paper
                    )
            except Exception as exc:          # one unreadable paper never stops the whole run
                try:                          # v26.2: never leave this thread's write transaction open (it held the
                    self.db.rollback()        # database lock, so the NEXT stage failed with 'database is locked')
                except Exception:
                    pass
                print(f"\n--- [{i}/{len(paper_ids)}] Skipped: {title[:75]} - extraction failed ({type(exc).__name__}: "
                      f"{str(exc)[:160]}) ---")
                return pid
            print(f"\n--- [{i}/{len(paper_ids)}] Analyzed: {title[:75]}" + (f" (lead AI: {lead})" if lead else "") + " ---")
            print(f"  [Design & Stats] Status: {p_res.get('status')} | Stats found: {p_res.get('n_statistics', 0)}")
            print(f"  [Claims] Status: {c_res.get('status')} | Verified: {c_res.get('claims', 0)}")
            try:
                self.db.commit()              # this reader thread's writes are published at once (never left open)
            except Exception:
                pass
            return pid

        items = list(enumerate(paper_ids, 1))
        if workers > 1:
            with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
                list(ex.map(_extract, items))
        else:
            for it in items:
                _extract(it)

        def _link(pid):
            try:
                b_res = self.linker.link_paper(pid)
                linked_count = len(b_res.get("linked", []))
                sample = f": {', '.join(b_res.get('linked', [])[:3])}" if linked_count else ""
                print(f"  [Entities] Linked: {linked_count} symbols (NCBI/UniProt/PubChem){sample}")
            except Exception as exc:
                print(f"  [Entities notice] {exc}")

        if link_bio:            # network look-ups, independent per paper: run side by side (NCBI pacing is shared)
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:
                list(ex.map(_link, paper_ids))
        gc.collect()

    # -------------------------------------------------------------------------
    # Contradictions, Pathways, and Loop Closure
    # -------------------------------------------------------------------------

    def detect_contradictions(self, research_question_id: int) -> Dict[str, Any]:
        return detect_and_store(self.db, research_question_id, router=self.router)

    def run_empirical_closure(
        self,
        research_question_id: int,
        conflicts: List[Dict[str, Any]],
        limit: int = 3,
    ) -> List[Dict[str, Any]]:
        results = []
        for c in conflicts[:limit]:
            query = f"{c['intervention']} {c['organism']}"
            print(f"\n[GEO Loop Closure] Evaluating disagreement on: '{c['intervention']}' -> '{c['outcome']}'")

            datasets = self.dataset_discovery.discover(
                question=query,
                organism=c["organism"],
                sources=("GEO",),
                per_query=4,
                min_relevance=0.20,
                research_question_id=research_question_id,
            )

            if datasets:
                top = datasets[0]
                stmt = (
                    f"Public dataset {top['accession']} found (not analysed here) that could help evaluate the disagreement on "
                    f"'{c['intervention']}' and '{c['outcome']}' in {c['organism']}."
                )
                eid = self.db.add_evidence(
                    research_question_id=research_question_id,
                    evidence_type="DATASET_FOUND",
                    statement=stmt,
                    source_type="dataset",
                    confidence="medium",
                    value={"geo_accession": top["accession"], "relevance_reason": top.get("relevance_reason")},
                )
                results.append({"conflict": c["group_key"], "dataset": top["accession"], "evidence_id": eid})
                print(f"  -> Linked empirical dataset: {top['accession']} ({top['title'][:65]}...)")
            else:
                print("  -> No public GEO series matrix found matching this conflict.")

        return results

    def synthesize_mechanistic_pathways(self, research_question_id: int) -> Dict[str, Any]:
        pathway_results = self.pathways.build_pathways(research_question_id)
        count = pathway_results.get("total_chains_discovered", 0)
        print(f"  -> Discovered {count} cross-study mechanistic pathway chains.")
        for ch in pathway_results.get("mechanistic_chains", [])[:3]:
            print(f"     * {ch['pathway']}")
        return pathway_results

    # -------------------------------------------------------------------------
    # Hypotheses & Research Gaps (Dynamic Universal Synthesis)
    # -------------------------------------------------------------------------

    def formulate_hypotheses_and_gaps(
        self,
        research_question_id: int,
        question: str,
    ) -> Dict[str, Any]:
        claims = self.db.primary_claims(research_question_id)      # a review's relayed results never seed a hypothesis
        contradictions = self.db.current_contradictions(research_question_id)

        evidence_summary = [
            f"- [C{c['id']}] {(c.get('evidence_statement') or c.get('statement') or '')[:200]} | model: "
            f"{(c.get('claim_model') or '?').replace('_', ' ')} | status: {c.get('claim_status') or '?'} | "
            f"{(c.get('canonical_outcome') or c.get('measurement') or '?')} {c.get('canonical_direction') or c.get('effect_direction')}"
            + (f" | combination regimen: {', '.join((c.get('semantic_audit') or {}).get('combination_components') or [])}"
               if (c.get('semantic_audit') or {}).get('combination_components') else "")
            + f" | role: {(c.get('semantic_audit') or {}).get('evidence_role') or 'direct'}"
            + f" | subtype: {', '.join(((c.get('semantic_audit') or {}).get('identity') or {}).get('disease_subtype') or ['not specified'])}"
            for c in claims[:20]
        ]
        conflict_summary = [
            f"- Conflict between claim {c['claim_a_id']} and {c['claim_b_id']}: verdict={c['verdict']} ({c.get('reasoning')})"
            for c in contradictions[:5]
        ]

        prompt = f"""You are a senior scientific synthesis engine in a biology research pipeline.

Research Question: {question}

Observed Evidence:
{chr(10).join(evidence_summary) if evidence_summary else 'No specific claim statistics recorded.'}

Detected Cross-Study Disagreements:
{chr(10).join(conflict_summary) if conflict_summary else 'No severe contradictions detected; findings cover complementary biological axes.'}

Generate:
1. Exactly 2 mechanistic, falsifiable hypotheses that explain the observed biological findings or reconcile disagreements.
2. For each hypothesis, design a concrete next-step validation experiment with clear controls and a falsification condition.
3. Exactly 2 critical research gaps that remain unaddressed by the current literature.

RULES:
- Every hypothesis and gap must list in "derived_from" the exact claim IDs (e.g. "C12") above that produced it.
  If none of the claims supports it, return an empty list - it will be shown as an AI hypothesis.
- A combination regimen is evidence for the regimen only; never build a hypothesis or gap on one component of it.
- Only propose a pharmacokinetic gap for a systemic drug; procedures, topical or combination regimens need other gaps.
- Any dose, sample size, threshold, effect size or duration you propose is an assumption, not evidence.
- Name only genes, translocations, pathways and exposures that appear in the derived claims. Anything else (e.g. a
  translocation or oncogene you know from background knowledge) makes the hypothesis an AI-generated mechanistic
  hypothesis informed by literature - it will be labelled that way, never as empirically observed.
- A "mechanistic" or "intermediate" claim is not a cause of the disease; a gap must not assume the untested link
  (e.g. "chromosome aberrations leading to leukemia") is established - say that link is what is missing.
- State the disease subtype each hypothesis/gap concerns; a heterogeneous disease is not one entity.

Return ONLY valid JSON matching this schema:
{{
  "hypotheses": [
    {{
      "statement": "Clear testable statement",
      "derived_from": ["C<id>", "..."],
      "mechanism": "Proposed pathway written as a signed chain so the virtual lab can test it: use -> for activates / increases and -| for inhibits / decreases, e.g. 'metformin -> AMPK -| mTOR -> proliferation'; add one short sentence after it if needed",
      "predictions": ["prediction 1", "prediction 2"],
      "competing_explanations": ["at least one rival explanation for the same observations (confounder, reverse causation, model-specific effect)"],
      "falsification_conditions": ["condition that would refute this"],
      "next_experiment": {{
        "experiment_title": "...",
        "controls": "...",
        "independent_variable": "...",
        "dependent_variable": "...",
        "power_notes": "sample size and replication requirements"
      }}
    }}
  ],
  "research_gaps": [
    {{
      "description": "...",
      "derived_from": ["C<id>"],
      "category": "dosage|timing|mechanism|organism_generalization",
      "supporting_observation": "..."
    }}
  ]
}}"""

        def validate(obj: Any) -> Dict[str, Any]:
            if not isinstance(obj, dict):
                raise ValueError("Response must be a JSON object.")

            for wrap_key in ("synthesis", "result", "data", "output", "response", "analysis"):
                if wrap_key in obj and isinstance(obj[wrap_key], dict):
                    obj = obj[wrap_key]
                    break

            raw_hypotheses = obj.get("hypotheses") or obj.get("hypothesis") or []
            raw_gaps = obj.get("research_gaps") or obj.get("gaps") or []

            if isinstance(raw_hypotheses, dict):
                raw_hypotheses = [raw_hypotheses]
            if isinstance(raw_gaps, dict):
                raw_gaps = [raw_gaps]

            if not isinstance(raw_hypotheses, list) or not isinstance(raw_gaps, list):
                raise ValueError("Expected lists for hypotheses and research_gaps.")

            clean_hypotheses = []
            for h in raw_hypotheses:
                if not isinstance(h, dict):
                    continue
                preds = h.get("predictions")
                if isinstance(preds, str):
                    preds = [preds]
                elif not isinstance(preds, list):
                    preds = ["Measurable biological response"]

                falsif = h.get("falsification_conditions")
                if isinstance(falsif, str):
                    falsif = [falsif]
                elif not isinstance(falsif, list):
                    falsif = ["No statistically significant effect detected upon replication"]

                next_exp = h.get("next_experiment")
                if not isinstance(next_exp, dict):
                    next_exp = {"experiment_title": str(next_exp or "Replication assay")}

                clean_hypotheses.append({
                    "derived_from": h.get("derived_from") or [],
                    "statement": str(h.get("statement") or "Proposed biological hypothesis"),
                    "mechanism": str(h.get("mechanism") or "Regulatory molecular pathway"),
                    "predictions": preds,
                    "falsification_conditions": falsif,
                    "next_experiment": next_exp,
                })

            clean_gaps = []
            for g in raw_gaps:
                if isinstance(g, str):
                    clean_gaps.append({
                        "description": g.strip(),
                        "category": "generalization",
                        "supporting_observation": "Identified from cross-study literature gaps",
                    })
                elif isinstance(g, dict):
                    clean_gaps.append({
                        "derived_from": g.get("derived_from") or [],
                        "description": str(g.get("description") or "Unaddressed research gap"),
                        "category": str(g.get("category") or "mechanism"),
                        "supporting_observation": str(g.get("supporting_observation") or "Literature boundary"),
                    })

            if not clean_hypotheses and not clean_gaps:
                raise ValueError("No valid hypotheses or research gaps parsed.")

            return {"hypotheses": clean_hypotheses, "research_gaps": clean_gaps}

        res = None
        if self.router and not getattr(self, "_quick", False):      # quick mode: evidence-derived hypotheses only (no AI call)
            try:
                res = ask_json(self.router, prompt, validate=validate,
                               providers=self.router.order_for("hypotheses") if hasattr(self.router, "order_for") else self.providers,
                               max_attempts=2)            # two tries at most, then the evidence-derived hypotheses
            except Exception as exc:
                print(f"  [AI synthesis notice] Falling back to dynamic hypothesis generator: {exc}")

        if not res or not res.get("hypotheses"):
            res = self._deterministic_hypothesis_fallback(question, claims, contradictions)

        from reasoning_guards import mechanism_status, parse_derived, validate_gaps
        self.db.retire_hypotheses(research_question_id)      # this run's hypotheses replace the previous run's
        valid_ids = {c["id"] for c in claims}
        claims_by_id = {c["id"]: c for c in claims}
        created_hypotheses = []
        for h in res.get("hypotheses", []):
            h["derived_from"] = parse_derived(h.get("derived_from"), valid_ids)
            h["mechanism_status"] = mechanism_status(h, claims_by_id)
            evidence_summary_h = {"derived_from": [f"C{i}" for i in h["derived_from"]],
                                  "mechanism_status": h["mechanism_status"]}
            hid = self.db.add_hypothesis(
                research_question_id=research_question_id,
                hypothesis={
                    "statement": h.get("statement", ""),
                    "mechanism": h.get("mechanism"),
                    "predictions": h.get("predictions", []),
                    "falsification_conditions": h.get("falsification_conditions", []),
                    "competing_explanations": h.get("competing_explanations", []),
                    "supporting_evidence": evidence_summary_h,
                    "next_experiment": json.dumps(h.get("next_experiment", {})),
                    "status": "proposed",
                },
            )
            if h.get("next_experiment"):
                self.db.add_experiment_design(
                    hypothesis_id=hid,
                    design=h["next_experiment"],
                    power_notes=h["next_experiment"].get("power_notes", "Sample size >= 12 across 3 independent replicates"),
                    confounder_notes="Randomized placement with untreated vehicle controls.",
                )
            created_hypotheses.append({"id": hid, **h})

        from evidence_identity import question_frame
        gaps_ok, gap_notes = validate_gaps(res.get("research_gaps", []), claims, question_frame(question))
        for n in gap_notes:
            print(f"  [Gap check] {n}")
        created_gaps = []
        self.db.clear_research_gaps(research_question_id)
        for g in gaps_ok:
            gid = self.db.add_research_gap(
                research_question_id=research_question_id,
                description=g.get("description", ""),
                category=g.get("category", "general"),
                supporting_observation=f"{g.get('evidence_status', '')} | {g.get('supporting_observation', '')}".strip(" |"),
            )
            created_gaps.append({"id": gid, **g})

        ranked = self.consolidate_hypotheses(research_question_id, question)
        return {"hypotheses": ranked or created_hypotheses, "research_gaps": created_gaps}

    def consolidate_hypotheses(self, research_question_id: int, question: str,
                               matrix: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Merge overlapping hypotheses (all modes, all rounds) and rank them by verified support
        weighted by evidence level. Merged duplicates are kept in the DB with status='merged'."""
        from deep_reasoner import question_focus
        from reasoning_guards import consolidate_hypotheses
        from study_classifier import evidence_level, get_or_classify

        focus = question_focus(question)
        refs = {r["paper_id"]: r["ref"] for r in (matrix or {}).get("rows", [])}
        evidence = []
        for c in self.db.primary_claims(research_question_id):
            if matrix is not None and c.get("paper_id") not in refs:
                continue                      # only studies that are in the evidence matrix can support a hypothesis
            cls = get_or_classify(self.db, c["paper_id"]) if c.get("paper_id") else {"evidence_type": "unclear"}
            evidence.append({**c, "level": evidence_level(c.get("claim_model") or cls["evidence_type"], focus, cls.get("organism_group")),
                             "ref": refs.get(c.get("paper_id"), f"paper {c.get('paper_id')}")})
        stored = self.db.list_hypotheses(research_question_id, include_merged=True)
        live = [h for h in stored if h.get("status") not in ("merged", "superseded")]   # earlier runs' hypotheses stay out
        if not live:
            return []
        ranked = consolidate_hypotheses(live, evidence)
        by_statement = {h["statement"]: h for h in live}
        for h in ranked:
            self.db.set_hypothesis_rank(h["id"], h["rank"], h["title"], h["support"])
            for dup in h.get("merged_from") or []:
                d = by_statement.get(dup)
                if d and d["id"] != h["id"]:
                    self.db.mark_hypothesis_merged(d["id"], h["id"])
        return ranked

    def _deterministic_hypothesis_fallback(
        self,
        question: str,
        claims: list,
        contradictions: list,
    ) -> Dict[str, Any]:
        """Dynamically derives universal hypotheses from actual question tokens and claims."""
        from semantic_audit import FAMILY_LABEL
        answering0 = [c for c in claims if (c.get("semantic_audit") or {}).get("answers_question")]
        if not answering0:
            # nothing verified answers the question: no canned template ('Can regulates biological response ...')
            # - an empty, honest section instead of an invented hypothesis
            print("  [Hypotheses] No verified finding answers the question - no template hypothesis generated "
                  "(add an AI key and/or raise the deep-read budget).")
            return {"hypotheses": [], "research_gaps": []}
        # Without an AI key no mechanism is invented. Each hypothesis is DERIVED from one verified finding that
        # answers the question: "what was observed, in which system / subtype - does it replicate one step closer
        # to the target organism?" (no 'concentration-dependent signaling threshold' templates, no question words).
        # Study-level coverage, not claim-level: one paper can hold a cell assay AND a mouse xenograft, and other papers
        # can hold the next level already. For every agent x endpoint group, the systems that ALREADY have verified
        # findings are collected across all papers; a hypothesis / gap proposes only the first level that is missing.
        LADDER = ["in_silico", "cell_culture", "animal", "human_observational", "human_clinical"]
        LEVEL_OF = {"organoid_ex_vivo": "cell_culture", "experimental_disease_model": "animal", "case_report": "human_observational"}
        LADDER_NAME = {"in_silico": "computational", "cell_culture": "cell culture", "animal": "animal (in vivo)",
                       "human_observational": "human observational", "human_clinical": "randomised controlled (human)"}
        GROUP = {"proliferation": "growth", "tumour_growth": "growth", "viability": "growth"}
        verb = {"increase": "increases", "decrease": "decreases", "no_effect": "does not change"}
        from semantic_audit import subgroup_of

        def agent(c):
            return ((c.get("intervention") or "").strip().lower().split() or [""])[0]

        from semantic_audit import finding_target

        def group(c):
            fam = c.get("canonical_outcome") or ""
            if fam in ("expression", "signalling", "protein_localization"):
                # METTL3 expression and p53 expression are different endpoints (never one 'expression' group)
                return f"{fam}:{finding_target(c) or 'C' + str(c.get('id'))}"
            return GROUP.get(fam, fam)

        def endpoint_name(c):
            fam = c.get("canonical_outcome") or ""
            t = finding_target(c)
            return f"{t} {FAMILY_LABEL.get(fam, fam)}" if t else FAMILY_LABEL.get(fam, c.get("measurement") or "the outcome")

        primary_all = [c for c in claims if (c.get("semantic_audit") or {}).get("evidence_role") not in ("secondary_report",
                                                                                                      "combination")]
        coverage: Dict[Any, Dict[str, list]] = {}
        for c in primary_all:
            m = LEVEL_OF.get(c.get("claim_model") or "", c.get("claim_model") or "")
            if m in LADDER and agent(c):
                coverage.setdefault((agent(c), group(c)), {}).setdefault(m, []).append(c)
        from semantic_audit import _READOUT_IV, _GENE_IV, _MANIPULATION

        def is_agent(iv: str) -> bool:
            # a measured read-out ('Rb phosphorylation') or a bare gene ('FOXA1 gene') is not something that was given:
            # it never becomes the subject of a replication hypothesis or a 'no evidence' gap
            if _MANIPULATION.search(iv):
                return True
            gm = _GENE_IV.match(iv)
            return not (_READOUT_IV.search(iv) or (gm and re.search(r"\d|[A-Z]{3,}", gm.group(1))))

        seen, hyps, gaps = set(), [], []

        def _lvl(c):
            m = LEVEL_OF.get(c.get("claim_model") or "", c.get("claim_model") or "")
            return LADDER.index(m) if m in LADDER else -1

        # each agent x endpoint is described by its STRONGEST verified finding (human before animal before cells)
        for c in sorted(answering0, key=lambda c: -_lvl(c)):
            interv = (c.get("intervention") or "").strip()
            fam = c.get("canonical_outcome") or ""
            key = (agent(c), group(c))
            if not interv or key in seen or not is_agent(interv):
                continue
            seen.add(key)
            have = coverage.get(key, {})
            levels = [m for m in LADDER if m in have]
            model = c.get("claim_model") or "unclear"
            endpoint = endpoint_name(c)
            subs = [x for x in (((c.get("semantic_audit") or {}).get("identity") or {}).get("disease_subtype") or [])
                    if x != "not specified"]
            sg = subgroup_of(c.get("quote"))
            from semantic_audit import separate_subgroups
            seps = separate_subgroups(c.get("quote"))
            if seps:
                # 'in women with HOMA > 2.8, ... and those with HER2-positive tumors' = SEPARATE subgroup analyses: never
                # written as one population that has all of them ('HER2-positive women with HOMA > 2.8')
                where = (model.replace("_", " ") + f"; in separately reported subgroups ({'; '.join(seps[:4])}"
                         + (" ..." if len(seps) > 4 else "") + ") - each analysed on its own, not their intersection")
                sg = None
            else:
                where = model.replace("_", " ") + (f", {', '.join(subs)}" if subs else "") + \
                    (f"; only in {sg}" if sg and sg != "overall" else "")
            others = "; ".join(f"{LADDER_NAME.get(m, m)}: " + ", ".join(f"C{x['id']}" for x in [y for y in have[m] if y is not c][:4])
                               for m in levels if [y for y in have[m] if y is not c])
            observed = "; ".join(f"{LADDER_NAME.get(m, m)}: " + ", ".join(f"C{x['id']}" for x in have[m][:4]) for m in levels) \
                or f"{model.replace('_', ' ')}: C{c['id']}"
            # study-level coverage: propose the first level ABOVE the highest one already verified (across all papers);
            # when the top of the ladder is covered, the next step is replication in another population of that design
            top_i = max([LADDER.index(m) for m in levels] or [LADDER.index(LEVEL_OF.get(model, model))
                                                              if LEVEL_OF.get(model, model) in LADDER else -1])
            missing = [m for m in LADDER[top_i + 1:]]
            step = LADDER_NAME.get(missing[0]) if missing else "randomised controlled (another population)"
            d = c.get("canonical_direction") or c.get("direction") or "increase"
            nulls = [x for x in answering0 if x is not c and (x.get("canonical_direction") or x.get("direction")) == "no_effect"
                     and agent(x) == key[0] and group(x) == key[1]]
            null_txt = (f"; no effect in {'the overall population' if subgroup_of(nulls[0].get('quote')) == 'overall' else 'a ' + (nulls[0].get('claim_model') or 'other').replace('_', ' ') + ' study'}"
                        f" (C{nulls[0]['id']})" if nulls and d != "no_effect" else "")
            if d == "no_effect":
                # the strongest finding is null: an effect seen in a weaker design is stated next to it, not hidden
                effs = [x for x in answering0 if x is not c and agent(x) == key[0] and group(x) == key[1]
                        and (x.get("canonical_direction") or x.get("direction")) in ("increase", "decrease")]
                if effs:
                    null_txt = (f"; {(effs[0].get('canonical_direction') or effs[0].get('direction'))} reported in a "
                                f"{(effs[0].get('claim_model') or 'other').replace('_', ' ')} study (C{effs[0]['id']})")
            human_null = [x for m in ("human_clinical", "human_observational") for x in have.get(m, [])
                          if (x.get("canonical_direction") or x.get("direction")) == "no_effect"]
            lab_effect = [x for m in ("cell_culture", "animal") for x in have.get(m, [])
                          if (x.get("canonical_direction") or x.get("direction")) in ("increase", "decrease")
                          # a cell experiment cannot measure a PATIENT outcome: never compared with trial survival
                          and not (m == "cell_culture" and (x.get("canonical_outcome") or "") in
                                   ("patient_survival", "recurrence_incidence", "clinical_response"))]
            obs_eff = [x for x in have.get("human_observational", []) if (x.get("canonical_direction") or x.get("direction")) in ("increase", "decrease")]
            rct_null = [x for x in have.get("human_clinical", []) if (x.get("canonical_direction") or x.get("direction")) == "no_effect"]
            if obs_eff and rct_null:
                # the randomised evidence already exists and disagrees: the gap is the discrepancy, not 'no RCT'
                gaps.append({"description": f"Observational data (C{obs_eff[0]['id']}) link {interv} to {endpoint} "
                                            f"{(obs_eff[0].get('canonical_direction') or '').replace('_', ' ')}, but a randomised trial "
                                            f"(C{rct_null[0]['id']}) found no statistically significant difference - unexplained (confounding, population or dose "
                                            f"differences) in the deep-read sample.",
                             "category": "discrepancy", "derived_from": [f"C{obs_eff[0]['id']}", f"C{rct_null[0]['id']}"],
                             "supporting_observation": observed})
            elif human_null and lab_effect:
                gaps.append({"description": f"Laboratory studies show {interv} {verb.get(lab_effect[0].get('canonical_direction'), 'changes')} "
                                            f"{endpoint} ({', '.join('C' + str(x['id']) for x in lab_effect[:4])}), but human evidence shows "
                                            f"no statistically significant difference ({', '.join('C' + str(x['id']) for x in human_null[:3])}) - unexplained (dose reached "
                                            f"in patients, population, endpoint) in the deep-read sample.",
                             "category": "discrepancy", "derived_from": [f"C{x['id']}" for x in (lab_effect[:2] + human_null[:1])],
                             "supporting_observation": observed})
            elif missing:
                gaps.append({"description": f"No {step} evidence on {interv} and {endpoint}"
                                            + (f" in {', '.join(subs)}" if subs else "") + " in the deep-read sample"
                                            + (f" (verified so far: {observed})." if observed else "."),
                             "category": "translation", "derived_from": [f"C{x['id']}" for m in levels for x in have[m][:2]] or [f"C{c['id']}"],
                             "supporting_observation": observed})
            hyps.append({
                "statement": f"{interv[0].upper() + interv[1:]} {verb.get(d, 'changes')} {endpoint} ({where}; observed in C{c['id']}"
                             f"{null_txt}{'; also verified in ' + others if others else ''}) and the "
                             f"{'null result holds' if d == 'no_effect' else 'effect replicates'} in an independent "
                             f"{step.replace('independent ', '')} study{' restricted to ' + sg if sg and sg != 'overall' else ''}.",
                "mechanism": "Not specified - no mechanistic evidence was verified for this effect.",
                "predictions": [f"An independent {step} study finds "
                                f"{ {'decrease': 'a decrease in', 'increase': 'an increase in'}.get(d, 'no change in') } "
                                f"{endpoint} with {interv}."],
                "falsification_conditions": [f"An adequately powered {step} study finds a significant change in {endpoint} with {interv}."
                                             if d == "no_effect" else
                                             f"An adequately powered {step} study finds no difference in {endpoint}."],
                "next_experiment": {"experiment_title": f"Independent {step} study: {interv} and {endpoint}",
                                    "model": step, "independent_variable": interv, "dependent_variable": endpoint,
                                    "controls": "concurrent untreated / standard-care comparator"},
                "derived_from": [f"C{c['id']}"] + [f"C{x['id']}" for m in levels for x in have[m][:2] if x is not c][:4],
            })
            if len(hyps) >= 3:
                break
        # across endpoints: laboratory growth effects next to a null randomised trial on a clinical outcome
        lab_growth = [x for (ag, gp), lv in coverage.items() if gp == "growth" for m in ("cell_culture", "animal")
                      for x in lv.get(m, []) if (x.get("canonical_direction") or x.get("direction")) in ("increase", "decrease")]
        human_growth = [x for (ag, gp), lv in coverage.items() if gp == "growth" for m in ("human_observational", "human_clinical")
                        for x in lv.get(m, [])]
        rct_null_any = [x for lv in coverage.values() for x in lv.get("human_clinical", [])
                        if (x.get("canonical_direction") or x.get("direction")) == "no_effect" and group(x) != "growth"]
        if lab_growth and rct_null_any and not human_growth:
            gaps.append({"description": f"Laboratory studies show reduced proliferation / tumour growth "
                                        f"({', '.join('C' + str(x['id']) for x in lab_growth[:4])}), but the randomised human evidence "
                                        f"({', '.join('C' + str(x['id']) for x in rct_null_any[:2])}) found no benefit on a clinical outcome, "
                                        f"and no human study in the deep-read sample measured proliferation itself (e.g. Ki-67).",
                         "category": "discrepancy", "derived_from": [f"C{x['id']}" for x in lab_growth[:2] + rct_null_any[:1]],
                         "supporting_observation": "study-level coverage across all deep-read papers"})
        # MECHANISTIC hypotheses - only where a verified passage itself shows HOW the agent acts (a mediator that is
        # required: effect lost after knockdown = M3; or a route the source names: 'via FOXA1 inhibition' = M2).
        # Nothing is invented: no verified mediator, no mechanistic hypothesis.
        mech_seen = set()
        for c in answering0:
            me = (c.get("semantic_audit") or {}).get("mechanism_evidence") or {}
            interv = (c.get("intervention") or "").strip()
            if not me.get("mediator") or not interv or not is_agent(interv):
                continue
            med = me["mediator"]
            mkey = (agent(c), med.lower())
            if mkey in mech_seen:
                continue
            mech_seen.add(mkey)
            fam = c.get("canonical_outcome") or ""
            endpoint = FAMILY_LABEL.get(fam, c.get("measurement") or "the outcome")
            d = c.get("canonical_direction") or c.get("direction") or "decrease"
            dverb = {"increase": "increases", "decrease": "reduces", "no_effect": "does not change"}.get(d, "changes")
            model = (c.get("claim_model") or "model").replace("_", " ")
            required = me.get("level") == "M3"
            hyps.append({
                "statement": f"{interv[0].upper() + interv[1:]} {dverb} {endpoint} through {med} "
                             f"({model}; {me['level']}: {me['kind']}; C{c['id']}).",
                "mechanism": f"{interv} -> {med} -> {endpoint}. Verified: {me['kind']} (C{c['id']}, {model}). "
                             + ("Not yet shown: that the agent changes " + med + " in other models or in people."
                                if required else
                                f"Not yet shown: that {med} is required for the effect (no perturbation of {med} was verified)."),
                "kind": "mechanistic",
                "predictions": ([f"Knocking down / inhibiting {med} in an independent model blunts the effect of {interv} on {endpoint}.",
                                 f"{interv[0].upper() + interv[1:]} changes {med} activity in vivo at doses reached in patients."]
                                if required else
                                [f"Knocking down / inhibiting {med} blunts the effect of {interv} on {endpoint}.",
                                 f"Restoring {med} (over-expression) rescues {endpoint} in {interv}-treated cells."]),
                "falsification_conditions": [f"{interv[0].upper() + interv[1:]} still {dverb} {endpoint} to the same extent when {med} is knocked down "
                                             f"or inhibited in an adequately replicated experiment."],
                "next_experiment": {"experiment_title": f"{med} dependence of the {interv} effect on {endpoint}",
                                    "model": model, "independent_variable": f"{interv} x {med} knockdown (2 x 2 design)",
                                    "dependent_variable": endpoint, "controls": f"non-targeting knockdown; vehicle; {med} rescue"},
                "derived_from": [f"C{c['id']}"],
            })
            if len(mech_seen) >= 2:
                break
        return {"hypotheses": hyps, "research_gaps": gaps}

    # -------------------------------------------------------------------------
    # Quality Auditing & Empirical Evaluation
    # -------------------------------------------------------------------------

    def audit_findings(self, research_question_id: int) -> List[Dict[str, Any]]:
        claims = self.db.verified_claims(research_question_id)
        flags = []
        with self.db._lock:   # claim-level flags are recomputed per run (no duplicates across runs)
            self.db._conn.execute("DELETE FROM audit_flags WHERE research_question_id IS ? AND entity_type='claim'",
                                  (research_question_id,))
            self.db._conn.commit()
        # Quarantined claims never reach the evidence; they are listed here with the exact reason.
        for q in self.db.quarantined_claims(research_question_id):
            reasons = (q.get("semantic_audit") or {}).get("reasons") or ["failed verification"]
            fid = self.db.add_audit_flag(
                research_question_id=research_question_id,
                entity_type="claim",
                entity_id=q["id"],
                claim_text=q.get("statement", ""),
                evidence_reference=f"paper:{q.get('paper_id')}",
                issue="QUARANTINED - " + "; ".join(reasons)[:600],
                severity="HIGH",
            )
            flags.append({"flag_id": fid, "claim_id": q["id"], "severity": "HIGH",
                          "issue": f"Claim C{q['id']} quarantined - " + "; ".join(reasons)[:300], "reasons": reasons})

        for c in claims:
            if c.get("effect_direction") == "no_effect" and c.get("sample_size") and c["sample_size"] < 10:
                fid = self.db.add_audit_flag(
                    research_question_id=research_question_id,
                    entity_type="claim",
                    entity_id=c["id"],
                    claim_text=c.get("statement", ""),
                    evidence_reference=f"paper:{c.get('paper_id')}",
                    issue=f"Reported 'no_effect' with small sample size (n={c['sample_size']}). Risk of Type II error.",
                    severity="MEDIUM",
                )
                flags.append({"flag_id": fid, "claim_id": c["id"], "severity": "MEDIUM", "issue": "underpowered_null"})

            if c.get("nonsignificant_direction"):
                fid = self.db.add_audit_flag(
                    research_question_id=research_question_id,
                    entity_type="claim",
                    entity_id=c["id"],
                    claim_text=c.get("statement", ""),
                    evidence_reference=f"paper:{c.get('paper_id')}",
                    issue="Directional biological claim reported without p < 0.05 statistical significance.",
                    severity="HIGH",
                )
                flags.append({"flag_id": fid, "claim_id": c["id"], "severity": "HIGH", "issue": "nonsignificant_trend"})


        return flags

    def evaluate_hypothesis_with_data(
        self,
        research_question_id: int,
        dataset_path: str,
        group_column: str,
        outcome_column: str,
    ) -> Optional[Dict[str, Any]]:
        hypotheses = self.db.list_hypotheses(research_question_id)
        if not hypotheses:
            return None
        # test the hypothesis that is about the uploaded outcome / treatment (not blindly the first one)
        from semantic_audit import canonical_outcome
        fam = canonical_outcome(outcome_column.replace("_", " "))["family"]
        words = set(re.findall(r"[a-z][a-z0-9\-]{2,}", f"{group_column} {outcome_column}".lower().replace("_", " ")))

        def fit(h):
            txt = f"{h.get('statement') or ''} {h.get('mechanism') or ''}".lower()
            return (canonical_outcome(txt)["family"] == fam) * 2 + sum(1 for w in words if w in txt)
        target_h = max(hypotheses, key=fit)
        original_status = target_h.get("status") or "proposed"
        eval_res = self.evaluator.evaluate_hypothesis_with_data(
            hypothesis_id=target_h["id"],
            dataset_path=dataset_path,
            group_column=group_column,
            outcome_column=outcome_column,
        )
        # SEGREGATION: the user's data is reported in its own section; it never changes the literature-based
        # status / ranking of a hypothesis
        self.db.update_hypothesis_status(target_h["id"], original_status)
        eval_res = {**eval_res, "hypothesis_id": target_h["id"], "hypothesis": target_h.get("statement"),
                    "match": "matches the uploaded outcome" if fit(target_h) else "no hypothesis mentions the uploaded "
                    "outcome - tested against the top-ranked one"}
        print(f"\n[Your data vs hypothesis] Hypothesis #{target_h['id']} -> {eval_res['new_status'].upper()} on your data")
        print(f"  -> {eval_res['verdict']}")
        return eval_res

    # -------------------------------------------------------------------------
    # PRISMA acquisition, protein interactions, evidence matrix & deep reasoning
    # -------------------------------------------------------------------------

    def plan_queries(self, question: str) -> List[str]:
        # concept queries built from the question (disease + spelling variants + subtypes x what is asked); the raw
        # sentence is NOT sent to keyword databases (PubMed returns nothing for 'Can leukemia be cured?', fuzzy sources
        # return word matches) - it is still used for OpenAlex's semantic search in the PRISMA identification step
        from research_planner import build_keyword_queries
        queries = build_keyword_queries(question) or [question]
        for q in getattr(self, "_extra_queries", None) or []:      # added by the auto-healer for a repair round
            if q and q.strip().lower() not in {x.strip().lower() for x in queries}:
                queries.append(q)
        print(f"  [Planner] Concept queries: {' | '.join(queries)}")
        if self.router and not getattr(self, "_quick", False):       # quick mode: concept queries only (no AI planner call)
            try:
                plan = generate_model_plan(self.router, question, provider="groq")
                if plan and plan.get("search_queries"):
                    have = {q.strip().lower() for q in queries}
                    queries = queries[:3] + [q for q in plan["search_queries"][:4] if q.strip().lower() not in have
                                             and q.strip().lower() != question.strip().lower()] + queries[3:]
            except Exception as exc:
                print(f"  [Planner notice] AI planner unavailable, concept queries only: {exc}")
        # endpoint -> measurable biomarker expansion (e.g. proliferation -> Ki-67, PCNA);
        # deterministic, so it also runs without an AI key
        extra = endpoint_expansion_queries(question, existing=queries)
        if extra:
            print(f"  [Planner] Biomarker expansion: {' | '.join(extra)}")
            queries += extra
        return queries

    def acquire_literature_prisma(
        self,
        question: str,
        research_question_id: int,
        screen_n: int = 150,
        deep_read_k: int = 25,
        use_unpaywall: bool = True,
        sources: Optional[List[str]] = None,
        citation_chasing: bool = True,
        abstract_only: bool = False,
    ) -> Dict[str, Any]:
        queries = self.plan_queries(question)
        screener = PrismaScreener(
            source_manager=self.source_manager,
            vector_store=self.vector_store,
            unpaywall=self.unpaywall if use_unpaywall else None,
            sources=sources,
            citation_chasing=citation_chasing,
            abstract_only=abstract_only,
            user_collections=getattr(self, "user_collections", None),
        )
        if use_unpaywall and not self.unpaywall.enabled:
            print("  [Unpaywall] No valid contact email supplied - OA PDF resolution skipped.")
        result = screener.run(self.db, queries, question, research_question_id,
                              screen_n=screen_n, deep_k=deep_read_k)
        result["queries"] = queries
        return result

    def collect_gene_symbols(self, research_question_id: int, paper_ids: List[int], question: str) -> List[str]:
        """Genes named in the question or in VERIFIED claims of the included papers only.
        (Titles/abstracts are no longer mined: that produced STRING networks unrelated to the evidence.)"""
        from claim_extraction import abbreviations as _abbr
        from study_classifier import get_or_classify
        # primary-study claims that passed the relevance gate; reviews only restate other work
        claims = [c for c in self.db.verified_claims(research_question_id) if c.get("paper_id") in set(paper_ids or [])
                  and get_or_classify(self.db, c["paper_id"]).get("evidence_type") != "review_meta"]
        # abbreviations a paper defines for a NON-gene term are never genes: 'metformin (MET)' is not the MET receptor,
        # 'tamoxifen (TAM)', 'primary breast cancer cells (PBCCs)'
        not_genes = set()
        for pid in {c["paper_id"] for c in claims}:
            p = self.db.get_paper(pid) or {}
            secs = " ".join(x["text"] for x in self.db.get_sections(pid))
            for a, full in _abbr(f"{p.get('abstract') or ''} {secs}").items():
                if full.split()[0].islower() or len(full.split()) >= 3 and not re.search(r"\b(?:protein|kinase|receptor|factor|gene)\b", full, re.I):
                    not_genes.add(a.upper())
        interv = {str(c.get("intervention") or "").upper() for c in claims}
        # standard drug abbreviations of the agents in the claims ('MET' / 'MF' = metformin) are never genes, even in a
        # paper that uses the abbreviation without defining it ('MET-R cells', 'MET-treated')
        from clinical_context import _DRUG_ABBR
        agents_txt = " ".join(interv).lower() + " " + (question or "").lower()
        not_genes |= {k.upper() for k, v in _DRUG_ABBR.items() if any(w in agents_txt for w in v.split() if len(w) >= 5)}
        text = " ".join(f"{c.get('measurement') or ''} {c.get('statement') or ''} {c.get('quote') or ''}" for c in claims)
        text = drop_non_genes(text, not_genes | interv)
        symbols: List[str] = [s for s in symbols_from_text(question, limit=5)]
        if paper_ids:
            marks = ",".join("?" for _ in paper_ids)
            rows = self.db._conn.execute(
                f"""SELECT be.symbol, COUNT(*) AS n FROM entity_mentions em
                    JOIN biological_entities be ON be.id = em.entity_id
                    WHERE em.source_type='paper' AND em.source_id IN ({marks}) AND be.entity_type='gene'
                    GROUP BY be.symbol ORDER BY n DESC LIMIT 25""", tuple(paper_ids)).fetchall()
            for r in rows:   # NCBI-linked genes count only if a verified claim mentions them
                if r["symbol"] and re.search(rf"\b{re.escape(r['symbol'])}\b", text) and r["symbol"] not in symbols:
                    symbols.append(r["symbol"])
        for s in symbols_from_text(text, limit=12):
            if s not in symbols:
                symbols.append(s)
        self._evidence_genes = set(symbols_from_text(text, limit=30))
        return symbols[:15]

    @staticmethod
    def choose_species(question: str, matrix: Optional[Dict[str, Any]] = None) -> int:
        """Organism for STRING: named in the question first, else the most common organism
        among included whole-organism studies, else human (works for plants and microbes too)."""
        named = find_species(question)
        if named:
            return named
        rows = (matrix or {}).get("rows", [])
        if any(r.get("bucket") == "human" for r in rows):
            return 9606
        votes: Dict[int, int] = {}
        for r in rows:
            if r.get("bucket") in ("animal", "plant", "microbial", "in_vitro"):
                tid = find_species(f"{r.get('model') or ''} {r.get('title') or ''}")
                if tid:
                    votes[tid] = votes.get(tid, 0) + 1
        return max(votes, key=votes.get) if votes else 9606

    def protein_interactions(self, research_question_id: int, paper_ids: List[int], question: str,
                             matrix: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        symbols = self.collect_gene_symbols(research_question_id, paper_ids, question)
        species = self.choose_species(question, matrix)
        levels = {"protein_interaction": "what STRING reports: physical/functional association between proteins",
                  "disease_association": "NOT assessed by STRING",
                  "causal_mechanism": "requires perturbation evidence in a verified study (see mechanism levels M3/M4)",
                  "therapeutic_target": "NOT established by any of this"}
        if not symbols:
            print("  -> No gene/protein named in the question or in verified claims - STRING not queried.")
            return {"species": species, "query_genes": [], "unmapped": [], "mode": "none", "nodes": [], "edges": [], "link": None,
                    "evidence_link": None, "evidence_link_note": "no gene/protein named in the question or in verified evidence - not queried",
                    "interpretation_levels": levels}
        print(f"  -> Querying STRING-db (taxon {species}) for: {', '.join(symbols)}")
        net = self.string_db.build_for_genes(symbols, species=species)
        ev_genes = getattr(self, "_evidence_genes", set())
        linked = [e for e in net.get("edges", []) if e.get("a") in ev_genes or e.get("b") in ev_genes]
        net["evidence_link"] = "linked" if linked else "background"
        net["evidence_link_note"] = (f"{len(linked)} network association(s) connect proteins appearing in verified evidence. "
                                     f"These associations do not establish that the proteins participate in the mechanism "
                                     f"of the observed effect"
                                     if linked else "no interaction involves a protein named in the verified evidence - background biology only")
        net["interpretation_levels"] = levels
        print(f"  -> {len(net['edges'])} physical interactions across {len(net['nodes'])} proteins ({net['evidence_link']}).")
        return net

    # -------------------------------------------------------------------------
    # Autonomous ReAct Agent Entrypoint
    # -------------------------------------------------------------------------

    def run_agent(self, question: str, max_steps: Optional[int] = None) -> Dict[str, Any]:
        """The agent spends a budget by objective (tokens, tool calls, time, papers, turns) and stops when the question's
        objective is met; max_steps is only an optional extra ceiling on reasoning turns."""
        """Run an autonomous, dynamic tool-calling ReAct agent scoped to the question's queue."""
        if not self.router:
            raise RuntimeError("Autonomous agent requires an active AI router.")
        from evidence_identity import normalize_question
        question = normalize_question(question)["question"]
        rq_id = self.db.create_research_question(question)
        agent = AutonomousResearchAgent(
            db=self.db,
            router=self.router,
            max_steps=max_steps,
            research_question_id=rq_id,
        )
        return agent.run(question)

    # -------------------------------------------------------------------------
    # Batch Pipeline Investigation Entrypoint
    # -------------------------------------------------------------------------

    def investigate(self, question: str, *args, auto_heal: bool = True, **kwargs) -> Dict[str, Any]:
        """One investigation + the AUTO-HEALER: the finished run is checked for known failure patterns
        (report_auditor); repairable ones trigger a repair round (broader concept queries, larger budget,
        forced re-extraction), the best round is kept, in-place repairs are applied, and anything left is
        reported in the dossier and written to BUG_REPORT.md. The program's own code is never modified."""
        if getattr(self, "router", None) is None and not getattr(self, "allow_without_ai", True):
            raise NoAIKeyError(NO_KEY_MESSAGE)
        import report_auditor as ra
        budgets = {"screen_n": kwargs.get("screen_n", 150), "deep_read_k": kwargs.get("deep_read_k", 25),
                   "max_papers": kwargs.get("max_papers", 6)}
        from core_utils import output_path
        output_report = str(output_path(kwargs.get("output_report", "RESEARCH_REPORT.md")))
        kwargs["output_report"] = output_report
        history, rounds = [], []
        self._extra_queries, self._force_reextract = [], False
        round_no = 0
        while True:
            kwargs.update(budgets)
            dossier = self._investigate_once(question, *args, **kwargs)
            try:
                report_text = Path(output_report).read_text(encoding="utf-8")
            except OSError:
                report_text = ""
            q_used = dossier.get("research_question") or question
            issues = ra.audit_run(dossier, report_text, q_used, budgets)
            rounds.append((dossier, report_text, issues))
            plan = ra.repair_plan(issues, budgets, q_used, round_no, kwargs.get("use_prisma", False)) if auto_heal else None
            history.append({"round": round_no, "issues": [{k: i[k] for k in ("code", "severity", "detail")} for i in issues],
                            "repair": (f"re-run with screen {plan['screen_n']}, deep-read {plan['deep_read_k']}"
                                       + (f", {len(plan['extra_queries'])} extra concept queries" if plan["extra_queries"] else "")
                                       + (", forced re-extraction" if plan["force_reextract"] else "")) if plan else None,
                            "quality": list(ra.run_quality(dossier))})
            if not plan:
                break
            print(f"\n[Auto-heal] Round {round_no} problems: {', '.join(i['code'] for i in issues)} -> {history[-1]['repair']}")
            if len(rounds) >= 2 and ra.run_quality(rounds[-1][0]) <= ra.run_quality(rounds[-2][0]):
                history[-1]["repair"] = None
                history[-1]["note"] = "stopped: the previous repair round did not add verified evidence"
                break                     # the last repair did not improve the evidence - stop
            budgets = {"screen_n": plan["screen_n"], "deep_read_k": plan["deep_read_k"], "max_papers": plan["max_papers"]}
            self._extra_queries, self._force_reextract = plan["extra_queries"], plan["force_reextract"]
            codes = ", ".join(sorted({i["code"] for i in issues}))
            note = (f"repair round - the automatic report check found a fixable problem ({codes}), so the run is "
                    f"repeated: {history[-1]['repair']}. The better of the two runs is kept.")
            print(f"[Pass] {note}")
            hook = getattr(self, "progress", None)
            if hook:                          # announced on the results page (stage 'pass'), never a silent reset
                try:
                    hook("pass", note)
                except Exception:
                    pass
            round_no += 1
        self._extra_queries, self._force_reextract = [], False
        best_i = max(range(len(rounds)), key=lambda i: ra.run_quality(rounds[i][0]))
        dossier, report_text, issues = rounds[best_i]
        if getattr(self, "_hypothesis_tests", None):  # v27: laboratory tests of the hypotheses (reader report)
            dossier["hypothesis_tests"] = [{k: v for k, v in t.items() if k != "views"} for t in self._hypothesis_tests]
            dossier["hypothesis_test_views"] = [{"statement": t["statement"], "views": t.get("views")} for t in self._hypothesis_tests]
        if getattr(self, "_auto_history", None):     # v26.2 Auto: the in-run widening decisions (report + dossier)
            dossier["tiers"] = list(self._auto_history)
            dossier["analysis_tier"] = getattr(self, "_auto_level", None)
        inplace = ra.apply_inplace(dossier, issues)
        fixed_codes = {"Q6"} if inplace else set()
        remaining = [i for i in issues if i["code"] not in fixed_codes]
        # v22 additive layers: canonical evidence record + consistency (B), hypothesis-vs-discovery audit (C), gold score (A).
        # They only ADD keys / a report section; a failure inside them is recorded and never raises.
        v22_md = ""
        try:
            import v22_layers
            v22 = v22_layers.apply(dossier, report_text)
            remaining = remaining + [i for i in v22["issues"] if i.get("severity") in ("critical", "major")]
            v22_md = v22["markdown"]
        except Exception as exc:
            print(f"[v22] skipped: {type(exc).__name__}: {exc}")
        # v25 additive layers: guard statistics, calibrated verdict, RCT meta-analysis + publication bias, dependency
        # graph, cross-scale chain, knowledge graph, risk of bias, numerical audit, publication status, versioned
        # memory, reproducibility manifest, metrics, and the three labelled dossier sections (see v25_layers.py)
        v25_md = ""
        try:
            import v25_layers
            v25 = v25_layers.apply(dossier, report_text, engine=self)
            remaining = remaining + [i for i in v25["issues"] if i.get("severity") in ("critical", "major")]
            report_text = v25["report_text"] or report_text
            v25_md = v25["markdown"]
        except Exception as exc:
            print(f"[v25] skipped: {type(exc).__name__}: {exc}")
        # v26 layers: local library, quality profiles, causal checks, plausibility, absolute / relative risk, calibrated
        # confidence, subgroups, registered-but-missing studies, value of more research, adversarial review
        v26_md = ""
        try:
            import v26_layers
            v26 = v26_layers.apply(dossier, report_text, engine=self)
            remaining = remaining + v26["issues"]
            report_text = v26["report_text"] or report_text
            v26_md = v26["markdown"]
        except Exception as exc:
            print(f"[v26] skipped: {type(exc).__name__}: {exc}")
        heal = {"enabled": auto_heal, "rounds": len(rounds), "kept_round": best_i if len(rounds) > 1 else None,
                "history": history, "inplace": inplace, "remaining": remaining}
        bug = ra.write_bug_report(str(Path(output_report).with_name("BUG_REPORT.md")), question, heal)
        heal["bug_report"] = bug
        dossier["auto_heal"] = heal
        # v27: the READER report (answer, explanation, key studies, certainty, sources - plain language) comes first;
        # everything technical (screening, audits, quotes, run log, provider messages) becomes an optional appendix
        technical = report_text + ra.heal_markdown(heal) + v22_md + v25_md + v26_md
        try:
            import reader_report
            reader = reader_report.build(dossier, router=getattr(self, "router", None))
            dossier["reader_report"] = reader
            report_text_final = (reader + "\n\n<!-- technical-appendix -->\n\n# Technical appendix (optional reading)\n\n"
                                 "How the studies were found and screened, every verified quote, the automatic quality "
                                 "checks and the run log.\n\n" + technical)
        except Exception as exc:
            print(f"[Reader report] skipped: {type(exc).__name__}: {exc}")
            report_text_final = technical
        try:
            Path(output_report).write_text(report_text_final, encoding="utf-8")
            output_path(f"research_dossier_{dossier.get('research_question_id')}.json").write_text(
                json.dumps(dossier, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
        except OSError:
            pass
        if remaining:
            print(f"[Auto-heal] {len(remaining)} problem(s) not repaired automatically -> {bug}")
        return dossier

    def _test_hypotheses(self, synthesis: Dict[str, Any], exec_synthesis: Any, question: str) -> None:
        import ai_lab
        from reader_report import _strip_codes
        hyps = synthesis.get("hypotheses") or []
        user = getattr(self, "_user_hypothesis", None)
        items = ([{"statement": user, "_user": True}] if user else []) + [h for h in hyps[:5] if h.get("statement")]
        if not items:
            return
        if self.progress:
            self.progress("verify", f"Testing {len(items)} hypothesis(es) in the virtual laboratory before presenting them")
        rank = {"supported by the simulation": 0, "supported, but depends on assumptions": 1,
                "explored (no testable prediction)": 2, "not testable in the lab": 3, "ambiguous claim": 3,
                "contradicted, but depends on assumptions": 4, "contradicted by the simulation": 5}
        tests = []
        for n, h in enumerate(items):
            claim = _strip_codes(h["statement"])
            ctx = " ".join(str(h.get(k) or "") for k in ("mechanism", "next_experiment"))[:800]
            r = ai_lab.investigate(claim, router=self.router if n < 3 else None, context=f"Research question: {question}. {ctx}",
                                   time_budget_s=45)
            how = "; ".join(f"{x['tool']} ({x.get('role')}): {(x.get('result') or {}).get('text', 'run')}"
                            for x in r.get("experiments") or [] if not x.get("error"))[:300]
            test = {"statement": claim, "verdict": r["verdict"], "verdict_plain": r["verdict_plain"], "how": how,
                    "user_hypothesis": bool(h.get("_user")), "experiments": r.get("experiments"),
                    "views": r.get("views")[:6], "planner": r.get("planner")}
            tests.append(test)
            print(f"  -> Laboratory: {r['verdict']} - {claim[:90]}")
            if not h.get("_user"):
                h["lab_test"] = {k: test[k] for k in ("verdict", "verdict_plain", "how", "planner")}
                # confidence = verified literature support + the laboratory test
                sup = (h.get("support") or {}).get("score") or 0
                lab = rank.get(r["verdict"], 3)
                level = ("moderate" if sup and lab <= 1 else "low" if lab <= 3 or sup else "very low")
                if lab >= 4:
                    level = "very low"
                h["confidence"] = {"level": level, "reason":
                                   f"{'verified studies support it' if sup else 'no verified study supports it yet'}; "
                                   f"laboratory: {r['verdict_plain'].lower()}"}
                h["_lab_rank"] = lab
        hyps.sort(key=lambda h: (h.get("_lab_rank", 3), -((h.get("support") or {}).get("score") or 0)))
        for h in hyps:
            h.pop("_lab_rank", None)
        if isinstance(exec_synthesis, dict):
            exec_synthesis["hypothesis_tests"] = tests
        self._hypothesis_tests = tests

    def _widen_in_run(self, question: str, rq_id: int, paper_ids: List[int], prisma: Optional[Dict[str, Any]],
                      use_unpaywall: bool, sources: Optional[List[str]], link_bio: bool):
        """v26.2 Auto mode, ONE continuous run: after the papers are read, the evidence is checked (the same rules as
        the tier escalation); if it is thin or conflicting, the search is widened to the next level IN THIS RUN - more
        abstracts screened, more papers read in full, citations chased - and only the new or newly full-text papers
        are read. Nothing already done is repeated and the later stages (contradictions, synthesis, report) run once.
        The decisions are stored in self._auto_history (-> dossier['tiers'])."""
        import analysis_tiers as AT
        import verdict_calibration as vc
        level = getattr(self, "_auto_level", "quick")
        hist = self.__dict__.setdefault("_auto_history", [])
        t0 = time.time()
        while self._auto_levels:
            try:
                interim = build_matrix(self.db, rq_id, paper_ids, fulltext_sources=(prisma or {}).get("text_basis"),
                                       question=question, screening=(prisma or {}).get("screening_table"))
                verdict = vc.from_matrix(interim, question)
                comp = None
                tot = self.router.health.totals() if self.router is not None and hasattr(self.router, "health") else {}
                if tot.get("requests"):
                    comp = {"complete": tot.get("requests_failed", 0) / tot["requests"] < 0.05,
                            "success_rate": round(1 - tot.get("requests_failed", 0) / tot["requests"], 3)}
                dec = AT.escalate({"research_question": question, "evidence_matrix": interim,
                                   "calibrated_verdict": verdict, "completeness": comp}, level)
            except Exception as exc:
                print(f"  [Auto] evidence check skipped ({type(exc).__name__}: {exc}) - keeping the {level} level")
                break
            hist.append({"tier": level, "seconds": round(time.time() - t0, 1), "decision": dec})
            if not dec["escalate"]:
                print(f"  [Auto] staying at the {level} level: " + "; ".join(dec["reasons"] or ["the evidence is sufficient"]))
                break
            nxt = self._auto_levels.pop(0)
            p = AT.PARAMS[nxt]
            why = "; ".join(str(r) for r in dec["reasons"][:2])
            msg = (f"Widening the search in this run: {level} -> {nxt} level ({p['screen_n']} abstracts screened, "
                   f"{p['deep_read_k']} read in full), because {why}. Papers already read are kept - nothing restarts.")
            print(f"  [Auto] {msg}")
            if self.progress:
                self.progress("widen", msg)
            self._quick = bool(p.get("quick"))
            t0 = time.time()
            more = self.acquire_literature_prisma(question=question, research_question_id=rq_id,
                                                  screen_n=p["screen_n"], deep_read_k=p["deep_read_k"],
                                                  use_unpaywall=use_unpaywall, sources=sources,
                                                  citation_chasing=p["citation_chasing"],
                                                  abstract_only=p["abstract_only"])
            new_ids = [i for i in more.get("paper_ids") or [] if i not in paper_ids]
            # the earlier papers are passed again ONLY so that a paper first read from its abstract is now read in
            # full (its text changed, so the cache misses); everything already read with the same text is a cache hit
            paper_ids = list(paper_ids) + new_ids
            self.db.set_paper_scope(paper_ids)
            print(f"  [Auto] {len(new_ids)} new paper(s) to read at the {nxt} level")
            self.process_papers(paper_ids, rq_id, question, link_bio=link_bio)
            adopt_existing_claims(self.db, paper_ids, rq_id)
            merged = dict(more)
            tb = dict((prisma or {}).get("text_basis") or {})
            tb.update(more.get("text_basis") or {})
            merged["text_basis"] = tb
            merged["paper_ids"] = paper_ids
            prisma = merged
            level = nxt
        self._auto_level = level
        return paper_ids, prisma

    def _investigate_once(
        self,
        question: str,
        experimental_dataset: Optional[str] = None,
        exp_group_col: Optional[str] = None,
        exp_outcome_col: Optional[str] = None,
        max_papers: int = 6,
        min_relevance: float = 0.30,
        enable_geo: bool = True,
        output_report: str = "RESEARCH_REPORT.md",
        use_prisma: bool = False,
        screen_n: int = 150,
        deep_read_k: int = 25,
        use_unpaywall: bool = True,
        use_string: bool = True,
        use_deep_reasoning: bool = True,
        enhanced_report: Optional[bool] = None,
        link_bio: bool = True,
        sources: Optional[List[str]] = None,
        citation_chasing: bool = True,
        use_structures: bool = True,
        abstract_only: bool = False,
        quick: bool = False,
        user_collections: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        from core_utils import output_path
        self.user_collections = list(user_collections or [])   # v26: the user's own papers, only when asked for
        from evidence_identity import normalize_question
        output_report = str(output_path(output_report))
        self._quick = bool(quick)
        self._timings, self._cur_stage = [], None
        norm = normalize_question(question)          # spelling fixed BEFORE searching ('leukimia' -> 'leukemia')
        question = norm["question"]
        if not os.getenv("BIOSYN_OFFLINE"):
            from evidence_identity import online_spelling
            online = online_spelling(question)       # PubMed ESpell for words outside the local vocabulary
            if online["corrections"]:
                question = online["question"]
                norm = {**norm, "question": question, "corrections": norm["corrections"] + online["corrections"],
                        "note": "; ".join(x for x in (norm.get("note"), online["note"]) if x)}
        print("=" * 80)
        print("BIOLOGY AI RESEARCH ENGINE - CLOSED-LOOP INVESTIGATION")
        print(f"Question: {question}")
        assign = self.router.assignments() if self.router is not None and hasattr(self.router, "assignments") else {}
        if assign:
            print("  -> AI task assignment: " + "; ".join(f"{k}: {v}" for k, v in assign.items()))
        if norm["corrections"]:
            print(f"  -> {norm['note']}")
        print("=" * 80)

        rq_id = self.db.create_research_question(question)
        enhanced = use_prisma if enhanced_report is None else enhanced_report
        prisma = None

        if use_prisma:
            self._stage("screening", f"PRISMA funnel - screen {screen_n} abstracts, deep-read top {deep_read_k}")
            prisma = self.acquire_literature_prisma(
                question=question,
                research_question_id=rq_id,
                screen_n=screen_n,
                deep_read_k=deep_read_k,
                use_unpaywall=use_unpaywall,
                sources=sources,
                citation_chasing=citation_chasing,
                abstract_only=abstract_only,
            )
            paper_ids = prisma["paper_ids"]
        else:
            self._stage("retrieval", "Multi-source literature acquisition")
            paper_ids = self.acquire_literature(
                question=question,
                research_question_id=rq_id,
                max_papers=max_papers,
                min_relevance=min_relevance,
            )

        if not paper_ids:
            paper_ids = [p["id"] for p in self.db.list_papers(rq_id)]
        # evidence for THIS run = this run's included papers only (the question may have been asked before)
        self.db.set_paper_scope(paper_ids)

        self._stage("extraction", f"Extracting design, statistics and quote-grounded claims from {len(paper_ids)} papers")
        print(f"\nProcessing {len(paper_ids)} papers in active knowledge base...")
        self.process_papers(paper_ids, rq_id, question, link_bio=link_bio)
        if use_prisma:
            adopted = adopt_existing_claims(self.db, paper_ids, rq_id)
            if adopted:
                print(f"  -> Re-used {adopted} verified claims already extracted for these papers in earlier runs.")
        if use_prisma and getattr(self, "_auto_levels", None):
            # v26.2 Auto: widen WITHIN this run (no restart) while the evidence is thin or conflicting
            paper_ids, prisma = self._widen_in_run(question, rq_id, paper_ids, prisma, use_unpaywall, sources,
                                                   link_bio)

        bridge_res = None
        if experimental_dataset and exp_group_col and exp_outcome_col:
            print("\n" + "=" * 80)
            print("CONNECTING USER EXPERIMENTAL DATA TO LITERATURE")
            print("=" * 80)
            bridge_res = self.bridge.compare_experiment_to_literature(
                research_question_id=rq_id,
                dataset_path=experimental_dataset,
                group_col=exp_group_col,
                outcome_col=exp_outcome_col,
            )
            print(f"  -> {bridge_res['statement']}")

        self._stage("contradictions", "Cross-study contradiction detection")
        contradictions = self.detect_contradictions(rq_id)

        all_claims = self.db.verified_claims(rq_id)
        conflict_targets = extract_conflicts_for_geo(contradictions, all_claims)

        geo_results = []
        if enable_geo and conflict_targets:
            geo_results = self.run_empirical_closure(rq_id, conflict_targets)

        self._stage("pathways", "Constructing multi-hop mechanistic pathways")
        pathway_results = self.synthesize_mechanistic_pathways(rq_id)

        self._stage("hypotheses", "Formulating hypotheses & experimental designs")
        synthesis = self.formulate_hypotheses_and_gaps(rq_id, question)
        print(f"  -> Generated {len(synthesis['hypotheses'])} falsifiable hypotheses.")
        print(f"  -> Identified {len(synthesis['research_gaps'])} unaddressed research gaps.")

        falsification_res = None
        if experimental_dataset and exp_group_col and exp_outcome_col:
            falsification_res = self.evaluate_hypothesis_with_data(
                research_question_id=rq_id,
                dataset_path=experimental_dataset,
                group_column=exp_group_col,
                outcome_column=exp_outcome_col,
            )

        audit_flags = self.audit_findings(rq_id)
        print(f"  -> Logged {len(audit_flags)} deterministic audit flags.")

        original_report = self.reporter.generate(rq_id, output_path=output_report)

        matrix, string_net, exec_synthesis, matrix_csv_text = None, None, None, None
        structures: List[Dict[str, Any]] = []
        if enhanced:
            self._stage("evidence_matrix", "Sorting evidence into Human / Animal / In vitro buckets")
            matrix = build_matrix(self.db, rq_id, paper_ids, fulltext_sources=(prisma or {}).get("text_basis"),
                                  question=question, screening=(prisma or {}).get("screening_table"))
            if norm.get("corrections"):
                matrix["question_normalisation"] = norm      # dossier + website show 'as typed' and the correction
            matrix_csv_text = matrix_csv(matrix)
            print(f"  -> Evidence types: {matrix['type_counts']} | levels: {matrix['level_counts']}")
            t = matrix["totals"]
            print(f"  -> {t['papers']} papers, {t['experiments']} experiments, {t['independent_findings']} independent "
                  f"findings from {t['verified_claims']} verified claims ({t['quarantined_claims']} quarantined)")
            synthesis["hypotheses"] = self.consolidate_hypotheses(rq_id, question, matrix) or synthesis["hypotheses"]

            if use_string and not structures_relevant(question):
                # an OUTCOME question ('does X reduce Y in humans?'): a protein network cannot change the answer and
                # in practice only ever produced a 'background biology' appendix - skip it (saves its web requests)
                print("  [STRING] skipped - the question asks about an outcome, not a protein mechanism "
                      "(name a protein or ask 'how / through what' to include protein interactions)")
                string_net = {"query_genes": [], "edges": [], "skipped": True, "evidence_link": "not_run",
                              "evidence_link_note": "Not run: the question asks about an outcome, not a protein "
                                                    "mechanism, so a protein network could not change the answer."}
            elif use_string:
                self._stage("protein_interactions", "STRING-db physical protein-protein interactions")
                try:
                    string_net = self.protein_interactions(rq_id, paper_ids, question, matrix)
                except Exception as exc:
                    print(f"  [STRING notice] {exc}")

            # proteins modelled in 3D: those named in VERIFIED claims first, then those that also have STRING links;
            # each model says why it is shown (a structure is context, never evidence)
            _qg = [g for g in ((string_net or {}).get("query_genes") or [])]
            _in_edges = {x for e in ((string_net or {}).get("edges") or []) for x in (e.get("a"), e.get("b"))}
            _vc = self.db.primary_claims(rq_id)
            _named = {g: [c["id"] for c in _vc if re.search(rf"(?<![A-Za-z0-9]){re.escape(g)}(?![A-Za-z0-9])",
                                                            f"{c.get('quote') or ''} {c.get('statement') or ''}")]
                      for g in _qg}
            genes = sorted(_qg, key=lambda g: (not _named.get(g), g not in _in_edges))[:3]
            # 3D models answer mechanism questions; for a question about a clinical / biological OUTCOME ('does X reduce
            # proliferation in humans?') they cannot change the answer, so they are skipped (and the log says why)
            mech_q = structures_relevant(question)
            if use_structures and genes and not mech_q:
                print("  -> 3D structures skipped: the question asks about an outcome, not a protein mechanism "
                      "(ask a 'how / through which protein' question to get structures).")
            if use_structures and genes and mech_q and not os.getenv("BIOSYN_OFFLINE"):
                self._stage("structures", "3D structures of the proteins named in the evidence (AlphaFold DB)")
                try:
                    from structure import structures_for_genes
                    taxon = int((string_net or {}).get("species") or 9606)
                    structures = structures_for_genes(genes, taxon=taxon, limit=3)
                    for st in structures:
                        ids_ = _named.get(st["gene"]) or []
                        st["why"] = ((f"named in verified claim(s) {', '.join('C' + str(i) for i in ids_[:4])}" if ids_ else
                                      "named in an included paper (no verified claim names it)")
                                     + ("; has STRING links in this run" if st["gene"] in _in_edges else ""))
                        print(f"  -> {st['gene']}: " + (f"AlphaFold model {st['id']}, mean pLDDT "
                              f"{st['analysis']['confidence'].get('mean_plddt')}, {len(st['top_pockets'])} pocket(s)"
                              if not st.get("error") else f"no model ({st['error'][:80]})"))
                except Exception as exc:
                    print(f"  [Structure notice] {exc}")

            self._stage("deep_reasoning", "Multi-step synthesis with a free open-weight reasoning model")
            reasoner = DeepReasoner(self.router if use_deep_reasoning else None)
            reasoner.reflect = not quick                  # quick mode: one synthesis call, no reflection round
            exec_synthesis = reasoner.synthesize(question, matrix, self.db.current_contradictions(rq_id), string_net, prisma=prisma)
            from reasoning_guards import build_traceability
            claims_by_id = {c["id"]: c for c in self.db.verified_claims(rq_id)}
            exec_synthesis["traceability"] = build_traceability(exec_synthesis, matrix, claims_by_id)
            try:      # Layer 1 of the website: one plain verdict + the level-by-level answer
                from enhanced_report import level_answer_obj
                _la = level_answer_obj(question, matrix)
                exec_synthesis["verdict"] = _la.get("verdict")
                exec_synthesis["level_answer"] = _la.get("text")
                exec_synthesis["level_discordance"] = _la.get("discordance")
                # ONE answer in the whole dossier: the detailed report's own tally (counted from raw claims, which
                # counts duplicate claims separately) is replaced by the matrix's, so the two can never disagree
                if _la.get("text") and isinstance(original_report, str):
                    from report_generator import ANSWER_MARK
                    i = original_report.find(ANSWER_MARK)
                    if i >= 0:
                        j = original_report.find("## Executive Summary", i)
                        if j > i:
                            original_report = (original_report[:i] + _la["text"] + "\n\n" + original_report[j:])
            except Exception as exc:
                print(f"  [Answer notice] {exc}")
            if not synthesis.get("research_gaps"):
                # never 'No research gaps' while the evidence itself (or the report's own key uncertainties) shows open
                # questions: gaps derived from the verified findings, each with its basis
                from reasoning_guards import evidence_gaps
                synthesis["research_gaps"] = evidence_gaps(matrix, exec_synthesis, question)
                if synthesis["research_gaps"] and isinstance(original_report, str):
                    n_g = len(synthesis["research_gaps"])
                    original_report = re.sub(r"- None recorded for this run[^\n]*",
                                             f"- {n_g} evidence-derived gap(s) - listed in the 'Research Gaps' section of the "
                                             f"dossier above.", original_report, count=1)
                    original_report = re.sub(r"Identified \*\*0 key unaddressed research gaps\*\*",
                                             f"Identified **{n_g} research gap(s)** (derived from the verified evidence and "
                                             f"the key uncertainties)", original_report, count=1)
            # a dose gap is shown next to what the report's own dose check found ('dose translation from mouse 250 mg/kg'
            # while that dose was assessed as within human exposure)
            _dose_rows = [r for r in (matrix.get("rows") or []) if (r.get("dose_relevance") or {}).get("category")
                          in ("physiological", "borderline", "supra-physiological")]
            for _g in synthesis.get("research_gaps") or []:
                if isinstance(_g, dict) and _dose_rows and re.search(r"\bdos(?:e|es|ing|age)\b", _g.get("description") or "", re.I):
                    _g["dose_note"] = ("The report's dose check already assessed: " + "; ".join(
                        f"{r['ref']}: {(r['dose_relevance'] or {}).get('category')}" for r in _dose_rows[:5])
                        + " (see 'Dose / Exposure Relevance').")
            # registered vs published primary outcome (needs the registry; skipped quietly when offline)
            try:
                from methodology_flags import outcome_switch_flags
                _sw = outcome_switch_flags(matrix.get("rows") or [])
                from methodology_flags import LAST_REGISTRY_RUN as _lrr
                exec_synthesis["registry_check"] = dict(_lrr)
                if _sw:
                    exec_synthesis["outcome_switching"] = _sw
                    print(f"  -> Outcome switching: {len(_sw)} trial(s) report a different primary outcome than registered.")
                elif _lrr.get("stopped"):
                    print(f"  -> Registered-outcome check stopped early ({_lrr['stopped']}) after "
                          f"{_lrr['looked']} lookup(s) in {_lrr['seconds']} s - reported as not checked.")
            except Exception as exc:
                print(f"  [Registry notice] {str(exc)[:120]}")
            _mf = [f for r in (matrix.get("rows") or []) for f in (r.get("method_flags") or [])]
            if _mf:
                exec_synthesis["method_flags"] = _mf
                _crit = [f for f in _mf if f.get("severity") == "critical"]
                print(f"  -> Methodological flags: {len(_mf)}"
                      + (f" ({len(_crit)} critical: " + ", ".join(sorted({f.get('line') or f.get('compound') or '?'
                                                                          for f in _crit})) + ")" if _crit else ""))
            from report_validator import validate_report
            exec_synthesis["report_validation"] = validate_report(
                exec_synthesis, matrix, prisma, string_net, synthesis["hypotheses"],
                contradictions=self.db.current_contradictions(rq_id), verified_ids=set(claims_by_id),
                gaps=synthesis["research_gaps"])
            print(f"  -> Pre-publication validation: {exec_synthesis['report_validation']['summary']}")
            print(f"  -> Synthesis via {exec_synthesis.get('provider') or 'deterministic'} "
                  f"({exec_synthesis.get('model') or 'no model'}); certainty={exec_synthesis.get('certainty')}")

            self._stage("report", "Composing evidence dossier")
            if isinstance(exec_synthesis, dict):
                exec_synthesis['ai_assignments'] = assign
                exec_synthesis['user_experiment'] = ({"dataset": Path(experimental_dataset).name, "analysis": bridge_res,
                                                       "hypothesis_test": falsification_res}
                                                      if experimental_dataset and exp_group_col and exp_outcome_col else None)
            try:            # VIRTUAL LAB: every hypothesis gets an in-silico consistency / testability check
                from virtual_lab import check_hypotheses
                lab = check_hypotheses(synthesis.get("hypotheses") or [], question)
                print(f"  -> Virtual lab checked {len(lab)} hypothesis(es): " + ", ".join(x["verdict"] for x in lab))
                if isinstance(exec_synthesis, dict):
                    exec_synthesis["virtual_lab"] = lab          # v25: kept for the dossier's simulation section (B)
            except Exception as exc:
                print(f"  [Virtual lab notice] {exc}")
            # v27: EVERY hypothesis is tested in BioSynthetica's own laboratory before it is presented (whatever the
            # prompt): experiments designed with predictions written down first, run and judged by the code, repeated
            # over plausible biology. Each hypothesis gets the result and a confidence level; supported ones come first.
            try:
                self._test_hypotheses(synthesis, exec_synthesis, question)
            except Exception as exc:
                print(f"  [Laboratory notice] hypothesis tests skipped: {type(exc).__name__}: {exc}")
            reg = (exec_synthesis.get("report_validation") or {}).get("registry") if isinstance(exec_synthesis, dict) else None
            if reg and isinstance(original_report, str):
                # ONE registry for every count: the detailed engine report quotes the same deep-read number as the
                # PRISMA table, the evidence matrix and the validator
                original_report = re.sub(
                    r"This autonomous investigation deep-read \*\*\d+ papers\*\* \([^)]*\)",
                    f"This autonomous investigation deep-read **{reg['deep_read']} papers** (study registry: "
                    f"{reg['statement']})", original_report, count=1)
            compose_enhanced_report(question, exec_synthesis, matrix, prisma, string_net,
                                    original_report, output_path=output_report,
                                    hypotheses=synthesis["hypotheses"], gaps=synthesis["research_gaps"],
                                    contradictions=self.db.current_contradictions(rq_id),
                                    pathways=pathway_results.get("mechanistic_chains", []),
                                    structures=structures)
            if quick:
                try:            # the report says plainly that it is a quick scan, not a full dossier
                    rp = Path(output_report)
                    txt = rp.read_text(encoding="utf-8")
                    banner = (f"> **Quick scan** - abstracts of the top {len(paper_ids)} papers only (no full texts, no automatic "
                              f"repair round, no 3D structures). Use a full run for a complete evidence dossier.\n\n")
                    head, sep, rest = txt.partition("\n")
                    rp.write_text(head + sep + "\n" + banner + rest, encoding="utf-8")
                except OSError:
                    pass

        print("\n" + "=" * 80)
        print(f"INVESTIGATION COMPLETE: Report saved to {output_report}")
        print("=" * 80)

        summary = self.db.research_question_summary(rq_id)
        dossier = {
            "research_question_id": rq_id,
            "research_question": question,
            "question_normalisation": norm,
            "ai_assignments": assign,
            "papers_analyzed": len(paper_ids),
            "evidence_summary": summary,
            "contradiction_summary": {
                "claim_groups_analyzed": len(contradictions.get("groups", [])),
                "conflicting_groups_found": contradictions.get("n_conflict_groups", 0),
                "conflicting_pairs": contradictions.get("n_conflicting_pairs", 0),
                "verdict_counts": contradictions.get("verdict_counts", {}),
                "same_system_disagreements": contradictions.get("n_same_system_disagreements", 0),
                "different_system_pairs": contradictions.get("n_different_system_pairs", 0),
                # reporting vocabulary: Direct contradictions / Cross-context disagreements / Evidence-level conflict
                **__import__("contradiction_detection").contradiction_counts(
                    self.db.current_contradictions(rq_id),
                    [c for c in self.db.verified_claims(rq_id)
                     if (c.get("semantic_audit") or {}).get("evidence_role") != "secondary_report"]),
            },
            "contradictions": [
                {"claim_a": c["claim_a_id"], "claim_b": c["claim_b_id"], "verdict": c["verdict"],
                 "kind": (c.get("comparison") or {}).get("kind", "same_system"), "why": c.get("reasoning"), "comparison": (c.get("comparison") or {}).get("comparison", [])}
                for c in self.db.current_contradictions(rq_id)],
            "quarantined_claims": [
                {"id": q["id"], "paper_id": q.get("paper_id"), "paper_title": q.get("paper_title"),
                 "statement": q.get("statement"), "quote": q.get("quote"),
                 "reasons": (q.get("semantic_audit") or {}).get("reasons", [])}
                for q in self.db.quarantined_claims(rq_id)],
            "pathway_chains": pathway_results.get("mechanistic_chains", []),
            "empirical_geo_closures": geo_results,
            "hypotheses": synthesis["hypotheses"],
            "research_gaps": synthesis["research_gaps"],
            "falsification_evaluation": falsification_res,
            # the user's uploaded experiment, analysed SEPARATELY from the literature evidence
            "user_experiment": ({"dataset": Path(experimental_dataset).name if experimental_dataset else None,
                                 "group_column": exp_group_col, "outcome_column": exp_outcome_col,
                                 "analysis": bridge_res, "hypothesis_test": falsification_res}
                                if experimental_dataset and exp_group_col and exp_outcome_col else None),
            "audit_flags": audit_flags,
            "speed": "quick" if quick else "full",
            "report_path": output_report,
            "status": "investigation_complete",
        }
        if enhanced:
            dossier.update({
                "prisma": {k: v for k, v in (prisma or {}).items() if k != "paper_ids"} or None,
                "included_paper_ids": paper_ids,
                "evidence_matrix": matrix,
                "evidence_matrix_csv": matrix_csv_text,
                "evidence_matrix_markdown": matrix_markdown(matrix) if matrix else None,
                "protein_interactions": string_net,
                "protein_structures": structures,
                "executive_synthesis": exec_synthesis,
            })

        self._close_stage()
        dossier["timings"] = list(getattr(self, "_timings", []))
        dossier["timings_total_seconds"] = round(sum(t["seconds"] for t in dossier["timings"]), 1)
        print("  [Time] stages: " + " · ".join(f"{t['stage'].replace('_', ' ')} {t['seconds']:.0f}s" for t in dossier["timings"])
              + f" · total {dossier['timings_total_seconds']:.0f}s")
        self._timings = []
        try:            # the report shows where the time went (stage start/end, AI calls, retries, tokens, cache hits)
            rp = Path(output_report)
            if rp.exists() and dossier["timings"]:
                rp.write_text(rp.read_text(encoding="utf-8").rstrip() + "\n\n" + timings_markdown(dossier["timings"]),
                              encoding="utf-8")
        except OSError:
            pass
        from core_utils import output_path
        out_path = output_path(f"research_dossier_{rq_id}.json")
        out_path.write_text(json.dumps(dossier, indent=2, ensure_ascii=False), encoding="utf-8")
        self.db.set_paper_scope(None)
        return dossier

    def close(self):
        self.db.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


# -----------------------------------------------------------------------------
# Command Line Interface (CLI) Entrypoint
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Biology AI Research Engine CLI.")
    parser.add_argument("question", help="Biological research question to investigate")
    parser.add_argument("--max-papers", type=int, default=6, help="Max papers to retrieve and analyze")
    parser.add_argument("--min-relevance", type=float, default=0.30, help="Relevance cutoff (0.0 to 1.0)")
    parser.add_argument("--data", default=None, help="Optional user experimental dataset CSV/Excel")
    parser.add_argument("--group-col", default=None, help="Grouping/treatment column in user dataset")
    parser.add_argument("--outcome-col", default=None, help="Measured outcome column in user dataset")
    parser.add_argument("--output", default="RESEARCH_REPORT.md", help="Output Markdown report path")
    parser.add_argument("--agent", action="store_true", help="Run in autonomous ReAct tool-calling agent mode")
    parser.add_argument("--steps", type=int, default=None, help="Optional ceiling on agent reasoning turns (the agent otherwise spends a budget by objective)")
    parser.add_argument("--no-geo", action="store_true", help="Skip automated GEO loop closure")
    parser.add_argument("--prisma", action="store_true", help="PRISMA funnel + evidence matrix + STRING + deep reasoning")
    parser.add_argument("--screen", type=int, default=150, help="Abstracts to screen in PRISMA mode")
    parser.add_argument("--deep-read", type=int, default=25, help="Top papers to deep-read in PRISMA mode")
    parser.add_argument("--email", default=None, help="Contact email for Unpaywall (or set UNPAYWALL_EMAIL)")
    parser.add_argument("--no-string", action="store_true", help="Skip STRING-db protein interactions")
    parser.add_argument("--no-reasoning", action="store_true", help="Skip deep-reasoning executive summary")
    parser.add_argument("--sources", default=None,
                        help="Comma list for PRISMA: europe_pmc,pubmed,pmc,crossref,openalex,semantic_scholar (default: all)")
    parser.add_argument("--no-citation-chasing", action="store_true", help="Skip Semantic Scholar citation chasing")
    args = parser.parse_args()

    probe = AIRouter()
    if not probe.configured_providers():
        print("=" * 80 + "\n" + NO_KEY_MESSAGE + "\n" + "=" * 80)
        raise SystemExit(2)
    with ResearchEngine(contact_email=args.email) as engine:
        if args.agent:
            engine.run_agent(args.question, max_steps=args.steps)
        else:
            engine.investigate(
                question=args.question,
                experimental_dataset=args.data,
                exp_group_col=args.group_col,
                exp_outcome_col=args.outcome_col,
                max_papers=args.max_papers,
                min_relevance=args.min_relevance,
                enable_geo=not args.no_geo,
                output_report=args.output,
                use_prisma=args.prisma,
                screen_n=args.screen,
                deep_read_k=args.deep_read,
                use_string=not args.no_string,
                use_deep_reasoning=not args.no_reasoning,
                sources=[s.strip() for s in args.sources.split(",")] if args.sources else None,
                citation_chasing=not args.no_citation_chasing,
            )