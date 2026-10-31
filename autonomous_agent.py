"""
autonomous_agent.py
-------------------
High-Speed General Biology Autonomous Research Agent with Integrated Adversarial Critic:
  1. Searches PubMed, PMC, Europe PMC in parallel.
  2. Batch-reads papers and extracts verified quote-grounded claims into SQLite.
  3. Evaluates competing explanations and red-teams hypotheses.
  4. Automatically runs the Adversarial Critic on finish to prevent false claims.
  5. Generates a balanced RESEARCH_REPORT.md on completion.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

from adversarial_critic import audit_agent_synthesis
from ai_router import AIRouter
from claim_extraction import ClaimExtractor
from database import ResearchDB
from llm_json import parse_json_response
from report_generator import ResearchReportGenerator
from scientific_tools import ScientificToolbox


@dataclass
class KnowledgePillar:
    pillar_id: str
    topic: str
    required_evidence: str
    confidence: float = 0.0
    evidence_anchors: List[str] = field(default_factory=list)
    source_paper_ids: Set[int] = field(default_factory=set)


@dataclass
class ResearchBudget:
    """Budget by OBJECTIVE, not by a count of steps: a 'step' is not a unit (a search, reading a 30-page paper and
    checking one quotation cost very different amounts). Each resource has its own allowance, every action is charged
    what it actually consumes, and the agent sees what is left and decides how to spend it. The run ends when the
    objective is met (enough papers read, competing explanations checked, every proof pillar covered by verified
    evidence) or when ONE of the resources is spent - never because an arbitrary step number was reached.

      tokens       LLM tokens (prompt + answer; estimated as characters / 4) for the agent's reasoning and paper reading
      tool_calls   searches, paper reads, gene / chemical look-ups, virtual-lab runs
      seconds      wall-clock time the user allows the investigation (a limit, not a prediction)
      sources      papers read in full
      iterations   reasoning turns (a safety ceiling against a model that loops without acting)
    """
    tokens: int = 600_000
    tool_calls: int = 80
    seconds: float = 3600.0
    sources: int = 40
    iterations: int = 120
    min_papers_read: int = 10
    max_searches: int = 12
    used: Dict[str, float] = field(default_factory=lambda: {"tokens": 0, "tool_calls": 0, "seconds": 0.0, "sources": 0,
                                                           "iterations": 0})
    started: float = 0.0
    searches_used: int = 0

    COSTS = {  # what each action is charged besides the tokens it really used
        "search_literature": {"tool_calls": 1}, "batch_read_papers": {"tool_calls": 1}, "lookup_gene": {"tool_calls": 1},
        "lookup_chemical": {"tool_calls": 1}, "virtual_lab": {"tool_calls": 1}, "investigate_competing_explanation": {},
    }

    @property
    def steps_used(self) -> int:              # kept for callers that report how many turns were taken
        return int(self.used["iterations"])

    def start(self) -> None:
        import time
        self.started = time.time()

    def charge(self, **amounts: float) -> None:
        for k, v in amounts.items():
            self.used[k] = self.used.get(k, 0) + v

    def tick(self) -> None:
        import time
        if self.started:
            self.used["seconds"] = time.time() - self.started

    def limits(self) -> Dict[str, float]:
        return {"tokens": self.tokens, "tool_calls": self.tool_calls, "seconds": self.seconds, "sources": self.sources,
                "iterations": self.iterations}

    def remaining(self) -> Dict[str, float]:
        self.tick()
        return {k: max(0.0, lim - self.used.get(k, 0)) for k, lim in self.limits().items()}

    def exhausted(self) -> Optional[str]:
        """Name of the first resource that is spent, or None."""
        rem = self.remaining()
        return next((k for k in ("seconds", "tokens", "tool_calls", "iterations") if rem[k] <= 0), None)

    def describe(self) -> str:
        r = self.remaining()
        return (f"tokens {int(r['tokens']):,} of {self.tokens:,} · tool calls {int(r['tool_calls'])} of {self.tool_calls} · "
                f"papers {int(r['sources'])} of {self.sources} · time {-(-int(r['seconds']) // 60)} of {int(self.seconds // 60)} min · "
                f"turns {int(r['iterations'])} of {self.iterations}")


@dataclass
class ActivePaperQueue:
    research_question_id: int
    queue: List[int] = field(default_factory=list)
    read_papers: Set[int] = field(default_factory=set)

    def enqueue(self, paper_ids: List[int]) -> None:
        for pid in paper_ids:
            if pid not in self.queue and pid not in self.read_papers:
                self.queue.append(pid)

    def mark_read(self, paper_id: int) -> None:
        self.read_papers.add(paper_id)
        if paper_id in self.queue:
            self.queue.remove(paper_id)

    def next_unread_batch(self, n: int = 3) -> List[int]:
        return self.queue[:n]


AGENT_PROMPT = """You are an Autonomous Scientific Research Controller investigating:
"{question}"

ACTIVE EVIDENCE SCORECARD:
{scorecard_summary}

AVAILABLE TOOLS (Use batch reading for high speed):
1. "search_literature": {{"query": "string", "max_results": 3}}
   -> Searches PubMed, PMC, and Europe PMC in parallel; returns numeric Paper IDs (max_results up to 10).
2. "batch_read_papers": {{"paper_ids": [int, int, int]}}
   -> Reads multiple papers simultaneously in one step!
3. "lookup_gene": {{"symbol": "string", "organism": "string"}}
   -> Queries NCBI Gene for verified function and locus identity.
4. "lookup_chemical": {{"name": "string"}}
   -> Queries PubChem for drug or chemical compound structure.
5. "investigate_competing_explanation": {{"alternative_mechanism": "string", "testable_contrast": "string"}}
   -> Evaluates alternative hypotheses or counter-mechanisms to prevent confirmation bias.
6. "virtual_lab": {{"tool": "simulate_virus|protein_properties|analyze_dna|mutate_dna|align|dose_response|pk_one_compartment|exposure_vs_potency|simulate_growth|perturb_pathway|power_two_groups|compare_groups|fit_dose_response|simulate_epidemic|protein_structure", "params": {{...}}}}
   -> Runs an in-silico test locally (genes, proteins, 3D structures, viruses, drugs, pathways, statistics).
      protein_structure: {{"source": "pdb|alphafold", "identifier": "6LU7 or a gene / UniProt ID", "residue": 145}}
      gives the structure's confidence, pockets, bound-ligand contacts and a residue's structural context.
      Results are simulations / structural context, NOT evidence; use them to check whether a hypothesis is
      consistent and testable.
7. "finish": {{
       "executive_summary": "string",
       "key_mechanism": "string",
       "evaluated_competing_explanations": ["..."],
       "falsifiable_hypotheses": [{{ "statement": "...", "mechanism": "...", "predictions": ["..."], "falsification_condition": "..." }}],
       "unresolved_research_gaps": ["..."]
   }}
   -> Finalizes the investigation.

BUDGET LEFT (spend it where it answers the question best; cheap actions first when they suffice):
{budget}
OBJECTIVE: at least {min_papers} papers read, competing explanations checked, every proof pillar backed by verified claims.
LATEST OBSERVATION:
{observation}

DIRECTIVES:
- If unread papers are in the queue, use "batch_read_papers" to read several at once.
- Search again with a different query when a pillar has no verified evidence yet; read human trials first for clinical questions.
- Call "finish" when the objective is met, or earlier only if the remaining budget cannot buy useful evidence.

Output strictly valid JSON:
{{
  "thought": "What evidence am I addressing next?",
  "action": "tool_name",
  "params": {{ ... }}
}}"""


class AutonomousResearchAgent:
    def __init__(
        self,
        db: ResearchDB,
        router: AIRouter,
        max_steps: Optional[int] = None,
        research_question_id: Optional[int] = None,
        budget: Optional[ResearchBudget] = None,
    ):
        self.db = db
        self.router = router
        self.tools = ScientificToolbox(db=self.db)
        self.claim_extractor = ClaimExtractor(self.db, self.router)
        self.reporter = ResearchReportGenerator(db=self.db)
        self.budget = budget or ResearchBudget()
        if max_steps:                    # an explicit turn ceiling from an old caller is honoured as the iteration limit only
            self.budget.iterations = int(max_steps)
        self.rq_id = research_question_id or 1
        self.queue = ActivePaperQueue(research_question_id=self.rq_id)
        self.pillars: List[KnowledgePillar] = []
        self.competing_checked: bool = False

    def initialize_knowledge_map(self, question: str) -> None:
        print("\n[Metacognition] Deconstructing biological question into proof pillars...")
        prompt = f"""Deconstruct this biological question into 3 essential empirical proof pillars:
"{question}"
Return strictly valid JSON:
{{"pillars": [{{"pillar_id": "P1", "topic": "Phenotypic Outcome / Dose Response", "required_evidence": "..."}},
              {{"pillar_id": "P2", "topic": "Molecular Pathway / Mechanism", "required_evidence": "..."}},
              {{"pillar_id": "P3", "topic": "Causal Specificity / Counter-Mechanisms", "required_evidence": "..."}}]}}"""
        try:
            raw = self._ask(prompt, task="agent")
            parsed = parse_json_response(raw)
            for p in parsed.get("pillars", []):
                self.pillars.append(KnowledgePillar(p["pillar_id"], p["topic"], p.get("required_evidence", "")))
        except Exception:
            self.pillars = [
                KnowledgePillar("P1", "Phenotypic Response / Endpoint", "Quantitative experimental measurements"),
                KnowledgePillar("P2", "Molecular Pathway / Cellular Cascade", "Specific genes, receptors, and enzymes"),
                KnowledgePillar("P3", "Causal Specificity & Controls", "Controlled comparisons excluding confounders"),
            ]

        for p in self.pillars:
            print(f"  * [{p.pillar_id}] {p.topic}")

    def summary(self) -> str:
        sat = self.objective_met()
        status = "OBJECTIVE MET - you may call 'finish'." if sat else "STATUS: GATHERING EVIDENCE."
        lines = [
            status,
            f"- Papers Read in Run: {len(self.queue.read_papers)}/{self.budget.min_papers_read} (Unread in Queue: {self.queue.queue})",
            f"- Competing Explanations Checked: {'✓' if self.competing_checked else '✗'}",
            "- Knowledge Pillars:"
        ]
        for p in self.pillars:
            lines.append(f"  * [{p.pillar_id}] {p.topic} (Confidence: {p.confidence:.2f})")
        return "\n".join(lines)

    def objective_met(self) -> bool:
        return (len(self.queue.read_papers) >= self.budget.min_papers_read and self.competing_checked
                and all(p.confidence >= 0.75 for p in self.pillars))

    def _ask(self, prompt: str, task: str) -> str:
        """Every model call is charged its real size against the token budget."""
        raw = self.router.ask(prompt, task=task) or ""
        self.budget.charge(tokens=(len(prompt) + len(raw)) // 4)
        return raw

    def execute_tool(self, action: str, params: Dict[str, Any], question: str) -> str:
        for k, v in self.budget.COSTS.get(action, {}).items():
            self.budget.charge(**{k: v})
        try:
            if action in ("search_literature", "search_papers"):
                if self.queue.queue:
                    return f"You already have unread papers {self.queue.queue} in queue! Call 'batch_read_papers' on them first."
                if self.budget.searches_used >= self.budget.max_searches:
                    return "Search allowance used up: read the queued papers or finish."

                query = params.get("query") or question
                paper_ids = self.tools.source_manager.retrieve_and_ingest(
                    db=self.db,
                    queries=[query],
                    question=question,
                    research_question_id=self.rq_id,
                    max_papers=max(1, min(10, int(params.get("max_results", 8) or 8))),
                    min_relevance=0.25,
                )
                self.budget.searches_used += 1

                if not paper_ids:
                    existing = self.db.list_papers_summary(self.rq_id)
                    paper_ids = [p["id"] for p in existing[:3]]

                self.queue.enqueue(paper_ids)
                res = [f"[Paper ID: {pid}] \"{(self.tools.get_paper_metadata(pid).get('title') or 'Untitled')[:65]}\"" for pid in paper_ids if self.tools.get_paper_metadata(pid)]
                return f"Retrieved {len(res)} papers into queue:\n" + "\n".join(res)

            elif action in ("batch_read_papers", "read_paper"):
                pids = params.get("paper_ids")
                if isinstance(pids, (str, int)):
                    pids = re.findall(r"\d+", str(pids))
                if not pids:
                    pids = [params.get("paper_id")] if params.get("paper_id") else self.queue.next_unread_batch(3)

                texts = []
                for pid in [int(x) for x in pids if str(x).strip().isdigit()][:8]:
                    if self.budget.remaining()["sources"] <= 0:
                        texts.append("Paper allowance used up: finish with what was read.")
                        break
                    t = self.tools.get_full_text(pid, section_type="results") or self.tools.get_full_text(pid, section_type="abstract") or ""
                    self.queue.mark_read(pid)
                    self.budget.charge(sources=1, tokens=min(len(t), 60_000) // 4 + 1500)   # extraction reads the text
                    n_claims = 0
                    try:
                        c_res = self.claim_extractor.extract(pid, self.rq_id, question, force=False, max_claims=7)
                        n_claims = int(c_res.get("claims", 0) or 0)
                        print(f"    [Claim Extractor] Paper #{pid} -> {n_claims} verified claims saved to DB.")
                    except Exception:
                        pass

                    texts.append(f"=== Paper #{pid} Results ({n_claims} verified claim(s)) ===\n{t[:900]}")
                    # a pillar gains confidence only from VERIFIED evidence, not from the act of reading
                    for p in self.pillars:
                        if n_claims:
                            p.confidence = min(1.0, p.confidence + 0.15 * min(n_claims, 3))
                            p.source_paper_ids.add(pid)

                return "\n\n".join(texts) if texts else "No valid paper IDs provided."

            elif action == "investigate_competing_explanation":
                self.competing_checked = True
                for p in self.pillars:
                    p.confidence = min(1.0, p.confidence + 0.25)
                return f"Evaluated competing explanation: {params.get('alternative_mechanism')} (Contrast: {params.get('testable_contrast')})."

            elif action == "virtual_lab":
                from virtual_lab import run_tool
                res = run_tool(str(params.get("tool") or ""), **(params.get("params") or {}))

                def slim(x: Any) -> Any:          # coordinates, curves and plot points are for the website, not the model
                    if isinstance(x, dict):
                        return {k: slim(v) for k, v in x.items()
                                if k not in ("curve", "model_text", "points", "volcano", "embedding_pc12", "profile", "string")}
                    if isinstance(x, list):
                        return [slim(v) for v in x[:12]]
                    return x
                res = slim(res)
                return "Virtual lab (in silico, not evidence): " + json.dumps(res, default=str)[:1500]

            elif action in ("lookup_gene", "search_gene"):
                res = self.tools.search_gene(params.get("symbol", ""), params.get("organism"))
                return f"NCBI Gene [{res.get('symbol')}]: ID={res.get('gene_id')} - {res.get('description')}" if res else "Gene not found."

            elif action == "lookup_chemical":
                res = self.tools.bio.pubchem_compound(params.get("name", ""))
                return f"PubChem [{res.get('name')}]: CID={res.get('cid')}" if res else "Compound not found."

            return f"Unknown tool: '{action}'."
        except Exception as exc:
            return f"Tool execution notice: {exc}"

    def persist_synthesis(self, params: Dict[str, Any], question: str) -> None:
        """Saves hypotheses, triggers the Adversarial Critic by default, and compiles report."""
        key_mech = params.get("key_mechanism") or "Biological regulation"
        hypotheses_records = []

        # 1. Save Hypotheses
        for h in params.get("falsifiable_hypotheses", []):
            if isinstance(h, dict):
                hid = self.db.add_hypothesis(
                    research_question_id=self.rq_id,
                    hypothesis={
                        "statement": h.get("statement", "Hypothesis"),
                        "mechanism": h.get("mechanism", key_mech),
                        "predictions": h.get("predictions", []),
                        "falsification_conditions": [h.get("falsification_condition", "Refuted under experimental contrast")],
                        "status": "proposed",
                    },
                )
                hypotheses_records.append({"id": hid, **h})

        # 2. Save Research Gaps
        for g in params.get("unresolved_research_gaps", []):
            desc = g.get("description", str(g)) if isinstance(g, dict) else str(g)
            self.db.add_research_gap(
                research_question_id=self.rq_id,
                description=desc,
                category="general",
                supporting_observation="Identified during autonomous agent investigation",
            )

        # 3. Log Audit Flags for underpowered claims
        claims = self.db.verified_claims(self.rq_id)   # critic and flags see verified evidence only
        for c in claims:
            if c.get("effect_direction") == "no_effect" and c.get("sample_size") and c["sample_size"] < 10:
                self.db.add_audit_flag(
                    research_question_id=self.rq_id,
                    entity_type="claim",
                    entity_id=c["id"],
                    claim_text=c.get("statement", ""),
                    evidence_reference=f"paper:{c.get('paper_id')}",
                    issue=f"Reported 'no_effect' with small sample size (n={c['sample_size']}). Risk of Type II error.",
                    severity="MEDIUM",
                )

        # 4. RUN ADVERSARIAL CRITIC BY DEFAULT ON SYNTHESIZED HYPOTHESES
        print("\n" + "=" * 80)
        print("[AUTOMATIC AUDIT] Running Adversarial Critic on Hypotheses...")
        print("=" * 80)
        try:
            critic_flags = audit_agent_synthesis(
                router=self.router,
                question=question,
                hypotheses=hypotheses_records,
                claims=claims,
            )
            for flag in critic_flags:
                self.db.add_audit_flag(
                    research_question_id=self.rq_id,
                    entity_type=flag["entity_type"],
                    entity_id=None,
                    claim_text=flag["claim_text"],
                    evidence_reference=flag["evidence_reference"],
                    issue=flag["issue"],
                    severity=flag["severity"],
                )
            print(f"  -> Critic Audit Complete: Logged {len(critic_flags)} adversarial checks into knowledge base.")
        except Exception as exc:
            print(f"  [Critic Notice] Automatic review encountered notice: {exc}")

        # 5. Compile Dossier
        try:
            self.reporter.generate(self.rq_id, output_path="RESEARCH_REPORT.md")
        except Exception:
            pass

    def run(self, question: str) -> Dict[str, Any]:
        print("=" * 80)
        print("GENERAL BIOLOGY AUTONOMOUS RESEARCH AGENT (HIGH-SPEED MODE)")
        print(f"Goal: {question}")
        print("=" * 80)

        self.budget.start()
        self.initialize_knowledge_map(question)
        observation = "Investigation started. Ready to retrieve primary literature."
        stop_reason = None
        recent: List[str] = []

        while True:
            stop_reason = self.budget.exhausted()
            if stop_reason:
                break
            self.budget.charge(iterations=1)
            step = self.budget.steps_used
            print(f"\n[Agent turn {step}] Budget left: {self.budget.describe()}")

            prompt = AGENT_PROMPT.format(
                question=question,
                budget=self.budget.describe(),
                min_papers=self.budget.min_papers_read,
                scorecard_summary=self.summary(),
                observation=observation[:1200],
            )

            try:
                raw = self._ask(prompt, task="agent")
            except Exception as exc:
                print(f"  [Agent] All AI providers failed this step ({str(exc)[:120]}); using scripted next action.")
                raw = ""
            try:
                plan = parse_json_response(raw)
            except Exception:
                match = re.search(r"\{.*\}", raw, re.DOTALL)
                try:
                    plan = json.loads(match.group(0)) if match else {}
                except json.JSONDecodeError:
                    plan = {}

            if isinstance(plan, list):
                plan = plan[0] if (plan and isinstance(plan[0], dict)) else {}
            elif not isinstance(plan, dict):
                plan = {}

            thought = plan.get("thought", "Evaluating evidence sufficiency.")
            action = plan.get("action", "")
            params = plan.get("params", {})
            if not isinstance(params, dict):
                params = {}

            if not action:
                if self.objective_met():
                    action = "finish"
                elif self.queue.queue:
                    action = "batch_read_papers"
                    params = {"paper_ids": self.queue.next_unread_batch(3)}
                elif not self.competing_checked:
                    action = "investigate_competing_explanation"
                    params = {
                        "alternative_mechanism": "Non-specific physiological stress or off-target interaction",
                        "testable_contrast": "Equimolar vehicle control and orthogonal knockout",
                    }
                else:
                    action = "search_literature"
                    params = {"query": question, "max_results": 8}

            # no progress: the same action with the same parameters again and again buys nothing - take the next useful
            # action instead (read queued papers, search, or finish when the objective is met)
            sig = f"{action}|{json.dumps(params, sort_keys=True, default=str)[:300]}"
            recent.append(sig)
            if len(recent) >= 3 and len(set(recent[-3:])) == 1 and action != "batch_read_papers":
                if self.objective_met():
                    action, params = "finish", params if action == "finish" else {}
                elif self.queue.queue:
                    action, params = "batch_read_papers", {"paper_ids": self.queue.next_unread_batch(5)}
                elif self.budget.searches_used < self.budget.max_searches:
                    action, params = "search_literature", {"query": f"{question} {len(recent)}".rsplit(" ", 1)[0], "max_results": 8}
                else:
                    stop_reason = "progress"
                    break
                recent.clear()

            print(f"  Thought: {thought}")
            print(f"  -> Action: {action}({params})")

            if action == "finish":
                nothing_left = not self.queue.queue and self.budget.searches_used >= self.budget.max_searches
                if not self.objective_met() and not nothing_left:
                    observation = (f"Cannot finish yet: objective not met (papers read {len(self.queue.read_papers)}/"
                                   f"{self.budget.min_papers_read}, competing explanations "
                                   f"{'checked' if self.competing_checked else 'not checked'}, pillars "
                                   + ", ".join(f"{p.pillar_id} {p.confidence:.2f}" for p in self.pillars) + ").")
                    continue
                else:
                    print("\n" + "=" * 80)
                    print("INVESTIGATION COMPLETE (Synthesis Finalized)")
                    print("=" * 80)
                    print(f"Executive Summary:\n{params.get('executive_summary', '')}\n")
                    if "key_mechanism" in params:
                        print(f"Key Mechanism:\n{params['key_mechanism']}\n")
                    if "falsifiable_hypotheses" in params:
                        print(f"Falsifiable Hypotheses:\n{json.dumps(params['falsifiable_hypotheses'], indent=2)}\n")

                    self.persist_synthesis(params, question)
                    return {"status": "complete", "steps_used": step, "budget_used": dict(self.budget.used),
                            "final_synthesis": params}

            observation = self.execute_tool(action, params, question)
            print(f"  -> Observation:\n{observation[:250]}...")

        # a resource ran out before the objective was met: finish with what was gathered, and say which one
        print(f"\n[Budget] The {stop_reason.replace('_', ' ')} budget is spent ({self.budget.describe()}). Finalizing synthesis...")
        try:
            final_raw = self.router.ask(task="synthesis", prompt=
                f"Synthesize biological findings for: '{question}'. Return JSON: {{\"executive_summary\": \"...\", \"key_mechanism\": \"...\", \"falsifiable_hypotheses\": [...]}}",
            )
            final_plan = parse_json_response(final_raw)
            if not isinstance(final_plan, dict):
                raise ValueError("non-object synthesis")
        except Exception:
            final_plan = {"executive_summary": "Investigation completed across primary studies."}

        self.persist_synthesis(final_plan, question)
        return {"status": "complete_budget_spent", "stopped_because": f"{stop_reason} budget spent",
                "budget_used": dict(self.budget.used), "final_synthesis": final_plan}