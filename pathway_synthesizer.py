"""
pathway_synthesizer.py
----------------------
Cross-Study Mechanistic Pathway Synthesizer:
Connects claims across independent papers into multi-hop causal chains:
    Intervention -> Transporters/Sensors -> Signaling/TFs -> Morphology -> Biomass
"""

from __future__ import annotations

from typing import Any, Dict, List
from database import ResearchDB


class PathwaySynthesizer:
    def __init__(self, db: ResearchDB):
        self.db = db

    def build_pathways(self, research_question_id: int) -> Dict[str, Any]:
        claims = self.db.verified_claims(research_question_id)
        if not claims:
            return {"total_chains_discovered": 0, "mechanistic_chains": [], "chains": [],
                    "summary": "No verified claims available to construct pathways."}

        # 1. Normalize nodes
        nodes: Dict[str, List[Dict[str, Any]]] = {}
        for c in claims:
            subj = str(c.get("intervention") or "").lower().strip()
            outc = str(c.get("measurement") or c.get("outcome") or "").lower().strip()
            direction = c.get("canonical_direction") or c.get("effect_direction") or "unclear"
            if not subj or not outc or direction == "unclear":
                continue

            edge = {
                "claim_id": c["id"],
                "paper_id": c["paper_id"],
                "paper_title": c.get("paper_title"),
                "source": subj,
                "target": outc,
                "direction": direction,
                "quote": c.get("quote"),
                "p_value": c.get("p_value"),
                "model": c.get("claim_model"),
            }
            nodes.setdefault(subj, []).append(edge)

        # 2. Multi-Hop Chain Discovery (e.g. A -> B and B -> C)
        def _linked(a: str, b: str) -> bool:
            # substring matching on very short labels ("ki", "ros") produced nonsense chains
            if a == b:
                return True
            shorter = a if len(a) < len(b) else b
            return len(shorter) >= 4 and (a in b or b in a)

        chains = []
        for start_node, outgoing_edges in nodes.items():
            for edge_1 in outgoing_edges:
                intermediate = edge_1["target"]
                # Look for matching hops where intermediate is the cause
                for hop_node in nodes:
                    if hop_node == start_node:
                        continue
                    if _linked(hop_node, intermediate):
                        for edge_2 in nodes[hop_node]:
                            if (edge_2["claim_id"] == edge_1["claim_id"]
                                    or edge_2["paper_id"] == edge_1["paper_id"]   # must be cross-study
                                    or edge_2["target"] in (start_node, intermediate)):
                                continue
                            models = [m for m in (edge_1.get("model"), edge_2.get("model")) if m]
                            mixed = len(set(models)) > 1
                            chain = {
                                "pathway": f"{start_node} -> ({edge_1['direction']}) -> {intermediate} -> ({edge_2['direction']}) -> {edge_2['target']}",
                                "evidence_level": "M1 candidate (cross-study inference)" + (
                                    f"; links different model systems ({' -> '.join(m.replace('_', ' ') for m in models)})" if mixed else ""),
                                "step_1": edge_1,
                                "step_2": edge_2,
                                "biological_interpretation": (
                                    f"Cross-study mechanism: Paper '{(edge_1['paper_title'] or 'Untitled')[:40]}' shows {start_node} {edge_1['direction']}s "
                                    f"{intermediate}, which Paper '{(edge_2['paper_title'] or 'Untitled')[:40]}' links to {edge_2['direction']} in {edge_2['target']}. "
                                    f"Candidate mechanism only - no single experiment demonstrated the full sequence."
                                ),
                            }
                            chains.append(chain)

        # 3. Deduplicate and rank chains
        seen_paths = set()
        unique_chains = []
        for ch in chains:
            if ch["pathway"] not in seen_paths:
                seen_paths.add(ch["pathway"])
                unique_chains.append(ch)

        return {
            "total_chains_discovered": len(unique_chains),
            "mechanistic_chains": unique_chains[:6],
        }