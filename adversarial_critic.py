"""
adversarial_critic.py
---------------------
Fault-tolerant second-pass adversarial critic with targeted context slicing.
Supports both standalone CLI auditing and direct programmatic integration.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ai_router import AIRouter

SCHEMA_VERSION = "biology_ai_critique_v0.3"
BASE_DIR = Path(__file__).resolve().parent
DEFAULT_EVIDENCE = BASE_DIR / "evidence_with_reasoning.json"
DEFAULT_OUTPUT = BASE_DIR / "evidence_critique.json"

SECTION_CONTEXT_KEYS = {
    "overall_assessment": [
        "scientific_question",
        "raw_data.data_quality",
        "computed_evidence.experiment.statistics",
        "reasoning.hypotheses",
    ],
    "unsupported_claims": [
        "scientific_question",
        "computed_evidence.experiment",
        "reasoning.observations",
        "reasoning.causal_claims_supported",
        "reasoning.biological_mechanisms",
    ],
    "statistical_concerns": [
        "scientific_question",
        "raw_data.data_quality",
        "computed_evidence.experiment",
    ],
    "alternative_explanations": [
        "scientific_question",
        "raw_data.pearson_correlations",
        "computed_evidence.experiment",
        "reasoning.observations",
        "reasoning.biological_mechanisms",
    ],
    "hypothesis_quality": [
        "scientific_question",
        "reasoning.hypotheses",
        "computed_evidence.experiment.statistics",
    ],
    "research_gaps": [
        "scientific_question",
        "raw_data.data_quality",
        "reasoning.hypotheses",
        "reasoning.research_gaps",
    ],
    "next_experiments": [
        "scientific_question",
        "reasoning.hypotheses",
        "reasoning.research_gaps",
        "reasoning.alternative_explanations",
    ],
}


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def resolve_project_path(value: Optional[str], default_path: Path) -> Path:
    if not value:
        return default_path
    p = Path(value)
    if p.is_absolute():
        return p
    cwd_candidate = Path.cwd() / p
    return cwd_candidate if cwd_candidate.exists() else BASE_DIR / p


def finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def extract_balanced_json(text: str) -> Optional[str]:
    if not text:
        return None
    starts = [(i, ch) for i, ch in enumerate(text) if ch in "{["]
    for start, opener in starts:
        closer = "}" if opener == "{" else "]"
        depth = 0
        in_string = False
        escaped = False
        for i in range(start, len(text)):
            ch = text[i]
            if in_string:
                if escaped:
                    escaped = False
                elif ch == "\\":
                    escaped = True
                elif ch == '"':
                    in_string = False
                continue
            if ch == '"':
                in_string = True
            elif ch == opener:
                depth += 1
            elif ch == closer:
                depth -= 1
                if depth == 0:
                    return text[start : i + 1]
    return None


def strip_markdown_fences(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    return re.sub(r"\s*```$", "", text).strip()


def parse_json_value(text: str) -> Any:
    if not text:
        raise ValueError("Empty AI response.")
    cleaned = strip_markdown_fences(text)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    candidate = extract_balanced_json(cleaned)
    if candidate is None:
        raise ValueError("Could not find complete balanced JSON object or array.")
    return json.loads(candidate)


def get_nested(record: Dict[str, Any], path: str, default: Any = None) -> Any:
    current: Any = record
    for part in path.split("."):
        if not isinstance(current, dict) or part not in current:
            return default
        current = current[part]
    return current


def set_nested(target: Dict[str, Any], path: str, value: Any) -> None:
    parts = path.split(".")
    curr = target
    for p in parts[:-1]:
        curr = curr.setdefault(p, {})
    curr[parts[-1]] = value


def find_reasoning_sections(record: Dict[str, Any]) -> Dict[str, Any]:
    for key in ("ai_interpretation", "reasoning", "scientific_reasoning", "ai_reasoning"):
        value = record.get(key)
        if isinstance(value, dict) and any(
            k in value for k in ("observations", "hypotheses", "statistical_interpretation")
        ):
            return value
    return record


def deterministic_critique(record: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], List[str]]:
    findings: List[Dict[str, Any]] = []
    strengths: List[str] = []

    rows = get_nested(record, "raw_data.data_quality.rows") or get_nested(record, "raw_data.rows")
    if finite_number(rows) and rows < 30:
        findings.append({
            "severity": "HIGH",
            "category": "sample_size",
            "finding": f"Dataset contains only {int(rows)} rows. Limited statistical power.",
            "recommendation": "Expand biological replicates and perform independent experimental runs.",
        })

    missing = get_nested(record, "raw_data.data_quality.total_missing_values") or get_nested(record, "raw_data.missing_values")
    if finite_number(missing):
        if int(missing) == 0:
            strengths.append("No missing values were detected.")
        else:
            findings.append({
                "severity": "HIGH",
                "category": "missing_data",
                "finding": f"Dataset contains {int(missing)} missing values.",
                "recommendation": "Examine missingness pattern and justify imputation method.",
            })

    correlations = get_nested(record, "raw_data.pearson_correlations") or get_nested(record, "computed_evidence.correlations")
    if isinstance(correlations, dict):
        for name, value in correlations.items():
            r = value.get("pearson_r", value.get("r")) if isinstance(value, dict) else value
            if finite_number(r) and abs(float(r)) >= 0.8:
                findings.append({
                    "severity": "MEDIUM",
                    "category": "association_vs_causation",
                    "finding": f"Association {name} is strong (r={float(r):.3f}), but correlation does not establish causation.",
                    "recommendation": "Evaluate confounding variables and design perturbation experiments.",
                })

    reasoning = find_reasoning_sections(record)
    required_sections = ["observations", "statistical_interpretation", "hypotheses", "research_gaps"]
    for section in required_sections:
        if section not in reasoning:
            findings.append({
                "severity": "MEDIUM",
                "category": "missing_reasoning_section",
                "finding": f"Scientific reasoning is missing the '{section}' section.",
                "recommendation": "Ensure reasoning engine completes all sections.",
            })

    return findings, strengths


def slice_record_for_section(record: Dict[str, Any], section: str) -> Dict[str, Any]:
    keys = SECTION_CONTEXT_KEYS.get(section, ["scientific_question"])
    sliced: Dict[str, Any] = {}
    reasoning = find_reasoning_sections(record)
    full_context = copy.deepcopy(record)
    full_context["reasoning"] = reasoning

    for k in keys:
        val = get_nested(full_context, k)
        if val is not None:
            set_nested(sliced, k, val)

    return sliced


SECTION_SPECS = {
    "overall_assessment": {
        "instruction": """Return a concise overall assessment (max 3 sentences).
Return ONLY JSON:
{"overall_assessment": "..."}""",
        "validator": lambda x: isinstance(x, dict) and bool(str(x.get("overall_assessment", "")).strip()),
    },
    "unsupported_claims": {
        "instruction": """Identify up to 3 claims in the reasoning that exceed empirical evidence.
Return ONLY JSON:
{"unsupported_claims": [{"claim": "...", "reason": "...", "required_evidence": ["..."]}]}
If none are unsupported, return: {"unsupported_claims": []}""",
        "validator": lambda x: isinstance(x, dict) and isinstance(x.get("unsupported_claims"), list),
    },
    "statistical_concerns": {
        "instruction": """Identify up to 3 statistical or design concerns present in the evidence.
Return ONLY JSON:
{"statistical_concerns": [{"concern": "...", "evidence": ["..."], "recommendation": "..."}]}
If none, return: {"statistical_concerns": []}""",
        "validator": lambda x: isinstance(x, dict) and isinstance(x.get("statistical_concerns"), list),
    },
    "alternative_explanations": {
        "instruction": """List up to 3 plausible alternative biological or technical explanations.
Return ONLY JSON:
{"alternative_explanations": [{"explanation": "...", "why_plausible": "...", "test": "..."}]}""",
        "validator": lambda x: isinstance(x, dict) and isinstance(x.get("alternative_explanations"), list),
    },
    "hypothesis_quality": {
        "instruction": """Critique existing hypotheses for testability and falsifiability (max 2).
Return ONLY JSON:
{"hypothesis_quality": [{"hypothesis_id": "...", "assessment": "...", "strengths": ["..."], "weaknesses": ["..."], "missing_test": "..."}]}
If none exist, return: {"hypothesis_quality": []}""",
        "validator": lambda x: isinstance(x, dict) and isinstance(x.get("hypothesis_quality"), list),
    },
    "research_gaps": {
        "instruction": """Identify up to 3 important research gaps revealed by the record.
Return ONLY JSON:
{"research_gaps": [{"gap": "...", "why_it_matters": "...", "how_to_address": "..."}]}""",
        "validator": lambda x: isinstance(x, dict) and isinstance(x.get("research_gaps"), list),
    },
    "next_experiments": {
        "instruction": """Propose up to 2 high-impact experiments to test hypotheses or resolve gaps.
Return ONLY JSON:
{"next_experiments": [{"experiment": "...", "question": "...", "key_measurements": ["..."], "expected_interpretation": "...", "falsification_condition": "..."}]}""",
        "validator": lambda x: isinstance(x, dict) and isinstance(x.get("next_experiments"), list),
    },
}


def build_prompt(
    section: str,
    record: Dict[str, Any],
    deterministic_findings: List[Dict[str, Any]],
    strengths: List[str],
) -> str:
    sliced_record = slice_record_for_section(record, section)
    sliced_record["deterministic_findings_summary"] = [f["finding"] for f in deterministic_findings[:3]]
    sliced_record["strengths"] = strengths

    return (
        "You are an adversarial scientific reviewer.\n"
        "Critique the research record without agreeing uncritically.\n"
        "Use ONLY the supplied record slice. Keep output concise to fit strict JSON quotas.\n\n"
        f"TARGET SECTION: {section}\n\n"
        f"{SECTION_SPECS[section]['instruction']}\n\n"
        "RESEARCH RECORD SLICE:\n"
        + json.dumps(sliced_record, indent=2, ensure_ascii=False)
    )


def generate_section(
    router: AIRouter,
    section: str,
    record: Dict[str, Any],
    deterministic_findings: List[Dict[str, Any]],
    strengths: List[str],
    providers: List[str],
    retries_per_provider: int = 2,
) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
    prompt = build_prompt(section, record, deterministic_findings, strengths)
    attempts: List[Dict[str, Any]] = []

    for provider in providers:
        for attempt in range(1, retries_per_provider + 1):
            started = time.time()
            try:
                raw = router.ask(prompt, provider=provider)
                elapsed = time.time() - started
                try:
                    parsed = parse_json_value(raw)
                except Exception as parse_error:
                    attempts.append({
                        "provider": provider,
                        "attempt": attempt,
                        "status": "invalid_json",
                        "error": str(parse_error),
                    })
                    repair_prompt = (
                        "Format the following as strictly valid JSON matching the schema below. No commentary.\n"
                        f"{SECTION_SPECS[section]['instruction']}\n\nInput content to format:\n{raw[:1500]}"
                    )
                    try:
                        repaired_raw = router.ask(repair_prompt, provider=provider)
                        repaired = parse_json_value(repaired_raw)
                        if SECTION_SPECS[section]["validator"](repaired):
                            return repaired, {"status": "validated", "provider": provider, "attempts": attempts}
                    except Exception:
                        pass
                    continue

                if SECTION_SPECS[section]["validator"](parsed):
                    attempts.append({"provider": provider, "attempt": attempt, "status": "validated", "elapsed": round(elapsed, 2)})
                    return parsed, {"status": "validated", "provider": provider, "attempts": attempts}

                attempts.append({"provider": provider, "attempt": attempt, "status": "schema_invalid"})
            except Exception as exc:
                attempts.append({"provider": provider, "attempt": attempt, "status": "error", "error": str(exc)})

    return None, {"status": "failed", "attempts": attempts}


def audit_agent_synthesis(
    router: AIRouter,
    question: str,
    hypotheses: List[Dict[str, Any]],
    claims: List[Dict[str, Any]],
    providers: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """
    Direct programmatic audit of an agent's synthesized hypotheses.
    Returns structured audit flags with specific weaknesses and missing controls.
    """
    provs = providers or ["groq", "nvidia", "gemini"]
    flags: List[Dict[str, Any]] = []

    record = {
        "scientific_question": question,
        "reasoning": {
            "hypotheses": hypotheses,
            "observations": [c.get("statement") for c in claims[:6]],
        },
        "raw_data": {"data_quality": {"claims_count": len(claims)}},
        "computed_evidence": {},
    }

    res, _ = generate_section(
        router=router,
        section="hypothesis_quality",
        record=record,
        deterministic_findings=[],
        strengths=[],
        providers=provs,
    )

    if res and "hypothesis_quality" in res:
        for item in res["hypothesis_quality"]:
            hid = item.get("hypothesis_id", "Hypothesis")
            for weakness in item.get("weaknesses", []):
                flags.append({
                    "entity_type": "hypothesis",
                    "claim_text": hid,
                    "issue": f"[Critic Weakness] {weakness}",
                    "severity": "HIGH",
                    "evidence_reference": "Adversarial Critic",
                })
            missing = item.get("missing_test")
            if missing and str(missing).strip():
                flags.append({
                    "entity_type": "hypothesis",
                    "claim_text": hid,
                    "issue": f"[Missing Orthogonal Control] {missing}",
                    "severity": "MEDIUM",
                    "evidence_reference": "Adversarial Critic",
                })

    return flags


def critique(
    evidence_file: Optional[str] = None,
    output_file: Optional[str] = None,
    provider: str = "groq",
    use_ai: bool = True,
) -> Dict[str, Any]:
    evidence_path = resolve_project_path(evidence_file, DEFAULT_EVIDENCE)
    output_path = resolve_project_path(output_file, DEFAULT_OUTPUT)

    if not evidence_path.exists():
        raise FileNotFoundError(f"Evidence file not found: {evidence_path}")

    record = load_json(evidence_path)
    deterministic_findings, strengths = deterministic_critique(record)

    ai_critique: Dict[str, Any] = {}
    validation = {
        "schema_version": SCHEMA_VERSION,
        "sections_requested": list(SECTION_SPECS.keys()),
        "sections_passed": 0,
        "sections_failed": 0,
        "section_status": {},
    }

    if use_ai:
        router = AIRouter()
        provider_order = [provider] + [p for p in ("groq", "nvidia", "gemini") if p != provider]

        for section in SECTION_SPECS:
            print(f"[{section}] generating critique (compact context)...")
            result, meta = generate_section(
                router=router,
                section=section,
                record=record,
                deterministic_findings=deterministic_findings,
                strengths=strengths,
                providers=provider_order,
                retries_per_provider=2,
            )
            validation["section_status"][section] = meta
            if result is not None:
                ai_critique[section] = result
                validation["sections_passed"] += 1
                print(f"  -> [{section}] validated.")
            else:
                validation["sections_failed"] += 1
                ai_critique[section] = {"status": "failed", "message": "Critique unavailable."}
                print(f"  -> [{section}] generation failed. Preserving deterministic findings.")

    output = {
        "schema_version": SCHEMA_VERSION,
        "source_evidence_file": str(evidence_path),
        "deterministic_findings": deterministic_findings,
        "strengths": strengths,
        "ai_critique": ai_critique,
        "validation": validation,
    }
    save_json(output_path, output)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Run adversarial critic with targeted context slicing.")
    parser.add_argument("--evidence", default=None, help="Path to evidence_with_reasoning.json")
    parser.add_argument("--output", default=None, help="Path to evidence_critique.json")
    parser.add_argument("--provider", default="groq", choices=["gemini", "groq", "nvidia"])
    parser.add_argument("--no-ai", action="store_true", help="Deterministic critique only")
    args = parser.parse_args()

    try:
        critique(evidence_file=args.evidence, output_file=args.output, provider=args.provider, use_ai=not args.no_ai)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()