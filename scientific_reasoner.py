import json
import re
import time
from pathlib import Path
from typing import Any, Dict, List

from ai_router import AIRouter

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = BASE_DIR / "evidence_record.json"
DEFAULT_OUTPUT = BASE_DIR / "evidence_with_reasoning.json"

SECTIONS = [
    "observations",
    "statistical_interpretation",
    "associations",
    "causal_assessment",
    "alternative_explanations",
    "biological_mechanisms",
    "hypotheses",
    "research_gaps",
    "recommended_next_actions",
]
MAX_ATTEMPTS = 3


def resolve_project_path(value: str, default: Path) -> Path:
    if not value:
        return default
    p = Path(value)
    return p if p.is_absolute() else BASE_DIR / p


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    temp.replace(path)


def extract_json_object(text: str) -> str:
    if not isinstance(text, str) or not text.strip():
        raise ValueError("AI returned an empty response.")

    text = text.strip()
    text = re.sub(r"^\s*```(?:json)?\s*", "", text, flags=re.I)
    text = re.sub(r"\s*```\s*$", "", text)

    start = text.find("{")
    if start < 0:
        raise ValueError("AI response did not contain a JSON object.")

    in_string = False
    escaped = False
    depth = 0

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
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start:i + 1]

    if in_string:
        raise ValueError("AI response contained an unterminated JSON string.")
    raise ValueError("AI response contained an incomplete JSON object.")


def parse_json(text: str) -> Dict[str, Any]:
    return json.loads(extract_json_object(text))


def compact_evidence(evidence: Dict[str, Any], section: str) -> str:
    if section == "observations":
        payload = {
            "scientific_question": evidence.get("scientific_question", ""),
            "raw_data": evidence.get("raw_data", {}),
            "experiment": evidence.get("computed_evidence", {}).get("experiment", {}),
        }
    else:
        payload = {
            "scientific_question": evidence.get("scientific_question", ""),
            "raw_data": evidence.get("raw_data", {}),
            "computed_evidence": evidence.get("computed_evidence", {}),
        }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def section_schema(section: str) -> str:
    if section == "hypotheses":
        return (
            '{"items":[{"id":"H1","statement":"...",'
            '"mechanism":"...",'
            '"supporting_evidence":["..."],'
            '"competing_explanations":["..."],'
            '"predictions":["..."],'
            '"falsification_conditions":["..."],'
            '"next_experiment":"..."}]}'
        )
    return (
        '{"items":[{"statement":"...",'
        '"evidence_reference":"...",'
        '"confidence":"high|medium|low"}]}'
    )


def prompt_for_section(section: str, evidence: Dict[str, Any]) -> str:
    return f"""
You are a scientific reasoning component in a biology evidence engine.

Python has already computed the numerical evidence. Do NOT recalculate numbers
and do NOT invent data.

Return ONLY one complete JSON object. No markdown. No commentary.
Keep it SHORT: normally 0-4 items. Never write long paragraphs.
Every substantive claim must be grounded in the supplied evidence.
Do not turn correlation into causation.
Do not present a possible biological mechanism as established fact.
For this n=12 dataset, distinguish statistical evidence from generalization.
If evidence is insufficient, return an empty items list.

Requested section: {section}

Required JSON shape:
{section_schema(section)}

Evidence:
{compact_evidence(evidence, section)}
""".strip()


def validate_section(section: str, obj: Dict[str, Any]) -> List[Dict[str, Any]]:
    if not isinstance(obj, dict):
        raise ValueError("Section result is not an object.")
    items = obj.get("items")
    if not isinstance(items, list):
        raise ValueError("Missing 'items' array.")
    if len(items) > 4:
        raise ValueError("More than 4 items returned.")

    for item in items:
        if not isinstance(item, dict):
            raise ValueError("Section item is not an object.")

        if section == "hypotheses":
            required = [
                "id", "statement", "mechanism", "supporting_evidence",
                "competing_explanations", "predictions",
                "falsification_conditions", "next_experiment"
            ]
            for key in required:
                if key not in item:
                    raise ValueError(f"Hypothesis missing '{key}'.")
            for key in (
                "supporting_evidence", "competing_explanations",
                "predictions", "falsification_conditions"
            ):
                if not isinstance(item[key], list):
                    raise ValueError(f"Hypothesis '{key}' must be a list.")
        else:
            if "statement" not in item or "confidence" not in item:
                raise ValueError("Section item missing statement/confidence.")
            if item["confidence"] not in {"high", "medium", "low"}:
                raise ValueError("Invalid confidence value.")

    return items


def ask_section(router: AIRouter, section: str, evidence: Dict[str, Any]) -> List[Dict[str, Any]]:
    prompt = prompt_for_section(section, evidence)
    last_error = None

    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            raw = router.ask(prompt, task="hypotheses") if hasattr(router, "order_for") else router.ask(prompt, provider="groq")
            parsed = parse_json(raw)
            return validate_section(section, parsed)
        except Exception as exc:
            last_error = exc
            print(f"  attempt {attempt}/{MAX_ATTEMPTS} failed: {exc}")
            if attempt < MAX_ATTEMPTS:
                time.sleep(1.5 * attempt)

    raise RuntimeError(
        f"Section '{section}' failed after {MAX_ATTEMPTS} attempts: {last_error}"
    )


def build_reasoning(router: AIRouter, evidence: Dict[str, Any]) -> Dict[str, Any]:
    result = {
        "scientific_question": evidence.get("scientific_question", ""),
        "observations": [],
        "statistical_interpretation": [],
        "associations": [],
        "causal_claims_supported": [],
        "causal_claims_not_supported": [],
        "limitations": [],
        "alternative_explanations": [],
        "biological_mechanisms": [],
        "hypotheses": [],
        "research_gaps": [],
        "recommended_next_actions": [],
    }

    deterministic = evidence.get("limitations", [])
    if isinstance(deterministic, list):
        result["limitations"].extend(deterministic)

    # Track which AI-generated sections were successfully validated.
    result["_reasoning_metadata"] = {
        "engine_version": "sectioned_v2_fixed",
        "provider": "groq",
        "validated_sections": [],
    }

    for section in SECTIONS:
        print(f"\n[{section}] generating...")
        items = ask_section(router, section, evidence)
        print(f"[{section}] validated {len(items)} item(s).")
        result["_reasoning_metadata"]["validated_sections"].append(section)

        if section == "causal_assessment":
            # Keep causal support conservative: statements explicitly framed as
            # unsupported are placed in the not-supported field.
            for item in items:
                text = str(item.get("statement", "")).lower()
                if any(x in text for x in (
                    "not supported", "cannot establish", "cannot infer",
                    "insufficient", "does not establish"
                )):
                    result["causal_claims_not_supported"].append(item)
                else:
                    result["causal_claims_supported"].append(item)
        elif section == "hypotheses":
            result["hypotheses"] = items
        else:
            result[section] = items

    return result


def main() -> int:
    print("=" * 80)
    print("SCIENTIFIC REASONING ENGINE - ROBUST SECTIONED VERSION")
    print("=" * 80)

    raw_input = input(f"\nEvidence file (default: {DEFAULT_INPUT.name}): ").strip()
    evidence_path = resolve_project_path(raw_input, DEFAULT_INPUT)

    if not evidence_path.exists():
        print(f"\nERROR: Evidence file not found:\n{evidence_path}")
        return 1

    try:
        evidence = load_json(evidence_path)
    except Exception as exc:
        print(f"\nERROR: Could not read evidence JSON: {exc}")
        return 1

    print(f"\nUsing evidence: {evidence_path}")

    router = AIRouter()

    try:
        reasoning = build_reasoning(router, evidence)
    except Exception as exc:
        print("\nERROR: Scientific reasoning failed.")
        print(str(exc))
        print("\nNo incomplete reasoning file was written.")
        return 1

    output = dict(evidence)
    output["ai_interpretation"] = reasoning

    raw_output = input(
        f"\nOutput file (default: {DEFAULT_OUTPUT.name}): "
    ).strip()
    output_path = resolve_project_path(raw_output, DEFAULT_OUTPUT)

    save_json(output_path, output)

    print("\n" + "=" * 80)
    print("SCIENTIFIC REASONING COMPLETE")
    print("=" * 80)
    print(f"Saved: {output_path}")
    print("All generated sections passed JSON/schema validation.")
    return 0


import sys  # noqa: E402

if __name__ == "__main__":
    # --help and closed / interrupted input exit cleanly instead of with a traceback
    import builtins as _bi
    if any(_a in ("-h", "--help") for _a in sys.argv[1:]):
        print("usage: python scientific_reasoner.py\n\nInteractive scientific reasoner: asks for a research question and reasons over stored evidence.\nPress Ctrl+C at any prompt to quit.")
        raise SystemExit(0)
    _real_input = _bi.input

    def _safe_input(prompt=""):
        try:
            return _real_input(prompt)
        except (EOFError, KeyboardInterrupt):
            print("\nNo input - exiting.")
            raise SystemExit(0)
    _bi.input = _safe_input
    raise SystemExit(main())
