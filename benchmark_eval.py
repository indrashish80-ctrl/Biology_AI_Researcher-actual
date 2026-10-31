"""
benchmark_eval.py  (v24.2)
--------------------------
Public benchmark scorer compatible with Evidence Inference format.

Reads JSONL of {question, evidence_text, intervention, outcome, direction, direction_certainty, evidence_source}
and scores the engine's output against these gold labels.

Usage:
  python benchmark_eval.py benchmark.jsonl output_dir/

Output: JSON report with accuracy, precision, recall, F1 by score level.
"""
import json
from pathlib import Path
from typing import Any, Dict, List
from collections import Counter


def load_benchmark(path: str) -> List[Dict[str, Any]]:
    """Load JSONL benchmark file."""
    cases = []
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    cases.append(json.loads(line))
    except FileNotFoundError:
        return []
    return cases


def evaluate(cases: List[Dict[str, Any]], engine_results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """
    Score engine output against benchmark.

    Args:
        cases: List of benchmark cases (with 'question', 'intervention', 'outcome', 'direction', 'evidence_source')
        engine_results: Dict mapping case_id to {direction, certainty, evidence_source, ...}

    Returns:
        Evaluation report with metrics by certainty level.
    """
    metrics = {
        "total": len(cases),
        "correct": 0,
        "by_certainty": {},
        "confusion": Counter(),
    }

    for case in cases:
        case_id = case.get("id") or f"{case.get('question', '')}_{case.get('intervention', '')}"
        gold_direction = case.get("direction")  # "increase", "decrease", "no_effect"
        gold_certainty = case.get("direction_certainty")  # 0-5 scale or low/moderate/high

        result = engine_results.get(case_id, {})
        pred_direction = result.get("direction")
        pred_certainty = result.get("certainty")  # 0-1 scale

        # Convert scales if needed
        if isinstance(pred_certainty, float):
            cert_level = "high" if pred_certainty > 0.66 else "moderate" if pred_certainty > 0.33 else "low"
        else:
            cert_level = pred_certainty or "unknown"

        # Score match
        is_correct = pred_direction == gold_direction and pred_certainty == gold_certainty
        if is_correct:
            metrics["correct"] += 1

        metrics["confusion"][(gold_direction, pred_direction)] += 1

        # Track by certainty level
        if cert_level not in metrics["by_certainty"]:
            metrics["by_certainty"][cert_level] = {"total": 0, "correct": 0}
        metrics["by_certainty"][cert_level]["total"] += 1
        if is_correct:
            metrics["by_certainty"][cert_level]["correct"] += 1

    # Calculate rates
    metrics["accuracy"] = metrics["correct"] / metrics["total"] if metrics["total"] > 0 else 0.0
    for level in metrics["by_certainty"]:
        tot = metrics["by_certainty"][level]["total"]
        correct = metrics["by_certainty"][level]["correct"]
        metrics["by_certainty"][level]["accuracy"] = correct / tot if tot > 0 else 0.0

    return metrics


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("benchmark_file", help="JSONL benchmark file")
    ap.add_argument("--results", help="JSON results file to score against")
    ap.add_argument("--output", default="benchmark_report.json", help="Output report path")
    args = ap.parse_args()

    cases = load_benchmark(args.benchmark_file)
    results = {}
    if args.results:
        with open(args.results, encoding="utf-8") as f:
            results = json.load(f)

    report = evaluate(cases, results)
    Path(args.output).write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(f"Benchmark evaluation: {report['correct']}/{report['total']} correct ({report['accuracy']*100:.1f}%)")
