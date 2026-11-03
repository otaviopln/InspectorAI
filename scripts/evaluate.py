"""Behavioral evaluation for InspectorAI's held-out dataset split.

This evaluator measures decision recognition and output-structure compliance.
It does not claim to measure semantic correctness or requirement preservation;
those require independent human review.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from inspector.engine import DATASET_PATH, generate_response, load_components

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "inspector" / "data" / "processed"
DECISIONS = ("CLARIFY", "FORMAT_RCFO")
PREDICTED_LABELS = (*DECISIONS, "UNKNOWN")
SECTION_LABELS = (
    "Decision",
    "Known requirements",
    "Missing information",
    "Questions",
    "Role",
    "Context",
    "Focus",
    "Output",
)
EVALUATOR_VERSION = "1.0"


def parse_decision(text: str) -> str:
    """Return the generated decision, or UNKNOWN if no valid marker exists."""
    match = re.search(
        r"(?im)^[ \t]*Decision[ \t]*:[ \t]*(CLARIFY|FORMAT_RCFO)\b",
        text,
    )
    return match.group(1).upper() if match else "UNKNOWN"


def extract_section(text: str, label: str) -> str:
    """Extract a named section up to the next recognized section heading."""
    alternatives = "|".join(re.escape(item) for item in SECTION_LABELS)
    pattern = re.compile(
        rf"(?ims)^[ \t]*{re.escape(label)}[ \t]*:[ \t]*(.*?)(?=^[ \t]*(?:{alternatives})[ \t]*:|\Z)"
    )
    match = pattern.search(text)
    return match.group(1).strip() if match else ""


def score_output(record: dict[str, Any], output: str) -> dict[str, Any]:
    """Score task decision and required output sections for one example."""
    task = str(record.get("task", "")).strip()
    ambiguity = str(record.get("ambiguity", "")).strip()
    if task == "clarify":
        expected_decision = "CLARIFY"
        required_sections = ["Questions"]
        if ambiguity == "partial":
            required_sections.extend(["Known requirements", "Missing information"])
    elif task == "format_rcfo":
        expected_decision = "FORMAT_RCFO"
        required_sections = ["Role", "Context", "Focus", "Output"]
    else:
        raise ValueError(f"Tarefa desconhecida no dataset: {task!r}")

    predicted_decision = parse_decision(output)
    section_status = {
        label: bool(extract_section(output, label))
        for label in required_sections
    }
    missing_sections = [
        label for label, present in section_status.items() if not present
    ]
    partial_sections_complete = None
    if ambiguity == "partial":
        partial_sections_complete = all(
            bool(extract_section(output, label))
            for label in ("Known requirements", "Missing information")
        )

    return {
        "id": record.get("id"),
        "task": task,
        "ambiguity": ambiguity,
        "expected_decision": expected_decision,
        "predicted_decision": predicted_decision,
        "decision_recognized": predicted_decision != "UNKNOWN",
        "decision_correct": predicted_decision == expected_decision,
        "required_sections": section_status,
        "missing_sections": missing_sections,
        "expected_structure_valid": (
            predicted_decision == expected_decision and not missing_sections
        ),
        "partial_sections_complete": partial_sections_complete,
    }


def _rate(numerator: int, denominator: int) -> float | None:
    return round(numerator / denominator, 4) if denominator else None


def summarize_scores(scores: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate accuracy, coverage, class metrics, and task structure rates."""
    total = len(scores)
    recognized = sum(bool(item["decision_recognized"]) for item in scores)
    correct = sum(bool(item["decision_correct"]) for item in scores)
    structurally_valid = sum(bool(item["expected_structure_valid"]) for item in scores)

    confusion = {
        expected: {predicted: 0 for predicted in PREDICTED_LABELS}
        for expected in DECISIONS
    }
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in scores:
        confusion[item["expected_decision"]][item["predicted_decision"]] += 1
        by_task[item["task"]].append(item)

    task_metrics = {}
    for task, items in sorted(by_task.items()):
        task_metrics[task] = {
            "examples": len(items),
            "decision_accuracy": _rate(
                sum(bool(item["decision_correct"]) for item in items), len(items)
            ),
            "expected_structure_validity": _rate(
                sum(bool(item["expected_structure_valid"]) for item in items),
                len(items),
            ),
        }

    class_metrics = {}
    for label in DECISIONS:
        true_positive = confusion[label][label]
        false_positive = sum(
            confusion[other][label] for other in DECISIONS if other != label
        )
        false_negative = sum(
            confusion[label][other] for other in PREDICTED_LABELS if other != label
        )
        precision = _rate(true_positive, true_positive + false_positive)
        recall = _rate(true_positive, true_positive + false_negative)
        f1 = (
            round(2 * precision * recall / (precision + recall), 4)
            if precision is not None and recall is not None and precision + recall
            else None
        )
        class_metrics[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": sum(confusion[label].values()),
        }

    partial_items = [item for item in scores if item["ambiguity"] == "partial"]
    partial_complete = sum(
        item["partial_sections_complete"] is True for item in partial_items
    )

    return {
        "examples": total,
        "decision_accuracy": _rate(correct, total),
        "decision_coverage": _rate(recognized, total),
        "expected_structure_validity": _rate(structurally_valid, total),
        "partial_sections_completeness": _rate(
            partial_complete, len(partial_items)
        ),
        "partial_examples": len(partial_items),
        "by_task": task_metrics,
        "class_metrics": class_metrics,
        "confusion_matrix": confusion,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate InspectorAI decisions and output structure on a held-out split."
    )
    parser.add_argument(
        "--split",
        choices=("train", "validation", "test"),
        default="test",
        help="Dataset split to evaluate (default: test).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional maximum number of examples, useful for a quick smoke test.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for evaluation_report.json and evaluation_predictions.jsonl.",
    )
    args = parser.parse_args()

    if args.limit is not None and args.limit < 1:
        parser.error("--limit precisa ser maior que zero.")

    # Load the model only after validating its tokenizer/dataset fingerprints.
    tokenizer, model = load_components()

    with DATASET_PATH.open("r", encoding="utf-8") as stream:
        dataset = json.load(stream)
    splits = dataset.get("splits", {})
    records = splits.get(args.split) if isinstance(splits, dict) else None
    if not isinstance(records, list) or not records:
        raise ValueError(f"O split {args.split!r} está vazio ou ausente.")

    if args.limit is not None:
        records = records[: args.limit]

    scored = []
    predictions = []
    for index, record in enumerate(records, start=1):
        error = None
        try:
            response = generate_response(model, tokenizer, record["input"])
        except Exception as exc:
            response = ""
            error = f"{type(exc).__name__}: {exc}"

        score = score_output(record, response)
        scored.append(score)
        predictions.append({**score, "response": response, "error": error})
        print(
            f"[{index}/{len(records)}] {record.get('id', 'sem_id')}: "
            f"{score['predicted_decision']}"
        )

    metrics = summarize_scores(scored)
    report = {
        "evaluator_version": EVALUATOR_VERSION,
        "split": args.split,
        "dataset_schema": dataset.get("schema_version"),
        "limited_run": args.limit is not None,
        "metrics": metrics,
        "interpretation": (
            "Automated structural metrics only. Semantic preservation, question "
            "relevance, and unsupported assumptions still require human review."
        ),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.output_dir / "evaluation_report.json"
    predictions_path = args.output_dir / "evaluation_predictions.jsonl"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    with predictions_path.open("w", encoding="utf-8") as stream:
        for prediction in predictions:
            stream.write(json.dumps(prediction, ensure_ascii=False) + "\n")

    print("\nBehavioral evaluation completed.")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"Report: {report_path}")
    print(f"Predictions: {predictions_path}")


if __name__ == "__main__":
    main()
