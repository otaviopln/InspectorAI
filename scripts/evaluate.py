"""Behavioral evaluation for InspectorAI.

The evaluator separates:
- decision classification;
- strict output/section integrity;
- annotated requirement-anchor coverage;
- explicitly forbidden phrase detection;
- manual semantic review.

Anchor and forbidden-phrase checks are deterministic screening aids. They
cannot prove semantic equivalence or detect every unsupported inference.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from inspector.engine import DATASET_PATH, generate_result, load_components
from inspector.rcfo import validate_rcfo_structure

DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "inspector" / "data" / "processed"
DEFAULT_SEMANTIC_CASES = PROJECT_ROOT / "tests" / "fixtures" / "semantic_evaluation_cases.jsonl"

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
SECTION_LOOKUP = {label.casefold(): label for label in SECTION_LABELS}
HEADER_RE = re.compile(
    r"^[ \t]*(Decision|Known requirements|Missing information|Questions|Role|Context|Focus|Output)[ \t]*:[ \t]*(.*)$",
    re.IGNORECASE,
)
QUESTION_RE = re.compile(r"^[ \t]*\d+[.)][ \t]*(.+?)[ \t]*$")
EVALUATOR_VERSION = "2.0"


def normalize_phrase(value: str) -> str:
    """Normalize Unicode, case, punctuation and whitespace for phrase checks."""
    value = unicodedata.normalize("NFKC", str(value)).casefold()
    value = re.sub(r"[^\w]+", " ", value, flags=re.UNICODE)
    return " ".join(value.split())


def contains_phrase(text: str, phrase: str) -> bool:
    """Match a normalized phrase as a sequence of whole normalized tokens."""
    normalized_text = normalize_phrase(text)
    normalized_phrase = normalize_phrase(phrase)
    if not normalized_phrase:
        return False
    return f" {normalized_phrase} " in f" {normalized_text} "


def parse_sections(text: str) -> dict[str, Any]:
    """Parse recognized headings, preserving duplicates and their original order."""
    sections: dict[str, list[str]] = defaultdict(list)
    order: list[str] = []
    current_label: str | None = None
    current_lines: list[str] = []

    def flush() -> None:
        if current_label is not None:
            sections[current_label].append("\n".join(current_lines).strip())

    for line in str(text).splitlines():
        match = HEADER_RE.match(line)
        if match:
            flush()
            current_label = SECTION_LOOKUP[match.group(1).casefold()]
            order.append(current_label)
            current_lines = [match.group(2).strip()] if match.group(2).strip() else []
        elif current_label is not None:
            current_lines.append(line.rstrip())

    flush()
    return {
        "sections": dict(sections),
        "order": order,
        "counts": {label: len(values) for label, values in sections.items()},
    }


def parse_decision(text: str) -> str:
    """Return one recognized decision; duplicates or malformed markers are UNKNOWN."""
    parsed = parse_sections(text)
    values = parsed["sections"].get("Decision", [])
    if len(values) != 1:
        return "UNKNOWN"
    match = re.match(r"^[ \t]*(CLARIFY|FORMAT_RCFO)\b", values[0], re.IGNORECASE)
    return match.group(1).upper() if match else "UNKNOWN"


def _question_diagnostics(body: str) -> dict[str, Any]:
    questions: list[str] = []
    unnumbered_nonempty: list[str] = []
    for line in body.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        match = QUESTION_RE.match(line)
        if match:
            questions.append(match.group(1).strip())
        elif stripped:
            unnumbered_nonempty.append(stripped)

    normalized = [normalize_phrase(question) for question in questions]
    duplicates = sorted({
        question for question, count in Counter(normalized).items()
        if question and count > 1
    })
    return {
        "numbered_count": len(questions),
        "duplicate_questions": duplicates,
        "unnumbered_lines": unnumbered_nonempty,
        "valid": bool(questions) and not duplicates,
    }


def _field_integrity(parsed: dict[str, Any], output: str) -> dict[str, Any]:
    """Delegate R.C.F.O. checks to the same validator used at inference time."""
    return validate_rcfo_structure(output)


def _semantic_screen(record: dict[str, Any], parsed: dict[str, Any], output: str) -> dict[str, Any]:
    """Run explicit, human-authored lexical anchors and forbidden-phrase checks."""
    annotations = record.get("semantic_checks")
    if not isinstance(annotations, dict):
        return {
            "available": False,
            "requirements_total": 0,
            "requirements_preserved": 0,
            "requirement_coverage": None,
            "missing_requirements": [],
            "forbidden_phrases": [],
            "forbidden_phrase_count": 0,
            "passed": None,
            "human_review_required": True,
        }

    selected_sections = annotations.get("search_sections")
    if isinstance(selected_sections, list) and selected_sections:
        text_parts = []
        for label in selected_sections:
            if label in parsed["sections"]:
                text_parts.extend(parsed["sections"][label])
        search_text = "\n".join(text_parts)
    else:
        search_text = output

    requirement_results = []
    for requirement in annotations.get("requirements", []):
        if not isinstance(requirement, dict):
            continue
        requirement_id = str(requirement.get("id", "unnamed"))
        concept_groups = requirement.get("concept_groups", [])
        missing_groups = []
        for index, aliases in enumerate(concept_groups):
            if isinstance(aliases, str):
                aliases = [aliases]
            if not isinstance(aliases, list) or not aliases:
                missing_groups.append(index)
                continue
            if not any(contains_phrase(search_text, str(alias)) for alias in aliases):
                missing_groups.append(index)
        requirement_results.append({
            "id": requirement_id,
            "preserved": not missing_groups,
            "missing_concept_groups": missing_groups,
            "concept_group_count": len(concept_groups),
        })

    forbidden_matches = [
        str(phrase)
        for phrase in annotations.get("forbidden_phrases", [])
        if isinstance(phrase, str) and contains_phrase(output, phrase)
    ]
    requirements_total = len(requirement_results)
    requirements_preserved = sum(
        bool(item["preserved"]) for item in requirement_results
    )
    requirements_with_concepts = sum(
        item["concept_group_count"] > 0 for item in requirement_results
    )
    preserved_with_concepts = sum(
        bool(item["preserved"]) and item["concept_group_count"] > 0
        for item in requirement_results
    )
    return {
        "available": True,
        "requirements_total": requirements_total,
        "requirements_preserved": requirements_preserved,
        "requirements_with_concepts": requirements_with_concepts,
        "preserved_with_concepts": preserved_with_concepts,
        "requirement_coverage": (
            round(preserved_with_concepts / requirements_with_concepts, 4)
            if requirements_with_concepts else None
        ),
        "missing_requirements": [
            item["id"] for item in requirement_results if not item["preserved"]
        ],
        "requirement_details": requirement_results,
        "forbidden_phrases": forbidden_matches,
        "forbidden_phrase_count": len(forbidden_matches),
        "passed": (
            not any(not item["preserved"] for item in requirement_results)
            and not forbidden_matches
        ),
        "human_review_required": True,
    }


def score_output(record: dict[str, Any], output: str) -> dict[str, Any]:
    """Score decision, structural integrity, question duplication and annotations."""
    task = str(record.get("task", "")).strip()
    ambiguity = str(record.get("ambiguity", "")).strip()
    if task == "clarify":
        expected_decision = "CLARIFY"
        required_sections = ["Questions"]
        expected_order = ["Decision", "Questions"]
        forbidden_sections = ["Role", "Context", "Focus", "Output"]
        if ambiguity == "partial":
            required_sections = ["Known requirements", "Missing information", "Questions"]
            expected_order = ["Decision", *required_sections]
    elif task == "format_rcfo":
        expected_decision = "FORMAT_RCFO"
        required_sections = ["Role", "Context", "Focus", "Output"]
        expected_order = ["Decision", *required_sections]
        forbidden_sections = ["Known requirements", "Missing information", "Questions"]
    else:
        raise ValueError(f"Tarefa desconhecida no dataset: {task!r}")

    parsed = parse_sections(output)
    sections = parsed["sections"]
    counts = parsed["counts"]
    predicted_decision = parse_decision(output)

    missing_sections = [
        label for label in required_sections
        if counts.get(label, 0) == 0 or (
            counts.get(label, 0) == 1 and not sections[label][0].strip()
        )
    ]
    duplicate_sections = [
        label for label, count in counts.items() if count > 1
    ]
    empty_sections = [
        label for label, values in sections.items()
        if any(not value.strip() for value in values)
    ]
    unexpected_sections = [
        label for label in forbidden_sections if counts.get(label, 0) > 0
    ]
    order_valid = parsed["order"] == expected_order

    questions_body = "\n".join(sections.get("Questions", []))
    question_diagnostics = _question_diagnostics(questions_body) if task == "clarify" else {
        "numbered_count": 0,
        "duplicate_questions": [],
        "unnumbered_lines": [],
        "valid": None,
    }
    field_integrity = (
        _field_integrity(parsed, output) if task == "format_rcfo"
        else {"applicable": False, "valid": None}
    )
    decision_occurrences = counts.get("Decision", 0)

    flags: list[str] = []
    if predicted_decision != expected_decision:
        flags.append("decision_mismatch")
    if decision_occurrences != 1:
        flags.append("decision_header_count_invalid")
    if missing_sections:
        flags.append("missing_or_empty_required_sections")
    if duplicate_sections:
        flags.append("duplicate_section_headers")
    if empty_sections:
        flags.append("empty_section_body")
    if unexpected_sections:
        flags.append("unexpected_task_sections")
    if not order_valid:
        flags.append("section_order_invalid")
    if task == "clarify" and not question_diagnostics["numbered_count"]:
        flags.append("no_numbered_questions")
    if question_diagnostics["duplicate_questions"]:
        flags.append("duplicate_questions")
    if task == "format_rcfo" and not field_integrity["valid"]:
        flags.append("rcfo_field_integrity_invalid")

    structure_valid = not flags
    semantic = _semantic_screen(record, parsed, output)

    return {
        "id": record.get("id"),
        "task": task,
        "ambiguity": ambiguity,
        "evaluation_source": record.get("_evaluation_source", "held_out_split"),
        "expected_decision": expected_decision,
        "predicted_decision": predicted_decision,
        "decision_recognized": predicted_decision != "UNKNOWN",
        "decision_correct": predicted_decision == expected_decision,
        "required_sections": {
            label: (
                counts.get(label, 0) == 1 and bool(sections[label][0].strip())
            )
            for label in required_sections
        },
        "missing_sections": missing_sections,
        "duplicate_sections": duplicate_sections,
        "empty_sections": empty_sections,
        "unexpected_sections": unexpected_sections,
        "section_order": parsed["order"],
        "section_order_valid": order_valid,
        "field_integrity": field_integrity,
        "question_diagnostics": question_diagnostics,
        "structure_flags": flags,
        "expected_structure_valid": structure_valid,
        "partial_sections_complete": (
            all(
                counts.get(label, 0) == 1 and bool(sections[label][0].strip())
                for label in ("Known requirements", "Missing information")
            )
            if ambiguity == "partial" else None
        ),
        "semantic_screen": semantic,
        "human_review_required": True,
    }


def _rate(numerator: int, denominator: int) -> float | None:
    return round(numerator / denominator, 4) if denominator else None


def summarize_scores(scores: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate decision, structural, R.C.F.O., and annotated screening metrics."""
    total = len(scores)
    recognized = sum(bool(item["decision_recognized"]) for item in scores)
    correct = sum(bool(item["decision_correct"]) for item in scores)
    structurally_valid = sum(bool(item["expected_structure_valid"]) for item in scores)

    confusion = {
        expected: {predicted: 0 for predicted in PREDICTED_LABELS}
        for expected in DECISIONS
    }
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in scores:
        confusion[item["expected_decision"]][item["predicted_decision"]] += 1
        by_task[item["task"]].append(item)
        by_source[item["evaluation_source"]].append(item)

    task_metrics = {}
    for task, items in sorted(by_task.items()):
        task_metrics[task] = {
            "examples": len(items),
            "decision_accuracy": _rate(
                sum(bool(item["decision_correct"]) for item in items), len(items)
            ),
            "strict_structure_validity": _rate(
                sum(bool(item["expected_structure_valid"]) for item in items),
                len(items),
            ),
        }

    source_metrics = {}
    for source, items in sorted(by_source.items()):
        source_metrics[source] = {
            "examples": len(items),
            "decision_accuracy": _rate(
                sum(bool(item["decision_correct"]) for item in items), len(items)
            ),
            "strict_structure_validity": _rate(
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
    rcfo_items = [
        item for item in scores if item["task"] == "format_rcfo"
    ]
    rcfo_valid = sum(
        item["field_integrity"].get("valid") is True for item in rcfo_items
    )
    clarify_items = [item for item in scores if item["task"] == "clarify"]
    duplicate_question_examples = sum(
        bool(item["question_diagnostics"]["duplicate_questions"])
        for item in clarify_items
    )
    numbered_question_examples = sum(
        bool(item["question_diagnostics"]["numbered_count"])
        for item in clarify_items
    )

    annotated = [
        item for item in scores if item["semantic_screen"]["available"]
    ]
    requirements_total = sum(
        item["semantic_screen"].get("requirements_with_concepts", 0)
        for item in annotated
    )
    requirements_preserved = sum(
        item["semantic_screen"].get("preserved_with_concepts", 0)
        for item in annotated
    )
    forbidden_examples = sum(
        item["semantic_screen"]["forbidden_phrase_count"] > 0
        for item in annotated
    )
    forbidden_matches = sum(
        item["semantic_screen"]["forbidden_phrase_count"] for item in annotated
    )
    semantic_passes = sum(
        item["semantic_screen"]["passed"] is True for item in annotated
    )

    return {
        "examples": total,
        "decision_accuracy": _rate(correct, total),
        "decision_coverage": _rate(recognized, total),
        "strict_structure_validity": _rate(structurally_valid, total),
        "expected_structure_validity": _rate(structurally_valid, total),
        "partial_sections_completeness": _rate(
            partial_complete, len(partial_items)
        ),
        "partial_examples": len(partial_items),
        "rcfo_field_integrity": {
            "examples": len(rcfo_items),
            "valid_examples": rcfo_valid,
            "validity_rate": _rate(rcfo_valid, len(rcfo_items)),
        },
        "clarification_questions": {
            "examples": len(clarify_items),
            "with_numbered_questions": numbered_question_examples,
            "with_duplicate_questions": duplicate_question_examples,
            "numbered_question_rate": _rate(numbered_question_examples, len(clarify_items)),
            "duplicate_question_rate": _rate(duplicate_question_examples, len(clarify_items)),
        },
        "annotated_semantic_screen": {
            "annotated_examples": len(annotated),
            "unannotated_examples": total - len(annotated),
            "requirements_total": requirements_total,
            "requirements_preserved": requirements_preserved,
            "requirement_anchor_coverage": _rate(
                requirements_preserved, requirements_total
            ),
            "cases_passing_screen": semantic_passes,
            "cases_with_forbidden_phrases": forbidden_examples,
            "forbidden_phrase_matches": forbidden_matches,
            "screen_pass_rate": _rate(semantic_passes, len(annotated)),
            "human_review_still_required": True,
        },
        "human_review_pending": total,
        "by_task": task_metrics,
        "by_source": source_metrics,
        "class_metrics": class_metrics,
        "confusion_matrix": confusion,
    }


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Arquivo JSONL não encontrado: {path}")
    records = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"JSON inválido em {path}, linha {line_number}: {exc}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"Registro não é um objeto em {path}, linha {line_number}.")
            records.append(record)
    return records


def _review_queue_item(record: dict[str, Any], prediction: dict[str, Any]) -> dict[str, Any]:
    """Create a blank, editable human-review record for each generated answer."""
    task = record.get("task")
    return {
        "id": record.get("id"),
        "input": record.get("input", ""),
        "response": prediction.get("response", ""),
        "evaluation_source": prediction.get("evaluation_source"),
        "automatic_flags": prediction.get("structure_flags", []),
        "semantic_screen": prediction.get("semantic_screen", {}),
        "human_review": {
            "intent_preserved": None,
            "user_requirements_preserved": None,
            "unsupported_content_absent": None,
            "semantic_fidelity": None,
            "rcfo_fields_faithful": None if task == "format_rcfo" else "not_applicable",
            "clarification_questions_relevant": None if task == "clarify" else "not_applicable",
            "notes": "",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate decision accuracy, strict structure, annotated requirement "
            "anchors, forbidden phrases, and human-review needs."
        )
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
        help="Optional maximum number of held-out records; semantic cases are added separately.",
    )
    parser.add_argument(
        "--semantic-cases",
        type=Path,
        default=DEFAULT_SEMANTIC_CASES,
        help="JSONL file with manually annotated benchmark prompts.",
    )
    parser.add_argument(
        "--skip-semantic-cases",
        action="store_true",
        help="Do not append the annotated semantic benchmark.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=(
            "Directory for evaluation_report.json, evaluation_predictions.jsonl "
            "and evaluation_review_queue.jsonl."
        ),
    )
    args = parser.parse_args()

    if args.limit is not None and args.limit < 1:
        parser.error("--limit precisa ser maior que zero.")

    with DATASET_PATH.open("r", encoding="utf-8") as stream:
        dataset = json.load(stream)
    splits = dataset.get("splits", {})
    dataset_records = splits.get(args.split) if isinstance(splits, dict) else None
    if not isinstance(dataset_records, list) or not dataset_records:
        raise ValueError(f"O split {args.split!r} está vazio ou ausente.")
    if args.limit is not None:
        dataset_records = dataset_records[: args.limit]

    records = [
        {**record, "_evaluation_source": "held_out_split"}
        for record in dataset_records
    ]
    semantic_records: list[dict[str, Any]] = []
    if not args.skip_semantic_cases and args.semantic_cases.is_file():
        semantic_records = [
            {**record, "_evaluation_source": "semantic_suite"}
            for record in _load_jsonl(args.semantic_cases)
        ]
        records.extend(semantic_records)

    ids = [str(record.get("id", "")) for record in records]
    if any(not item for item in ids):
        raise ValueError("Todos os registros avaliados precisam ter um id.")
    duplicate_ids = sorted(
        item for item, count in Counter(ids).items() if count > 1
    )
    if duplicate_ids:
        raise ValueError(f"IDs duplicados entre os conjuntos avaliados: {duplicate_ids}")

    # Checkpoint, tokenizer and prepared dataset are validated before inference.
    tokenizer, model = load_components(validate_dataset=True)

    scored = []
    predictions = []
    review_queue = []
    for index, record in enumerate(records, start=1):
        error = None
        generation = None
        try:
            generation = generate_result(model, tokenizer, record["input"])
            response = (
                generation.text
                if generation.status in {"ok", "invalid_structure"}
                else ""
            )
            if generation.status != "ok":
                error = generation.status
        except Exception as exc:
            response = ""
            error = f"{type(exc).__name__}: {exc}"

        score = score_output(record, response)
        score["evaluation_source"] = record["_evaluation_source"]
        scored.append(score)
        prediction = {
            **score,
            "input": record["input"],
            "response": response,
            "error": error,
            "generation_status": generation.status if generation else "exception",
            "generation_finished": generation.finished if generation else False,
            "input_tokens": generation.input_tokens if generation else None,
            "max_input_tokens": generation.max_input_tokens if generation else None,
        }
        predictions.append(prediction)
        review_queue.append(_review_queue_item(record, prediction))
        print(
            f"[{index}/{len(records)}] {record.get('id', 'sem_id')}: "
            f"{score['predicted_decision']} | "
            f"structure={'OK' if score['expected_structure_valid'] else 'FAIL'}"
        )

    metrics = summarize_scores(scored)
    metrics["generation_failures"] = sum(
        prediction["generation_status"] not in {"ok", "invalid_structure"}
        for prediction in predictions
    )
    metrics["structural_rejections"] = sum(
        prediction["generation_status"] == "invalid_structure"
        for prediction in predictions
    )
    statuses = sorted({
        prediction["generation_status"] for prediction in predictions
    })
    metrics["generation_statuses"] = {
        status: sum(
            prediction["generation_status"] == status
            for prediction in predictions
        )
        for status in statuses
    }
    report = {
        "evaluator_version": EVALUATOR_VERSION,
        "split": args.split,
        "dataset_schema": dataset.get("schema_version"),
        "dataset_examples": len(dataset_records),
        "semantic_suite_examples": len(semantic_records),
        "limited_run": args.limit is not None,
        "metrics": metrics,
        "interpretation": (
            "Strict structure checks verify headings, counts, non-empty sections, "
            "ordering and duplicate clarification questions. Requirement coverage "
            "uses manually annotated lexical concept groups; forbidden-phrase checks "
            "only detect explicitly listed phrases. These screens do not prove "
            "semantic equivalence or detect every unsupported assumption. Review "
            "evaluation_review_queue.jsonl manually before claiming semantic quality."
        ),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.output_dir / "evaluation_report.json"
    predictions_path = args.output_dir / "evaluation_predictions.jsonl"
    review_path = args.output_dir / "evaluation_review_queue.jsonl"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    with predictions_path.open("w", encoding="utf-8") as stream:
        for prediction in predictions:
            stream.write(json.dumps(prediction, ensure_ascii=False) + "\n")
    with review_path.open("w", encoding="utf-8") as stream:
        for item in review_queue:
            stream.write(json.dumps(item, ensure_ascii=False) + "\n")

    print("\nBehavioral evaluation completed.")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"Report: {report_path}")
    print(f"Predictions: {predictions_path}")
    print(f"Human review queue: {review_path}")


if __name__ == "__main__":
    main()
