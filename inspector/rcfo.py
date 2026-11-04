"""Strict structural validation for InspectorAI R.C.F.O. responses."""
from __future__ import annotations

import re
from collections import defaultdict
from typing import Any

RCFO_FIELDS = ("Role", "Context", "Focus", "Output")
CLARIFY_FIELDS = ("Known requirements", "Missing information", "Questions")
ALL_HEADINGS = ("Decision", *CLARIFY_FIELDS, *RCFO_FIELDS)
_LOOKUP = {label.casefold(): label for label in ALL_HEADINGS}
_HEADER = re.compile(
    r"^[ \t]*(Decision|Known requirements|Missing information|Questions|Role|Context|Focus|Output)[ \t]*:[ \t]*(.*)$",
    re.IGNORECASE,
)
_DECISION = re.compile(r"^FORMAT_RCFO[ \t]*$", re.IGNORECASE)


def _parse(text: str) -> dict[str, Any]:
    sections: dict[str, list[str]] = defaultdict(list)
    order: list[str] = []
    preamble: list[str] = []
    current: str | None = None
    body: list[str] = []

    def flush() -> None:
        if current is not None:
            sections[current].append("\n".join(body).strip())

    for line in str(text).splitlines():
        match = _HEADER.match(line)
        if match:
            flush()
            current = _LOOKUP[match.group(1).casefold()]
            order.append(current)
            body = [match.group(2).strip()] if match.group(2).strip() else []
        elif current is None:
            if line.strip():
                preamble.append(line.strip())
        else:
            body.append(line.rstrip())
    flush()
    return {
        "sections": dict(sections),
        "order": order,
        "counts": {key: len(value) for key, value in sections.items()},
        "preamble": preamble,
    }


def is_rcfo_candidate(text: str) -> bool:
    parsed = _parse(text)
    sections = parsed["sections"]
    return (
        any(_DECISION.fullmatch(value.strip()) for value in sections.get("Decision", []))
        or any(sections.get(field) for field in RCFO_FIELDS)
    )


def validate_rcfo_structure(text: str) -> dict[str, Any]:
    parsed = _parse(text)
    sections, counts, order = parsed["sections"], parsed["counts"], parsed["order"]
    missing = [field for field in RCFO_FIELDS if counts.get(field, 0) == 0]
    duplicates = [field for field in RCFO_FIELDS if counts.get(field, 0) > 1]
    empty = [
        field for field in RCFO_FIELDS
        if counts.get(field, 0) == 1 and not sections[field][0].strip()
    ]
    decision_values = sections.get("Decision", [])
    decision_valid = (
        len(decision_values) == 1
        and bool(_DECISION.fullmatch(decision_values[0].strip()))
        and bool(order)
        and order[0] == "Decision"
    )
    field_order = [label for label in order if label in RCFO_FIELDS]
    order_valid = field_order == list(RCFO_FIELDS)
    unexpected = [label for label in CLARIFY_FIELDS if counts.get(label, 0) > 0]
    duplicate_headings = [label for label, count in counts.items() if count > 1]

    errors: list[str] = []
    if not decision_valid:
        errors.append("decision_header_invalid")
    if missing:
        errors.append("missing_fields")
    if duplicates:
        errors.append("duplicate_fields")
    if empty:
        errors.append("empty_fields")
    if not order_valid:
        errors.append("field_order_invalid")
    if unexpected:
        errors.append("unexpected_clarification_sections")
    if parsed["preamble"]:
        errors.append("unassigned_text_before_sections")
    if duplicate_headings:
        errors.append("duplicate_headings")

    return {
        "applicable": True,
        "valid": not errors,
        "expected_fields": list(RCFO_FIELDS),
        "field_counts": {field: counts.get(field, 0) for field in RCFO_FIELDS},
        "missing_fields": missing,
        "duplicate_fields": duplicates,
        "empty_fields": empty,
        "field_order": field_order,
        "order_valid": order_valid,
        "unexpected_sections": unexpected,
        "decision_header_valid": decision_valid,
        "preamble_lines": parsed["preamble"],
        "errors": errors,
    }


def format_rcfo_validation_error(validation: dict[str, Any]) -> str:
    labels = {
        "decision_header_invalid": "cabeçalho Decision inválido ou ausente",
        "missing_fields": "campos obrigatórios ausentes",
        "duplicate_fields": "campos R.C.F.O. duplicados",
        "empty_fields": "campos sem conteúdo",
        "field_order_invalid": "ordem dos campos incorreta",
        "unexpected_clarification_sections": "seções CLARIFY em resposta R.C.F.O.",
        "unassigned_text_before_sections": "texto fora das seções",
        "duplicate_headings": "cabeçalhos duplicados",
    }
    return "; ".join(labels.get(item, item) for item in validation.get("errors", [])) or "estrutura inválida"
