"""Tests for the runtime R.C.F.O. structural validator."""
import pytest

from inspector.rcfo import (
    format_rcfo_validation_error,
    is_rcfo_candidate,
    validate_rcfo_structure,
)

VALID = (
    "Decision: FORMAT_RCFO\n"
    "Role: SQL developer\n"
    "Context: A customer orders database.\n"
    "Focus: Retrieve the five most recent customer orders.\n"
    "Output: An SQL query.\n"
)


def test_accepts_complete_rcfo_response():
    result = validate_rcfo_structure(VALID)
    assert result["valid"] is True
    assert result["errors"] == []


@pytest.mark.parametrize("field", ["Role", "Context", "Focus", "Output"])
def test_rejects_each_missing_required_field(field):
    output = "\n".join(line for line in VALID.splitlines() if not line.startswith(field + ":"))
    result = validate_rcfo_structure(output)
    assert result["valid"] is False
    assert field in result["missing_fields"]


@pytest.mark.parametrize("field", ["Role", "Context", "Focus", "Output"])
def test_rejects_duplicate_required_field(field):
    line = next(line for line in VALID.splitlines() if line.startswith(field + ":"))
    result = validate_rcfo_structure(VALID + line + "\n")
    assert result["valid"] is False
    assert field in result["duplicate_fields"]


def test_rejects_empty_field():
    result = validate_rcfo_structure(VALID.replace("Context: A customer orders database.", "Context:"))
    assert result["empty_fields"] == ["Context"]
    assert result["valid"] is False


def test_rejects_wrong_field_order():
    output = (
        "Decision: FORMAT_RCFO\nRole: Analyst\nFocus: Retrieve orders.\n"
        "Context: An orders database.\nOutput: SQL.\n"
    )
    result = validate_rcfo_structure(output)
    assert result["valid"] is False
    assert result["order_valid"] is False


def test_rejects_missing_or_invalid_decision():
    result = validate_rcfo_structure(VALID.replace("Decision: FORMAT_RCFO\n", ""))
    assert result["valid"] is False
    assert result["decision_header_valid"] is False


def test_rejects_duplicate_decision_heading():
    output = VALID.replace(
        "Decision: FORMAT_RCFO\n",
        "Decision: FORMAT_RCFO\nDecision: FORMAT_RCFO\n",
    )
    result = validate_rcfo_structure(output)
    assert result["valid"] is False
    assert result["decision_header_valid"] is False


def test_rejects_clarify_sections_in_rcfo_response():
    result = validate_rcfo_structure(VALID + "Questions:\n1. Which database?\n")
    assert result["valid"] is False
    assert result["unexpected_sections"] == ["Questions"]


def test_rejects_unassigned_preamble():
    result = validate_rcfo_structure("Here is the result:\n" + VALID)
    assert result["valid"] is False
    assert "unassigned_text_before_sections" in result["errors"]


def test_detects_rcfo_candidate_from_decision_or_field():
    assert is_rcfo_candidate(VALID)
    assert is_rcfo_candidate("Role: Analyst\n")
    assert not is_rcfo_candidate("Decision: CLARIFY\nQuestions:\n1. Which database?\n")


def test_validation_error_is_actionable():
    result = validate_rcfo_structure("Decision: FORMAT_RCFO\nRole: Analyst\n")
    assert "campos obrigatórios ausentes" in format_rcfo_validation_error(result)
