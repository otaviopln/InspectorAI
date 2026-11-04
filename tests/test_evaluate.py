"""Regression tests for semantic and structural InspectorAI evaluation."""

from scripts.evaluate import score_output, summarize_scores


def _record(task="format_rcfo", ambiguity="clear", **extra):
    return {
        "id": "case-1",
        "task": task,
        "ambiguity": ambiguity,
        **extra,
    }


def test_rcfo_requires_each_field_exactly_once_and_in_order():
    output = (
        "Decision: FORMAT_RCFO\n"
        "Role: SQL developer\n"
        "Context: A customer database.\n"
        "Focus: Retrieve the five most recent customer orders.\n"
        "Output: An SQL query.\n"
    )
    score = score_output(_record(), output)
    assert score["expected_structure_valid"] is True
    assert score["field_integrity"]["valid"] is True
    assert score["field_integrity"]["field_counts"] == {
        "Role": 1, "Context": 1, "Focus": 1, "Output": 1
    }


def test_rcfo_detects_missing_output_field():
    output = (
        "Decision: FORMAT_RCFO\n"
        "Role: SQL developer\n"
        "Context: A customer database.\n"
        "Focus: Retrieve customer orders.\n"
    )
    score = score_output(_record(), output)
    assert score["expected_structure_valid"] is False
    assert score["field_integrity"]["missing_fields"] == ["Output"]
    assert "rcfo_field_integrity_invalid" in score["structure_flags"]


def test_rcfo_detects_duplicate_context_heading():
    output = (
        "Decision: FORMAT_RCFO\n"
        "Role: Data analyst\n"
        "Context: A Python developer.\n"
        "Context: A Python script.\n"
        "Focus: Write an email.\n"
        "Output: An email.\n"
    )
    score = score_output(_record(), output)
    assert score["field_integrity"]["duplicate_fields"] == ["Context"]
    assert "duplicate_section_headers" in score["structure_flags"]
    assert score["expected_structure_valid"] is False


def test_rcfo_detects_empty_field_body():
    output = (
        "Decision: FORMAT_RCFO\n"
        "Role: Data analyst\n"
        "Context:\n"
        "Focus: Summarize sales.\n"
        "Output: A report.\n"
    )
    score = score_output(_record(), output)
    assert score["field_integrity"]["empty_fields"] == ["Context"]
    assert score["expected_structure_valid"] is False


def test_rcfo_detects_fields_in_wrong_order():
    output = (
        "Decision: FORMAT_RCFO\n"
        "Role: Data analyst\n"
        "Focus: Summarize sales.\n"
        "Context: A sales database.\n"
        "Output: A report.\n"
    )
    score = score_output(_record(), output)
    assert score["field_integrity"]["order_valid"] is False
    assert score["expected_structure_valid"] is False


def test_rcfo_rejects_clarification_only_sections():
    output = (
        "Decision: FORMAT_RCFO\n"
        "Role: Data analyst\n"
        "Context: A sales database.\n"
        "Focus: Summarize sales.\n"
        "Output: A report.\n"
        "Questions:\n"
        "1. Which year?\n"
    )
    score = score_output(_record(), output)
    assert "unexpected_task_sections" in score["structure_flags"]


def test_clarify_partial_requires_known_missing_and_questions_sections():
    output = (
        "Decision: CLARIFY\n"
        "Known requirements:\n"
        "- Build a mobile app.\n"
        "Missing information:\n"
        "- Preferred platform.\n"
        "Questions:\n"
        "1. Which platform should it support?\n"
    )
    score = score_output(_record("clarify", "partial"), output)
    assert score["expected_structure_valid"] is True
    assert score["partial_sections_complete"] is True


def test_clarify_detects_duplicate_questions():
    output = (
        "Decision: CLARIFY\n"
        "Questions:\n"
        "1. Which database are you using?\n"
        "2. Which database are you using?\n"
    )
    score = score_output(_record("clarify", "vague"), output)
    assert score["question_diagnostics"]["duplicate_questions"]
    assert "duplicate_questions" in score["structure_flags"]
    assert score["expected_structure_valid"] is False


def test_clarify_requires_numbered_questions():
    output = (
        "Decision: CLARIFY\n"
        "Questions:\n"
        "Which database are you using?\n"
    )
    score = score_output(_record("clarify", "vague"), output)
    assert "no_numbered_questions" in score["structure_flags"]


def test_requirement_anchor_groups_accept_curated_synonyms():
    record = _record(
        semantic_checks={
            "search_sections": ["Focus", "Output"],
            "requirements": [
                {
                    "id": "five-most-recent-customer-orders",
                    "concept_groups": [
                        ["five", "5"],
                        ["most recent", "latest", "newest"],
                        ["customer orders", "orders from customers"],
                    ],
                }
            ],
            "forbidden_phrases": ["Python developer", "write an email"],
        }
    )
    output = (
        "Decision: FORMAT_RCFO\n"
        "Role: Database analyst\n"
        "Context: An orders database.\n"
        "Focus: Retrieve the latest orders from customers, limited to 5.\n"
        "Output: An SQL query.\n"
    )
    score = score_output(record, output)
    screen = score["semantic_screen"]
    assert screen["available"] is True
    assert screen["requirement_coverage"] == 1.0
    assert screen["missing_requirements"] == []
    assert screen["passed"] is True


def test_requirement_screen_detects_omitted_user_requirements():
    record = _record(
        semantic_checks={
            "requirements": [
                {
                    "id": "customer-orders-goal",
                    "concept_groups": [
                        ["customer orders", "orders from customers"],
                        ["five", "5"],
                        ["most recent", "latest", "newest"],
                    ],
                }
            ]
        }
    )
    output = (
        "Decision: FORMAT_RCFO\n"
        "Role: Data analyst\n"
        "Context: A database.\n"
        "Focus: Write an email.\n"
        "Output: An email.\n"
    )
    screen = score_output(record, output)["semantic_screen"]
    assert screen["requirements_preserved"] == 0
    assert screen["requirement_coverage"] == 0.0
    assert screen["missing_requirements"] == ["customer-orders-goal"]


def test_forbidden_phrase_screen_detects_explicitly_unsupported_content():
    record = _record(
        semantic_checks={
            "requirements": [],
            "forbidden_phrases": ["Python developer", "Python script", "write an email"],
        }
    )
    output = (
        "Decision: FORMAT_RCFO\n"
        "Role: Data analyst\n"
        "Context: A Python developer.\n"
        "Focus: Write an email.\n"
        "Output: A Python script.\n"
    )
    screen = score_output(record, output)["semantic_screen"]
    assert screen["forbidden_phrase_count"] == 3
    assert screen["passed"] is False


def test_unannotated_output_is_marked_for_human_review():
    score = score_output(_record(), "Decision: FORMAT_RCFO")
    assert score["semantic_screen"]["available"] is False
    assert score["semantic_screen"]["human_review_required"] is True
    assert score["human_review_required"] is True


def test_duplicate_decision_heading_is_not_accepted():
    output = (
        "Decision: FORMAT_RCFO\n"
        "Decision: FORMAT_RCFO\n"
        "Role: Analyst\nContext: A database.\n"
        "Focus: Retrieve orders.\nOutput: SQL.\n"
    )
    score = score_output(_record(), output)
    assert score["predicted_decision"] == "UNKNOWN"
    assert score["decision_correct"] is False
    assert "decision_header_count_invalid" in score["structure_flags"]


def test_summary_reports_field_and_semantic_screen_metrics():
    record = _record(
        semantic_checks={
            "requirements": [
                {"id": "goal", "concept_groups": [["customer orders"]]}
            ],
            "forbidden_phrases": ["write an email"],
        }
    )
    output = (
        "Decision: FORMAT_RCFO\n"
        "Role: Database analyst\n"
        "Context: A customer orders database.\n"
        "Focus: Retrieve customer orders.\n"
        "Output: A query.\n"
    )
    score = score_output(record, output)
    metrics = summarize_scores([score])
    assert metrics["rcfo_field_integrity"]["validity_rate"] == 1.0
    assert metrics["annotated_semantic_screen"]["requirement_anchor_coverage"] == 1.0
    assert metrics["human_review_pending"] == 1
