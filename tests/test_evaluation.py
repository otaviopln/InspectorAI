from evaluate import parse_decision, score_output, summarize_scores


def test_parse_decision_returns_unknown_for_unrecognized_output():
    assert parse_decision("I need more information.") == "UNKNOWN"
    assert parse_decision("Decision: clarify\nQuestions:") == "CLARIFY"


def test_vague_clarification_requires_questions_section():
    record = {"id": "vague_1", "task": "clarify", "ambiguity": "vague"}
    valid = score_output(
        record,
        "Decision: CLARIFY\nQuestions:\n1. What result do you need?",
    )
    invalid = score_output(
        record,
        "Decision: CLARIFY\nI need more details.",
    )
    assert valid["expected_structure_valid"] is True
    assert invalid["expected_structure_valid"] is False
    assert invalid["missing_sections"] == ["Questions"]


def test_partial_clarification_requires_known_and_missing_sections():
    record = {"id": "partial_1", "task": "clarify", "ambiguity": "partial"}
    output = (
        "Decision: CLARIFY\n"
        "Known requirements:\n- Build a dashboard.\n"
        "Missing information:\n- Data source.\n"
        "Questions:\n1. Which data source should be used?"
    )
    result = score_output(record, output)
    assert result["expected_structure_valid"] is True
    assert result["partial_sections_complete"] is True


def test_rcfo_output_requires_all_four_nonempty_sections():
    record = {"id": "clear_1", "task": "format_rcfo", "ambiguity": "clear"}
    valid = (
        "Decision: FORMAT_RCFO\n"
        "Role: Analyst\n"
        "Context: A defined business task.\n"
        "Focus: Analyze the request.\n"
        "Output: A concise report."
    )
    invalid = (
        "Decision: FORMAT_RCFO\n"
        "Role: Analyst\n"
        "Context: A defined business task.\n"
        "Focus: Analyze the request."
    )
    assert score_output(record, valid)["expected_structure_valid"] is True
    assert score_output(record, invalid)["missing_sections"] == ["Output"]


def test_summary_counts_unknown_decisions_as_errors_and_reports_task_rates():
    scores = [
        score_output(
            {"id": "a", "task": "clarify", "ambiguity": "vague"},
            "Decision: CLARIFY\nQuestions:\n1. What is the goal?",
        ),
        score_output(
            {"id": "b", "task": "format_rcfo", "ambiguity": "clear"},
            "No structured output.",
        ),
    ]
    summary = summarize_scores(scores)
    assert summary["examples"] == 2
    assert summary["decision_accuracy"] == 0.5
    assert summary["decision_coverage"] == 0.5
    assert summary["expected_structure_validity"] == 0.5
    assert summary["confusion_matrix"]["FORMAT_RCFO"]["UNKNOWN"] == 1
