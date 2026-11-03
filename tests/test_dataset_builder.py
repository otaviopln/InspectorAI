"""Regression tests for the new curated dataset pipeline.

The former combinatorial build_dataset() API is intentionally retired.
"""

import json
from pathlib import Path
import importlib.util

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "prepare_dataset.py"
SPEC = importlib.util.spec_from_file_location("prepare_dataset_builder", SCRIPT)
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


def _record(record_id, ambiguity, group_id):
    if ambiguity == "vague":
        task = "clarify"
        target = (
            "Decision: CLARIFY\nQuestions:\n"
            "1. What result do you need?\n2. What information is available?"
        )
    elif ambiguity == "partial":
        task = "clarify"
        target = (
            "Decision: CLARIFY\nKnown requirements:\n- A known task.\n"
            "Missing information:\n- The intended scope.\n"
            "Questions:\n1. What scope should be included?"
        )
    else:
        task = "format_rcfo"
        target = (
            "Decision: FORMAT_RCFO\nRole: Analyst\nContext: A defined task.\n"
            "Focus: Complete the requested analysis.\nOutput: A concise report."
        )
    return {
        "id": record_id,
        "task": task,
        "ambiguity": ambiguity,
        "language": "en",
        "input": f"Perform a unique request for {record_id}.",
        "target": target,
        "source": "unit_test",
        "group_id": group_id,
        "review_status": "approved",
    }


def test_new_schema_requires_all_three_ambiguity_types():
    rows = [
        _record(f"{kind}_{i}", kind, f"{kind}_{i}")
        for kind in ("vague", "partial", "clear")
        for i in range(3)
    ]
    splits = module.grouped_stratified_split(rows, seed=7)
    for split_name in ("train", "validation", "test"):
        observed = {row["ambiguity"] for row in splits[split_name]}
        assert observed == {"vague", "partial", "clear"}


def test_jsonl_is_validated_and_prepared(tmp_path):
    rows = [
        _record(f"{kind}_{i}", kind, f"{kind}_{i}")
        for kind in ("vague", "partial", "clear")
        for i in range(3)
    ]
    source = tmp_path / "curated.jsonl"
    source.write_text(
        "\n".join(json.dumps(row) for row in rows),
        encoding="utf-8",
    )
    prepared_path = tmp_path / "processed" / "prepared_dataset.json"
    prepared, report = module.prepare_dataset(source, prepared_path, seed=7)
    assert prepared_path.exists()
    assert report["counts"]["total"] == 9
    assert set(prepared["splits"]) == {"train", "validation", "test"}
