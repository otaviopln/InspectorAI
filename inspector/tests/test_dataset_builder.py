"""Regression test for the curated dataset preparation pipeline."""

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "prepare_dataset.py"
SPEC = importlib.util.spec_from_file_location("prepare_dataset_nested_test", SCRIPT)
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


def test_prepared_dataset_keeps_all_strata_in_each_split(tmp_path):
    rows = []
    for ambiguity in ("vague", "partial", "clear"):
        for index in range(3):
            task = "format_rcfo" if ambiguity == "clear" else "clarify"
            if ambiguity == "vague":
                target = "Decision: CLARIFY\nQuestions:\n1. What is the goal?"
            elif ambiguity == "partial":
                target = (
                    "Decision: CLARIFY\nKnown requirements:\n- Some details.\n"
                    "Missing information:\n- The scope.\nQuestions:\n1. What scope?"
                )
            else:
                target = (
                    "Decision: FORMAT_RCFO\nRole: Analyst\nContext: A task.\n"
                    "Focus: Analyze it.\nOutput: A report."
                )
            rows.append({
                "id": f"{ambiguity}_{index}",
                "task": task,
                "ambiguity": ambiguity,
                "language": "en",
                "input": f"Unique request {ambiguity} {index}",
                "target": target,
                "source": "unit_test",
                "group_id": f"{ambiguity}_group_{index}",
                "review_status": "approved",
            })

    source = tmp_path / "source.jsonl"
    source.write_text(
        "\n".join(json.dumps(row) for row in rows),
        encoding="utf-8",
    )
    prepared, report = module.prepare_dataset(
        source,
        tmp_path / "prepared_dataset.json",
        seed=42,
    )
    assert report["counts"]["total"] == 9
    for split in prepared["splits"].values():
        assert {row["ambiguity"] for row in split} == {"vague", "partial", "clear"}
