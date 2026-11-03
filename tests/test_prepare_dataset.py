import importlib.util
import json
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "prepare_dataset.py"
SPEC = importlib.util.spec_from_file_location("prepare_dataset", SCRIPT)
prepare_module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(prepare_module)


def make_record(record_id, task, ambiguity, group_id, status="approved"):
    if ambiguity == "vague":
        target = (
            "Decision: CLARIFY\nQuestions:\n"
            "1. What is the intended goal?\n"
            "2. What information can you provide?"
        )
        prompt = f"Please help with task {record_id}."
    elif ambiguity == "partial":
        target = (
            "Decision: CLARIFY\nKnown requirements:\n- The user specified a task.\n"
            "Missing information:\n- The desired scope.\n"
            "Questions:\n1. What scope is required?"
        )
        prompt = f"Please partially specify task {record_id}."
    else:
        target = (
            "Decision: FORMAT_RCFO\nRole: Analyst\nContext: A defined task.\n"
            "Focus: Complete the stated analysis.\nOutput: A concise report."
        )
        prompt = f"Perform the defined task {record_id} and return a concise report."

    return {
        "id": record_id,
        "task": "format_rcfo" if ambiguity == "clear" else "clarify",
        "ambiguity": ambiguity,
        "language": "en",
        "input": prompt,
        "target": target,
        "source": "unit_test",
        "group_id": group_id,
        "review_status": status,
    }


def write_jsonl(path, rows):
    path.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )



def test_prepared_dataset_is_deterministic_across_python_hash_seeds(tmp_path):
    import os
    import subprocess
    import sys

    rows = [
        make_record(f"{kind}_{index}", "", kind, f"{kind}_family_{index}")
        for kind in ("vague", "partial", "clear")
        for index in range(8)
    ]
    source = tmp_path / "curated.jsonl"
    write_jsonl(source, rows)

    script_dir = Path(__file__).resolve().parents[1] / "scripts"
    code = (
        "import sys; "
        "from pathlib import Path; "
        "sys.path.insert(0, sys.argv[3]); "
        "from prepare_dataset import prepare_dataset; "
        "prepare_dataset(Path(sys.argv[1]), Path(sys.argv[2]), seed=42)"
    )

    outputs = []
    for hash_seed in ("7", "98765"):
        output = tmp_path / f"prepared_{hash_seed}.json"
        env = os.environ.copy()
        env["PYTHONHASHSEED"] = hash_seed
        subprocess.run(
            [sys.executable, "-c", code, str(source), str(output), str(script_dir)],
            check=True,
            capture_output=True,
            text=True,
            env=env,
        )
        outputs.append(output.read_bytes())

    assert outputs[0] == outputs[1]


def test_source_fingerprint_ignores_json_whitespace_and_line_endings(tmp_path):
    rows = [
        make_record(f"{kind}_{index}", "", kind, f"{kind}_family_{index}")
        for kind in ("vague", "partial", "clear")
        for index in range(6)
    ]
    compact = tmp_path / "compact.jsonl"
    formatted = tmp_path / "formatted.jsonl"

    compact.write_bytes(
        ("\n".join(json.dumps(row, separators=(",", ":")) for row in rows) + "\n").encode("utf-8")
    )
    formatted.write_bytes(
        ("\r\n".join(json.dumps(row, ensure_ascii=False, indent=2) for row in rows) + "\r\n").encode("utf-8")
    )

    compact_prepared, _ = prepare_module.prepare_dataset(
        compact, tmp_path / "compact" / "prepared_dataset.json", seed=42
    )
    formatted_prepared, _ = prepare_module.prepare_dataset(
        formatted, tmp_path / "formatted" / "prepared_dataset.json", seed=42
    )

    assert compact_prepared["source_sha256"] == formatted_prepared["source_sha256"]
    assert compact_prepared["splits"] == formatted_prepared["splits"]


def test_grouped_split_has_no_group_leakage():
    rows = []
    for category in ("vague", "partial", "clear"):
        for index in range(6):
            rows.append(
                make_record(
                    f"{category}_{index}",
                    "",
                    category,
                    f"{category}_family_{index}",
                )
            )

    splits = prepare_module.grouped_stratified_split(rows, seed=123)
    group_sets = {
        name: {row["group_id"] for row in values}
        for name, values in splits.items()
    }

    assert group_sets["train"].isdisjoint(group_sets["validation"])
    assert group_sets["train"].isdisjoint(group_sets["test"])
    assert group_sets["validation"].isdisjoint(group_sets["test"])
    assert all(splits[name] for name in splits)


def test_pending_records_are_excluded(tmp_path):
    path = tmp_path / "curated.jsonl"
    rows = []
    for category in ("vague", "partial", "clear"):
        for index in range(6):
            rows.append(
                make_record(
                    f"{category}_{index}",
                    "",
                    category,
                    f"{category}_family_{index}",
                    "pending_review" if index == 0 else "approved",
                )
            )
    write_jsonl(path, rows)

    approved, pending = prepare_module.load_approved_records(path)
    assert pending == 3
    assert len(approved) == 15


def test_partial_record_requires_known_requirements():
    row = make_record("partial_bad", "", "partial", "partial_bad")
    row["target"] = "Decision: CLARIFY\nQuestions:\n1. What is missing?"
    try:
        prepare_module.validate_record(row)
    except ValueError as exc:
        assert "Known requirements" in str(exc)
    else:
        raise AssertionError("Expected partial target validation to fail.")


def test_related_group_can_span_categories_but_stays_in_one_split(tmp_path):
    rows = []
    for category in ("vague", "partial", "clear"):
        for index in range(6):
            group_id = "shared_school_scenario" if index == 0 else f"{category}_family_{index}"
            rows.append(
                make_record(
                    f"{category}_{index}",
                    "",
                    category,
                    group_id,
                )
            )

    source = tmp_path / "curated.jsonl"
    write_jsonl(source, rows)
    approved, _ = prepare_module.load_approved_records(source)
    splits = prepare_module.grouped_stratified_split(approved, seed=42)

    locations = {
        split_name
        for split_name, split_rows in splits.items()
        if any(row["group_id"] == "shared_school_scenario" for row in split_rows)
    }
    assert len(locations) == 1
