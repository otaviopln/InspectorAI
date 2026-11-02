"""Tests for dataset_builder.py (expand_dataset.py)."""

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT / "scripts"

# Carregar expand_dataset.py diretamente via importlib para evitar
# conflitos de nome com outros módulos.
expand_path = SCRIPTS_DIR / "expand_dataset.py"
spec = importlib.util.spec_from_file_location("expand_dataset_module", expand_path)
expand_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(expand_module)

build_dataset = expand_module.build_dataset


def test_dataset_has_required_structure():
    """Dataset must have version, description, splits, and counts."""
    dataset = build_dataset(target_pairs=10)

    assert "version" in dataset
    assert "description" in dataset
    assert "splits" in dataset
    assert "counts" in dataset


def test_dataset_has_both_splits():
    """Dataset must contain both vague_to_questions and clear_to_rcfo splits."""
    dataset = build_dataset(target_pairs=10)

    splits = dataset["splits"]
    assert "vague_to_questions" in splits
    assert "clear_to_rcfo" in splits


def test_dataset_pair_structure():
    """Each pair must have 'input' and 'target' keys."""
    dataset = build_dataset(target_pairs=10)

    for split_name, pairs in dataset["splits"].items():
        for pair in pairs:
            assert "input" in pair
            assert "target" in pair
            assert isinstance(pair["input"], str)
            assert isinstance(pair["target"], str)
            assert len(pair["input"]) > 0
            assert len(pair["target"]) > 0


def test_dataset_counts_match():
    """counts must match the actual number of pairs in splits."""
    dataset = build_dataset(target_pairs=20)

    counts = dataset["counts"]
    splits = dataset["splits"]

    assert counts["vague_to_questions"] == len(splits["vague_to_questions"])
    assert counts["clear_to_rcfo"] == len(splits["clear_to_rcfo"])
    assert counts["total"] == len(splits["vague_to_questions"]) + len(splits["clear_to_rcfo"])


def test_dataset_total_matches_target():
    """Total pairs should approximately match the target."""
    dataset = build_dataset(target_pairs=50)

    assert dataset["counts"]["total"] == 50


def test_vague_pairs_have_question_structure():
    """Vague prompt targets should contain R.C.F.O-aligned questions."""
    dataset = build_dataset(target_pairs=20)

    for pair in dataset["splits"]["vague_to_questions"]:
        target = pair["target"]
        assert "Role:" in target
        assert "Context:" in target
        assert "Focus:" in target
        assert "Output:" in target


def test_clear_pairs_have_rcfo_structure():
    """Clear prompt targets should contain R.C.F.O fields."""
    dataset = build_dataset(target_pairs=20)

    for pair in dataset["splits"]["clear_to_rcfo"]:
        target = pair["target"]
        assert "Role:" in target
        assert "Context:" in target
        assert "Focus:" in target
        assert "Output:" in target