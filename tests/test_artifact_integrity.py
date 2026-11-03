"""Regression tests for checkpoint artifact integrity metadata."""

import json
from pathlib import Path

import pytest

from inspector import engine


def _write_json(path: Path, data, *, indent=None):
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=indent),
        encoding="utf-8",
    )
    return path


def _checkpoint_for(tokenizer_path: Path, dataset_path: Path) -> dict:
    return {
        "format_version": 4,
        "artifact_hashes": {
            "algorithm": engine.HASH_ALGORITHM,
            "tokenizer": engine._canonical_json_sha256(tokenizer_path),
            "prepared_dataset": engine._canonical_json_sha256(dataset_path),
        },
    }


def test_matching_tokenizer_and_dataset_hashes_are_accepted(tmp_path):
    tokenizer_path = _write_json(
        tmp_path / "tokenizer.json",
        {"model": {"vocab": {"hello": 4}}, "version": 1},
        indent=2,
    )
    dataset_path = _write_json(
        tmp_path / "prepared_dataset.json",
        {"schema_version": "inspector.dataset.v2", "splits": {"train": []}},
        indent=4,
    )
    checkpoint = _checkpoint_for(tokenizer_path, dataset_path)

    assert engine._validate_checkpoint_artifact_hashes(
        checkpoint,
        tokenizer_path=tokenizer_path,
        dataset_path=dataset_path,
    ) is True


def test_tokenizer_hash_mismatch_is_rejected(tmp_path):
    tokenizer_path = _write_json(tmp_path / "tokenizer.json", {"vocab": {"a": 4}})
    dataset_path = _write_json(tmp_path / "prepared_dataset.json", {"splits": {}})
    checkpoint = _checkpoint_for(tokenizer_path, dataset_path)

    _write_json(tokenizer_path, {"vocab": {"b": 4}})

    with pytest.raises(ValueError, match="tokenizer"):
        engine._validate_checkpoint_artifact_hashes(
            checkpoint,
            tokenizer_path=tokenizer_path,
            dataset_path=dataset_path,
        )


def test_dataset_hash_mismatch_is_rejected(tmp_path):
    tokenizer_path = _write_json(tmp_path / "tokenizer.json", {"vocab": {"a": 4}})
    dataset_path = _write_json(tmp_path / "prepared_dataset.json", {"splits": {"train": [1]}})
    checkpoint = _checkpoint_for(tokenizer_path, dataset_path)

    _write_json(dataset_path, {"splits": {"train": [1, 2]}})

    with pytest.raises(ValueError, match="prepared_dataset"):
        engine._validate_checkpoint_artifact_hashes(
            checkpoint,
            tokenizer_path=tokenizer_path,
            dataset_path=dataset_path,
        )


def test_canonical_hash_ignores_json_formatting(tmp_path):
    path = tmp_path / "artifact.json"
    _write_json(path, {"z": 1, "a": [2, 3]}, indent=2)
    first_hash = engine._canonical_json_sha256(path)

    _write_json(path, {"a": [2, 3], "z": 1}, indent=None)
    second_hash = engine._canonical_json_sha256(path)

    assert first_hash == second_hash


def test_checkpoint_v4_requires_artifact_hashes():
    with pytest.raises(ValueError, match="artifact_hashes"):
        engine._validate_checkpoint_artifact_hashes({"format_version": 4})


def test_legacy_checkpoint_without_hashes_warns_and_remains_compatible(capsys):
    assert engine._validate_checkpoint_artifact_hashes({"format_version": 3}) is False
    assert "checkpoint legado sem hashes" in capsys.readouterr().out
