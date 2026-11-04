"""Tests for training data alignment and prepared split loading."""

import json
import sys
from pathlib import Path

import pytest

pytest.importorskip("tokenizers")

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))
import train  # noqa: E402


class FakeTokenizer:
    def encode(self, text):
        mapping = {
            "entrada": [10, 11],
            "resposta": [20, 21],
            "long_input": list(range(20, 30)),
        }
        return mapping[text]


def test_teacher_forcing_uses_next_token_shift(monkeypatch):
    monkeypatch.setattr(train, "MAX_LEN", 8)
    src, tgt_input, tgt_labels = train.collate_batch(
        [("entrada", "resposta")],
        FakeTokenizer(),
    )
    assert src[0, :2].tolist() == [10, 11]
    assert tgt_input[0, :3].tolist() == [train.SOS_IDX, 20, 21]
    assert tgt_labels[0, :3].tolist() == [20, 21, train.EOS_IDX]


def test_collate_rejects_silent_target_truncation(monkeypatch):
    monkeypatch.setattr(train, "MAX_LEN", 3)
    try:
        train.collate_batch(
            [("entrada", "long_input")],
            FakeTokenizer(),
        )
    except ValueError as exc:
        assert "target excede MAX_LEN" in str(exc)
    else:
        raise AssertionError("Expected target length overflow to fail.")


def test_loader_requires_all_explicit_splits(tmp_path):
    row = {
        "id": "example_1",
        "task": "clarify",
        "ambiguity": "vague",
        "language": "en",
        "input": "Help me.",
        "target": "Decision: CLARIFY\\nQuestions:\\n1. What do you need?",
        "source": "unit_test",
        "group_id": "example_group",
        "review_status": "approved",
    }
    data = {
        "schema_version": "inspector.dataset.v2",
        "language": "en",
        "splits": {
            "train": [row],
            "validation": [dict(row, id="example_2", group_id="validation_group")],
            "test": [dict(row, id="example_3", group_id="test_group")],
        },
    }
    path = tmp_path / "prepared.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    train_pairs, validation_pairs, _ = train.load_prepared_dataset(path)
    assert len(train_pairs) == len(validation_pairs) == 1



def test_loader_does_not_require_test_examples_to_be_trainable(tmp_path):
    row = {
        "id": "example_1",
        "task": "clarify",
        "ambiguity": "vague",
        "language": "en",
        "input": "Help me.",
        "target": "Decision: CLARIFY\\nQuestions:\\n1. What do you need?",
        "source": "unit_test",
        "group_id": "train_group",
        "review_status": "approved",
    }
    data = {
        "schema_version": "inspector.dataset.v2",
        "language": "en",
        "splits": {
            "train": [row],
            "validation": [dict(row, id="val", group_id="val_group")],
            "test": [dict(row, id="test", group_id="test_group", input="", target="")],
        },
    }
    path = tmp_path / "prepared.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    train_pairs, validation_pairs, _ = train.load_prepared_dataset(path)
    assert len(train_pairs) == len(validation_pairs) == 1


def test_padding_is_dynamic_and_never_truncates(monkeypatch):
    monkeypatch.setattr(train, "MAX_LEN", 8)
    padded = train.pad_sequences([[4, 5], [6, 7, 8]], max_len=8)
    assert padded.shape == (2, 3)
    with pytest.raises(ValueError, match="não deve truncar"):
        train.pad_sequences([[1, 2, 3, 4]], max_len=3)



def test_sequence_length_precheck_fails_before_training(monkeypatch):
    monkeypatch.setattr(train, "MAX_LEN", 8)
    with pytest.raises(ValueError, match="excedem MAX_LEN"):
        train.validate_sequence_lengths(
            [("entrada", "long_input")],
            FakeTokenizer(),
            "train",
        )
