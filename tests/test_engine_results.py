"""Inference regression tests for explicit input limits and completion state."""

from types import SimpleNamespace

import torch

from inspector.engine import generate_result


class DummyModel:
    def __init__(self):
        self.encoder = SimpleNamespace(
            pos_enc=SimpleNamespace(pe=torch.zeros(1, 4, 8))
        )

    def eval(self):
        return self


class TooLongTokenizer:
    vocab = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>"}

    def encode(self, text):
        return [4, 5, 6, 7, 8]

    def decode(self, ids, skip_special=False):
        return ""


def test_generate_result_rejects_long_input_without_truncating():
    result = generate_result(DummyModel(), TooLongTokenizer(), "a long prompt")

    assert result.status == "input_too_long"
    assert result.input_tokens == 5
    assert result.max_input_tokens == 4
    assert "no content was discarded" in result.text
    assert result.finished is False
