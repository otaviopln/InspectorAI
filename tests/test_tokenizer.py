from pathlib import Path

import tokenizer as tokenizer_module
from tokenizer import BPETokenizer


def test_tokenizer_round_trip():
    tokenizer = BPETokenizer(vocab_size=300)

    sample = "Inspector AI creates structured prompts."

    tokenizer.train(
        [sample] * 10,
        max_base_chars=64,
    )

    encoded = tokenizer.encode(sample)

    decoded = tokenizer.decode(
        encoded,
        skip_special=True,
    )

    assert decoded == sample


def test_special_tokens_have_reserved_ids():
    from tokenizer import PAD_IDX, SOS_IDX, EOS_IDX

    assert PAD_IDX == 0
    assert SOS_IDX == 1
    assert EOS_IDX == 2


def test_standalone_main_does_not_overwrite_official_tokenizer(monkeypatch):
    sample = "Inspector AI creates structured prompts."
    official_path = (
        tokenizer_module.ROOT
        / "inspector"
        / "models"
        / "tokenizer_vocab.json"
    )
    official_before = (
        official_path.read_bytes()
        if official_path.exists()
        else None
    )
    saved_paths = []

    class FakeTokenizer:
        def __init__(self, vocab_size=3000):
            self.vocab_size = vocab_size

        def train(self, texts):
            assert texts

        def encode(self, text, add_special=False):
            return [1, 4, 2] if add_special else [4]

        def decode(self, ids, skip_special=False):
            return sample

        def save(self, path):
            path = Path(path)
            path.write_text("temporary tokenizer", encoding="utf-8")
            saved_paths.append(path)

        @classmethod
        def load(cls, path):
            assert Path(path).is_file()
            return cls()

    monkeypatch.setattr(
        tokenizer_module,
        "BPETokenizer",
        FakeTokenizer,
    )

    tokenizer_module.main()

    assert len(saved_paths) == 1
    assert saved_paths[0].resolve() != official_path.resolve()
    assert not saved_paths[0].exists()  # TemporaryDirectory cleaned it up.

    official_after = (
        official_path.read_bytes()
        if official_path.exists()
        else None
    )
    assert official_after == official_before
