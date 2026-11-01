
"""
Fast BPE Tokenizer for InspectorAI.

Uses Hugging Face Tokenizers, whose BPE implementation
is written in Rust.

Maintains the interface required by InspectorAI:
- train()
- encode()
- decode()
- save()
- load()

Special token IDs:
PAD = 0
SOS = 1
EOS = 2
UNK = 3
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

from tokenizers import (
    Tokenizer,
    models,
    pre_tokenizers,
    decoders,
    trainers,
)

# ---------------------------------------------------------------
# Paths
# ---------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]

DATASET_PATH = (
    ROOT / "inspector" / "data" / "processed"
    / "mock_dataset.json"
)

VOCAB_PATH = (
    ROOT / "inspector" / "models"
    / "tokenizer_vocab.json"
)

# ---------------------------------------------------------------
# Special tokens
# ---------------------------------------------------------------

PAD_IDX = 0
SOS_IDX = 1
EOS_IDX = 2
UNK_IDX = 3

SPECIAL_TOKENS = [
    "<PAD>",
    "<SOS>",
    "<EOS>",
    "<UNK>",
]


def _read_dataset_texts(path: Path) -> List[str]:
    """Reads input and target texts from the dataset JSON."""

    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}"
        )

    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    texts = []

    for pairs in data.get("splits", {}).values():
        for pair in pairs:
            if not isinstance(pair, dict):
                continue

            user_input = str(
                pair.get("input", "")
            ).strip()

            target = str(
                pair.get("target", "")
            ).strip()

            if user_input:
                texts.append(user_input)

            if target:
                texts.append(target)

    return texts


class BPETokenizer:
    """
    Rust-accelerated Byte-Level BPE tokenizer.

    The public API is compatible with InspectorAI.
    """

    def __init__(self, vocab_size: int = 3000) -> None:

        if vocab_size < 260:
            raise ValueError(
                "Byte-Level BPE requires a vocabulary size "
                "of at least 260 (256 byte symbols + 4 specials)."
            )

        self.vocab_size = vocab_size

        self.char_to_id: Dict[str, int] = {}
        self.id_to_char: Dict[int, str] = {}

        self.vocab: Dict[int, str] = {}
        self.merges: Dict = {}

        self._backend: Optional[Tokenizer] = None
        self._trained = False

    # -----------------------------------------------------------
    # Training
    # -----------------------------------------------------------

    def train(
        self,
        texts: List[str],
        max_base_chars: int = 500,
    ) -> None:
        """
        Trains BPE using the Rust implementation.

        max_base_chars is retained for compatibility with the
        previous tokenizer interface. Byte-Level BPE uses its
        complete 256-symbol byte alphabet instead.
        """

        clean_texts = [
            str(text)
            for text in texts
            if text is not None and str(text).strip()
        ]

        if not clean_texts:
            raise ValueError(
                "Cannot train tokenizer with an empty corpus."
            )

        print(
            f"Training fast BPE tokenizer on "
            f"{len(clean_texts)} text segments..."
        )

        # Create BPE model.
        backend = Tokenizer(
            models.BPE(
                unk_token="<UNK>"
            )
        )

        # Byte-level pre-tokenization.
        backend.pre_tokenizer = (
            pre_tokenizers.ByteLevel(
                add_prefix_space=False
            )
        )

        # Corresponding decoder.
        backend.decoder = decoders.ByteLevel()

        # Rust BPE trainer.
        trainer = trainers.BpeTrainer(
            vocab_size=self.vocab_size,
            min_frequency=2,
            special_tokens=SPECIAL_TOKENS,
            initial_alphabet=(
                pre_tokenizers.ByteLevel.alphabet()
            ),
            show_progress=True,
        )

        # Train directly from memory.
        backend.train_from_iterator(
            clean_texts,
            trainer=trainer,
            length=len(clean_texts),
        )

        # Validate reserved token IDs.
        for expected_id, token in enumerate(SPECIAL_TOKENS):

            actual_id = backend.token_to_id(token)

            if actual_id != expected_id:
                raise RuntimeError(
                    f"Invalid special token ID: {token}. "
                    f"Expected {expected_id}, got {actual_id}."
                )

        self._backend = backend

        self._refresh_vocabulary()

        self._trained = True

        print(
            f"Fast BPE training completed. "
            f"Vocabulary: {len(self.vocab)} tokens."
        )

    # -----------------------------------------------------------
    # Vocabulary
    # -----------------------------------------------------------

    def _refresh_vocabulary(self) -> None:
        """Synchronizes the public vocabulary with the backend."""

        if self._backend is None:
            raise RuntimeError(
                "Tokenizer backend is not initialized."
            )

        backend_vocab = self._backend.get_vocab()

        self.char_to_id = {
            token: token_id
            for token, token_id in backend_vocab.items()
        }

        self.id_to_char = {
            token_id: token
            for token, token_id in backend_vocab.items()
        }

        self.vocab = dict(self.id_to_char)

        self.vocab_size = len(self.vocab)

    # -----------------------------------------------------------
    # Encoding
    # -----------------------------------------------------------

    def encode(
        self,
        text: str,
        add_special: bool = False,
    ) -> List[int]:
        """Converts text into token IDs."""

        if not self._trained or self._backend is None:
            raise RuntimeError(
                "Tokenizer must be trained or loaded before encoding."
            )

        encoded = self._backend.encode(
            str(text),
            add_special_tokens=False,
        )

        ids = list(encoded.ids)

        if add_special:
            ids = (
                [SOS_IDX]
                + ids
                + [EOS_IDX]
            )

        return ids

    # -----------------------------------------------------------
    # Decoding
    # -----------------------------------------------------------

    def decode(
        self,
        ids: List[int],
        skip_special: bool = False,
    ) -> str:
        """Converts token IDs back into text."""

        if not self._trained or self._backend is None:
            raise RuntimeError(
                "Tokenizer must be trained or loaded before decoding."
            )

        valid_ids = [
            int(token_id)
            for token_id in ids
            if 0 <= int(token_id) < len(self.vocab)
        ]

        return self._backend.decode(
            valid_ids,
            skip_special_tokens=skip_special,
        )

    # -----------------------------------------------------------
    # Saving
    # -----------------------------------------------------------

    def save(self, path: Path) -> None:
        """Saves the complete tokenizer configuration."""

        if not self._trained or self._backend is None:
            raise RuntimeError(
                "Cannot save an untrained tokenizer."
            )

        path = Path(path)

        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        self._backend.save(str(path))

        print(
            f"Tokenizer saved to: {path}"
        )

    # -----------------------------------------------------------
    # Loading
    # -----------------------------------------------------------

    @classmethod
    def load(cls, path: Path) -> "BPETokenizer":
        """Loads a previously saved Rust BPE tokenizer."""

        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(
                f"Tokenizer file not found: {path}"
            )

        backend = Tokenizer.from_file(
            str(path)
        )

        tokenizer = cls(
            vocab_size=max(
                backend.get_vocab_size(),
                260,
            )
        )

        tokenizer._backend = backend
        tokenizer._refresh_vocabulary()

        # Validate special token IDs.
        for expected_id, token in enumerate(SPECIAL_TOKENS):

            actual_id = backend.token_to_id(token)

            if actual_id != expected_id:
                raise ValueError(
                    f"Incompatible tokenizer: {token} "
                    f"has ID {actual_id}, expected {expected_id}."
                )

        tokenizer._trained = True

        print(
            f"Tokenizer loaded: {len(tokenizer.vocab)} tokens."
        )

        return tokenizer


# ---------------------------------------------------------------
# Standalone test
# ---------------------------------------------------------------

def main() -> None:

    texts = [
        "Inspector AI creates structured prompts.",
        "The system analyzes ambiguous user requests.",
        "Clear requests are transformed into RCFO format.",
    ] * 100

    tokenizer = BPETokenizer(
        vocab_size=3000
    )

    tokenizer.train(texts)

    sample = (
        "Inspector AI creates structured prompts."
    )

    encoded = tokenizer.encode(
        sample,
        add_special=True,
    )

    decoded = tokenizer.decode(
        encoded,
        skip_special=True,
    )

    print("Original:", sample)
    print("Encoded:", encoded)
    print("Decoded:", decoded)
    print("Round trip:", sample == decoded)

    tokenizer.save(VOCAB_PATH)


if __name__ == "__main__":
    main()
