"""Training loop for Inspector AI.

Trains the custom Seq2Seq Transformer (InspectorTransformer) on the OpenAssistant
dataset (preferred) or the mock dataset (fallback), using the from-scratch
BPE tokenizer.

Pipeline:
  1. Load openassistant_dataset.json (or mock_dataset.json as fallback)
     (input -> target pairs)
  2. Load / train BPETokenizer on the same corpus
  3. Encode pairs into integer sequences and batch them with padding
  4. Forward pass through InspectorTransformer (teacher forcing)
  5. CrossEntropyLoss on non-padding positions
  6. Backpropagation + Adam optimizer step
  7. Save trained weights to inspector/models/inspector_model.pth

Device support:
  - Auto-detects NVIDIA CUDA, Intel/AMD DirectML, or falls back to CPU.
  - Set env var INSPECTOR_DEVICE=cpu|cuda|directml to force a device.
"""

from __future__ import annotations

import json
import os
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from tokenizer import BPETokenizer, SOS_IDX, EOS_IDX, PAD_IDX, SPECIAL_TOKENS  # noqa: E402
from model import InspectorTransformer  # noqa: E402

# ---------------------------------------------------------------------------
# Paths
# =============================================================================
EXPANDED_DATASET_PATH = ROOT / "inspector" / "data" / "processed" / "expanded_dataset.json"
EXPANDED_60K_DATASET_PATH = ROOT / "inspector" / "data" / "processed" / "expanded_60k_dataset.json"
OA_DATASET_PATH = ROOT / "inspector" / "data" / "processed" / "openassistant_dataset.json"
MOCK_DATASET_PATH = ROOT / "inspector" / "data" / "processed" / "mock_dataset.json"

TOKENIZER_PATH = ROOT / "inspector" / "models" / "tokenizer_vocab.json"
MODEL_PATH = ROOT / "inspector" / "models" / "inspector_model.pth"

# Priority: expanded_60k > expanded > openassistant > mock
DATASET_PATH = (
    EXPANDED_60K_DATASET_PATH
    if EXPANDED_60K_DATASET_PATH.exists()
    else EXPANDED_DATASET_PATH
    if EXPANDED_DATASET_PATH.exists()
    else OA_DATASET_PATH
    if OA_DATASET_PATH.exists()
    else MOCK_DATASET_PATH
)

# =============================================================================
# Device selection (Colab GPU / TPU / CPU)
# =============================================================================
_FORCED_DEVICE = os.environ.get("INSPECTOR_DEVICE", "").strip().lower()


def pick_device() -> torch.device:
    """Return the best available torch device.

    Priority: user env var INSPECTOR_DEVICE > CUDA > DirectML > CPU.
    """
    if _FORCED_DEVICE in {"cpu", "cuda", "directml", "tpu"}:
        if _FORCED_DEVICE == "cuda" and not torch.cuda.is_available():
            print("WARNING: CUDA unavailable. Falling back to CPU.")
            return torch.device("cpu")
        if _FORCED_DEVICE == "tpu":
            try:
                import torch_xla.core.xla_model as xm  # type: ignore
                return xm.xla_device()
            except Exception as exc:
                print(f"WARNING: TPU unavailable ({exc}). Falling back to CPU.")
                return torch.device("cpu")
        if _FORCED_DEVICE == "directml":
            try:
                import torch_directml  # type: ignore
                return torch_directml.device()
            except Exception as exc:
                print(f"WARNING: DirectML unavailable ({exc}). Falling back to CPU.")
                return torch.device("cpu")
        return torch.device(_FORCED_DEVICE)

    if torch.cuda.is_available():
        return torch.device("cuda")

    try:
        import torch_xla.core.xla_model as xm  # type: ignore
        return xm.xla_device()
    except Exception:
        pass

    try:
        import torch_directml  # type: ignore
        if torch_directml.is_available():
            return torch_directml.device()
    except Exception:
        pass

    return torch.device("cpu")


DEVICE = pick_device()

print(f"🖥️  Device: {DEVICE}")
if torch.cuda.is_available():
    print(f"🚀 GPU: {torch.cuda.get_device_name(0)}")
    print(f"🧠 VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

# =============================================================================
# Hyperparameters — Optimized for Colab GPU (60k+ pairs)
# =============================================================================
VOCAB_SIZE = 3000          # Tokenizer vocabulary size
D_MODEL = 256              # Model hidden dimension (máximo desempenho equilibrado)
NUM_HEADS = 8              # Attention heads (256 / 8 = 32, divisível)
NUM_LAYERS = 6             # Camadas Transformer
D_FF = 1024                # Feed-forward inner dimension (256 × 4)
MAX_LEN = 128              # Max sequence length
BATCH_SIZE = 32            # Samples per batch (T4 suporta 32 com D_MODEL=256)
EPOCHS = 15                # Training epochs (15 conforme solicitado)
LR = 5e-4                  # Learning rate
PAD_IDX = 0                # Padding token id
MAX_PAIRS = None           # None = use all available pairs (~60k+)

# Validation and early stopping
VALIDATION_RATIO = 0.10
RANDOM_SEED = 42
EARLY_STOPPING_PATIENCE = 5
MIN_DELTA = 0.0001

# =============================================================================
# Dataset loading
# =============================================================================


def load_pairs(path: Path) -> List[Tuple[str, str]]:
    """Load (input, target) text pairs from a dataset JSON."""
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    pairs: List[Tuple[str, str]] = []
    for _split_name, split_pairs in data.get("splits", {}).items():
        for pair in split_pairs:
            pairs.append((pair.get("input", ""), pair.get("target", "")))
    return pairs


def pad_sequences(
    seqs: List[List[int]], pad_val: int = PAD_IDX, max_len: int = MAX_LEN
) -> torch.Tensor:
    """Pad/truncate a list of id-sequences to the same length."""
    padded: List[List[int]] = []
    for seq in seqs:
        if len(seq) > max_len:
            seq = seq[:max_len]
        padded.append(seq + [pad_val] * (max_len - len(seq)))
    return torch.tensor(padded, dtype=torch.long)


def collate_batch(
    batch: List[Tuple[str, str]], tokenizer: BPETokenizer
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Encode a raw batch and return (src, tgt_input, tgt_labels).

    tgt_input is prefixed with <SOS> and fed into the decoder (teacher forcing).
    tgt_labels is suffixed with <EOS> and used for loss computation.
    Padding positions are ignored by CrossEntropyLoss(ignore_index=PAD_IDX).
    """
    src_ids = [tokenizer.encode(inp) for inp, _ in batch]
    tgt_ids = [
        [SOS_IDX] + tokenizer.encode(tar) + [EOS_IDX] for _, tar in batch
    ]

    src = pad_sequences(src_ids)
    tgt_input = pad_sequences(tgt_ids)   # fed into decoder
    tgt_labels = pad_sequences(tgt_ids)  # used for loss (teacher forcing)
    return src, tgt_input, tgt_labels


# =============================================================================
# Training
# =============================================================================


def main() -> None:
    # 1. Load dataset
    pairs = load_pairs(DATASET_PATH)
    if MAX_PAIRS is not None and len(pairs) > MAX_PAIRS:
        pairs = pairs[:MAX_PAIRS]

    print(f"📊 Loaded {len(pairs)} training pairs from {DATASET_PATH.name}")
    print(f"🖥️  Device: {DEVICE}")

    # 2. Load / train tokenizer
    tokenizer = BPETokenizer(vocab_size=VOCAB_SIZE)
    if TOKENIZER_PATH.exists():
        tokenizer = BPETokenizer.load(TOKENIZER_PATH)
        saved_vocab_size = len(tokenizer.vocab)
        if saved_vocab_size < VOCAB_SIZE:
            print(
                f"🔄 Retraining tokenizer: {saved_vocab_size} → {VOCAB_SIZE} tokens"
            )
            texts = [
                text
                for pair in pairs
                for text in pair
            ]
            tokenizer.train(
                texts,
                max_base_chars=VOCAB_SIZE - len(SPECIAL_TOKENS) - 1,
            )
            tokenizer.save(TOKENIZER_PATH)
        else:
            print(f"📥 Loaded tokenizer: {saved_vocab_size} tokens")
    else:
        texts = [
            text
            for pair in pairs
            for text in pair
        ]
        print(f"🏗️  Training new tokenizer on {len(texts)} text segments...")
        tokenizer.train(
            texts,
            max_base_chars=VOCAB_SIZE - len(SPECIAL_TOKENS) - 1,
        )
        tokenizer.save(TOKENIZER_PATH)

    vocab_size = len(tokenizer.vocab)
    print(f"📚 Final vocabulary: {vocab_size} tokens")

    if vocab_size <= EOS_IDX:
        raise ValueError("Invalid tokenizer vocabulary.")

    # 3. Split train / validation
    rng = random.Random(RANDOM_SEED)
    rng.shuffle(pairs)

    val_count = max(1, int(len(pairs) * VALIDATION_RATIO))
    val_count = min(val_count, len(pairs) - 1)

    val_pairs = pairs[:val_count]
    train_pairs = pairs[val_count:]

    print(f"📈 Train pairs: {len(train_pairs)}")
    print(f"📉 Validation pairs: {len(val_pairs)}")

    # 4. Model, loss, optimizer
    model = InspectorTransformer(
        vocab_size=vocab_size,
        d_model=D_MODEL,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
        d_ff=D_FF,
        max_len=MAX_LEN,
        pad_idx=PAD_IDX,
    ).to(DEVICE)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"🧠 Model parameters: {total_params:,}")

    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX, reduction="sum")
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    num_batches = (len(train_pairs) + BATCH_SIZE - 1) // BATCH_SIZE

    print(
        f"🚀 Starting training: epochs={EPOCHS}, batch_size={BATCH_SIZE}, "
        f"lr={LR}, device={DEVICE.type}"
    )

    # 5. Training loop
    best_validation_loss = float("inf")
    epochs_without_improvement = 0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        rng = random.Random(RANDOM_SEED + epoch)
        rng.shuffle(train_pairs)

        total_train_loss = 0.0
        total_train_tokens = 0

        for i in range(0, len(train_pairs), BATCH_SIZE):
            batch = train_pairs[i : i + BATCH_SIZE]
            src, tgt_input, tgt_labels = collate_batch(batch, tokenizer)

            src = src.to(DEVICE)
            tgt_input = tgt_input.to(DEVICE)
            tgt_labels = tgt_labels.to(DEVICE)

            logits = model(src, tgt_input)

            loss_sum = criterion(
                logits.reshape(-1, logits.size(-1)), tgt_labels.reshape(-1)
            )

            token_count = int((tgt_labels != PAD_IDX).sum().item())
            if token_count == 0:
                continue

            loss = loss_sum / token_count

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_train_loss += loss_sum.item()
            total_train_tokens += token_count

        if total_train_tokens == 0:
            raise RuntimeError("No valid tokens used in training.")

        train_loss = total_train_loss / total_train_tokens

        # Validation
        model.eval()
        total_val_loss = 0.0
        total_val_tokens = 0
        with torch.no_grad():
            for i in range(0, len(val_pairs), BATCH_SIZE):
                batch = val_pairs[i : i + BATCH_SIZE]
                src, tgt_input, tgt_labels = collate_batch(batch, tokenizer)

                src = src.to(DEVICE)
                tgt_input = tgt_input.to(DEVICE)
                tgt_labels = tgt_labels.to(DEVICE)

                logits = model(src, tgt_input)
                loss_sum = criterion(
                    logits.reshape(-1, logits.size(-1)),
                    tgt_labels.reshape(-1),
                )
                token_count = int((tgt_labels != PAD_IDX).sum().item())
                total_val_loss += loss_sum.item()
                total_val_tokens += token_count

        val_loss = total_val_loss / max(total_val_tokens, 1)

        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"Train loss: {train_loss:.4f} | "
            f"Val loss: {val_loss:.4f}"
        )

        # Save best checkpoint
        improved = val_loss < best_validation_loss - MIN_DELTA

        if improved:
            best_validation_loss = val_loss
            epochs_without_improvement = 0

            MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

            checkpoint = {
                "format_version": 2,
                "model_state_dict": {
                    k: v.detach().cpu() for k, v in model.state_dict().items()
                },
                "model_config": {
                    "vocab_size": vocab_size,
                    "d_model": D_MODEL,
                    "num_heads": NUM_HEADS,
                    "num_layers": NUM_LAYERS,
                    "d_ff": D_FF,
                    "max_len": MAX_LEN,
                    "pad_idx": PAD_IDX,
                },
                "training_config": {
                    "epoch": epoch,
                    "epochs": EPOCHS,
                    "batch_size": BATCH_SIZE,
                    "learning_rate": LR,
                    "validation_loss": val_loss,
                    "random_seed": RANDOM_SEED,
                },
                "optimizer_state_dict": optimizer.state_dict(),
            }

            torch.save(checkpoint, MODEL_PATH)
            print(
                f"   ✅ Best checkpoint saved. Val loss: {val_loss:.4f}"
            )

        else:
            epochs_without_improvement += 1
            print(
                f"   ⏳ No improvement: {epochs_without_improvement}/"
                f"{EARLY_STOPPING_PATIENCE}"
            )

        # Early stopping
        if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
            print("🛑 Early stopping triggered.")
            break

    print("\n✅ Training completed.")
    print(f"📊 Best validation loss: {best_validation_loss:.4f}")
    print(f"💾 Model: {MODEL_PATH}")
    print(f"🔤 Tokenizer: {TOKENIZER_PATH}")


if __name__ == "__main__":
    main()
