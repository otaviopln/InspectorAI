"""Train InspectorAI on a reviewed, pre-split dataset.

Expected input: inspector/data/processed/prepared_dataset.json
Generated artifacts:
  inspector/models/tokenizer_vocab.json
  inspector/models/inspector_model.pth
  inspector/models/training_metrics.json

The dataset must be prepared with scripts/prepare_dataset.py first.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import shutil
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.tokenizer import (
    BPETokenizer,
    SOS_IDX,
    EOS_IDX,
    PAD_IDX,
    UNK_IDX,
)
from scripts.model import InspectorTransformer

PREPARED_DATASET_PATH = (
    ROOT / "inspector" / "data" / "processed" / "prepared_dataset.json"
)
TOKENIZER_PATH = ROOT / "inspector" / "models" / "tokenizer_vocab.json"
MODEL_PATH = ROOT / "inspector" / "models" / "inspector_model.pth"
METRICS_PATH = ROOT / "inspector" / "models" / "training_metrics.json"
STAGING_MODEL_PATH = ROOT / "inspector" / "models" / ".inspector_model.pth.tmp"
STAGING_TOKENIZER_PATH = ROOT / "inspector" / "models" / ".tokenizer_vocab.json.tmp"

# Colab-friendly baseline. Do not scale these until the data pipeline is stable.
VOCAB_SIZE = 3000
D_MODEL = 256
NUM_HEADS = 8
NUM_LAYERS = 4
D_FF = 1024
MAX_LEN = 192
BATCH_SIZE = 16
EPOCHS = 20
LR = 5e-4
RETRAIN_TOKENIZER = True

RANDOM_SEED = 42
EARLY_STOPPING_PATIENCE = 5
MIN_DELTA = 0.0001

def pick_device() -> torch.device:
    forced_device = os.environ.get("INSPECTOR_DEVICE", "").strip().lower()
    if forced_device == "cpu":
        return torch.device("cpu")
    if forced_device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("INSPECTOR_DEVICE=cuda, mas CUDA não está disponível.")
        return torch.device("cuda")
    if forced_device == "directml":
        try:
            import torch_directml  # type: ignore
            return torch_directml.device()
        except Exception as exc:
            raise RuntimeError(f"DirectML não está disponível: {exc}") from exc
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


HASH_ALGORITHM = "sha256-canonical-json-v1"


def _canonical_json_sha256(data: Any) -> str:
    """Hash JSON content independently of indentation and line endings."""
    canonical = json.dumps(
        data,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _canonical_json_file_sha256(path: Path) -> str:
    """Load a JSON artifact and return its canonical SHA-256 fingerprint."""
    if not path.is_file():
        raise FileNotFoundError(f"Artifact JSON não encontrado: {path}")
    with path.open("r", encoding="utf-8") as stream:
        data = json.load(stream)
    return _canonical_json_sha256(data)


def _records_to_pairs(records: Any, split_name: str) -> list[tuple[str, str]]:
    if not isinstance(records, list) or not records:
        raise ValueError(f"O conjunto '{split_name}' está vazio ou inválido.")

    pairs: list[tuple[str, str]] = []
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError(f"Registro {index} de '{split_name}' não é um objeto.")
        user_input = str(record.get("input", "")).strip()
        target = str(record.get("target", "")).strip()
        if not user_input or not target:
            raise ValueError(
                f"Registro {index} de '{split_name}' tem input/target vazio."
            )
        pairs.append((user_input, target))
    return pairs


def load_prepared_dataset(path: Path = PREPARED_DATASET_PATH):
    """Load the explicit train/validation/test splits and guard against leakage."""
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset preparado não encontrado: {path}\n"
            "Execute primeiro: python scripts/prepare_dataset.py"
        )

    with path.open("r", encoding="utf-8") as stream:
        data = json.load(stream)

    if data.get("schema_version") != "inspector.dataset.v2":
        raise ValueError(
            "Schema incompatível. Esperado 'inspector.dataset.v2'. "
            "Gere novamente com scripts/prepare_dataset.py."
        )
    if data.get("language") != "en":
        raise ValueError("Esta versão do treinamento espera dados em inglês.")

    splits = data.get("splits")
    if not isinstance(splits, dict):
        raise ValueError("O dataset precisa conter o objeto 'splits'.")

    train_records = splits.get("train")
    validation_records = splits.get("validation")
    test_records = splits.get("test")

    train_pairs = _records_to_pairs(train_records, "train")
    validation_pairs = _records_to_pairs(validation_records, "validation")
    if not isinstance(test_records, list) or not test_records:
        raise ValueError("O split 'test' está vazio ou inválido.")

    group_sets = {
        name: {
            str(record.get("group_id", "")).strip()
            for record in splits[name]
        }
        for name in ("train", "validation", "test")
    }
    if group_sets["train"] & group_sets["validation"]:
        raise ValueError("Vazamento de group_id entre train e validation.")
    if group_sets["train"] & group_sets["test"]:
        raise ValueError("Vazamento de group_id entre train e test.")
    if group_sets["validation"] & group_sets["test"]:
        raise ValueError("Vazamento de group_id entre validation e test.")

    # Held-out test data is evaluated only by scripts/evaluate.py.
    return train_pairs, validation_pairs, data


def pad_sequences(
    sequences: list[list[int]],
    pad_value: int = PAD_IDX,
    max_len: int = MAX_LEN,
) -> torch.Tensor:
    if max_len < 2:
        raise ValueError("MAX_LEN precisa ser pelo menos 2.")
    if not sequences:
        raise ValueError("Não é possível aplicar padding a um lote vazio.")
    if any(len(sequence) > max_len for sequence in sequences):
        raise ValueError(
            "Sequência acima de max_len; padding não deve truncar conteúdo."
        )
    batch_max_len = max(len(sequence) for sequence in sequences)
    padded = [
        sequence + [pad_value] * (batch_max_len - len(sequence))
        for sequence in sequences
    ]
    return torch.tensor(padded, dtype=torch.long)


def collate_batch(
    batch: list[tuple[str, str]],
    tokenizer: BPETokenizer,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create source, shifted decoder input and next-token labels."""
    src_ids: list[list[int]] = []
    tgt_input_ids: list[list[int]] = []
    tgt_label_ids: list[list[int]] = []

    for user_input, target in batch:
        source = tokenizer.encode(user_input)
        if not source:
            source = [UNK_IDX]
        if len(source) > MAX_LEN:
            raise ValueError(
                f"A entrada excede MAX_LEN={MAX_LEN} após tokenização: "
                f"{user_input[:100]!r}. Aumente MAX_LEN ou revise o exemplo."
            )
        src_ids.append(source)

        target_tokens = tokenizer.encode(target)
        if len(target_tokens) > MAX_LEN - 1:
            raise ValueError(
                f"O target excede MAX_LEN={MAX_LEN} após tokenização. "
                "Aumente MAX_LEN ou torne o target mais conciso. "
                f"Início do target: {target[:100]!r}"
            )
        full_target = [SOS_IDX] + target_tokens + [EOS_IDX]
        tgt_input_ids.append(full_target[:-1])
        tgt_label_ids.append(full_target[1:])

    return (
        pad_sequences(src_ids),
        pad_sequences(tgt_input_ids),
        pad_sequences(tgt_label_ids),
    )


def _percentile(values: list[int], percentile: int) -> int:
    ordered = sorted(values)
    if not ordered:
        return 0
    index = round((percentile / 100) * (len(ordered) - 1))
    return ordered[index]


def validate_sequence_lengths(
    pairs: list[tuple[str, str]],
    tokenizer: BPETokenizer,
    split_name: str,
) -> None:
    """Fail before epoch 1 if any train/validation pair exceeds MAX_LEN."""
    if not pairs:
        raise ValueError(f"O conjunto {split_name!r} está vazio.")

    source_lengths = [len(tokenizer.encode(source)) for source, _ in pairs]
    target_lengths = [len(tokenizer.encode(target)) + 1 for _, target in pairs]
    print(
        f"Lengths {split_name} | "
        f"input p50={_percentile(source_lengths, 50)} "
        f"p95={_percentile(source_lengths, 95)} max={max(source_lengths)} | "
        f"target p50={_percentile(target_lengths, 50)} "
        f"p95={_percentile(target_lengths, 95)} max={max(target_lengths)}"
    )

    problems = [
        (index, source_len, target_len)
        for index, (source_len, target_len) in enumerate(
            zip(source_lengths, target_lengths)
        )
        if source_len > MAX_LEN or target_len > MAX_LEN
    ]
    if problems:
        raise ValueError(
            f"{len(problems)} exemplos de {split_name} excedem MAX_LEN={MAX_LEN}. "
            f"Primeiros índices (índice, input_tokens, target_tokens): {problems[:10]}. "
            "Aumente MAX_LEN ou revise os exemplos; nenhum conteúdo foi truncado."
        )


def evaluate_loss(
    model: InspectorTransformer,
    pairs: list[tuple[str, str]],
    tokenizer: BPETokenizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    if not pairs:
        raise ValueError("Não é possível avaliar um conjunto vazio.")

    model.eval()
    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for start in range(0, len(pairs), BATCH_SIZE):
            batch = pairs[start : start + BATCH_SIZE]
            src, tgt_input, tgt_labels = collate_batch(batch, tokenizer)
            src = src.to(device)
            tgt_input = tgt_input.to(device)
            tgt_labels = tgt_labels.to(device)

            logits = model(src, tgt_input)
            loss_sum = criterion(
                logits.reshape(-1, logits.size(-1)),
                tgt_labels.reshape(-1),
            )
            token_count = int((tgt_labels != PAD_IDX).sum().item())
            total_loss += float(loss_sum.item())
            total_tokens += token_count

    if total_tokens == 0:
        raise RuntimeError("Nenhum token válido encontrado durante a avaliação.")
    return total_loss / total_tokens


def main() -> None:
    device = pick_device()
    random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(RANDOM_SEED)

    train_pairs, validation_pairs, dataset_info = load_prepared_dataset()
    prepared_dataset_sha256 = _canonical_json_sha256(dataset_info)

    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(
        "Examples | "
        f"train={len(train_pairs)} | "
        f"validation={len(validation_pairs)}"
    )
    if len(train_pairs) < 400:
        print(
            "WARNING: fewer than 400 training examples. This run is useful "
            "for pipeline smoke-testing, not for claiming generalization."
        )

    # Fit tokenizer only on training data to avoid validation/test leakage.
    train_texts = [text for pair in train_pairs for text in pair]
    tokenizer_is_staged = False
    if RETRAIN_TOKENIZER or not TOKENIZER_PATH.exists():
        tokenizer = BPETokenizer(vocab_size=VOCAB_SIZE)
        tokenizer.train(train_texts)
        tokenizer.save(STAGING_TOKENIZER_PATH)
        tokenizer_is_staged = True
    else:
        tokenizer = BPETokenizer.load(TOKENIZER_PATH)

    # Measure train/validation lengths with the actual tokenizer before epoch 1.
    # The held-out test split is intentionally not inspected during training.
    validate_sequence_lengths(train_pairs, tokenizer, "train")
    validate_sequence_lengths(validation_pairs, tokenizer, "validation")

    tokenizer_artifact_path = (
        STAGING_TOKENIZER_PATH if tokenizer_is_staged else TOKENIZER_PATH
    )
    artifact_hashes = {
        "algorithm": HASH_ALGORITHM,
        "tokenizer": _canonical_json_file_sha256(tokenizer_artifact_path),
        "prepared_dataset": prepared_dataset_sha256,
    }

    vocab_size = len(tokenizer.vocab)
    if vocab_size <= EOS_IDX:
        raise ValueError(f"Vocabulário inválido: {vocab_size} tokens.")

    model_config = {
        "vocab_size": vocab_size,
        "d_model": D_MODEL,
        "num_heads": NUM_HEADS,
        "num_layers": NUM_LAYERS,
        "d_ff": D_FF,
        "max_len": MAX_LEN,
        "pad_idx": PAD_IDX,
    }
    model = InspectorTransformer(**model_config).to(device)

    criterion = nn.CrossEntropyLoss(
        ignore_index=PAD_IDX,
        reduction="sum",
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    best_validation_loss = float("inf")
    best_state_dict: dict[str, torch.Tensor] | None = None
    epochs_without_improvement = 0
    rng = random.Random(RANDOM_SEED)
    history: list[dict[str, float | int]] = []

    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_pairs = list(train_pairs)
        rng.shuffle(epoch_pairs)
        total_loss = 0.0
        total_tokens = 0

        for start in range(0, len(epoch_pairs), BATCH_SIZE):
            batch = epoch_pairs[start : start + BATCH_SIZE]
            src, tgt_input, tgt_labels = collate_batch(batch, tokenizer)
            src = src.to(device)
            tgt_input = tgt_input.to(device)
            tgt_labels = tgt_labels.to(device)

            logits = model(src, tgt_input)
            loss_sum = criterion(
                logits.reshape(-1, logits.size(-1)),
                tgt_labels.reshape(-1),
            )
            token_count = int((tgt_labels != PAD_IDX).sum().item())
            if token_count == 0:
                continue

            loss = loss_sum / token_count
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += float(loss_sum.item())
            total_tokens += token_count

        if total_tokens == 0:
            raise RuntimeError("Nenhum token válido foi utilizado no treinamento.")

        train_loss = total_loss / total_tokens
        validation_loss = evaluate_loss(
            model, validation_pairs, tokenizer, criterion, device
        )
        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "validation_loss": validation_loss,
            "learning_rate": optimizer.param_groups[0]["lr"],
        })
        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"Train loss: {train_loss:.4f} | "
            f"Validation loss: {validation_loss:.4f}"
        )

        if validation_loss < best_validation_loss - MIN_DELTA:
            best_validation_loss = validation_loss
            epochs_without_improvement = 0
            best_state_dict = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }

            MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
            checkpoint = {
                "format_version": 4,
                "model_state_dict": best_state_dict,
                "model_config": model_config,
                "training_config": {
                    "epoch": epoch,
                    "epochs": EPOCHS,
                    "batch_size": BATCH_SIZE,
                    "learning_rate": LR,
                    "validation_loss": validation_loss,
                    "random_seed": RANDOM_SEED,
                    "language": "en",
                    "decision_tokens": ["CLARIFY", "FORMAT_RCFO"],
                },
                "dataset_info": {
                    "schema_version": dataset_info.get("schema_version"),
                    "split_strategy": dataset_info.get("split_strategy"),
                    "train_examples": len(train_pairs),
                    "validation_examples": len(validation_pairs),
                },
                "artifact_hashes": artifact_hashes,
            }
            torch.save(checkpoint, STAGING_MODEL_PATH)
            print(f"  Best checkpoint staged: val_loss={validation_loss:.4f}")
        else:
            epochs_without_improvement += 1
            print(
                "  No validation improvement: "
                f"{epochs_without_improvement}/{EARLY_STOPPING_PATIENCE}"
            )

        if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
            print("Early stopping.")
            break

    if best_state_dict is None:
        raise RuntimeError("Nenhum checkpoint válido foi produzido.")

    # Restore the best validation checkpoint. Test evaluation happens separately.
    model.load_state_dict(best_state_dict)

    # Refuse to publish a checkpoint if either source artifact changed
    # while training was running.
    current_dataset_sha256 = _canonical_json_file_sha256(PREPARED_DATASET_PATH)
    if current_dataset_sha256 != artifact_hashes["prepared_dataset"]:
        raise RuntimeError(
            "O dataset preparado mudou durante o treinamento. "
            "O checkpoint não será publicado; execute o treinamento novamente."
        )

    current_tokenizer_sha256 = _canonical_json_file_sha256(tokenizer_artifact_path)
    if current_tokenizer_sha256 != artifact_hashes["tokenizer"]:
        raise RuntimeError(
            "O tokenizador mudou durante o treinamento. "
            "O checkpoint não será publicado; execute o treinamento novamente."
        )

    # Commit the new model/tokenizer only after training and held-out
    # evaluation have completed. A failed run therefore leaves the old pair intact.
    model_backup = MODEL_PATH.with_name(".inspector_model.pth.backup")
    tokenizer_backup = TOKENIZER_PATH.with_name(".tokenizer_vocab.json.backup")
    had_model = MODEL_PATH.exists()
    had_tokenizer = TOKENIZER_PATH.exists()

    try:
        if had_model:
            shutil.copy2(MODEL_PATH, model_backup)
        if tokenizer_is_staged and had_tokenizer:
            shutil.copy2(TOKENIZER_PATH, tokenizer_backup)

        if tokenizer_is_staged:
            os.replace(STAGING_TOKENIZER_PATH, TOKENIZER_PATH)
        os.replace(STAGING_MODEL_PATH, MODEL_PATH)
    except Exception:
        # Restore the previous matching pair if either replacement fails.
        if had_model and model_backup.exists():
            os.replace(model_backup, MODEL_PATH)
        elif not had_model and MODEL_PATH.exists():
            MODEL_PATH.unlink()

        if tokenizer_is_staged:
            if had_tokenizer and tokenizer_backup.exists():
                os.replace(tokenizer_backup, TOKENIZER_PATH)
            elif not had_tokenizer and TOKENIZER_PATH.exists():
                TOKENIZER_PATH.unlink()
        raise
    finally:
        model_backup.unlink(missing_ok=True)
        tokenizer_backup.unlink(missing_ok=True)

    metrics = {
        "best_validation_loss": best_validation_loss,
        "train_examples": len(train_pairs),
        "validation_examples": len(validation_pairs),
        "device": str(device),
        "model_config": model_config,
        "history": history,
        "test_evaluation": "Run scripts/evaluate.py after model selection.",
    }
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with METRICS_PATH.open("w", encoding="utf-8") as stream:
        json.dump(metrics, stream, ensure_ascii=False, indent=2)

    backup_dir_env = os.environ.get("INSPECTOR_BACKUP_DIR", "").strip()
    if backup_dir_env:
        backup_dir = Path(backup_dir_env)
        backup_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(MODEL_PATH, backup_dir / MODEL_PATH.name)
        shutil.copy2(TOKENIZER_PATH, backup_dir / TOKENIZER_PATH.name)
        shutil.copy2(METRICS_PATH, backup_dir / METRICS_PATH.name)

    print("\nTraining completed.")
    print(f"Best validation loss: {best_validation_loss:.4f}")
    print("Held-out test was not evaluated during training.")
    print(f"Model: {MODEL_PATH}")
    print(f"Tokenizer: {TOKENIZER_PATH}")
    print(f"Metrics: {METRICS_PATH}")


if __name__ == "__main__":
    main()
