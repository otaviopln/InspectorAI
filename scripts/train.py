"""Training loop for Inspector AI."""

from __future__ import annotations

import json
import os
import shutil
import random
import sys

from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn

# ---------------------------------------------------------------
# Diretórios do projeto
# ---------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]

SCRIPTS_DIR = ROOT / "scripts"

if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

# ---------------------------------------------------------------
# Importações locais
# ---------------------------------------------------------------
from tokenizer import (
    BPETokenizer,
    SOS_IDX,
    EOS_IDX,
    PAD_IDX,
    UNK_IDX,
    SPECIAL_TOKENS,
)

from model import InspectorTransformer

# ---------------------------------------------------------------
# Caminhos
# ---------------------------------------------------------------
OA_DATASET_PATH = (
    ROOT / "inspector" / "data" / "processed"
    / "openassistant_dataset.json"
)

MOCK_DATASET_PATH = (
    ROOT / "inspector" / "data" / "processed"
    / "mock_dataset.json"
)

TOKENIZER_PATH = (
    ROOT / "inspector" / "models" / "tokenizer_vocab.json"
)

MODEL_PATH = (
    ROOT / "inspector" / "models" / "inspector_model.pth"
)

DATASET_PATH = (
    OA_DATASET_PATH
    if OA_DATASET_PATH.exists()
    else MOCK_DATASET_PATH
)

# ---------------------------------------------------------------
# Configuração do dispositivo
# ---------------------------------------------------------------
_FORCED_DEVICE = os.environ.get(
    "INSPECTOR_DEVICE", ""
).strip().lower()

# ---------------------------------------------------------------
# Configuração para treinamento em GPU CUDA
# ---------------------------------------------------------------

VOCAB_SIZE = 3000

D_MODEL = 128
NUM_HEADS = 4
NUM_LAYERS = 2
D_FF = 512

MAX_LEN = 128

BATCH_SIZE = 32
EPOCHS = 10
LR = 5e-4

MAX_PAIRS = 12000
RETRAIN_TOKENIZER = True

# Validação e Early Stopping
VALIDATION_RATIO = 0.10
RANDOM_SEED = 42

EARLY_STOPPING_PATIENCE = 4
MIN_DELTA = 0.0001


def pick_device() -> torch.device:
    """Return the best available torch device.

    Priority: user env var INSPECTOR_DEVICE > CUDA > DirectML > CPU.
    """
    if _FORCED_DEVICE in {"cpu", "cuda", "directml"}:
        if _FORCED_DEVICE == "cuda" and not torch.cuda.is_available():
            print("WARNING: INSPECTOR_DEVICE=cuda requested but CUDA is unavailable. Falling back to CPU.")
            return torch.device("cpu")
        if _FORCED_DEVICE == "directml":
            try:
                import torch_directml  # type: ignore
                return torch_directml.device()
            except Exception as exc:  # pragma: no cover
                print(f"WARNING: DirectML unavailable ({exc}). Falling back to CPU.")
                return torch.device("cpu")
        return torch.device(_FORCED_DEVICE)

    if torch.cuda.is_available():
        return torch.device("cuda")

    try:
        import torch_directml  # type: ignore
        if torch_directml.is_available():
            return torch_directml.device()
    except Exception:
        pass

    return torch.device("cpu")


DEVICE = pick_device()

def load_dataset_splits(
    path: Path,
) -> Dict[str, List[Tuple[str, str]]]:
    """Carrega os conjuntos classificados do arquivo JSON."""

    if not path.exists():
        raise FileNotFoundError(
            f"Dataset não encontrado: {path}"
        )

    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    raw_splits = data.get("splits", {})

    if not isinstance(raw_splits, dict):
        raise ValueError(
            "Formato inválido: 'splits' precisa ser um objeto."
        )

    result: Dict[str, List[Tuple[str, str]]] = {}

    for split_name, entries in raw_splits.items():
        if not isinstance(entries, list):
            continue

        pairs = []

        for entry in entries:
            if not isinstance(entry, dict):
                continue

            user_input = str(
                entry.get("input", "")
            ).strip()

            target = str(
                entry.get("target", "")
            ).strip()

            if user_input and target:
                pairs.append((user_input, target))

        if pairs:
            result[split_name] = pairs

    if not result:
        raise ValueError(
            "Nenhum exemplo válido foi encontrado no dataset."
        )

    return result

def split_dataset(
    dataset_splits: Dict[str, List[Tuple[str, str]]],
    validation_ratio: float = VALIDATION_RATIO,
    seed: int = RANDOM_SEED,
) -> Tuple[
    List[Tuple[str, str]],
    List[Tuple[str, str]],
]:
    """
    Separa os exemplos em treinamento e validação.

    A divisão ocorre separadamente para cada categoria do dataset.
    """

    if not 0 < validation_ratio < 1:
        raise ValueError(
            "validation_ratio precisa estar entre 0 e 1."
        )

    rng = random.Random(seed)

    train_pairs: List[Tuple[str, str]] = []
    validation_pairs: List[Tuple[str, str]] = []

    for category_pairs in dataset_splits.values():
        pairs = list(category_pairs)
        rng.shuffle(pairs)

        # Uma categoria com somente um exemplo permanece no treino.
        if len(pairs) < 2:
            train_pairs.extend(pairs)
            continue

        validation_count = max(
            1,
            round(len(pairs) * validation_ratio),
        )

        # Garante pelo menos um exemplo de treinamento por categoria.
        validation_count = min(
            validation_count,
            len(pairs) - 1,
        )

        validation_pairs.extend(
            pairs[:validation_count]
        )

        train_pairs.extend(
            pairs[validation_count:]
        )

    # Se todas as categorias tinham apenas um exemplo,
    # desloca um exemplo para validação, desde que seja possível.
    if not validation_pairs:
        if len(train_pairs) < 2:
            raise ValueError(
                "São necessários pelo menos dois exemplos "
                "para criar treinamento e validação."
            )

        validation_pairs.append(train_pairs.pop())

    rng.shuffle(train_pairs)
    rng.shuffle(validation_pairs)

    if not train_pairs or not validation_pairs:
        raise ValueError(
            "A divisão produziu um conjunto vazio."
        )

    return train_pairs, validation_pairs

def limit_dataset_splits(
    dataset_splits: Dict[str, List[Tuple[str, str]]],
    max_pairs: int,
    seed: int = RANDOM_SEED,
) -> Dict[str, List[Tuple[str, str]]]:
    """
    Limita o dataset preservando suas categorias.

    Garante, sempre que possível:
        - Pelo menos um exemplo de treinamento por categoria.
        - Pelo menos um exemplo de validação por categoria
          que possua dois ou mais registros.
    """

    if max_pairs < 1:
        raise ValueError(
            "MAX_PAIRS precisa ser maior que zero."
        )

    available = {
        name: list(pairs)
        for name, pairs in dataset_splits.items()
        if pairs
    }

    if not available:
        raise ValueError(
            "Não existem categorias com exemplos válidos."
        )

    total = sum(
        len(pairs)
        for pairs in available.values()
    )

    if max_pairs >= total:
        return available

    # Reserva dois exemplos para categorias que permitem
    # ter treinamento e validação.
    quotas = {
        name: min(2, len(pairs))
        for name, pairs in available.items()
    }

    minimum_required = sum(quotas.values())

    if max_pairs < minimum_required:
        raise ValueError(
            f"MAX_PAIRS precisa ser pelo menos {minimum_required} "
            "para preservar as categorias e permitir uma divisão "
            "adequada entre treinamento e validação."
        )

    remaining = max_pairs - minimum_required

    # Distribui as vagas restantes proporcionalmente.
    while remaining > 0:
        candidates = [
            name
            for name, pairs in available.items()
            if quotas[name] < len(pairs)
        ]

        if not candidates:
            break

        selected = max(
            candidates,
            key=lambda name: (
                len(available[name]) / (quotas[name] + 1)
            ),
        )

        quotas[selected] += 1
        remaining -= 1

    # Embaralha e seleciona os exemplos de cada categoria.
    rng = random.Random(seed)

    limited = {}

    for name, pairs in available.items():
        shuffled = list(pairs)
        rng.shuffle(shuffled)

        limited[name] = shuffled[:quotas[name]]

    return limited

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
    batch: List[Tuple[str, str]],
    tokenizer: BPETokenizer,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Prepara entrada, entrada do decoder e alvos."""

    if MAX_LEN < 2:
        raise ValueError(
            "MAX_LEN precisa ser pelo menos 2."
        )

    src_ids = []
    tgt_input_ids = []
    tgt_label_ids = []

    for user_input, target in batch:
        source = tokenizer.encode(user_input)[:MAX_LEN]

        if not source:
            source = [UNK_IDX]

        src_ids.append(source)

        target_tokens = tokenizer.encode(
            target
        )[:MAX_LEN - 1]

        full_target = (
            [SOS_IDX]
            + target_tokens
            + [EOS_IDX]
        )

        tgt_input_ids.append(full_target[:-1])
        tgt_label_ids.append(full_target[1:])

    src = pad_sequences(src_ids)
    tgt_input = pad_sequences(tgt_input_ids)
    tgt_labels = pad_sequences(tgt_label_ids)

    return src, tgt_input, tgt_labels

def evaluate_loss(
    model: InspectorTransformer,
    pairs: List[Tuple[str, str]],
    tokenizer: BPETokenizer,
    criterion: nn.Module,
) -> float:
    """Calcula a loss média por token no conjunto de validação."""

    if not pairs:
        raise ValueError(
            "O conjunto de validação está vazio."
        )

    model.eval()

    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for i in range(0, len(pairs), BATCH_SIZE):
            batch = pairs[i:i + BATCH_SIZE]

            src, tgt_input, tgt_labels = collate_batch(
                batch,
                tokenizer,
            )

            src = src.to(DEVICE)
            tgt_input = tgt_input.to(DEVICE)
            tgt_labels = tgt_labels.to(DEVICE)

            logits = model(
                src,
                tgt_input,
            )

            loss_sum = criterion(
                logits.reshape(-1, logits.size(-1)),
                tgt_labels.reshape(-1),
            )

            token_count = int(
                (tgt_labels != PAD_IDX).sum().item()
            )

            total_loss += loss_sum.item()
            total_tokens += token_count

    if total_tokens == 0:
        raise ValueError(
            "Nenhum token válido no conjunto de validação."
        )

    return total_loss / total_tokens


def main() -> None:
    backup_dir_env = os.environ.get(
    "INSPECTOR_BACKUP_DIR"
    )

    BACKUP_DIR = (
        Path(backup_dir_env)
        if backup_dir_env
        else None
    )

    if BACKUP_DIR is not None:
        BACKUP_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )

        print(
            f"Backups serão salvos em: {BACKUP_DIR}"
        )
    # 1. Carrega o dataset.
    dataset_splits = load_dataset_splits(DATASET_PATH)

    print("MAX_PAIRS configurado:", MAX_PAIRS)

    # 2. Aplica o limite antes da divisão.
    if MAX_PAIRS is not None:
        dataset_splits = limit_dataset_splits(
            dataset_splits,
            MAX_PAIRS,
            seed=RANDOM_SEED,
        )

    total_examples = sum(
        len(pairs)
        for pairs in dataset_splits.values()
    )

    print("Total após aplicação do limite:", total_examples)

    # 3. Divide em treinamento e validação.
    train_pairs, validation_pairs = split_dataset(
        dataset_splits,
        validation_ratio=VALIDATION_RATIO,
        seed=RANDOM_SEED,
    )

    print(f"Training examples: {len(train_pairs)}")
    print(f"Validation examples: {len(validation_pairs)}")

    # Continue aqui com o restante do main()
    # que já implementamos.

    # ---------------------------------------------------------------
    # Treinamento ou reutilização do tokenizer
    # ---------------------------------------------------------------

    train_texts = [
        text
        for pair in train_pairs
        for text in pair
    ]

    if (
        TOKENIZER_PATH.exists()
        and not RETRAIN_TOKENIZER
    ):

        print("Loading previously trained tokenizer...")

        tokenizer = BPETokenizer.load(
            TOKENIZER_PATH
        )

    else:

        print("Training new Rust-accelerated BPE tokenizer...")

        tokenizer = BPETokenizer(
            vocab_size=VOCAB_SIZE
        )

        tokenizer.train(
            train_texts,
        )

        tokenizer.save(
            TOKENIZER_PATH
        )

    # Tamanho real do vocabulário.
    vocab_size = len(tokenizer.vocab)

    print(
        f"Final tokenizer vocabulary: {vocab_size}"
    )

    if vocab_size <= EOS_IDX:
        raise ValueError(
            "Invalid tokenizer vocabulary."
        )

    # ---------------------------------------------------------------
    # 3. Cria modelo, loss e otimizador
    # ---------------------------------------------------------------
    model_config = {
        "vocab_size": vocab_size,
        "d_model": D_MODEL,
        "num_heads": NUM_HEADS,
        "num_layers": NUM_LAYERS,
        "d_ff": D_FF,
        "max_len": MAX_LEN,
        "pad_idx": PAD_IDX,
    }

    model = InspectorTransformer(
        **model_config,
    ).to(DEVICE)

    # Soma as perdas para calcular a média por token.
    criterion = nn.CrossEntropyLoss(
        ignore_index=PAD_IDX,
        reduction="sum",
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LR,
    )

    # ---------------------------------------------------------------
    # 4. Treinamento e validação
    # ---------------------------------------------------------------
    best_validation_loss = float("inf")
    epochs_without_improvement = 0

    rng = random.Random(RANDOM_SEED)

    print(
        f"Starting training: epochs={EPOCHS}, "
        f"batch_size={BATCH_SIZE}, "
        f"learning_rate={LR}"
    )

    for epoch in range(1, EPOCHS + 1):
        model.train()

        rng.shuffle(train_pairs)

        total_train_loss = 0.0
        total_train_tokens = 0

        for i in range(0, len(train_pairs), BATCH_SIZE):
            batch = train_pairs[i:i + BATCH_SIZE]

            src, tgt_input, tgt_labels = collate_batch(
                batch,
                tokenizer,
            )

            src = src.to(DEVICE)
            tgt_input = tgt_input.to(DEVICE)
            tgt_labels = tgt_labels.to(DEVICE)

            logits = model(
                src,
                tgt_input,
            )

            loss_sum = criterion(
                logits.reshape(-1, logits.size(-1)),
                tgt_labels.reshape(-1),
            )

            token_count = int(
                (tgt_labels != PAD_IDX).sum().item()
            )

            if token_count == 0:
                continue

            # Backpropaga a média por token.
            loss = loss_sum / token_count

            optimizer.zero_grad(set_to_none=True)
            loss.backward()

            # Evita gradientes excessivamente grandes.
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0,
            )

            optimizer.step()

            total_train_loss += loss_sum.item()
            total_train_tokens += token_count

        if total_train_tokens == 0:
            raise RuntimeError(
                "Nenhum token válido foi utilizado no treinamento."
            )

        train_loss = (
            total_train_loss / total_train_tokens
        )

        validation_loss = evaluate_loss(
            model,
            validation_pairs,
            tokenizer,
            criterion,
        )

        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"Train loss: {train_loss:.4f} | "
            f"Validation loss: {validation_loss:.4f}"
        )

        # -----------------------------------------------------------
        # 5. Salva somente quando há melhora na validação
        # -----------------------------------------------------------
        improved = (
            validation_loss
            < best_validation_loss - MIN_DELTA
        )

        if improved:
            best_validation_loss = validation_loss
            epochs_without_improvement = 0

            MODEL_PATH.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            checkpoint = {
                "format_version": 2,
                "model_state_dict": {
                    key: value.detach().cpu()
                    for key, value in model.state_dict().items()
                },
                "model_config": model_config,
                "training_config": {
                    "epoch": epoch,
                    "epochs": EPOCHS,
                    "batch_size": BATCH_SIZE,
                    "learning_rate": LR,
                    "validation_loss": validation_loss,
                    "random_seed": RANDOM_SEED,
                },
                "optimizer_state_dict": optimizer.state_dict(),
            }

            torch.save(
                checkpoint,
                MODEL_PATH,
            )

            if BACKUP_DIR is not None:
              shutil.copy2(
                  MODEL_PATH,
                  BACKUP_DIR / "inspector_model.pth",
              )

            print(
                f"Best checkpoint saved. "
                f"Validation loss: {validation_loss:.4f}"
            )

        else:
            epochs_without_improvement += 1

            print(
                "Validation did not improve: "
                f"{epochs_without_improvement}/"
                f"{EARLY_STOPPING_PATIENCE}"
            )

        # -----------------------------------------------------------
        # 6. Early stopping
        # -----------------------------------------------------------
        if (
            epochs_without_improvement
            >= EARLY_STOPPING_PATIENCE
        ):
            print(
                "Early stopping: validation loss did not improve."
            )
            break

    print("\nTraining completed.")
    print(
        f"Best validation loss: {best_validation_loss:.4f}"
    )
    print(f"Checkpoint: {MODEL_PATH}")
    print(f"Tokenizer: {TOKENIZER_PATH}")

if __name__ == "__main__":
    main()
