
"""Inspect tokenized sequence lengths in the prepared InspectorAI dataset."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT / "scripts"

if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from tokenizer import BPETokenizer

DATASET_PATH = (
    ROOT / "inspector" / "data" / "processed"
    / "prepared_dataset.json"
)

MODELS_DIR = ROOT / "inspector" / "models"

STAGING_TOKENIZER = MODELS_DIR / ".tokenizer_vocab.json.tmp"
FINAL_TOKENIZER = MODELS_DIR / "tokenizer_vocab.json"

# O tokenizador temporário pertence à tentativa de treinamento atual.
# Ele deve ser priorizado para medir os mesmos tokens usados no erro.
TOKENIZER_PATH = (
    STAGING_TOKENIZER
    if STAGING_TOKENIZER.exists()
    else FINAL_TOKENIZER
)

if not DATASET_PATH.exists():
    raise FileNotFoundError(
        f"Dataset preparado não encontrado: {DATASET_PATH}"
    )

if not TOKENIZER_PATH.exists():
    raise FileNotFoundError(
        "Tokenizador não encontrado. "
        "Verifique os arquivos temporários e definitivos."
    )

tokenizer = BPETokenizer.load(TOKENIZER_PATH)

import json

with DATASET_PATH.open("r", encoding="utf-8") as file:
    dataset = json.load(file)

def percentile(values, percent):
    values = sorted(values)
    index = round((len(values) - 1) * percent)
    return values[index]

for split_name in ("train", "validation", "test"):
    records = dataset["splits"][split_name]

    lengths = []
    oversized = []

    for record in records:
        source_tokens = len(tokenizer.encode(record["input"]))
        target_tokens = len(tokenizer.encode(record["target"]))

        # O decoder recebe SOS + target.
        required_length = max(
            source_tokens,
            target_tokens + 1,
        )

        lengths.append(required_length)

        if required_length > 128:
            oversized.append({
                "id": record.get("id", "sem_id"),
                "group_id": record.get("group_id", "sem_group"),
                "source_tokens": source_tokens,
                "target_tokens": target_tokens,
                "required_length": required_length,
                "input": record["input"],
                "target": record["target"],
            })

    print("\n" + "=" * 65)
    print(f"DATASET: {split_name.upper()}")
    print("=" * 65)

    print("Total de exemplos:", len(records))
    print("Maior comprimento:", max(lengths))
    print("P50:", percentile(lengths, 0.50))
    print("P90:", percentile(lengths, 0.90))
    print("P95:", percentile(lengths, 0.95))
    print("P99:", percentile(lengths, 0.99))
    print("Exemplos acima de 128:", len(oversized))

    for item in sorted(
        oversized,
        key=lambda x: x["required_length"],
        reverse=True,
    )[:10]:

        print("\nID:", item["id"])
        print("Grupo:", item["group_id"])
        print("Tokens da entrada:", item["source_tokens"])
        print("Tokens do alvo:", item["target_tokens"])
        print("Comprimento necessário:", item["required_length"])
        print("Entrada:", item["input"][:180])
        print("Alvo:", item["target"][:300])
