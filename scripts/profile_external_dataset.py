"""Profile a small, reproducible sample of a Hugging Face dataset.

This script streams a bounded sample. It does not download the complete dataset,
modify source records, approve a license, or create InspectorAI training targets.

Install:
    python -m pip install datasets huggingface_hub

Example:
    python scripts/profile_external_dataset.py ^
        --dataset cnmoro/GPT4-500k-Augmented-PTBR-Clean ^
        --sample-size 200 ^
        --seed 42
"""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
from collections import Counter
from datetime import datetime, timezone
from itertools import islice
from pathlib import Path
from typing import Any

DEFAULT_DATASET = "cnmoro/GPT4-500k-Augmented-PTBR-Clean"
DEFAULT_SPLIT = "train"
DEFAULT_OUTPUT_DIR = (
    Path(__file__).resolve().parents[1]
    / "inspector"
    / "data"
    / "external"
    / "cache"
    / "gpt4_500k_ptbr"
)
INPUT_KEYS = ("instruction", "input", "prompt", "question", "utt_br", "utt", "text")
TARGET_KEYS = ("response", "output", "target", "answer")


def normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", value.casefold()).strip()


def percentile(values: list[int], fraction: float) -> int | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, math.ceil(fraction * len(ordered)) - 1)
    return ordered[index]


def first_present_key(keys: set[str], candidates: tuple[str, ...]) -> str | None:
    for candidate in candidates:
        if candidate in keys:
            return candidate
    return None


def json_safe(value: Any) -> Any:
    """Convert common Hugging Face/Python values into JSON-compatible values."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return str(value)


def scan_field_values(
    stream: Any,
    field: str,
) -> dict[str, Any]:
    """Count every value of a field across one streamed split."""
    counts: Counter[str] = Counter()
    missing = 0
    total = 0

    for row in stream:
        total += 1
        value = row.get(field)
        if value is None:
            missing += 1
        else:
            counts[str(value)] += 1

    return {
        "field": field,
        "records_scanned": total,
        "missing_or_null": missing,
        "value_counts": dict(counts.most_common()),
        "note": "Contagem integral do split em streaming; nenhum registro foi salvo.",
    }


def build_profile(
    rows: list[dict[str, Any]],
    dataset_id: str,
    revision: str,
    split: str,
    seed: int,
    buffer_size: int,
    filter_field: str | None,
    filter_value: str | None,
) -> dict[str, Any]:
    keys = sorted({str(key) for row in rows for key in row})
    field_stats: dict[str, Any] = {}

    for key in keys:
        values = [row.get(key) for row in rows]
        strings = [value for value in values if isinstance(value, str)]
        lengths = [len(value) for value in strings]
        non_null = [value for value in values if value is not None]
        category_counts = Counter(str(value) for value in non_null)
        field_stats[key] = {
            "category_counts": (
                dict(category_counts.most_common(30))
                if len(category_counts) <= 50 else None
            ),
            "missing_or_null": sum(value is None for value in values),
            "empty_strings": sum(
                isinstance(value, str) and not value.strip() for value in values
            ),
            "non_string_values": sum(
                value is not None and not isinstance(value, str) for value in values
            ),
            "string_values": len(strings),
            "string_length_chars": {
                "min": min(lengths) if lengths else None,
                "median": int(statistics.median(lengths)) if lengths else None,
                "p95": percentile(lengths, 0.95),
                "max": max(lengths) if lengths else None,
            },
        }

    input_key = first_present_key(set(keys), INPUT_KEYS)
    target_key = first_present_key(set(keys), TARGET_KEYS)

    normalized_inputs: list[str] = []
    empty_inputs = 0
    empty_targets = 0
    ptbr_diacritic_heuristic = 0

    for row in rows:
        input_value = row.get(input_key) if input_key else None
        target_value = row.get(target_key) if target_key else None

        if isinstance(input_value, str) and input_value.strip():
            normalized_inputs.append(normalize_text(input_value))
            if re.search(r"[ãõáéíóúâêôç]", input_value.casefold()):
                ptbr_diacritic_heuristic += 1
        else:
            empty_inputs += 1

        if not isinstance(target_value, str) or not target_value.strip():
            empty_targets += 1

    duplicates = len(normalized_inputs) - len(set(normalized_inputs))
    return {
        "profile_version": "1.0",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": dataset_id,
        "resolved_revision": revision,
        "split": split,
        "sample": {
            "records_inspected": len(rows),
            "sampling_method": "optional exact-value filter, then streaming shuffle with bounded buffer",
            "seed": seed,
            "buffer_size": buffer_size,
            "filter": (
                {"field": filter_field, "value": filter_value}
                if filter_field is not None else None
            ),
            "note": (
                "A amostra é exploratória; não representa necessariamente "
                "a distribuição completa do dataset."
            ),
        },
        "schema": {
            "fields": keys,
            "input_field_detected": input_key,
            "target_field_detected": target_key,
            "field_statistics": field_stats,
        },
        "quality_checks": {
            "empty_input_records": empty_inputs,
            "empty_target_records": empty_targets if target_key else None,
            "duplicate_inputs_in_sample": duplicates if input_key else None,
            "records_with_ptbr_diacritic_characters": (
                ptbr_diacritic_heuristic if input_key else None
            ),
            "language_heuristic_warning": (
                "A presença de caracteres acentuados não confirma pt-BR. "
                "A classificação de idioma/variante exige detector e revisão."
            ),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Inspeciona uma amostra limitada de um dataset Hugging Face."
    )
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--split", default=DEFAULT_SPLIT)
    parser.add_argument("--revision", default="main")
    parser.add_argument("--sample-size", type=int, default=200)
    parser.add_argument("--buffer-size", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--count-field",
        help="Conta todos os valores de um campo no split inteiro e encerra; "
        "por exemplo: --count-field lang.",
    )
    parser.add_argument(
        "--filter-field",
        help="Campo a filtrar antes da amostragem, por exemplo lang.",
    )
    parser.add_argument(
        "--filter-value",
        help="Valor textual exato do campo, por exemplo pt.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    if args.sample_size < 1:
        parser.error("--sample-size precisa ser maior que zero.")
    if args.buffer_size < args.sample_size:
        parser.error("--buffer-size deve ser maior ou igual a --sample-size.")
    if (args.filter_field is None) != (args.filter_value is None):
        parser.error("--filter-field e --filter-value devem ser usados juntos.")

    try:
        from datasets import load_dataset
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise SystemExit(
            "Dependências ausentes. Execute: "
            "python -m pip install datasets huggingface_hub"
        ) from exc

    print(f"Consultando metadados de {args.dataset}...")
    info = HfApi().dataset_info(args.dataset, revision=args.revision)
    resolved_revision = info.sha or args.revision
    print(f"Revisão resolvida: {resolved_revision}")
    print(f"Carregando split {args.split!r} em streaming...")

    stream = load_dataset(
        args.dataset,
        split=args.split,
        streaming=True,
        revision=resolved_revision,
    )

    if args.count_field:
        counts = scan_field_values(stream, args.count_field)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        counts_path = args.output_dir / f"{args.count_field}_value_counts.json"
        counts_path.write_text(
            json.dumps(
                {
                    "dataset": args.dataset,
                    "resolved_revision": resolved_revision,
                    "split": args.split,
                    "created_at_utc": datetime.now(timezone.utc).isoformat(),
                    **counts,
                },
                ensure_ascii=False,
                indent=2,
            ) + "\\n",
            encoding="utf-8",
        )
        print(f"Registros examinados: {counts['records_scanned']}")
        print(f"Valores de {args.count_field}:")
        for value, count in counts["value_counts"].items():
            print(f"  {value}: {count}")
        print(f"Ausentes: {counts['missing_or_null']}")
        print(f"Relatório: {counts_path}")
        return 0

    if args.filter_field is not None:
        field = args.filter_field
        value = args.filter_value
        stream = stream.filter(lambda row: str(row.get(field)) == value)

    stream = stream.shuffle(seed=args.seed, buffer_size=args.buffer_size)

    rows = [json_safe(row) for row in islice(iter(stream), args.sample_size)]
    if not rows:
        raise SystemExit(
            "Nenhum registro foi retornado. Confira dataset, split e acesso."
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    sample_path = args.output_dir / "sample.jsonl"
    profile_path = args.output_dir / "profile.json"

    with sample_path.open("w", encoding="utf-8", newline="\n") as output:
        for row in rows:
            output.write(json.dumps(row, ensure_ascii=False) + "\n")

    profile = build_profile(
        rows=rows,
        dataset_id=args.dataset,
        revision=resolved_revision,
        split=args.split,
        seed=args.seed,
        buffer_size=args.buffer_size,
        filter_field=args.filter_field,
        filter_value=args.filter_value,
    )
    profile_path.write_text(
        json.dumps(profile, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print("\nPerfil concluído.")
    print(f"Registros inspecionados: {len(rows)}")
    print(f"Campos: {', '.join(profile['schema']['fields'])}")
    print(f"Campo de entrada: {profile['schema']['input_field_detected']}")
    print(f"Campo de resposta: {profile['schema']['target_field_detected']}")
    print(f"Amostra: {sample_path}")
    print(f"Relatório: {profile_path}")
    print(
        "\nAtenção: o perfil é exploratório. Ele não valida licença, "
        "não aprova conteúdo e não gera dados de treinamento."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
