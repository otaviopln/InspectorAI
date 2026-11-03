"""Validate and split curated InspectorAI examples without synthetic mixing.

Input:  inspector/data/raw/curated_examples.jsonl
Output: inspector/data/processed/prepared_dataset.json
        inspector/data/processed/dataset_report.json

Only records explicitly marked review_status="approved" are eligible for
training. Groups are kept intact across train/validation/test to reduce
near-duplicate leakage.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "inspector" / "data" / "raw" / "curated_examples.jsonl"
PROCESSED_DIR = ROOT / "inspector" / "data" / "processed"
DEFAULT_OUTPUT = PROCESSED_DIR / "prepared_dataset.json"
REPORT_PATH = PROCESSED_DIR / "dataset_report.json"

VALID_TASKS = {"clarify", "format_rcfo"}
VALID_AMBIGUITY = {"vague", "partial", "clear"}
REQUIRED_FIELDS = {
    "id", "task", "ambiguity", "language", "input", "target",
    "source", "group_id", "review_status",
}
RCFO_FIELDS = ("Role:", "Context:", "Focus:", "Output:")


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().casefold())


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset de origem não encontrado: {path}\n"
            "Crie curated_examples.jsonl em inspector/data/raw/."
        )

    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"JSON inválido na linha {line_number}: {exc}"
                ) from exc
            if not isinstance(record, dict):
                raise ValueError(
                    f"A linha {line_number} precisa conter um objeto JSON."
                )
            record["_line"] = line_number
            records.append(record)

    if not records:
        raise ValueError("O arquivo JSONL não contém registros.")
    return records


def validate_record(record: dict[str, Any]) -> None:
    line = record.get("_line", "?")
    missing = REQUIRED_FIELDS - record.keys()
    if missing:
        raise ValueError(f"Linha {line}: campos ausentes: {sorted(missing)}")

    for field in REQUIRED_FIELDS:
        if not isinstance(record[field], str) or not record[field].strip():
            raise ValueError(
                f"Linha {line}: '{field}' precisa ser uma string não vazia."
            )

    if record["task"] not in VALID_TASKS:
        raise ValueError(f"Linha {line}: task inválida: {record['task']}")
    if record["ambiguity"] not in VALID_AMBIGUITY:
        raise ValueError(
            f"Linha {line}: ambiguity inválida: {record['ambiguity']}"
        )
    if record["language"] != "en":
        raise ValueError(
            f"Linha {line}: a versão atual aceita somente language='en'."
        )

    expected_task = "format_rcfo" if record["ambiguity"] == "clear" else "clarify"
    if record["task"] != expected_task:
        raise ValueError(
            f"Linha {line}: ambiguity='{record['ambiguity']}' exige "
            f"task='{expected_task}'."
        )

    prompt = record["input"].strip()
    target = record["target"].strip()
    if "{" in prompt or "}" in prompt:
        raise ValueError(
            f"Linha {line}: input contém placeholder não resolvido."
        )

    if record["task"] == "clarify":
        if not target.startswith("Decision: CLARIFY\n"):
            raise ValueError(
                f"Linha {line}: target de clarify deve começar com "
                "'Decision: CLARIFY'."
            )
        if "Questions:" not in target:
            raise ValueError(f"Linha {line}: target precisa conter 'Questions:'.")
        if record["ambiguity"] == "partial":
            if "Known requirements:" not in target:
                raise ValueError(
                    f"Linha {line}: exemplos partial precisam preservar "
                    "o que já é conhecido em 'Known requirements:'."
                )
            if "Missing information:" not in target:
                raise ValueError(
                    f"Linha {line}: exemplos partial precisam declarar "
                    "'Missing information:'."
                )
    else:
        if not target.startswith("Decision: FORMAT_RCFO\n"):
            raise ValueError(
                f"Linha {line}: target claro deve começar com "
                "'Decision: FORMAT_RCFO'."
            )
        for label in RCFO_FIELDS:
            if not re.search(rf"(?m)^{re.escape(label)}\s*\S", target):
                raise ValueError(
                    f"Linha {line}: campo R.C.F.O. ausente ou vazio: {label}"
                )


def load_approved_records(path: Path) -> tuple[list[dict[str, Any]], int]:
    raw_records = read_jsonl(path)
    approved: list[dict[str, Any]] = []
    pending_count = 0
    seen_ids: set[str] = set()
    seen_inputs: dict[str, str] = {}

    for record in raw_records:
        validate_record(record)
        line = record.get("_line", "?")

        record_id = record["id"].strip()
        if record_id in seen_ids:
            raise ValueError(f"Linha {line}: id duplicado: {record_id}")
        seen_ids.add(record_id)

        input_key = _normalize(record["input"])
        previous_target = seen_inputs.get(input_key)
        if previous_target is not None:
            if _normalize(previous_target) != _normalize(record["target"]):
                raise ValueError(
                    f"Linha {line}: a mesma entrada possui targets diferentes."
                )
            raise ValueError(
                f"Linha {line}: entrada duplicada; remova ou consolide o registro."
            )
        seen_inputs[input_key] = record["target"]

        status = record["review_status"].strip().lower()
        if status not in {"pending_review", "approved", "rejected"}:
            raise ValueError(
                f"Linha {line}: review_status inválido: {status}"
            )
        if status != "approved":
            pending_count += 1
            continue

        clean_record = {
            key: value
            for key, value in record.items()
            if key != "_line"
        }
        approved.append(clean_record)

    if not approved:
        raise ValueError(
            f"Nenhum registro aprovado. Foram encontrados {pending_count} "
            "registros pendentes/reprovados. Revise cada exemplo e altere "
            "'review_status' para 'approved' somente após confirmar a qualidade."
        )

    return approved, pending_count


def grouped_stratified_split(
    records: list[dict[str, Any]],
    train_ratio: float = 0.8,
    validation_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
    min_groups_per_stratum: int = 3,
    attempts: int = 300,
) -> dict[str, list[dict[str, Any]]]:
    """Split by group_id while approximately preserving all task strata.

    A group may contain vague, partial, and/or clear versions of the same
    underlying scenario. The entire group is assigned to exactly one split,
    preventing related examples from leaking across train/validation/test.
    """
    ratios = (train_ratio, validation_ratio, test_ratio)
    if any(r <= 0 for r in ratios) or abs(sum(ratios) - 1.0) > 1e-8:
        raise ValueError("As proporções devem ser positivas e somar 1.0.")
    if attempts < 1:
        raise ValueError("attempts precisa ser maior que zero.")

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    groups_by_stratum: dict[tuple[str, str], set[str]] = defaultdict(set)
    totals: Counter = Counter()

    for record in records:
        group_id = record["group_id"].strip()
        stratum = (record["task"], record["ambiguity"])
        grouped[group_id].append(record)
        groups_by_stratum[stratum].add(group_id)
        totals[stratum] += 1

    for stratum, group_ids in groups_by_stratum.items():
        if len(group_ids) < min_groups_per_stratum:
            raise ValueError(
                f"A categoria {stratum} aparece em apenas {len(group_ids)} "
                f"famílias; são necessárias pelo menos {min_groups_per_stratum} "
                "famílias distintas para representá-la em train, validation e "
                "test sem dividir grupos relacionados."
            )

    split_names = ("train", "validation", "test")
    split_ratios = dict(zip(split_names, ratios))
    strata = list(totals)
    target = {
        split: {stratum: totals[stratum] * split_ratios[split] for stratum in strata}
        for split in split_names
    }
    group_ids = list(grouped)
    best_assignment: dict[str, str] | None = None
    best_score = float("inf")

    group_strata_map = {
        group_id: {
            (record["task"], record["ambiguity"])
            for record in group_records
        }
        for group_id, group_records in grouped.items()
    }

    for attempt in range(attempts):
        rng = random.Random(seed + attempt)
        assignment: dict[str, str] = {}
        actual: dict[str, Counter] = {
            split: Counter() for split in split_names
        }

        # First reserve enough whole groups so that every stratum appears
        # in every split. Rare strata are handled first. A group may satisfy
        # several strata at once (for example, vague and clear versions of
        # the same school-portal scenario).
        strata_order = list(strata)
        rng.shuffle(strata_order)
        strata_order.sort(key=lambda item: len(groups_by_stratum[item]))
        feasible = True

        covered_by_split: dict[str, set[tuple[str, str]]] = {
            split: set() for split in split_names
        }

        def split_already_has(split: str, stratum: tuple[str, str]) -> bool:
            return stratum in covered_by_split[split]

        for stratum in strata_order:
            candidate_ids = list(groups_by_stratum[stratum])
            rng.shuffle(candidate_ids)

            for split in split_names:
                if split_already_has(split, stratum):
                    continue

                candidates = [
                    group_id
                    for group_id in candidate_ids
                    if group_id not in assignment or assignment[group_id] == split
                ]
                if not candidates:
                    feasible = False
                    break

                def coverage_score(group_id: str) -> tuple[int, int]:
                    newly_covered = sum(
                        1
                        for other_stratum in group_strata_map[group_id]
                        if not split_already_has(split, other_stratum)
                    )
                    # Prefer a group that covers several currently missing
                    # labels; prefer smaller groups when coverage is equal.
                    return (newly_covered, -len(grouped[group_id]))

                best_coverage = max(coverage_score(group_id) for group_id in candidates)
                tied_candidates = [
                    group_id
                    for group_id in candidates
                    if coverage_score(group_id) == best_coverage
                ]
                chosen_group = rng.choice(tied_candidates)
                assignment[chosen_group] = split
                covered_by_split[split].update(group_strata_map[chosen_group])

        if not feasible:
            continue

        # Account for the reserved groups.
        for group_id, split in assignment.items():
            actual[split].update(
                Counter(
                    (record["task"], record["ambiguity"])
                    for record in grouped[group_id]
                )
            )

        # Assign remaining groups by minimizing normalized deviation from
        # the requested per-stratum proportions.
        remaining_ids = [group_id for group_id in grouped if group_id not in assignment]
        rng.shuffle(remaining_ids)

        def rarity(group_id: str) -> tuple[float, int]:
            weight = sum(
                1.0 / len(groups_by_stratum[stratum])
                for stratum in group_strata_map[group_id]
            )
            return (-weight, -len(grouped[group_id]))

        remaining_ids.sort(key=rarity)

        for group_id in remaining_ids:
            group_counts = Counter(
                (record["task"], record["ambiguity"])
                for record in grouped[group_id]
            )
            deltas: dict[str, float] = {}

            for split in split_names:
                delta = 0.0
                for stratum, count in group_counts.items():
                    before = actual[split][stratum]
                    desired = target[split][stratum]
                    after = before + count
                    scale = max(desired, 1.0)
                    delta += (
                        (after - desired) ** 2 - (before - desired) ** 2
                    ) / scale
                deltas[split] = delta

            minimum = min(deltas.values())
            tied = [split for split, delta in deltas.items() if abs(delta - minimum) < 1e-9]
            chosen = rng.choice(tied)
            assignment[group_id] = chosen
            actual[chosen].update(group_counts)

        score = 0.0
        for split in split_names:
            for stratum in strata:
                desired = target[split][stratum]
                observed = actual[split][stratum]
                score += ((observed - desired) ** 2) / max(desired, 1.0)

        if score < best_score:
            best_score = score
            best_assignment = assignment

    if best_assignment is None:
        raise ValueError(
            "Não foi possível criar três divisões sem separar grupos e mantendo "
            "vague, partial e clear em todos os conjuntos. Adicione mais famílias "
            "independentes ou revise os group_id."
        )

    result: dict[str, list[dict[str, Any]]] = {
        "train": [], "validation": [], "test": []
    }
    for group_id, split in best_assignment.items():
        result[split].extend(grouped[group_id])

    rng = random.Random(seed)
    for split_records in result.values():
        rng.shuffle(split_records)

    # Defensive check: no group can occur in more than one split.
    group_sets = {
        name: {record["group_id"] for record in split_records}
        for name, split_records in result.items()
    }
    if group_sets["train"] & group_sets["validation"]:
        raise RuntimeError("Vazamento de group_id entre train e validation.")
    if group_sets["train"] & group_sets["test"]:
        raise RuntimeError("Vazamento de group_id entre train e test.")
    if group_sets["validation"] & group_sets["test"]:
        raise RuntimeError("Vazamento de group_id entre validation e test.")

    return result


def build_report(
    splits: dict[str, list[dict[str, Any]]],
    pending_count: int,
    seed: int,
) -> dict[str, Any]:
    report: dict[str, Any] = {
        "schema_version": "inspector.dataset.v2",
        "seed": seed,
        "pending_or_rejected_records_excluded": pending_count,
        "counts": {},
        "by_task": {},
        "by_ambiguity": {},
        "unique_groups": {},
    }

    for split_name, records in splits.items():
        report["counts"][split_name] = len(records)
        report["unique_groups"][split_name] = len(
            {record["group_id"] for record in records}
        )
        report["by_task"][split_name] = dict(
            Counter(record["task"] for record in records)
        )
        report["by_ambiguity"][split_name] = dict(
            Counter(record["ambiguity"] for record in records)
        )

    report["counts"]["total"] = sum(report["counts"].values())
    return report


def prepare_dataset(
    input_path: Path = DEFAULT_INPUT,
    output_path: Path = DEFAULT_OUTPUT,
    seed: int = 42,
) -> tuple[dict[str, Any], dict[str, Any]]:
    records, pending_count = load_approved_records(input_path)
    splits = grouped_stratified_split(records, seed=seed)

    source_digest = hashlib.sha256(input_path.read_bytes()).hexdigest()
    prepared = {
        "schema_version": "inspector.dataset.v2",
        "language": "en",
        "decision_tokens": ["CLARIFY", "FORMAT_RCFO"],
        "split_strategy": "grouped_stratified_80_10_10",
        "seed": seed,
        "source_sha256": source_digest,
        "splits": splits,
    }
    report = build_report(splits, pending_count, seed)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as stream:
        json.dump(prepared, stream, ensure_ascii=False, indent=2)

    # Also write one JSONL file per split for convenient inspection/tooling.
    for split_name, split_records in splits.items():
        split_path = output_path.parent / f"{split_name}.jsonl"
        with split_path.open("w", encoding="utf-8") as stream:
            for record in split_records:
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")

    report["source_sha256"] = source_digest
    report_path = output_path.parent / "dataset_report.json"
    with report_path.open("w", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)

    return prepared, report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate and split curated InspectorAI training examples."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    prepared, report = prepare_dataset(args.input, args.output, args.seed)
    print("InspectorAI dataset preparation completed.")
    print(f"Approved examples: {report['counts']['total']}")
    print(f"Train:      {report['counts']['train']}")
    print(f"Validation: {report['counts']['validation']}")
    print(f"Test:       {report['counts']['test']}")
    print(
        "Excluded pending/rejected records:",
        report["pending_or_rejected_records_excluded"],
    )
    print(f"Prepared dataset: {args.output}")
    print(f"Report: {args.output.parent / 'dataset_report.json'}")


if __name__ == "__main__":
    main()
