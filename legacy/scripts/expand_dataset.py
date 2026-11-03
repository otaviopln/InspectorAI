"""Deprecated entry point for the old combinatorial data generator.

The previous implementation independently combined role/context/focus/output
banks and could create contradictory labels. Use the curated JSONL workflow:

1. Review inspector/data/raw/curated_examples.jsonl
2. Set review_status="approved" only after reviewing each record
3. Run python scripts/prepare_dataset.py
4. Run python scripts/train.py
"""


def main() -> None:
    raise SystemExit(
        "expand_dataset.py foi descontinuado porque gerava combinações "
        "semanticamente inconsistentes. Use curated_examples.jsonl e execute "
        "python scripts/prepare_dataset.py. Consulte README_DATA_V2.md."
    )


if __name__ == "__main__":
    main()
