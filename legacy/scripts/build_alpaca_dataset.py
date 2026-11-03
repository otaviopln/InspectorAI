"""Dataset builder: Alpaca -> Inspector AI format.

Tries to load the Stanford Alpaca dataset from HuggingFace (`tatsu-lab/alpaca`)
and converts it into Inspector AI training pairs:

  * Clear instructions  ->  R.C.F.O formatted targets
  * Vague / short instructions  ->  strategic clarification questions

If HuggingFace datasets library or internet access is unavailable, the script
falls back to a small built-in Alpaca-style sample so the pipeline can still
be tested locally.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import List, Tuple

ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DATA_DIR = ROOT / "inspector" / "data" / "processed"
RAW_DATA_DIR = ROOT / "inspector" / "data" / "raw"
OUTPUT_PATH = PROCESSED_DATA_DIR / "alpaca_dataset.json"

# Heuristic signals that suggest an instruction is vague / under-specified.
VAGUE_SIGNALS = (
    "something",
    "stuff",
    "thing",
    "fix it",
    "make it better",
    "help me",
    "i need",
    "can you",
    "maybe",
    "i guess",
    "not sure",
    "write something",
    "do something",
    "create something",
)

# Verbs that usually indicate a clear, actionable task.
CLEAR_VERBS = (
    "write",
    "create",
    "design",
    "build",
    "analyze",
    "review",
    "implement",
    "refactor",
    "migrate",
    "debug",
    "audit",
    "draft",
    "document",
    "prepare",
    "develop",
    "migrate",
    "optimize",
    "generate",
    "produce",
    "compose",
)


def _is_vague(text: str) -> bool:
    """Heuristic: short or vague-sounding instructions."""
    lowered = text.lower().strip()
    if len(lowered.split()) < 5:
        return True
    return any(signal in lowered for signal in VAGUE_SIGNALS)


def _is_clear(text: str) -> bool:
    """Heuristic: contains an action verb and enough detail."""
    lowered = text.lower().strip()
    if len(lowered.split()) < 8:
        return False
    return any(verb in lowered for verb in CLEAR_VERBS) and not _is_vague(text)


def _format_rcfo(instruction: str, response: str) -> str:
    """Best-effort conversion of an Alpaca pair into R.C.F.O fields.

    Since Alpaca responses are free-form, we attempt to extract role/context
    hints from the instruction and use the response as the focus/output.
    """
    role = "Unspecified role"
    context = instruction
    focus = instruction
    output = response.strip() if response else "Clear, actionable response."

    # Try to detect an explicit role like "As a ..."
    m = re.search(r"as\s+a\s+(.+?)(?:,|can|could|please|to)", instruction, re.IGNORECASE)
    if m:
        role = m.group(1).strip().capitalize()

    return (
        f"Role: {role}\n"
        f"Context: {context}\n"
        f"Focus: {focus}\n"
        f"Output: {output}"
    )


def _clarification_questions(instruction: str) -> str:
    """Generate R.C.F.O-aligned clarification questions for a vague prompt."""
    return (
        "Role: What role or persona should take on this task?\n"
        f"Context: What background or situation should I know about '{instruction}'?\n"
        f"Focus: What is the specific goal or task you want accomplished?\n"
        "Output: What format or deliverable do you expect?"
    )


def _load_alpaca_from_huggingface() -> List[dict]:
    """Attempt to load Alpaca from HuggingFace datasets library."""
    try:
        from datasets import load_dataset  # type: ignore

        ds = load_dataset("tatsu-lab/alpaca", split="train")
        return [
            {"instruction": row["instruction"], "output": row.get("output", "")}
            for row in ds
        ]
    except Exception:
        return []


def _fallback_sample() -> List[dict]:
    """Small Alpaca-style sample for offline testing."""
    return [
        {"instruction": "Write a professional email to a client requesting a project deadline extension.", "output": "Subject: Request for Deadline Extension...\nDear [Client Name], I hope this email finds you well..."},
        {"instruction": "Create a Python function that calculates compound interest.", "output": "def compound_interest(principal, rate, time):\n    return principal * (1 + rate) ** time"},
        {"instruction": "fix it", "output": ""},
        {"instruction": "write something about business", "output": ""},
        {"instruction": "As a data analyst, analyze Q3 sales figures from the regional CSV file and identify underperforming regions.", "output": "Q3 sales analysis: Region A underperformed by 12%..."},
        {"instruction": "help me with this", "output": ""},
        {"instruction": "Analyze the website traffic data and produce a summary report.", "output": "Traffic increased by 15% QoQ. Main sources: organic search (40%), direct (30%)..."},
        {"instruction": "make this better", "output": ""},
        {"instruction": "Draft a product requirements document for a mobile checkout feature.", "output": "PRD: Mobile Checkout...\n1. Overview...\n2. User Stories..."},
        {"instruction": "can you debug my code", "output": ""},
    ]


def build_alpaca_pairs(use_huggingface: bool = True) -> dict:
    """Build Inspector-style training pairs from Alpaca."""
    pairs: List[dict] = []

    if use_huggingface:
        alpaca = _load_alpaca_from_huggingface()
        source = "HuggingFace (tatsu-lab/alpaca)"
    else:
        alpaca = []
        source = "offline sample"

    if not alpaca:
        alpaca = _fallback_sample()
        source = "offline sample (fallback)"

    clear_count = 0
    vague_count = 0

    for item in alpaca:
        instruction = item.get("instruction", "").strip()
        output = item.get("output", "").strip()

        if not instruction:
            continue

        if _is_clear(instruction):
            pairs.append(
                {
                    "input": instruction,
                    "target": _format_rcfo(instruction, output),
                }
            )
            clear_count += 1
        elif _is_vague(instruction):
            pairs.append(
                {
                    "input": instruction,
                    "target": _clarification_questions(instruction),
                }
            )
            vague_count += 1

    return {
        "version": "1.0.0-alpaca",
        "source": source,
        "description": (
            "Alpaca-derived dataset for Inspector AI. Clear instructions mapped "
            "to R.C.F.O format; vague/short instructions mapped to clarification "
            "questions."
        ),
        "splits": {
            "clear_to_rcfo": [p for p in pairs if "Role:" in p["target"] and "Context:" in p["target"] and len(p["input"].split()) >= 8],
            "vague_to_questions": [p for p in pairs if len(p["input"].split()) < 8 or any(s in p["input"].lower() for s in VAGUE_SIGNALS)],
        },
        "counts": {
            "clear_to_rcfo": clear_count,
            "vague_to_questions": vague_count,
            "total": clear_count + vague_count,
        },
    }


def save_json(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def main() -> None:
    dataset = build_alpaca_pairs(use_huggingface=True)

    # Save a processed copy
    save_json(dataset, OUTPUT_PATH)
    # Also keep a raw copy for provenance
    save_json(dataset, RAW_DATA_DIR / "alpaca_dataset.json")

    print("Inspector AI — Alpaca dataset conversion complete.")
    print(f"Source: {dataset['source']}")
    print(f"Clear -> R.C.F.O pairs:   {dataset['counts']['clear_to_rcfo']}")
    print(f"Vague -> Questions pairs: {dataset['counts']['vague_to_questions']}")
    print(f"Total pairs:              {dataset['counts']['total']}")
    print(f"Written to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
