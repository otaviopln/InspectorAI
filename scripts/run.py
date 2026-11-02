"""Entry point for running Inspector AI interactively (CLI).

Loads the trained InspectorTransformer and the from-scratch BPE tokenizer,
then runs an interactive terminal loop:

  You:    <types an idea / prompt>
  Inspector:  <encoded -> Transformer -> decoded response>

The model output will naturally be either:
  * strategic clarification questions  (if the input was dubious / vague), or
  * a strict R.C.F.O formatted prompt  (if the input was clear enough).

No HuggingFace, no sentencepiece — everything is built from torch primitives
and the custom tokenizer trained in previous sessions.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from inspector import load_components, generate_response  # noqa: E402


def main() -> None:
    """Interactive terminal entry point."""
    print("=" * 60)
    print("  Inspector AI — Interactive Terminal")
    print("=" * 60)
    print("Type your idea below. Inspector will respond with either")
    print("strategic clarification questions (dubious input) or a strict")
    print("R.C.F.O formatted prompt (clear input).")
    print("Type 'quit', 'exit', or 'q' to leave.\n")

    try:
        tokenizer, model = load_components()
    except FileNotFoundError as exc:
        print(f"Error: {exc}")
        print("\nRun 'python scripts/train.py' first to train the model.")
        return

    print("-" * 60)
    print("Ready. Awaiting your input...\n")

    while True:
        try:
            user_input = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if user_input.lower() in {"quit", "exit", "q"}:
            print("Goodbye!")
            break
        if not user_input:
            continue

        response = generate_response(model, tokenizer, user_input)
        print(f"\nInspector: {response}\n")


if __name__ == "__main__":
    main()
