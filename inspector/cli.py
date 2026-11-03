"""Interactive terminal interface for InspectorAI."""

from __future__ import annotations

from .engine import generate_response, load_components


def main() -> None:
    """Load the local model and start an interactive terminal session."""
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
    except (RuntimeError, ValueError) as exc:
        print(f"Unable to initialize InspectorAI: {exc}")
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

        try:
            response = generate_response(model, tokenizer, user_input)
        except Exception as exc:
            print(f"\nInspector error: {exc}\n")
            continue

        print(f"\nInspector: {response}\n")


if __name__ == "__main__":
    main()
