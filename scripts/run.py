"""Entry point for running Inspector AI interactively."""

import sys
from pathlib import Path

# Allow running from scripts/ directory
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from inspector.core.analyzer import PromptAnalyzer
from inspector.core.formatter import RCFOFormatter
from inspector.core.questioner import QuestionGenerator


def main() -> None:
    analyzer = PromptAnalyzer()
    formatter = RCFOFormatter()
    questioner = QuestionGenerator()

    print("Inspector AI — enter a prompt (type 'quit' to exit):\n")
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

        classification = analyzer.classify(user_input)
        if classification == "dubious":
            questions = questioner.generate(user_input)
            print("\nInspector: Your input is a bit unclear. Could you clarify?")
            for i, q in enumerate(questions, 1):
                print(f"  {i}. {q}")
        else:
            rcfo = formatter.format(user_input)
            print("\nInspector: R.C.F.O formatted input:")
            print(rcfo)

        print()


if __name__ == "__main__":
    main()
