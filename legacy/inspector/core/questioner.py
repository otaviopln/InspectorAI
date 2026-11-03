"""Strategic question generator for Inspector AI.

When a prompt is classified as dubious, Inspector asks clarifying questions
instead of guessing. These questions map to the R.C.F.O fields so the user
can supply the missing pieces.
"""

from __future__ import annotations


class QuestionGenerator:
    """Generates strategic clarifying questions for dubious prompts."""

    QUESTIONS = [
        "What role or persona should take on this task? (Role)",
        "What background or situation should I be aware of? (Context)",
        "What is the specific goal or task you want accomplished? (Focus)",
        "What format or deliverable do you expect as output? (Output)",
    ]

    def generate(self, text: str) -> list[str]:
        """Generate clarifying questions for a dubious prompt.

        Args:
            text: The raw user prompt.

        Returns:
            A list of strategic questions for the user.
        """
        return list(self.QUESTIONS)
