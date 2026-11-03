"""R.C.F.O framework formatter for Inspector AI.

Reformats clear user prompts into the R.C.F.O structure:

- Role    — who should act
- Context — background and situation
- Focus   — the specific task or goal
- Output  — the expected deliverable or format
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RCFO:
    """A structured R.C.F.O representation."""

    role: str
    context: str
    focus: str
    output: str

    def __str__(self) -> str:
        return (
            f"Role:    {self.role}\n"
            f"Context: {self.context}\n"
            f"Focus:   {self.focus}\n"
            f"Output:  {self.output}"
        )


class RCFOFormatter:
    """Formats clear prompts into R.C.F.O structure."""

    def format(self, text: str) -> RCFO:
        """Format a clear prompt into R.C.F.O fields.

        Args:
            text: The raw user prompt (assumed clear).

        Returns:
            An RCFO dataclass populated with best-effort extractions.
        """
        role = self._extract_role(text)
        context = self._extract_context(text)
        focus = self._extract_focus(text)
        output = self._extract_output(text)
        return RCFO(role=role, context=context, focus=focus, output=output)

    def _extract_role(self, text: str) -> str:
        """Best-effort extraction of the acting role."""
        lowered = text.lower()
        for keyword in ("as a", "i am a", "i'm a", "as an", "i am an", "i'm an"):
            if keyword in lowered:
                start = lowered.index(keyword) + len(keyword)
                return text[start:].strip().capitalize()
        return "Unspecified role"

    def _extract_context(self, text: str) -> str:
        """Best-effort extraction of the background/situation."""
        return text

    def _extract_focus(self, text: str) -> str:
        """Best-effort extraction of the specific task/goal."""
        return text

    def _extract_output(self, text: str) -> str:
        """Best-effort extraction of the expected deliverable/format."""
        return "Clear, actionable response addressing the request."
