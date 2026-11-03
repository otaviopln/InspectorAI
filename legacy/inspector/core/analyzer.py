"""Prompt analyzer for Inspector AI.

Classifies user prompts as either "clear" or "dubious". A dubious prompt is
ambiguous, incomplete, or missing enough information to act on safely.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class AnalysisResult:
    """Result of analyzing a user prompt."""

    label: str  # "clear" or "dubious"
    confidence: float
    reason: str


class PromptAnalyzer:
    """Analyzes user prompts and classifies them as clear or dubious.

    This is a rule-based placeholder. In later iterations this will be backed
    by a PyTorch model trained on labeled prompt data.
    """

    # Heuristic signals that suggest a prompt is dubious / under-specified.
    DUBIOUS_SIGNALS = (
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
    )

    def classify(self, text: str) -> AnalysisResult:
        """Classify a prompt as clear or dubious.

        Args:
            text: The raw user prompt.

        Returns:
            An AnalysisResult with label, confidence, and reason.
        """
        lowered = text.lower().strip()
        if not lowered:
            return AnalysisResult(
                label="dubious",
                confidence=1.0,
                reason="Empty input provides no information to act on.",
            )

        # Too short to be meaningful.
        if len(lowered.split()) < 4:
            return AnalysisResult(
                label="dubious",
                confidence=0.8,
                reason="Input is too short to determine intent.",
            )

        # Contains vague / under-specified signals.
        for signal in self.DUBIOUS_SIGNALS:
            if signal in lowered:
                return AnalysisResult(
                    label="dubious",
                    confidence=0.7,
                    reason=f"Input contains vague term '{signal}'.",
                )

        return AnalysisResult(
            label="clear",
            confidence=0.9,
            reason="Input appears specific and actionable.",
        )
