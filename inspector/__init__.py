"""InspectorAI public package interface."""

__version__ = "0.2.0"
__author__ = "Inspector AI Project"

from .engine import (
    InspectorResult,
    generate_response,
    generate_result,
    load_components,
)

__all__ = [
    "__version__",
    "__author__",
    "InspectorResult",
    "generate_response",
    "generate_result",
    "load_components",
]
