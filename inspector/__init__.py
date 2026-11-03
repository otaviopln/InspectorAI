"""InspectorAI public package interface."""

__version__ = "0.2.0"
__author__ = "Inspector AI Project"

from .engine import generate_response, load_components

__all__ = [
    "__version__",
    "__author__",
    "generate_response",
    "load_components",
]
