"""Regression tests for the centralized InspectorAI engine imports."""

import inspector
from inspector import engine


def test_package_and_engine_export_the_same_api():
    assert inspector.generate_response is engine.generate_response
    assert inspector.load_components is engine.load_components


def test_engine_is_loaded_from_the_inspector_package():
    assert engine.__name__ == "inspector.engine"
    assert callable(engine.generate_response)
    assert callable(engine.load_components)
