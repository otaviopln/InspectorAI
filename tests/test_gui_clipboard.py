"""Tests for the desktop UI clipboard integration."""

import pytest

from inspector_ui.clipboard import (
    ClipboardWriteError,
    copy_text_to_clipboard,
)


class FakeRoot:
    def __init__(self):
        self.value = ""
        self.updated = False

    def clipboard_clear(self):
        self.value = ""

    def clipboard_append(self, text):
        self.value += text

    def update_idletasks(self):
        self.updated = True

    def clipboard_get(self):
        return self.value


def test_copy_preserves_multiline_unicode_response():
    root = FakeRoot()
    response = "Decision: CLARIFY\nQuestions:\n1. Qual é o objetivo?\n2. Ação — revisão."

    copy_text_to_clipboard(root, response)

    assert root.updated is True
    assert root.value == response


def test_copy_rejects_empty_text():
    with pytest.raises(ValueError, match="no response text"):
        copy_text_to_clipboard(FakeRoot(), "")


def test_copy_reports_tk_clipboard_failure_on_non_windows(monkeypatch):
    import inspector_ui.clipboard as clipboard

    class BrokenRoot(FakeRoot):
        def clipboard_clear(self):
            raise RuntimeError("clipboard unavailable")

    monkeypatch.setattr(clipboard.sys, "platform", "linux")

    with pytest.raises(ClipboardWriteError, match="clipboard unavailable"):
        clipboard.copy_text_to_clipboard(BrokenRoot(), "InspectorAI response")


def test_copy_uses_native_windows_unicode_clipboard(monkeypatch):
    import inspector_ui.clipboard as clipboard

    root = FakeRoot()
    captured = []
    monkeypatch.setattr(clipboard.sys, "platform", "win32")
    monkeypatch.setattr(
        clipboard,
        "_windows_clipboard_write",
        lambda value: captured.append(value),
    )

    response = "Line 1\nLine 2 — resposta completa"
    clipboard.copy_text_to_clipboard(root, response)

    assert captured == [response]
    assert root.value == ""
