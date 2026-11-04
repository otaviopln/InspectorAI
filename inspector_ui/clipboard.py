"""Reliable clipboard integration for the InspectorAI desktop UI."""

from __future__ import annotations

import ctypes
import sys
import time
from ctypes import wintypes


class ClipboardWriteError(RuntimeError):
    """Raised when the response could not be written to the system clipboard."""


def _windows_clipboard_write(text: str) -> None:
    """Write Unicode text through the Win32 API as a fallback for Tk on Windows."""
    if sys.platform != "win32":
        raise OSError("The Win32 clipboard API is available only on Windows.")

    user32 = ctypes.WinDLL("user32", use_last_error=True)
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)

    user32.OpenClipboard.argtypes = [wintypes.HWND]
    user32.OpenClipboard.restype = wintypes.BOOL
    user32.EmptyClipboard.argtypes = []
    user32.EmptyClipboard.restype = wintypes.BOOL
    user32.SetClipboardData.argtypes = [wintypes.UINT, wintypes.HANDLE]
    user32.SetClipboardData.restype = wintypes.HANDLE
    user32.CloseClipboard.argtypes = []
    user32.CloseClipboard.restype = wintypes.BOOL

    kernel32.GlobalAlloc.argtypes = [wintypes.UINT, ctypes.c_size_t]
    kernel32.GlobalAlloc.restype = wintypes.HGLOBAL
    kernel32.GlobalLock.argtypes = [wintypes.HGLOBAL]
    kernel32.GlobalLock.restype = ctypes.c_void_p
    kernel32.GlobalUnlock.argtypes = [wintypes.HGLOBAL]
    kernel32.GlobalUnlock.restype = wintypes.BOOL
    kernel32.GlobalFree.argtypes = [wintypes.HGLOBAL]
    kernel32.GlobalFree.restype = wintypes.HGLOBAL

    opened = False
    for attempt in range(5):
        if user32.OpenClipboard(None):
            opened = True
            break
        if attempt < 4:
            time.sleep(0.02)
    if not opened:
        raise ctypes.WinError(ctypes.get_last_error())

    memory = None
    transferred = False
    try:
        if not user32.EmptyClipboard():
            raise ctypes.WinError(ctypes.get_last_error())

        payload = text.encode("utf-16-le") + b"\x00\x00"
        memory = kernel32.GlobalAlloc(0x0002, len(payload))  # GMEM_MOVEABLE
        if not memory:
            raise ctypes.WinError(ctypes.get_last_error())

        pointer = kernel32.GlobalLock(memory)
        if not pointer:
            raise ctypes.WinError(ctypes.get_last_error())

        try:
            ctypes.memmove(pointer, payload, len(payload))
        finally:
            kernel32.GlobalUnlock(memory)

        if not user32.SetClipboardData(13, memory):  # CF_UNICODETEXT
            raise ctypes.WinError(ctypes.get_last_error())
        transferred = True
    finally:
        user32.CloseClipboard()
        if memory and not transferred:
            kernel32.GlobalFree(memory)


def copy_text_to_clipboard(root, text: str) -> None:
    """Copy and verify text, using Win32 as a fallback when Tk fails.

    This function must be called on the Tk main thread.
    """
    if not isinstance(text, str) or not text:
        raise ValueError("There is no response text to copy.")

    tk_error: Exception | None = None
    try:
        root.clipboard_clear()
        root.clipboard_append(text)
        root.update_idletasks()
        if root.clipboard_get() == text:
            return
        tk_error = ClipboardWriteError(
            "Tkinter did not return the exact text written to the clipboard."
        )
    except Exception as exc:
        tk_error = exc

    if sys.platform == "win32":
        try:
            _windows_clipboard_write(text)
            return
        except Exception as win_error:
            raise ClipboardWriteError(
                "Tkinter and the Windows clipboard API both failed. "
                f"Tkinter: {tk_error}; Windows: {win_error}"
            ) from win_error

    raise ClipboardWriteError(
        f"Tkinter could not write or verify the clipboard contents: {tk_error}"
    ) from tk_error
