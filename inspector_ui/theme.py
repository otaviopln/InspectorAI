"""Paleta minimalista, escala (DPI) e fontes."""

from __future__ import annotations

from tkinter import font as tkfont

SCALE = 1.0
FONT = "Segoe UI"

BG = "#0A0C10"
SURFACE = "#11141A"
SURFACE_HI = "#181C24"
USER = "#171C26"
BORDER = "#222832"
BORDER_HI = "#333B4A"

TEXT = "#E8EBF1"
MUTED = "#8B94A5"
SUBTLE = "#5B6474"

ACCENT = "#8DB4FF"
ACCENT_DK = "#0B1730"
OK = "#7DDC9A"
WARN = "#F2C46D"
ERR = "#FF7D8D"


def rgb(color: str) -> tuple[int, int, int]:
    color = color.lstrip("#")
    return int(color[0:2], 16), int(color[2:4], 16), int(color[4:6], 16)


def mix(a: str, b: str, t: float) -> str:
    t = max(0.0, min(1.0, t))
    ra, rb = rgb(a), rgb(b)
    return "#%02x%02x%02x" % tuple(round(x + (y - x) * t) for x, y in zip(ra, rb))


def set_scale(value: float) -> None:
    global SCALE
    SCALE = max(1.0, float(value))


def px(value: float) -> int:
    return max(1, int(round(value * SCALE)))


def pick_font(root) -> None:
    global FONT
    try:
        families = set(tkfont.families(root))
    except Exception:
        return
    for name in ("Segoe UI Variable Text", "Segoe UI", "SF Pro Text", "Inter",
                 "Ubuntu", "Noto Sans", "Helvetica Neue", "Arial"):
        if name in families:
            FONT = name
            return


def font(size: float, bold: bool = False) -> tuple:
    """Fonte para itens do Canvas e do Text (pixels, já com DPI)."""
    return (FONT, -px(size * 1.3), "bold" if bold else "normal")


_cache: dict = {}


def measure_font(size: float, bold: bool = False) -> tkfont.Font:
    key = (FONT, size, bold, SCALE)
    f = _cache.get(key)
    if f is None:
        f = tkfont.Font(family=FONT, size=-px(size * 1.3), weight="bold" if bold else "normal")
        _cache[key] = f
    return f
