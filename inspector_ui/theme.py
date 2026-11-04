"""Paleta, escala (DPI) e fontes."""

from __future__ import annotations

from tkinter import font as tkfont

SCALE = 1.0
FONT = "Segoe UI"

# Fundo
BG_TOP = "#030B1B"
BG_BOT = "#071B3A"

# Vidro e luz
GLASS_TINT = (8, 26, 56)
BLUE = "#3D8BFF"
CYAN = "#5CD6FF"
ICE = "#CFE8FF"
LIME = "#7CF29A"
AMBER = "#FFC857"
RED = "#FF6B81"

# Texto
TEXT = "#EAF3FF"
MUTED = "#8EAACB"
SUBTLE = "#5B7899"
NAVY = "#04162E"

# Balões
USER_TINT = (28, 86, 186)
AI_TINT = (12, 40, 82)
ERR_TINT = (96, 24, 48)


def rgb(color: str) -> tuple[int, int, int]:
    color = color.lstrip("#")
    return int(color[0:2], 16), int(color[2:4], 16), int(color[4:6], 16)


def hexcolor(c: tuple[int, int, int]) -> str:
    return "#%02x%02x%02x" % tuple(int(v) for v in c[:3])


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
    """Escolhe a melhor família disponível no sistema."""
    global FONT
    try:
        families = set(tkfont.families(root))
    except Exception:
        return
    for name in (
        "Segoe UI Variable Text", "Segoe UI", "Frutiger", "SF Pro Text",
        "Ubuntu", "Noto Sans", "Helvetica Neue", "Arial",
    ):
        if name in families:
            FONT = name
            return


def font(size: float, bold: bool = False) -> tuple:
    """Fonte para itens do Canvas (tamanho em pixels, já com DPI)."""
    return (FONT, -px(size * 1.3), "bold" if bold else "normal")


_measure_cache: dict = {}


def measure_font(size: float, bold: bool = False) -> tkfont.Font:
    key = (FONT, size, bold, SCALE)
    f = _measure_cache.get(key)
    if f is None:
        f = tkfont.Font(
            family=FONT, size=-px(size * 1.3), weight="bold" if bold else "normal"
        )
        _measure_cache[key] = f
    return f
