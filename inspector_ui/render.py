"""Formas e luzes desenhadas com Pillow sobre o fundo liso da interface."""

from __future__ import annotations

from functools import lru_cache

from PIL import Image, ImageChops, ImageDraw, ImageFilter

from . import theme as T

_RS = getattr(Image, "Resampling", Image)
LANCZOS = _RS.LANCZOS
BILINEAR = _RS.BILINEAR


def rmask(w: int, h: int, radius: float, ss: int = 3) -> Image.Image:
    w, h = max(1, int(w)), max(1, int(h))
    m = Image.new("L", (w * ss, h * ss), 0)
    ImageDraw.Draw(m).rounded_rectangle((0, 0, w * ss - 1, h * ss - 1), radius=radius * ss, fill=255)
    return m.resize((w, h), LANCZOS)


@lru_cache(maxsize=512)
def shape(w: int, h: int, radius: int, fill: str, border: str | None = None,
          bg: str = T.BG) -> Image.Image:
    """Retângulo arredondado com antialiasing, já composto sobre o fundo."""
    w, h, ss = max(2, int(w)), max(2, int(h)), 3
    img = Image.new("RGB", (w * ss, h * ss), T.rgb(bg))
    ImageDraw.Draw(img).rounded_rectangle(
        (0, 0, w * ss - 1, h * ss - 1), radius=min(radius, h / 2) * ss,
        fill=T.rgb(fill), outline=T.rgb(border) if border else None, width=ss,
    )
    return img.resize((w, h), LANCZOS)


@lru_cache(maxsize=256)
def circle(size: int, fill: str, bg: str = T.SURFACE) -> Image.Image:
    return shape(size, size, size // 2, fill, None, bg)


@lru_cache(maxsize=64)
def ghost(w: int, h: int, level: int) -> Image.Image:
    k = level / 4
    return shape(w, h, h // 2, T.mix(T.BG, T.SURFACE_HI, k), T.mix(T.BG, T.BORDER, k))


@lru_cache(maxsize=256)
def field(w: int, h: int, radius: int, level: int, pad: int) -> Image.Image:
    """Campo de texto: superfície lisa com borda que acende (level 0..8) e halo suave."""
    w, h, pad = int(w), int(h), int(pad)
    W, H = w + 2 * pad, h + 2 * pad
    k = level / 8
    base = Image.new("RGB", (W, H), T.rgb(T.BG))
    if k > 0:
        sc = 4
        g = Image.new("L", (W // sc + 2, H // sc + 2), 0)
        ImageDraw.Draw(g).rounded_rectangle(
            (pad / sc, pad / sc, (pad + w) / sc, (pad + h) / sc),
            radius=min(radius, h / 2) / sc, outline=255, width=2,
        )
        g = g.filter(ImageFilter.GaussianBlur(1.8)).resize((W, H), BILINEAR)
        g = g.point(lambda v: int(v * 0.5 * k))
        base = Image.composite(Image.new("RGB", (W, H), T.rgb(T.ACCENT)), base, g)
    border = T.mix(T.BORDER, T.ACCENT, k * 0.8)
    body = shape(w, h, radius, T.SURFACE, border)
    base.paste(body, (pad, pad), rmask(w, h, min(radius, h / 2)))
    return base


def glow(radius: int, color: str = T.ACCENT, intensity: float = 1.0) -> Image.Image:
    """Ponto de luz suave (RGBA)."""
    size = max(4, int(radius) * 2)
    g = Image.radial_gradient("L").resize((size, size), BILINEAR)
    g = g.point(lambda v: int(255 * intensity * max(0.0, 1 - v / (255 * 0.9)) ** 2.4))
    layer = Image.new("RGBA", (size, size), T.rgb(color) + (0,))
    layer.putalpha(g)
    return layer


def fade_strip(w: int, h: int, top: bool) -> Image.Image:
    """Faixa da cor do fundo que dissolve o conteúdo nas bordas da área de rolagem."""
    w, h = max(1, int(w)), max(2, int(h))
    g = Image.linear_gradient("L").resize((w, h), BILINEAR)
    g = g.point(lambda v: 255 - v) if top else g
    g = g.point(lambda v: int(255 * (v / 255) ** 0.9))
    layer = Image.new("RGBA", (w, h), T.rgb(T.BG) + (0,))
    layer.putalpha(g)
    return layer
