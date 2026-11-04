"""Renderização com Pillow: fundo, vidro fosco, bordas luminosas, flares e botões."""

from __future__ import annotations

import math
from functools import lru_cache

from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageOps

from . import theme as T

_RS = getattr(Image, "Resampling", Image)
LANCZOS = _RS.LANCZOS
BILINEAR = _RS.BILINEAR


# ---------------------------------------------------------------------------
# Primitivas
# ---------------------------------------------------------------------------
def vgrad(w: int, h: int, top: str, bottom: str) -> Image.Image:
    w, h = max(1, int(w)), max(1, int(h))
    g = Image.linear_gradient("L").resize((w, h), BILINEAR)
    return ImageOps.colorize(g, black=top, white=bottom)


def rmask(w: int, h: int, radius: float, ss: int = 2) -> Image.Image:
    w, h = max(1, int(w)), max(1, int(h))
    m = Image.new("L", (w * ss, h * ss), 0)
    ImageDraw.Draw(m).rounded_rectangle(
        (0, 0, w * ss - 1, h * ss - 1), radius=radius * ss, fill=255
    )
    return m.resize((w, h), LANCZOS)


def _tinted(size, color: str, alpha: Image.Image) -> Image.Image:
    layer = Image.new("RGBA", size, T.rgb(color) + (0,))
    layer.putalpha(alpha)
    return layer


# ---------------------------------------------------------------------------
# Fundo com luz
# ---------------------------------------------------------------------------
def make_background(w: int, h: int) -> Image.Image:
    """Gradiente azul-marinho com florescências de luz e feixes diagonais."""
    w, h = max(8, int(w)), max(8, int(h))
    base = vgrad(w, h, T.BG_TOP, T.BG_BOT).convert("RGBA")

    sw, sh = max(2, w // 4), max(2, h // 4)
    light = Image.new("RGBA", (sw, sh), (0, 0, 0, 0))
    for fx, fy, fr, col, a in (
        (0.08, 0.04, 0.62, "#2F6BFF", 100),
        (0.94, 0.96, 0.66, "#12B5FF", 74),
        (0.60, 0.42, 0.44, "#1E40C8", 50),
    ):
        g = Image.new("L", (sw, sh), 0)
        R = fr * max(sw, sh) / 2
        ImageDraw.Draw(g).ellipse(
            (fx * sw - R, fy * sh - R, fx * sw + R, fy * sh + R), fill=a
        )
        g = g.filter(ImageFilter.GaussianBlur(R * 0.6))
        light = Image.alpha_composite(light, _tinted((sw, sh), col, g))
    base = Image.alpha_composite(base, light.resize((w, h), BILINEAR))

    hw, hh = max(2, w // 2), max(2, h // 2)
    shafts = Image.new("L", (hw, hh), 0)
    d = ImageDraw.Draw(shafts)
    for cx, wid, a in ((0.20, 0.05, 40), (0.50, 0.025, 26), (0.80, 0.07, 36)):
        x, bw, skew = cx * hw, wid * hw, hh * 0.55
        d.polygon(
            [(x - bw + skew, 0), (x + bw + skew, 0), (x + bw - skew, hh), (x - bw - skew, hh)],
            fill=a,
        )
    shafts = shafts.filter(ImageFilter.GaussianBlur(hw * 0.006 + 2)).resize((w, h), BILINEAR)
    fade = Image.linear_gradient("L").resize((w, h), BILINEAR).point(lambda v: 255 - int(v * 0.85))
    shafts = ImageChops.multiply(shafts, fade)
    base = Image.alpha_composite(base, _tinted((w, h), "#9CC8FF", shafts))
    return base.convert("RGB")


# ---------------------------------------------------------------------------
# Vidro
# ---------------------------------------------------------------------------
def frost_fill(
    bg: Image.Image, x: int, y: int, w: int, h: int, radius: int,
    *, tint=T.GLASS_TINT, tint_a: int = 165, flat: bool = False, blur: int = 16,
) -> Image.Image:
    """Vidro fosco: recorta o fundo, desfoca, escurece e aplica reflexos suaves."""
    w, h = max(2, int(w)), max(2, int(h))
    bw, bh = bg.size
    m = int(radius + blur * 2)
    x0, y0 = max(0, x - m), max(0, y - m)
    x1, y1 = min(bw, x + w + m), min(bh, y + h + m)
    crop = bg.crop((x0, y0, x1, y1)).filter(ImageFilter.GaussianBlur(blur))
    region = Image.new("RGB", (w, h), T.rgb(T.BG_BOT))
    region.paste(crop, (x0 - x, y0 - y))
    region = Image.alpha_composite(
        region.convert("RGBA"), Image.new("RGBA", (w, h), tuple(tint) + (tint_a,))
    )
    if not flat:
        sheen_h = max(2, int(h * 0.5))
        sa = Image.linear_gradient("L").resize((w, sheen_h), BILINEAR)
        sa = sa.point(lambda v: int(26 * (1 - v / 255)))
        full = Image.new("L", (w, h), 0)
        full.paste(sa, (0, 0))
        region = Image.alpha_composite(region, _tinted((w, h), "#BEDCFF", full))

        rw, rh = max(2, w // 4), max(2, h // 4)
        refl = Image.new("L", (rw, rh), 0)
        ImageDraw.Draw(refl).polygon([(0, 0), (rw * 0.36, 0), (0, rh * 0.9)], fill=16)
        refl = refl.filter(ImageFilter.GaussianBlur(2)).resize((w, h), BILINEAR)
        region = Image.alpha_composite(region, _tinted((w, h), "#FFFFFF", refl))
    region.putalpha(rmask(w, h, radius))
    return region


def make_glass(
    w: int, h: int, radius: int, *, tint=(30, 80, 160),
    alpha_top: int = 120, alpha_bot: int = 70, border: str | None = None,
    border_a: int = 120, gloss: float = 0.16, ss: int = 2,
) -> Image.Image:
    """Vidro translúcido (RGBA) para balões, cartões e fichas."""
    w, h = max(2, int(w)), max(2, int(h))
    W, H, r = w * ss, h * ss, radius * ss
    base = T.hexcolor(tint)
    body = vgrad(W, H, T.mix(base, "#FFFFFF", 0.16), base).convert("RGBA")
    ag = Image.linear_gradient("L").resize((W, H), BILINEAR)
    ag = ag.point(lambda v: int(alpha_top + (alpha_bot - alpha_top) * v / 255))
    body.putalpha(ImageChops.multiply(ag, rmask(W, H, r, ss=1)))

    gh = max(2, int(H * 0.46))
    ga = Image.linear_gradient("L").resize((W, gh), BILINEAR)
    ga = ga.point(lambda v: int(255 * gloss * (1 - v / 255)))
    gfull = Image.new("L", (W, H), 0)
    gfull.paste(ga, (0, 0))
    gfull = ImageChops.multiply(gfull, rmask(W, H, r, ss=1))
    body = Image.alpha_composite(body, _tinted((W, H), "#FFFFFF", gfull))

    if border:
        st = Image.new("L", (W, H), 0)
        ImageDraw.Draw(st).rounded_rectangle(
            (0, 0, W - 1, H - 1), radius=r, outline=border_a, width=max(1, ss)
        )
        body = Image.alpha_composite(body, _tinted((W, H), border, st))
    return body.resize((w, h), LANCZOS)


@lru_cache(maxsize=256)
def make_bubble(w: int, h: int, kind: str) -> Image.Image:
    if kind == "user":
        return make_glass(w, h, 20, tint=T.USER_TINT, alpha_top=215, alpha_bot=170,
                          border="#9CCBFF", border_a=170, gloss=0.2)
    if kind == "error":
        return make_glass(w, h, 20, tint=T.ERR_TINT, alpha_top=210, alpha_bot=170,
                          border=T.RED, border_a=190, gloss=0.12)
    return make_glass(w, h, 20, tint=T.AI_TINT, alpha_top=200, alpha_bot=150,
                      border="#6FA8E6", border_a=120, gloss=0.14)


def make_rim(w: int, h: int, radius: int, *, intensity: float = 0.5,
             color: str = T.CYAN, pad: int = 16) -> Image.Image:
    """Borda luminosa do vidro: traço fino com brilho direcional + halo externo."""
    w, h, radius, pad = int(w), int(h), int(radius), int(pad)
    W, H = w + 2 * pad, h + 2 * pad
    out = Image.new("RGBA", (W, H), (0, 0, 0, 0))

    sc = 4
    g = Image.new("L", (W // sc + 2, H // sc + 2), 0)
    ImageDraw.Draw(g).rounded_rectangle(
        (pad / sc, pad / sc, (pad + w) / sc, (pad + h) / sc),
        radius=radius / sc, outline=255, width=2,
    )
    g = g.filter(ImageFilter.GaussianBlur(2.2)).resize((W, H), BILINEAR)
    g = g.point(lambda v: int(v * 0.62 * intensity))
    out = Image.alpha_composite(out, _tinted((W, H), color, g))

    ss = 2
    st = Image.new("L", (W * ss, H * ss), 0)
    ImageDraw.Draw(st).rounded_rectangle(
        (pad * ss, pad * ss, (pad + w) * ss - 1, (pad + h) * ss - 1),
        radius=radius * ss, outline=255, width=max(2, int(ss * (1.0 + 0.8 * intensity))),
    )
    st = st.resize((W, H), LANCZOS)
    hg = Image.linear_gradient("L").rotate(90).resize((W, H), BILINEAR)
    vg = Image.linear_gradient("L").resize((W, H), BILINEAR)
    diag = ImageOps.invert(ImageChops.difference(hg, vg))
    diag = diag.point(lambda v: int(95 + 160 * (v / 255) ** 1.5))
    st = ImageChops.multiply(st, diag).point(lambda v: int(v * (0.40 + 0.60 * intensity)))
    return Image.alpha_composite(out, _tinted((W, H), T.mix(color, "#FFFFFF", 0.55), st))


# ---------------------------------------------------------------------------
# Luzes
# ---------------------------------------------------------------------------
def make_flare(length: int, thick: int, color: str = T.CYAN,
               intensity: float = 1.0, vertical: bool = False) -> Image.Image:
    """Faixa de luz (como as dos cantos dos painéis de vidro) com bloom central."""
    L, Hh = max(8, int(length)), max(6, int(thick))
    cy = Hh // 2
    prof = Image.new("L", (L, Hh), 0)
    ImageDraw.Draw(prof).line((0, cy, L, cy), fill=255, width=max(1, Hh // 9))
    core = prof.filter(ImageFilter.GaussianBlur(max(0.6, Hh * 0.04)))
    soft = prof.filter(ImageFilter.GaussianBlur(Hh * 0.22)).point(lambda v: int(v * 0.9))
    alpha = ImageChops.add(core, soft)

    bloom = Image.new("L", (L, Hh), 0)
    ImageDraw.Draw(bloom).ellipse((L / 2 - Hh * 0.45, cy - Hh * 0.16, L / 2 + Hh * 0.45, cy + Hh * 0.16), fill=255)
    bloom = bloom.filter(ImageFilter.GaussianBlur(Hh * 0.13)).point(lambda v: int(v * 0.8))
    alpha = ImageChops.add(alpha, bloom)

    env = Image.linear_gradient("L").rotate(90).resize((L, Hh), BILINEAR)
    env = env.point(lambda v: int(255 * math.sin(math.pi * v / 255) ** 2))
    alpha = ImageChops.multiply(alpha, env).point(lambda v: int(v * intensity))
    img = _tinted((L, Hh), T.mix(color, "#FFFFFF", 0.62), alpha)
    return img.rotate(90, expand=True) if vertical else img


def make_glow(radius: int, color: str = T.CYAN, intensity: float = 1.0) -> Image.Image:
    """Ponto de luz suave (halo radial)."""
    size = max(4, int(radius) * 2)
    g = Image.radial_gradient("L").resize((size, size), BILINEAR)
    g = g.point(lambda v: int(255 * intensity * max(0.0, 1 - v / 255) ** 2.2))
    return _tinted((size, size), color, g)


def make_glint(w: int, h: int, radius: int, t: float, pad: int = 0,
               strength: float = 0.55) -> Image.Image:
    """Reflexo diagonal de luz que atravessa o vidro (t de 0 a 1)."""
    w, h, pad = int(w), int(h), int(pad)
    band = Image.new("L", (w, h), 0)
    cx = -0.25 * w + t * 1.5 * w
    bw, slant = max(6, w * 0.10), h * 0.8
    ImageDraw.Draw(band).polygon(
        [(cx - bw, 0), (cx + bw, 0), (cx + bw - slant, h), (cx - bw - slant, h)], fill=255
    )
    band = band.filter(ImageFilter.GaussianBlur(max(2, bw * 0.45)))
    band = ImageChops.multiply(band, rmask(w, h, radius)).point(lambda v: int(v * strength))
    layer = _tinted((w, h), "#FFFFFF", band)
    if pad:
        canvas = Image.new("RGBA", (w + 2 * pad, h + 2 * pad), (0, 0, 0, 0))
        canvas.paste(layer, (pad, pad))
        return canvas
    return layer


def make_fade_strip(fill: Image.Image, h: int, top: bool = True) -> Image.Image:
    """Faixa tirada do próprio vidro, com transparência que se dissolve para dentro."""
    W, H = fill.size
    h = max(2, min(int(h), H // 2))
    strip = fill.crop((0, 0, W, h)) if top else fill.crop((0, H - h, W, H))
    g = Image.linear_gradient("L").resize((W, h), BILINEAR)
    g = g.point(lambda v: 255 - v) if top else g
    r, gg, b, a = strip.split()
    return Image.merge("RGBA", (r, gg, b, ImageChops.multiply(a, g)))


# ---------------------------------------------------------------------------
# Botões de vidro
# ---------------------------------------------------------------------------
def _button_colors(style: str, disabled: bool):
    if disabled:
        return "#1B3354", "#12243F", "#2B4A74", 0.6, 0.0
    if style == "primary":
        return "#7BCBFF", "#1F5FE0", "#C4E6FF", 1.0, 0.28
    return "#2E5A9A", "#14305E", "#6FA8E6", 0.55, 0.0


def make_button(w: int, h: int, style: str = "primary", level: float = 0.0,
                pad: int = 10, disabled: bool = False) -> Image.Image:
    """Pílula de vidro com gloss e halo; level (0..1) controla o brilho de hover."""
    w, h, pad = int(w), int(h), int(pad)
    ss = 3
    W, H = w * ss, h * ss
    top, bot, bd, fill_a, idle_glow = _button_colors(style, disabled)
    k = 0.0 if disabled else level
    top = T.mix(top, "#FFFFFF", 0.20 * k)
    bot = T.mix(bot, "#FFFFFF", 0.14 * k)
    bd = T.mix(bd, "#FFFFFF", 0.30 * k)
    r = H / 2
    mask = rmask(W, H, r, ss=1)

    body = vgrad(W, H, top, bot).convert("RGBA")
    if style == "primary" and not disabled:
        gl = Image.new("L", (W, H), 0)
        ImageDraw.Draw(gl).ellipse((W * 0.10, H * 0.58, W * 0.90, H * 1.40), fill=150)
        gl = gl.filter(ImageFilter.GaussianBlur(H * 0.16))
        body = Image.alpha_composite(body, _tinted((W, H), "#9CF3FF", gl))
    gh = max(2, int(H * 0.5))
    gstr = 0.55 if style == "primary" else 0.24
    ga = Image.linear_gradient("L").resize((W, gh), BILINEAR)
    ga = ga.point(lambda v: int(255 * gstr * (1 - 0.9 * v / 255)))
    gfull = Image.new("L", (W, H), 0)
    gfull.paste(ga, (0, 0))
    gshape = Image.new("L", (W, H), 0)
    ins = ss * 2
    ImageDraw.Draw(gshape).rounded_rectangle(
        (ins, ins, W - 1 - ins, int(H * 0.52)), radius=max(2, r - ins), fill=255
    )
    body = Image.alpha_composite(
        body, _tinted((W, H), "#FFFFFF", ImageChops.multiply(gfull, gshape))
    )
    body.putalpha(mask.point(lambda v: int(v * fill_a)))
    st = Image.new("L", (W, H), 0)
    ImageDraw.Draw(st).rounded_rectangle((0, 0, W - 1, H - 1), radius=r, outline=255, width=ss)
    body = Image.alpha_composite(body, _tinted((W, H), bd, st))

    WP, HP = W + 2 * pad * ss, H + 2 * pad * ss
    canvas = Image.new("RGBA", (WP, HP), (0, 0, 0, 0))
    glow_k = 0.0 if disabled else (idle_glow + (1 - idle_glow) * k if style == "primary" else 0.8 * k)
    if glow_k > 0:
        gm = Image.new("L", (WP, HP), 0)
        gm.paste(mask, (pad * ss, pad * ss))
        gm = gm.filter(ImageFilter.GaussianBlur(pad * ss * 0.55)).point(lambda v: int(v * 0.9 * glow_k))
        canvas = Image.alpha_composite(canvas, _tinted((WP, HP), "#4FB0FF" if style == "primary" else "#7FB8FF", gm))
    canvas.alpha_composite(body, (pad * ss, pad * ss))
    return canvas.resize((w + 2 * pad, h + 2 * pad), LANCZOS)
