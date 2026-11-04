"""InspectorAI — interface desktop em estilo Frutiger Aero (modo escuro).

Principais características:
- Visual "aero" escuro: vidro, brilho aqua, orbes/bolhas glossy e bordas luminosas.
- Layout responsivo: a barra lateral recolhe sozinha em janelas estreitas, balões de
  mensagem e cartões de sugestão se reorganizam conforme a largura disponível.
- Animações leves: hover suave nos botões, foco luminoso no campo de texto, indicador
  de "digitando" com três bolhas, fade-in das mensagens e efeito de máquina de escrever.
- Carregamento do modelo e inferência em threads, com fila thread-safe.
- Guarda de inferência única para proteger o modelo local.
- Você pode digitar enquanto o modelo carrega; o envio é liberado quando ele estiver pronto.

Execute a partir da raiz do projeto: python inspector_gui.py
"""

from __future__ import annotations

import math
import queue
import random
import threading
import tkinter as tk
from datetime import datetime
from tkinter import font as tkfont
from typing import Optional

import customtkinter as ctk
from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageOps

try:  # ImageTk depende do Tk; o CustomTkinter já exige Pillow + Tk.
    from PIL import ImageTk
except Exception:  # pragma: no cover
    ImageTk = None

from inspector.engine import generate_result, load_components


# ---------------------------------------------------------------------------
# Tema Frutiger Aero (escuro)
# ---------------------------------------------------------------------------
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("dark-blue")

BG = "#04111D"            # fundo da janela
SIDEBAR = "#061A2A"       # barra lateral
WORK = "#051722"          # área de trabalho
CONV = "#071F31"          # painel da conversa (vidro profundo)
GLASS = "#0B2A3F"         # cartões de vidro
GLASS_HI = "#103A55"      # vidro em hover
BORDER = "#1A5575"        # borda sutil
BORDER_HI = "#38B6E6"     # borda luminosa

TEXT = "#EAF9FF"
MUTED = "#8DB8CF"
SUBTLE = "#5E8CA6"

AQUA = "#35E0F2"
AQUA_2 = "#18B6E8"
SKY = "#4FA9FF"
LIME = "#8DF26B"
AMBER = "#FFC857"
RED = "#FF6B81"
NAVY_TEXT = "#03263B"

USER_BUBBLE = "#0D4468"
USER_BORDER = "#3FB0E4"
AI_BUBBLE = "#09283C"
AI_BORDER = "#1D6483"
ERR_BUBBLE = "#2A1220"

# (topo, base, aro) das orbes
ORB_AQUA = ("#1593C9", "#8FF7FF", "#C9FCFF")
ORB_SKY = ("#1F63D6", "#8CCBFF", "#CBE6FF")
ORB_LIME = ("#2BA84F", "#CBFF93", "#E5FFC9")
ORB_AMBER = ("#C98A12", "#FFE08A", "#FFF1C4")

FONT = "Segoe UI"
MONO = "Cascadia Code"

SIDEBAR_FULL = 272
SIDEBAR_COMPACT = 84
NARROW_BREAKPOINT = 1040  # largura lógica abaixo da qual a sidebar recolhe


# ---------------------------------------------------------------------------
# Utilitários de cor e imagem (PIL)
# ---------------------------------------------------------------------------
_RS = getattr(Image, "Resampling", Image)
_LANCZOS = _RS.LANCZOS
_BILINEAR = _RS.BILINEAR


def _rgb(color: str) -> tuple[int, int, int]:
    color = color.lstrip("#")
    return int(color[0:2], 16), int(color[2:4], 16), int(color[4:6], 16)


def mix(a: str, b: str, t: float) -> str:
    """Interpolação linear entre duas cores hexadecimais."""
    t = max(0.0, min(1.0, t))
    ra, rb = _rgb(a), _rgb(b)
    return "#%02x%02x%02x" % tuple(round(x + (y - x) * t) for x, y in zip(ra, rb))


def _vgrad(w: int, h: int, top: str, bottom: str) -> Image.Image:
    w, h = max(1, int(w)), max(1, int(h))
    grad = Image.linear_gradient("L").resize((w, h), _BILINEAR)
    return ImageOps.colorize(grad, black=top, white=bottom)


def _rounded_mask(w: int, h: int, radius: float) -> Image.Image:
    mask = Image.new("L", (w, h), 0)
    ImageDraw.Draw(mask).rounded_rectangle(
        (0, 0, w - 1, h - 1), radius=max(0, radius), fill=255
    )
    return mask


def _flatten(img: Image.Image, bg: str) -> Image.Image:
    base = Image.new("RGB", img.size, _rgb(bg))
    base.paste(img, (0, 0), img)
    return base


_PILL_CACHE: dict = {}


def make_pill(
    w: int,
    h: int,
    *,
    top: str,
    bottom: str,
    border: str,
    bg: str,
    gloss: float = 0.5,
    glow: Optional[str] = None,
    radius: Optional[float] = None,
) -> Image.Image:
    """Botão em forma de pílula com brilho de vidro (estilo Aero)."""
    key = (w, h, top, bottom, border, bg, gloss, glow, radius)
    cached = _PILL_CACHE.get(key)
    if cached is not None:
        return cached

    ss = 3
    W, H = w * ss, h * ss
    r = min((radius if radius is not None else h / 2) * ss, H / 2)

    base = _vgrad(W, H, top, bottom).convert("RGBA")

    if glow:  # reflexo luminoso na parte de baixo
        g = Image.new("L", (W, H), 0)
        ImageDraw.Draw(g).ellipse((W * 0.10, H * 0.58, W * 0.90, H * 1.40), fill=170)
        g = g.filter(ImageFilter.GaussianBlur(H * 0.16))
        layer = Image.new("RGBA", (W, H), _rgb(glow) + (0,))
        layer.putalpha(g)
        base = Image.alpha_composite(base, layer)

    if gloss > 0:  # faixa de brilho superior
        gh = max(2, int(H * 0.52))
        hi, lo = int(255 * gloss), int(255 * gloss * 0.10)
        ga = Image.linear_gradient("L").resize((W, gh), _BILINEAR)
        ga = ga.point(lambda v: int(hi + (lo - hi) * v / 255))
        galpha = Image.new("L", (W, H), 0)
        galpha.paste(ga, (0, 0))
        shape = Image.new("L", (W, H), 0)
        inset = ss * 2
        ImageDraw.Draw(shape).rounded_rectangle(
            (inset, inset, W - 1 - inset, gh), radius=max(2, r - inset), fill=255
        )
        galpha = ImageChops.multiply(galpha, shape)
        white = Image.new("RGBA", (W, H), (255, 255, 255, 0))
        white.putalpha(galpha)
        base = Image.alpha_composite(base, white)

    ImageDraw.Draw(base).rounded_rectangle(
        (0, 0, W - 1, H - 1), radius=r, outline=border, width=ss
    )
    base.putalpha(_rounded_mask(W, H, r))

    out = Image.new("RGB", (W, H), _rgb(bg))
    out.paste(base, (0, 0), base)
    out = out.resize((w, h), _LANCZOS)
    _PILL_CACHE[key] = out
    return out


def make_tile(size: int, top: str, bottom: str, rim: str, ss: int = 4) -> Image.Image:
    """Ícone em quadrado arredondado com brilho de vidro (RGBA)."""
    S = max(8, int(size)) * ss
    r = S * 0.30
    body = _vgrad(S, S, top, bottom).convert("RGBA")
    shape = Image.new("L", (S, S), 0)
    inset = ss * 2
    ImageDraw.Draw(shape).rounded_rectangle(
        (inset, inset, S - 1 - inset, int(S * 0.52)), radius=r - inset, fill=255
    )
    fade = Image.linear_gradient("L").resize((S, S), _BILINEAR)
    fade = fade.point(lambda v: max(8, 190 - int(v * 1.5)))
    white = Image.new("RGBA", (S, S), (255, 255, 255, 0))
    white.putalpha(ImageChops.multiply(shape, fade))
    body = Image.alpha_composite(body, white)
    ImageDraw.Draw(body).rounded_rectangle(
        (0, 0, S - 1, S - 1), radius=r, outline=rim, width=max(1, int(ss * 1.2))
    )
    body.putalpha(_rounded_mask(S, S, r))
    return body.resize((int(size), int(size)), _LANCZOS)


def _draw_bubble(
    layer: Image.Image, cx: float, cy: float, r: float, tint: str, strength: float = 1.0
) -> Image.Image:
    """Bolha de sabão translúcida sobre uma camada RGBA."""
    ov = Image.new("RGBA", layer.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(ov)
    tr = _rgb(tint)
    d.ellipse(
        (cx - r, cy - r, cx + r, cy + r),
        fill=tr + (int(24 * strength),),
        outline=tr + (int(150 * strength),),
        width=max(1, int(r * 0.05)),
    )
    d.ellipse(
        (cx - r * 0.78, cy + r * 0.05, cx + r * 0.78, cy + r * 0.92),
        fill=tr + (int(34 * strength),),
    )
    d.ellipse(
        (cx - r * 0.62, cy - r * 0.84, cx + r * 0.20, cy - r * 0.32),
        fill=(255, 255, 255, int(105 * strength)),
    )
    d.ellipse(
        (cx + r * 0.36, cy + r * 0.38, cx + r * 0.56, cy + r * 0.56),
        fill=(255, 255, 255, int(120 * strength)),
    )
    return Image.alpha_composite(layer, ov)


def _fade_top(img: Image.Image, frac: float) -> Image.Image:
    """Esmaece a transparência no topo para a imagem se fundir ao fundo."""
    W, H = img.size
    g = Image.linear_gradient("L").resize((1, H), _BILINEAR)
    g = g.point(lambda v: min(255, int(v / max(frac, 0.01)))).resize((W, H))
    r, gg, b, a = img.split()
    img = Image.merge("RGBA", (r, gg, b, ImageChops.multiply(a, g)))
    return img


def _edge_fade(img: Image.Image, margin: float, sides: str = "tblr") -> Image.Image:
    """Dissolve as bordas escolhidas da imagem para que o brilho não forme um retângulo."""
    W, H = img.size
    m = int(margin)
    mask = Image.new("L", (W, H), 0)
    left = 0 if "l" in sides else -m * 3
    right = W if "r" in sides else W + m * 3
    top = 0 if "t" in sides else -m * 3
    bottom = H if "b" in sides else H + m * 3
    ImageDraw.Draw(mask).rectangle(
        (left + m, top + m, right - m - 1, bottom - m - 1), fill=255
    )
    mask = mask.filter(ImageFilter.GaussianBlur(m / 2.2))
    r, g, b, a = img.split()
    return Image.merge("RGBA", (r, g, b, ImageChops.multiply(a, mask)))


def make_decor(w: int, h: int, seed: int = 11) -> Image.Image:
    """Bolhas e brilho aqua para o rodapé da barra lateral (RGBA)."""
    ss = 2
    W, H = w * ss, h * ss
    img = Image.new("RGBA", (W, H), (0, 0, 0, 0))

    glow = Image.new("L", (W, H), 0)
    ImageDraw.Draw(glow).ellipse((-W * 0.3, H * 0.60, W * 1.3, H * 1.7), fill=180)
    glow = glow.filter(ImageFilter.GaussianBlur(H * 0.12)).point(lambda v: int(v * 0.55))
    layer = Image.new("RGBA", (W, H), _rgb(AQUA_2) + (0,))
    layer.putalpha(glow)
    img = Image.alpha_composite(img, layer)

    rng = random.Random(seed)
    for i in range(10):
        r = rng.uniform(0.04, 0.16) * W
        cx = rng.uniform(0.08, 0.92) * W
        cy = H - rng.uniform(0.02, 0.85) * H
        tint = (AQUA, SKY, LIME, AQUA)[i % 4]
        img = _draw_bubble(img, cx, cy, r, tint, 0.35 + 0.65 * (cy / H))

    img = _fade_top(img, 0.55)
    return img.resize((w, h), _LANCZOS)


def make_hero(w: int, h: int) -> Image.Image:
    """Orbe central com halo, reflexo e bolhas ao redor (RGBA transparente)."""
    ss = 2
    W, H = w * ss, h * ss
    img = Image.new("RGBA", (W, H), (0, 0, 0, 0))

    halo = Image.new("L", (W, H), 0)
    ImageDraw.Draw(halo).ellipse(
        (W / 2 - H * 0.55, H * 0.14, W / 2 + H * 0.55, H * 0.84), fill=150
    )
    halo = halo.filter(ImageFilter.GaussianBlur(H * 0.14))
    layer = Image.new("RGBA", (W, H), _rgb(AQUA_2) + (0,))
    layer.putalpha(halo.point(lambda v: int(v * 0.55)))
    img = Image.alpha_composite(img, layer)

    for fx, fy, fr, tint in (
        (0.50, 0.50, 0.30, AQUA),
        (0.18, 0.64, 0.11, SKY),
        (0.82, 0.30, 0.09, LIME),
        (0.74, 0.78, 0.06, AQUA),
        (0.27, 0.20, 0.05, AQUA),
        (0.90, 0.62, 0.04, SKY),
        (0.09, 0.28, 0.035, LIME),
    ):
        img = _draw_bubble(img, fx * W, fy * H, fr * H, tint, 1.0)

    img = _edge_fade(img, H * 0.10)
    return img.resize((w, h), _LANCZOS)


def make_glow_line(w: int, h: int = 3) -> Image.Image:
    """Linha luminosa aqua → lime, com as pontas desaparecendo."""
    w = max(2, int(w))
    img = Image.new("RGBA", (w, h))
    px = img.load()
    c0, c1 = _rgb(AQUA), _rgb(LIME)
    for x in range(w):
        t = x / (w - 1)
        a = int(255 * (math.sin(math.pi * t) ** 0.7))
        col = tuple(round(p + (q - p) * t) for p, q in zip(c0, c1))
        for y in range(h):
            px[x, y] = col + (a if y > 0 else a // 2,)
    return img


def ctk_img(img: Image.Image, size: tuple[int, int]) -> ctk.CTkImage:
    return ctk.CTkImage(light_image=img, dark_image=img, size=size)


# ---------------------------------------------------------------------------
# Botão glossy desenhado em Canvas (hover animado, ícones vetoriais)
# ---------------------------------------------------------------------------
class GlossButton(tk.Canvas):
    STYLES = {
        "aqua": dict(
            top="#52E8FA", bottom="#0B8BC6", border="#9AF8FF",
            text=NAVY_TEXT, gloss=0.62, glow="#B5FDFF",
        ),
        "glass": dict(
            top="#1B5675", bottom="#0A3048", border="#3A94BA",
            text=TEXT, gloss=0.30, glow=None,
        ),
    }

    def __init__(
        self,
        master,
        *,
        text: str = "",
        icon: Optional[str] = None,
        style: str = "aqua",
        width: int = 120,
        height: int = 40,
        command=None,
        bg: str = WORK,
        scale: float = 1.0,
        font_size: int = 12,
        radius: Optional[float] = None,
    ) -> None:
        self._scale = scale
        w = max(8, round(width * scale))
        h = max(8, round(height * scale))
        super().__init__(
            master, width=w, height=h, bg=bg, bd=0,
            highlightthickness=0, cursor="hand2", takefocus=0,
        )
        self._bg_color = bg
        self._text = text
        self._icon = icon
        self._style = self.STYLES[style]
        self._command = command
        self._font_size = font_size
        self._radius = radius
        self._bw, self._bh = w, h

        self._enabled = True
        self._pressed = False
        self._hover_step = 0
        self._hover_target = 0
        self._hover_job: Optional[str] = None
        self._pil: dict = {}
        self._photos: dict = {}
        self._img_item = None
        self._fg_items: list = []

        self.bind("<Enter>", self._on_enter)
        self.bind("<Leave>", self._on_leave)
        self.bind("<ButtonPress-1>", self._on_press)
        self.bind("<ButtonRelease-1>", self._on_release)
        self.bind("<Configure>", self._on_configure)
        self._build_items()

    # -- API pública ---------------------------------------------------
    def set_enabled(self, enabled: bool) -> None:
        if enabled == self._enabled:
            return
        self._enabled = enabled
        self._pressed = False
        self._hover_step = self._hover_target = 0
        self.configure(cursor="hand2" if enabled else "arrow")
        self._recolor()
        self._apply()

    def set_text(self, text: str) -> None:
        if text != self._text:
            self._text = text
            self._build_items()

    # -- render --------------------------------------------------------
    def _font(self) -> tkfont.Font:
        px = max(8, round(self._font_size * self._scale))
        return tkfont.Font(family=FONT, size=-px, weight="bold")

    def _fg(self) -> str:
        return self._style["text"] if self._enabled else "#5C8CA6"

    def _pil_img(self, name: str) -> Image.Image:
        img = self._pil.get(name)
        if img is not None:
            return img
        st = self._style
        top, bottom, border = st["top"], st["bottom"], st["border"]
        gloss, glow = st["gloss"], st["glow"]
        if name == "hover":
            top, bottom = mix(top, "#FFFFFF", 0.20), mix(bottom, "#FFFFFF", 0.14)
            border = mix(border, "#FFFFFF", 0.25)
        elif name == "pressed":
            top, bottom = mix(top, "#000000", 0.22), mix(bottom, "#000000", 0.22)
            gloss *= 0.55
        elif name == "disabled":
            top, bottom, border = "#12384F", "#0B2A3E", "#1E5068"
            gloss, glow = 0.07, None
        img = make_pill(
            self._bw, self._bh, top=top, bottom=bottom, border=border,
            bg=self._bg_color, gloss=gloss, glow=glow, radius=self._radius,
        )
        self._pil[name] = img
        return img

    def _photo_for(self, key):
        photo = self._photos.get(key)
        if photo is not None:
            return photo
        kind, step = key
        if kind == "hover":
            normal = self._pil_img("normal")
            img = normal if step == 0 else Image.blend(
                normal, self._pil_img("hover"), step / 3
            )
        else:
            img = self._pil_img(kind)
        photo = ImageTk.PhotoImage(img)
        self._photos[key] = photo
        return photo

    def _key(self):
        if not self._enabled:
            return ("disabled", 0)
        if self._pressed:
            return ("pressed", 0)
        return ("hover", self._hover_step)

    def _apply(self) -> None:
        if self._img_item is not None:
            self.itemconfigure(self._img_item, image=self._photo_for(self._key()))

    def _recolor(self) -> None:
        color = self._fg()
        for item in self._fg_items:
            self.itemconfigure(item, fill=color)

    def _build_items(self) -> None:
        self.delete("all")
        self._fg_items = []
        w, h, s = self._bw, self._bh, self._scale
        self._img_item = self.create_image(
            0, 0, anchor="nw", image=self._photo_for(self._key())
        )
        color = self._fg()
        font = self._font() if self._text else None
        tw = font.measure(self._text) if font else 0
        isz = round(self._font_size * 1.3 * s) if self._icon else 0
        gap = round(9 * s) if (self._icon and self._text) else 0
        x0 = (w - (isz + gap + tw)) / 2
        cy = h / 2
        if self._icon:
            self._draw_icon(self._icon, x0 + isz / 2, cy, isz, color)
        if self._text:
            self._fg_items.append(
                self.create_text(
                    x0 + isz + gap, cy, text=self._text, anchor="w",
                    fill=color, font=font,
                )
            )

    def _draw_icon(self, kind: str, cx: float, cy: float, size: float, color: str) -> None:
        lw = max(2, round(2.4 * self._scale))
        r = size / 2
        kw = dict(fill=color, width=lw, capstyle="round", joinstyle="round")
        if kind == "plus":
            items = [
                self.create_line(cx - r * 0.8, cy, cx + r * 0.8, cy, **kw),
                self.create_line(cx, cy - r * 0.8, cx, cy + r * 0.8, **kw),
            ]
        elif kind == "send":
            items = [
                self.create_line(cx, cy + r * 0.85, cx, cy - r * 0.85, **kw),
                self.create_line(
                    cx - r * 0.78, cy - r * 0.08, cx, cy - r * 0.85,
                    cx + r * 0.78, cy - r * 0.08, **kw,
                ),
            ]
        else:  # menu
            items = [
                self.create_line(cx - r * 0.8, cy + dy, cx + r * 0.8, cy + dy, **kw)
                for dy in (-r * 0.62, 0, r * 0.62)
            ]
        self._fg_items.extend(items)

    # -- eventos -------------------------------------------------------
    def _on_configure(self, event) -> None:
        if event.width < 8 or event.height < 8:
            return
        if (event.width, event.height) != (self._bw, self._bh):
            self._bw, self._bh = event.width, event.height
            self._pil.clear()
            self._photos.clear()
            self._build_items()

    def _on_enter(self, _event=None) -> None:
        if self._enabled:
            self._hover_target = 3
            self._kick()

    def _on_leave(self, _event=None) -> None:
        self._pressed = False
        self._hover_target = 0
        self._kick()

    def _on_press(self, _event=None) -> None:
        if self._enabled:
            self._pressed = True
            self._apply()

    def _on_release(self, event) -> None:
        if not self._enabled:
            return
        was_pressed = self._pressed
        self._pressed = False
        self._apply()
        inside = 0 <= event.x <= self._bw and 0 <= event.y <= self._bh
        if was_pressed and inside and self._command:
            self._command()

    def _kick(self) -> None:
        if self._hover_job is None:
            self._step()

    def _step(self) -> None:
        self._hover_job = None
        try:
            if self._hover_step < self._hover_target:
                self._hover_step += 1
            elif self._hover_step > self._hover_target:
                self._hover_step -= 1
            self._apply()
            if self._hover_step != self._hover_target:
                self._hover_job = self.after(28, self._step)
        except tk.TclError:
            pass


# ---------------------------------------------------------------------------
# Aplicação
# ---------------------------------------------------------------------------
STATUS_STATES = {
    "loading": (AMBER, "Carregando modelo…", "Preparando o motor local", "CARREGANDO"),
    "ready": (LIME, "Motor pronto", "Modelo local online", "ONLINE"),
    "busy": (AQUA, "Analisando…", "Gerando a resposta", "PROCESSANDO"),
    "error": (RED, "Motor indisponível", "Verifique o modelo", "ERRO"),
}


class InspectorGUI(ctk.CTk):
    """Interface de chat Frutiger Aero para o modelo local InspectorAI."""

    def __init__(self) -> None:
        super().__init__()
        self._pick_font()

        self.title("InspectorAI | Prompt Workspace")
        self.configure(fg_color=BG)
        self._closed = False
        self._dpi = self._ui_scale()

        screen_w = self.winfo_screenwidth() / self._dpi
        screen_h = self.winfo_screenheight() / self._dpi
        self.geometry(f"{int(min(1280, screen_w - 80))}x{int(min(860, screen_h - 120))}")
        self.minsize(820, 600)
        self.after(40, self._center_window)

        self.tokenizer = None
        self.model = None

        self._loading = False
        self._generating = False
        self._worker_results: queue.Queue = queue.Queue()

        self._next_row = 0
        self._welcome_visible = False
        self._body_labels: list[ctk.CTkLabel] = []
        self._suggestion_cards: list[tuple[ctk.CTkFrame, ctk.CTkLabel]] = []
        self._typing_row: Optional[ctk.CTkFrame] = None
        self._typing_dots: list[ctk.CTkLabel] = []

        self._phase = 0
        self._tick_id: Optional[str] = None
        self._pulsing = False
        self._dot_color = AMBER
        self._dot_dim = mix(AMBER, GLASS, 0.6)
        self._status_state = "loading"

        self._scroll_pending = False
        self._reflow_id: Optional[str] = None
        self._resize_id: Optional[str] = None
        self._hint_id: Optional[str] = None
        self._glow_width = 0
        self._was_narrow: Optional[bool] = None
        self._sidebar_collapsed = False
        self._tween_tokens: dict[str, int] = {}

        self._make_images()

        self.grid_columnconfigure(2, weight=1)
        self.grid_rowconfigure(0, weight=1)

        self._build_sidebar()
        ctk.CTkFrame(self, width=1, corner_radius=0, fg_color="#0F3A52").grid(
            row=0, column=1, sticky="ns"
        )
        self._build_workspace()
        self._show_welcome_screen()

        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self.bind("<Configure>", self._on_root_configure)
        self.bind("<Control-n>", lambda _e: self._clear_chat())
        self.bind("<Control-l>", lambda _e: self.input_box.focus_set())

        self.after(90, self._poll_worker_results)
        self._tick()
        self._set_status("loading")
        self._load_model_async()
        self.after(150, self._on_resized)

    # ------------------------------------------------------------------
    # Infra: escala, fonte, imagens, animações
    # ------------------------------------------------------------------
    def _ui_scale(self) -> float:
        try:
            return float(self._get_window_scaling())
        except Exception:
            return 1.0

    def _pick_font(self) -> None:
        global FONT
        try:
            families = set(tkfont.families(self))
        except Exception:
            return
        for candidate in (
            "Segoe UI Variable Text", "Segoe UI", "Frutiger", "SF Pro Text",
            "Ubuntu", "Noto Sans", "Helvetica Neue", "Arial",
        ):
            if candidate in families:
                FONT = candidate
                break

    def _center_window(self) -> None:
        if self._closed:
            return
        try:
            self.eval("tk::PlaceWindow . center")
        except Exception:
            pass

    def _px(self, value: float) -> int:
        return max(1, int(round(value * self._dpi)))

    def _make_images(self) -> None:
        def tile(size: int, palette) -> ctk.CTkImage:
            return ctk_img(make_tile(self._px(size), *palette), (size, size))

        self.img_mark = tile(46, ORB_AQUA)
        self.img_avatar_ai = tile(32, ORB_AQUA)
        self.img_avatar_user = tile(32, ORB_SKY)
        self.img_letters = [
            tile(30, ORB_AQUA), tile(30, ORB_SKY), tile(30, ORB_LIME), tile(30, ORB_AQUA)
        ]
        self.img_card_icons = [tile(34, ORB_AQUA), tile(34, ORB_LIME), tile(34, ORB_SKY)]
        self.img_hero = ctk_img(
            make_hero(self._px(440), self._px(190)), (440, 190)
        )

    def _tween(
        self, widget, option: str, c0: str, c1: str,
        *, steps: int = 8, delay: int = 16, key: Optional[str] = None,
    ) -> None:
        """Transição suave de cor para uma opção de widget."""
        token = 0
        if key is not None:
            token = self._tween_tokens.get(key, 0) + 1
            self._tween_tokens[key] = token

        def step(i: int) -> None:
            if self._closed:
                return
            if key is not None and self._tween_tokens.get(key) != token:
                return
            try:
                widget.configure(**{option: mix(c0, c1, i / steps)})
            except Exception:
                return
            if i < steps:
                self.after(delay, lambda: step(i + 1))

        step(0)

    def _tick(self) -> None:
        """Relógio único de animações (pulso de status e bolhas de 'digitando')."""
        if self._closed:
            return
        self._phase += 1
        t = self._phase

        if self._pulsing:
            k = 0.5 + 0.5 * math.sin(t * 0.24)
            color = mix(self._dot_dim, self._dot_color, k)
            for dot in (self.sidebar_status_dot, self.model_badge_dot):
                try:
                    dot.configure(text_color=color)
                except Exception:
                    pass

        for i, dot in enumerate(self._typing_dots):
            k = 0.5 + 0.5 * math.sin(t * 0.30 - i * 0.95)
            try:
                dot.configure(text_color=mix("#2B6A88", AQUA, k))
            except Exception:
                pass

        self._tick_id = self.after(60, self._tick)

    # ------------------------------------------------------------------
    # Layout: barra lateral
    # ------------------------------------------------------------------
    def _build_sidebar(self) -> None:
        self.sidebar = ctk.CTkFrame(
            self, width=SIDEBAR_FULL, corner_radius=0, fg_color=SIDEBAR, border_width=0
        )
        self.sidebar.grid(row=0, column=0, sticky="nsew")
        self.sidebar.grid_propagate(False)
        self.sidebar.grid_columnconfigure(0, weight=1)
        self.sidebar.grid_rowconfigure(3, weight=1)

        # Marca
        self.brand = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        self.brand.grid(row=0, column=0, padx=18, pady=(24, 22), sticky="ew")
        self.brand_mark = ctk.CTkLabel(
            self.brand, text="I", image=self.img_mark, compound="center",
            font=(FONT, 20, "bold"), text_color=NAVY_TEXT,
        )
        self.brand_mark.pack(side="left")
        self.brand_text = ctk.CTkFrame(self.brand, fg_color="transparent")
        self.brand_text.pack(side="left", padx=(12, 0))
        ctk.CTkLabel(
            self.brand_text, text="INSPECTOR", font=(FONT, 16, "bold"), text_color=TEXT
        ).pack(anchor="w")
        ctk.CTkLabel(
            self.brand_text, text="AI PROMPT WORKSPACE",
            font=(FONT, 9, "bold"), text_color=AQUA_2,
        ).pack(anchor="w", pady=(1, 0))

        # Nova conversa
        self.new_chat_button = GlossButton(
            self.sidebar, text="Nova conversa", icon="plus", style="aqua",
            width=236, height=46, command=self._clear_chat, bg=SIDEBAR,
            scale=self._dpi, font_size=13,
        )
        self.new_chat_button.grid(row=1, column=0, padx=18, pady=(0, 20), sticky="ew")

        # Cartão R.C.F.O.
        self.rcfo_card = ctk.CTkFrame(
            self.sidebar, fg_color=GLASS, corner_radius=18,
            border_width=1, border_color=BORDER,
        )
        self.rcfo_card.grid(row=2, column=0, padx=18, sticky="ew")
        self.rcfo_card.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(
            self.rcfo_card, text="MOTOR R.C.F.O.", font=(FONT, 10, "bold"), text_color=AQUA
        ).grid(row=0, column=0, padx=16, pady=(14, 8), sticky="w")

        rows = (
            ("R", "Role", "Quem responde"),
            ("C", "Context", "Cenário da tarefa"),
            ("F", "Focus", "O que priorizar"),
            ("O", "Output", "Formato da entrega"),
        )
        for index, (letter, word, desc) in enumerate(rows):
            line = ctk.CTkFrame(self.rcfo_card, fg_color="transparent")
            line.grid(
                row=index + 1, column=0, padx=12,
                pady=(0, 10 if index == len(rows) - 1 else 6), sticky="ew",
            )
            ctk.CTkLabel(
                line, text=letter, image=self.img_letters[index], compound="center",
                font=(FONT, 12, "bold"), text_color=NAVY_TEXT,
            ).pack(side="left")
            texts = ctk.CTkFrame(line, fg_color="transparent")
            texts.pack(side="left", padx=(10, 0))
            ctk.CTkLabel(
                texts, text=word, font=(FONT, 12, "bold"), text_color=TEXT
            ).pack(anchor="w")
            ctk.CTkLabel(
                texts, text=desc, font=(FONT, 10), text_color=MUTED
            ).pack(anchor="w")

        # Cartão de status (rodapé)
        self.status_card = ctk.CTkFrame(
            self.sidebar, fg_color=GLASS, corner_radius=16,
            border_width=1, border_color=BORDER,
        )
        self.status_card.grid(row=4, column=0, padx=18, pady=18, sticky="ew")
        self.status_card.grid_columnconfigure(1, weight=1)

        self.sidebar_status_dot = ctk.CTkLabel(
            self.status_card, text="●", font=(FONT, 16), text_color=AMBER, width=20
        )
        self.sidebar_status_dot.grid(row=0, column=0, padx=(14, 6), pady=12)

        self.status_texts = ctk.CTkFrame(self.status_card, fg_color="transparent")
        self.status_texts.grid(row=0, column=1, padx=(0, 12), pady=10, sticky="w")
        self.sidebar_status_text = ctk.CTkLabel(
            self.status_texts, text="Carregando modelo…",
            font=(FONT, 12, "bold"), text_color=TEXT, anchor="w",
        )
        self.sidebar_status_text.pack(anchor="w")
        self.sidebar_status_sub = ctk.CTkLabel(
            self.status_texts, text="Preparando o motor local",
            font=(FONT, 10), text_color=MUTED, anchor="w",
        )
        self.sidebar_status_sub.pack(anchor="w")

        # Bolhas decorativas (fundo; ficam atrás dos cartões)
        self._decor_photo = None
        self.sidebar_decor = None
        if ImageTk is not None:
            decor = make_decor(self._px(SIDEBAR_FULL), self._px(300))
            self._decor_photo = ImageTk.PhotoImage(_flatten(decor, SIDEBAR))
            self.sidebar_decor = tk.Label(
                self.sidebar, image=self._decor_photo, bd=0,
                highlightthickness=0, bg=SIDEBAR,
            )
            self.sidebar_decor.place(x=0, rely=1.0, anchor="sw")
            try:
                self.sidebar_decor.lift(self.sidebar._canvas)
            except Exception:
                self.sidebar_decor.lower()

    def _set_sidebar_collapsed(self, collapsed: bool) -> None:
        if collapsed == self._sidebar_collapsed:
            return
        self._sidebar_collapsed = collapsed
        self.sidebar.configure(width=SIDEBAR_COMPACT if collapsed else SIDEBAR_FULL)

        if collapsed:
            self.brand_text.pack_forget()
            self.rcfo_card.grid_remove()
            self.status_texts.grid_remove()
            self.new_chat_button.set_text("")
            self.brand.grid_configure(padx=(19, 0))
            self.new_chat_button.grid_configure(padx=16)
            self.status_card.grid_configure(padx=16)
            self.sidebar_status_dot.grid_configure(padx=(0, 0))
            self.sidebar_status_dot.configure(width=52)
            if self.sidebar_decor is not None:
                self.sidebar_decor.place_forget()
        else:
            self.brand_text.pack(side="left", padx=(12, 0))
            self.rcfo_card.grid()
            self.status_texts.grid()
            self.new_chat_button.set_text("Nova conversa")
            self.brand.grid_configure(padx=18)
            self.new_chat_button.grid_configure(padx=18)
            self.status_card.grid_configure(padx=18)
            self.sidebar_status_dot.grid_configure(padx=(14, 6))
            self.sidebar_status_dot.configure(width=20)
            if self.sidebar_decor is not None:
                self.sidebar_decor.place(x=0, rely=1.0, anchor="sw")
                try:
                    self.sidebar_decor.lift(self.sidebar._canvas)
                except Exception:
                    pass

    def _toggle_sidebar(self) -> None:
        self._set_sidebar_collapsed(not self._sidebar_collapsed)
        self._schedule_reflow()

    # ------------------------------------------------------------------
    # Layout: área de trabalho
    # ------------------------------------------------------------------
    def _build_workspace(self) -> None:
        self.workspace = ctk.CTkFrame(self, fg_color=WORK, corner_radius=0)
        self.workspace.grid(row=0, column=2, sticky="nsew")
        self.workspace.grid_columnconfigure(0, weight=1)
        self.workspace.grid_rowconfigure(2, weight=1)

        # Cabeçalho
        self.header = ctk.CTkFrame(
            self.workspace, height=82, corner_radius=0, fg_color=WORK
        )
        self.header.grid(row=0, column=0, sticky="ew")
        self.header.grid_propagate(False)
        self.header.grid_columnconfigure(1, weight=1)
        self.header.grid_rowconfigure(0, weight=1)

        self.menu_button = GlossButton(
            self.header, icon="menu", style="glass", width=42, height=42,
            command=self._toggle_sidebar, bg=WORK, scale=self._dpi, radius=14,
        )
        self.menu_button.grid(row=0, column=0, padx=(24, 14))

        title_box = ctk.CTkFrame(self.header, fg_color="transparent")
        title_box.grid(row=0, column=1, sticky="w")
        ctk.CTkLabel(
            title_box, text="Prompt Studio", font=(FONT, 20, "bold"), text_color=TEXT
        ).pack(anchor="w")
        self.subtitle = ctk.CTkLabel(
            title_box, text="Transforme solicitações em instruções mais precisas.",
            font=(FONT, 11), text_color=MUTED,
        )
        self.subtitle.pack(anchor="w", pady=(1, 0))

        actions = ctk.CTkFrame(self.header, fg_color="transparent")
        actions.grid(row=0, column=2, padx=(10, 26), sticky="e")

        self.model_badge = ctk.CTkFrame(
            actions, fg_color=GLASS, corner_radius=18, border_width=1, border_color=BORDER
        )
        self.model_badge.pack(side="left", padx=(0, 12))
        self.model_badge_dot = ctk.CTkLabel(
            self.model_badge, text="●", text_color=AMBER, font=(FONT, 12)
        )
        self.model_badge_dot.pack(side="left", padx=(13, 6), pady=7)
        self.model_badge_text = ctk.CTkLabel(
            self.model_badge, text="CARREGANDO", font=(FONT, 10, "bold"), text_color=MUTED
        )
        self.model_badge_text.pack(side="left", padx=(0, 15), pady=7)

        self.clear_button = GlossButton(
            actions, text="Limpar", style="glass", width=92, height=38,
            command=self._clear_chat, bg=WORK, scale=self._dpi, font_size=12,
        )
        self.clear_button.pack(side="left")

        # Linha luminosa sob o cabeçalho
        self.glow_label = tk.Label(self.workspace, bd=0, highlightthickness=0, bg=WORK)
        self.glow_label.grid(row=1, column=0, sticky="ew")
        self._update_glow_line(900)

        # Painel de conversa (vidro)
        self.conversation_panel = ctk.CTkFrame(
            self.workspace, fg_color=CONV, corner_radius=22,
            border_width=1, border_color=BORDER,
        )
        self.conversation_panel.grid(
            row=2, column=0, sticky="nsew", padx=24, pady=(16, 12)
        )
        self.conversation_panel.grid_columnconfigure(0, weight=1)
        self.conversation_panel.grid_rowconfigure(0, weight=1)

        self.messages_frame = ctk.CTkScrollableFrame(
            self.conversation_panel, fg_color="transparent", corner_radius=0,
            scrollbar_button_color="#1E6485", scrollbar_button_hover_color=AQUA,
        )
        self.messages_frame.grid(row=0, column=0, sticky="nsew", padx=8, pady=10)
        self.messages_frame.grid_columnconfigure(0, weight=1)
        try:
            self.messages_frame._parent_canvas.bind(
                "<Configure>", lambda _e: self._schedule_reflow(), add="+"
            )
        except Exception:
            pass

        # Compositor
        composer_wrap = ctk.CTkFrame(self.workspace, fg_color="transparent")
        composer_wrap.grid(row=3, column=0, sticky="ew", padx=24, pady=(0, 18))
        composer_wrap.grid_columnconfigure(0, weight=1)

        self.composer = ctk.CTkFrame(
            composer_wrap, fg_color=GLASS, corner_radius=22,
            border_width=1, border_color=BORDER,
        )
        self.composer.grid(row=0, column=0, sticky="ew")
        self.composer.grid_columnconfigure(0, weight=1)

        self.input_box = ctk.CTkTextbox(
            self.composer, height=48, corner_radius=0, fg_color="transparent",
            border_width=0, text_color=TEXT, font=(FONT, 13), wrap="word",
            scrollbar_button_color="#1E6485", scrollbar_button_hover_color=AQUA,
        )
        self.input_box.grid(row=0, column=0, padx=(22, 8), pady=12, sticky="ew")
        self.input_box.bind("<Return>", self._on_enter)
        self.input_box.bind("<KP_Enter>", self._on_enter)
        self.input_box.bind("<FocusIn>", lambda _e: self._set_composer_focus(True))
        self.input_box.bind("<FocusOut>", lambda _e: self._set_composer_focus(False))
        try:
            self.input_box._textbox.bind("<<Modified>>", self._on_input_modified)
        except Exception:
            self.input_box.bind("<KeyRelease>", lambda _e: self._after_input_changed())

        self.placeholder = ctk.CTkLabel(
            self.composer, text="Descreva sua ideia ou solicitação…",
            font=(FONT, 13), text_color=SUBTLE, fg_color="transparent", anchor="w",
        )
        try:
            self.placeholder.place(in_=self.input_box._textbox, x=3, y=1)
        except Exception:
            self.placeholder.place(x=24, y=16)
        self.placeholder.bind("<Button-1>", lambda _e: self.input_box.focus_set())

        self.send_button = GlossButton(
            self.composer, icon="send", style="aqua", width=46, height=46,
            command=self._send, bg=GLASS, scale=self._dpi, font_size=14,
        )
        self.send_button.grid(row=0, column=1, padx=(0, 12), pady=12, sticky="s")
        self.send_button.set_enabled(False)

        self.hint_label = ctk.CTkLabel(
            composer_wrap, text="", font=(FONT, 10), text_color=SUBTLE, anchor="e"
        )
        self.hint_label.grid(row=1, column=0, pady=(8, 0), padx=6, sticky="e")
        self._reset_hint()

    def _update_glow_line(self, width_px: int) -> None:
        if self._closed or ImageTk is None:
            return
        width_px = max(200, int(width_px))
        self._glow_width = width_px
        line = make_glow_line(width_px, max(2, self._px(3)))
        self._glow_photo = ImageTk.PhotoImage(_flatten(line, WORK))
        self.glow_label.configure(image=self._glow_photo)

    # ------------------------------------------------------------------
    # Tela de boas-vindas
    # ------------------------------------------------------------------
    def _show_welcome_screen(self) -> None:
        self.welcome = ctk.CTkFrame(self.messages_frame, fg_color="transparent")
        self.welcome.grid(row=0, column=0, sticky="ew", padx=16, pady=(22, 12))
        self.welcome.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(self.welcome, text="", image=self.img_hero).grid(
            row=0, column=0, pady=(0, 6)
        )
        self.headline_1 = ctk.CTkLabel(
            self.welcome, text="Boas ideias merecem",
            font=(FONT, 30, "bold"), text_color=TEXT,
        )
        self.headline_1.grid(row=1, column=0)
        self.headline_2 = ctk.CTkLabel(
            self.welcome, text="instruções extraordinárias.",
            font=(FONT, 30, "bold"), text_color=AQUA,
        )
        self.headline_2.grid(row=2, column=0, pady=(0, 12))

        self.welcome_text = ctk.CTkLabel(
            self.welcome,
            text=(
                "Conte o que você deseja realizar. O Inspector analisa sua solicitação "
                "e ajuda a estruturá-la usando Role, Context, Focus e Output."
            ),
            font=(FONT, 12), text_color=MUTED, justify="center", wraplength=560,
        )
        self.welcome_text.grid(row=3, column=0, pady=(0, 24))

        self.suggestions = ctk.CTkFrame(self.welcome, fg_color="transparent")
        self.suggestions.grid(row=4, column=0, sticky="ew")

        cards = (
            ("Estruturar uma ideia",
             "Quero criar uma plataforma para organizar estudos e tarefas."),
            ("Melhorar um projeto",
             "Analise meu projeto de software e identifique melhorias prioritárias."),
            ("Criar um planejamento",
             "Preciso de um plano de estudos de programação para três meses."),
        )
        self._suggestion_cards = []
        for index, (title, prompt) in enumerate(cards):
            self._suggestion_cards.append(self._build_suggestion_card(index, title, prompt))

        self._welcome_visible = True
        self._layout_suggestions(3, 190)

    def _build_suggestion_card(self, index: int, title: str, prompt: str):
        card = ctk.CTkFrame(
            self.suggestions, fg_color=GLASS, corner_radius=18,
            border_width=1, border_color=BORDER,
        )
        card.grid_columnconfigure(1, weight=1)

        icon = ctk.CTkLabel(
            card, text=("1", "2", "3")[index], compound="center",
            image=self.img_card_icons[index],
            font=(FONT, 14, "bold"), text_color=NAVY_TEXT,
        )
        icon.grid(row=0, column=0, rowspan=2, padx=(14, 10), pady=14, sticky="n")
        title_label = ctk.CTkLabel(
            card, text=title, font=(FONT, 13, "bold"), text_color=TEXT, anchor="w"
        )
        title_label.grid(row=0, column=1, padx=(0, 14), pady=(14, 2), sticky="w")
        desc = ctk.CTkLabel(
            card, text=prompt, font=(FONT, 10), text_color=MUTED,
            anchor="w", justify="left", wraplength=180,
        )
        desc.grid(row=1, column=1, padx=(0, 14), pady=(0, 14), sticky="nw")

        def hover(active: bool) -> None:
            self._tween(
                card, "border_color",
                BORDER if active else BORDER_HI, BORDER_HI if active else BORDER,
                steps=6, key=f"card{index}",
            )
            card.configure(fg_color=GLASS_HI if active else GLASS)

        def on_leave(_e=None) -> None:
            self.after(30, lambda: hover(False) if not self._pointer_inside(card) else None)

        for widget in (card, icon, title_label, desc):
            widget.bind("<Enter>", lambda _e: hover(True), add="+")
            widget.bind("<Leave>", on_leave, add="+")
            widget.bind("<Button-1>", lambda _e, p=prompt: self._use_suggestion(p), add="+")
            try:
                tk.Misc.configure(widget, cursor="hand2")
            except Exception:
                pass
        return card, desc

    def _layout_suggestions(self, columns: int, column_width: float) -> None:
        if not self._welcome_visible:
            return
        for col in range(4):
            self.suggestions.grid_columnconfigure(
                col, weight=1 if col < columns else 0, uniform="cards" if col < columns else ""
            )
        for index, (card, desc) in enumerate(self._suggestion_cards):
            card.grid(
                row=index // columns, column=index % columns,
                padx=6, pady=6, sticky="nsew",
            )
            desc.configure(wraplength=max(120, int(column_width - 100)))

    def _remove_welcome_screen(self) -> None:
        if self._welcome_visible:
            self.welcome.destroy()
            self._welcome_visible = False
            self._suggestion_cards = []

    def _pointer_inside(self, widget) -> bool:
        try:
            px, py = widget.winfo_pointerxy()
            x, y = widget.winfo_rootx(), widget.winfo_rooty()
            return x <= px <= x + widget.winfo_width() and y <= py <= y + widget.winfo_height()
        except Exception:
            return False

    # ------------------------------------------------------------------
    # Mensagens
    # ------------------------------------------------------------------
    def _wrap_width(self) -> int:
        try:
            canvas_w = self.messages_frame._parent_canvas.winfo_width()
        except Exception:
            canvas_w = 900
        avail = max(260, canvas_w / self._dpi - 36)
        bubble_max = max(260, min(820, avail * 0.82 - 48))
        return int(bubble_max - 38)

    def _build_row(self, is_user: bool, is_error: bool = False):
        """Cria a linha da mensagem: avatar + balão. Retorna (row, bubble, header)."""
        self._remove_welcome_screen()
        self._next_row += 1

        row = ctk.CTkFrame(self.messages_frame, fg_color="transparent")
        row.grid(row=self._next_row, column=0, sticky="ew", padx=10, pady=(4, 10))
        row.grid_columnconfigure(1, weight=1)

        avatar = ctk.CTkLabel(
            row, text="V" if is_user else "I", compound="center",
            image=self.img_avatar_user if is_user else self.img_avatar_ai,
            font=(FONT, 13, "bold"), text_color=NAVY_TEXT,
        )
        avatar.grid(row=0, column=2 if is_user else 0, padx=(10, 0) if is_user else (0, 10), sticky="n")

        if is_error:
            fill, border = ERR_BUBBLE, RED
        elif is_user:
            fill, border = USER_BUBBLE, USER_BORDER
        else:
            fill, border = AI_BUBBLE, AI_BORDER

        bubble = ctk.CTkFrame(
            row, fg_color=fill, corner_radius=18, border_width=1, border_color=border
        )
        bubble.grid(row=0, column=1, sticky="e" if is_user else "w")
        bubble.grid_columnconfigure(0, weight=1)

        header = ctk.CTkFrame(bubble, fg_color="transparent")
        header.grid(row=0, column=0, padx=16, pady=(10, 0), sticky="ew")
        header.grid_columnconfigure(1, weight=1)

        name_color = RED if is_error else (SKY if is_user else AQUA)
        ctk.CTkLabel(
            header, text="VOCÊ" if is_user else "INSPECTOR",
            font=(FONT, 10, "bold"), text_color=name_color,
        ).grid(row=0, column=0, sticky="w")
        ctk.CTkLabel(
            header, text=datetime.now().strftime("%H:%M"),
            font=(FONT, 9), text_color=SUBTLE,
        ).grid(row=0, column=1, padx=(12, 0), sticky="e")

        return row, bubble, header, fill

    def _add_message(
        self,
        sender: str,
        text: str,
        *,
        animate: bool = False,
        is_error: bool = False,
    ) -> ctk.CTkLabel:
        """Adiciona um balão de conversa e, opcionalmente, revela o texto aos poucos."""
        is_user = sender.lower() in {"você", "voce", "user"}
        row, bubble, header, fill = self._build_row(is_user, is_error)

        text_color = "#FFB3BF" if is_error else TEXT
        body = ctk.CTkLabel(
            bubble, text="" if animate else text, font=(FONT, 13),
            text_color=text_color if animate else fill,
            justify="left", anchor="w", wraplength=self._wrap_width(),
        )
        body.grid(row=1, column=0, padx=16, pady=(4, 14), sticky="w")
        self._body_labels.append(body)
        self._last_bubble = bubble

        if not animate:
            self._tween(body, "text_color", fill, text_color, steps=7)

        if not is_user and not is_error and text:
            copy_button = ctk.CTkButton(
                header, text="Copiar", width=58, height=22, corner_radius=8,
                fg_color="transparent", hover_color=GLASS_HI, text_color=MUTED,
                font=(FONT, 10, "bold"),
            )
            copy_button.configure(
                command=lambda b=copy_button, t=text: self._copy_text(b, t)
            )
            copy_button.grid(row=0, column=2, padx=(6, 0), sticky="e")

        self._scroll_to_bottom(force=True)
        if animate and text:
            self._reveal_text(body, text)
        return body

    def _copy_text(self, button: ctk.CTkButton, text: str) -> None:
        try:
            self.clipboard_clear()
            self.clipboard_append(text)
            button.configure(text="✓ Copiado", text_color=LIME)
            self.after(1500, lambda: self._restore_copy(button))
        except Exception:
            pass

    def _restore_copy(self, button: ctk.CTkButton) -> None:
        if self._closed:
            return
        try:
            button.configure(text="Copiar", text_color=MUTED)
        except Exception:
            pass

    def _reveal_text(self, label: ctk.CTkLabel, full_text: str, position: int = 0) -> None:
        """Efeito de máquina de escrever, agendado na thread principal do Tk."""
        if self._closed:
            return
        try:
            canvas = self.messages_frame._parent_canvas
            at_bottom = canvas.yview()[1] >= 0.985
        except Exception:
            at_bottom = True

        if position >= len(full_text):
            try:
                label.configure(text=full_text)
            except Exception:
                return
            if at_bottom:
                self._scroll_to_bottom()
            return

        step = max(6, len(full_text) // 110)
        next_position = min(position + step, len(full_text))
        try:
            label.configure(text=full_text[:next_position])
        except Exception:
            return
        if at_bottom:
            self._scroll_to_bottom()
        self.after(12, lambda: self._reveal_text(label, full_text, next_position))

    def _scroll_to_bottom(self, force: bool = False) -> None:
        if self._closed or self._scroll_pending:
            return
        self._scroll_pending = True

        def _scroll() -> None:
            self._scroll_pending = False
            if self._closed:
                return
            try:
                self.messages_frame._parent_canvas.yview_moveto(1.0)
            except Exception:
                pass

        self.after(30, _scroll)

    # ------------------------------------------------------------------
    # Indicador "digitando" (três bolhas)
    # ------------------------------------------------------------------
    def _show_typing_indicator(self) -> None:
        row, bubble, header, _fill = self._build_row(False)
        for child in header.winfo_children():
            if isinstance(child, ctk.CTkLabel) and child.cget("text") != "INSPECTOR":
                child.destroy()

        dots_box = ctk.CTkFrame(bubble, fg_color="transparent")
        dots_box.grid(row=1, column=0, padx=16, pady=(2, 12), sticky="w")
        self._typing_dots = []
        for i in range(3):
            dot = ctk.CTkLabel(dots_box, text="●", font=(FONT, 15), text_color="#2B6A88")
            dot.grid(row=0, column=i, padx=(0, 3))
            self._typing_dots.append(dot)
        ctk.CTkLabel(
            dots_box, text="Analisando sua solicitação", font=(FONT, 11), text_color=MUTED
        ).grid(row=0, column=3, padx=(8, 0))

        self._typing_row = row
        self._scroll_to_bottom(force=True)

    def _remove_typing_indicator(self) -> None:
        self._typing_dots = []
        if self._typing_row is not None:
            try:
                self._typing_row.destroy()
            except Exception:
                pass
            self._typing_row = None

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------
    def _set_status(self, state: str, title: Optional[str] = None) -> None:
        color, default_title, subtitle, badge = STATUS_STATES[state]
        self._status_state = state
        self._dot_color = color
        self._dot_dim = mix(color, GLASS, 0.62)
        self._pulsing = state in {"loading", "busy"}

        self.sidebar_status_text.configure(text=title or default_title)
        self.sidebar_status_sub.configure(text=subtitle)
        self.model_badge_text.configure(text=badge)
        for dot in (self.sidebar_status_dot, self.model_badge_dot):
            dot.configure(text_color=color)

    # ------------------------------------------------------------------
    # Carregamento do modelo e fila de resultados
    # ------------------------------------------------------------------
    def _load_model_async(self) -> None:
        if self._loading:
            return
        self._loading = True
        self._set_status("loading")
        self._update_send_state()

        def _worker() -> None:
            try:
                tokenizer, model = load_components()
                self._worker_results.put(("model_loaded", (tokenizer, model)))
            except Exception as exc:
                self._worker_results.put(("model_error", str(exc)))

        threading.Thread(target=_worker, name="inspector-model-loader", daemon=True).start()

    def _poll_worker_results(self) -> None:
        if self._closed:
            return
        while True:
            try:
                event, payload = self._worker_results.get_nowait()
            except queue.Empty:
                break
            if event == "model_loaded":
                self._on_model_loaded(*payload)
            elif event == "model_error":
                self._on_model_error(payload)
            elif event == "response":
                response, failed = payload
                self._on_response_ready(response, failed)
        if not self._closed:
            self.after(90, self._poll_worker_results)

    def _on_model_loaded(self, tokenizer, model) -> None:
        self.tokenizer = tokenizer
        self.model = model
        self._loading = False
        self._set_status("ready")
        self._update_send_state()
        self._reset_hint()

    def _on_model_error(self, message: str) -> None:
        self._loading = False
        self._set_status("error", "Falha ao carregar")
        self._update_send_state()
        self._add_message(
            "Inspector", "Não foi possível carregar o modelo.\n\n" + message, is_error=True
        )
        retry = ctk.CTkButton(
            self._last_bubble, text="Tentar novamente", height=30, corner_radius=10,
            fg_color=GLASS_HI, hover_color=BORDER, border_width=1, border_color=RED,
            text_color=TEXT, font=(FONT, 11, "bold"),
            command=lambda: self._retry_load(retry),
        )
        retry.grid(row=2, column=0, padx=16, pady=(0, 14), sticky="w")

    def _retry_load(self, button: ctk.CTkButton) -> None:
        try:
            button.destroy()
        except Exception:
            pass
        self._load_model_async()

    # ------------------------------------------------------------------
    # Campo de entrada: placeholder, altura automática, foco luminoso
    # ------------------------------------------------------------------
    def _get_input(self) -> str:
        return self.input_box.get("1.0", "end-1c").strip()

    def _on_input_modified(self, _event=None) -> None:
        box = self.input_box._textbox
        try:
            if box.edit_modified():
                box.edit_modified(False)
                self._after_input_changed()
        except tk.TclError:
            pass

    def _after_input_changed(self) -> None:
        if self._closed:
            return
        has_text = bool(self.input_box.get("1.0", "end-1c"))
        if has_text:
            self.placeholder.place_forget()
        else:
            try:
                self.placeholder.place(in_=self.input_box._textbox, x=3, y=1)
            except Exception:
                self.placeholder.place(x=24, y=16)
        self._autosize_input()
        self._update_send_state()

    def _autosize_input(self) -> None:
        box = self.input_box._textbox
        try:
            result = box.count("1.0", "end-1c", "displaylines")
            if isinstance(result, (tuple, list)):
                result = result[0] if result else 1
            lines = max(1, int(result or 1))
        except Exception:
            lines = self.input_box.get("1.0", "end-1c").count("\n") + 1
        height = max(48, min(156, lines * 22 + 8))
        if int(self.input_box.cget("height")) != height:
            self.input_box.configure(height=height)

    def _set_composer_focus(self, focused: bool) -> None:
        self._tween(
            self.composer, "border_color",
            BORDER if focused else BORDER_HI, BORDER_HI if focused else BORDER,
            steps=8, key="composer",
        )

    def _update_send_state(self) -> None:
        ready = self.model is not None and not self._loading and not self._generating
        self.send_button.set_enabled(ready and bool(self._get_input()))

    def _reset_hint(self) -> None:
        self._hint_id = None
        text = "Enter para enviar  ·  Shift + Enter para nova linha"
        if self._loading:
            text = "O motor está carregando — você já pode ir escrevendo."
        self.hint_label.configure(text=text, text_color=SUBTLE)

    def _flash_hint(self, text: str, color: str = AMBER, ms: int = 3000) -> None:
        self.hint_label.configure(text=text, text_color=color)
        if self._hint_id is not None:
            try:
                self.after_cancel(self._hint_id)
            except Exception:
                pass
        self._hint_id = self.after(ms, self._reset_hint)

    # ------------------------------------------------------------------
    # Interações do usuário
    # ------------------------------------------------------------------
    def _use_suggestion(self, prompt: str) -> None:
        self.input_box.delete("1.0", "end")
        self.input_box.insert("1.0", prompt)
        self.input_box.focus_set()
        self._after_input_changed()

    def _on_enter(self, event) -> Optional[str]:
        # Shift+Enter insere quebra de linha. Enter envia.
        if event.state & 0x1:
            return None  # deixa o Tk inserir a quebra de linha
        self._send()
        return "break"

    def _send(self) -> None:
        user_text = self._get_input()
        if not user_text:
            return
        if self._loading or self.model is None or self.tokenizer is None:
            self._flash_hint("O motor ainda está carregando — aguarde um instante.")
            return
        if self._generating:
            self._flash_hint("Aguarde a resposta atual terminar para enviar outra.")
            return

        self._generating = True
        self.input_box.delete("1.0", "end")
        self._after_input_changed()
        self._add_message("Você", user_text)

        self._set_status("busy")
        self.clear_button.set_enabled(False)
        self.new_chat_button.set_enabled(False)
        self._show_typing_indicator()

        model, tokenizer = self.model, self.tokenizer

        def _worker() -> None:
            try:
                result = generate_result(model, tokenizer, user_text)
                response = result.text
                failed = result.status != "ok"
                if result.status == "incomplete_generation":
                    response += "\n\n[Resposta incompleta: o limite de geração foi atingido.]"
                self._worker_results.put(("response", (response, failed)))
            except Exception as exc:
                self._worker_results.put(
                    ("response", (f"Ocorreu um erro durante a inferência:\n{exc}", True))
                )

        threading.Thread(target=_worker, name="inspector-inference", daemon=True).start()

    def _on_response_ready(self, response: str, failed: bool = False) -> None:
        self._remove_typing_indicator()
        self._generating = False

        self._add_message("Inspector", response, animate=not failed, is_error=failed)
        self._set_status("error" if failed else "ready", "Falha na inferência" if failed else None)

        self.clear_button.set_enabled(True)
        self.new_chat_button.set_enabled(True)
        self._update_send_state()
        self.input_box.focus_set()

    def _clear_chat(self) -> None:
        if self._generating:
            self._flash_hint("Aguarde a resposta terminar para limpar a conversa.")
            return

        for child in self.messages_frame.winfo_children():
            child.destroy()

        self._next_row = 0
        self._welcome_visible = False
        self._body_labels = []
        self._suggestion_cards = []
        self._show_welcome_screen()
        self._schedule_reflow()
        self.input_box.focus_set()

    # ------------------------------------------------------------------
    # Responsividade
    # ------------------------------------------------------------------
    def _on_root_configure(self, event) -> None:
        if event.widget is not self or self._closed:
            return
        if self._resize_id is not None:
            try:
                self.after_cancel(self._resize_id)
            except Exception:
                pass
        self._resize_id = self.after(70, self._on_resized)

    def _on_resized(self) -> None:
        self._resize_id = None
        if self._closed:
            return
        narrow = (self.winfo_width() / self._dpi) < NARROW_BREAKPOINT
        if narrow != self._was_narrow:
            self._was_narrow = narrow
            self._set_sidebar_collapsed(narrow)
        self._schedule_reflow()

    def _schedule_reflow(self) -> None:
        if self._closed:
            return
        if self._reflow_id is not None:
            try:
                self.after_cancel(self._reflow_id)
            except Exception:
                pass
        self._reflow_id = self.after(60, self._reflow)

    def _reflow(self) -> None:
        self._reflow_id = None
        if self._closed:
            return

        wrap = self._wrap_width()
        alive = []
        for label in self._body_labels:
            try:
                if label.winfo_exists():
                    label.configure(wraplength=wrap)
                    alive.append(label)
            except Exception:
                pass
        self._body_labels = alive

        try:
            canvas_w = self.messages_frame._parent_canvas.winfo_width()
        except Exception:
            canvas_w = 900
        avail = max(260, canvas_w / self._dpi - 36)

        if self._welcome_visible:
            columns = 3 if avail >= 780 else (2 if avail >= 520 else 1)
            col_w = (avail - 40) / columns
            self._layout_suggestions(columns, col_w)
            self.welcome_text.configure(wraplength=int(max(240, min(600, avail - 40))))
            size = 30 if avail >= 600 else (24 if avail >= 440 else 20)
            for headline in (self.headline_1, self.headline_2):
                headline.configure(font=(FONT, size, "bold"))

        if (self.workspace.winfo_width() / self._dpi) < 720:
            self.subtitle.pack_forget()
        elif not self.subtitle.winfo_ismapped():
            self.subtitle.pack(anchor="w", pady=(1, 0))

        target = self.workspace.winfo_width()
        if abs(target - self._glow_width) > 24:
            self._update_glow_line(target)

    # ------------------------------------------------------------------
    # Encerramento
    # ------------------------------------------------------------------
    def _on_close(self) -> None:
        self._closed = True
        for after_id in (self._tick_id, self._reflow_id, self._resize_id, self._hint_id):
            if after_id is not None:
                try:
                    self.after_cancel(after_id)
                except Exception:
                    pass
        self.destroy()


def main() -> None:
    app = InspectorGUI()
    app.mainloop()


if __name__ == "__main__":
    main()
