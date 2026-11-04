"""Componentes desenhados no Canvas: painéis de vidro, botões e luzes animadas."""

from __future__ import annotations

import math

from PIL import ImageTk

from . import render
from . import theme as T
from .anim import ease_in_out, linear


def photo(ui, pil_image):
    """Cria um PhotoImage e o mantém referenciado enquanto a cena existir."""
    return ui.keep(ImageTk.PhotoImage(pil_image))


def perimeter_point(x: float, y: float, w: float, h: float, r: float, s: float):
    """Ponto sobre o contorno de um retângulo arredondado (s de 0 a 1)."""
    sh, sv, arc = w - 2 * r, h - 2 * r, math.pi * r / 2
    d = (s % 1.0) * (2 * sh + 2 * sv + 4 * arc)
    if d < sh:
        return x + r + d, y
    d -= sh
    if d < arc:
        a = -math.pi / 2 + d / r
        return x + w - r + r * math.cos(a), y + r + r * math.sin(a)
    d -= arc
    if d < sv:
        return x + w, y + r + d
    d -= sv
    if d < arc:
        a = d / r
        return x + w - r + r * math.cos(a), y + h - r + r * math.sin(a)
    d -= arc
    if d < sh:
        return x + w - r - d, y + h
    d -= sh
    if d < arc:
        a = math.pi / 2 + d / r
        return x + r + r * math.cos(a), y + h - r + r * math.sin(a)
    d -= arc
    if d < sv:
        return x, y + h - r - d
    d -= sv
    a = math.pi + d / r
    return x + r + r * math.cos(a), y + r + r * math.sin(a)


class Hover:
    """Detecta entrada/saída do ponteiro em um grupo de itens sem 'piscar'."""

    def __init__(self, ui, tag: str, callback) -> None:
        self.ui, self.tag, self.callback = ui, tag, callback
        self.active = False
        ui.scene.tag_bind(tag, "<Enter>", self._enter, add="+")
        ui.scene.tag_bind(tag, "<Leave>", self._leave, add="+")

    def _enter(self, _e=None) -> None:
        if not self.active:
            self.active = True
            self.callback(True)

    def _leave(self, _e=None) -> None:
        self.ui.root.after(35, self._check)

    def _check(self) -> None:
        if self.active and not self._inside():
            self.active = False
            self.callback(False)

    def _inside(self) -> bool:
        s = self.ui.scene
        try:
            bbox = s.bbox(self.tag)
            if not bbox:
                return False
            px_ = s.winfo_pointerx() - s.winfo_rootx()
            py_ = s.winfo_pointery() - s.winfo_rooty()
            return bbox[0] <= px_ <= bbox[2] and bbox[1] <= py_ <= bbox[3]
        except Exception:
            return False


class GlassPanel:
    """Painel de vidro: preenchimento fosco/translúcido + borda luminosa com níveis."""

    LEVELS = 4

    def __init__(self, ui, x, y, w, h, radius, *, tags=(), mode="frost", rim=True,
                 rim_color=T.CYAN, flat=False, tint=T.GLASS_TINT, tint_a=165,
                 alpha=(120, 70)) -> None:
        self.ui = ui
        self.x, self.y, self.w, self.h = int(x), int(y), int(w), int(h)
        self.radius = int(radius)
        self.tags = tuple(tags)
        self.rim_color = rim_color
        self.pad = T.px(16)
        self.level = 0
        self._rims: dict = {}
        self.rim_item = None

        if mode == "frost":
            self.fill_img = render.frost_fill(
                ui.bg, self.x, self.y, self.w, self.h, self.radius,
                tint=tint, tint_a=tint_a, flat=flat,
            )
        else:
            self.fill_img = render.make_glass(
                self.w, self.h, self.radius, tint=tint,
                alpha_top=alpha[0], alpha_bot=alpha[1],
            )
        self.fill_item = ui.scene.create_image(
            self.x, self.y, anchor="nw", image=photo(ui, self.fill_img), tags=self.tags
        )
        if rim:
            self.add_rim()

    def average_color(self) -> str:
        """Cor média do centro do vidro (para casar o fundo de widgets opacos)."""
        w, h = self.fill_img.size
        c = self.fill_img.crop((w // 4, h // 4, 3 * w // 4, 3 * h // 4)).resize((1, 1))
        return T.hexcolor(c.convert("RGB").getpixel((0, 0)))

    def add_rim(self) -> None:
        self.rim_item = self.ui.scene.create_image(
            self.x - self.pad, self.y - self.pad, anchor="nw",
            image=self._rim_photo(0), tags=self.tags,
        )

    def _rim_photo(self, k: int):
        ph = self._rims.get(k)
        if ph is None:
            intensity = 0.42 + 0.58 * k / self.LEVELS
            ph = photo(self.ui, render.make_rim(
                self.w, self.h, self.radius, intensity=intensity,
                color=self.rim_color, pad=self.pad,
            ))
            self._rims[k] = ph
        return ph

    def set_level(self, k: int) -> None:
        k = max(0, min(self.LEVELS, int(k)))
        if k != self.level and self.rim_item is not None:
            self.level = k
            self.ui.scene.itemconfigure(self.rim_item, image=self._rim_photo(k))

    def glow_to(self, target: int, duration: float = 0.22, key=None) -> None:
        start = self.level

        def step(p: float) -> None:
            self.set_level(round(start + (target - start) * p))

        self.ui.anim.tween(duration, step, key=key or ("panel", id(self)))

    def add_flare(self, edge: str, *, length: int, speed: float = 0.08,
                  phase: float = 0.0, color: str = T.CYAN, group: str = "scene") -> None:
        """Faixa de luz que desliza lentamente ao longo de uma aresta do vidro."""
        s, ui = self.ui.scene, self.ui
        vertical = edge in ("left", "right")
        thick = T.px(26)
        photos = [
            photo(ui, render.make_flare(length, thick, color, k, vertical))
            for k in (0.35, 0.65, 1.0)
        ]
        item = s.create_image(0, 0, image=photos[1], tags=self.tags)
        r = self.radius
        if vertical:
            lo, hi = self.y + r + length / 2, self.y + self.h - r - length / 2
            fixed = self.x if edge == "left" else self.x + self.w
        else:
            lo, hi = self.x + r + length / 2, self.x + self.w - r - length / 2
            fixed = self.y if edge == "top" else self.y + self.h
        if hi <= lo:
            s.delete(item)
            return
        state = {"k": 1}

        def tick(t: float, _dt: float) -> None:
            u = ease_in_out(0.5 + 0.5 * math.sin(t * speed * 2 * math.pi + phase))
            pos = lo + (hi - lo) * u
            if vertical:
                s.coords(item, fixed, pos)
            else:
                s.coords(item, pos, fixed)
            b = 0.5 + 0.5 * math.sin(t * speed * 5.3 + phase * 1.7)
            k = 0 if b < 0.33 else (1 if b < 0.66 else 2)
            if k != state["k"]:
                state["k"] = k
                s.itemconfigure(item, image=photos[k])

        ui.anim.add(tick, group)


class GlassButton:
    """Botão de vidro com halo, hover suave e reflexo de luz que atravessa."""

    GLINT_FRAMES = 7

    def __init__(self, ui, x, y, w, h, *, text="", icon=None, style="primary",
                 command=None, font_size=13) -> None:
        self.ui = ui
        self.x, self.y, self.w, self.h = int(x), int(y), int(w), int(h)
        self.style, self.command = style, command
        self.pad = T.px(10)
        self.tag = f"btn{id(self)}"
        self.level = 0
        self.enabled = True
        self._pressed = False
        self._glint_on = False
        self._photos: dict = {}
        self._glints: list = []

        s = ui.scene
        self.item = s.create_image(
            self.x - self.pad, self.y - self.pad, anchor="nw",
            image=self._photo(0), tags=(self.tag,),
        )
        self.fg: list = []
        self._build_fg(text, icon, font_size)

        Hover(ui, self.tag, self._hover)
        s.tag_bind(self.tag, "<ButtonPress-1>", self._press, add="+")
        s.tag_bind(self.tag, "<ButtonRelease-1>", self._release, add="+")

    # -- imagens ------------------------------------------------------------
    def _photo(self, k: int):
        key = ("dis",) if not self.enabled else ("lvl", k)
        ph = self._photos.get(key)
        if ph is None:
            ph = photo(self.ui, render.make_button(
                self.w, self.h, self.style, 0.0 if not self.enabled else k / 4,
                self.pad, disabled=not self.enabled,
            ))
            self._photos[key] = ph
        return ph

    def _fg_color(self) -> str:
        if not self.enabled:
            return "#5C7FA6"
        return "#F4FAFF" if self.style == "primary" else T.TEXT

    def _build_fg(self, text, icon, font_size) -> None:
        s, ui = self.ui.scene, self.ui
        color = self._fg_color()
        cx, cy = self.x + self.w / 2, self.y + self.h / 2
        font = T.font(font_size, True)
        tw = T.measure_font(font_size, True).measure(text) if text else 0
        isz = T.px(font_size * 1.25) if icon else 0
        gap = T.px(9) if (icon and text) else 0
        x0 = cx - (isz + gap + tw) / 2
        tags = (self.tag,)
        if icon:
            self._icon(icon, x0 + isz / 2, cy, isz, color, tags)
        if text:
            self.fg.append(s.create_text(
                x0 + isz + gap, cy, text=text, anchor="w", fill=color, font=font, tags=tags
            ))

    def _icon(self, kind, cx, cy, size, color, tags) -> None:
        s = self.ui.scene
        lw = max(2, T.px(2.4))
        r = size / 2
        kw = dict(fill=color, width=lw, capstyle="round", joinstyle="round", tags=tags)
        if kind == "plus":
            lines = [s.create_line(cx - r * .8, cy, cx + r * .8, cy, **kw),
                     s.create_line(cx, cy - r * .8, cx, cy + r * .8, **kw)]
        elif kind == "send":
            lines = [s.create_line(cx, cy + r * .85, cx, cy - r * .85, **kw),
                     s.create_line(cx - r * .78, cy - r * .08, cx, cy - r * .85,
                                   cx + r * .78, cy - r * .08, **kw)]
        else:
            lines = [s.create_line(cx - r * .8, cy + d, cx + r * .8, cy + d, **kw)
                     for d in (-r * .62, 0, r * .62)]
        self.fg.extend(lines)

    # -- estados ------------------------------------------------------------
    def set_enabled(self, enabled: bool) -> None:
        if enabled == self.enabled:
            return
        self.enabled = enabled
        self.level = 0
        s = self.ui.scene
        s.itemconfigure(self.item, image=self._photo(0))
        for item in self.fg:
            try:
                if s.type(item) == "line":
                    s.itemconfigure(item, fill=self._fg_color())
                else:
                    s.itemconfigure(item, fill=self._fg_color())
            except Exception:
                pass

    def _set_level(self, k: int) -> None:
        k = max(0, min(4, int(k)))
        if k == self.level:
            return
        self.level = k
        if self.enabled and not self._glint_on:
            self.ui.scene.itemconfigure(self.item, image=self._photo(k))

    def _hover(self, on: bool) -> None:
        if not self.enabled:
            return
        self.ui.scene.configure(cursor="hand2" if on else "")
        start, target = self.level, 4 if on else 0
        self.ui.anim.tween(
            0.2, lambda p: self._set_level(round(start + (target - start) * p)),
            key=("btn", self.tag),
        )
        if on:
            self._play_glint()

    def _play_glint(self) -> None:
        if self._glint_on:
            return
        if not self._glints:
            base = render.make_button(self.w, self.h, self.style, 1.0, self.pad)
            for i in range(self.GLINT_FRAMES):
                t = i / (self.GLINT_FRAMES - 1)
                frame = base.copy()
                frame.alpha_composite(
                    render.make_glint(self.w, self.h, self.h // 2, t, self.pad,
                                      0.6 if self.style == "primary" else 0.4)
                )
                self._glints.append(photo(self.ui, frame))
        self._glint_on = True
        s = self.ui.scene

        def step(p: float) -> None:
            s.itemconfigure(self.item, image=self._glints[min(self.GLINT_FRAMES - 1, int(p * self.GLINT_FRAMES))])

        def done() -> None:
            self._glint_on = False
            if self.enabled:
                s.itemconfigure(self.item, image=self._photo(self.level))

        self.ui.anim.tween(0.55, step, ease=linear, key=("glint", self.tag), on_done=done)

    def _press(self, _e=None) -> None:
        if self.enabled:
            self._pressed = True

    def _release(self, e) -> None:
        was, self._pressed = self._pressed, False
        inside = (self.x <= e.x <= self.x + self.w) and (self.y <= e.y <= self.y + self.h)
        if self.enabled and was and inside and self.command:
            self.command()
