"""Componentes leves do Canvas: hover sem tremor, botão fantasma e botão de enviar."""

from __future__ import annotations

import math

from PIL import ImageTk

from . import render
from . import theme as T


def photo(ui, pil_image):
    """PhotoImage mantido vivo enquanto a cena existir."""
    return ui.keep(ImageTk.PhotoImage(pil_image))


def perimeter_point(x: float, y: float, w: float, h: float, r: float, s: float):
    """Ponto sobre o contorno de um retângulo arredondado (s de 0 a 1)."""
    r = min(r, h / 2, w / 2)
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
    """Entrada/saída do ponteiro em um grupo de itens, sem piscar entre itens."""

    def __init__(self, ui, tag: str, callback) -> None:
        self.ui, self.tag, self.callback, self.active = ui, tag, callback, False
        ui.scene.tag_bind(tag, "<Enter>", self._enter, add="+")
        ui.scene.tag_bind(tag, "<Leave>", self._leave, add="+")

    def _enter(self, _e=None) -> None:
        if not self.active:
            self.active = True
            self.ui.scene.configure(cursor="hand2")
            self.callback(True)

    def _leave(self, _e=None) -> None:
        self.ui.root.after(35, self._check)

    def _check(self) -> None:
        if self.active and not self._inside():
            self.active = False
            self.ui.scene.configure(cursor="")
            self.callback(False)

    def _inside(self) -> bool:
        s = self.ui.scene
        try:
            bb = s.bbox(self.tag)
            if not bb:
                return False
            x = s.winfo_pointerx() - s.winfo_rootx()
            y = s.winfo_pointery() - s.winfo_rooty()
            return bb[0] <= x <= bb[2] and bb[1] <= y <= bb[3]
        except Exception:
            return False


class GhostButton:
    """Botão de texto: a superfície e a cor surgem suavemente ao passar o mouse."""

    def __init__(self, ui, x, y, w, h, *, text, icon=None, command=None, size=12) -> None:
        self.ui, self.command = ui, command
        self.x, self.y, self.w, self.h = int(x), int(y), int(w), int(h)
        self.tag = f"gb{id(self)}"
        self.level = 0.0
        self.enabled = True
        self._pressed = False
        self._photos: dict = {}
        s = ui.scene
        self.item = s.create_image(self.x, self.y, anchor="nw", image=self._img(0), tags=(self.tag,))
        tw = T.measure_font(size, True).measure(text)
        isz = T.px(size * 1.0) if icon else 0
        gap = T.px(8) if icon else 0
        x0 = self.x + (self.w - (isz + gap + tw)) / 2
        cy = self.y + self.h / 2
        self.fg = []
        if icon == "plus":
            r, lw = isz / 2, max(2, T.px(1.8))
            kw = dict(fill=T.MUTED, width=lw, capstyle="round", tags=(self.tag,))
            cx = x0 + r
            self.fg += [s.create_line(cx - r * .7, cy, cx + r * .7, cy, **kw),
                        s.create_line(cx, cy - r * .7, cx, cy + r * .7, **kw)]
        self.fg.append(s.create_text(x0 + isz + gap, cy, text=text, anchor="w", fill=T.MUTED,
                                     font=T.font(size, True), tags=(self.tag,)))
        Hover(ui, self.tag, self._hover)
        s.tag_bind(self.tag, "<ButtonPress-1>", self._press, add="+")
        s.tag_bind(self.tag, "<ButtonRelease-1>", self._release, add="+")

    def _img(self, k: int):
        ph = self._photos.get(k)
        if ph is None:
            ph = photo(self.ui, render.ghost(self.w, self.h, k))
            self._photos[k] = ph
        return ph

    def _set(self, level: float) -> None:
        if abs(level - self.level) < 1e-3:
            return
        prev = round(self.level)
        self.level = level
        s = self.ui.scene
        if round(level) != prev:
            s.itemconfigure(self.item, image=self._img(round(level)))
        color = T.mix(T.MUTED, T.TEXT, level / 4)
        for it in self.fg:
            s.itemconfigure(it, fill=color)

    def _hover(self, on: bool) -> None:
        if not self.enabled:
            return
        start, target = self.level, 4.0 if on else 0.0
        self.ui.anim.tween(0.22, lambda p: self._set(start + (target - start) * p), key=("gb", self.tag))

    def set_enabled(self, enabled: bool) -> None:
        self.enabled = enabled
        color = T.mix(T.MUTED, T.TEXT, self.level / 4) if enabled else T.SUBTLE
        for it in self.fg:
            self.ui.scene.itemconfigure(it, fill=color)

    def _press(self, _e=None) -> None:
        self._pressed = self.enabled

    def _release(self, e) -> None:
        was, self._pressed = self._pressed, False
        if was and self.x <= e.x <= self.x + self.w and self.y <= e.y <= self.y + self.h and self.command:
            self.command()


class SendButton:
    """Círculo de envio: apaga/acende suavemente conforme pode ou não enviar."""

    def __init__(self, ui, size, command) -> None:
        self.ui, self.size, self.command = ui, int(size), command
        self.tag = f"sb{id(self)}"
        self.enabled = False
        self.k = 0.0       # 0 desativado .. 1 ativo
        self.hv = 0.0      # 0..1 hover
        self._pressed = False
        self.cx = self.cy = 0
        s = ui.scene
        self._photos: dict = {}
        self.item = s.create_image(0, 0, image=self._img(), tags=(self.tag,))
        kw = dict(width=max(2, T.px(2)), capstyle="round", joinstyle="round", tags=(self.tag,))
        self.l1 = s.create_line(0, 0, 0, 0, fill=T.SUBTLE, **kw)
        self.l2 = s.create_line(0, 0, 0, 0, 0, 0, fill=T.SUBTLE, **kw)
        Hover(ui, self.tag, self._hover)
        s.tag_bind(self.tag, "<ButtonPress-1>", self._press, add="+")
        s.tag_bind(self.tag, "<ButtonRelease-1>", self._release, add="+")

    def _color(self) -> str:
        base = T.mix(T.SURFACE_HI, T.ACCENT, self.k)
        return T.mix(base, "#FFFFFF", 0.18 * self.hv * self.k)

    def _img(self):
        color = self._color()
        ph = self._photos.get(color)
        if ph is None:
            ph = photo(self.ui, render.circle(self.size, color))
            self._photos[color] = ph
        return ph

    def _refresh(self) -> None:
        s = self.ui.scene
        s.itemconfigure(self.item, image=self._img())
        ic = T.mix(T.SUBTLE, T.ACCENT_DK, self.k)
        s.itemconfigure(self.l1, fill=ic)
        s.itemconfigure(self.l2, fill=ic)

    def place(self, cx: float, cy: float) -> None:
        self.cx, self.cy = cx, cy
        s, r = self.ui.scene, self.size / 2
        s.coords(self.item, cx, cy)
        s.coords(self.l1, cx, cy + r * .38, cx, cy - r * .36)
        s.coords(self.l2, cx - r * .32, cy - r * .06, cx, cy - r * .38, cx + r * .32, cy - r * .06)

    def set_enabled(self, enabled: bool) -> None:
        if enabled == self.enabled:
            return
        self.enabled = enabled
        start, target = self.k, 1.0 if enabled else 0.0

        def step(p: float) -> None:
            self.k = round((start + (target - start) * p) * 12) / 12
            self._refresh()

        self.ui.anim.tween(0.28, step, key=("send", self.tag))

    def _hover(self, on: bool) -> None:
        start, target = self.hv, 1.0 if on else 0.0

        def step(p: float) -> None:
            self.hv = round((start + (target - start) * p) * 6) / 6
            self._refresh()

        self.ui.anim.tween(0.2, step, key=("sendhv", self.tag))

    def _press(self, _e=None) -> None:
        self._pressed = self.enabled

    def _release(self, e) -> None:
        was, self._pressed = self._pressed, False
        if was and abs(e.x - self.cx) <= self.size / 2 and abs(e.y - self.cy) <= self.size / 2 and self.command:
            self.command()
