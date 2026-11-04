"""Aplicação principal: interface minimalista, escura e fluida (tkinter + Pillow)."""

from __future__ import annotations

import math
import queue
import threading
import tkinter as tk
from typing import Optional

from PIL import Image, ImageTk

from inspector.engine import generate_response, load_components

from . import render
from . import theme as T
from .anim import Animator, ease_in_out, ease_out, linear
from .widgets import GhostButton, Hover, SendButton, perimeter_point, photo

STATUS = {
    "loading": (T.WARN, "Carregando modelo"),
    "ready": (T.OK, "Pronto"),
    "busy": (T.ACCENT, "Pensando"),
    "error": (T.ERR, "Indisponível"),
}

SUGGESTIONS = (
    "Quero criar uma plataforma para organizar estudos e tarefas.",
    "Analise meu projeto de software e identifique melhorias prioritárias.",
    "Preciso de um plano de estudos de programação para três meses.",
)

WELCOME_TITLE = "O que vamos estruturar hoje?"
WELCOME_SUB = ("Descreva uma ideia ou tarefa. O Inspector ajuda a estruturá-la "
               "em Role, Context, Focus e Output.")

MAX_LINES = 6


class InspectorApp:
    def __init__(self) -> None:
        self.root = tk.Tk()
        T.pick_font(self.root)
        T.set_scale(self.root.winfo_fpixels("1i") / 96.0)
        self.root.title("InspectorAI")
        self.root.configure(bg=T.BG)
        sw, sh = self.root.winfo_screenwidth(), self.root.winfo_screenheight()
        w, h = min(T.px(1100), sw - T.px(80)), min(T.px(820), sh - T.px(120))
        self.root.geometry(f"{w}x{h}+{(sw - w) // 2}+{max(0, (sh - h) // 2 - 20)}")
        self.root.minsize(T.px(520), T.px(520))

        self.scene = tk.Canvas(self.root, bg=T.BG, highlightthickness=0, bd=0)
        self.scene.pack(fill="both", expand=True)
        self.anim = Animator(self.root)

        # modelo
        self.tokenizer = self.model = None
        self._loading = self._generating = self._closed = False
        self._results: queue.Queue = queue.Queue()

        # cena
        self._refs: list = []
        self.W = self.H = 0
        self._layout_key = None
        self._layout_job = None
        self.col = (0, 0)
        self.head = 0
        self.view = (0, 0, 0, 0)

        # mensagens
        self.msgs: list[dict] = []
        self.scroll = self.scroll_target = 0.0
        self.content_h = 0
        self._next_y = 0
        self._typing_h = 0
        self._bubbles: dict = {}
        self._reveal_keys: list = []
        self._welcome_visible = True
        self._welcome_items: list = []
        self._animate_welcome = True
        self._thumb_vis = 0.0
        self._thumb_shown = None

        # compositor
        self.lh = T.measure_font(12).metrics("linespace")
        self.pad_y = T.px(17)
        self.comp_h = float(self.lh + 2 * self.pad_y)
        self._comp_goal = self.comp_h
        self._field_level = 0.0
        self._field_cache: dict = {}
        self._field_rect = (0, 0, 0, 0)
        self._hint_job = None

        # status / luz
        self._status_state, self._status_title = "loading", None
        self._status_color = T.WARN
        self._pulsing = True
        self._light_cache: dict = {}
        self._dot = None
        self._chase_items: list = []
        self.send = self.btn_new = None

        self._make_input()
        self.scene.bind("<Configure>", self._on_configure)
        self.root.bind("<Control-n>", lambda _e: self._clear_chat())
        self.root.bind("<Control-l>", lambda _e: self.input.focus_set())
        for seq in ("<MouseWheel>", "<Button-4>", "<Button-5>"):
            self.root.bind_all(seq, self._on_wheel)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        self.anim.add(self._thumb_tick, "core")
        self.anim.add(self._scroll_tick, "core")
        self.root.after(90, self._poll)
        self.root.after(40, self._layout)
        self._load_model_async()

    # ------------------------------------------------------------------
    # Infra
    # ------------------------------------------------------------------
    def keep(self, ph):
        self._refs.append(ph)
        return ph

    def run(self) -> None:
        self.root.mainloop()

    def _on_close(self) -> None:
        self._closed = True
        self.anim.stop()
        self.root.destroy()

    def _make_input(self) -> None:
        self.input = tk.Text(
            self.scene, bd=0, highlightthickness=0, relief="flat", wrap="word",
            font=T.font(12), fg=T.TEXT, bg=T.SURFACE, insertbackground=T.ACCENT,
            insertwidth=max(2, T.px(2)), selectbackground="#2A3F66",
            selectforeground=T.TEXT, padx=0, pady=0, undo=True, spacing1=0, spacing3=0,
        )
        self.placeholder = tk.Label(
            self.input, text="Descreva sua ideia ou solicitação…", fg=T.SUBTLE,
            bg=T.SURFACE, font=T.font(12), bd=0, anchor="w",
        )
        self.placeholder.place(x=0, y=0)
        self.placeholder.bind("<Button-1>", lambda _e: self.input.focus_set())
        self.input.bind("<Return>", self._on_enter)
        self.input.bind("<KP_Enter>", self._on_enter)
        self.input.bind("<FocusIn>", lambda _e: self._field_glow(True))
        self.input.bind("<FocusOut>", lambda _e: self._field_glow(False))
        self.input.bind("<<Modified>>", self._on_modified)

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------
    def _on_configure(self, _e=None) -> None:
        if self._layout_job is None:  # throttle: acompanha o arraste sem refazer a cada pixel
            self._layout_job = self.root.after(30, self._layout)

    def _layout(self, force: bool = False) -> None:
        self._layout_job = None
        if self._closed:
            return
        s = self.scene
        W, H = s.winfo_width(), s.winfo_height()
        if W < T.px(300) or H < T.px(300):
            return
        if (W, H) == self._layout_key and not force:
            return
        self._layout_key = (W, H)
        at_bottom = self.scroll_target >= self._max_scroll() - 1

        for group in ("scene", "msgs", "chase"):
            self.anim.clear(group)
        for key in self._reveal_keys:
            self.anim.cancel(key)
        self._reveal_keys = []
        s.delete("all")
        self._refs, self._bubbles, self._field_cache = [], {}, {}
        self._chase_items, self._dot = [], None
        self.W, self.H = W, H

        colw = min(T.px(780), W - 2 * T.px(20))
        x0 = (W - colw) // 2
        self.col = (x0, x0 + colw)
        self.head = T.px(60)
        self._compute_view()

        s.create_rectangle(0, 0, W, self.head, fill=T.BG, outline="", tags=("cover",))
        self.cover_b = s.create_rectangle(0, self.view[3], W, H, fill=T.BG, outline="", tags=("cover",))
        fh = T.px(30)
        s.create_image(0, self.head, anchor="nw", tags=("cover",),
                       image=photo(self, render.fade_strip(W, fh, True)))
        self.fade_b = s.create_image(0, self.view[3] - fh, anchor="nw", tags=("cover",),
                                     image=photo(self, render.fade_strip(W, fh, False)))
        self.fade_h = fh
        self.thumb = s.create_line(0, 0, 0, 0, fill=T.BG, width=max(3, T.px(3)),
                                   capstyle="round", state="hidden")

        self._build_messages()
        self._build_header()
        self._build_composer()
        self._apply_status()
        self.anim.add(self._tick_light, "scene")
        if self._generating:
            self._start_chase()

        self.scroll = min(self.scroll, self._max_scroll())
        if at_bottom:
            self.scroll_target = self._max_scroll()
        self.scroll_target = min(self.scroll_target, self._max_scroll())
        self._update_send_state()
        self._after_input_changed(resize=False)

    def _compute_view(self) -> None:
        comp_y = self.H - T.px(40) - int(round(self.comp_h))
        self.view = (0, self.head, self.W, comp_y - T.px(14))

    # -- cabeçalho ----------------------------------------------------------------
    def _build_header(self) -> None:
        s = self.scene
        cx0, cx1 = self.col
        cy = self.head / 2
        s.create_text(cx0, cy, text="Inspector", anchor="w", fill=T.TEXT, font=T.font(14, True))
        ww = T.measure_font(14, True).measure("Inspector")
        dx = cx0 + ww + T.px(22)
        glow = s.create_image(dx, cy, image=self._light_photos(self._status_color)[3])
        r = T.px(3.5)
        core = s.create_oval(dx - r, cy - r, dx + r, cy + r, fill=self._status_color, outline="")
        label = s.create_text(dx + T.px(14), cy, text="", anchor="w", fill=T.MUTED, font=T.font(10))
        self._dot = (glow, core, label)

        bw, bh = T.px(150), T.px(36)
        self.btn_new = GhostButton(self, cx1 - bw, (self.head - bh) / 2, bw, bh,
                                   text="Nova conversa", icon="plus", command=self._clear_chat)

    # -- compositor -----------------------------------------------------------------
    def _build_composer(self) -> None:
        s = self.scene
        self.field_item = s.create_image(0, 0, anchor="nw")
        self.text_win = s.create_window(0, 0, window=self.input, anchor="nw")
        self.send = SendButton(self, T.px(40), self._send)
        self.hint = s.create_text(self.W / 2, self.H - T.px(16), text="", fill=T.SUBTLE,
                                  font=T.font(9))
        self._reset_hint()
        self._apply_composer()
        self.send.set_enabled(False)

    def _apply_composer(self, stick: bool = False) -> None:
        s = self.scene
        cx0, cx1 = self.col
        colw = cx1 - cx0
        h = int(round(self.comp_h))
        comp_bottom = self.H - T.px(40)
        comp_y = comp_bottom - h
        self._field_rect = (cx0, comp_y, colw, h)

        pad = T.px(28)
        radius = T.px(28)
        lv = int(round(self._field_level * 8))
        key = (colw, h, lv)
        ph = self._field_cache.get(key)
        if ph is None:
            if len(self._field_cache) > 80:
                self._field_cache.clear()
            ph = ImageTk.PhotoImage(render.field(colw, h, radius, lv, pad))
            self._field_cache[key] = ph
        s.itemconfigure(self.field_item, image=ph)
        s.coords(self.field_item, cx0 - pad, comp_y - pad)

        size = self.send.size
        tw = colw - T.px(22) - T.px(14) - size - T.px(10)
        s.coords(self.text_win, cx0 + T.px(22), comp_y + self.pad_y)
        s.itemconfigure(self.text_win, width=tw, height=max(self.lh, h - 2 * self.pad_y))
        self.send.place(cx1 - T.px(10) - size / 2, comp_bottom - T.px(8) - size / 2)
        s.coords(self.hint, self.W / 2, self.H - T.px(16))

        self._compute_view()
        vy1 = self.view[3]
        s.coords(self.cover_b, 0, vy1, self.W, self.H)
        s.coords(self.fade_b, 0, vy1 - self.fade_h)
        if stick:
            self._apply_scroll(self._max_scroll())
            self.scroll_target = self._max_scroll()
        self._update_thumb()

    def _field_glow(self, focused: bool) -> None:
        start = self._field_level
        target = 1.0 if focused else (0.4 if self._generating else 0.0)

        def step(p: float) -> None:
            self._field_level = start + (target - start) * p
            self._apply_composer()

        self.anim.tween(0.32, step, key="field")

    def _on_modified(self, _e=None) -> None:
        try:
            if self.input.edit_modified():
                self.input.edit_modified(False)
                self._after_input_changed()
        except tk.TclError:
            pass

    def _after_input_changed(self, resize: bool = True) -> None:
        if self.input.get("1.0", "end-1c"):
            self.placeholder.place_forget()
        else:
            self.placeholder.place(x=0, y=0)
        self._update_send_state()
        if not resize:
            return
        try:
            res = self.input.count("1.0", "end-1c", "displaylines")
            if isinstance(res, (tuple, list)):
                res = res[0] if res else 1
            lines = max(1, int(res or 1))
        except Exception:
            lines = self.input.get("1.0", "end-1c").count("\n") + 1
        goal = float(min(lines, MAX_LINES) * self.lh + 2 * self.pad_y)
        if abs(goal - self._comp_goal) < 0.5:
            return
        self._comp_goal = goal
        start = self.comp_h

        def step(p: float) -> None:
            self.comp_h = start + (goal - start) * p
            self._apply_composer(stick=at_bottom())

        def at_bottom() -> bool:
            return self.scroll_target >= self._max_scroll() - self.lh * 2

        self.anim.tween(0.22, step, key="grow")

    def _update_send_state(self) -> None:
        ready = self.model is not None and not self._loading and not self._generating
        if self.send is not None:
            self.send.set_enabled(ready and bool(self._get_input()))
        if self.btn_new is not None:
            self.btn_new.set_enabled(not self._generating)

    def _get_input(self) -> str:
        return self.input.get("1.0", "end-1c").strip()

    def _reset_hint(self) -> None:
        self._hint_job = None
        text = "Enter envia  ·  Shift + Enter quebra a linha"
        if self._loading:
            text = "O modelo está carregando — você já pode ir escrevendo"
        self.scene.itemconfigure(self.hint, text=text, fill=T.SUBTLE)

    def _flash_hint(self, text: str, ms: int = 3000) -> None:
        self.scene.itemconfigure(self.hint, text=text, fill=T.WARN)
        if self._hint_job is not None:
            self.root.after_cancel(self._hint_job)
        self._hint_job = self.root.after(ms, self._reset_hint)

    # ------------------------------------------------------------------
    # Luz: status e processamento
    # ------------------------------------------------------------------
    def _light_photos(self, color: str):
        arr = self._light_cache.get(color)
        if arr is None:
            arr = [ImageTk.PhotoImage(render.glow(T.px(13), color, k))
                   for k in (0.15, 0.3, 0.5, 0.7, 0.9, 1.0)]
            self._light_cache[color] = arr
        return arr

    def _tick_light(self, t: float, _dt: float) -> None:
        if self._dot is None:
            return
        k = int(2.5 + 2.5 * math.sin(t * 3.6)) if self._pulsing else 3
        k = max(0, min(5, k))
        if k != getattr(self, "_light_k", -1):
            self._light_k = k
            self.scene.itemconfigure(self._dot[0], image=self._light_photos(self._status_color)[k])

    def _set_status(self, state: str, title: Optional[str] = None) -> None:
        self._status_state, self._status_title = state, title
        self._apply_status()

    def _apply_status(self) -> None:
        color, label = STATUS[self._status_state]
        self._status_color = color
        self._pulsing = self._status_state in {"loading", "busy"}
        self._light_k = -1
        if self._dot is not None:
            glow, core, text = self._dot
            self.scene.itemconfigure(core, fill=color)
            self.scene.itemconfigure(text, text=self._status_title or label)

    def _chase_photo(self, k: float):
        key = ("chase", k)
        ph = self._light_cache.get(key)
        if ph is None:
            ph = ImageTk.PhotoImage(render.glow(T.px(15), T.ACCENT, k))
            self._light_cache[key] = ph
        return ph

    def _start_chase(self) -> None:
        if self._chase_items:
            return
        s = self.scene
        items = [s.create_image(0, 0, image=self._chase_photo(k), tags=("chase",))
                 for k in (1.0, 0.6, 0.36, 0.2, 0.1)]
        self._chase_items = items

        def tick(t: float, _dt: float) -> None:
            x, y, w, h = self._field_rect
            base = (t * 0.28) % 1.0
            for j, it in enumerate(items):
                px_, py_ = perimeter_point(x, y, w, h, T.px(28), base - j * 0.01)
                s.coords(it, px_, py_)

        self.anim.add(tick, "chase")

    def _stop_chase(self) -> None:
        self.anim.clear("chase")
        self.scene.delete("chase")
        self._chase_items = []

    # ------------------------------------------------------------------
    # Mensagens
    # ------------------------------------------------------------------
    EDGE = 34
    GAP = 26

    def _Y(self, y_rel: float) -> float:
        return self.view[1] + y_rel - self.scroll

    def _max_scroll(self) -> float:
        return max(0.0, self.content_h - (self.view[3] - self.view[1]))

    def _build_messages(self) -> None:
        for m in self.msgs:
            m["_pending"] = False
        edge, gap = T.px(self.EDGE), T.px(self.GAP)
        self._welcome_items = []
        if not self.msgs and self._welcome_visible:
            self.content_h = self._build_welcome(edge)
        else:
            self._welcome_visible = False
            y = edge
            for i, m in enumerate(self.msgs):
                y += self._build_item(i, m, y) + gap
            self._next_y = y
            self.content_h = (y - gap + edge) if self.msgs else 0
        self.scene.tag_lower("msgs", "cover")
        self._update_thumb()

    def _build_item(self, i: int, m: dict, y: float, pending: bool = False) -> int:
        if m.get("typing"):
            self._typing_h = self._build_typing(y)
            return self._typing_h
        return self._build_message(i, m, y, pending)

    def _build_message(self, i: int, m: dict, y: float, pending: bool) -> int:
        s = self.scene
        cx0, cx1 = self.col
        avail = cx1 - cx0
        tags = ("msgs", f"m{i}")
        y0 = self._Y(y)
        err = bool(m.get("error"))
        info = {"texts": [], "image": None, "body": None, "copy": None}
        font = T.font(12)

        if m["role"] == "user":
            padx, pady = T.px(18), T.px(12)
            body = s.create_text(0, 0, text=m["text"], anchor="nw", fill=T.TEXT, font=font,
                                 width=int(avail * 0.82) - 2 * padx, tags=tags)
            bb = s.bbox(body)
            bw, bh = bb[2] - bb[0] + 2 * padx, bb[3] - bb[1] + 2 * pady
            x0 = cx1 - bw
            item = s.create_image(x0, y0, anchor="nw", tags=tags,
                                  image=photo(self, render.shape(bw, bh, T.px(20), T.USER)))
            s.tag_lower(item, body)
            s.coords(body, x0 + padx, y0 + pady)
            info.update(texts=[(body, T.TEXT)], image=(item, bw, bh), body=body)
            self._bubbles[i] = info
            return int(bh)

        color = T.mix(T.ERR, T.TEXT, 0.35) if err else T.TEXT
        label = s.create_text(cx0, y0, anchor="nw", text="Erro" if err else "Inspector",
                              fill=T.ERR if err else T.SUBTLE, font=T.font(10, True), tags=tags)
        body = s.create_text(cx0, y0 + T.px(24), anchor="nw", text=m["text"], fill=color,
                             font=font, width=min(avail, T.px(720)), tags=tags)
        bb = s.bbox(body)
        th = bb[3] - bb[1]
        if pending:
            s.itemconfigure(body, text="")
        texts = [(label, T.ERR if err else T.SUBTLE), (body, color)]
        if not err:
            tag = f"copy{i}"
            copy = s.create_text(cx0, y0 + T.px(24) + th + T.px(10), anchor="nw", text="Copiar",
                                 fill=T.SUBTLE, font=T.font(10, True), tags=tags + (tag,))
            Hover(self, tag, lambda on, c=copy: self.anim.tween(
                0.18, lambda p, c=c, on=on: s.itemconfigure(
                    c, fill=T.mix(T.SUBTLE, T.ACCENT, p) if on else T.mix(T.ACCENT, T.SUBTLE, p)),
                key=("cp", c)))
            s.tag_bind(tag, "<Button-1>", lambda _e, c=copy, t=m["text"]: self._copy(c, t))
            if pending:
                s.itemconfigure(copy, state="hidden")
            info["copy"] = copy
        info.update(texts=texts, body=body)
        self._bubbles[i] = info
        return T.px(24) + th + T.px(10) + T.px(18)

    def _build_typing(self, y: float) -> int:
        s = self.scene
        cx0 = self.col[0]
        y0 = self._Y(y)
        r = T.px(3.5)
        dots = [s.create_oval(0, 0, 1, 1, fill=T.SUBTLE, outline="", tags=("msgs", "typing"))
                for _ in range(3)]
        h = T.px(28)

        def tick(t: float, _dt: float) -> None:
            base_y = self._Y(y) + h / 2
            for j, d in enumerate(dots):
                ph = t * 5.2 - j * 0.9
                x = cx0 + T.px(6) + j * T.px(16)
                yy = base_y - max(0.0, math.sin(ph)) * T.px(5)
                s.coords(d, x - r, yy - r, x + r, yy + r)
                s.itemconfigure(d, fill=T.mix(T.SUBTLE, T.ACCENT, max(0.0, math.sin(ph))))

        self.anim.add(tick, "msgs")
        tick(0.0, 0.0)
        return h

    # -- boas-vindas -------------------------------------------------------------------
    def _build_welcome(self, edge: int) -> int:
        s = self.scene
        cx0, cx1 = self.col
        avail = cx1 - cx0
        cx = (cx0 + cx1) // 2
        vh = self.view[3] - self.view[1]
        tags = ("msgs", "welcome")
        items: list = []

        tsize = 22 if avail >= T.px(520) else 17
        title = s.create_text(cx, 0, text=WELCOME_TITLE, anchor="n", fill=T.TEXT,
                              font=T.font(tsize), width=avail, justify="center", tags=tags)
        sub = s.create_text(cx, 0, text=WELCOME_SUB, anchor="n", fill=T.MUTED, font=T.font(11),
                            width=min(T.px(500), avail), justify="center", tags=tags)
        tb, sb = s.bbox(title), s.bbox(sub)
        th, sh = tb[3] - tb[1], sb[3] - sb[1]
        row_h = T.px(54)
        total = th + T.px(14) + sh + T.px(34) + len(SUGGESTIONS) * row_h
        top = max(edge, (vh - total) // 2)
        s.coords(title, cx, self._Y(top))
        s.coords(sub, cx, self._Y(top + th + T.px(14)))
        items += [(title, T.TEXT), (sub, T.MUTED)]

        glow = s.create_image(cx, self._Y(top + th / 2), tags=tags,
                              image=photo(self, render.glow(T.px(280), T.ACCENT, 0.13)))
        s.tag_lower(glow, title)
        self._welcome_glow = glow
        base_rel = top + th / 2

        def drift(t: float, _dt: float) -> None:
            s.coords(glow, cx + math.sin(t * 0.35) * T.px(70),
                     self._Y(base_rel) + math.cos(t * 0.27) * T.px(26))

        self.anim.add(drift, "msgs")

        rw = min(T.px(560), avail)
        rx = cx - rw // 2
        ry0 = top + th + T.px(14) + sh + T.px(34)
        hit = photo(self, Image.new("RGBA", (rw, row_h), (0, 0, 0, 0)))
        for i, text in enumerate(SUGGESTIONS):
            ry = self._Y(ry0 + i * row_h)
            tag = f"sug{i}"
            rt = tags + (tag,)
            line = s.create_line(rx, ry, rx + rw, ry, fill=T.BORDER, tags=rt)
            s.create_image(rx, ry, anchor="nw", image=hit, tags=rt)
            label = s.create_text(rx + T.px(4), ry + row_h / 2, text=text, anchor="w",
                                  fill=T.MUTED, font=T.font(11), width=rw - T.px(44), tags=rt)
            cxx, cyy, a = rx + rw - T.px(14), ry + row_h / 2, T.px(5)
            chev = s.create_line(cxx - a * .6, cyy - a, cxx + a * .4, cyy, cxx - a * .6, cyy + a,
                                 fill=T.SUBTLE, width=max(2, T.px(1.8)), capstyle="round",
                                 joinstyle="round", tags=rt)
            items += [(line, T.BORDER), (label, T.MUTED), (chev, T.SUBTLE)]
            state = {"x": 0.0}

            def hover(on, label=label, chev=chev, state=state) -> None:
                start, target = state["x"], 1.0 if on else 0.0

                def step(p, label=label, chev=chev, state=state, start=start, target=target):
                    v = start + (target - start) * p
                    s.move(chev, (v - state["x"]) * T.px(5), 0)
                    state["x"] = v
                    s.itemconfigure(label, fill=T.mix(T.MUTED, T.TEXT, v))
                    s.itemconfigure(chev, fill=T.mix(T.SUBTLE, T.ACCENT, v))

                self.anim.tween(0.22, step, key=("sug", label))

            Hover(self, tag, hover)
            s.tag_bind(tag, "<Button-1>", lambda _e, t=text: self._use_suggestion(t))
        s.create_line(rx, self._Y(ry0 + len(SUGGESTIONS) * row_h), rx + rw,
                      self._Y(ry0 + len(SUGGESTIONS) * row_h), fill=T.BORDER, tags=tags)

        self._welcome_items = items
        if self._animate_welcome:
            self._animate_welcome = False
            self._fade_items(items, delay=60, stagger=70, dur=0.5, start_color=T.BG)
        return top + total + edge

    def _fade_items(self, items, *, delay=0, stagger=0, dur=0.4, start_color=T.BG, reverse=False,
                    on_done=None) -> None:
        s = self.scene
        for n, (it, final) in enumerate(items):
            a, b = (final, start_color) if reverse else (start_color, final)
            if not reverse:
                s.itemconfigure(it, fill=a)

            def go(it=it, a=a, b=b, last=(n == len(items) - 1)):
                self.anim.tween(dur, lambda p: s.itemconfigure(it, fill=T.mix(a, b, p)),
                                key=("fade", it), on_done=on_done if last else None)

            self.root.after(delay + stagger * n // 2, go)

    def _dismiss_welcome(self, done) -> None:
        if not self._welcome_visible:
            done()
            return
        self._welcome_visible = False
        try:
            self.scene.delete(self._welcome_glow)
        except Exception:
            pass
        self.anim.clear("msgs")

        def finish() -> None:
            self.scene.delete("welcome")
            self._welcome_items = []
            self._reset_flow()
            done()

        if self._welcome_items:
            self._fade_items(self._welcome_items, dur=0.18, reverse=True, on_done=finish)
        else:
            finish()

    def _reset_flow(self) -> None:
        self._next_y = T.px(self.EDGE)
        self.content_h = 0
        self.scroll = self.scroll_target = 0.0

    # -- adicionar / animar mensagens ------------------------------------------------------
    def _append_item(self, m: dict, pending: bool = False) -> None:
        i = len(self.msgs) - 1
        edge, gap = T.px(self.EDGE), T.px(self.GAP)
        if len(self.msgs) == 1 and self._next_y < edge:
            self._next_y = edge
        y = self._next_y
        h = self._build_item(i, m, y, pending)
        self._next_y = y + h + gap
        self.content_h = self._next_y - gap + edge
        self.scene.tag_lower("msgs", "cover")
        self._update_thumb()

    def _add_message(self, role: str, text: str, *, error: bool = False,
                     animate: bool = False) -> None:
        if self._welcome_visible:  # entrada direta (ex.: erro antes de qualquer envio)
            self._welcome_visible = False
            self.scene.delete("welcome")
            self.anim.clear("msgs")
            self._reset_flow()
        m = {"role": role, "text": text, "error": error, "_pending": animate and not error}
        self.msgs.append(m)
        self._append_item(m, pending=m["_pending"])
        self.scroll_target = self._max_scroll()
        i = len(self.msgs) - 1
        self._animate_in(i)
        if m["_pending"]:
            self.root.after(160, lambda: self._reveal(i))

    def _animate_in(self, i: int) -> None:
        info = self._bubbles.get(i)
        if not info:
            return
        s = self.scene
        dy0 = T.px(14)
        for it, _c in info["texts"]:
            s.itemconfigure(it, fill=T.BG)
        frames = None
        if info["image"]:
            item, bw, bh = info["image"]
            frames = [photo(self, render.shape(bw, bh, T.px(20), T.mix(T.BG, T.USER, k / 5)))
                      for k in range(6)]
            s.itemconfigure(item, image=frames[0])
        tag = f"m{i}"
        s.move(tag, 0, dy0)
        cur = {"dy": float(dy0)}

        def step(p: float) -> None:
            target = dy0 * (1 - p)
            s.move(tag, 0, target - cur["dy"])
            cur["dy"] = target
            for it, final in info["texts"]:
                s.itemconfigure(it, fill=T.mix(T.BG, final, p))
            if frames:
                s.itemconfigure(info["image"][0], image=frames[min(5, int(p * 6))])

        self.anim.tween(0.38, step, ease=ease_out, key=("in", i))

    def _reveal(self, i: int) -> None:
        info = self._bubbles.get(i)
        if not info or i >= len(self.msgs):
            return
        s = self.scene
        text = self.msgs[i]["text"]
        body, copy = info["body"], info["copy"]
        key = ("reveal", i)
        self._reveal_keys.append(key)

        def done() -> None:
            s.itemconfigure(body, text=text)
            self.msgs[i]["_pending"] = False
            if copy:
                s.itemconfigure(copy, state="normal", fill=T.BG)
                self.anim.tween(0.4, lambda p: s.itemconfigure(copy, fill=T.mix(T.BG, T.SUBTLE, p)),
                                key=("cpin", i))

        self.anim.tween(max(0.6, min(5.0, len(text) / 230)),
                        lambda p: s.itemconfigure(body, text=text[: int(len(text) * p)]),
                        ease=linear, key=key, on_done=done)

    def _add_typing(self) -> None:
        self.msgs.append({"role": "assistant", "typing": True, "text": ""})
        self._append_item(self.msgs[-1])
        self.scroll_target = self._max_scroll()

    def _remove_typing(self) -> None:
        idx = [n for n, m in enumerate(self.msgs) if m.get("typing")]
        if not idx:
            return
        self.scene.delete("typing")
        self.anim.clear("msgs")
        self.msgs.pop(idx[0])
        gap = T.px(self.GAP)
        self._next_y -= self._typing_h + gap
        self.content_h = (self._next_y - gap + T.px(self.EDGE)) if self.msgs else 0
        # reativa o ticker das animações restantes (apenas o indicador usa o grupo "msgs")

    def _copy(self, item, text: str) -> None:
        try:
            self.root.clipboard_clear()
            self.root.clipboard_append(text)
        except Exception:
            return
        self.scene.itemconfigure(item, text="Copiado", fill=T.OK)
        self.root.after(1400, lambda: self._restore_copy(item))

    def _restore_copy(self, item) -> None:
        try:
            self.scene.itemconfigure(item, text="Copiar", fill=T.SUBTLE)
        except tk.TclError:
            pass

    # ------------------------------------------------------------------
    # Rolagem suave
    # ------------------------------------------------------------------
    def _apply_scroll(self, value: float) -> None:
        value = max(0.0, min(self._max_scroll(), value))
        dy = value - self.scroll
        if abs(dy) < 0.01:
            return
        self.scroll = value
        self.scene.move("msgs", 0, -dy)
        self._thumb_vis = 1.0
        self._update_thumb()

    def _scroll_tick(self, _t: float, dt: float) -> None:
        diff = self.scroll_target - self.scroll
        if abs(diff) < 0.4:
            if diff:
                self._apply_scroll(self.scroll_target)
            return
        self._apply_scroll(self.scroll + diff * min(1.0, dt * 13))

    def _on_wheel(self, e) -> None:
        oy = self.scene.winfo_rooty()
        if not (oy + self.view[1] <= e.y_root <= oy + self.view[3]):
            return
        num = getattr(e, "num", 0)
        if num == 4:
            steps = 1
        elif num == 5:
            steps = -1
        else:
            d = getattr(e, "delta", 0)
            steps = d / 120 if abs(d) >= 120 else (1 if d > 0 else -1)
        self.scroll_target = max(0.0, min(self._max_scroll(), self.scroll_target - steps * T.px(90)))
        self._thumb_vis = 1.0

    def _update_thumb(self) -> None:
        vh = self.view[3] - self.view[1]
        if self.content_h <= vh + 1 or self._thumb_vis <= 0:
            if self._thumb_shown != "hidden":
                self.scene.itemconfigure(self.thumb, state="hidden")
                self._thumb_shown = "hidden"
            return
        track = vh - 2 * T.px(20)
        length = max(T.px(40), track * vh / self.content_h)
        frac = self.scroll / max(1.0, self.content_h - vh)
        top = self.view[1] + T.px(20) + (track - length) * frac
        x = self.W - T.px(8)
        self.scene.coords(self.thumb, x, top, x, top + length)
        self.scene.itemconfigure(self.thumb, state="normal",
                                 fill=T.mix(T.BG, "#3B4558", min(1.0, self._thumb_vis)))
        self._thumb_shown = "normal"

    def _thumb_tick(self, _t: float, dt: float) -> None:
        if self._thumb_vis > 0:
            self._thumb_vis = max(0.0, self._thumb_vis - dt * 0.9)
            self._update_thumb()

    # ------------------------------------------------------------------
    # Interações
    # ------------------------------------------------------------------
    def _use_suggestion(self, prompt: str) -> None:
        self.input.delete("1.0", "end")
        self.input.insert("1.0", prompt)
        self.input.focus_set()
        self._after_input_changed()

    def _on_enter(self, event):
        if event.state & 0x1:
            return None
        self._send()
        return "break"

    def _clear_chat(self) -> None:
        if self._generating:
            self._flash_hint("Aguarde a resposta terminar para iniciar outra conversa.")
            return
        for key in self._reveal_keys:
            self.anim.cancel(key)
        self.msgs = []
        self._bubbles = {}
        self.scene.delete("msgs")
        self.anim.clear("msgs")
        self._welcome_visible = True
        self._animate_welcome = True
        self.scroll = self.scroll_target = 0.0
        self.content_h = 0
        self._build_messages()
        self.input.focus_set()

    # ------------------------------------------------------------------
    # Modelo, fila e envio
    # ------------------------------------------------------------------
    def _load_model_async(self) -> None:
        if self._loading:
            return
        self._loading = True
        self._set_status("loading")
        self._update_send_state()

        def worker() -> None:
            try:
                self._results.put(("model_loaded", load_components()))
            except Exception as exc:
                self._results.put(("model_error", str(exc)))

        threading.Thread(target=worker, name="inspector-model-loader", daemon=True).start()

    def _poll(self) -> None:
        if self._closed:
            return
        while True:
            try:
                event, payload = self._results.get_nowait()
            except queue.Empty:
                break
            if event == "model_loaded":
                self.tokenizer, self.model = payload
                self._loading = False
                self._set_status("ready")
                self._update_send_state()
                self._reset_hint()
            elif event == "model_error":
                self._loading = False
                self._set_status("error", "Falha ao carregar")
                self._update_send_state()
                self._add_message("assistant", "Não foi possível carregar o modelo.\n\n" + payload,
                                  error=True)
            elif event == "response":
                self._on_response(*payload)
        self.root.after(90, self._poll)

    def _send(self) -> None:
        text = self._get_input()
        if not text:
            return
        if self._loading or self.model is None or self.tokenizer is None:
            self._flash_hint("O modelo ainda está carregando — aguarde um instante.")
            return
        if self._generating:
            self._flash_hint("Aguarde a resposta atual terminar para enviar outra.")
            return

        self._generating = True
        self.input.delete("1.0", "end")
        self._after_input_changed()
        self._set_status("busy")
        self._update_send_state()
        self._start_chase()
        self._field_glow(self.input.focus_get() is self.input)
        model, tokenizer = self.model, self.tokenizer

        def proceed() -> None:
            self._add_message("user", text)
            self._add_typing()

            def worker() -> None:
                try:
                    self._results.put(("response", (generate_response(model, tokenizer, text), False)))
                except Exception as exc:
                    self._results.put(("response", (f"Ocorreu um erro durante a inferência:\n{exc}", True)))

            threading.Thread(target=worker, name="inspector-inference", daemon=True).start()

        self._dismiss_welcome(proceed)

    def _on_response(self, response: str, failed: bool) -> None:
        self._remove_typing()
        self._generating = False
        self._stop_chase()
        self._add_message("assistant", response, error=failed, animate=not failed)
        self._set_status("error" if failed else "ready", "Falha na inferência" if failed else None)
        self._update_send_state()
        self._field_glow(self.input.focus_get() is self.input)
        self.input.focus_set()
