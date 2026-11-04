"""Aplicação principal: cena em Canvas com vidro, luz e animações fluidas."""

from __future__ import annotations

import math
import queue
import threading
import tkinter as tk
from datetime import datetime
from typing import Optional

from PIL import ImageTk

from inspector.engine import generate_response, load_components

from . import render
from . import theme as T
from .anim import Animator, linear
from .widgets import GlassButton, GlassPanel, Hover, perimeter_point, photo

NARROW = 1040  # largura lógica abaixo da qual a barra lateral recolhe

STATUS = {
    "loading": (T.AMBER, "Carregando…", "Preparando o motor local", "CARREGANDO"),
    "ready": (T.LIME, "Motor pronto", "Modelo local online", "ONLINE"),
    "busy": (T.CYAN, "Analisando…", "Gerando a resposta", "PROCESSANDO"),
    "error": (T.RED, "Indisponível", "Verifique o modelo", "ERRO"),
}

SUGGESTIONS = (
    ("Estruturar uma ideia", "Quero criar uma plataforma para organizar estudos e tarefas."),
    ("Melhorar um projeto", "Analise meu projeto de software e identifique melhorias prioritárias."),
    ("Criar um planejamento", "Preciso de um plano de estudos de programação para três meses."),
)

WELCOME_TEXT = (
    "Conte o que você deseja realizar. O Inspector analisa sua solicitação e ajuda "
    "a estruturá-la usando Role, Context, Focus e Output."
)

RCFO = (
    ("R", "Role", "Quem responde"),
    ("C", "Context", "Cenário da tarefa"),
    ("F", "Focus", "O que priorizar"),
    ("O", "Output", "Formato da entrega"),
)


class InspectorApp:
    def __init__(self) -> None:
        self.root = tk.Tk()
        T.pick_font(self.root)
        T.set_scale(self.root.winfo_fpixels("1i") / 96.0)
        self.root.title("InspectorAI | Prompt Workspace")
        self.root.configure(bg=T.BG_TOP)

        sw, sh = self.root.winfo_screenwidth(), self.root.winfo_screenheight()
        w, h = min(T.px(1280), sw - T.px(80)), min(T.px(860), sh - T.px(120))
        self.root.geometry(f"{w}x{h}+{(sw - w) // 2}+{max(0, (sh - h) // 2 - 20)}")
        self.root.minsize(T.px(820), T.px(600))

        self.scene = tk.Canvas(self.root, bg=T.BG_TOP, highlightthickness=0, bd=0)
        self.scene.pack(fill="both", expand=True)
        self.anim = Animator(self.root)

        # modelo / fila
        self.tokenizer = None
        self.model = None
        self._loading = False
        self._generating = False
        self._results: queue.Queue = queue.Queue()
        self._closed = False

        # cena
        self._refs: list = []
        self.bg = None
        self._bg_size = (0, 0)
        self._layout_key = None
        self._layout_job = None
        self._was_narrow: Optional[bool] = None
        self._sidebar_pref: Optional[bool] = None
        self.collapsed = False

        # mensagens
        self.msgs: list[dict] = []
        self.scroll = 0.0
        self.scroll_target = 0.0
        self.content_h = 0
        self.view = (0, 0, 0, 0)
        self._conv_rect = (0, 0, 0, 0)
        self._bubbles: dict = {}
        self._reveal_keys: list = []

        # status / luzes
        self._status_state = "loading"
        self._status_title: Optional[str] = None
        self._status_color = T.AMBER
        self._pulsing = True
        self._light_cache: dict = {}
        self._lights: list = []
        self._light_k = -1
        self._status_items: dict = {}

        # widgets recriados a cada layout
        self.btn_new = self.btn_clear = self.btn_send = self.btn_menu = None
        self.composer: Optional[GlassPanel] = None
        self._comp_rect = None
        self._chase_items: list = []
        self._hint_item = None
        self._hint_job = None
        self.thumb = None

        self._make_input()
        self.scene.bind("<Configure>", self._on_configure)
        self.root.bind("<Control-n>", lambda _e: self._clear_chat())
        self.root.bind("<Control-l>", lambda _e: self.input.focus_set())
        self.root.bind_all("<MouseWheel>", self._on_wheel)
        self.root.bind_all("<Button-4>", self._on_wheel)
        self.root.bind_all("<Button-5>", self._on_wheel)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        self.anim.add(self._scroll_tick, "core")
        self.root.after(90, self._poll)
        self.root.after(60, self._layout)
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
            font=T.font(14), fg=T.TEXT, bg=T.BG_BOT, insertbackground=T.CYAN,
            insertwidth=max(2, T.px(2)), selectbackground="#2B5FA8",
            selectforeground=T.TEXT, padx=0, pady=0, undo=True,
        )
        self.placeholder = tk.Label(
            self.input, text="Descreva sua ideia ou solicitação…", fg=T.SUBTLE,
            bg=T.BG_BOT, font=T.font(14), bd=0, anchor="w",
        )
        self.placeholder.place(x=0, y=0)
        self.placeholder.bind("<Button-1>", lambda _e: self.input.focus_set())
        self.input.bind("<Return>", self._on_enter)
        self.input.bind("<KP_Enter>", self._on_enter)
        self.input.bind("<FocusIn>", lambda _e: self._composer_focus(True))
        self.input.bind("<FocusOut>", lambda _e: self._composer_focus(False))
        self.input.bind("<<Modified>>", self._on_modified)

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------
    def _on_configure(self, _e=None) -> None:
        if self._layout_job is not None:
            self.root.after_cancel(self._layout_job)
        self._layout_job = self.root.after(110, self._layout)

    def _layout(self, force: bool = False) -> None:
        self._layout_job = None
        if self._closed:
            return
        s = self.scene
        W, H = s.winfo_width(), s.winfo_height()
        if W < T.px(300) or H < T.px(300):
            return
        narrow = W < T.px(NARROW)
        if narrow != self._was_narrow:
            self._was_narrow = narrow
            self._sidebar_pref = None
        self.collapsed = self._sidebar_pref if self._sidebar_pref is not None else narrow
        key = (W, H, self.collapsed)
        if key == self._layout_key and not force:
            return
        self._layout_key = key

        at_bottom = self.scroll_target >= self._max_scroll() - 1
        self.anim.clear("scene")
        self.anim.clear("msgs")
        self.anim.clear("chase")
        self._chase_items = []
        self._lights = []
        self._status_items = {}
        self._light_k = -1
        s.delete("all")
        self._refs = []
        self._bubbles = {}

        if (W, H) != self._bg_size:
            self.bg = render.make_background(W, H)
            self._bg_size = (W, H)
        s.create_image(0, 0, anchor="nw", image=photo(self, self.bg))

        M = T.px(14)
        SB = T.px(72) if self.collapsed else T.px(262)
        wx0, wx1 = M + SB + M, W - M
        hh = T.px(64)
        cy0 = M + hh + T.px(8)
        comp_h = T.px(104)
        comp_y = H - M - T.px(30) - comp_h
        conv_bottom = comp_y - T.px(14)
        conv_h = max(T.px(120), conv_bottom - cy0)
        conv_w = wx1 - wx0

        self._build_conversation(wx0, cy0, conv_w, conv_h)
        self._build_sidebar(M, M, SB, H - 2 * M)
        self._build_header(wx0, M, wx1, hh)
        self._build_composer(wx0, comp_y, conv_w, comp_h)

        self._build_messages()
        self._apply_status()
        self.anim.add(self._tick_lights, "scene")
        if self._generating:
            self._start_chase()
        self._update_send_state()
        self._after_input_changed()

        self.scroll = min(self.scroll, self._max_scroll())
        if at_bottom:
            self.scroll_target = self._max_scroll()
        self.scroll_target = min(self.scroll_target, self._max_scroll())

    # -- conversa -----------------------------------------------------------
    def _build_conversation(self, x, y, w, h) -> None:
        s = self.scene
        radius = T.px(28)
        conv = GlassPanel(self, x, y, w, h, radius, tags=("chrome",), rim=False)
        inset = T.px(8)
        vx0, vy0, vx1, vy1 = x, y + inset, x + w, y + h - inset
        self.view = (vx0, vy0, vx1, vy1)
        self._conv_rect = (x, y, w, h)

        top = self.bg.crop((x, 0, x + w, vy0)).convert("RGBA")
        top.alpha_composite(conv.fill_img.crop((0, 0, w, vy0 - y)), (0, y))
        bot = self.bg.crop((x, vy1, x + w, self.bg.size[1])).convert("RGBA")
        bot.alpha_composite(conv.fill_img.crop((0, vy1 - y, w, h)), (0, 0))
        s.create_image(x, 0, anchor="nw", image=photo(self, top), tags=("cover",))
        s.create_image(x, vy1, anchor="nw", image=photo(self, bot), tags=("cover",))

        fh = T.px(26)
        inner = conv.fill_img.crop((0, inset, w, h - inset))
        s.create_image(x, vy0, anchor="nw",
                       image=photo(self, render.make_fade_strip(inner, fh, True)), tags=("cover",))
        s.create_image(x, vy1 - fh, anchor="nw",
                       image=photo(self, render.make_fade_strip(inner, fh, False)), tags=("cover",))

        conv.add_rim()
        conv.add_flare("top", length=T.px(300), speed=0.055, phase=0.0)
        conv.add_flare("bottom", length=T.px(240), speed=0.045, phase=2.4, color=T.BLUE)
        conv.add_flare("left", length=T.px(160), speed=0.05, phase=4.0, color=T.BLUE)

        self.thumb = s.create_line(0, 0, 0, 0, fill="#4F86C8", width=max(3, T.px(4)),
                                   capstyle="round", state="hidden")

    # -- barra lateral ---------------------------------------------------------
    def _build_sidebar(self, x, y, w, h) -> None:
        s = self.scene
        panel = GlassPanel(self, x, y, w, h, T.px(26), tags=("chrome",), rim_color=T.BLUE)
        panel.add_flare("right", length=T.px(260), speed=0.06, phase=1.0)
        panel.add_flare("left", length=T.px(160), speed=0.045, phase=3.3, color=T.BLUE)

        def tile(tx, ty):
            size = T.px(44)
            GlassPanel(self, tx, ty, size, size, T.px(14), mode="glass", tint=(36, 100, 214),
                       alpha=(215, 165), rim_color=T.CYAN)
            s.create_text(tx + size / 2, ty + size / 2, text="I", fill="#F4FAFF",
                          font=T.font(20, True))

        if self.collapsed:
            tile(x + (w - T.px(44)) // 2, y + T.px(22))
            self.btn_new = GlassButton(self, x + T.px(10), y + T.px(88), w - T.px(20), T.px(46),
                                       icon="plus", command=self._clear_chat)
            lx, ly = x + w / 2, y + h - T.px(40)
            self._light(lx, ly)
            return

        tile(x + T.px(20), y + T.px(22))
        s.create_text(x + T.px(76), y + T.px(24), text="INSPECTOR", anchor="nw",
                      fill=T.TEXT, font=T.font(16, True))
        s.create_text(x + T.px(76), y + T.px(46), text="AI PROMPT WORKSPACE", anchor="nw",
                      fill=T.CYAN, font=T.font(9, True))

        self.btn_new = GlassButton(self, x + T.px(18), y + T.px(92), w - T.px(36), T.px(46),
                                   text="Nova conversa", icon="plus", command=self._clear_chat)

        cx, cy, cw, ch = x + T.px(18), y + T.px(160), w - T.px(36), T.px(232)
        GlassPanel(self, cx, cy, cw, ch, T.px(22), mode="glass", tint=(14, 44, 98),
                   alpha=(120, 80), rim_color=T.BLUE)
        s.create_text(cx + T.px(18), cy + T.px(16), text="MOTOR R.C.F.O.", anchor="nw",
                      fill=T.CYAN, font=T.font(10, True))
        for i, (letter, word, desc) in enumerate(RCFO):
            ry = cy + T.px(46) + i * T.px(44)
            chip = T.px(30)
            GlassPanel(self, cx + T.px(16), ry, chip, chip, T.px(10), mode="glass",
                       tint=(30, 84, 180), alpha=(190, 140), rim_color=T.CYAN)
            s.create_text(cx + T.px(16) + chip / 2, ry + chip / 2, text=letter,
                          fill="#F4FAFF", font=T.font(12, True))
            s.create_text(cx + T.px(58), ry - T.px(1), text=word, anchor="nw",
                          fill=T.TEXT, font=T.font(12, True))
            s.create_text(cx + T.px(58), ry + T.px(16), text=desc, anchor="nw",
                          fill=T.MUTED, font=T.font(10))

        sx, sy, sww, shh = x + T.px(18), y + h - T.px(76), w - T.px(36), T.px(58)
        GlassPanel(self, sx, sy, sww, shh, T.px(20), mode="glass", tint=(14, 44, 98),
                   alpha=(130, 90), rim_color=T.CYAN)
        self._light(sx + T.px(26), sy + shh / 2)
        self._status_items["title"] = s.create_text(
            sx + T.px(48), sy + T.px(12), text="", anchor="nw", fill=T.TEXT, font=T.font(12, True))
        self._status_items["sub"] = s.create_text(
            sx + T.px(48), sy + T.px(31), text="", anchor="nw", fill=T.MUTED, font=T.font(10))

    # -- cabeçalho ---------------------------------------------------------------
    def _build_header(self, x0, y0, x1, hh) -> None:
        s = self.scene
        self.btn_menu = GlassButton(self, x0, y0 + T.px(11), T.px(42), T.px(42), icon="menu",
                                    style="glass", command=self._toggle_sidebar)
        tx = x0 + T.px(58)
        s.create_text(tx, y0 + T.px(8), text="Prompt Studio", anchor="nw",
                      fill=T.TEXT, font=T.font(20, True))
        wide = (x1 - x0) >= T.px(700)
        if wide:
            s.create_text(tx, y0 + T.px(38), anchor="nw", fill=T.MUTED, font=T.font(11),
                          text="Transforme solicitações em instruções mais precisas.")

        bw, bh = T.px(96), T.px(40)
        self.btn_clear = GlassButton(self, x1 - bw, y0 + T.px(12), bw, bh, text="Limpar",
                                     style="glass", command=self._clear_chat, font_size=12)
        if (x1 - x0) >= T.px(560):
            pw = T.px(168)
            px0 = x1 - bw - T.px(12) - pw
            GlassPanel(self, px0, y0 + T.px(12), pw, bh, bh // 2, mode="glass", tint=(20, 56, 120),
                       alpha=(140, 96), rim_color=T.CYAN)
            self._light(px0 + T.px(24), y0 + T.px(12) + bh / 2)
            self._status_items["badge"] = s.create_text(
                px0 + T.px(44), y0 + T.px(12) + bh / 2, text="", anchor="w",
                fill=T.MUTED, font=T.font(10, True))

    # -- compositor ----------------------------------------------------------------
    def _build_composer(self, x, y, w, h) -> None:
        s = self.scene
        self.composer = GlassPanel(self, x, y, w, h, T.px(28), flat=True, tint_a=190)
        self.composer.add_flare("top", length=T.px(320), speed=0.05, phase=0.8)
        self.composer.add_flare("bottom", length=T.px(200), speed=0.04, phase=3.0, color=T.BLUE)
        self._comp_rect = (x, y, w, h)

        bg = self.composer.average_color()
        self.input.configure(bg=bg)
        self.placeholder.configure(bg=bg)

        send = T.px(52)
        pad = T.px(24)
        tw = w - pad - send - T.px(36)
        th = h - T.px(36)
        s.create_window(x + pad, y + T.px(18), window=self.input, anchor="nw", width=tw, height=th)
        self.btn_send = GlassButton(self, x + w - T.px(18) - send, y + (h - send) // 2, send, send,
                                    icon="send", command=self._send)
        self._hint_item = s.create_text(x + w - T.px(8), y + h + T.px(18), anchor="e",
                                        text="", fill=T.SUBTLE, font=T.font(10))
        self._reset_hint()
        if self.input.focus_get() is self.input:
            self.composer.set_level(4)

    # ------------------------------------------------------------------
    # Luzes de status
    # ------------------------------------------------------------------
    def _light_photos(self, color: str, key: str = "light"):
        cache_key = (key, color)
        arr = self._light_cache.get(cache_key)
        if arr is None:
            radius = T.px(13 if key == "light" else 10)
            arr = [ImageTk.PhotoImage(render.make_glow(radius, color, k))
                   for k in (0.18, 0.34, 0.52, 0.72, 0.9, 1.0)]
            self._light_cache[cache_key] = arr
        return arr

    def _light(self, x, y) -> None:
        s = self.scene
        glow = s.create_image(x, y, image=self._light_photos(self._status_color)[4])
        r = T.px(4)
        core = s.create_oval(x - r, y - r, x + r, y + r, fill=self._status_color, outline="")
        self._lights.append((glow, core))

    def _tick_lights(self, t: float, _dt: float) -> None:
        k = int(2.5 + 2.5 * math.sin(t * 4.4)) if self._pulsing else 4
        k = max(0, min(5, k))
        if k == self._light_k:
            return
        self._light_k = k
        photos = self._light_photos(self._status_color)
        for glow, _core in self._lights:
            self.scene.itemconfigure(glow, image=photos[k])

    def _set_status(self, state: str, title: Optional[str] = None) -> None:
        self._status_state, self._status_title = state, title
        self._apply_status()

    def _apply_status(self) -> None:
        color, default_title, sub, badge = STATUS[self._status_state]
        self._status_color = color
        self._pulsing = self._status_state in {"loading", "busy"}
        self._light_k = -1
        s = self.scene
        for _glow, core in self._lights:
            s.itemconfigure(core, fill=color)
        items = self._status_items
        if "title" in items:
            s.itemconfigure(items["title"], text=self._status_title or default_title)
            s.itemconfigure(items["sub"], text=sub)
        if "badge" in items:
            s.itemconfigure(items["badge"], text=badge)

    # ------------------------------------------------------------------
    # Mensagens
    # ------------------------------------------------------------------
    def _Y(self, y_rel: float) -> float:
        return self.view[1] + y_rel - self.scroll

    def _max_scroll(self) -> float:
        return max(0.0, self.content_h - (self.view[3] - self.view[1]))

    def _rebuild_messages(self) -> None:
        for key in self._reveal_keys:
            self.anim.cancel(key)
        self._reveal_keys = []
        self.scene.delete("msgs")
        self.anim.clear("msgs")
        for m in self.msgs:
            m.pop("_hidden", None)
        self._bubbles = {}
        self._build_messages()
        mx = self._max_scroll()
        if self.scroll > mx:
            self._apply_scroll(mx)
        self.scroll_target = min(self.scroll_target, mx)

    def _build_messages(self) -> None:
        vx0, vy0, vx1, vy1 = self.view
        pad, gap = T.px(18), T.px(14)
        edge = T.px(34)  # maior que a faixa de esmaecimento
        avail = vx1 - vx0 - 2 * pad
        if not self.msgs:
            self.content_h = self._build_welcome(vx0, vx1, vy1 - vy0, pad, avail, edge)
        else:
            maxw = min(T.px(800), int(avail * 0.86))
            y = edge
            for i, m in enumerate(self.msgs):
                if m.get("typing"):
                    h = self._build_typing(vx0, pad, y)
                else:
                    h = self._build_bubble(i, m, vx0, vx1, pad, maxw, y)
                y += h + gap
            self.content_h = y - gap + edge
        self.scene.tag_lower("msgs", "cover")
        self._update_thumb()

    def _build_bubble(self, i, m, vx0, vx1, pad, maxw, y) -> int:
        s = self.scene
        is_user = m["role"] == "user"
        kind = "error" if m.get("error") else ("user" if is_user else "ai")
        tags = ("msgs", f"m{i}")
        padx, head = T.px(18), T.px(24)
        text = "" if m.get("_hidden") else m["text"]
        body = s.create_text(0, 0, text=m["text"], anchor="nw", width=maxw - 2 * padx,
                             fill="#FFC4CD" if m.get("error") else T.TEXT,
                             font=T.font(13), tags=tags)
        x1, y1, x2, y2 = s.bbox(body)
        tw, th = x2 - x1, y2 - y1
        bw = max(tw, T.px(160)) + 2 * padx
        bh = T.px(10) + head + th + T.px(14)
        x0 = vx1 - pad - bw if is_user else vx0 + pad
        y0 = self._Y(y)

        img = render.make_bubble(int(bw), int(bh), kind)
        ph = photo(self, img)
        item = s.create_image(x0, y0, anchor="nw", image=ph, tags=tags)
        s.tag_lower(item, body)
        s.coords(body, x0 + padx, y0 + T.px(10) + head)
        s.itemconfigure(body, text=text)

        name_color = T.RED if m.get("error") else (T.ICE if is_user else T.CYAN)
        s.create_text(x0 + padx, y0 + T.px(10), anchor="nw", fill=name_color,
                      text="VOCÊ" if is_user else "INSPECTOR", font=T.font(10, True), tags=tags)
        tstr = m["time"]
        tfont = T.measure_font(9)
        s.create_text(x0 + bw - padx, y0 + T.px(11), anchor="ne", text=tstr,
                      fill="#A9C8F0" if is_user else T.SUBTLE, font=T.font(9), tags=tags)
        if not is_user and not m.get("error"):
            cx = x0 + bw - padx - tfont.measure(tstr) - T.px(14)
            tag = f"copy{i}"
            copy = s.create_text(cx, y0 + T.px(10), anchor="ne", text="Copiar",
                                 fill=T.MUTED, font=T.font(10, True), tags=tags + (tag,))
            Hover(self, tag, lambda on, c=copy: s.itemconfigure(c, fill=T.CYAN if on else T.MUTED))
            s.tag_bind(tag, "<Button-1>", lambda _e, c=copy, t=m["text"]: self._copy(c, t))

        self._bubbles[i] = {"item": item, "photo": ph, "pil": img, "body": body,
                            "w": int(bw), "h": int(bh)}
        return int(bh)

    def _build_typing(self, vx0, pad, y) -> int:
        s = self.scene
        tags = ("msgs", "typing")
        bw, bh = T.px(280), T.px(54)
        x0, y0 = vx0 + pad, self._Y(y)
        s.create_image(x0, y0, anchor="nw", image=photo(self, render.make_bubble(bw, bh, "ai")),
                       tags=tags)
        photos = self._light_photos(T.CYAN, "dot")
        dots = [s.create_image(x0 + T.px(30) + j * T.px(22), y0 + bh / 2, image=photos[0], tags=tags)
                for j in range(3)]
        s.create_text(x0 + T.px(100), y0 + bh / 2, anchor="w", text="Analisando sua solicitação",
                      fill=T.MUTED, font=T.font(11), tags=tags)

        def tick(t: float, _dt: float) -> None:
            for j, d in enumerate(dots):
                k = max(0, min(5, int(2.5 + 2.5 * math.sin(t * 5.0 - j * 1.0))))
                s.itemconfigure(d, image=photos[k])

        self.anim.add(tick, "msgs")
        return bh

    def _build_welcome(self, vx0, vx1, vh, pad, avail, edge) -> int:
        s = self.scene
        tags = ("msgs", "welcome")
        cx = (vx0 + vx1) // 2
        tsize = 30 if avail >= T.px(600) else 22
        probe = s.create_text(0, 0, text="Ag", font=T.font(tsize, True), anchor="nw")
        pb = s.bbox(probe)
        ls = max(pb[3] - pb[1], T.px(tsize * 1.3))
        s.delete(probe)
        cols = 3 if avail >= T.px(780) else (2 if avail >= T.px(520) else 1)
        gapc = T.px(12)
        card_w = (avail - gapc * (cols - 1)) // cols
        card_h = T.px(108) if cols > 1 else T.px(92)
        rows = (len(SUGGESTIONS) + cols - 1) // cols
        cards_h = rows * card_h + (rows - 1) * gapc

        sub_w = min(T.px(560), avail)
        sub = s.create_text(cx, 0, text=WELCOME_TEXT, font=T.font(13), fill=T.MUTED, anchor="n",
                            width=sub_w, justify="center", tags=tags)
        sb = s.bbox(sub)
        sub_h = sb[3] - sb[1]

        beam_gap = T.px(52)
        total = ls * 2 + beam_gap + sub_h + T.px(28) + cards_h
        top = max(edge, (vh - total) // 2)

        glow = photo(self, render.make_glow(T.px(220), T.BLUE, 0.42))
        s.create_image(cx, self._Y(top + ls), image=glow, tags=tags)
        s.create_text(cx, self._Y(top), anchor="n", text="Boas ideias merecem",
                      fill=T.TEXT, font=T.font(tsize, True), tags=tags)
        s.create_text(cx, self._Y(top + ls), anchor="n", text="instruções extraordinárias.",
                      fill=T.CYAN, font=T.font(tsize, True), tags=tags)

        beams = [photo(self, render.make_flare(T.px(420), T.px(30), T.CYAN, k))
                 for k in (0.3, 0.55, 0.8, 1.0)]
        beam = s.create_image(cx, self._Y(top + 2 * ls + T.px(26)), image=beams[2], tags=tags)
        state = {"k": 2}

        def breathe(t: float, _dt: float) -> None:
            k = max(0, min(3, int(1.5 + 1.5 * math.sin(t * 1.4))))
            if k != state["k"]:
                state["k"] = k
                s.itemconfigure(beam, image=beams[k])

        self.anim.add(breathe, "msgs")
        s.coords(sub, cx, self._Y(top + 2 * ls + beam_gap))

        y_cards = top + 2 * ls + beam_gap + sub_h + T.px(28)
        for i, (title, prompt) in enumerate(SUGGESTIONS):
            r, c = divmod(i, cols)
            x = vx0 + pad + c * (card_w + gapc)
            yy = self._Y(y_cards + r * (card_h + gapc))
            tag = f"card{i}"
            ctags = tags + (tag,)
            panel = GlassPanel(self, x, yy, card_w, card_h, T.px(20), tags=ctags, mode="glass",
                               tint=(24, 64, 140), alpha=(120, 78), rim_color=T.CYAN)
            s.create_text(x + T.px(20), yy + T.px(16), anchor="nw", text=title,
                          fill=T.TEXT, font=T.font(13, True), tags=ctags)
            s.create_text(x + T.px(20), yy + T.px(42), anchor="nw", text=prompt,
                          fill=T.MUTED, font=T.font(11), width=card_w - T.px(40), tags=ctags)
            Hover(self, tag, lambda on, p=panel, k=tag: p.glow_to(4 if on else 0, key=("card", k)))
            s.tag_bind(tag, "<Button-1>", lambda _e, p=prompt: self._use_suggestion(p))
        return top + total + edge

    # -- animações de mensagem ----------------------------------------------------
    def _play_glint(self, i: int) -> None:
        b = self._bubbles.get(i)
        if not b:
            return
        n = 7
        frames = []
        for k in range(n):
            frame = b["pil"].copy()
            frame.alpha_composite(render.make_glint(b["w"], b["h"], T.px(20), k / (n - 1), 0, 0.5))
            frames.append(photo(self, frame))
        s = self.scene
        self.anim.tween(
            0.7, lambda p: s.itemconfigure(b["item"], image=frames[min(n - 1, int(p * n))]),
            ease=linear, key=("bglint", i),
            on_done=lambda: s.itemconfigure(b["item"], image=b["photo"]),
        )

    def _reveal(self, i: int) -> None:
        b = self._bubbles.get(i)
        if not b:
            return
        text = self.msgs[i]["text"]
        self.scene.itemconfigure(b["body"], text="")
        key = ("reveal", i)
        self._reveal_keys.append(key)
        duration = max(0.5, min(4.0, len(text) / 260))
        self.anim.tween(
            duration,
            lambda p: self.scene.itemconfigure(b["body"], text=text[: int(len(text) * p)]),
            ease=linear, key=key,
            on_done=lambda: self.scene.itemconfigure(b["body"], text=text),
        )

    def _copy(self, item, text: str) -> None:
        try:
            self.root.clipboard_clear()
            self.root.clipboard_append(text)
        except Exception:
            return
        self.scene.itemconfigure(item, text="Copiado ✓", fill=T.LIME)
        self.root.after(1500, lambda: self._restore_copy(item))

    def _restore_copy(self, item) -> None:
        try:
            self.scene.itemconfigure(item, text="Copiar", fill=T.MUTED)
        except tk.TclError:
            pass

    def _add_message(self, role: str, text: str, *, error: bool = False,
                     animate: bool = False) -> None:
        self.msgs.append({"role": role, "text": text, "error": error,
                          "time": datetime.now().strftime("%H:%M")})
        self._rebuild_messages()
        self._scroll_to_bottom()
        idx = len(self.msgs) - 1
        self._play_glint(idx)
        if animate and not error:
            self._reveal(idx)

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
        self._update_thumb()

    def _scroll_tick(self, _t: float, dt: float) -> None:
        diff = self.scroll_target - self.scroll
        if abs(diff) < 0.4:
            if diff:
                self._apply_scroll(self.scroll_target)
            return
        self._apply_scroll(self.scroll + diff * min(1.0, dt * 14))

    def _scroll_to_bottom(self) -> None:
        self.scroll_target = self._max_scroll()

    def _on_wheel(self, e) -> None:
        if not self._over_conv(e.x_root, e.y_root):
            return
        num = getattr(e, "num", 0)
        if num == 4:
            steps = 1
        elif num == 5:
            steps = -1
        else:
            d = getattr(e, "delta", 0)
            steps = d / 120 if abs(d) >= 120 else (1 if d > 0 else -1)
        self.scroll_target = max(0.0, min(self._max_scroll(), self.scroll_target - steps * T.px(80)))

    def _over_conv(self, rx: int, ry: int) -> bool:
        x, y, w, h = self._conv_rect
        ox, oy = self.scene.winfo_rootx(), self.scene.winfo_rooty()
        return ox + x <= rx <= ox + x + w and oy + y <= ry <= oy + y + h

    def _update_thumb(self) -> None:
        if self.thumb is None:
            return
        vx0, vy0, vx1, vy1 = self.view
        vh = vy1 - vy0
        if self.content_h <= vh + 1:
            self.scene.itemconfigure(self.thumb, state="hidden")
            return
        track = vh - 2 * T.px(18)
        length = max(T.px(36), track * vh / self.content_h)
        frac = self.scroll / max(1.0, self.content_h - vh)
        top = vy0 + T.px(18) + (track - length) * frac
        x = vx1 - T.px(9)
        self.scene.coords(self.thumb, x, top, x, top + length)
        self.scene.itemconfigure(self.thumb, state="normal")

    # ------------------------------------------------------------------
    # Compositor
    # ------------------------------------------------------------------
    def _get_input(self) -> str:
        return self.input.get("1.0", "end-1c").strip()

    def _on_modified(self, _e=None) -> None:
        try:
            if self.input.edit_modified():
                self.input.edit_modified(False)
                self._after_input_changed()
        except tk.TclError:
            pass

    def _after_input_changed(self) -> None:
        if self.input.get("1.0", "end-1c"):
            self.placeholder.place_forget()
        else:
            self.placeholder.place(x=0, y=0)
        self._update_send_state()

    def _composer_focus(self, focused: bool) -> None:
        if self.composer is not None:
            self.composer.glow_to(4 if focused else (3 if self._generating else 0),
                                  duration=0.3, key="composer")

    def _update_send_state(self) -> None:
        ready = self.model is not None and not self._loading and not self._generating
        if self.btn_send is not None:
            self.btn_send.set_enabled(ready and bool(self._get_input()))
        for btn in (self.btn_new, self.btn_clear):
            if btn is not None:
                btn.set_enabled(not self._generating)

    def _reset_hint(self) -> None:
        self._hint_job = None
        text = "Enter para enviar  ·  Shift + Enter para nova linha"
        if self._loading:
            text = "O motor está carregando — você já pode ir escrevendo."
        if self._hint_item is not None:
            self.scene.itemconfigure(self._hint_item, text=text, fill=T.SUBTLE)

    def _flash_hint(self, text: str, color: str = T.AMBER, ms: int = 3000) -> None:
        if self._hint_item is not None:
            self.scene.itemconfigure(self._hint_item, text=text, fill=color)
        if self._hint_job is not None:
            self.root.after_cancel(self._hint_job)
        self._hint_job = self.root.after(ms, self._reset_hint)

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

    # -- luz que percorre o compositor durante o processamento -------------------------
    def _start_chase(self) -> None:
        if not self._comp_rect or self._chase_items:
            return
        s = self.scene
        x, y, w, h = self._comp_rect
        r = T.px(28)
        photos = [self._chase_photo(k) for k in (1.0, 0.62, 0.38, 0.22, 0.1)]
        items = [s.create_image(0, 0, image=p, tags=("chase",)) for p in photos]
        self._chase_items = items

        def tick(t: float, _dt: float) -> None:
            base = (t * 0.30) % 1.0
            for j, it in enumerate(items):
                ax, ay = perimeter_point(x, y, w, h, r, base - j * 0.011)
                s.coords(it, ax, ay)

        self.anim.add(tick, "chase")
        if self.composer is not None:
            self.composer.glow_to(3, duration=0.4, key="composer")

    def _chase_photo(self, k: float):
        key = ("chase", k)
        ph = self._light_cache.get(key)
        if ph is None:
            ph = ImageTk.PhotoImage(render.make_glow(T.px(17), T.CYAN, k))
            self._light_cache[key] = ph
        return ph

    def _stop_chase(self) -> None:
        self.anim.clear("chase")
        self.scene.delete("chase")
        self._chase_items = []
        if self.composer is not None:
            focused = self.root.focus_get() is self.input
            self.composer.glow_to(4 if focused else 0, duration=0.5, key="composer")

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
                tokenizer, model = load_components()
                self._results.put(("model_loaded", (tokenizer, model)))
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
            self._flash_hint("O motor ainda está carregando — aguarde um instante.")
            return
        if self._generating:
            self._flash_hint("Aguarde a resposta atual terminar para enviar outra.")
            return

        self._generating = True
        self.input.delete("1.0", "end")
        self._after_input_changed()
        self._add_message("user", text)
        self.msgs.append({"role": "assistant", "typing": True, "text": ""})
        self._rebuild_messages()
        self._scroll_to_bottom()

        self._set_status("busy")
        self._update_send_state()
        self._start_chase()

        model, tokenizer = self.model, self.tokenizer

        def worker() -> None:
            try:
                self._results.put(("response", (generate_response(model, tokenizer, text), False)))
            except Exception as exc:
                self._results.put(("response", (f"Ocorreu um erro durante a inferência:\n{exc}", True)))

        threading.Thread(target=worker, name="inspector-inference", daemon=True).start()

    def _on_response(self, response: str, failed: bool) -> None:
        self.msgs = [m for m in self.msgs if not m.get("typing")]
        self._generating = False
        self._stop_chase()
        self._add_message("assistant", response, error=failed, animate=not failed)
        self._set_status("error" if failed else "ready", "Falha na inferência" if failed else None)
        self._update_send_state()
        self.input.focus_set()

    # ------------------------------------------------------------------
    # Ações
    # ------------------------------------------------------------------
    def _clear_chat(self) -> None:
        if self._generating:
            self._flash_hint("Aguarde a resposta terminar para limpar a conversa.")
            return
        self.msgs = []
        self.scroll = self.scroll_target = 0.0
        self._rebuild_messages()
        self.input.focus_set()

    def _toggle_sidebar(self) -> None:
        self._sidebar_pref = not self.collapsed
        self._layout(force=True)
