"""InspectorAI — modern desktop interface.

Modern CustomTkinter chat UI with:
- Dark workspace, sidebar and responsive chat area.
- Animated loading/status indicators and typewriter response reveal.
- Background model loading/inference with a thread-safe queue.
- Single-inference guard to protect the local model.

Run from the project root: python inspector_gui.py
"""

from __future__ import annotations

from __future__ import annotations

import queue
import threading
from typing import Optional

import customtkinter as ctk

from inspector.engine import generate_response, load_components


# ---------------------------------------------------------------------------
# Theme
# ---------------------------------------------------------------------------
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("dark-blue")

BG = "#0B1020"
SIDEBAR = "#0E1528"
PANEL = "#111B30"
PANEL_ALT = "#16223A"
BORDER = "#24334F"
TEXT = "#F1F5F9"
MUTED = "#8FA0BA"
SUBTLE = "#64748B"
ACCENT = "#7C9CFF"
ACCENT_HOVER = "#91ABFF"
ACCENT_DARK = "#283D79"
GREEN = "#39D6A3"
RED = "#FF7185"
USER_BUBBLE = "#23365A"
ASSISTANT_BUBBLE = "#141F34"

FONT = "Segoe UI"
MONO = "Cascadia Code"


class InspectorGUI(ctk.CTk):
    """Modern chat interface for the local InspectorAI model."""

    def __init__(self) -> None:
        super().__init__()

        self.title("InspectorAI | Prompt Workspace")
        self.geometry("1280x850")
        self.minsize(940, 680)
        self.configure(fg_color=BG)

        self.tokenizer = None
        self.model = None

        self._loading = False
        self._generating = False
        self._closed = False
        self._worker_results: queue.Queue = queue.Queue()
        self._typing_after_id: Optional[str] = None
        self._status_after_id: Optional[str] = None
        self._messages_count = 0

        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        self._build_sidebar()
        self._build_workspace()
        self._show_welcome_screen()

        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self.after(90, self._poll_worker_results)
        self._load_model_async()

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------
    def _build_sidebar(self) -> None:
        self.sidebar = ctk.CTkFrame(
            self,
            width=250,
            corner_radius=0,
            fg_color=SIDEBAR,
            border_width=0,
        )
        self.sidebar.grid(row=0, column=0, sticky="nsew")
        self.sidebar.grid_propagate(False)
        self.sidebar.grid_columnconfigure(0, weight=1)
        self.sidebar.grid_rowconfigure(5, weight=1)

        brand = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        brand.grid(row=0, column=0, padx=22, pady=(27, 30), sticky="ew")

        mark = ctk.CTkFrame(
            brand,
            width=42,
            height=42,
            corner_radius=13,
            fg_color=ACCENT_DARK,
            border_width=1,
            border_color="#4966B3",
        )
        mark.pack(side="left")
        mark.pack_propagate(False)
        ctk.CTkLabel(
            mark,
            text="✳",
            font=(FONT, 24, "bold"),
            text_color="#DDE5FF",
        ).place(relx=0.5, rely=0.48, anchor="center")

        brand_text = ctk.CTkFrame(brand, fg_color="transparent")
        brand_text.pack(side="left", padx=(12, 0))
        ctk.CTkLabel(
            brand_text,
            text="INSPECTOR",
            font=(FONT, 15, "bold"),
            text_color=TEXT,
        ).pack(anchor="w")
        ctk.CTkLabel(
            brand_text,
            text="AI PROMPT WORKSPACE",
            font=(FONT, 9, "bold"),
            text_color=SUBTLE,
        ).pack(anchor="w", pady=(1, 0))

        ctk.CTkLabel(
            self.sidebar,
            text="WORKSPACE",
            font=(FONT, 10, "bold"),
            text_color=SUBTLE,
        ).grid(row=1, column=0, padx=24, pady=(0, 10), sticky="w")

        self.new_chat_button = ctk.CTkButton(
            self.sidebar,
            text="＋   Nova conversa",
            height=46,
            corner_radius=12,
            anchor="w",
            font=(FONT, 13, "bold"),
            fg_color=ACCENT_DARK,
            hover_color="#344F94",
            text_color="#F4F6FF",
            command=self._clear_chat,
        )
        self.new_chat_button.grid(
            row=2, column=0, padx=17, pady=(0, 24), sticky="ew"
        )

        ctk.CTkLabel(
            self.sidebar,
            text="SEU ASSISTENTE",
            font=(FONT, 10, "bold"),
            text_color=SUBTLE,
        ).grid(row=3, column=0, padx=24, pady=(0, 10), sticky="w")

        info_card = ctk.CTkFrame(
            self.sidebar,
            fg_color=PANEL,
            corner_radius=14,
            border_width=1,
            border_color=BORDER,
        )
        info_card.grid(row=4, column=0, padx=17, sticky="ew")
        info_card.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(
            info_card,
            text="◈  R.C.F.O. ENGINE",
            font=(FONT, 12, "bold"),
            text_color=TEXT,
        ).grid(row=0, column=0, padx=14, pady=(14, 4), sticky="w")

        ctk.CTkLabel(
            info_card,
            text="Role · Context · Focus · Output",
            font=(FONT, 10),
            text_color=MUTED,
        ).grid(row=1, column=0, padx=14, pady=(0, 13), sticky="w")

        footer = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        footer.grid(row=6, column=0, padx=20, pady=20, sticky="ew")
        footer.grid_columnconfigure(0, weight=1)

        ctk.CTkFrame(
            footer, height=1, fg_color=BORDER
        ).grid(row=0, column=0, sticky="ew", pady=(0, 15))

        self.sidebar_status_dot = ctk.CTkLabel(
            footer,
            text="●",
            font=(FONT, 12),
            text_color="#EABF65",
        )
        self.sidebar_status_dot.grid(row=1, column=0, sticky="w")
        self.sidebar_status_text = ctk.CTkLabel(
            footer,
            text="Inicializando motor",
            font=(FONT, 11),
            text_color=MUTED,
        )
        self.sidebar_status_text.grid(row=1, column=0, padx=(17, 0), sticky="w")

    def _build_workspace(self) -> None:
        self.workspace = ctk.CTkFrame(
            self,
            fg_color=BG,
            corner_radius=0,
        )
        self.workspace.grid(row=0, column=1, sticky="nsew")
        self.workspace.grid_columnconfigure(0, weight=1)
        self.workspace.grid_rowconfigure(1, weight=1)

        # Top bar
        topbar = ctk.CTkFrame(
            self.workspace,
            height=76,
            corner_radius=0,
            fg_color=BG,
            border_width=0,
        )
        topbar.grid(row=0, column=0, sticky="ew", padx=30)
        topbar.grid_propagate(False)
        topbar.grid_columnconfigure(0, weight=1)

        title_box = ctk.CTkFrame(topbar, fg_color="transparent")
        title_box.grid(row=0, column=0, sticky="w", pady=14)
        ctk.CTkLabel(
            title_box,
            text="Prompt Studio",
            font=(FONT, 19, "bold"),
            text_color=TEXT,
        ).pack(anchor="w")
        ctk.CTkLabel(
            title_box,
            text="Transforme solicitações em instruções mais precisas.",
            font=(FONT, 11),
            text_color=MUTED,
        ).pack(anchor="w", pady=(2, 0))

        actions = ctk.CTkFrame(topbar, fg_color="transparent")
        actions.grid(row=0, column=1, sticky="e")

        self.model_badge = ctk.CTkFrame(
            actions,
            fg_color=PANEL,
            corner_radius=20,
            border_width=1,
            border_color=BORDER,
        )
        self.model_badge.pack(side="left", padx=(0, 12))

        self.model_badge_dot = ctk.CTkLabel(
            self.model_badge,
            text="●",
            text_color="#EABF65",
            font=(FONT, 9),
        )
        self.model_badge_dot.pack(side="left", padx=(11, 5), pady=8)
        self.model_badge_text = ctk.CTkLabel(
            self.model_badge,
            text="MODELO LOCAL",
            font=(FONT, 9, "bold"),
            text_color=MUTED,
        )
        self.model_badge_text.pack(side="left", padx=(0, 12), pady=8)

        self.clear_button = ctk.CTkButton(
            actions,
            text="Limpar",
            width=76,
            height=34,
            corner_radius=10,
            fg_color=PANEL,
            hover_color=PANEL_ALT,
            border_width=1,
            border_color=BORDER,
            text_color=MUTED,
            font=(FONT, 11, "bold"),
            command=self._clear_chat,
        )
        self.clear_button.pack(side="left")

        separator = ctk.CTkFrame(
            self.workspace,
            height=1,
            fg_color=BORDER,
        )
        separator.grid(row=0, column=0, sticky="sew")

        # Conversation surface
        self.conversation_area = ctk.CTkFrame(
            self.workspace,
            fg_color="transparent",
        )
        self.conversation_area.grid(
            row=1, column=0, sticky="nsew", padx=30, pady=(18, 10)
        )
        self.conversation_area.grid_columnconfigure(0, weight=1)
        self.conversation_area.grid_rowconfigure(0, weight=1)

        self.messages_frame = ctk.CTkScrollableFrame(
            self.conversation_area,
            fg_color="transparent",
            corner_radius=0,
            scrollbar_button_color="#344563",
            scrollbar_button_hover_color=ACCENT,
        )
        self.messages_frame.grid(row=0, column=0, sticky="nsew")
        self.messages_frame.grid_columnconfigure(0, weight=1)

        # Composer
        composer_wrap = ctk.CTkFrame(
            self.workspace,
            fg_color="transparent",
        )
        composer_wrap.grid(
            row=2, column=0, sticky="ew", padx=30, pady=(4, 22)
        )
        composer_wrap.grid_columnconfigure(0, weight=1)

        self.composer = ctk.CTkFrame(
            composer_wrap,
            fg_color=PANEL,
            corner_radius=18,
            border_width=1,
            border_color=BORDER,
        )
        self.composer.grid(row=0, column=0, sticky="ew")
        self.composer.grid_columnconfigure(0, weight=1)

        self.input_box = ctk.CTkTextbox(
            self.composer,
            height=66,
            corner_radius=13,
            fg_color="transparent",
            border_width=0,
            text_color=TEXT,
            font=(FONT, 13),
            wrap="word",
        )
        self.input_box.grid(
            row=0, column=0, padx=(15, 8), pady=9, sticky="ew"
        )
        self.input_box.bind("<Return>", self._on_enter)
        self.input_box.bind("<FocusIn>", self._clear_input_placeholder)
        self.input_box.bind("<FocusOut>", self._restore_input_placeholder)
        self._placeholder_text = "Descreva sua ideia ou solicitação..."
        self._placeholder_active = False
        self._show_input_placeholder()

        self.send_button = ctk.CTkButton(
            self.composer,
            text="↑",
            width=43,
            height=43,
            corner_radius=13,
            fg_color=ACCENT,
            hover_color=ACCENT_HOVER,
            text_color="#081124",
            font=(FONT, 23, "bold"),
            command=self._send,
        )
        self.send_button.grid(
            row=0, column=1, padx=(0, 12), pady=10, sticky="s"
        )

        helper = ctk.CTkLabel(
            composer_wrap,
            text="Enter para enviar  ·  Shift + Enter para nova linha",
            font=(FONT, 10),
            text_color=SUBTLE,
        )
        helper.grid(row=1, column=0, pady=(8, 0), sticky="e")

    # ------------------------------------------------------------------
    # Welcome screen and messages
    # ------------------------------------------------------------------
    def _show_welcome_screen(self) -> None:
        self.welcome = ctk.CTkFrame(
            self.messages_frame,
            fg_color="transparent",
        )
        self.welcome.grid(
            row=0, column=0, sticky="nsew", padx=18, pady=(55, 25)
        )
        self.welcome.grid_columnconfigure(0, weight=1)

        icon_card = ctk.CTkFrame(
            self.welcome,
            width=62,
            height=62,
            corner_radius=20,
            fg_color=ACCENT_DARK,
            border_width=1,
            border_color="#4966B3",
        )
        icon_card.grid(row=0, column=0, pady=(0, 22))
        icon_card.grid_propagate(False)
        ctk.CTkLabel(
            icon_card,
            text="✳",
            font=(FONT, 32, "bold"),
            text_color="#E2E8FF",
        ).place(relx=0.5, rely=0.48, anchor="center")

        ctk.CTkLabel(
            self.welcome,
            text="Boas ideias merecem\ninstruções extraordinárias.",
            font=(FONT, 30, "bold"),
            text_color=TEXT,
            justify="center",
        ).grid(row=1, column=0, pady=(0, 12))

        ctk.CTkLabel(
            self.welcome,
            text=(
                "Conte o que você deseja realizar. O Inspector analisa sua solicitação "
                "e ajuda a estruturá-la usando Role, Context, Focus e Output."
            ),
            font=(FONT, 12),
            text_color=MUTED,
            justify="center",
            wraplength=620,
        ).grid(row=2, column=0, pady=(0, 28))

        suggestions = ctk.CTkFrame(
            self.welcome,
            fg_color="transparent",
        )
        suggestions.grid(row=3, column=0)

        cards = [
            (
                "✦  Estruturar uma ideia",
                "Quero criar uma plataforma para organizar estudos e tarefas.",
            ),
            (
                "⌘  Melhorar um projeto",
                "Analise meu projeto de software e identifique melhorias prioritárias.",
            ),
            (
                "◎  Criar um planejamento",
                "Preciso de um plano de estudos de programação para três meses.",
            ),
        ]

        for index, (label, prompt) in enumerate(cards):
            button = ctk.CTkButton(
                suggestions,
                text=label,
                width=190,
                height=58,
                corner_radius=13,
                fg_color=PANEL,
                hover_color=PANEL_ALT,
                border_width=1,
                border_color=BORDER,
                text_color="#DCE5F5",
                font=(FONT, 11, "bold"),
                command=lambda value=prompt: self._use_suggestion(value),
            )
            button.grid(
                row=0,
                column=index,
                padx=6,
                pady=5,
            )

        self._welcome_visible = True

    def _remove_welcome_screen(self) -> None:
        if getattr(self, "_welcome_visible", False):
            self.welcome.destroy()
            self._welcome_visible = False

    def _add_message(
        self,
        sender: str,
        text: str,
        *,
        animate: bool = False,
        is_error: bool = False,
    ) -> ctk.CTkLabel:
        """Adds a chat bubble and optionally reveals its text progressively."""
        self._remove_welcome_screen()
        self._messages_count += 1

        row = ctk.CTkFrame(
            self.messages_frame,
            fg_color="transparent",
        )
        row.grid(
            row=self._messages_count,
            column=0,
            sticky="ew",
            padx=7,
            pady=(5, 13),
        )
        row.grid_columnconfigure(0, weight=1)

        is_user = sender.lower() in {"você", "voce", "user"}
        bubble_color = USER_BUBBLE if is_user else ASSISTANT_BUBBLE
        border_color = "#344D79" if is_user else BORDER

        bubble = ctk.CTkFrame(
            row,
            fg_color=bubble_color,
            corner_radius=17,
            border_width=1,
            border_color=border_color,
        )

        if is_user:
            bubble.grid(row=0, column=0, sticky="e", padx=(90, 0))
        else:
            bubble.grid(row=0, column=0, sticky="w", padx=(0, 90))

        bubble.grid_columnconfigure(0, weight=1)

        name_color = ACCENT_HOVER if is_user else GREEN
        if is_error:
            name_color = RED

        header = ctk.CTkFrame(
            bubble,
            fg_color="transparent",
        )
        header.grid(
            row=0, column=0, padx=16, pady=(11, 3), sticky="ew"
        )

        avatar = "●" if is_user else "✳"
        ctk.CTkLabel(
            header,
            text=avatar,
            font=(FONT, 10, "bold"),
            text_color=name_color,
        ).pack(side="left", padx=(0, 7))

        ctk.CTkLabel(
            header,
            text="Você" if is_user else "INSPECTOR",
            font=(FONT, 9, "bold"),
            text_color=name_color,
        ).pack(side="left")

        body = ctk.CTkLabel(
            bubble,
            text="" if animate else text,
            font=(FONT, 13),
            text_color=RED if is_error else TEXT,
            justify="left",
            anchor="w",
            wraplength=max(320, min(690, self.winfo_width() - 440)),
        )
        body.grid(
            row=1,
            column=0,
            padx=16,
            pady=(3, 14),
            sticky="w",
        )

        self._scroll_to_bottom()

        if animate and text:
            self._reveal_text(body, text)

        return body

    def _reveal_text(
        self,
        label: ctk.CTkLabel,
        full_text: str,
        position: int = 0,
    ) -> None:
        """Typewriter-style reveal, scheduled on Tk's main thread."""
        if self._closed:
            return

        if position >= len(full_text):
            label.configure(text=full_text)
            self._scroll_to_bottom()
            return

        next_position = min(position + 12, len(full_text))
        label.configure(text=full_text[:next_position])
        self._scroll_to_bottom()

        self.after(
            9,
            lambda: self._reveal_text(
                label,
                full_text,
                next_position,
            ),
        )

    def _scroll_to_bottom(self) -> None:
        if self._closed:
            return

        def _scroll() -> None:
            if self._closed:
                return
            try:
                self.messages_frame._parent_canvas.yview_moveto(1.0)
            except (AttributeError, RuntimeError):
                pass

        self.after(45, _scroll)

    # ------------------------------------------------------------------
    # Status and animation
    # ------------------------------------------------------------------
    def _set_status(
        self,
        message: str,
        color: str,
        *,
        badge_message: Optional[str] = None,
    ) -> None:
        self.sidebar_status_text.configure(text=message)
        self.sidebar_status_dot.configure(text_color=color)
        self.model_badge_dot.configure(text_color=color)

        if badge_message:
            self.model_badge_text.configure(text=badge_message)

    def _animate_startup_status(self, frame: int = 0) -> None:
        if self._closed or not self._loading:
            return

        dots = "." * (frame % 4)
        self._set_status(
            f"Carregando modelo{dots}",
            "#EABF65",
            badge_message="CARREGANDO",
        )

        self._status_after_id = self.after(
            380,
            lambda: self._animate_startup_status(frame + 1),
        )

    def _show_typing_indicator(self) -> None:
        self._typing_label = self._add_message(
            "Inspector",
            "",
            animate=False,
        )
        self._animate_typing_dots(0)

    def _animate_typing_dots(self, frame: int) -> None:
        if self._closed or not self._generating:
            return

        dots = "." * (frame % 4)
        try:
            self._typing_label.configure(
                text=f"Analisando sua solicitação{dots}"
            )
        except (AttributeError, RuntimeError):
            return

        self._typing_after_id = self.after(
            330,
            lambda: self._animate_typing_dots(frame + 1),
        )

    def _remove_typing_indicator(self) -> None:
        if self._typing_after_id is not None:
            try:
                self.after_cancel(self._typing_after_id)
            except Exception:
                pass
            self._typing_after_id = None

        if hasattr(self, "_typing_label"):
            try:
                parent = self._typing_label.master
                # The label is nested in the bubble; remove its whole message row.
                bubble = parent
                row = bubble.master
                row.destroy()
                self._messages_count -= 1
            except (AttributeError, RuntimeError):
                pass

    # ------------------------------------------------------------------
    # Model loading and background result queue
    # ------------------------------------------------------------------
    def _load_model_async(self) -> None:
        if self._loading:
            return

        self._loading = True
        self.send_button.configure(state="disabled")
        self.input_box.configure(state="disabled")
        self._animate_startup_status()

        def _worker() -> None:
            try:
                tokenizer, model = load_components()
                self._worker_results.put(
                    ("model_loaded", (tokenizer, model))
                )
            except Exception as exc:
                self._worker_results.put(
                    ("model_error", str(exc))
                )

        threading.Thread(
            target=_worker,
            name="inspector-model-loader",
            daemon=True,
        ).start()

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

        if self._status_after_id is not None:
            try:
                self.after_cancel(self._status_after_id)
            except Exception:
                pass
            self._status_after_id = None

        self._set_status(
            "Motor pronto",
            GREEN,
            badge_message="ONLINE",
        )
        self.send_button.configure(state="normal")
        self.input_box.configure(state="normal")
        self._show_input_placeholder()

    def _on_model_error(self, message: str) -> None:
        self._loading = False
        self._set_status(
            "Falha ao carregar",
            RED,
            badge_message="ERRO",
        )
        self.send_button.configure(state="disabled")
        self.input_box.configure(state="disabled")
        self._add_message(
            "Inspector",
            "Não foi possível carregar o modelo.\n\n" + message,
            is_error=True,
        )

    # ------------------------------------------------------------------
    # Textbox placeholder compatibility (CTkTextbox has no native placeholder)
    # ------------------------------------------------------------------
    def _show_input_placeholder(self) -> None:
        """Draws a lightweight placeholder using the textbox content itself."""
        if not hasattr(self, "input_box") or self._placeholder_active:
            return

        try:
            current = self.input_box.get("1.0", "end-1c")
            if current.strip():
                return

            self.input_box.configure(text_color=SUBTLE)
            self.input_box.insert("1.0", self._placeholder_text)
            self._placeholder_active = True
        except Exception:
            # The widget may be disabled or being destroyed during shutdown.
            pass

    def _clear_input_placeholder(self, event=None) -> None:
        """Removes the simulated placeholder when the user focuses the box."""
        if not getattr(self, "_placeholder_active", False):
            return

        try:
            self.input_box.delete("1.0", "end")
            self.input_box.configure(text_color=TEXT)
            self._placeholder_active = False
        except Exception:
            pass

    def _restore_input_placeholder(self, event=None) -> None:
        """Restores the placeholder when the empty textbox loses focus."""
        if not hasattr(self, "input_box"):
            return

        try:
            if self.input_box.cget("state") == "disabled":
                return
            self._show_input_placeholder()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # User interactions
    # ------------------------------------------------------------------
    def _use_suggestion(self, prompt: str) -> None:
        if self._loading or self._generating or self.model is None:
            return

        if self._placeholder_active:
            self._clear_input_placeholder()
        self.input_box.delete("1.0", "end")
        self.input_box.configure(text_color=TEXT)
        self.input_box.insert("1.0", prompt)
        self.input_box.focus_set()

    def _on_enter(self, event) -> str:
        # Shift+Enter inserts a line break. Enter sends the message.
        if event.state & 0x1:
            return "break"

        self._send()
        return "break"

    def _send(self) -> None:
        if (
            self._loading
            or self._generating
            or self.model is None
            or self.tokenizer is None
        ):
            return

        if self._placeholder_active:
            return

        user_text = self.input_box.get(
            "1.0", "end-1c"
        ).strip()

        if not user_text:
            return

        self._generating = True
        self.input_box.delete("1.0", "end")
        self.input_box.configure(text_color=TEXT)
        self._placeholder_active = False
        self._add_message("Você", user_text)

        self._set_status(
            "Analisando solicitação",
            ACCENT,
            badge_message="PROCESSANDO",
        )

        self.send_button.configure(state="disabled")
        self.input_box.configure(state="disabled")
        self._show_typing_indicator()

        # Capture stable references before starting the worker.
        model = self.model
        tokenizer = self.tokenizer

        def _worker() -> None:
            try:
                response = generate_response(
                    model,
                    tokenizer,
                    user_text,
                )
                self._worker_results.put(
                    ("response", (response, False))
                )
            except Exception as exc:
                self._worker_results.put(
                    (
                        "response",
                        (
                            f"Ocorreu um erro durante a inferência:\n{exc}",
                            True,
                        ),
                    )
                )

        threading.Thread(
            target=_worker,
            name="inspector-inference",
            daemon=True,
        ).start()

    def _on_response_ready(
        self,
        response: str,
        failed: bool = False,
    ) -> None:
        self._remove_typing_indicator()
        self._generating = False

        self._add_message(
            "Inspector",
            response,
            animate=not failed,
            is_error=failed,
        )

        self._set_status(
            "Falha na inferência" if failed else "Motor pronto",
            RED if failed else GREEN,
            badge_message="ERRO" if failed else "ONLINE",
        )

        if self.model is not None:
            self.send_button.configure(state="normal")
            self.input_box.configure(state="normal")
            self._show_input_placeholder()

    def _clear_chat(self) -> None:
        if self._generating:
            return

        for child in self.messages_frame.winfo_children():
            child.destroy()

        self._messages_count = 0
        self._welcome_visible = False
        self._show_welcome_screen()
        self._scroll_to_bottom()

    def _on_close(self) -> None:
        self._closed = True

        for after_id in (
            self._typing_after_id,
            self._status_after_id,
        ):
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
