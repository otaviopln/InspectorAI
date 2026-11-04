"""InspectorAI — ponto de entrada da interface em vidro.

Dependências: Pillow (python -m pip install pillow). Não requer customtkinter.
Execute a partir da raiz do projeto: python inspector_gui.py
"""

from __future__ import annotations


def main() -> None:
    try:  # nitidez em telas de alta densidade (Windows)
        import ctypes

        ctypes.windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        pass

    from inspector_ui.app import InspectorApp

    InspectorApp().run()


if __name__ == "__main__":
    main()
