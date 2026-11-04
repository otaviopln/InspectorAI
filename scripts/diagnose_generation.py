
"""Diagnóstico de reprodutibilidade da geração do InspectorAI."""

from __future__ import annotations

import sys
from hashlib import sha256
from pathlib import Path

# Identifica e registra a raiz do projeto.
ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Imports realizados somente após registrar a raiz.
from inspector.engine import (
    MODEL_PATH,
    TOKENIZER_PATH,
    generate_result,
    load_components,
)


def file_hash(path: Path) -> str:
    """Calcula SHA-256 de um arquivo."""
    if not path.is_file():
        raise FileNotFoundError(f"Arquivo não encontrado: {path}")

    digest = sha256()

    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def main() -> None:
    print("=" * 60)
    print("INSPECTORAI - DIAGNÓSTICO DE REPRODUTIBILIDADE")
    print("=" * 60)

    print(f"\nProject root: {ROOT}")

    print("\nARTIFACT INTEGRITY")
    print("Model SHA-256:", file_hash(MODEL_PATH))
    print("Tokenizer SHA-256:", file_hash(TOKENIZER_PATH))

    print("\nLoading model and tokenizer...")
    tokenizer, model = load_components()

    prompt = "Create a system."
    results: list[str] = []

    print(f"\nINPUT: {prompt}")

    for iteration in range(10):
        result = generate_result(model, tokenizer, prompt)
        results.append(result.text)

        print(f"\nEXECUTION {iteration + 1}")
        print("Status:", result.status)
        print("Finished:", result.finished)
        print("Stop reason:", result.stop_reason)
        print("-" * 50)
        print(result.text)

    unique_results = len(set(results))

    print("\n" + "=" * 60)
    print("FINAL DIAGNOSTIC")
    print("=" * 60)
    print("Total executions:", len(results))
    print("Unique responses:", unique_results)
    print("Deterministic:", unique_results == 1)

    if unique_results == 1:
        print("\nSUCCESS: All executions generated identical responses.")
    else:
        print("\nWARNING: Different responses were generated.")
        print("Further investigation is required.")


if __name__ == "__main__":
    main()
