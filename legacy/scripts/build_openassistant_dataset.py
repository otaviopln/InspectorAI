"""Dataset builder: OpenAssistant (oasst1) -> Inspector AI format.

Loads the OpenAssistant oasst1 dataset from HuggingFace (`OpenAssistant/oasst1`)
and converts it into Inspector AI training pairs:

  * Clear user messages  ->  R.C.F.O formatted targets
  * Vague / under-specified user messages  ->  strategic clarification questions

OpenAssistant contains multi-turn conversations with quality annotations. This
script extracts user -> assistant turns and classifies them using heuristics.
If HuggingFace or internet access is unavailable, it falls back to a small
built-in OpenAssistant-style sample.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import List, Tuple

ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DATA_DIR = ROOT / "inspector" / "data" / "processed"
RAW_DATA_DIR = ROOT / "inspector" / "data" / "raw"
OUTPUT_PATH = PROCESSED_DATA_DIR / "openassistant_dataset.json"

# Heuristic signals that suggest a user message is vague / under-specified.
VAGUE_SIGNALS = (
    r"^(?:please\s+)?help me[.!?]*$",
    r"^(?:please\s+)?help me with (?:this|it|that)[.!?]*$",
    r"^(?:can you )?help me with (?:this|it|that)[.!?]*$",
    r"^(?:i need|i want) (?:some )?help(?: with (?:this|it|that))?[.!?]*$",
    r"^(?:please )?(?:fix|improve|change|rewrite|optimize|update) "
    r"(?:it|this|that)[.!?]*$",
    r"^(?:please )?(?:write|create|make|do) "
    r"(?:something|anything)(?: about .+)?[.!?]*$",
    r"^(?:i am|i'm) not sure(?: what to do)?[.!?]*$",
    r"^what should i do(?: about .+)?[.!?]*$",
)

# Verbs that usually indicate a clear, actionable task.
CLEAR_VERBS = (
"write", "create", "design", "build", "analyze",
    "analyse", "review", "implement", "refactor",
    "migrate", "debug", "audit", "draft", "document",
    "prepare", "develop", "optimize", "optimise",
    "generate", "produce", "compose", "translate",
    "summarize", "summarise", "explain", "calculate",
    "compare", "classify", "organize", "organise",
    "calculate", "plan", "evaluate", "convert",
)


def _is_vague(text: str) -> bool:
    """Heuristic: short or vague-sounding user message."""
    lowered = text.lower().strip()
    if len(lowered.split()) < 5:
        return True
    return any(signal in lowered for signal in VAGUE_SIGNALS)


def _is_clear(text: str) -> bool:
    """Heuristic: contains an action verb and enough detail."""
    lowered = text.lower().strip()
    if len(lowered.split()) < 8:
        return False
    return any(verb in lowered for verb in CLEAR_VERBS) and not _is_vague(text)


def _format_rcfo(user_message: str, assistant_response: str) -> str:
    """Best-effort conversion of an OpenAssistant turn into R.C.F.O fields."""
    role = "Unspecified role"
    context = user_message
    focus = user_message
    output = assistant_response.strip() if assistant_response else "Clear, actionable response."

    # Try to detect an explicit role like "As a ..."
    m = re.search(r"as\s+a\s+(.+?)(?:,|can|could|please|to|\.)", user_message, re.IGNORECASE)
    if m:
        role = m.group(1).strip().capitalize()

    return (
        f"Role: {role}\n"
        f"Context: {context}\n"
        f"Focus: {focus}\n"
        f"Output: {output}"
    )


def _clarification_questions(user_message: str) -> str:
    """Generate R.C.F.O-aligned clarification questions for a vague prompt."""
    return (
        "Role: What role or persona should take on this task?\n"
        f"Context: What background or situation should I know about '{user_message}'?\n"
        "Focus: What is the specific goal or task you want accomplished?\n"
        "Output: What format or deliverable do you expect?"
    )


def _load_openassistant_from_huggingface() -> List[Tuple[str, str]]:
    """
    Carrega o OpenAssistant e associa cada resposta do assistente
    diretamente à mensagem do usuário indicada por parent_id.

    Retorna:
        Lista de tuplas (mensagem_do_usuario, resposta_do_assistente).
    """
    try:
        from datasets import load_dataset

        dataset = load_dataset(
            "OpenAssistant/oasst1",
            split="train",
        )

        # Primeiro passo: indexa todas as mensagens pelo message_id.
        messages_by_id = {}

        for row in dataset:
            message_id = row.get("message_id")

            if message_id:
                messages_by_id[str(message_id)] = row

        pairs: List[Tuple[str, str]] = []
        seen = set()

        # Segundo passo: encontra respostas cujo pai é uma mensagem do usuário.
        for row in dataset:
            role = str(row.get("role", "")).lower()
            assistant_text = str(row.get("text", "") or "").strip()
            parent_id = row.get("parent_id")

            # Considera somente respostas do assistente.
            if role != "assistant":
                continue

            if not assistant_text or not parent_id:
                continue

            # Ignora mensagens excluídas.
            if row.get("deleted", False):
                continue

            parent = messages_by_id.get(str(parent_id))

            if parent is None:
                continue

            parent_role = str(parent.get("role", "")).lower()
            user_text = str(parent.get("text", "") or "").strip()

            # O pai precisa ser uma mensagem do usuário.
            if parent_role not in {"user", "prompter"}:
                continue

            if not user_text:
                continue

            if parent.get("deleted", False):
                continue

            pair = (user_text, assistant_text)

            # Evita duplicações.
            if pair not in seen:
                pairs.append(pair)
                seen.add(pair)

        print(
            f"OpenAssistant: {len(pairs)} pares válidos encontrados."
        )

        return pairs

    except Exception as exc:
        print(
            "Não foi possível carregar o OpenAssistant: "
            f"{exc}"
        )
        return []


def _fallback_sample() -> List[Tuple[str, str]]:
    """Small OpenAssistant-style sample for offline testing."""
    return [
        ("Write a professional email to a client requesting a project deadline extension.",
         "Subject: Request for Deadline Extension...\nDear [Client Name], I hope this email finds you well..."),
        ("Create a Python function that calculates compound interest.",
         "def compound_interest(principal, rate, time):\n    return principal * (1 + rate) ** time"),
        ("fix it", ""),
        ("write something about business", ""),
        ("As a data analyst, analyze Q3 sales figures from the regional CSV file and identify underperforming regions.",
         "Q3 sales analysis: Region A underperformed by 12%..."),
        ("help me with this", ""),
        ("Analyze the website traffic data and produce a summary report.",
         "Traffic increased by 15% QoQ. Main sources: organic search (40%), direct (30%)..."),
        ("make this better", ""),
        ("Draft a product requirements document for a mobile checkout feature.",
         "PRD: Mobile Checkout...\n1. Overview...\n2. User Stories..."),
        ("can you debug my code", ""),
        ("I need a marketing strategy for our new SaaS product targeting small businesses.",
         "Marketing Strategy for SaaS...\nTarget: Small businesses with <50 employees..."),
        ("what should i do about my career", ""),
        ("Create a 30-60-90 day onboarding plan for new software engineers.",
         "30-60-90 Day Onboarding Plan...\nWeek 1-2: Setup and orientation..."),
        ("make my website faster", ""),
        ("Review the vendor service agreement for liability clauses.",
         "Liability clauses review:\n1. Indemnification...\n2. Limitation of liability..."),
    ]


def build_openassistant_pairs(use_huggingface: bool = True) -> dict:
    """Constrói os pares de treinamento do Inspector AI."""

    clear_pairs: List[dict] = []
    vague_pairs: List[dict] = []

    if use_huggingface:
        oa = _load_openassistant_from_huggingface()
        source = "HuggingFace (OpenAssistant/oasst1)"
    else:
        oa = []
        source = "offline sample"

    if not oa:
        oa = _fallback_sample()
        source = "offline sample (fallback)"

    for user_msg, assistant_resp in oa:
        user_msg = user_msg.strip()
        assistant_resp = assistant_resp.strip()

        if not user_msg:
            continue

        if _is_clear(user_msg):
            clear_pairs.append(
                {
                    "input": user_msg,
                    "target": _format_rcfo(
                        user_msg,
                        assistant_resp,
                    ),
                }
            )

        elif _is_vague(user_msg):
            vague_pairs.append(
                {
                    "input": user_msg,
                    "target": _clarification_questions(user_msg),
                }
            )

    total = len(clear_pairs) + len(vague_pairs)

    return {
        "version": "1.0.1-openassistant",

        "source": source,

        "description": (
            "OpenAssistant-derived dataset for Inspector AI. "
            "Clear user messages mapped to R.C.F.O format; "
            "vague or under-specified messages mapped to "
            "clarification questions. Messages are paired "
            "through their parent_id relationships."
        ),

        "splits": {
            "clear_to_rcfo": clear_pairs,
            "vague_to_questions": vague_pairs,
        },

        "counts": {
            "clear_to_rcfo": len(clear_pairs),
            "vague_to_questions": len(vague_pairs),
            "total": total,
        },
    }


def save_json(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def main() -> None:
    dataset = build_openassistant_pairs(use_huggingface=True)

    save_json(dataset, OUTPUT_PATH)
    save_json(dataset, RAW_DATA_DIR / "openassistant_dataset.json")

    print("Inspector AI — OpenAssistant dataset conversion complete.")
    print(f"Source: {dataset['source']}")
    print(f"Clear -> R.C.F.O pairs:   {dataset['counts']['clear_to_rcfo']}")
    print(f"Vague -> Questions pairs: {dataset['counts']['vague_to_questions']}")
    print(f"Total pairs:              {dataset['counts']['total']}")
    print(f"Written to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
