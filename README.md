# Inspector AI

Inspector is a custom AI assistant built from absolute zero with Python and PyTorch.

## Purpose

Inspector analyzes user prompts and behaves in one of two ways:

1. **Dubious input** — If the prompt is ambiguous, incomplete, or dubious, Inspector asks strategic clarifying questions instead of guessing.
2. **Clear input** — If the prompt is clear enough, Inspector reformats it into the **R.C.F.O** framework:
   - **R**ole — who should act
   - **C**ontext — background and situation
   - **F**ocus — the specific task or goal
   - **O**utput — the expected deliverable or format

## Project Structure

```
InspectorAI/
├── History.md                  # Project changelog (all development sessions)
├── README.md                   # This file
├── requirements.txt            # Python dependencies
├── .gitignore                  # Git ignore rules
├── LICENSE                     # License (MIT)
├── inspector.py                # Interactive terminal application
├── inspector_gui.py            # Desktop GUI application (CustomTkinter)
├── inspector/                  # Core package
│   ├── __init__.py
│   ├── config.py               # Configuration and paths
│   ├── core/
│   │   ├── __init__.py
│   │   ├── analyzer.py         # Prompt analyzer (dubious vs clear) — legacy
│   │   ├── formatter.py        # R.C.F.O formatter — legacy
│   │   └── questioner.py       # Strategic question generator — legacy
│   ├── data/
│   │   ├── __init__.py
│   │   ├── raw/                # Raw prompt datasets
│   │   │   ├── mock_dataset.json
│   │   │   ├── openassistant_dataset.json
│   │   │   └── expanded_dataset.json
│   │   └── processed/          # Processed prompt datasets
│   │       ├── mock_dataset.json
│   │       ├── openassistant_dataset.json
│   │       └── expanded_dataset.json
│   ├── models/                 # Trained models
│   │   ├── __init__.py
│   │   ├── inspector_model.pth  # Trained PyTorch weights
│   │   └── tokenizer_vocab.json # BPE tokenizer vocabulary
│   ├── tests/                  # Unit tests
│   │   ├── __init__.py
│   │   ├── conftest.py
│   │   ├── test_dataset_builder.py
│   │   ├── test_model.py
│   │   ├── test_tokenizer.py
│   │   └── test_training.py
│   └── utils/                  # Utilities (empty, reserved)
│       ├── __init__.py
│       └── .gitkeep
├── scripts/                    # Entry-point and utility scripts
│   ├── __init__.py
│   ├── train.py                # Main training loop (CPU/GPU/DirectML)
│   ├── train_colab.py          # Training script optimized for Google Colab
│   ├── model.py                # Transformer Seq2Seq model (from scratch)
│   ├── tokenizer.py            # BPE tokenizer (HuggingFace tokenizers/Rust)
│   ├── dataset_builder.py      # Mock dataset builder (600 pairs)
│   ├── build_openassistant_dataset.py  # OpenAssistant dataset converter
│   ├── build_alpaca_dataset.py   # Alpaca dataset converter
│   ├── expand_dataset.py       # Synthetic dataset generator (20,000+ pairs)
│   ├── run.py                  # Interactive CLI entry point
│   └── colab_train.ipynb       # Google Colab notebook for GPU training
└── tests/                      # Root-level tests (mirror of inspector/tests)
    ├── conftest.py
    ├── test_dataset_builder.py
    ├── test_model.py
    ├── test_tokenizer.py
    └── test_training.py
```

## Installation

```bash
cd InspectorAI
pip install -r requirements.txt
```

## Usage

### Interactive Terminal

```bash
python inspector.py
```

### Desktop GUI (Recommended)

```bash
python inspector_gui.py
```

Requires `customtkinter` (included in requirements.txt).

### Training

**Local (CPU):**
```bash
python scripts/train.py
```

**Google Colab (GPU — Recommended for production quality):**

1. Open [colab.research.google.com](https://colab.research.google.com)
2. Upload: `scripts/train_colab.py`, `scripts/model.py`, `scripts/tokenizer.py`, `inspector/data/processed/expanded_dataset.json`
3. Runtime → Change runtime type → GPU
4. Run all cells in `scripts/colab_train.ipynb`
5. Download `inspector_model.pth` and `tokenizer_vocab.json`
6. Copy them to `InspectorAI/inspector/models/`

## Datasets

| Dataset | Pairs | Source | License |
|---------|-------|--------|---------|
| `mock_dataset.json` | 600 | Synthetic (hand-written) | MIT |
| `openassistant_dataset.json` | 9,202 | HuggingFace `OpenAssistant/oasst1` | Apache 2.0 |
| `expanded_dataset.json` | 20,000 | Synthetic (template-based) | MIT |
| **Total** | **~31,202** | | |

## Model Architecture

- **Type:** Encoder-Decoder Transformer (Seq2Seq)
- **Attention:** Multi-Head Self-Attention + Cross-Attention
- **Positional Encoding:** Sinusoidal
- **Decoder:** Autoregressive with Beam Search
- **Training:** Teacher forcing, CrossEntropyLoss, Adam optimizer

## License

MIT License — see `LICENSE` file.

## Development Roadmap

- [x] Project structure and core package
- [x] BPE Tokenizer (from scratch)
- [x] Transformer Seq2Seq model (from scratch)
- [x] Training loop with validation and early stopping
- [x] Dataset generation (mock, OpenAssistant, synthetic 20k pairs)
- [x] Interactive terminal application
- [x] Desktop GUI application
- [ ] Beam search fine-tuning
- [ ] Validation metrics (BLEU/ROUGE)
- [ ] Domain-specific fine-tuning
- [ ] REST API endpoint

# InspectorAI dataset and training pipeline — version 2

## Decisions adopted

- **Partial requests:** preserve the facts already supplied and ask only targeted questions about missing information.
- **Language:** English (`language: "en"`) for this training version.
- **Architecture:** one Encoder–Decoder Transformer. The target explicitly starts with either `Decision: CLARIFY` or `Decision: FORMAT_RCFO`.

## Data layout

```text
inspector/data/raw/curated_examples.jsonl
inspector/data/processed/prepared_dataset.json
inspector/data/processed/dataset_report.json
inspector/models/inspector_model.pth
inspector/models/tokenizer_vocab.json
inspector/models/training_metrics.json
```

Each JSONL line is one independently reviewable record. Required fields:

- `id`: unique record ID.
- `task`: `clarify` or `format_rcfo`.
- `ambiguity`: `vague`, `partial`, or `clear`.
- `language`: `en`.
- `input`: original user request.
- `target`: ideal model response.
- `source`: provenance, such as `human_authored`, `assistant_seed`, or a named public dataset.
- `group_id`: family of semantically related examples (it may contain vague, partial, and clear versions of the same underlying scenario); all variants in a group stay in one split.
- `review_status`: `pending_review`, `approved`, or `rejected`.

The starter file contains 18 seed examples. They are deliberately marked `pending_review`: they are drafts for inspection, not claims of human validation. Review each input/target pair. Change the status to `approved` only after checking that the target is faithful, complete, and semantically consistent. Do not approve all records mechanically.

## Target contract

### Vague

```text
Decision: CLARIFY
Questions:
1. ...
2. ...
```

Ask only questions needed to identify the request's goal, subject, users, constraints, or deliverable. Do not invent a role or project context.

### Partial

```text
Decision: CLARIFY
Known requirements:
- ...
Missing information:
- ...
Questions:
1. ...
```

The `Known requirements` section must preserve facts present in the input. Questions must address only material gaps. Do not ask again for information already supplied.

### Clear

```text
Decision: FORMAT_RCFO
Role: ...
Context: ...
Focus: ...
Output: ...
```

Preserve the user's requirements. Do not add frameworks, platforms, databases, deadlines, or deliverables that were not requested or logically entailed.

## Prepare the dataset

From the repository root:

```powershell
python scripts/prepare_dataset.py
```

The preparation script:

1. validates schema and task/ambiguity compatibility;
2. rejects duplicate IDs and duplicate inputs;
3. excludes every record not explicitly marked `approved`;
4. keeps each `group_id` intact;
5. splits each task/ambiguity stratum into 80% train, 10% validation, 10% test;
6. writes `prepared_dataset.json` and `dataset_report.json`.

At least three distinct groups are required in each of the three ambiguity strata. The splitter uses a deterministic, repeated greedy assignment to keep whole groups intact while approximating the requested class proportions. With a small dataset, ratios are approximate because groups are indivisible.

## Train

```powershell
python scripts/train.py
```

The new training script reads only `prepared_dataset.json`; it does not silently fall back to the old synthetic or OpenAssistant data. It fits the tokenizer only on training text, uses shifted teacher-forcing targets, rejects examples that would be silently truncated, selects the checkpoint by validation loss, restores that best checkpoint, and reports held-out test loss once at the end. New model/tokenizer artifacts are staged and replace the previous pair only after the run and test evaluation complete successfully. For Colab, use `scripts/colab_train_v2.ipynb`, not the older notebook that imports `train_colab.py`.

## Important limitations

- The 18 starter records are a **pipeline smoke-test seed**, not enough data to train a useful general-purpose language model. After reviewing them, build a larger human-checked set with broad task coverage before interpreting generation quality.
- A low token-level loss does not prove useful prompt analysis. Keep a separate fixed evaluation set and manually score decision accuracy, question relevance, information preservation, and R.C.F.O. fidelity.
- The old `expand_dataset.py` combinatorial generator is disabled because independent combinations of role/context/focus/output created contradictory labels.
- Keep the model checkpoint and tokenizer from the same training run. Back up the current files before replacing them.
