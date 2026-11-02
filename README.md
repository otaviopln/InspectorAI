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
