# InspectorAI

**A specialized AI assistant for prompt analysis, clarification, and R.C.F.O. structuring.**

InspectorAI is an experimental AI project built with Python and PyTorch. It is designed to analyze user requests, identify missing information, and transform sufficiently clear prompts into a structured format.

The project implements a custom Encoder-Decoder Transformer, a Byte-Level BPE tokenizer, a controlled training pipeline, and a desktop interface built with CustomTkinter.

> InspectorAI is an educational and experimental machine learning project. It is not a general-purpose language model or a production-ready AI assistant.

## Overview

InspectorAI learns to perform two primary tasks:

### 1. Clarification

When a request is vague, incomplete, or under-specified, the model generates targeted questions instead of making unsupported assumptions.

For partially specified requests, it should preserve the information already provided and ask only about material gaps.

### 2. R.C.F.O. Formatting

When a request contains sufficient information, the model restructures it using the R.C.F.O. framework:

| Component | Description |
|---|---|
| Role | Who should perform the task |
| Context | Relevant background and circumstances |
| Focus | The specific task or objective |
| Output | The expected deliverable or format |

The intended behavior is to preserve the user's original requirements without introducing unsupported constraints, deliverables, or assumptions.

## Architecture

InspectorAI uses a custom sequence-to-sequence Transformer implemented with PyTorch primitives.

### Model components

- **Encoder:** processes the input prompt using self-attention.
- **Decoder:** generates the output autoregressively.
- **Multi-Head Attention:** implements scaled dot-product attention.
- **Cross-Attention:** connects decoder representations to the encoder output.
- **Positional Encoding:** injects sinusoidal positional information.
- **Feed-Forward Networks:** process token representations.
- **Layer Normalization and Residual Connections:** stabilize model training.

The model includes padding masks and causal masks to support sequence-to-sequence learning.

### Tokenizer

InspectorAI uses a Byte-Level BPE tokenizer powered by Hugging Face Tokenizers, whose implementation is written in Rust.

Special token IDs:

| Token | ID |
|---|---:|
| `<PAD>` | 0 |
| `<SOS>` | 1 |
| `<EOS>` | 2 |
| `<UNK>` | 3 |

The tokenizer is trained on the training split and is saved alongside the model checkpoint.

### Text generation

The inference implementation includes autoregressive generation with Beam Search.

The current implementation uses configurable generation parameters, including beam size, length penalty, and maximum generated tokens.

## Training Pipeline

The current training pipeline uses a reviewed and explicitly prepared dataset.

The training script does not silently fall back to the previous synthetic or OpenAssistant datasets.

### Dataset preparation

The expected source dataset is:

`inspector/data/raw/curated_examples.jsonl`

Each JSONL record represents an independently reviewable training example.

Required fields:

| Field | Description |
|---|---|
| `id` | Unique record identifier |
| `task` | `clarify` or `format_rcfo` |
| `ambiguity` | `vague`, `partial`, or `clear` |
| `language` | Dataset language |
| `input` | Original user request |
| `target` | Expected model response |
| `source` | Data provenance |
| `group_id` | Identifier for related examples |
| `review_status` | Review state |

The current training version expects English-language data.

Records must be explicitly approved before they can be included in the prepared dataset.

The preparation pipeline:

1. Validates the dataset schema and task compatibility.
2. Rejects duplicate IDs and duplicate inputs.
3. Excludes records that have not been approved.
4. Keeps related examples with the same `group_id` in the same split.
5. Creates training, validation, and test sets.
6. Generates a prepared dataset and a dataset report.

The intended split proportions are 80% training, 10% validation, and 10% test, with actual proportions depending on group sizes.

### Prepare the dataset

From the project root:

```powershell
python scripts/prepare_dataset.py
```

The resulting dataset is stored at:

`inspector/data/processed/prepared_dataset.json`

### Train the model

```powershell
python scripts/train.py
```

The training pipeline includes:

- Dataset validation.
- Tokenizer training.
- Teacher-forcing training.
- Cross-entropy loss.
- Adam optimization.
- Validation loss monitoring.
- Early stopping.
- Best-checkpoint selection.
- Held-out test evaluation.

The tokenizer and model artifacts are staged and replace the previous artifacts only after the training run and test evaluation complete successfully. The checkpoint records canonical JSON SHA-256 fingerprints for both the exact tokenizer artifact and the prepared dataset used in that run. Training aborts before publishing artifacts if either source changes while the run is in progress.

## Project Structure

```text
InspectorAI/
├── README.md
├── History.md
├── LICENSE
├── requirements.txt
├── .gitignore
│
├── inspector_gui.py
│
├── inspector/
│   ├── __init__.py
│   ├── __main__.py
│   ├── cli.py
│   ├── engine.py
│   ├── data/
│   │   ├── raw/
│   │   └── processed/
│   └── models/
│
├── scripts/
│   ├── model.py
│   ├── tokenizer.py
│   ├── train.py
│   ├── prepare_dataset.py
│   ├── inspect_sequence_lengths.py
│   ├── run.py
│   └── colab_train_v2.ipynb
│
├── tests/
│
└── legacy/
    ├── README.md
    ├── inspector/
    │   ├── config.py
    │   ├── core/
    │   └── tests/
    └── scripts/
        ├── train_colab.py
        ├── dataset_builder.py
        ├── expand_dataset.py
        ├── expand_dataset_60k.py
        ├── build_alpaca_dataset.py
        └── build_openassistant_dataset.py
```

The supported training and inference workflow is maintained outside
`legacy/`. The archive preserves the former heuristic implementation,
previous synthetic dataset generators, external-dataset conversion
experiments, and an older Colab trainer. Archived files are for historical
reference only and are not part of the supported execution path.

The inference engine lives exclusively in `inspector/engine.py`. The desktop
GUI and terminal interface use that same module.

Run the terminal interface with:

```powershell
python -m inspector
```

The compatibility launcher `python scripts/run.py` is also available.

## Installation

### Requirements

- Python 3.10 or newer
- PyTorch 2.0 or newer
- NumPy
- Hugging Face Tokenizers
- CustomTkinter

Install the project dependencies:

```powershell
git clone https://github.com/otaviopln/InspectorAI.git
cd InspectorAI

python -m venv .venv
.venv\Scripts\activate

python -m pip install --upgrade pip
pip install -r requirements.txt
```

For environments using CUDA, install a PyTorch build compatible with the installed NVIDIA driver and CUDA runtime.

## Usage

### Desktop application

Launch the graphical interface:

```powershell
python inspector_gui.py
```

The desktop application provides a chat-oriented interface for interacting with the locally loaded model.

The current GUI implementation includes:

- Dark workspace and sidebar.
- Responsive chat area.
- Animated loading and status indicators.
- Typewriter-style response rendering.
- Background model loading and inference.
- Thread-safe communication between inference and the interface.
- A single-inference guard.

The GUI requires the trained model and tokenizer artifacts to be available.

### Terminal application

Launch the interactive terminal interface:

```powershell
python -m inspector
```

The package entry point (`python -m inspector`) delegates to `inspector/cli.py`. Both the terminal application and desktop GUI import the same inference implementation from `inspector/engine.py`; checkpoint loading, architecture validation, and response generation are maintained in one place.\n\nThe previous root-level `inspector.py` engine has been removed to avoid a module/package name collision with the `inspector/` package. The compatibility launcher `python scripts/run.py` remains available.

## Training Configuration

The current training configuration is defined in `scripts/train.py`.

| Parameter | Value |
|---|---:|
| Vocabulary size | 3,000 |
| Model dimension | 256 |
| Attention heads | 8 |
| Transformer layers | 4 |
| Feed-forward dimension | 1,024 |
| Maximum sequence length | 192 |
| Batch size | 16 |
| Maximum epochs | 20 |
| Learning rate | 0.0005 |
| Early stopping patience | 5 |
| Random seed | 42 |

These values represent the current code configuration, not a guarantee of model quality or a universal recommendation.

The training script automatically selects CUDA when available, unless a device is explicitly configured.

Supported device overrides:

```powershell
$env:INSPECTOR_DEVICE="cpu"
python scripts/train.py
```

```powershell
$env:INSPECTOR_DEVICE="cuda"
python scripts/train.py
```

```powershell
$env:INSPECTOR_DEVICE="directml"
python scripts/train.py
```

The `directml` option requires the corresponding PyTorch DirectML dependency.

## Google Colab

The project includes a Colab-oriented training notebook:

`scripts/colab_train_v2.ipynb`

For GPU training:

1. Open the notebook in Google Colab.
2. Select a GPU runtime.
3. Make the project files and approved dataset available.
4. Execute the training pipeline.
5. Download the resulting model and tokenizer artifacts.
6. Place the artifacts in `inspector/models/`.

Keep the model checkpoint and tokenizer from the same training run. They are a matched pair and should not be mixed with artifacts from different runs.

## Model Artifacts

The training process produces:

| Artifact | Purpose |
|---|---|
| `inspector_model.pth` | Trained Transformer checkpoint |
| `tokenizer_vocab.json` | BPE tokenizer vocabulary and configuration |
| `training_metrics.json` | Training and evaluation metrics |

The checkpoint stores the model state, model configuration, and artifact integrity metadata.

### Artifact integrity

New format-v4 checkpoints record:

- `artifact_hashes.tokenizer`: SHA-256 fingerprint of `tokenizer_vocab.json`.
- `artifact_hashes.prepared_dataset`: SHA-256 fingerprint of `prepared_dataset.json`.
- `artifact_hashes.algorithm`: `sha256-canonical-json-v1`.

The fingerprints are calculated from canonicalized JSON (sorted object keys, compact separators, UTF-8 encoding). This makes the verification independent of indentation and Windows/Linux line endings while still detecting changes to the tokenizer vocabulary, merge rules, dataset records, or split assignments.

Before loading a format-v4 model, the inference engine verifies both files against the hashes stored in the checkpoint. A mismatch, or a missing prepared dataset, prevents the model from loading. Keep `inspector/data/processed/prepared_dataset.json`, `inspector/models/tokenizer_vocab.json`, and `inspector/models/inspector_model.pth` from the same training run.

Older checkpoints without hash metadata remain loadable for backward compatibility, but the engine prints a warning because their tokenizer/dataset pairing cannot be verified. Run `python scripts/train.py` to produce a format-v4 checkpoint with integrity validation.

The inference code supports checkpoints with explicit model configuration and provides compatibility handling for older checkpoints.

## Behavioral Evaluation

The behavioral evaluator reports more than decision recognition. It checks output integrity and runs a small, manually annotated semantic screening suite.

Run the full held-out test split and the annotated semantic suite:

    python scripts/evaluate.py

Run a small smoke test on the held-out split (the semantic suite is still included):

    python scripts/evaluate.py --limit 5

Run only the held-out split:

    python scripts/evaluate.py --skip-semantic-cases

Use a different annotated benchmark file:

    python scripts/evaluate.py --semantic-cases tests/fixtures/semantic_evaluation_cases.jsonl

The evaluator measures:

- Decision accuracy, coverage, class precision/recall/F1, and confusion matrix.
- Exact section counts, required non-empty sections, section order, and incompatible sections for each task.
- R.C.F.O. integrity: exactly one Role, Context, Focus, and Output, all non-empty and in the required order.
- Clarification format: at least one numbered question and detection of exact duplicate questions.
- Requirement-anchor coverage for manually annotated benchmark cases.
- Explicit forbidden-phrase matches for unsupported content listed in those annotations.

**Important limitation:** lexical anchors and forbidden phrases are deterministic screening checks, not semantic understanding. A paraphrase not included in the aliases can be marked as missing, and unsupported claims not explicitly listed cannot be detected by this mechanism. They do not prove that intent was preserved or that all invented content is absent.

The evaluator writes three artifacts to inspector/data/processed/:

| File | Purpose |
|---|---|
| evaluation_report.json | Aggregate metrics and limitations |
| evaluation_predictions.jsonl | Per-example outputs and automatic diagnostics |
| evaluation_review_queue.jsonl | Editable human-review template for every generated answer |

The human-review queue includes fields for intent preservation, requirement preservation, unsupported content, semantic fidelity, R.C.F.O. faithfulness, and clarification-question relevance. Complete this review before making claims about semantic quality.

Run the evaluator regression tests:

    pytest tests/test_evaluate.py -v

Run the full test suite:

    pytest -q

The evaluator uses the checkpoint's artifact-integrity validation. Keep inspector_model.pth, tokenizer_vocab.json, and prepared_dataset.json from the same training run.
## Limitations

InspectorAI is an experimental project. Its capabilities depend heavily on the quality, size, and diversity of the training dataset.

Important limitations include:

- A small dataset is insufficient for reliable general-purpose language understanding.
- Training loss alone does not establish useful prompt-analysis behavior.
- Generated clarification questions may be irrelevant or incomplete.
- R.C.F.O. formatting may omit requirements or introduce unsupported assumptions.
- English-language training data limits the reliability of other languages.
- The model does not have the broad knowledge or capabilities of a large pretrained language model.

The initial curated examples are intended to validate the data and training pipeline. They should not be interpreted as a sufficiently large or comprehensively human-validated training corpus.

Evaluation should include a separate held-out dataset and manual assessment of:

- Decision accuracy.
- Relevance of clarification questions.
- Preservation of user requirements.
- R.C.F.O. formatting fidelity.
- Unsupported assumptions.
- Output consistency.

## Development

InspectorAI is developed as an educational machine learning project focused on understanding the complete workflow of a specialized text-generation model.

The repository contains the model implementation, tokenizer, data preparation pipeline, training code, inference applications, and tests.

Run the checkpoint artifact integrity regression tests from the project root:

```powershell
pytest tests/test_artifact_integrity.py -v
```

Contributions, experiments, and improvements to the training data and evaluation methodology are welcome.

## License

This project is licensed under the MIT License.

See [LICENSE](LICENSE) for details.
