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
Inspector AI/
├── History.md                  # Project changelog
├── README.md                   # This file
├── requirements.txt            # Python dependencies
├── .gitignore                  # Git ignore rules
├── scripts/                    # Entry-point scripts
│   ├── train.py
│   └── run.py
└── inspector/                  # Core package
    ├── __init__.py
    ├── config.py               # Configuration and paths
    ├── core/
    │   ├── __init__.py
    │   ├── analyzer.py         # Prompt analyzer (dubious vs clear)
    │   ├── formatter.py        # R.C.F.O formatter
    │   └── questioner.py       # Strategic question generator
    ├── models/
    │   └── __init__.py         # PyTorch model definitions
    ├── data/
    │   ├── __init__.py
    │   ├── raw/                # Raw prompt datasets
    │   └── processed/          # Processed prompt datasets
    ├── utils/
    │   └── __init__.py         # Logging and helpers
    └── tests/
        └── __init__.py         # Unit and integration tests
```

## Installation

```bash
cd "Inspector AI"
pip install -r requirements.txt
```

## Usage

Coming soon — core modules are being implemented.
