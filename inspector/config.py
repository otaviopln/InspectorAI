"""Configuration settings for Inspector AI."""

import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = PROJECT_ROOT / "inspector"
DATA_DIR = PACKAGE_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = PACKAGE_ROOT / "models"
LOGS_DIR = PROJECT_ROOT / "logs"

# Model settings
MODEL_NAME = "inspector-prompt-analyzer"
EMBEDDING_DIM = 128
HIDDEN_DIM = 256
NUM_CLASSES = 2  # clear vs dubious
LEARNING_RATE = 1e-3
BATCH_SIZE = 32
EPOCHS = 10

# R.C.F.O framework fields
RCFO_FIELDS = ["Role", "Context", "Focus", "Output"]

# Logging
LOG_LEVEL = "INFO"

# Ensure directories exist
for _dir in [RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, LOGS_DIR]:
    _dir.mkdir(parents=True, exist_ok=True)
