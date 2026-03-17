"""
Configuration module for AI Customer Support System.

Uses relative paths based on the project root directory.
"""

import os
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "model"
INTENT_DATA_PATH = MODEL_DIR / "intents.json"
INTENT_MODEL_PATH = MODEL_DIR / "data.pth"
NER_MODEL_DIR = MODEL_DIR / "ner_model"

# ── Model settings ─────────────────────────────────────────────────────
INTENT_CONFIDENCE_THRESHOLD = 0.75
NER_MODEL_NAME = "dslim/bert-base-NER"

# ── API settings ───────────────────────────────────────────────────────
API_HOST = "0.0.0.0"
API_PORT = int(os.environ.get("PORT", 8000))
MAX_REQUEST_SIZE = 1024  # bytes
CORS_ORIGINS = ["*"]

# ── Rate limiting (placeholder values) ─────────────────────────────────
RATE_LIMIT_REQUESTS = 60
RATE_LIMIT_WINDOW_SECONDS = 60
