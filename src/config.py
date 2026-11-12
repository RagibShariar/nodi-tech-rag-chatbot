"""Project settings. Read from environment variables / .env, with defaults that work out of the box."""

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")  # does nothing if .env doesn't exist

DATA_DIR = ROOT / "data"
EVAL_DIR = ROOT / "eval"

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
EMBED_MODEL = os.getenv("EMBED_MODEL", "bge-m3")
QDRANT_PATH = ROOT / os.getenv("QDRANT_PATH", "artifacts/qdrant")
QDRANT_URL = os.getenv("QDRANT_URL") or None  # set -> use a Qdrant server instead of local mode
