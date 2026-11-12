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

RERANK_MODEL = os.getenv("RERANK_MODEL", "BAAI/bge-reranker-v2-m3")
SEARCH_DEPTH = int(os.getenv("SEARCH_DEPTH", "20"))  # how many results vector and BM25 each return
CANDIDATES = int(os.getenv("CANDIDATES", "5"))       # how many fused results the reranker re-scores
# 5, not 20: same scores on the eval (E2), 11s vs 32s per question on CPU. See docs/experiments.md.
RRF_K = int(os.getenv("RRF_K", "60"))            # RRF constant (standard: 60)
RETRIEVAL_MODE = os.getenv("RETRIEVAL_MODE", "rerank")  # vector | bm25 | hybrid | rerank

CHAT_MODEL = os.getenv("CHAT_MODEL", "qwen3.5:4b")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.1"))  # low = factual, repeatable
LLM_SEED = int(os.getenv("LLM_SEED", "42"))                   # fixed seed = same answer for same prompt
NUM_CTX = int(os.getenv("NUM_CTX", "8192"))                   # context window, in tokens
HISTORY_TURNS = int(os.getenv("HISTORY_TURNS", "6"))          # 1 turn = 1 question + 1 answer
