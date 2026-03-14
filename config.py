"""Configuration and API key management."""

import os
from pathlib import Path

BASE_DIR = Path(__file__).parent
PODCASTS_DIR = BASE_DIR / "podcasts"
OUTPUT_DIR = BASE_DIR / "output"


def load_key(filename: str) -> str:
    """Load an API key from a key file."""
    key_path = BASE_DIR / filename
    if not key_path.exists():
        raise FileNotFoundError(f"Key file not found: {key_path}")
    return key_path.read_text().strip()


def get_claude_key() -> str:
    return load_key("claude.key.txt")


def get_openai_key() -> str:
    return load_key("openai.key.txt")


def get_xai_key() -> str:
    return load_key("xai.key.txt")


def get_mistral_key() -> str:
    return load_key("mistral.key.txt")


# Whisper model settings
WHISPER_MODEL = "large-v3"
WHISPER_DEVICE = "cuda"
WHISPER_COMPUTE_TYPE = "float16"
