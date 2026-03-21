"""Configuration and API key management."""

import os
from pathlib import Path

BASE_DIR = Path(__file__).parent
PODCASTS_DIR = BASE_DIR / "podcasts"
OUTPUT_DIR = BASE_DIR / "output"
FEEDS_FILE = BASE_DIR / "feeds.json"
FETCH_STATE_FILE = BASE_DIR / ".fetch_state.json"


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
WHISPER_COMPUTE_TYPE = "float16"


def detect_gpu_backend() -> str:
    """Detect available GPU backend: 'cuda' (NVIDIA or ROCm) or 'cpu'.

    ROCm's HIP runtime presents through the CUDA API, so
    torch.cuda.is_available() returns True on both NVIDIA CUDA
    and AMD ROCm systems. CTranslate2 v4.7+ also supports ROCm
    natively via this same mechanism.
    """
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
    except ImportError:
        pass
    return "cpu"


WHISPER_DEVICE = detect_gpu_backend()


def detect_whisper_backend() -> str:
    """Detect which whisper library is available.

    Prefers faster-whisper (CTranslate2) for performance.
    Falls back to openai-whisper (PyTorch) which has broader
    GPU compatibility (works on any PyTorch-supported device
    including ROCm without special CTranslate2 builds).
    """
    try:
        import faster_whisper
        return "faster-whisper"
    except ImportError:
        pass
    try:
        import whisper
        return "openai-whisper"
    except ImportError:
        pass
    raise ImportError(
        "No whisper backend found. Install either:\n"
        "  pip install faster-whisper   (preferred, supports CUDA and ROCm via CTranslate2 v4.7+)\n"
        "  pip install openai-whisper   (fallback, works with any PyTorch backend)"
    )


WHISPER_BACKEND = detect_whisper_backend()
