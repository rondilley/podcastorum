# Podcast Summarizer

A Python CLI tool that transcribes podcast audio locally using GPU-accelerated Whisper and produces editorial-style analysis through a multi-LLM cooperative/adversarial pipeline.

Transcription runs entirely on local hardware (no audio leaves the machine). The analysis pipeline sends transcripts to multiple LLM providers, has them independently analyze and then adversarially critique each other's work, and synthesizes the strongest insights into a final editorial written in the voice of Ron Dilley.

## Requirements

- Python 3.12+
- NVIDIA GPU with CUDA support (for transcription)
- PyTorch with CUDA
- API keys for one or more LLM providers (Claude, OpenAI, xAI, Mistral)

## Setup

```bash
pip install -r requirements.txt
```

Place API keys in the project root as plain text files:

```
claude.key.txt      # Anthropic API key
openai.key.txt      # OpenAI API key
xai.key.txt         # xAI API key
mistral.key.txt     # Mistral API key
```

Only the providers with valid key files will be used. The pipeline works with as few as one provider but produces richer output with multiple.

## Usage

```bash
# Full pipeline: transcribe + multi-LLM analysis
python summarizer.py "podcasts/episode.m4a"

# Skip transcription if already done (uses saved JSONL segments)
python summarizer.py "podcasts/episode.m4a" --skip-transcribe

# Use a smaller Whisper model for faster transcription
python summarizer.py "podcasts/episode.m4a" --whisper-model medium
```

Accepts any audio format supported by ffmpeg (m4a, mp3, wav, flac, ogg, etc.).

## Output

All output is written to the `output/` directory:

| File | Contents |
|------|----------|
| `{title}.segments.jsonl` | Incremental transcription segments (crash-safe) |
| `{title}.transcript.md` | Formatted transcript with metadata and timestamps |
| `{title}.analysis.md` | Final editorial analysis in markdown |

## How It Works

### Step 1: Local Transcription

Uses [faster-whisper](https://github.com/SYSTRAN/faster-whisper) with the `large-v3` model running on CUDA. Segments are written incrementally to a JSONL file so that progress is preserved even if the process is interrupted. VAD (Voice Activity Detection) filtering skips silence automatically.

### Step 2: Multi-LLM Editorial Analysis

The analysis pipeline runs three phases inspired by the [llm_compare](../llm_compare) framework:

**Phase 1 -- Independent Analyses (Cooperative)**
Each available LLM provider independently analyzes the transcript, producing its own editorial take on the podcast content.

**Phase 2 -- Adversarial Critiques**
Each provider critiques every other provider's analysis, identifying blind spots, weak arguments, missed connections, and factual issues. With 4 providers, this produces 12 cross-critiques.

**Phase 3 -- Collaborative Synthesis**
A synthesizer model (Claude, by default) combines the strongest insights from all analyses while addressing valid critiques raised during the adversarial phase. The result is a single editorial that benefits from multiple perspectives and has been stress-tested through debate.

### Editorial Voice

All output is written in the voice, tone, and style of Ron Dilley -- cybersecurity expert, former CISO, and self-described "cyber security curmudgeon (in training)." The editorial style favors conversational authority, irreverent humor, honest assessment, and practical takeaways over dry summarization.

## Project Structure

```
podcast_summarizer/
    summarizer.py       # CLI entry point, orchestrates the pipeline
    transcriber.py      # Local GPU transcription via faster-whisper
    analyzer.py         # Multi-LLM analysis (independent, adversarial, synthesis)
    config.py           # API key loading, model and device settings
    requirements.txt    # Python dependencies
    podcasts/           # Input audio files
    output/             # Generated transcripts and analyses
```

## Whisper Model Options

| Model | VRAM | Speed | Accuracy |
|-------|------|-------|----------|
| `tiny` | ~1 GB | Fastest | Low |
| `base` | ~1 GB | Fast | Fair |
| `small` | ~2 GB | Moderate | Good |
| `medium` | ~5 GB | Slower | Better |
| `large-v3` | ~10 GB | Slowest | Best (default) |

Override with `--whisper-model`:

```bash
python summarizer.py "podcasts/episode.m4a" --whisper-model small
```

## LLM Providers

| Provider | Model | API |
|----------|-------|-----|
| Anthropic Claude | claude-sonnet-4 | Anthropic SDK |
| OpenAI | gpt-4.1 | OpenAI SDK |
| xAI Grok | grok-3 | OpenAI-compatible |
| Mistral | mistral-large-latest | OpenAI-compatible |

Providers are auto-discovered based on which `*.key.txt` files exist. The pipeline degrades gracefully -- with a single provider it skips the adversarial phase and returns the solo analysis directly.
