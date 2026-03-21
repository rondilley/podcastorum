# Architecture

## System Overview

```mermaid
graph TD
    Z[fetcher.py<br/>RSS Discovery + Download] -->|podcasts/*.mp3| A
    A[summarizer.py<br/>CLI + Pipeline] --> B[transcriber.py<br/>Local GPU]
    A --> C[analyzer.py<br/>Multi-LLM]
    B --> D[faster-whisper / openai-whisper<br/>large-v3 / CUDA or ROCm]
    C --> E[Claude<br/>Sonnet 4]
    C --> F[OpenAI<br/>GPT-4.1]
    C --> G[xAI<br/>Grok-3]
    C --> H[Mistral<br/>Large-latest]
    Z --> Y[(feeds.json)]
    Z --> X[(.fetch_state.json)]
    D --> I[(segments.jsonl)]
    I --> J[(transcript.md)]
    E & F & G & H --> K[(analysis.md)]
```

## Module Responsibilities

### config.py

Centralized configuration. Loads API keys from `*.key.txt` files in the project root. Defines Whisper model settings (model size, compute type) and directory paths.

No environment variables -- keys are plain files for simplicity and compatibility with the llm_compare project structure.

**GPU auto-detection.** `detect_gpu_backend()` checks `torch.cuda.is_available()` which returns `True` for both NVIDIA CUDA and AMD ROCm (since ROCm's HIP runtime presents through the CUDA API). Falls back to `"cpu"` if no GPU is available.

**Whisper backend auto-detection.** `detect_whisper_backend()` tries to import `faster_whisper` first (preferred for performance), then falls back to `whisper` (OpenAI's original, broader GPU compatibility). Raises a clear error if neither is installed.

### transcriber.py

Handles local audio-to-text conversion with two backend options, auto-detected via `config.WHISPER_BACKEND`.

**Backend: faster-whisper (CTranslate2)**

The preferred backend. Supports NVIDIA CUDA and AMD ROCm (CTranslate2 v4.7+). Key characteristics:

- **Incremental JSONL output.** Each transcribed segment is flushed to disk immediately via a JSONL file. This is critical because the whisper process frequently exits with code 127 near completion (likely a CUDA/ctranslate2 cleanup issue). The JSONL file preserves all segments regardless.
- **VAD filtering.** Voice Activity Detection skips silence, reducing processing time and improving transcript quality.
- **Progress reporting.** Reports every 5% to avoid flooding stdout.

**Backend: openai-whisper (PyTorch)**

The fallback backend. Works on any PyTorch-supported GPU including ROCm. Key differences:

- **No incremental output.** All segments are returned at once after transcription completes, then written to JSONL for consistency with the faster-whisper format.
- **No VAD filtering.** OpenAI's whisper does not have built-in VAD.
- **2-4x slower** than faster-whisper for the same model and accuracy.

**Common design:**

- **JSONL format.** Both backends write identical output: line 1 is a metadata record (`{"_info": {...}}`), followed by one segment per line (`{"start", "end", "text"}`). The `load_segments_from_jsonl()` function reconstructs the full transcript from either backend's output.
- **Public API is backend-agnostic.** `transcribe()`, `load_segments_from_jsonl()`, `assess_completeness()`, and `transcribe_in_subprocess()` work identically regardless of backend.

### analyzer.py

Multi-LLM editorial analysis pipeline with three phases:

```mermaid
graph LR
    subgraph Phase 1: Independent Analyses
        T[Transcript] --> CA[Claude]
        T --> OA[OpenAI]
        T --> XA[xAI]
        T --> MA[Mistral]
        CA --> A1[Analysis A]
        OA --> A2[Analysis B]
        XA --> A3[Analysis C]
        MA --> A4[Analysis D]
    end
```

```mermaid
graph LR
    subgraph Phase 2: Adversarial Critiques
        A1[Analysis A<br/>Claude] --- C1[Critiqued by<br/>OpenAI, xAI, Mistral]
        A2[Analysis B<br/>OpenAI] --- C2[Critiqued by<br/>Claude, xAI, Mistral]
        A3[Analysis C<br/>xAI] --- C3[Critiqued by<br/>Claude, OpenAI, Mistral]
        A4[Analysis D<br/>Mistral] --- C4[Critiqued by<br/>Claude, OpenAI, xAI]
    end
```

```mermaid
graph TD
    subgraph Phase 3: Collaborative Synthesis
        A1[4 Analyses] --> S[Claude Synthesizer]
        C1[12 Critiques] --> S
        T1[Transcript] --> S
        S --> F[Final Editorial]
    end
```

**Provider abstraction.** The `LLMProvider` class wraps different API clients behind a `generate(system, user) -> str` interface. Anthropic uses its native SDK; OpenAI, xAI, and Mistral all use the OpenAI SDK with different base URLs.

**Auto-discovery.** `get_providers()` attempts to initialize each provider from its key file. Missing or invalid keys are skipped with a warning. The pipeline works with 1-N providers.

**Graceful degradation.** With a single provider, phases 2 and 3 are skipped entirely and the solo analysis is returned directly.

**Prompt architecture.** Three distinct prompt templates:
- `ANALYSIS_PROMPT_TEMPLATE` -- Asks for editorial analysis (Phase 1)
- `CRITIQUE_PROMPT_TEMPLATE` -- Asks for adversarial critique of another analysis (Phase 2)
- `SYNTHESIS_PROMPT_TEMPLATE` -- Asks for final synthesis from all analyses and critiques (Phase 3)

All prompts operate under `RON_DILLEY_SYSTEM_PROMPT`, which defines the editorial voice.

### fetcher.py

Podcast discovery and episode download via RSS feeds. Operates independently from the analysis pipeline but can optionally trigger it via `--analyze`.

Key design decisions:

- **RSS auto-discovery.** Given a bare website URL (e.g., `darknetdiaries.com`), the fetcher parses the HTML for `<link rel="alternate" type="application/rss+xml">` tags. If none are found, it probes common RSS paths (`/feed`, `/rss`, `/feed.xml`, `/rss.xml`, `/atom.xml`, `/podcast.xml`, `/feed/podcast`, `/index.xml`). Uses stdlib `html.parser.HTMLParser` to avoid a BeautifulSoup dependency.

- **Two-file state model.** `feeds.json` stores user-managed podcast sources (name, feed URL, website, date added). `.fetch_state.json` stores runtime download state (downloaded GUIDs, file path mappings, last-checked timestamps). Separation keeps the user config clean and the auto-generated state gitignored.

- **GUID-based dedup.** Episodes are identified by their RSS `<guid>` element, which is more stable than titles (which can be edited after publication). The state file tracks all downloaded GUIDs per podcast.

- **Atomic downloads.** Audio files download to a `.part` temp file first, then `rename()` to the final path on completion. Interrupted downloads never leave partial files that could be mistaken for complete episodes.

- **Feed parsing via feedparser.** Handles both `<enclosure>` tags (standard for podcasts) and `<media:content>` elements. Extracts GUID, title, audio URL, publication date, and file size.

- **URL normalization.** All URLs (both `--rss` and website URLs) are normalized with `https://` if no scheme is provided, and HTML responses to `--rss` URLs produce a clear error directing the user to use `add <url>` for auto-discovery instead.

### summarizer.py

CLI entry point and pipeline orchestrator. Coordinates the two-step flow:

1. Transcribe (or load existing JSONL/transcript)
2. Analyze (multi-LLM pipeline)

Handles the `--skip-transcribe` flag by checking for existing JSONL files first (preferred, since they contain structured data), then falling back to existing transcript markdown files.

Builds the transcript markdown document with metadata table, timestamped segments, and full text.

## Data Flow

```mermaid
graph TD
    P[Podcast URL] --> Q[fetcher: discover RSS feed]
    Q --> R[fetcher: parse feed, filter new episodes]
    R --> S[fetcher: download to podcasts/]
    S --> A
    A[podcasts/title.m4a] --> B[transcriber.transcribe]
    B -->|incremental write| C[output/title.segments.jsonl]
    C --> D[build_transcript_md]
    D --> E[output/title.transcript.md]
    C --> F[analyzer.analyze]
    F --> G{Phase 1}
    G -->|4 analyses| H{Phase 2}
    H -->|12 critiques| I{Phase 3}
    I -->|synthesis| J[output/title.analysis.md]
```

## Transcription Recovery Flow

```mermaid
graph TD
    A[Start] --> B{--skip-transcribe?}
    B -->|No| C[Run transcriber.transcribe]
    C --> D[Segments saved incrementally to JSONL]
    D --> E{Process completed?}
    E -->|Yes| F[Build transcript.md]
    E -->|No / exit 127| G[JSONL has all flushed segments]
    G -->|Re-run with --skip-transcribe| H[load_segments_from_jsonl]
    B -->|Yes| I{JSONL exists?}
    I -->|Yes| H
    I -->|No| J{transcript.md exists?}
    J -->|Yes| K[Read transcript text directly]
    J -->|No| C
    H --> F
    F --> L[Run analysis pipeline]
    K --> L
```

## API Call Pattern

With 4 providers, the analysis pipeline makes:

| Phase | API Calls | Purpose |
|-------|-----------|---------|
| Phase 1 | 4 | One analysis per provider |
| Phase 2 | 12 | Each provider critiques the other 3 |
| Phase 3 | 1 | Synthesis by first provider (Claude) |
| **Total** | **17** | |

All calls are sequential (no concurrency) to keep the implementation simple and avoid rate limiting.

## Fetcher State Model

The fetcher uses two JSON files to separate user configuration from runtime state:

```mermaid
graph LR
    subgraph User-Managed
        F[feeds.json<br/>Podcast sources]
    end
    subgraph Auto-Managed
        S[.fetch_state.json<br/>Download history]
    end
    F --> FE[fetcher.py]
    FE --> S
    FE -->|new episodes| P[podcasts/]
    FE -->|--analyze| SU[summarizer.py]
```

**feeds.json** stores registered podcasts (name, feed URL, website, date added). Managed by `add`/`remove` subcommands.

**.fetch_state.json** tracks per-podcast download state: downloaded GUIDs, file path mappings, and last-checked timestamps. Updated automatically after each successful download. Gitignored.

## Known Issues

- **Whisper exit code 127.** The faster-whisper process frequently exits with code 127 near 95-100% completion. The incremental JSONL output ensures no data is lost. Root cause is likely a CUDA or ctranslate2 cleanup issue, not a transcription failure.

- **Large transcript context.** The full transcript is sent to each LLM in every phase. For very long podcasts (2+ hours), this may approach context limits on some models. No chunking strategy is currently implemented.
