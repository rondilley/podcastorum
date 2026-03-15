"""Podcast Summarizer — transcribe locally, analyze with multi-LLM pipeline."""

import argparse
import sys
from pathlib import Path

import config
import transcriber
import analyzer


def build_transcript_md(title: str, source_name: str, segments: list[dict],
                        full_text: str, info: dict) -> str:
    """Build a markdown transcript document from transcription results."""
    timestamped = transcriber.format_transcript_with_timestamps(segments)
    duration_min = info["duration"] / 60

    # Transcription time may not be available if loaded from interrupted JSONL
    time_line = ""
    if "transcription_time" in info and info["transcription_time"]:
        t = info["transcription_time"]
        time_line = f"| **Transcription time** | {t:.1f}s ({info['duration'] / t:.1f}x realtime) |\n"

    return (
        f"# {title} — Transcript\n\n"
        f"| Detail | Value |\n"
        f"|--------|-------|\n"
        f"| **Source** | `{source_name}` |\n"
        f"| **Language** | {info['language']} (confidence: {info['language_probability']:.0%}) |\n"
        f"| **Duration** | {duration_min:.1f} minutes |\n"
        f"{time_line}\n"
        f"---\n\n"
        f"## Timestamped Transcript\n\n"
        f"```\n{timestamped}\n```\n\n"
        f"---\n\n"
        f"## Full Text\n\n"
        f"{full_text}\n"
    )


def process_podcast(audio_path: Path, whisper_model: str = None,
                    skip_transcribe: bool = False) -> None:
    """Full pipeline: transcribe audio, then multi-LLM analysis."""
    title = audio_path.stem
    config.OUTPUT_DIR.mkdir(exist_ok=True)

    jsonl_path = config.OUTPUT_DIR / f"{title}.segments.jsonl"
    transcript_path = config.OUTPUT_DIR / f"{title}.transcript.md"
    analysis_path = config.OUTPUT_DIR / f"{title}.analysis.md"

    # Step 1: Transcribe (or load existing)
    # Check if we already have a usable JSONL from a previous run
    coverage, seg_count, _ = transcriber.assess_completeness(jsonl_path)
    have_usable_jsonl = coverage >= 0.90

    if skip_transcribe and transcript_path.exists():
        print(f"Using existing transcript: {transcript_path}")
        transcript_text = transcript_path.read_text(encoding="utf-8")
        _run_analysis(transcript_text, title, analysis_path)
        return
    elif have_usable_jsonl and not skip_transcribe:
        print(f"Found existing JSONL with {coverage:.0%} coverage ({seg_count} segments) — reusing")
        result = transcriber.load_segments_from_jsonl(jsonl_path)
    elif skip_transcribe and jsonl_path.exists():
        print(f"Loading existing segments: {jsonl_path}")
        result = transcriber.load_segments_from_jsonl(jsonl_path)
    else:
        print("=" * 60)
        print("STEP 1: Local Transcription")
        print("=" * 60)
        # Run whisper in a subprocess so exit-code-127 crashes don't kill us
        transcriber.transcribe_in_subprocess(audio_path, model_size=whisper_model)

        coverage, seg_count, _ = transcriber.assess_completeness(jsonl_path)
        if seg_count == 0:
            print("Error: Transcription produced no segments.", file=sys.stderr)
            sys.exit(1)

        if coverage < 0.90:
            print(f"  Partial transcription: {coverage:.0%} coverage ({seg_count} segments)")
        else:
            print(f"  Transcription coverage: {coverage:.0%} ({seg_count} segments)")

        result = transcriber.load_segments_from_jsonl(jsonl_path)

    transcript_text = result["text"]

    # Save transcript markdown
    transcript_md = build_transcript_md(
        title, audio_path.name, result["segments"], transcript_text, result["info"]
    )
    transcript_path.write_text(transcript_md, encoding="utf-8")
    print(f"Saved transcript: {transcript_path}")

    # Step 2: Analyze
    _run_analysis(transcript_text, title, analysis_path)


def _run_analysis(transcript_text: str, title: str, analysis_path: Path) -> None:
    """Run multi-LLM analysis pipeline and save results."""
    print()
    print("=" * 60)
    print("STEP 2: Multi-LLM Editorial Analysis")
    print("=" * 60)

    try:
        editorial = analyzer.analyze(transcript_text, title)
    except Exception as e:
        print(f"Error during analysis: {e}", file=sys.stderr)
        sys.exit(1)

    analysis_path.write_text(editorial, encoding="utf-8")
    print(f"\nSaved analysis: {analysis_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Podcast Summarizer — local transcription + multi-LLM editorial analysis"
    )
    parser.add_argument(
        "audio_file",
        help="Path to the podcast audio file (any format ffmpeg supports)",
    )
    parser.add_argument(
        "--whisper-model",
        default=None,
        help=f"Whisper model size (default: {config.WHISPER_MODEL})",
    )
    parser.add_argument(
        "--skip-transcribe",
        action="store_true",
        help="Skip transcription if JSONL segments or transcript already exists",
    )

    args = parser.parse_args()
    audio_path = Path(args.audio_file)

    if not audio_path.exists():
        print(f"Error: File not found: {audio_path}", file=sys.stderr)
        sys.exit(1)

    process_podcast(
        audio_path,
        whisper_model=args.whisper_model,
        skip_transcribe=args.skip_transcribe,
    )


if __name__ == "__main__":
    main()
