"""Local podcast transcription using faster-whisper with GPU acceleration."""

import json
import sys
import time
from pathlib import Path

from faster_whisper import WhisperModel

import config


def transcribe(audio_path: str | Path, model_size: str = None,
               output_dir: Path = None) -> dict:
    """Transcribe an audio file locally using faster-whisper.

    Writes segments incrementally to a JSONL file so progress is not lost
    if the process is interrupted. Returns a dict with 'text', 'segments',
    and 'info'.
    """
    audio_path = Path(audio_path)
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    model_size = model_size or config.WHISPER_MODEL
    output_dir = output_dir or config.OUTPUT_DIR
    output_dir.mkdir(exist_ok=True)

    # Incremental output file
    jsonl_path = output_dir / f"{audio_path.stem}.segments.jsonl"

    print(f"Loading Whisper model '{model_size}' on {config.WHISPER_DEVICE}...")
    model = WhisperModel(
        model_size,
        device=config.WHISPER_DEVICE,
        compute_type=config.WHISPER_COMPUTE_TYPE,
    )

    print(f"Transcribing: {audio_path.name}")
    start_time = time.time()

    segments_iter, info = model.transcribe(
        str(audio_path),
        beam_size=5,
        vad_filter=True,
        vad_parameters=dict(
            min_silence_duration_ms=500,
        ),
    )

    print(f"Detected language: {info.language} (probability {info.language_probability:.2f})")
    print(f"Duration: {info.duration:.1f}s ({info.duration / 60:.1f} min)")

    # Write info header as first line
    info_dict = {
        "language": info.language,
        "language_probability": info.language_probability,
        "duration": info.duration,
    }
    with open(jsonl_path, "w", encoding="utf-8") as f:
        f.write(json.dumps({"_info": info_dict}) + "\n")

    segments = []
    full_text_parts = []
    last_pct_reported = -1

    with open(jsonl_path, "a", encoding="utf-8") as f:
        for segment in segments_iter:
            seg_dict = {
                "start": segment.start,
                "end": segment.end,
                "text": segment.text.strip(),
            }
            segments.append(seg_dict)
            full_text_parts.append(segment.text.strip())

            # Write each segment immediately to disk
            f.write(json.dumps(seg_dict) + "\n")
            f.flush()

            # Progress indicator — report every 5%
            pct = int((segment.end / info.duration) * 100) if info.duration > 0 else 0
            pct_bucket = (pct // 5) * 5
            if pct_bucket > last_pct_reported:
                last_pct_reported = pct_bucket
                print(f"  Progress: {pct_bucket}%", flush=True)

    elapsed = time.time() - start_time
    print(f"Transcription complete in {elapsed:.1f}s ({info.duration / elapsed:.1f}x realtime)")

    info_dict["transcription_time"] = elapsed

    return {
        "text": " ".join(full_text_parts),
        "segments": segments,
        "info": info_dict,
    }


def load_segments_from_jsonl(jsonl_path: Path) -> dict:
    """Load a previously saved JSONL transcription file."""
    segments = []
    info = {}
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            if "_info" in data:
                info = data["_info"]
            else:
                segments.append(data)

    full_text = " ".join(seg["text"] for seg in segments)
    return {"text": full_text, "segments": segments, "info": info}


def format_transcript_with_timestamps(segments: list[dict]) -> str:
    """Format segments into a readable timestamped transcript."""
    lines = []
    for seg in segments:
        start_min, start_sec = divmod(int(seg["start"]), 60)
        start_hr, start_min = divmod(start_min, 60)
        timestamp = f"{start_hr:02d}:{start_min:02d}:{start_sec:02d}"
        lines.append(f"[{timestamp}] {seg['text']}")
    return "\n".join(lines)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <audio_file>")
        sys.exit(1)

    result = transcribe(sys.argv[1])
    print("\n--- Transcript ---")
    print(format_transcript_with_timestamps(result["segments"]))
