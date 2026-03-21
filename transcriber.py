"""Local podcast transcription with GPU acceleration.

Supports two backends:
  - faster-whisper (CTranslate2) — preferred, works on NVIDIA CUDA and AMD ROCm (CTranslate2 v4.7+)
  - openai-whisper (PyTorch) — fallback, works on any PyTorch-supported device including ROCm

Backend is auto-detected by config.WHISPER_BACKEND.
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import config


def check_gpu_available() -> None:
    """Check that the GPU is available and not locked by another process.

    On ROCm systems, /dev/kfd is the GPU device node. If another process
    holds it exclusively, whisper will segfault instead of giving a useful
    error. This check catches that before we waste time loading a model.
    """
    if config.WHISPER_DEVICE == "cpu":
        return

    # Check if /dev/kfd is in use (ROCm systems)
    kfd = Path("/dev/kfd")
    if kfd.exists():
        try:
            result = subprocess.run(
                ["fuser", str(kfd)],
                capture_output=True, text=True, timeout=5,
            )
            pids = result.stdout.strip().split()
            # Exclude our own process tree (subprocess spawns python3
            # which may touch /dev/kfd during PyTorch import)
            own_pids = {str(os.getpid()), str(os.getppid())}
            other_pids = [p for p in pids if p not in own_pids]
            if other_pids:
                # Identify what's using the GPU
                procs = []
                for pid in other_pids:
                    try:
                        comm = Path(f"/proc/{pid}/comm").read_text().strip()
                        procs.append(f"{comm} (PID {pid})")
                    except (FileNotFoundError, PermissionError):
                        procs.append(f"PID {pid}")
                proc_list = ", ".join(procs)
                print(
                    f"Error: GPU is in use by: {proc_list}\n"
                    f"Whisper needs exclusive GPU access. "
                    f"Stop the other process(es) or wait for them to finish.",
                    file=sys.stderr,
                )
                sys.exit(1)
        except FileNotFoundError:
            pass  # fuser not installed, skip check
        except subprocess.TimeoutExpired:
            pass  # fuser hung, skip check


# ---------------------------------------------------------------------------
# Backend: faster-whisper (CTranslate2)
# ---------------------------------------------------------------------------

def _transcribe_faster_whisper(audio_path: Path, model_size: str,
                                jsonl_path: Path) -> dict:
    """Transcribe using faster-whisper with incremental segment output."""
    from faster_whisper import WhisperModel

    print(f"Loading Whisper model '{model_size}' on {config.WHISPER_DEVICE} (faster-whisper)...")
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

            f.write(json.dumps(seg_dict) + "\n")
            f.flush()

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


# ---------------------------------------------------------------------------
# Backend: openai-whisper (PyTorch)
# ---------------------------------------------------------------------------

def _transcribe_openai_whisper(audio_path: Path, model_size: str,
                                jsonl_path: Path) -> dict:
    """Transcribe using OpenAI whisper (PyTorch backend).

    This backend works on any PyTorch-supported device including ROCm.
    Unlike faster-whisper, segments are returned all at once after
    transcription completes (no incremental streaming), but we still
    write them to JSONL for consistency.
    """
    import whisper

    device = config.WHISPER_DEVICE
    print(f"Loading Whisper model '{model_size}' on {device} (openai-whisper)...")
    model = whisper.load_model(model_size, device=device)

    print(f"Transcribing: {audio_path.name}")
    start_time = time.time()

    result = model.transcribe(
        str(audio_path),
        beam_size=5,
        verbose=False,
    )

    language = result.get("language", "unknown")
    # OpenAI whisper doesn't provide language probability or duration directly;
    # estimate duration from the last segment's end time
    raw_segments = result.get("segments", [])
    duration = raw_segments[-1]["end"] if raw_segments else 0.0

    print(f"Detected language: {language}")
    print(f"Duration: {duration:.1f}s ({duration / 60:.1f} min)")

    info_dict = {
        "language": language,
        "language_probability": 0.0,
        "duration": duration,
    }

    # Write JSONL in same format as faster-whisper backend
    with open(jsonl_path, "w", encoding="utf-8") as f:
        f.write(json.dumps({"_info": info_dict}) + "\n")
        for seg in raw_segments:
            seg_dict = {
                "start": seg["start"],
                "end": seg["end"],
                "text": seg["text"].strip(),
            }
            f.write(json.dumps(seg_dict) + "\n")

    elapsed = time.time() - start_time
    print(f"Transcription complete in {elapsed:.1f}s ({duration / elapsed:.1f}x realtime)")

    info_dict["transcription_time"] = elapsed

    segments = [
        {"start": s["start"], "end": s["end"], "text": s["text"].strip()}
        for s in raw_segments
    ]

    return {
        "text": " ".join(s["text"] for s in segments),
        "segments": segments,
        "info": info_dict,
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def transcribe(audio_path: str | Path, model_size: str = None,
               output_dir: Path = None) -> dict:
    """Transcribe an audio file locally using the best available backend.

    Writes segments to a JSONL file so progress is not lost if the
    process is interrupted. Returns a dict with 'text', 'segments',
    and 'info'.
    """
    audio_path = Path(audio_path)
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    model_size = model_size or config.WHISPER_MODEL
    output_dir = output_dir or config.OUTPUT_DIR
    output_dir.mkdir(exist_ok=True)

    jsonl_path = output_dir / f"{audio_path.stem}.segments.jsonl"

    backend = config.WHISPER_BACKEND
    print(f"Using whisper backend: {backend}")
    print(f"GPU device: {config.WHISPER_DEVICE}")

    check_gpu_available()

    if backend == "faster-whisper":
        return _transcribe_faster_whisper(audio_path, model_size, jsonl_path)
    else:
        return _transcribe_openai_whisper(audio_path, model_size, jsonl_path)


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


def assess_completeness(jsonl_path: Path) -> tuple[float, int, float]:
    """Check how complete a JSONL transcription is.

    Returns (coverage_ratio, segment_count, total_duration).
    """
    if not jsonl_path.exists():
        return 0.0, 0, 0.0

    info = {}
    last_segment = None
    segment_count = 0

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            if "_info" in data:
                info = data["_info"]
            else:
                last_segment = data
                segment_count += 1

    duration = info.get("duration", 0.0)
    if not last_segment or duration <= 0:
        return 0.0, segment_count, duration

    coverage = last_segment["end"] / duration
    return coverage, segment_count, duration


def transcribe_in_subprocess(audio_path: str | Path, model_size: str = None,
                             output_dir: Path = None) -> Path:
    """Run transcription in a subprocess so exit-code-127 crashes are contained."""
    audio_path = Path(audio_path)
    model_size = model_size or config.WHISPER_MODEL
    output_dir = output_dir or config.OUTPUT_DIR

    jsonl_path = output_dir / f"{audio_path.stem}.segments.jsonl"

    cmd = [
        sys.executable, __file__,
        str(audio_path),
        "--model", model_size,
        "--output-dir", str(output_dir),
    ]

    print(f"Starting transcription subprocess...")
    result = subprocess.run(cmd)

    if result.returncode != 0:
        print(f"  Whisper process exited with code {result.returncode} (expected near completion)")

    return jsonl_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transcribe audio with whisper")
    parser.add_argument("audio_file", help="Path to audio file")
    parser.add_argument("--model", default=None, help="Whisper model size")
    parser.add_argument("--output-dir", default=None, help="Output directory")
    args = parser.parse_args()

    output_dir = Path(args.output_dir) if args.output_dir else None
    transcribe(args.audio_file, model_size=args.model, output_dir=output_dir)
