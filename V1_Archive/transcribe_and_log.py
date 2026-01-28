#!/usr/bin/env python3
"""Transcribe audio and persist results using a GPU-optimized backend."""
from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Optional

import mysql.connector

try:  # Optional accelerated backend
    from faster_whisper import WhisperModel
except ImportError:  # pragma: no cover - optional dependency
    WhisperModel = None  # type: ignore

try:  # Legacy fallback
    import whisper as openai_whisper
except ImportError:  # pragma: no cover
    openai_whisper = None

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None

# ------------------------------
# Configurable via environment
# ------------------------------
BACKEND = os.getenv("TRANSCRIBE_BACKEND", "faster-whisper").lower()
LANGUAGE = os.getenv("WHISPER_LANGUAGE", "en")

FASTER_MODEL_NAME = os.getenv("FASTER_WHISPER_MODEL", os.getenv("WHISPER_MODEL", "medium.en"))
FASTER_DEVICE = os.getenv("FASTER_WHISPER_DEVICE", "cuda")
FASTER_COMPUTE = os.getenv("FASTER_WHISPER_COMPUTE_TYPE", "float16")
FASTER_BEAM = int(os.getenv("FASTER_WHISPER_BEAM_SIZE", "5"))
FASTER_VAD = os.getenv("FASTER_WHISPER_VAD", "1") == "1"
FASTER_TEMPERATURE = float(os.getenv("FASTER_WHISPER_TEMPERATURE", "0.0"))

WHISPER_CPP_BIN = os.getenv("WHISPER_CPP_BIN", "")
WHISPER_CPP_MODEL = os.getenv("WHISPER_CPP_MODEL", "ggml-medium.en.bin")
WHISPER_CPP_EXTRA = os.getenv("WHISPER_CPP_EXTRA_ARGS", "--threads 8 --gpu 1 --no-timestamps")

LEGACY_MODEL = os.getenv("OPENAI_WHISPER_MODEL", os.getenv("WHISPER_MODEL", "medium.en"))
LEGACY_THREADS = int(os.getenv("WHISPER_THREADS", "8"))

DB_HOST = os.getenv("DB_HOST", "localhost")
DB_USER = os.getenv("DB_USER", "repeateruser")
DB_PASS = os.getenv("DB_PASS", "changeme123")
DB_NAME = os.getenv("DB_NAME", "repeater")

_FA_MODEL: Optional[WhisperModel] = None


def ensure_audio_path() -> Path:
    if len(sys.argv) < 2:
        print("No input file provided. Exiting cleanly.")
        sys.exit(0)
    audio_path = Path(sys.argv[1]).expanduser().resolve()
    if not audio_path.exists():
        print(f"Audio file not found: {audio_path}")
        sys.exit(1)
    return audio_path


def get_faster_model() -> WhisperModel:
    if WhisperModel is None:
        raise RuntimeError("faster-whisper not installed")
    global _FA_MODEL
    if _FA_MODEL is None:
        _FA_MODEL = WhisperModel(
            FASTER_MODEL_NAME,
            device=FASTER_DEVICE,
            compute_type=FASTER_COMPUTE,
        )
    return _FA_MODEL


def transcribe_with_faster(audio_path: str) -> str:
    model = get_faster_model()
    segments, _ = model.transcribe(
        audio_path,
        beam_size=FASTER_BEAM,
        vad_filter=FASTER_VAD,
        language=LANGUAGE,
        temperature=FASTER_TEMPERATURE,
        condition_on_previous_text=False,
    )
    text = " ".join(seg.text.strip() for seg in segments).strip()
    if not text:
        raise RuntimeError("faster-whisper returned empty transcript")
    return text


def transcribe_with_whisper_cpp(audio_path: str) -> str:
    if not WHISPER_CPP_BIN:
        raise RuntimeError("Set WHISPER_CPP_BIN for whisper-cpp backend")
    out_dir = Path(tempfile.mkdtemp(prefix="whispercpp_"))
    stem = Path(audio_path).stem
    cmd = [
        WHISPER_CPP_BIN,
        "-m",
        WHISPER_CPP_MODEL,
        "-f",
        audio_path,
        "-l",
        LANGUAGE,
        "-otxt",
        "-oj",
        "-od",
        str(out_dir),
        "-of",
        stem,
    ]
    if WHISPER_CPP_EXTRA.strip():
        cmd.extend(shlex.split(WHISPER_CPP_EXTRA))
    try:
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0:
            raise RuntimeError(f"whisper.cpp exited {res.returncode}: {res.stderr.strip()}")
        json_path = out_dir / f"{stem}.json"
        if json_path.exists():
            payload = json.loads(json_path.read_text(encoding="utf-8"))
            text = " ".join(seg.get("text", "").strip() for seg in payload.get("segments", []))
        else:
            txt_path = out_dir / f"{stem}.txt"
            if not txt_path.exists():
                raise RuntimeError("whisper.cpp produced no output file")
            text = txt_path.read_text(encoding="utf-8")
        text = text.strip()
        if not text:
            raise RuntimeError("whisper.cpp returned empty transcript")
        return text
    finally:
        shutil.rmtree(out_dir, ignore_errors=True)


def transcribe_with_legacy(audio_path: str) -> str:
    if openai_whisper is None:
        raise RuntimeError("openai-whisper not installed")
    if torch is not None:
        torch.set_num_threads(LEGACY_THREADS)
    model = openai_whisper.load_model(LEGACY_MODEL)
    result = model.transcribe(audio_path, language=LANGUAGE)
    text = (result.get("text") or "").strip()
    if not text:
        raise RuntimeError("whisper returned empty transcript")
    return text


def transcribe_audio(audio_path: Path) -> str:
    if BACKEND == "faster-whisper":
        return transcribe_with_faster(str(audio_path))
    if BACKEND == "whisper-cpp":
        return transcribe_with_whisper_cpp(str(audio_path))
    if BACKEND == "whisper":
        return transcribe_with_legacy(str(audio_path))
    raise RuntimeError(f"Unknown backend '{BACKEND}'")


def insert_transcription(filename: str, transcript: str):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    db = mysql.connector.connect(
        host=DB_HOST,
        user=DB_USER,
        password=DB_PASS,
        database=DB_NAME,
    )
    try:
        cur = db.cursor()
        cur.execute(
            """
            INSERT INTO transcriptions (filename, timestamp, transcription)
            VALUES (%s, %s, %s)
            """,
            (filename, ts, transcript),
        )
        db.commit()
    finally:
        cur.close()
        db.close()


def main() -> int:
    audio_path = ensure_audio_path()
    try:
        transcript = transcribe_audio(audio_path)
    except Exception as exc:
        print(f"Transcription failed: {exc}", file=sys.stderr)
        return 1

    try:
        insert_transcription(audio_path.name, transcript)
    except Exception as exc:
        print(f"Database insert failed: {exc}", file=sys.stderr)
        return 2

    print(f"Transcribed and logged: {audio_path.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
