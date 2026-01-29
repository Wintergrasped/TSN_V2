"""
Utility functions for TSN.
"""

import hashlib
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import BinaryIO, TypedDict


class FilenameMetadata(TypedDict, total=False):
    """Structured metadata extracted from audio filenames."""

    node_id: str | None
    recorded_at: datetime | None
    format: str
    is_archive: bool


def compute_sha256(file_path: Path) -> str:
    """
    Compute SHA256 hash of a file.
    
    Args:
        file_path: Path to the file
        
    Returns:
        Hex string of SHA256 hash
    """
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


def compute_sha256_stream(stream: BinaryIO) -> str:
    """
    Compute SHA256 hash from a stream.
    
    Args:
        stream: Binary stream to hash
        
    Returns:
        Hex string of SHA256 hash
    """
    sha256_hash = hashlib.sha256()
    for byte_block in iter(lambda: stream.read(4096), b""):
        sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


def normalize_callsign(callsign: str) -> str:
    """
    Normalize a callsign to uppercase, remove suffixes.
    
    Examples:
        K7ABC/P -> K7ABC
        w7xyz-1 -> W7XYZ
        
    Args:
        callsign: Raw callsign string
        
    Returns:
        Normalized callsign
    """
    # Remove common suffixes
    callsign = callsign.upper().strip()
    
    # Remove /portable, /mobile, etc.
    if "/" in callsign:
        callsign = callsign.split("/")[0]
    
    # Remove SSID (e.g., -1, -2)
    if "-" in callsign:
        callsign = callsign.split("-")[0]
    
    return callsign


def _parse_timestamp(ts: str) -> datetime | None:
    """Parse a YYYYMMDDHHMMSS timestamp into an aware datetime."""

    try:
        return datetime.strptime(ts, "%Y%m%d%H%M%S").replace(tzinfo=timezone.utc)
    except ValueError:
        return None


_NEW_FILENAME_PATTERN = re.compile(
    r"^(?P<node>\d+|[A-Za-z0-9\-_]+)_(?P<ts>\d{14,16})(?:[._-].+)?$"
)
_LEGACY_PATTERN = re.compile(r"^(?P<ts>\d{14})$")
_UNDERSCORE_PATTERN = re.compile(r"^(?P<date>\d{8})_(?P<time>\d{6})")


def parse_audio_filename_metadata(filename: str) -> FilenameMetadata:
    """Extract node ID and capture timestamp clues from a filename."""

    stem = Path(filename).stem
    info: FilenameMetadata = {
        "node_id": None,
        "recorded_at": None,
        "format": "unknown",
        "is_archive": False,
    }

    match = _NEW_FILENAME_PATTERN.match(stem)
    if match:
        info["node_id"] = match.group("node")
        info["recorded_at"] = _parse_timestamp(match.group("ts"))
        info["format"] = "node_timestamp"
        return info

    match = _LEGACY_PATTERN.match(stem)
    if match:
        info["recorded_at"] = _parse_timestamp(match.group("ts"))
        info["format"] = "legacy_timestamp"
        info["is_archive"] = True
        return info

    match = _UNDERSCORE_PATTERN.match(stem)
    if match:
        info["recorded_at"] = _parse_timestamp(match.group("date") + match.group("time"))
        info["format"] = "legacy_timestamp"
        info["is_archive"] = True

    return info


def extract_timestamp_from_filename(filename: str) -> str | None:
    """Backward-compatible wrapper retaining legacy behavior."""

    parsed = parse_audio_filename_metadata(filename)
    if parsed["recorded_at"]:
        return parsed["recorded_at"].isoformat()
    return None


def get_audio_duration(file_path: Path) -> float | None:
    """
    Get duration of audio file in seconds.
    
    Args:
        file_path: Path to audio file
        
    Returns:
        Duration in seconds, or None if cannot be determined
    """
    try:
        import soundfile as sf
        
        info = sf.info(str(file_path))
        return float(info.duration)
    except Exception:
        return None


def get_audio_metadata(file_path: Path) -> dict:
    """
    Get audio file metadata (sample rate, channels, duration).
    
    Args:
        file_path: Path to audio file
        
    Returns:
        Dict with metadata, or empty dict on error
    """
    try:
        import soundfile as sf
        
        info = sf.info(str(file_path))
        return {
            "sample_rate": info.samplerate,
            "channels": info.channels,
            "duration_sec": float(info.duration),
            "format": info.format,
            "subtype": info.subtype,
        }
    except Exception:
        return {}
