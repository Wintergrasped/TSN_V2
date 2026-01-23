"""
Utility functions for TSN.
"""

import hashlib
from pathlib import Path
from typing import BinaryIO


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


def extract_timestamp_from_filename(filename: str) -> str | None:
    """
    Extract timestamp from filename if present.
    
    Expected format: YYYYMMDD_HHMMSS.wav
    
    Args:
        filename: Audio file name
        
    Returns:
        ISO timestamp string or None
    """
    import re
    from datetime import datetime
    
    # Pattern: 20260122_123456.wav
    pattern = r"(\d{8})_(\d{6})"
    match = re.search(pattern, filename)
    
    if match:
        date_str = match.group(1)  # YYYYMMDD
        time_str = match.group(2)  # HHMMSS
        
        try:
            dt = datetime.strptime(f"{date_str}{time_str}", "%Y%m%d%H%M%S")
            return dt.isoformat()
        except ValueError:
            return None
    
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
