"""Build micro-windows of transcripts for vLLM evaluation."""

import uuid
from datetime import datetime, timedelta
from typing import Any

from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from tsn_common.models import AudioFile, Transcription, CallsignLog, Callsign
from tsn_common.logging import get_logger

logger = get_logger(__name__)


async def build_micro_window(
    session: AsyncSession,
    node_id: str | None,
    window_start: datetime,
    window_end: datetime,
) -> dict[str, Any]:
    """
    Build a micro-window summary for vLLM evaluation.
    
    Returns:
        {
            "window_start": ISO timestamp,
            "window_end": ISO timestamp,
            "node_id": str,
            "transmissions": 5,
            "unique_callsigns": ["K7XYZ", "W1ABC"],
            "top_callsigns": [{"callsign": "K7XYZ", "count": 3}],
            "duration_sec": 240,
            "excerpts": ["transcript snippet 1", ...],
            "transcript_ids": [uuid, ...],
            "audio_file_ids": [uuid, ...]
        }
    """
    # Query transcripts in window
    stmt = (
        select(Transcription, AudioFile)
        .join(AudioFile, Transcription.audio_file_id == AudioFile.id)
        .where(
            AudioFile.created_at >= window_start,
            AudioFile.created_at < window_end,
        )
    )
    if node_id:
        stmt = stmt.where(AudioFile.node_id == node_id)
    
    stmt = stmt.order_by(AudioFile.created_at)
    
    result = await session.execute(stmt)
    rows = result.all()
    
    if not rows:
        return {
            "window_start": window_start.isoformat(),
            "window_end": window_end.isoformat(),
            "node_id": node_id,
            "transmissions": 0,
            "unique_callsigns": [],
            "top_callsigns": [],
            "duration_sec": 0,
            "excerpts": [],
            "transcript_ids": [],
            "audio_file_ids": [],
        }
    
    # Extract data
    transcript_ids = []
    audio_file_ids = []
    all_text = []
    total_duration = 0
    
    for trans, audio in rows:
        transcript_ids.append(trans.id)
        audio_file_ids.append(audio.id)
        all_text.append(trans.transcript_text or "")
        total_duration += audio.duration_sec or 0
    
    # Get callsigns in window
    # CallsignLog.transcription_id -> Transcription -> AudioFile
    # CallsignLog.callsign_id -> Callsign.callsign
    callsign_stmt = (
        select(
            Callsign.callsign,
            func.count(CallsignLog.id).label("count")
        )
        .join(Callsign, CallsignLog.callsign_id == Callsign.id)
        .join(Transcription, CallsignLog.transcription_id == Transcription.id)
        .join(AudioFile, Transcription.audio_file_id == AudioFile.id)
        .where(
            AudioFile.created_at >= window_start,
            AudioFile.created_at < window_end,
        )
        .group_by(Callsign.callsign)
        .order_by(func.count(CallsignLog.id).desc())
    )
    if node_id:
        callsign_stmt = callsign_stmt.where(AudioFile.node_id == node_id)
    
    callsign_result = await session.execute(callsign_stmt)
    callsign_rows = callsign_result.all()
    
    unique_callsigns = [row.callsign for row in callsign_rows]
    top_callsigns = [
        {"callsign": row.callsign, "count": row.count}
        for row in callsign_rows[:10]
    ]
    
    return {
        "window_start": window_start.isoformat(),
        "window_end": window_end.isoformat(),
        "node_id": node_id,
        "transmissions": len(rows),
        "unique_callsigns": unique_callsigns,
        "top_callsigns": top_callsigns,
        "duration_sec": int(total_duration),
        "excerpts": all_text,  # Will be curated by excerpt_selector
        "transcript_ids": [str(tid) for tid in transcript_ids],
        "audio_file_ids": [str(aid) for aid in audio_file_ids],
    }


async def get_active_nodes(
    session: AsyncSession,
    since: datetime,
) -> list[str]:
    """Get list of nodes with activity since given time."""
    stmt = (
        select(AudioFile.node_id)
        .where(
            AudioFile.created_at >= since,
            AudioFile.node_id.isnot(None),
        )
        .group_by(AudioFile.node_id)
    )
    result = await session.execute(stmt)
    return [row[0] for row in result.all() if row[0]]
