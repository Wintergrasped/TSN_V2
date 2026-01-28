"""Helpers for the Net Control operator page."""

from __future__ import annotations

import asyncio
import csv
import io
import uuid
from datetime import datetime, timezone

from sqlalchemy import select, distinct, func
from sqlalchemy.exc import SQLAlchemyError

from tsn_common.models import CallsignLog, NetControlSession, Transcription, AudioFile, Callsign

_net_ctl_table_ready = False
_net_ctl_lock = asyncio.Lock()


async def _ensure_net_control_table(session) -> None:
    """Create the NetControlSession table on-the-fly if migrations were skipped."""

    global _net_ctl_table_ready
    if _net_ctl_table_ready:
        return

    async with _net_ctl_lock:
        if _net_ctl_table_ready:
            return
        try:
            async def _create_table(sync_conn):
                NetControlSession.__table__.create(bind=sync_conn, checkfirst=True)

            await session.run_sync(_create_table)
            _net_ctl_table_ready = True
        except SQLAlchemyError:
            # If creation fails we surface the original error during normal query/insert
            pass


async def list_sessions(session, limit: int = 10) -> list[dict]:
    await _ensure_net_control_table(session)
    result = await session.execute(
        select(NetControlSession)
        .order_by(NetControlSession.started_at.desc())
        .limit(limit)
    )
    return [serialize_session(row) for row in result.scalars()]


async def get_active_session(session) -> dict | None:
    await _ensure_net_control_table(session)
    result = await session.execute(
        select(NetControlSession)
        .where(NetControlSession.status == "active")
        .order_by(NetControlSession.started_at.desc())
        .limit(1)
    )
    row = result.scalar_one_or_none()
    return serialize_session(row) if row else None


async def start_session(
    session,
    *,
    name: str,
    started_by: str,
    started_by_callsign: str | None,
    notes: str | None = None,
    node_id: str | None = None,
) -> dict:
    await _ensure_net_control_table(session)
    
    # Store node_id in metadata if provided
    metadata = {}
    if node_id:
        metadata["node_id"] = node_id
    
    record = NetControlSession(
        name=name.strip() or "Unlabeled Net",
        status="active",
        started_by=started_by,
        started_by_callsign=started_by_callsign,
        notes=notes,
        started_at=datetime.now(timezone.utc),
        metadata_=metadata,
    )
    session.add(record)
    await session.flush()
    return serialize_session(record)


async def stop_session(session, session_id: uuid.UUID) -> dict | None:
    await _ensure_net_control_table(session)
    record = await session.get(NetControlSession, session_id)
    if record is None:
        return None
    record.status = "closed"
    record.ended_at = datetime.now(timezone.utc)
    await session.flush()
    return serialize_session(record)


def serialize_session(record: NetControlSession | None) -> dict | None:
    if record is None:
        return None
    
    # Extract node_id from metadata if present
    node_id = None
    if record.metadata_:
        node_id = record.metadata_.get("node_id")
    
    return {
        "id": str(record.id),
        "name": record.name,
        "status": record.status,
        "started_by": record.started_by,
        "started_by_callsign": record.started_by_callsign,
        "started_at": record.started_at.isoformat(),
        "ended_at": record.ended_at.isoformat() if record.ended_at else None,
        "notes": record.notes,
        "download_url": record.download_url,
        "node_id": node_id,
    }


async def fetch_checkin_feed(session, limit: int = 50, node_id: str | None = None) -> list[dict]:
    query = (
        select(CallsignLog, Transcription, AudioFile)
        .join(Transcription, CallsignLog.transcription_id == Transcription.id)
        .join(AudioFile, Transcription.audio_file_id == AudioFile.id)
    )
    
    # Filter by node if specified
    if node_id:
        query = query.where(AudioFile.node_id == node_id)
    
    query = query.order_by(CallsignLog.detected_at.desc()).limit(limit)
    
    result = await session.execute(query)
    feed: list[dict] = []
    for log, tx, audio in result.all():
        feed.append(
            {
                "callsign_id": str(log.callsign_id),
                "detected_at": log.detected_at.isoformat(),
                "confidence": log.confidence,
                "context": (log.context_snippet or "")[:240],
                "transcript": (tx.transcript_text or "")[:400],
                "transcription_id": str(tx.id),
                "node_id": audio.node_id,
            }
        )
    return feed


async def get_available_nodes(session) -> list[str]:
    """Get list of all unique node_ids from audio files."""
    result = await session.execute(
        select(distinct(AudioFile.node_id))
        .order_by(AudioFile.node_id)
    )
    return [node for node in result.scalars() if node != "unknown"]


async def fetch_live_callsigns(session, limit: int = 5, node_id: str | None = None) -> list[dict]:
    """
    Fetch most recent callsigns heard in live transcripts.
    Uses QRZ validation status if available.
    
    For active nets, this provides real-time awareness of who's on frequency.
    """
    query = (
        select(
            Callsign.callsign,
            Callsign.validated,
            Callsign.validation_method,
            CallsignLog.detected_at,
            CallsignLog.confidence,
            CallsignLog.context_snippet,
            AudioFile.node_id,
        )
        .join(CallsignLog, Callsign.id == CallsignLog.callsign_id)
        .join(Transcription, CallsignLog.transcription_id == Transcription.id)
        .join(AudioFile, Transcription.audio_file_id == AudioFile.id)
    )
    
    # Filter by node if specified
    if node_id:
        query = query.where(AudioFile.node_id == node_id)
    
    # Get most recent detections
    query = query.order_by(CallsignLog.detected_at.desc()).limit(limit)
    
    result = await session.execute(query)
    
    callsigns = []
    seen_callsigns = set()
    
    for row in result.all():
        # Deduplicate - show each callsign only once
        if row.callsign in seen_callsigns:
            continue
        seen_callsigns.add(row.callsign)
        
        # Determine QRZ check status
        qrz_status = "unknown"
        if row.validated:
            qrz_status = "valid" if row.validation_method and "qrz" in row.validation_method.value.lower() else "valid_other"
        
        callsigns.append({
            "callsign": row.callsign,
            "detected_at": row.detected_at.isoformat(),
            "confidence": row.confidence,
            "context": (row.context_snippet or "")[:120],
            "qrz_status": qrz_status,
            "validated": row.validated,
            "node_id": row.node_id,
        })
    
    return callsigns


async def export_feed_csv(session, limit: int = 250) -> str:
    feed = await fetch_checkin_feed(session, limit=limit)
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(["detected_at", "callsign_id", "confidence", "context", "transcription_id"])
    for row in feed:
        writer.writerow(
            [
                row["detected_at"],
                row["callsign_id"],
                row.get("confidence"),
                row.get("context", "").replace("\n", " "),
                row["transcription_id"],
            ]
        )
    return buffer.getvalue()
