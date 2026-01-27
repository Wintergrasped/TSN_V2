"""Helpers for the Net Control operator page."""

from __future__ import annotations

import asyncio
import csv
import io
import uuid
from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError

from tsn_common.models import CallsignLog, NetControlSession, Transcription

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
        bind = session.get_bind() if hasattr(session, "get_bind") else getattr(session, "bind", None)
        if bind is None:
            return
        try:
            async with bind.begin() as conn:
                await conn.run_sync(NetControlSession.__table__.create, checkfirst=True)
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
) -> dict:
    await _ensure_net_control_table(session)
    record = NetControlSession(
        name=name.strip() or "Unlabeled Net",
        status="active",
        started_by=started_by,
        started_by_callsign=started_by_callsign,
        notes=notes,
        started_at=datetime.now(timezone.utc),
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
    }


async def fetch_checkin_feed(session, limit: int = 50) -> list[dict]:
    result = await session.execute(
        select(CallsignLog, Transcription)
        .join(Transcription, CallsignLog.transcription_id == Transcription.id)
        .order_by(CallsignLog.detected_at.desc())
        .limit(limit)
    )
    feed: list[dict] = []
    for log, tx in result.all():
        feed.append(
            {
                "callsign_id": str(log.callsign_id),
                "detected_at": log.detected_at.isoformat(),
                "confidence": log.confidence,
                "context": (log.context_snippet or "")[:240],
                "transcript": (tx.transcript_text or "")[:400],
                "transcription_id": str(tx.id),
            }
        )
    return feed


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
