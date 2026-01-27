"""Helpers for net summary views and APIs."""

from __future__ import annotations

import uuid
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from tsn_common.models.audio import AudioFile
from tsn_common.models.callsign import Callsign
from tsn_common.models.net import NetParticipation, NetSession


async def fetch_net_summary(session: AsyncSession, net_id: str) -> dict[str, Any] | None:
    """Return a rich summary for a single net session."""

    try:
        net_uuid = uuid.UUID(net_id)
    except (ValueError, TypeError):
        return None

    stmt = (
        select(NetSession)
        .options(
            selectinload(NetSession.ncs_callsign),
            selectinload(NetSession.participations).joinedload(NetParticipation.callsign),
            selectinload(NetSession.audio_file).selectinload(AudioFile.transcriptions),
        )
        .where(NetSession.id == net_uuid)
    )
    result = await session.execute(stmt)
    net: NetSession | None = result.scalar_one_or_none()
    if net is None:
        return None

    ncs_callsign: Callsign | None = net.ncs_callsign
    audio_file: AudioFile | None = net.audio_file

    participation_rows = sorted(
        net.participations,
        key=lambda part: (
            part.callsign.callsign if part.callsign else "zzz",
            part.last_seen,
        ),
        reverse=False,
    )

    transcripts: list[dict[str, Any]] = []
    if audio_file:
        ordered = sorted(audio_file.transcriptions, key=lambda tx: tx.created_at)
        for tx in ordered:
            transcripts.append(
                {
                    "id": str(tx.id),
                    "created_at": tx.created_at.isoformat(),
                    "backend": tx.backend.value,
                    "confidence": tx.confidence,
                    "text": tx.transcript_text,
                }
            )

    participant_payload = [
        {
            "callsign": part.callsign.callsign if part.callsign else "(unknown)",
            "validated": bool(part.callsign.validated) if part.callsign else False,
            "checkin_type": part.checkin_type.value,
            "transmissions": part.transmission_count,
            "talk_seconds": part.estimated_talk_seconds,
            "first_seen": part.first_seen.isoformat(),
            "last_seen": part.last_seen.isoformat(),
        }
        for part in participation_rows
    ]

    total_talk = sum(p["talk_seconds"] or 0 for p in participant_payload)
    total_transmissions = sum(p["transmissions"] or 0 for p in participant_payload)
    checkins = len(participant_payload)
    metrics = {
        "checkins": checkins,
        "validated_checkins": sum(1 for p in participant_payload if p["validated"]),
        "late_checkins": sum(1 for p in participant_payload if p["checkin_type"].lower() == "late"),
        "relay_checkins": sum(
            1 for p in participant_payload if p["checkin_type"].lower() in {"relay", "proxy"}
        ),
        "total_talk_seconds": total_talk,
        "avg_talk_seconds": int(total_talk / checkins) if checkins else 0,
        "avg_transmissions": round(total_transmissions / checkins, 1) if checkins else 0,
    }

    return {
        "id": str(net.id),
        "name": net.net_name,
        "club": net.club_name,
        "start_time": net.start_time.isoformat(),
        "end_time": net.end_time.isoformat(),
        "duration_sec": net.duration_sec,
        "participant_count": net.participant_count,
        "confidence": net.confidence,
        "ncs": ncs_callsign.callsign if ncs_callsign else None,
        "ncs_validated": bool(ncs_callsign.validated) if ncs_callsign else None,
        "ncs_method": ncs_callsign.validation_method.value if ncs_callsign and ncs_callsign.validation_method else None,
        "summary": net.summary,
        "topics": net.topics or [],
        "statistics": net.statistics or {},
        "metrics": metrics,
        "source_segments": net.source_segments or [],
        "participants": participant_payload,
        "transcripts": transcripts,
    }
