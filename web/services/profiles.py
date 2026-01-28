"""Profile-centric query helpers for callsigns and clubs."""

from __future__ import annotations

from datetime import datetime, timezone, timedelta

from sqlalchemy import select
from sqlalchemy.orm import joinedload

from tsn_common.models import (
    Callsign,
    CallsignLog,
    CallsignProfile,
    ClubMembership,
    ClubProfile,
    NetParticipation,
    NetSession,
    Transcription,
)
from tsn_common.utils import normalize_callsign
from web.services.ai import summarize_callsign, summarize_club


async def fetch_callsign_profile(session, callsign: str) -> dict | None:
    normalized = normalize_callsign(callsign)
    result = await session.execute(
        select(Callsign)
        .options(joinedload(Callsign.profile))
        .where(Callsign.callsign == normalized)
    )
    record: Callsign | None = result.scalar_one_or_none()
    if record is None:
        return None

    profile: CallsignProfile | None = record.profile

    memberships = await session.execute(
        select(ClubMembership, ClubProfile)
        .join(ClubProfile, ClubMembership.club_id == ClubProfile.id)
        .where(ClubMembership.callsign_id == record.id)
        .order_by(ClubMembership.last_seen.desc())
    )

    nets = await session.execute(
        select(NetParticipation, NetSession)
        .join(NetSession, NetParticipation.net_session_id == NetSession.id)
        .where(NetParticipation.callsign_id == record.id)
        .order_by(NetParticipation.last_seen.desc())
        .limit(20)
    )

    logs = await session.execute(
        select(CallsignLog, Transcription)
        .join(Transcription, CallsignLog.transcription_id == Transcription.id)
        .where(CallsignLog.callsign_id == record.id)
        .order_by(CallsignLog.detected_at.desc())
        .limit(25)
    )

    payload = {
        "callsign": record.callsign,
        "validated": record.validated,
        "validation_method": record.validation_method.value if record.validation_method else None,
        "first_seen": record.first_seen.isoformat(),
        "last_seen": record.last_seen.isoformat(),
        "seen_count": record.seen_count,
        "metadata": record.metadata_,
        "profile": {
            "summary": profile.profile_summary if profile else None,
            "topics": profile.primary_topics if profile else None,
            "activity_score": profile.activity_score if profile else None,
            "engagement_score": profile.engagement_score if profile else None,
            "last_analyzed_at": profile.last_analyzed_at.isoformat() if profile and profile.last_analyzed_at else None,
            "metadata": profile.metadata_ if profile else {},
        },
        "memberships": [
            {
                "club": club.name,
                "role": membership.role.value,
                "last_seen": membership.last_seen.isoformat(),
            }
            for membership, club in memberships.all()
        ],
        "net_activity": [
            {
                "net": net_session.net_name,
                "club": net_session.club_name,
                "start_time": net_session.start_time.isoformat(),
                "duration_sec": net_session.duration_sec,
                "transmissions": participation.transmission_count,
                "talk_seconds": participation.estimated_talk_seconds,
                "checkin_type": participation.checkin_type.value,
            }
            for participation, net_session in nets.all()
        ],
        "recent_logs": [
            {
                "detected_at": log.detected_at.isoformat(),
                "confidence": log.confidence,
                "context": (log.context_snippet or "")[:320],
                "transcription_preview": (tx.transcript_text or "")[:320],
            }
            for log, tx in logs.all()
        ],
    }

    # Check for cached AI summary first (performance optimization)
    cached_summary = None
    if profile and profile.metadata_:
        cached_summary = profile.metadata_.get("ai_summary")
        cache_timestamp = profile.metadata_.get("ai_summary_timestamp")
        # Regenerate if summary is older than 7 days
        if cached_summary and cache_timestamp:
            from datetime import datetime, timezone, timedelta
            cache_age = datetime.now(timezone.utc) - datetime.fromisoformat(cache_timestamp)
            if cache_age > timedelta(days=7):
                cached_summary = None
    
    if cached_summary:
        payload["ai_summary"] = cached_summary
    else:
        # Generate new summary and cache it
        new_summary = await summarize_callsign(record.callsign, payload)
        payload["ai_summary"] = new_summary
        
        # Store in profile metadata for future use
        if profile:
            metadata = dict(profile.metadata_ or {})
            metadata["ai_summary"] = new_summary
            metadata["ai_summary_timestamp"] = datetime.now(timezone.utc).isoformat()
            profile.metadata_ = metadata
            await session.flush()
    
    return payload


async def fetch_ncs_profile(session, callsign: str) -> dict | None:
    normalized = normalize_callsign(callsign)
    result = await session.execute(select(Callsign).where(Callsign.callsign == normalized))
    record = result.scalar_one_or_none()
    if record is None:
        return None

    nets = await session.execute(
        select(NetSession)
        .where(NetSession.ncs_callsign_id == record.id)
        .order_by(NetSession.start_time.desc())
        .limit(50)
    )

    return {
        "callsign": record.callsign,
        "nets": [
            {
                "name": net.net_name,
                "club": net.club_name,
                "start_time": net.start_time.isoformat(),
                "duration_sec": net.duration_sec,
                "participants": net.participant_count,
                "topics": net.topics or [],
                "summary": net.summary,
            }
            for net in nets.scalars()
        ],
    }


async def fetch_club_profile(session, club_name: str) -> dict | None:
    trimmed = club_name.strip()
    result = await session.execute(
        select(ClubProfile)
        .options(joinedload(ClubProfile.memberships))
        .where(ClubProfile.name == trimmed)
    )
    club = result.unique().scalar_one_or_none()
    if club is None:
        return None

    membership_rows = await session.execute(
        select(ClubMembership, Callsign)
        .join(Callsign, ClubMembership.callsign_id == Callsign.id)
        .where(ClubMembership.club_id == club.id)
        .order_by(ClubMembership.last_seen.desc())
    )

    nets = await session.execute(
        select(NetSession)
        .where(NetSession.club_name == club.name)
        .order_by(NetSession.start_time.desc())
        .limit(25)
    )

    payload = {
        "name": club.name,
        "summary": club.summary,
        "schedule": club.schedule,
        "last_analyzed_at": club.last_analyzed_at.isoformat() if club.last_analyzed_at else None,
        "metadata": club.metadata_ or {},
        "members": [
            {
                "callsign": callsign.callsign,
                "role": membership.role.value,
                "last_seen": membership.last_seen.isoformat(),
                "notes": membership.notes,
            }
            for membership, callsign in membership_rows.all()
        ],
        "nets": [
            {
                "name": net.net_name,
                "start_time": net.start_time.isoformat(),
                "duration_sec": net.duration_sec,
                "participants": net.participant_count,
                "summary": net.summary,
            }
            for net in nets.scalars()
        ],
    }

    if club.summary:
        payload["ai_summary"] = club.summary
    else:
        payload["ai_summary"] = await summarize_club(club.name, payload)
    return payload
