"""Helpers powering the personalized /user/dashboard experience."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import joinedload

from tsn_common.models import (
    AudioFile,
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
from web.models.user import PortalUser


@dataclass
class ProfilePreferences:
    bio: str
    photo_url: str | None
    club_memberships: list[str]

    @property
    def clubs_text(self) -> str:
        return "\n".join(self.club_memberships)


def _clean_text(value: str | None) -> str:
    return (value or "").strip()


def _clean_optional(value: str | None) -> str | None:
    trimmed = (value or "").strip()
    return trimmed or None


def _parse_club_entries(raw: str | None) -> list[str]:
    entries: list[str] = []
    for chunk in (raw or "").replace("\r", "").split("\n"):
        for part in chunk.split(","):
            cleaned = part.strip()
            if cleaned:
                entries.append(cleaned)
    # Deduplicate while preserving order
    seen: set[str] = set()
    unique: list[str] = []
    for club in entries:
        key = club.lower()
        if key in seen:
            continue
        seen.add(key)
        unique.append(club)
    return unique


def get_profile_preferences(user: PortalUser) -> ProfilePreferences:
    prefs: dict[str, Any] = dict(user.preferences or {})
    profile: dict[str, Any] = dict(prefs.get("profile") or {})
    clubs = profile.get("club_memberships") or []
    if isinstance(clubs, str):  # legacy newline-delimited storage
        clubs = _parse_club_entries(clubs)
    return ProfilePreferences(
        bio=profile.get("bio") or "",
        photo_url=profile.get("photo_url"),
        club_memberships=[club for club in clubs if isinstance(club, str)],
    )


async def update_profile_preferences(
    session: AsyncSession,
    user: PortalUser,
    *,
    bio: str,
    photo_url: str,
    club_memberships: str,
) -> None:
    prefs = dict(user.preferences or {})
    profile = dict(prefs.get("profile") or {})
    profile["bio"] = _clean_text(bio)
    profile["photo_url"] = _clean_optional(photo_url)
    profile["club_memberships"] = _parse_club_entries(club_memberships)
    prefs["profile"] = profile
    user.preferences = prefs
    await session.flush()


async def build_user_dashboard_payload(
    session: AsyncSession,
    user: PortalUser,
    *,
    transcript_limit: int = 12,
) -> dict:
    prefs = get_profile_preferences(user)
    normalized = normalize_callsign(user.callsign or "") if user.callsign else None
    payload: dict[str, Any] = {
        "profile": {
            "display_name": user.display_name,
            "email": user.email,
            "callsign": user.callsign,
            "is_admin": user.is_admin,
        },
        "preferences": {
            "bio": prefs.bio,
            "photo_url": prefs.photo_url,
            "club_memberships": prefs.club_memberships,
            "club_memberships_text": prefs.clubs_text,
        },
        "memberships": {
            "detected": [],
            "manual": prefs.club_memberships,
        },
        "recent_transcripts": [],
        "net_activity": [],
        "stats": {
            "mentions_total": 0,
            "nets_total": 0,
            "transmissions_total": 0,
            "talk_seconds_total": 0,
            "last_active": None,
            "validated": False,
            "validation_method": None,
            "first_seen": None,
            "last_seen": None,
            "avg_talk_length": None,
        },
        "callsign_missing": normalized is None,
        "callsign_profile": None,
    }

    if not normalized:
        return payload

    result = await session.execute(
        select(Callsign)
        .options(joinedload(Callsign.profile))
        .where(Callsign.callsign == normalized)
    )
    record: Callsign | None = result.scalar_one_or_none()
    if record is None:
        return payload

    payload["callsign_missing"] = False
    profile: CallsignProfile | None = record.profile
    payload["callsign_profile"] = {
        "summary": profile.profile_summary if profile else None,
        "topics": profile.primary_topics if profile else [],
        "activity_score": profile.activity_score if profile else None,
        "engagement_score": profile.engagement_score if profile else None,
        "last_analyzed_at": profile.last_analyzed_at.isoformat() if profile and profile.last_analyzed_at else None,
    }

    payload["stats"].update(
        {
            "mentions_total": record.seen_count,
            "first_seen": record.first_seen.isoformat(),
            "last_seen": record.last_seen.isoformat(),
            "validated": record.validated,
            "validation_method": record.validation_method.value if record.validation_method else None,
        }
    )

    aggregates = await session.execute(
        select(
            func.count(NetParticipation.id),
            func.coalesce(func.sum(NetParticipation.transmission_count), 0),
            func.coalesce(func.sum(NetParticipation.estimated_talk_seconds), 0),
            func.max(NetParticipation.last_seen),
        ).where(NetParticipation.callsign_id == record.id)
    )
    net_count, tx_total, talk_total, last_active = aggregates.one()
    payload["stats"]["nets_total"] = net_count or 0
    payload["stats"]["transmissions_total"] = int(tx_total or 0)
    payload["stats"]["talk_seconds_total"] = int(talk_total or 0)
    payload["stats"]["last_active"] = last_active.isoformat() if last_active else None
    payload["stats"]["avg_talk_length"] = (
        int(talk_total) / net_count if net_count else None
    )

    membership_rows = await session.execute(
        select(ClubMembership, ClubProfile)
        .join(ClubProfile, ClubMembership.club_id == ClubProfile.id)
        .where(ClubMembership.callsign_id == record.id)
        .order_by(ClubMembership.last_seen.desc())
    )
    payload["memberships"]["detected"] = [
        {
            "club": club.name,
            "role": membership.role.value,
            "last_seen": membership.last_seen.isoformat(),
        }
        for membership, club in membership_rows.all()
    ]

    net_rows = await session.execute(
        select(NetParticipation, NetSession)
        .join(NetSession, NetParticipation.net_session_id == NetSession.id)
        .where(NetParticipation.callsign_id == record.id)
        .order_by(NetParticipation.last_seen.desc())
        .limit(8)
    )
    payload["net_activity"] = [
        {
            "name": net.net_name,
            "club": net.club_name,
            "start_time": net.start_time.isoformat(),
            "duration_sec": net.duration_sec,
            "participants": net.participant_count,
            "transmissions": participation.transmission_count,
            "talk_seconds": participation.estimated_talk_seconds,
            "checkin_type": participation.checkin_type.value,
        }
        for participation, net in net_rows.all()
    ]

    transcript_rows = await session.execute(
        select(CallsignLog, Transcription, AudioFile)
        .join(Transcription, CallsignLog.transcription_id == Transcription.id)
        .join(AudioFile, Transcription.audio_file_id == AudioFile.id)
        .where(CallsignLog.callsign_id == record.id)
        .order_by(CallsignLog.detected_at.desc())
        .limit(transcript_limit)
    )
    payload["recent_transcripts"] = [
        {
            "detected_at": log.detected_at.isoformat(),
            "confidence": log.confidence,
            "context": (log.context_snippet or "")[:360],
            "transcription_preview": (tx.transcript_text or "")[:360],
            "audio_filename": audio.filename,
            "audio_started_at": audio.created_at.isoformat() if audio.created_at else None,
            "transcription_id": str(tx.id),
        }
        for log, tx, audio in transcript_rows.all()
    ]

    return payload
