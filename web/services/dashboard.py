"""Database query helpers backing the portal dashboards."""

import asyncio
from collections import defaultdict

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from tsn_common.logging import get_logger
from tsn_common.models.audio import AudioFile
from tsn_common.models.callsign import Callsign, ValidationMethod
from tsn_common.models.club import ClubProfile
from tsn_common.models.net import NetSession
from tsn_common.models.support import SystemHealth
from tsn_common.models.transcription import Transcription
from tsn_common.models.trend import TrendSnapshot
from web.services.ai import merge_entities, summarize_dashboard_sections


logger = get_logger(__name__)
_AI_TIMEOUT_SEC = 6
_DEFAULT_AI_SECTIONS = {
    "queue": "AI summary unavailable.",
    "nets": "AI summary unavailable.",
    "clubs": "AI summary unavailable.",
    "callsigns": "AI summary unavailable.",
    "health": "AI summary unavailable.",
}


async def get_audio_queue_snapshot(session: AsyncSession) -> dict:
    """Return counts for each AudioFile state and totals."""

    state_counts: dict[str, int] = defaultdict(int)
    result = await session.execute(
        select(AudioFile.state, func.count()).group_by(AudioFile.state)
    )
    for state, count in result:
        state_counts[state.value if hasattr(state, "value") else state] = count

    total_files = sum(state_counts.values())
    return {
        "total": total_files,
        "states": state_counts,
    }


async def get_recent_callsigns(session: AsyncSession, limit: int = 25) -> list[dict]:
    stmt = (
        select(Callsign)
        .where(
            Callsign.validated.is_(True),
            Callsign.validation_method == ValidationMethod.QRZ,
        )
        .order_by(Callsign.last_seen.desc())
        .limit(limit)
    )
    result = await session.execute(stmt)
    return [
        {
            "callsign": c.callsign,
            "validated": c.validated,
            "validation_method": c.validation_method.value if c.validation_method else None,
            "first_seen": c.first_seen.isoformat(),
            "last_seen": c.last_seen.isoformat(),
            "seen_count": c.seen_count,
        }
        for c in result.scalars().all()
    ]


async def get_recent_transcriptions(session: AsyncSession, limit: int = 10) -> list[dict]:
    stmt = (
        select(Transcription)
        .order_by(Transcription.created_at.desc())
        .limit(limit)
    )
    result = await session.execute(stmt)
    return [
        {
            "id": str(t.id),
            "audio_file_id": str(t.audio_file_id),
            "created_at": t.created_at.isoformat(),
            "confidence": t.confidence,
            "preview": t.transcript_text[:280],
        }
        for t in result.scalars().all()
    ]


async def get_recent_nets(session: AsyncSession, limit: int = 10) -> list[dict]:
    stmt = (
        select(NetSession)
        .order_by(NetSession.start_time.desc())
        .limit(limit)
    )
    result = await session.execute(stmt)
    return [
        {
            "id": str(net.id),
            "name": net.net_name,
            "club": net.club_name,
            "start_time": net.start_time.isoformat(),
            "duration_sec": net.duration_sec,
            "participants": net.participant_count,
            "confidence": net.confidence,
        }
        for net in result.scalars().all()
    ]


async def get_trend_highlights(session: AsyncSession, limit: int = 5) -> list[dict]:
    stmt = (
        select(TrendSnapshot)
        .order_by(TrendSnapshot.window_start.desc())
        .limit(limit)
    )
    result = await session.execute(stmt)
    return [
        {
            "window_start": t.window_start.isoformat(),
            "window_end": t.window_end.isoformat(),
            "topics": t.trending_topics or [],
            "callsigns": t.trending_callsigns or [],
            "notable_nets": t.notable_nets or [],
            "notes": t.notes,
        }
        for t in result.scalars().all()
    ]


async def get_club_profiles(session: AsyncSession, limit: int = 25) -> list[dict]:
    stmt = (
        select(ClubProfile)
        # MySQL/MariaDB do not support NULLS LAST, so emulate it with a boolean sort key.
        .order_by(
            ClubProfile.last_analyzed_at.is_(None),
            ClubProfile.last_analyzed_at.desc(),
        )
        .limit(limit)
    )
    result = await session.execute(stmt)
    return [
        {
            "name": club.name,
            "summary": club.summary,
            "schedule": club.schedule,
            "members": club.metadata_.get("member_count") if club.metadata_ else None,
            "last_analyzed_at": club.last_analyzed_at.isoformat() if club.last_analyzed_at else None,
        }
        for club in result.scalars().unique().all()
    ]


async def get_system_health(session: AsyncSession) -> list[dict]:
    result = await session.execute(select(SystemHealth))
    return [
        {
            "component": row.component,
            "status": row.status,
            "last_heartbeat": row.last_heartbeat.isoformat(),
            "cpu_percent": row.cpu_percent,
            "memory_mb": row.memory_mb,
            "disk_free_gb": row.disk_free_gb,
            "metrics": row.metrics,
        }
        for row in result.scalars().all()
    ]


async def get_dashboard_payload(session: AsyncSession) -> dict:
    """Aggregate everything the landing page requires."""

    queue = await get_audio_queue_snapshot(session)
    callsigns = await get_recent_callsigns(session)
    transcriptions = await get_recent_transcriptions(session)
    nets = await get_recent_nets(session)
    trends = await get_trend_highlights(session)
    clubs = await get_club_profiles(session)
    health = await get_system_health(session)

    alias_map: dict[str, str] = {}
    club_names = [club["name"] for club in clubs if club.get("name")]
    if club_names:
        try:
            alias_map = await asyncio.wait_for(
                merge_entities("club", club_names),
                timeout=_AI_TIMEOUT_SEC,
            )
        except asyncio.TimeoutError:
            logger.warning("dashboard_alias_timeout", timeout_sec=_AI_TIMEOUT_SEC)
        except Exception as exc:  # pragma: no cover - network variability
            logger.error("dashboard_alias_failed", error=str(exc))
    grouped: dict[str, set[str]] = defaultdict(set)
    for alias, canonical in alias_map.items():
        grouped[canonical].add(alias)
    for club in clubs:
        canonical = alias_map.get(club["name"], club["name"])
        club["canonical_name"] = canonical
        aliases = grouped.get(canonical, set())
        club["aliases"] = sorted(a for a in aliases if a != canonical)

    ai_sections = _DEFAULT_AI_SECTIONS.copy()
    try:
        ai_sections = await asyncio.wait_for(
            summarize_dashboard_sections(
                {
                    "queue": queue,
                    "callsigns": callsigns[:10],
                    "nets": nets[:10],
                    "clubs": clubs[:10],
                    "health": health[:5],
                }
            ),
            timeout=_AI_TIMEOUT_SEC,
        )
    except asyncio.TimeoutError:
        logger.warning("dashboard_ai_timeout", timeout_sec=_AI_TIMEOUT_SEC)
    except Exception as exc:  # pragma: no cover - network variability
        logger.error("dashboard_ai_failed", error=str(exc))

    return {
        "queue": queue,
        "callsigns": callsigns,
        "transcriptions": transcriptions,
        "nets": nets,
        "trends": trends,
        "clubs": clubs,
        "health": health,
        "ai_summaries": ai_sections,
    }
