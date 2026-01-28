"""Database query helpers backing the portal dashboards."""

from __future__ import annotations

import asyncio
import copy
import time
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from tsn_common.logging import get_logger
from tsn_common.models.audio import AudioFile
from tsn_common.models.callsign import Callsign, ValidationMethod
from tsn_common.models.club import ClubProfile
from tsn_common.models.net import NetSession
from tsn_common.models.support import ProcessingMetric, SystemHealth
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

_ALIAS_CACHE_TTL_SEC = 300
_SUMMARY_CACHE_TTL_SEC = 120

_alias_cache: dict[str, Any] = {
    "key": None,
    "expires": 0.0,
    "data": {},
}
_alias_task: asyncio.Task | None = None

_summary_cache: dict[str, Any] = {
    "expires": 0.0,
    "data": _DEFAULT_AI_SECTIONS.copy(),
}
_summary_task: asyncio.Task | None = None


def _club_alias_key(club_names: list[str]) -> str:
    normalized = sorted({name.strip() for name in club_names if name})
    return "|".join(normalized)


def _cached_alias_map(key: str) -> dict[str, str]:
    now = time.time()
    if (
        _alias_cache.get("key") == key
        and _alias_cache.get("expires", 0) > now
        and _alias_cache.get("data")
    ):
        return dict(_alias_cache["data"])
    if _alias_cache.get("data") and _alias_cache.get("expires", 0) > now:
        return dict(_alias_cache["data"])
    return {}


def _schedule_alias_refresh(club_names: list[str], cache_key: str) -> None:
    global _alias_task
    if not club_names:
        return
    if _alias_task and not _alias_task.done():
        return

    async def _refresh() -> None:
        try:
            result = await asyncio.wait_for(
                merge_entities("club", club_names[:50]),
                timeout=_AI_TIMEOUT_SEC,
            )
        except asyncio.TimeoutError:
            logger.warning("dashboard_alias_timeout", timeout_sec=_AI_TIMEOUT_SEC)
            return
        except Exception as exc:  # pragma: no cover - network variability
            logger.error("dashboard_alias_failed", error=str(exc))
            return

        _alias_cache.update(
            {
                "key": cache_key,
                "expires": time.time() + _ALIAS_CACHE_TTL_SEC,
                "data": result,
            }
        )

    _alias_task = asyncio.create_task(_refresh())
    _alias_task.add_done_callback(_log_task_failure("alias_refresh"))


def _schedule_summary_refresh(snapshot: dict[str, Any]) -> None:
    global _summary_task
    if _summary_task and not _summary_task.done():
        return

    async def _refresh() -> None:
        try:
            result = await asyncio.wait_for(
                summarize_dashboard_sections(snapshot),
                timeout=_AI_TIMEOUT_SEC,
            )
        except asyncio.TimeoutError:
            logger.warning("dashboard_ai_timeout", timeout_sec=_AI_TIMEOUT_SEC)
            return
        except Exception as exc:  # pragma: no cover - network variability
            logger.error("dashboard_ai_failed", error=str(exc))
            return

        _summary_cache.update(
            {
                "data": result,
                "expires": time.time() + _SUMMARY_CACHE_TTL_SEC,
            }
        )

    _summary_task = asyncio.create_task(_refresh())
    _summary_task.add_done_callback(_log_task_failure("dashboard_ai"))


def _log_task_failure(label: str):
    def _handler(task: asyncio.Task) -> None:
        try:
            task.result()
        except asyncio.CancelledError:  # pragma: no cover - noisy debug only
            logger.debug("background_task_cancelled", task_label=label)
        except Exception as exc:  # pragma: no cover - background failure
            logger.error("background_task_exception", task_label=label, error=str(exc))

    return _handler


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
            "avg_checkin_length_sec": (net.statistics or {}).get("avg_checkin_length_sec"),
            "total_talk_seconds": (net.statistics or {}).get("total_talk_seconds"),
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


async def get_global_stats(session: AsyncSession) -> dict[str, int]:
    now = datetime.now(timezone.utc)
    recent_cutoff = now - timedelta(days=7)

    ai_stages = ("ai_pass_vllm", "ai_pass_openai")

    total_callsigns = await session.scalar(select(func.count(Callsign.id))) or 0
    validated_callsigns = await session.scalar(
        select(func.count(Callsign.id)).where(Callsign.validated.is_(True))
    ) or 0
    recent_callsigns = await session.scalar(
        select(func.count(Callsign.id)).where(Callsign.last_seen >= recent_cutoff)
    ) or 0

    audio_total = await session.scalar(select(func.count(AudioFile.id))) or 0
    audio_recent = await session.scalar(
        select(func.count(AudioFile.id)).where(AudioFile.created_at >= recent_cutoff)
    ) or 0
    transcript_total = await session.scalar(select(func.count(Transcription.id))) or 0
    net_total = await session.scalar(select(func.count(NetSession.id))) or 0

    ai_passes_total = await session.scalar(
        select(func.count(ProcessingMetric.id)).where(ProcessingMetric.stage.in_(ai_stages))
    ) or 0
    ai_passes_recent = await session.scalar(
        select(func.count(ProcessingMetric.id)).where(
            ProcessingMetric.stage.in_(ai_stages),
            ProcessingMetric.timestamp >= recent_cutoff,
        )
    ) or 0

    return {
        "callsigns_total": total_callsigns,
        "callsigns_validated": validated_callsigns,
        "callsigns_recent": recent_callsigns,
        "audio_total": audio_total,
        "audio_recent": audio_recent,
        "transcripts_total": transcript_total,
        "nets_total": net_total,
        "ai_passes_total": ai_passes_total,
        "ai_passes_recent": ai_passes_recent,
    }


async def get_dashboard_payload(session: AsyncSession) -> dict:
    """Aggregate everything the landing page requires."""

    queue = await get_audio_queue_snapshot(session)
    callsigns = await get_recent_callsigns(session)
    transcriptions = await get_recent_transcriptions(session)
    nets = await get_recent_nets(session)
    trends = await get_trend_highlights(session)
    clubs = await get_club_profiles(session)
    health = await get_system_health(session)
    stats = await get_global_stats(session)

    club_names = [club["name"] for club in clubs if club.get("name")]
    alias_key = _club_alias_key(club_names)
    alias_map: dict[str, str] = _cached_alias_map(alias_key)
    if club_names:
        _schedule_alias_refresh(club_names, alias_key)
    grouped: dict[str, set[str]] = defaultdict(set)
    for alias, canonical in alias_map.items():
        grouped[canonical].add(alias)
    for club in clubs:
        canonical = alias_map.get(club["name"], club["name"])
        club["canonical_name"] = canonical
        aliases = grouped.get(canonical, set())
        club["aliases"] = sorted(a for a in aliases if a != canonical)

    ai_sections = dict(_summary_cache.get("data", _DEFAULT_AI_SECTIONS))
    summary_snapshot = {
        "queue": queue,
        "callsigns": callsigns[:10],
        "nets": nets[:10],
        "clubs": clubs[:10],
        "health": health[:5],
    }
    if (
        not _summary_cache.get("data")
        or _summary_cache.get("expires", 0) <= time.time()
        or ai_sections == _DEFAULT_AI_SECTIONS
    ):
        _schedule_summary_refresh(copy.deepcopy(summary_snapshot))

    if not ai_sections:
        ai_sections = _DEFAULT_AI_SECTIONS.copy()

    stats_cards = [
        {
            "label": "AI Passes",
            "value": stats["ai_passes_total"],
            "detail": f"{stats['ai_passes_recent']:,} in 7d",
        },
        {
            "label": "Callsigns Logged",
            "value": stats["callsigns_total"],
            "detail": f"{stats['callsigns_recent']:,} new this week",
        },
        {
            "label": "QRZ Validated",
            "value": stats["callsigns_validated"],
            "detail": "Auto-verified",
        },
        {
            "label": "Audio Files",
            "value": stats["audio_total"],
            "detail": f"{stats['transcripts_total']:,} transcripts",
        },
        {
            "label": "Nets Logged",
            "value": stats["nets_total"],
            "detail": "Organized sessions",
        },
    ]

    return {
        "queue": queue,
        "callsigns": callsigns,
        "transcriptions": transcriptions,
        "nets": nets,
        "trends": trends,
        "clubs": clubs,
        "health": health,
        "ai_summaries": ai_sections,
        "stats": stats,
        "stats_cards": stats_cards,
    }
