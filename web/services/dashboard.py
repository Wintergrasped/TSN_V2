"""Database query helpers backing the portal dashboards."""

from __future__ import annotations

import asyncio
import copy
import json
import time
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import Any

from sqlalchemy import case, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from tsn_common.logging import get_logger
from tsn_common.models.audio import AudioFile
from tsn_common.models.callsign import Callsign, CallsignLog, ValidationMethod
from tsn_common.models.club import ClubProfile
from tsn_common.models.net import NetSession
from tsn_common.models.support import ProcessingMetric, SystemHealth
from tsn_common.models.transcription import Transcription
from tsn_common.models.trend import TrendSnapshot
from web.services.ai import merge_entities, summarize_dashboard_sections
from web.services.validation import schedule_qrz_backfill


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

_PUBLIC_NODE_IDS = ("unknown", "public")


def normalize_node_scope(raw: str | None) -> str:
    if raw is None:
        return "all"
    trimmed = raw.strip()
    if not trimmed:
        return "all"
    lowered = trimmed.lower()
    if lowered == "all":
        return "all"
    if lowered == "public":
        return "public"
    return trimmed


def _node_clause(node_scope: str):
    if node_scope == "all":
        return None
    if node_scope == "public":
        return AudioFile.node_id.in_(_PUBLIC_NODE_IDS)
    return AudioFile.node_id == node_scope


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


def _flatten_ai_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (int, float)):
        if isinstance(value, int):
            return f"{value:,}"
        return f"{value:.2f}".rstrip("0").rstrip(".")
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        parts: list[str] = []
        for item in value:
            flattened = _flatten_ai_value(item)
            if flattened:
                parts.append(flattened)
        if not parts:
            return ""
        preview = ", ".join(parts[:4])
        if len(parts) > 4:
            preview = f"{preview}, ..."
        return preview
    if isinstance(value, dict):
        inner: list[str] = []
        for key, val in value.items():
            flattened = _flatten_ai_value(val)
            if flattened:
                inner.append(f"{key}: {flattened}")
        return "; ".join(inner)
    return str(value)


def _humanize_ai_summary(value: Any) -> str:
    default = "AI summary unavailable."
    if value is None:
        return default
    if isinstance(value, (dict, list)):
        flattened = _flatten_ai_value(value)
        return flattened or default
    text = str(value)
    snippet = text.strip()
    if not snippet:
        return default
    if snippet[0] in "[{":
        try:
            parsed = json.loads(snippet)
        except json.JSONDecodeError:
            return snippet
        if isinstance(parsed, dict):
            parts = []
            for key, value in parsed.items():
                flattened = _flatten_ai_value(value)
                if flattened:
                    parts.append(f"{key.title()}: {flattened}")
            return "; ".join(parts) if parts else default
        if isinstance(parsed, list):
            flattened = _flatten_ai_value(parsed)
            return flattened or default
    return snippet


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


async def get_node_selector_options(session: AsyncSession) -> list[dict[str, str]]:
    stmt = (
        select(AudioFile.node_id, func.max(AudioFile.created_at).label("last_seen"))
        .group_by(AudioFile.node_id)
        .order_by(func.max(AudioFile.created_at).desc())
    )
    result = await session.execute(stmt)
    nodes: list[str] = []
    has_public = False
    for node_id, _ in result.all():
        if not node_id:
            continue
        if node_id in _PUBLIC_NODE_IDS:
            has_public = True
        else:
            nodes.append(node_id)

    nodes = sorted(set(nodes))
    options: list[dict[str, Any]] = [
        {"value": "all", "label": "All Nodes", "disabled": False},
        {"value": "public", "label": "Public (unknown)", "disabled": not has_public},
    ]
    options.extend({"value": node, "label": f"Node {node}", "disabled": False} for node in nodes)
    return options


def _log_task_failure(label: str):
    def _handler(task: asyncio.Task) -> None:
        try:
            task.result()
        except asyncio.CancelledError:  # pragma: no cover - noisy debug only
            logger.debug("background_task_cancelled", task_label=label)
        except Exception as exc:  # pragma: no cover - background failure
            logger.error("background_task_exception", task_label=label, error=str(exc))

    return _handler


async def get_audio_queue_snapshot(session: AsyncSession, node_scope: str = "all") -> dict:
    """Return counts for each AudioFile state and totals."""

    state_counts: dict[str, int] = defaultdict(int)
    clause = _node_clause(node_scope)
    stmt = select(AudioFile.state, func.count()).group_by(AudioFile.state)
    if clause is not None:
        stmt = stmt.where(clause)
    result = await session.execute(stmt)
    for state, count in result:
        state_counts[state.value if hasattr(state, "value") else state] = count

    total_files = sum(state_counts.values())
    return {
        "total": total_files,
        "states": state_counts,
    }


async def get_recent_callsigns(
    session: AsyncSession,
    limit: int = 25,
    node_scope: str = "all",
    order_by: str = "recent",
    search: str | None = None,
) -> list[dict]:
    clause = _node_clause(node_scope)
    order_mode = (order_by or "recent").lower()
    search_pattern = f"%{search.strip().upper()}%" if search else None

    if clause is None:
        priority = case((Callsign.validation_method == ValidationMethod.QRZ, 0), else_=1)
        stmt = select(Callsign).where(Callsign.validated.is_(True))
        if search_pattern:
            stmt = stmt.where(func.upper(Callsign.callsign).like(search_pattern))
        if order_mode == "mentions":
            stmt = stmt.order_by(Callsign.seen_count.desc(), Callsign.last_seen.desc())
        else:
            stmt = stmt.order_by(priority, Callsign.last_seen.desc())
        stmt = stmt.limit(limit)
        result = await session.execute(stmt)
        records = [
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
        return records

    activity_subquery = (
        select(
            CallsignLog.callsign_id.label("callsign_id"),
            func.max(CallsignLog.detected_at).label("last_seen"),
            func.count(CallsignLog.id).label("segment_count"),
        )
        .join(Transcription, CallsignLog.transcription_id == Transcription.id)
        .join(AudioFile, Transcription.audio_file_id == AudioFile.id)
        .where(clause)
        .group_by(CallsignLog.callsign_id)
        .subquery()
    )

    stmt = (
        select(Callsign, activity_subquery.c.last_seen, activity_subquery.c.segment_count)
        .join(activity_subquery, activity_subquery.c.callsign_id == Callsign.id)
    )
    if search_pattern:
        stmt = stmt.where(func.upper(Callsign.callsign).like(search_pattern))
    if order_mode == "mentions":
        stmt = stmt.order_by(
            func.coalesce(activity_subquery.c.segment_count, 0).desc(),
            activity_subquery.c.last_seen.desc(),
        )
    else:
        stmt = stmt.order_by(activity_subquery.c.last_seen.desc())
    stmt = stmt.limit(limit)

    result = await session.execute(stmt)

    records: list[dict] = []
    for callsign, last_seen, segment_count in result.all():
        records.append(
            {
                "callsign": callsign.callsign,
                "validated": callsign.validated,
                "validation_method": callsign.validation_method.value if callsign.validation_method else None,
                "first_seen": callsign.first_seen.isoformat(),
                "last_seen": (last_seen or callsign.last_seen).isoformat() if last_seen or callsign.last_seen else None,
                "seen_count": int(segment_count or 0),
            }
        )
    return records


async def get_recent_transcriptions(
    session: AsyncSession,
    limit: int = 10,
    node_scope: str = "all",
) -> list[dict]:
    clause = _node_clause(node_scope)
    stmt = (
        select(Transcription)
        .join(AudioFile, Transcription.audio_file_id == AudioFile.id)
        .order_by(Transcription.created_at.desc())
        .limit(limit)
    )
    if clause is not None:
        stmt = stmt.where(clause)
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


async def get_recent_nets(
    session: AsyncSession,
    limit: int = 10,
    node_scope: str = "all",
) -> list[dict]:
    clause = _node_clause(node_scope)
    stmt = (
        select(NetSession)
        .join(AudioFile, NetSession.audio_file_id == AudioFile.id)
        .order_by(NetSession.start_time.desc())
        .limit(limit)
    )
    if clause is not None:
        stmt = stmt.where(clause)
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
            "formal_structure": net.formal_structure,
            "checkin_sequence": net.checkin_sequence,
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


async def get_club_profiles(
    session: AsyncSession,
    limit: int = 25,
    order_by: str = "recent",
    search: str | None = None,
) -> list[dict]:
    order_mode = (order_by or "recent").lower()
    search_pattern = f"%{search.strip().upper()}%" if search else None

    mention_stats = (
        select(
            NetSession.club_name.label("club_name"),
            func.count(NetSession.id).label("mention_count"),
            func.max(NetSession.start_time).label("last_session_at"),
        )
        .where(NetSession.club_name.isnot(None))
        .group_by(NetSession.club_name)
        .subquery()
    )

    stmt = (
        select(ClubProfile, mention_stats.c.mention_count, mention_stats.c.last_session_at)
        .outerjoin(mention_stats, mention_stats.c.club_name == ClubProfile.name)
    )
    if search_pattern:
        stmt = stmt.where(func.upper(ClubProfile.name).like(search_pattern))
    if order_mode == "mentions":
        stmt = stmt.order_by(
            func.coalesce(mention_stats.c.mention_count, 0).desc(),
            mention_stats.c.last_session_at.is_(None),
            mention_stats.c.last_session_at.desc(),
            ClubProfile.name.asc(),
        )
    else:
        stmt = stmt.order_by(
            ClubProfile.last_analyzed_at.is_(None),
            ClubProfile.last_analyzed_at.desc(),
        )
    stmt = stmt.limit(limit)

    result = await session.execute(stmt)
    clubs: list[dict] = []
    for club, mention_count, last_session_at in result.all():
        clubs.append(
            {
                "name": club.name,
                "summary": club.summary,
                "schedule": club.schedule,
                "members": club.metadata_.get("member_count") if club.metadata_ else None,
                "last_analyzed_at": club.last_analyzed_at.isoformat() if club.last_analyzed_at else None,
                "mention_count": int(mention_count or 0),
                "last_mentioned_at": last_session_at.isoformat() if last_session_at else None,
            }
        )
    return clubs


async def get_system_health(session: AsyncSession) -> list[dict]:
    result = await session.execute(select(SystemHealth))
    health_rows = [
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
    
    # Add GPU utilization statistics
    gpu_stats = await get_gpu_statistics(session)
    if gpu_stats:
        health_rows.append({
            "component": "vllm_gpu",
            "status": gpu_stats["status"],
            "last_heartbeat": gpu_stats["last_sample_at"],
            "cpu_percent": None,
            "memory_mb": None,
            "disk_free_gb": None,
            "metrics": gpu_stats,
        })
    
    # Add analysis worker idle statistics
    idle_stats = await get_idle_statistics(session)
    if idle_stats:
        health_rows.append({
            "component": "analysis_workers",
            "status": idle_stats["status"],
            "last_heartbeat": idle_stats["last_update_at"],
            "cpu_percent": None,
            "memory_mb": None,
            "disk_free_gb": None,
            "metrics": idle_stats,
        })
    
    return health_rows


async def get_gpu_statistics(session: AsyncSession) -> dict[str, Any] | None:
    """Get recent GPU utilization statistics."""
    from tsn_common.models.support import GpuUtilizationSample
    
    # Get samples from last hour
    window = datetime.now(timezone.utc) - timedelta(hours=1)
    result = await session.execute(
        select(
            func.avg(GpuUtilizationSample.utilization_pct).label("avg_util"),
            func.min(GpuUtilizationSample.utilization_pct).label("min_util"),
            func.max(GpuUtilizationSample.utilization_pct).label("max_util"),
            func.count(GpuUtilizationSample.id).label("sample_count"),
            func.max(GpuUtilizationSample.created_at).label("last_sample"),
        )
        .where(GpuUtilizationSample.created_at >= window)
    )
    row = result.first()
    
    if not row or row.sample_count == 0:
        return None
    
    avg_util = float(row.avg_util or 0)
    status = "healthy" if avg_util >= 75.0 else "degraded" if avg_util >= 50.0 else "down"
    
    return {
        "status": status,
        "avg_utilization_pct": round(avg_util, 2),
        "min_utilization_pct": round(float(row.min_util or 0), 2),
        "max_utilization_pct": round(float(row.max_util or 0), 2),
        "sample_count": int(row.sample_count),
        "last_sample_at": row.last_sample.isoformat() if row.last_sample else None,
        "window_hours": 1,
    }


async def get_idle_statistics(session: AsyncSession) -> dict[str, Any] | None:
    """Get analysis worker idle time statistics."""
    # Get idle metrics from last hour
    window = datetime.now(timezone.utc) - timedelta(hours=1)
    result = await session.execute(
        select(
            func.sum(ProcessingMetric.processing_time_ms).label("total_idle_ms"),
            func.count(ProcessingMetric.id).label("idle_event_count"),
            func.max(ProcessingMetric.timestamp).label("last_update"),
        )
        .where(
            ProcessingMetric.stage == "analysis_idle",
            ProcessingMetric.timestamp >= window,
        )
    )
    row = result.first()
    
    # Get worker statistics
    worker_result = await session.execute(
        select(ProcessingMetric.metadata_)
        .where(
            ProcessingMetric.stage.like("analysis_worker_%"),
            ProcessingMetric.timestamp >= window,
        )
        .order_by(ProcessingMetric.timestamp.desc())
        .limit(10)
    )
    worker_rows = worker_result.scalars().all()
    
    if not row or row.total_idle_ms is None:
        return None
    
    total_idle_ms = int(row.total_idle_ms or 0)
    idle_pct = 0.0
    work_pct = 0.0
    
    # Calculate aggregate work/idle percentages from worker metadata
    if worker_rows:
        work_pcts = []
        idle_pcts = []
        for metadata in worker_rows:
            if metadata:
                if "work_percent" in metadata:
                    work_pcts.append(float(metadata["work_percent"]))
                if "idle_percent" in metadata:
                    idle_pcts.append(float(metadata["idle_percent"]))
        
        if work_pcts:
            work_pct = sum(work_pcts) / len(work_pcts)
        if idle_pcts:
            idle_pct = sum(idle_pcts) / len(idle_pcts)
    
    # Status based on idle percentage
    status = "healthy" if idle_pct <= 10.0 else "degraded" if idle_pct <= 30.0 else "down"
    
    return {
        "status": status,
        "total_idle_ms": total_idle_ms,
        "total_idle_seconds": round(total_idle_ms / 1000.0, 2),
        "idle_event_count": int(row.idle_event_count or 0),
        "last_update_at": row.last_update.isoformat() if row.last_update else None,
        "avg_work_percent": round(work_pct, 2),
        "avg_idle_percent": round(idle_pct, 2),
        "window_hours": 1,
    }


async def get_global_stats(session: AsyncSession, node_scope: str = "all") -> dict[str, int]:
    now = datetime.now(timezone.utc)
    recent_cutoff = now - timedelta(days=7)

    ai_stage_filter = ProcessingMetric.stage.like("ai_pass_%")

    node_clause = _node_clause(node_scope)

    if node_clause is None:
        total_callsigns = await session.scalar(select(func.count(Callsign.id))) or 0
        validated_callsigns = await session.scalar(
            select(func.count(Callsign.id)).where(Callsign.validated.is_(True))
        ) or 0
        recent_callsigns = await session.scalar(
            select(func.count(Callsign.id)).where(Callsign.last_seen >= recent_cutoff)
        ) or 0
    else:
        callsign_activity = (
            select(CallsignLog.callsign_id.label("callsign_id"))
            .join(Transcription, CallsignLog.transcription_id == Transcription.id)
            .join(AudioFile, Transcription.audio_file_id == AudioFile.id)
            .where(node_clause)
            .group_by(CallsignLog.callsign_id)
            .subquery()
        )
        total_callsigns = await session.scalar(
            select(func.count(callsign_activity.c.callsign_id))
        ) or 0
        validated_callsigns = await session.scalar(
            select(func.count(Callsign.id))
            .join(callsign_activity, callsign_activity.c.callsign_id == Callsign.id)
            .where(Callsign.validated.is_(True))
        ) or 0
        recent_callsigns = await session.scalar(
            select(func.count(Callsign.id))
            .join(callsign_activity, callsign_activity.c.callsign_id == Callsign.id)
            .where(Callsign.last_seen >= recent_cutoff)
        ) or 0

    audio_stmt = select(func.count(AudioFile.id))
    if node_clause is not None:
        audio_stmt = audio_stmt.where(node_clause)
    audio_total = await session.scalar(audio_stmt) or 0

    audio_recent_stmt = select(func.count(AudioFile.id)).where(AudioFile.created_at >= recent_cutoff)
    if node_clause is not None:
        audio_recent_stmt = audio_recent_stmt.where(node_clause)
    audio_recent = await session.scalar(audio_recent_stmt) or 0

    transcript_stmt = select(func.count(Transcription.id)).join(AudioFile, Transcription.audio_file_id == AudioFile.id)
    if node_clause is not None:
        transcript_stmt = transcript_stmt.where(node_clause)
    transcript_total = await session.scalar(transcript_stmt) or 0

    net_stmt = select(func.count(NetSession.id)).join(AudioFile, NetSession.audio_file_id == AudioFile.id)
    if node_clause is not None:
        net_stmt = net_stmt.where(node_clause)
    net_total = await session.scalar(net_stmt) or 0

    ai_passes_total = await session.scalar(
        select(func.count(ProcessingMetric.id)).where(ai_stage_filter)
    ) or 0
    ai_passes_recent = await session.scalar(
        select(func.count(ProcessingMetric.id)).where(
            ai_stage_filter,
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


async def get_dashboard_payload(session: AsyncSession, node_scope: str | None = None) -> dict:
    """Aggregate everything the landing page requires."""

    resolved_scope = normalize_node_scope(node_scope)

    queue = await get_audio_queue_snapshot(session, resolved_scope)
    callsigns = await get_recent_callsigns(session, node_scope=resolved_scope)
    transcriptions = await get_recent_transcriptions(session, node_scope=resolved_scope)
    nets = await get_recent_nets(session, node_scope=resolved_scope)
    trends = await get_trend_highlights(session)
    clubs = await get_club_profiles(session)
    health = await get_system_health(session)
    stats = await get_global_stats(session, node_scope=resolved_scope)
    node_options = await get_node_selector_options(session)

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
    ai_sections = {key: _humanize_ai_summary(value) for key, value in ai_sections.items()}

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

    schedule_qrz_backfill()

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
        "node_filter": resolved_scope,
        "node_options": node_options,
    }
