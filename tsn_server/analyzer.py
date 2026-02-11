"""
Analyzer - batches transcripts and drives deep analysis via vLLM.
Uses the full 32k-context window to detect nets, build profiles, surface clubs,

"""

from __future__ import annotations

import asyncio
import json
import math
import random
import shutil
import subprocess
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Sequence

import httpx
from openai import AsyncOpenAI
from sqlalchemy import func, select

from tsn_common.config import AnalysisSettings, VLLMSettings, get_settings
from tsn_common.db import get_session
from tsn_common.logging import get_logger
from tsn_common.resource_lock import get_resource_lock
from tsn_common.models import (
    AiRunLog,
    AnalysisAudit,
    AudioFile,
    AudioFileState,
    Callsign,
    CallsignLog,
    CallsignProfile,
    CallsignTopic,
    CheckinType,
    ClubMembership,
    ClubProfile,
    ClubRole,
    GpuUtilizationSample,
    NetParticipation,
    NetSession,
    ProcessingMetric,
    TrendSnapshot,
    Transcription,
)
from tsn_common.utils import normalize_callsign

logger = get_logger(__name__)

_MIN_NET_TRANSCRIPTS = 3
_MIN_NET_PARTICIPANTS = 3
_MIN_NON_NCS_PARTICIPANTS = 2
_MIN_NET_DURATION_SEC = 180
_MIN_NET_CONFIDENCE = 0.65
_MIN_NET_SUMMARY_LEN = 40


@dataclass
class ContextEntry:
    """Lightweight snapshot of an audio/transcription pair for prompting."""

    index: int
    audio_id: uuid.UUID
    audio_filename: str
    audio_created_at: datetime
    audio_duration_sec: float | None
    transcription_id: uuid.UUID
    transcript_text: str
    callsigns: list[str]
    snippet: str = ""
    pass_type: str = "primary"
    smoothed_text: str | None = None
    smoothed_metadata: dict[str, Any] | None = None
    smoothed_at: datetime | None = None


class TranscriptAnalyzer:
    """Analyzes batches of transcripts using vLLM with prioritization controls."""

    def __init__(self, vllm_settings: VLLMSettings, analysis_settings: AnalysisSettings):
        self.vllm_settings = vllm_settings
        self.analysis_settings = analysis_settings
        self.http_client = httpx.AsyncClient(timeout=vllm_settings.timeout_sec)
        self._openai_client: AsyncOpenAI | None = None
        self._gpu_last_sample: float = 0.0
        self._gpu_last_value: float | None = None
        self._gpu_warned_missing: bool = False
        self._gpu_last_persist: float = 0.0
        self._idle_start_time: float | None = None
        self._total_idle_ms: int = 0
        self._idle_streak_count: int = 0

        logger.info(
            "analyzer_initialized",
            vllm_url=vllm_settings.base_url,
            model=vllm_settings.model,
            analysis_workers=analysis_settings.worker_count,
            batch_size=analysis_settings.max_batch_size,
        )

    async def close(self) -> None:
        await self.http_client.aclose()
        if self._openai_client is not None:
            await self._openai_client.close()

    async def _analysis_backlog_depth(self) -> int:
        backlog_states = (
            AudioFileState.QUEUED_ANALYSIS,
            AudioFileState.ANALYZING,
        )
        async with get_session() as session:
            result = await session.execute(
                select(func.count(AudioFile.id)).where(AudioFile.state.in_(backlog_states))
            )
            return result.scalar_one()

    async def _should_pause_for_transcription(self) -> bool:
        """Return True when transcription/extraction backlog should pre-empt analysis."""
        threshold = self.analysis_settings.transcription_backlog_pause
        if threshold <= 0:
            return False

        async with get_session() as session:
            backlog_states = (
                AudioFileState.QUEUED_TRANSCRIPTION,
                AudioFileState.TRANSCRIBING,
                AudioFileState.QUEUED_EXTRACTION,
                AudioFileState.EXTRACTING,
            )
            result = await session.execute(
                select(func.count(AudioFile.id)).where(AudioFile.state.in_(backlog_states))
            )
            backlog = result.scalar_one()

        if backlog < threshold:
            return False

        analysis_backlog = await self._analysis_backlog_depth()
        priority_floor = max(0, self.analysis_settings.analysis_queue_priority_floor)
        if priority_floor > 0 and analysis_backlog >= priority_floor:
            logger.debug(
                "analysis_overriding_transcription_pause",
                transcription_backlog=backlog,
                analysis_backlog=analysis_backlog,
                priority_floor=priority_floor,
            )
            return False

        if analysis_backlog > 0:
            logger.debug(
                "analysis_continuing_despite_transcription_backlog",
                transcription_backlog=backlog,
                analysis_backlog=analysis_backlog,
            )
            return False

        logger.debug(
            "analysis_paused_for_transcription",
            backlog=backlog,
            threshold=threshold,
        )
        return True

    async def _load_callsign_strings(self, transcription_ids: Sequence[uuid.UUID]) -> dict[uuid.UUID, list[str]]:
        if not transcription_ids:
            return {}

        async with get_session() as session:
            result = await session.execute(
                select(Callsign.callsign, CallsignLog.transcription_id)
                .join(CallsignLog, CallsignLog.callsign_id == Callsign.id)
                .where(CallsignLog.transcription_id.in_(transcription_ids))
            )

            mapping: dict[uuid.UUID, list[str]] = defaultdict(list)
            for callsign, transcription_id in result.all():
                mapping[transcription_id].append(callsign)

        return mapping

    async def _reserve_batch(self) -> list[ContextEntry]:
        """Reserve the next batch of audio/transcription pairs for analysis."""
        async with get_session() as session:
            result = await session.execute(
                select(AudioFile, Transcription)
                .join(Transcription, AudioFile.id == Transcription.audio_file_id)
                .where(AudioFile.state == AudioFileState.QUEUED_ANALYSIS)
                .order_by(AudioFile.created_at)
                .limit(self.analysis_settings.max_batch_size)
                .with_for_update(skip_locked=True)
            )
            rows = result.all()

            if not rows:
                return []

            for audio_file, _ in rows:
                audio_file.state = AudioFileState.ANALYZING
            await session.flush()

        callsign_map = await self._load_callsign_strings([t.id for _, t in rows])

        contexts: list[ContextEntry] = []
        for idx, (audio_file, transcription) in enumerate(rows, start=1):
            metadata = audio_file.metadata_ or {}
            contexts.append(
                ContextEntry(
                    index=idx,
                    audio_id=audio_file.id,
                    audio_filename=audio_file.filename,
                    audio_created_at=audio_file.created_at,
                    audio_duration_sec=audio_file.duration_sec,
                    transcription_id=transcription.id,
                    transcript_text=transcription.transcript_text,
                    callsigns=callsign_map.get(transcription.id, []),
                    pass_type=metadata.get("analysis_mode", "primary"),
                    smoothed_text=transcription.smoothed_text,
                    smoothed_metadata=transcription.smoothed_metadata,
                    smoothed_at=transcription.smoothed_at,
                )
            )

        return contexts

    async def _reserve_additional_after(self, after: datetime, limit: int) -> list[ContextEntry]:
        """Reserve extra contexts that occur after the provided timestamp."""

        async with get_session() as session:
            result = await session.execute(
                select(AudioFile, Transcription)
                .join(Transcription, AudioFile.id == Transcription.audio_file_id)
                .where(
                    AudioFile.state == AudioFileState.QUEUED_ANALYSIS,
                    AudioFile.created_at > after,
                )
                .order_by(AudioFile.created_at)
                .limit(limit)
                .with_for_update(skip_locked=True)
            )
            rows = result.all()

            if not rows:
                return []

            for audio_file, _ in rows:
                audio_file.state = AudioFileState.ANALYZING
            await session.flush()

        callsign_map = await self._load_callsign_strings([t.id for _, t in rows])
        extra_contexts: list[ContextEntry] = []
        for idx, (audio_file, transcription) in enumerate(rows, start=1):
            metadata = audio_file.metadata_ or {}
            extra_contexts.append(
                ContextEntry(
                    index=idx,
                    audio_id=audio_file.id,
                    audio_filename=audio_file.filename,
                    audio_created_at=audio_file.created_at,
                    audio_duration_sec=audio_file.duration_sec,
                    transcription_id=transcription.id,
                    transcript_text=transcription.transcript_text,
                    callsigns=callsign_map.get(transcription.id, []),
                    pass_type=metadata.get("analysis_mode", "primary"),
                    smoothed_text=transcription.smoothed_text,
                    smoothed_metadata=transcription.smoothed_metadata,
                    smoothed_at=transcription.smoothed_at,
                )
            )

        return extra_contexts

    def _query_gpu_utilization_sync(self) -> float | None:
        """Execute nvidia-smi (if available) to get the current GPU utilization percent."""

        binary = shutil.which("nvidia-smi")
        if not binary:
            if not self._gpu_warned_missing:
                logger.debug("gpu_utilization_binary_missing")
                self._gpu_warned_missing = True
            return None

        try:
            result = subprocess.run(
                [binary, "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True,
                check=True,
            )
            line = (result.stdout or "").strip().splitlines()[0]
            return float(line)
        except Exception as exc:  # pragma: no cover - depends on host GPU
            logger.debug("gpu_utilization_sample_failed", error=str(exc))
            return None

    async def _sample_gpu_utilization(self) -> float | None:
        """Return cached GPU utilization, throttling expensive shell calls."""

        if not self.analysis_settings.gpu_watch_enabled:
            return None

        interval = max(1.0, self.analysis_settings.gpu_check_interval_sec)
        now = time.monotonic()
        if self._gpu_last_value is not None and (now - self._gpu_last_sample) < interval:
            return self._gpu_last_value

        loop = asyncio.get_running_loop()
        value = await loop.run_in_executor(None, self._query_gpu_utilization_sync)
        self._gpu_last_sample = now
        self._gpu_last_value = value

        if value is not None:
            persist_interval = max(1.0, self.analysis_settings.gpu_check_interval_sec)
            if (now - self._gpu_last_persist) >= persist_interval:
                await self._record_gpu_sample(value)
                self._gpu_last_persist = now

        return value

    async def _record_gpu_sample(self, value: float) -> None:
        """Persist GPU utilization samples for later capacity analysis."""

        async with get_session() as session:
            sample = GpuUtilizationSample(
                utilization_pct=value,
                sample_source="nvidia-smi",
                is_saturated=value >= self.analysis_settings.gpu_saturation_threshold_pct,
                notes=f"idle_streak={self._idle_streak_count}, total_idle_ms={self._total_idle_ms}",
            )
            session.add(sample)
            await session.flush()
        
        # Log aggressive warnings when GPU is underutilized
        if value < self.analysis_settings.gpu_low_utilization_pct:
            logger.warning(
                "gpu_underutilized_warning",
                utilization=value,
                threshold=self.analysis_settings.gpu_low_utilization_pct,
                idle_streak=self._idle_streak_count,
                total_idle_ms=self._total_idle_ms,
                aggressive_backfill=self.analysis_settings.aggressive_backfill_enabled,
            )

    async def _gpu_is_underutilized(self) -> bool:
        """True when GPU utilization is below the configured threshold."""

        if not self.analysis_settings.gpu_watch_enabled:
            return False
        value = await self._sample_gpu_utilization()
        if value is None:
            return False
        threshold = max(0.0, min(100.0, self.analysis_settings.gpu_low_utilization_pct))
        if value < threshold:
            logger.debug("gpu_underutilized", utilization=value, threshold=threshold)
            return True
        return False

    async def _extend_context_window(
        self,
        contexts: list[ContextEntry],
        request: dict[str, Any],
    ) -> list[ContextEntry]:
        """Honor an LLM context_request by appending more transcripts when available."""

        if not contexts:
            return []

        after_index_raw = request.get("after_context")
        try:
            after_index = int(after_index_raw)
        except (TypeError, ValueError):
            after_index = len(contexts)
        after_index = max(1, min(after_index, len(contexts)))
        pivot = contexts[after_index - 1]

        limit_raw = request.get("min_segments") or request.get("min_transcripts")
        try:
            limit = int(limit_raw)
        except (TypeError, ValueError):
            limit = self.analysis_settings.max_batch_size
        limit = max(1, min(limit, self.analysis_settings.max_batch_size))

        return await self._reserve_additional_after(pivot.audio_created_at, limit)

    async def _retry_failed_analysis(self) -> int:
        """Move failed analysis files back into the queue when idle."""

        minutes = max(1, self.analysis_settings.failed_analysis_rescue_minutes)
        rescue_batch = max(1, self.analysis_settings.failed_analysis_rescue_batch)
        retry_limit = max(1, self.analysis_settings.failed_analysis_retry_limit)
        cutoff = datetime.now(timezone.utc) - timedelta(minutes=minutes)

        async with get_session() as session:
            result = await session.execute(
                select(AudioFile)
                .where(
                    AudioFile.state == AudioFileState.FAILED_ANALYSIS,
                    AudioFile.updated_at <= cutoff,
                    AudioFile.retry_count < retry_limit,
                )
                .order_by(AudioFile.updated_at)
                .limit(rescue_batch)
                .with_for_update(skip_locked=True)
            )
            records = result.scalars().all()
            if not records:
                return 0

            now = datetime.now(timezone.utc)
            for record in records:
                metadata = dict(record.metadata_ or {})
                history = list(metadata.get("failed_analysis_rescues") or [])
                history.append({"rescued_at": now.isoformat()})
                metadata["failed_analysis_rescues"] = history[-10:]
                record.metadata_ = metadata
                record.state = AudioFileState.QUEUED_ANALYSIS

            await session.flush()
            logger.info(
                "failed_analysis_rescued",
                count=len(records),
                rescue_minutes=minutes,
                retry_limit=retry_limit,
            )
            return len(records)

    async def _queue_refinement_work(self) -> int:
        """Requeue completed audio for idle refinement passes."""

        window_start = datetime.now(timezone.utc) - timedelta(
            hours=max(1, self.analysis_settings.refinement_window_hours)
        )
        batch_target = max(1, self.analysis_settings.refinement_batch_size)

        async with get_session() as session:
            result = await session.execute(
                select(AudioFile, NetSession.id)
                    .outerjoin(NetSession, NetSession.audio_file_id == AudioFile.id)
                    .where(
                        AudioFile.state == AudioFileState.COMPLETE,
                        AudioFile.created_at >= window_start,
                    )
                    .order_by(AudioFile.created_at.desc())
                    .limit(batch_target * 4)
                    .with_for_update(skip_locked=True)
            )
            rows = [(row[0], row[1]) for row in result.all()]
            if not rows:
                return 0

            # Prefer audio that never spawned a net first.
            rows.sort(key=lambda pair: pair[1] is not None)

            now = datetime.now(timezone.utc)
            queued = 0
            for audio_file, net_id in rows:
                metadata = dict(audio_file.metadata_ or {})
                passes = int(metadata.get("analysis_passes", 1))
                if passes >= max(1, self.analysis_settings.max_refinement_passes):
                    continue
                if metadata.get("pending_refinement"):
                    continue

                metadata.update(
                    {
                        "analysis_mode": "refinement",
                        "pending_refinement": True,
                        "refinement_reason": "missing_net" if net_id is None else "idle_deeppass",
                        "refinement_requested_at": now.isoformat(),
                    }
                )
                audio_file.metadata_ = metadata
                audio_file.state = AudioFileState.QUEUED_ANALYSIS
                queued += 1
                if queued >= batch_target:
                    break

            if not queued:
                return 0

            await session.flush()
            logger.info("analysis_refinement_batch_enqueued", count=queued)
            return queued

    async def _queue_overdrive_work(self) -> int:
        """Force completed audio back through analysis when the GPU is idle."""

        window_hours = max(1, self.analysis_settings.overdrive_window_hours)
        batch_target = max(1, self.analysis_settings.overdrive_batch_size)
        cooldown_hours = max(1, self.analysis_settings.overdrive_cooldown_hours)
        window_start = datetime.now(timezone.utc) - timedelta(hours=window_hours)
        cooldown_delta = timedelta(hours=cooldown_hours)

        async with get_session() as session:
            result = await session.execute(
                select(AudioFile)
                .where(
                    AudioFile.state == AudioFileState.COMPLETE,
                    AudioFile.created_at >= window_start,
                )
                .order_by(AudioFile.created_at.desc())
                .limit(batch_target * 6)
                .with_for_update(skip_locked=True)
            )
            records = result.scalars().all()
            if not records:
                return 0

            now = datetime.now(timezone.utc)
            queued = 0
            for record in records:
                metadata = dict(record.metadata_ or {})
                last_run = self._parse_iso_datetime(metadata.get("overdrive_last_run"))
                if last_run and (now - last_run) < cooldown_delta:
                    continue

                history = list(metadata.get("overdrive_history") or [])
                history.append({"queued_at": now.isoformat(), "reason": "gpu_keepalive"})
                metadata.update(
                    {
                        "analysis_mode": "overdrive",
                        "pending_refinement": True,
                        "refinement_reason": "gpu_overdrive",
                        "overdrive_last_run": now.isoformat(),
                        "overdrive_history": history[-10:],
                    }
                )

                record.metadata_ = metadata
                record.state = AudioFileState.QUEUED_ANALYSIS
                queued += 1
                if queued >= batch_target:
                    break

            if not queued:
                return 0

            await session.flush()
            logger.info(
                "analysis_overdrive_batch_enqueued",
                count=queued,
                window_hours=window_hours,
                cooldown_hours=cooldown_hours,
            )
            return queued

    async def _maybe_queue_gpu_backfill(self) -> str | None:
        """Kick off additional work (refinements or profiles) when the GPU is idle."""

        if not await self._gpu_is_underutilized():
            return None

        smoothing = await self._run_idle_smoothing_pass()
        if smoothing:
            return "smoothing_backfill"

        refinement = await self._queue_refinement_work()
        if refinement:
            return "audio_requeued"

        refreshed = await self._run_profile_refresh()
        if refreshed:
            return "profile_refresh"

        overdrive = await self._queue_overdrive_work()
        if overdrive:
            return "overdrive_requeue"

        return None

    async def _aggressive_backfill_work(self) -> int:
        """Chain multiple idle work types to keep vLLM continuously loaded.
        
        Returns the number of work items queued/completed.
        """
        if not self.analysis_settings.aggressive_backfill_enabled:
            logger.debug("aggressive_backfill_disabled")
            return 0

        logger.info("aggressive_backfill_starting", chain_limit=self.analysis_settings.idle_work_chain_limit)
        
        total_work = 0
        chain_limit = max(1, self.analysis_settings.idle_work_chain_limit)

        # Priority 1: Rescue failed analysis (always good to retry)
        rescued = await self._retry_failed_analysis()
        if rescued > 0:
            total_work += rescued
            logger.debug("aggressive_backfill_rescued_failed", count=rescued)

        # Priority 2: Smoothing (quick wins, improves quality)
        for _ in range(min(3, chain_limit)):
            if await self._run_idle_smoothing_pass():
                total_work += 1
            else:
                break

        # Priority 3: Refinement passes (deeper analysis)
        queued = await self._queue_refinement_work()
        if queued > 0:
            total_work += queued
            logger.debug("aggressive_backfill_refinement", count=queued)

        # Priority 4: Profile refresh (maintains profiles)
        for _ in range(min(2, chain_limit)):
            if await self._run_profile_refresh():
                total_work += 1
            else:
                break

        # Priority 5: Overdrive re-analysis (heaviest work)
        overdrive_queued = await self._queue_overdrive_work()
        if overdrive_queued > 0:
            total_work += overdrive_queued
            logger.debug("aggressive_backfill_overdrive", count=overdrive_queued)

        # Priority 6: Generate preemptive profiles for new callsigns
        preemptive = await self._queue_preemptive_profiles()
        if preemptive > 0:
            total_work += preemptive
            logger.debug("aggressive_backfill_preemptive_profiles", count=preemptive)

        # Priority 7: Deep club analysis
        club_work = await self._queue_club_deep_analysis()
        if club_work > 0:
            total_work += club_work
            logger.debug("aggressive_backfill_club_analysis", count=club_work)

        if total_work > 0:
            logger.info(
                "aggressive_backfill_completed",
                total_work_items=total_work,
                rescued=rescued,
                refinement=queued,
                overdrive=overdrive_queued,
                preemptive=preemptive,
                club=club_work,
            )
        else:
            logger.info(
                "aggressive_backfill_no_work_found",
                checked_failed=True,
                checked_smoothing=True,
                checked_refinement=True,
                checked_profiles=True,
                checked_overdrive=True,
                checked_preemptive=True,
                checked_clubs=True,
            )

        return total_work

    async def _queue_preemptive_profiles(self) -> int:
        """Generate profiles for callsigns that have activity but no profile yet."""
        async with get_session() as session:
            # Find callsigns with 3+ segments but no profile or stale profile
            stmt = (
                select(Callsign.id, func.count(CallsignLog.id).label("segment_count"))
                .join(CallsignLog, CallsignLog.callsign_id == Callsign.id)
                .outerjoin(CallsignProfile, CallsignProfile.callsign_id == Callsign.id)
                .where(
                    CallsignProfile.id.is_(None)
                    | (CallsignProfile.last_analyzed_at < datetime.now(timezone.utc) - timedelta(hours=48))
                )
                .group_by(Callsign.id)
                .having(func.count(CallsignLog.id) >= 3)
                .limit(5)
            )
            result = await session.execute(stmt)
            callsign_ids = [row[0] for row in result.all()]

        if not callsign_ids:
            return 0

        # The profile refresh system will pick these up
        logger.debug("preemptive_profiles_identified", count=len(callsign_ids))
        return len(callsign_ids)

    async def _queue_club_deep_analysis(self) -> int:
        """Queue deep club analysis for clubs with recent activity."""
        async with get_session() as session:
            # Find clubs with nets in last 7 days that haven't been deeply analyzed recently
            window = datetime.now(timezone.utc) - timedelta(days=7)
            stmt = (
                select(ClubProfile.id)
                .join(NetSession, NetSession.club_name == ClubProfile.name)
                .where(
                    NetSession.start_time >= window,
                    (ClubProfile.last_analyzed_at.is_(None))
                    | (ClubProfile.last_analyzed_at < datetime.now(timezone.utc) - timedelta(hours=24))
                )
                .group_by(ClubProfile.id)
                .limit(3)
            )
            result = await session.execute(stmt)
            club_ids = [row[0] for row in result.all()]

        if not club_ids:
            return 0

        # Mark clubs for deep analysis
        async with get_session() as session:
            for club_id in club_ids:
                club = await session.get(ClubProfile, club_id)
                if club:
                    metadata = dict(club.metadata_ or {})
                    metadata["pending_deep_analysis"] = True
                    metadata["deep_analysis_requested_at"] = datetime.now(timezone.utc).isoformat()
                    club.metadata_ = metadata
            await session.flush()

        logger.debug("club_deep_analysis_queued", count=len(club_ids))
        return len(club_ids)

    async def _record_idle_period(self, duration_ms: int) -> None:
        """Record idle time to database for monitoring."""
        async with get_session() as session:
            metric = ProcessingMetric(
                stage="analysis_idle",
                processing_time_ms=duration_ms,
                success=True,
                error_message=None,
                timestamp=datetime.now(timezone.utc),
                metadata_={
                    "idle_streak": self._idle_streak_count,
                    "total_idle_ms": self._total_idle_ms,
                    "gpu_utilization": await self._sample_gpu_utilization(),
                },
            )
            session.add(metric)
            await session.flush()

    async def _track_idle_time(self) -> None:
        """Track when analysis enters and exits idle state."""
        if self._idle_start_time is None:
            # Entering idle
            self._idle_start_time = time.monotonic()
            self._idle_streak_count += 1
        else:
            # Already idle, update cumulative
            now = time.monotonic()
            idle_duration_ms = int((now - self._idle_start_time) * 1000)
            self._total_idle_ms += idle_duration_ms

            # Log to database every 5 seconds of idle time
            if idle_duration_ms >= 5000:
                await self._record_idle_period(idle_duration_ms)
                self._idle_start_time = now

    def _reset_idle_tracking(self) -> None:
        """Reset idle tracking when work is found."""
        if self._idle_start_time is not None:
            # Exiting idle
            now = time.monotonic()
            idle_duration_ms = int((now - self._idle_start_time) * 1000)
            self._total_idle_ms += idle_duration_ms
            self._idle_start_time = None

    def _build_context_block(
        self,
        contexts: list[ContextEntry],
        *,
        overdrive: bool = False,
    ) -> tuple[list[ContextEntry], str]:
        """Trim transcripts into the configured context budget and build prompt text."""
        if not contexts:
            return [], ""

        remaining = max(self.analysis_settings.context_char_budget, 2000)
        if overdrive:
            remaining = max(remaining, self.analysis_settings.gpu_overdrive_budget)
        included: list[ContextEntry] = []
        block_parts: list[str] = []

        for ctx in contexts:
            header = (
                f"### TRANSCRIPT {len(included) + 1}\n"
                f"Filename: {ctx.audio_filename}\n"
                f"Captured: {ctx.audio_created_at.isoformat()}\n"
                f"Duration(sec): {ctx.audio_duration_sec or 'unknown'}\n"
                f"Detected Callsigns: {', '.join(ctx.callsigns) or 'Unknown'}\n"
                "---\n"
            )
            source_text = ctx.smoothed_text or ctx.transcript_text
            body = (source_text or "").strip()
            budget_for_body = remaining - len(header) - 1

            if budget_for_body <= 0:
                break

            snippet = body if len(body) <= budget_for_body else body[:budget_for_body]
            ctx.index = len(included) + 1
            ctx.snippet = snippet
            included.append(ctx)
            block_parts.append(f"{header}{snippet}\n")
            remaining -= len(header) + len(snippet) + 1

            if remaining <= 500:  # leave room for instructions
                break

        return included, "".join(block_parts)

    def _build_smoothing_prompt(self, contexts: Sequence[ContextEntry]) -> str:
        """Create instructions for transcript smoothing batches."""

        max_chars = max(2000, self.analysis_settings.context_char_budget // 4)
        payload = []
        for ctx in contexts:
            raw_text = (ctx.transcript_text or "").strip()
            payload.append(
                {
                    "transcription_id": str(ctx.transcription_id),
                    "audio_file": ctx.audio_filename,
                    "captured": ctx.audio_created_at.isoformat(),
                    "raw_text": raw_text[:max_chars],
                }
            )

        instructions = {
            "normalize": [
                "Fix casing and punctuation",
                "Remove obvious ASR artifacts (um, ah, repeated filler)",
                "Preserve callsigns and technical terms exactly as heard",
                "Return concise but faithful conversational text",
            ],
            "output_schema": {
                "smoothed": [
                    {
                        "transcription_id": "uuid",
                        "smoothed_text": "cleaned transcript",
                        "notes": "optional observations",
                    }
                ]
            },
        }
        prompt_blob = json.dumps({"instructions": instructions, "transcripts": payload}, ensure_ascii=False)
        return (
            "You polish amateur radio transcripts so downstream AI models spend"
            " more time analyzing nets than fixing ASR mistakes."
            " Clean each transcript independently and respond ONLY with JSON"
            " matching the provided schema."
            f"\nINPUT:\n{prompt_blob}"
        )

    async def _persist_smoothed_transcripts(
        self,
        entries: list[dict[str, Any]],
        *,
        latency_ms: int | None = None,
    ) -> dict[uuid.UUID, dict[str, Any]]:
        """Update transcription rows with smoothed variants."""

        if not entries:
            return {}

        now = datetime.now(timezone.utc)
        updated: dict[uuid.UUID, dict[str, Any]] = {}
        async with get_session() as session:
            for entry in entries:
                raw_id = entry.get("transcription_id") or entry.get("id")
                cleaned = (entry.get("smoothed_text") or entry.get("text") or "").strip()
                if not raw_id or not cleaned:
                    continue
                try:
                    transcription_id = uuid.UUID(str(raw_id))
                except ValueError:
                    continue
                record = await session.get(Transcription, transcription_id)
                if not record:
                    continue

                metadata = dict(entry.get("metadata") or {})
                metadata.update(
                    {
                        "source": "vllm_smoothing",
                        "latency_ms": latency_ms,
                        "notes": entry.get("notes") or entry.get("issues"),
                    }
                )

                record.smoothed_text = cleaned
                record.smoothed_metadata = metadata
                record.smoothed_at = now
                updated[transcription_id] = {
                    "smoothed_text": cleaned,
                    "smoothed_metadata": metadata,
                    "smoothed_at": now,
                }
            await session.flush()
        return updated

    async def _smooth_context_transcripts(self, contexts: list[ContextEntry]) -> None:
        """Request smoothed transcripts via vLLM and persist results."""

        if not contexts or not self.analysis_settings.transcript_smoothing_enabled:
            return

        pending = [ctx for ctx in contexts if not ctx.smoothed_text]
        if not pending:
            return

        batch_size = max(1, self.analysis_settings.transcript_smoothing_batch_size)
        for start in range(0, len(pending), batch_size):
            chunk = pending[start : start + batch_size]
            prompt = self._build_smoothing_prompt(chunk)
            audio_ids = [ctx.audio_id for ctx in chunk]
            metadata = {
                "transcription_ids": [str(ctx.transcription_id) for ctx in chunk],
                "chunk_size": len(chunk),
            }

            try:
                response_text, response_meta = await self.call_vllm(
                    prompt,
                    pass_label="smoothing",
                    audio_file_ids=audio_ids,
                    extra_metadata=metadata,
                )
                latency_ms = (response_meta or {}).get("latency_ms")
                payload = json.loads(response_text)
            except Exception as exc:
                logger.warning("transcript_smoothing_failed", error=str(exc))
                continue

            entries = payload.get("smoothed") or payload.get("transcripts") or []
            updated = await self._persist_smoothed_transcripts(entries, latency_ms=latency_ms)
            if not updated:
                continue
            logger.info(
                "transcript_smoothing_applied",
                chunk=len(updated),
                transcription_ids=[str(key) for key in updated.keys()],
            )
            for ctx in chunk:
                result = updated.get(ctx.transcription_id)
                if not result:
                    continue
                ctx.smoothed_text = result.get("smoothed_text")
                ctx.smoothed_metadata = result.get("smoothed_metadata")
                ctx.smoothed_at = result.get("smoothed_at")

    async def _select_unsmoothed_contexts(self, limit: int) -> list[ContextEntry]:
        """Return arbitrary transcript contexts that still need smoothing."""

        async with get_session() as session:
            result = await session.execute(
                select(AudioFile, Transcription)
                .join(Transcription, AudioFile.id == Transcription.audio_file_id)
                .where(Transcription.smoothed_text.is_(None))
                .order_by(Transcription.created_at.desc())
                .limit(limit)
            )
            rows = result.all()

        if not rows:
            return []

        callsign_map = await self._load_callsign_strings([tx.id for _, tx in rows])
        contexts: list[ContextEntry] = []
        for idx, (audio_file, transcription) in enumerate(rows, start=1):
            contexts.append(
                ContextEntry(
                    index=idx,
                    audio_id=audio_file.id,
                    audio_filename=audio_file.filename,
                    audio_created_at=audio_file.created_at,
                    audio_duration_sec=audio_file.duration_sec,
                    transcription_id=transcription.id,
                    transcript_text=transcription.transcript_text,
                    callsigns=callsign_map.get(transcription.id, []),
                    pass_type="smoothing_backfill",
                    smoothed_text=transcription.smoothed_text,
                    smoothed_metadata=transcription.smoothed_metadata,
                    smoothed_at=transcription.smoothed_at,
                )
            )

        return contexts

    async def _run_idle_smoothing_pass(self) -> bool:
        """Smooth historical transcripts when the GPU would otherwise sit idle."""

        if not self.analysis_settings.transcript_smoothing_enabled:
            return False

        batch_window = max(1, self.analysis_settings.transcript_smoothing_batch_size) * 4
        contexts = await self._select_unsmoothed_contexts(batch_window)
        if not contexts:
            return False

        await self._smooth_context_transcripts(contexts)
        return True

    async def call_vllm(
        self,
        prompt: str,
        *,
        pass_label: str = "analysis",
        audio_file_ids: Sequence[uuid.UUID] | None = None,
        extra_metadata: dict[str, Any] | None = None,
    ) -> tuple[str, dict[str, Any]]:
        """Call the vLLM server with automatic loopback fallback and deep logging."""
        
        # Acquire vLLM lock (waits for transcription + cooldown)
        resource_lock = get_resource_lock()
        await resource_lock.acquire_vllm()
        
        try:
            def candidate_urls() -> list[str]:
                urls = [self.vllm_settings.base_url.rstrip("/")]
                fallback = "http://127.0.0.1:8001"
                if fallback not in urls:
                    urls.append(fallback)
                return urls

            def _endpoint(url: str) -> str:
                url = url.rstrip("/")
                if url.endswith("/v1"):
                    return f"{url}/chat/completions"
                return f"{url}/v1/chat/completions"

            payload = {
                "model": self.vllm_settings.model,
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "You are an expert amateur radio analyst. Use the entire"
                            " context to detect nets, clubs, operator behavior, and"
                            " emerging topics. Respond with strict JSON only."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0.2,
                "max_tokens": self.analysis_settings.max_response_tokens,
                "response_format": {"type": "json_object"},
            }

            headers = {
                "Authorization": f"Bearer {self.vllm_settings.api_key.get_secret_value()}",
                "Content-Type": "application/json",
            }

            gpu_snapshot = await self._sample_gpu_utilization()
            errors: list[tuple[str, str]] = []
            endpoints = candidate_urls()
            for attempt, endpoint in enumerate(endpoints, start=1):
                try:
                    start_time = time.perf_counter()
                    response = await self.http_client.post(endpoint, headers=headers, json=payload)
                    response.raise_for_status()
                    data = response.json()
                    latency_ms = int((time.perf_counter() - start_time) * 1000)
                    usage = data.get("usage") or {}
                    prompt_tokens = self._extract_usage_value(usage, ("prompt_tokens", "input_tokens"))
                    completion_tokens = self._extract_usage_value(usage, ("completion_tokens", "output_tokens"))
                    total_tokens = self._extract_usage_value(usage, ("total_tokens",)) or (
                        (prompt_tokens or 0) + (completion_tokens or 0)
                        if prompt_tokens is not None and completion_tokens is not None
                        else None
                    )
                    content = data["choices"][0]["message"]["content"]
                    meta = {
                        "model": data.get("model"),
                        "usage": usage,
                        "latency_ms": latency_ms,
                    }
                    await self._record_ai_pass_metric(
                        backend="vllm",
                        pass_label=pass_label,
                        latency_ms=latency_ms,
                        success=True,
                        prompt_tokens=prompt_tokens,
                        completion_tokens=completion_tokens,
                        total_tokens=total_tokens,
                    )
                    await self._log_ai_run(
                        backend="vllm",
                        model=data.get("model") or self.vllm_settings.model,
                        pass_label=pass_label,
                        prompt_text=prompt,
                        response_text=content,
                        success=True,
                        latency_ms=latency_ms,
                        prompt_tokens=prompt_tokens,
                        completion_tokens=completion_tokens,
                        total_tokens=total_tokens,
                        audio_file_ids=audio_file_ids,
                        metadata={
                            **(extra_metadata or {}),
                            "endpoint": endpoint,
                            "attempt": attempt,
                            "usage": usage,
                        },
                        gpu_utilization_pct=gpu_snapshot,
                    )
                    return content, meta
                except httpx.ConnectError as exc:  # pragma: no cover - network heavy
                    error_msg = str(exc)
                    errors.append((endpoint, error_msg))
                    logger.warning(
                        "vllm_call_attempt_failed",
                        base_url=endpoint,
                        error=error_msg,
                        prompt_size=len(prompt),
                    )
                    await self._log_ai_run(
                        backend="vllm",
                        model=self.vllm_settings.model,
                        pass_label=pass_label,
                        prompt_text=prompt,
                        response_text=None,
                        success=False,
                        latency_ms=None,
                        audio_file_ids=audio_file_ids,
                        metadata={
                            **(extra_metadata or {}),
                            "endpoint": endpoint,
                            "attempt": attempt,
                            "failure": "connect_error",
                        },
                        error_message=error_msg,
                        gpu_utilization_pct=gpu_snapshot,
                    )
                    continue
                except Exception as exc:  # pragma: no cover - network heavy
                    error_msg = str(exc)
                    logger.error("vllm_call_failed", base_url=endpoint, error=error_msg)
                    await self._log_ai_run(
                        backend="vllm",
                        model=self.vllm_settings.model,
                        pass_label=pass_label,
                        prompt_text=prompt,
                        response_text=None,
                        success=False,
                        latency_ms=None,
                        audio_file_ids=audio_file_ids,
                        metadata={
                            **(extra_metadata or {}),
                            "endpoint": endpoint,
                            "attempt": attempt,
                            "failure": "exception",
                        },
                        error_message=error_msg,
                        gpu_utilization_pct=gpu_snapshot,
                    )
                    await self._record_ai_pass_metric(
                        backend="vllm",
                        pass_label=pass_label,
                        latency_ms=None,
                        success=False,
                        error=error_msg,
                    )
                    raise

            await self._record_ai_pass_metric(
                backend="vllm",
                pass_label=pass_label,
                latency_ms=None,
                success=False,
                error="all_endpoints_failed",
            )
            await self._log_ai_run(
                backend="vllm",
                model=self.vllm_settings.model,
                pass_label=pass_label,
                prompt_text=prompt,
                response_text=None,
                success=False,
                latency_ms=None,
                audio_file_ids=audio_file_ids,
                metadata={**(extra_metadata or {}), "endpoints": endpoints, "failure": "exhausted"},
                error_message="all_endpoints_failed",
                gpu_utilization_pct=gpu_snapshot,
            )
            logger.error("vllm_call_failed_all_endpoints", errors=errors, prompt_size=len(prompt))
            raise RuntimeError("All vLLM endpoints failed")
        finally:
            resource_lock.release_vllm()

    def _build_prompt(self, context_block: str) -> str:
        """Build the instruction payload referencing the attached context."""
        return f"""You have a 32,768-token context window. Consume as much of the
provided context as needed to keep the analysis GPU saturated.

CONTEXT START
{context_block}
CONTEXT END

CRITICAL: Amateur radio NETS have FORMAL STRUCTURE - detect these specific markers:

1. NET DETECTION (Primary Goal):
   FORMAL NETS ALWAYS HAVE:
   a) OPENING: NCS announces net name, club, purpose ("This is the [name] net")
   b) CHECK-INS: Formal roll call with callsigns, locations, names
      - NCS says: "Any check-ins?", "Go ahead with your call", "We have [callsign]"
      - Stations say: "This is [callsign], [name], [location]"
   c) CLOSING: Formal sign-off ("This concludes the [name] net", "73s", "Thanks for checking in")
   
   EVIDENCE REQUIRED FOR NET:
   - >=3 transcript sources spanning >=180 seconds
   - >=2 non-NCS participants with CLEAR check-ins
   - Formal opening OR closing statement
   - Net Control Station (NCS) managing traffic
   
   Extract these MANDATORY fields for each net:
   - "formal_structure": {{opening/checkins/closing booleans + exact quotes}}
   - "ncs_script": [List of NCS management statements]
   - "checkin_sequence": [Ordered list of formal check-ins with quotes]

2. VALIDATION: Each proposed net must have confidence >=0.85 and justify:
   - Which transcript contains opening/closing?
   - How many formal check-ins were detected?
   - What proves this is organized (not random QSO)?

3. Callsign Profiles: Track net participation, roles, and topics.

4. Club Detection: Match net names to clubs, extract schedules.

5. Trends: Analyze patterns even when no complete nets detected.

Respond ONLY with JSON in this schema:
{{
  "nets": [
    {{
      "name": "...",
      "net_type": "rag_chew|traffic|emergency|technical|other",
      "club": "optional club name",
      "ncs": "CALLSIGN",
      "summary": "one paragraph",
      "topics": ["..."],
      "formal_structure": {{
        "has_opening": true|false,
        "has_checkins": true|false,
        "has_closing": true|false,
        "opening_text": "Exact quote from transcript",
        "closing_text": "Exact quote from transcript",
        "opening_source": transcript_number,
        "closing_source": transcript_number
      }},
      "ncs_script": ["List of NCS statements like 'This is...', 'Any check-ins?', etc"],
      "checkin_sequence": [
        {{"callsign": "W1ABC", "time": transcript_number, "statement": "exact check-in quote"}}
      ],
      "stats": {{
         "duration_minutes": 0,
         "checkins": 0,
         "formal_checkins": 0,
         "avg_checkin_length_sec": 0,
         "confidence": 0.0
      }},
      "participants": [
        {{"callsign": "W1ABC", "checkin_type": "regular|late|io|proxy|relay",
          "talk_time_sec": 45, "notes": "...", "checkin_statement": "exact quote"}}
      ],
            "sources": [transcript_numbers]
    }}
  ],
  "callsigns": [
    {{
      "callsign": "W1ABC",
      "favorite_topics": ["..."],
      "net_roles": ["ncs", "participant", "guest"],
      "open_qso_refs": ["transcript notes"],
      "profile_summary": "2-3 sentences",
      "stats": {{"net_checkins": 0, "nets_as_ncs": 0}},
      "sources": [transcript_numbers]
    }}
  ],
  "clubs": [
    {{
      "name": "Club Name",
      "summary": "...",
      "schedule": "optional schedule info",
      "ncs_callsigns": ["CALL1"],
      "members": [{{"callsign": "CALL", "role": "member|ncs|guest"}}],
      "related_nets": ["net names"],
      "sources": [transcript_numbers]
    }}
  ],
  "trends": {{
     "topics": ["..."],
     "callsigns": ["..."],
     "notes": "brief commentary",
     "window_hours": 4
    }},
    "merge_suggestions": [
        {{
            "type": "club|callsign",
            "entity": "PSRG",
            "canonical": "Puget Sound Repeater Group",
            "confidence": 0.85,
            "reason": "context evidence"
        }}
    ]
}}

If you need more context to finish a net, include a top-level
"context_request" object such as
{{"after_context": 4, "min_segments": 2, "reason": "Net still in progress"}}.
"""

    async def _mark_audio_files(self, audio_ids: Sequence[uuid.UUID], state: AudioFileState, failed: bool = False) -> None:
        if not audio_ids:
            return
        async with get_session() as session:
            # Use no_autoflush to prevent premature flush during queries
            with session.no_autoflush:
                for audio_id in audio_ids:
                    record = await session.get(AudioFile, audio_id)
                    if not record:
                        continue
                    if failed:
                        record.state = AudioFileState.FAILED_ANALYSIS
                        record.retry_count += 1
                    else:
                        record.state = AudioFileState.ANALYZED
                        record.state = AudioFileState.COMPLETE
            
            # Now flush all changes at once
            await session.flush()

    async def _release_unprocessed(self, audio_ids: Sequence[uuid.UUID]) -> None:
        if not audio_ids:
            return
        async with get_session() as session:
            # Use no_autoflush to prevent premature flush during queries
            with session.no_autoflush:
                for audio_id in audio_ids:
                    record = await session.get(AudioFile, audio_id)
                    if record:
                        record.state = AudioFileState.QUEUED_ANALYSIS
            
            # Now flush all changes at once
            await session.flush()

    async def _get_or_create_callsign(self, session, raw_callsign: str) -> Callsign | None:
        normalized = normalize_callsign(raw_callsign)
        if not normalized:
            return None

        result = await session.execute(select(Callsign).where(Callsign.callsign == normalized))
        record = result.scalar_one_or_none()
        now = datetime.now(timezone.utc)

        if record:
            record.last_seen = now
            record.seen_count += 1
            return record

        record = Callsign(
            callsign=normalized,
            validated=False,
            first_seen=now,
            last_seen=now,
            seen_count=1,
        )

    async def _select_profile_candidates(self) -> list[dict[str, Any]]:
        batch_size = max(0, self.analysis_settings.profile_batch_size)
        if batch_size <= 0:
            return []

        now = datetime.now(timezone.utc)
        refresh_cutoff = now - timedelta(hours=max(1, self.analysis_settings.profile_refresh_hours))
        window_start = now - timedelta(hours=max(1, self.analysis_settings.profile_context_hours))

        async with get_session() as session:
            base_query = (
                select(Callsign, CallsignProfile)
                .outerjoin(CallsignProfile, CallsignProfile.callsign_id == Callsign.id)
                .where(
                    Callsign.seen_count >= self.analysis_settings.profile_min_seen_count,
                    func.coalesce(
                        CallsignProfile.last_analyzed_at,
                        datetime(1970, 1, 1, tzinfo=timezone.utc),
                    )
                    < refresh_cutoff,
                )
                .order_by(Callsign.last_seen.desc())
                .limit(batch_size * 4)
                .with_for_update(skip_locked=True)
            )
            result = await session.execute(base_query)
            rows = result.all()
            if not rows:
                return []

            candidates: list[dict[str, Any]] = []
            for callsign, profile in rows:
                candidates.append(
                    {
                        "callsign": callsign.callsign,
                        "callsign_id": callsign.id,
                        "profile_id": profile.id if profile else None,
                        "last_seen": callsign.last_seen,
                        "seen_count": callsign.seen_count,
                        "metadata": dict(callsign.metadata_ or {}),
                    }
                )
                if len(candidates) >= batch_size:
                    break

            if not candidates:
                return []

            callsign_ids = [entry["callsign_id"] for entry in candidates]
            stats_map: dict[uuid.UUID, dict[str, Any]] = {
                entry["callsign_id"]: {
                    **entry,
                    "segments": 0,
                    "net_count": 0,
                    "net_transmissions": 0,
                    "ncs_count": 0,
                    "topics": [],
                    "window_start": window_start,
                }
                for entry in candidates
            }

            log_stmt = (
                select(
                    CallsignLog.callsign_id,
                    func.count().label("segments"),
                    func.max(CallsignLog.detected_at).label("last_log_seen"),
                )
                .where(
                    CallsignLog.callsign_id.in_(callsign_ids),
                    CallsignLog.detected_at >= window_start,
                )
                .group_by(CallsignLog.callsign_id)
            )
            log_rows = await session.execute(log_stmt)
            for callsign_id, segment_count, last_log in log_rows.all():
                stats = stats_map.get(callsign_id)
                if not stats:
                    continue
                stats["segments"] = int(segment_count or 0)
                stats["last_activity"] = last_log

            net_stmt = (
                select(
                    NetParticipation.callsign_id,
                    func.count(func.distinct(NetParticipation.net_session_id)).label("distinct_nets"),
                    func.sum(NetParticipation.transmission_count).label("net_tx"),
                    func.max(NetParticipation.last_seen).label("last_net"),
                )
                .where(
                    NetParticipation.callsign_id.in_(callsign_ids),
                    NetParticipation.last_seen >= window_start,
                )
                .group_by(NetParticipation.callsign_id)
            )
            net_rows = await session.execute(net_stmt)
            for callsign_id, distinct_nets, net_tx, last_net in net_rows.all():
                stats = stats_map.get(callsign_id)
                if not stats:
                    continue
                stats["net_count"] = int(distinct_nets or 0)
                stats["net_transmissions"] = int(net_tx or 0)
                stats["last_net_activity"] = last_net

            ncs_stmt = (
                select(
                    NetSession.ncs_callsign_id,
                    func.count(NetSession.id).label("ncs_count"),
                    func.max(NetSession.start_time).label("last_led"),
                )
                .where(
                    NetSession.ncs_callsign_id.in_(callsign_ids),
                    NetSession.start_time >= window_start,
                )
                .group_by(NetSession.ncs_callsign_id)
            )
            ncs_rows = await session.execute(ncs_stmt)
            for callsign_id, ncs_count, last_led in ncs_rows.all():
                stats = stats_map.get(callsign_id)
                if not stats:
                    continue
                stats["ncs_count"] = int(ncs_count or 0)
                stats["last_led"] = last_led

            topic_stmt = (
                select(
                    CallsignTopic.callsign_id,
                    CallsignTopic.topic,
                    func.count().label("topic_hits"),
                )
                .where(
                    CallsignTopic.callsign_id.in_(callsign_ids),
                    CallsignTopic.detected_at >= window_start,
                )
                .group_by(CallsignTopic.callsign_id, CallsignTopic.topic)
            )
            topic_rows = await session.execute(topic_stmt)
            topic_map: dict[uuid.UUID, list[tuple[str, int]]] = defaultdict(list)
            for callsign_id, topic, hits in topic_rows.all():
                topic_map[callsign_id].append((topic, int(hits or 0)))
            for callsign_id, stats in stats_map.items():
                ranked = sorted(topic_map.get(callsign_id, []), key=lambda item: (-item[1], item[0]))
                stats["topics"] = [topic for topic, _ in ranked[:5]]

            window_hours = max(1, self.analysis_settings.profile_context_hours)
            for stats in stats_map.values():
                segments = stats.get("segments", 0)
                net_tx = stats.get("net_transmissions", 0)
                open_segments = max(0, segments - (net_tx or 0))
                stats["bias_score"] = self._compute_bias_score(stats.get("net_count", 0), open_segments)
                stats["window_hours"] = window_hours

            ordered = [stats_map[entry["callsign_id"]] for entry in candidates]
            return ordered

    @staticmethod
    def _compute_bias_score(net_events: int, open_segments: int) -> float:
        ratio = (float(net_events) + 1.0) / (float(open_segments) + 1.0)
        try:
            return round(math.tanh(math.log(ratio)), 4)
        except (ValueError, OverflowError):
            return 0.0

    def _summarize_callsign_metadata(self, metadata: dict[str, Any]) -> str:
        if not metadata:
            return "n/a"
        interesting_keys = ("home_club", "club", "license_class", "grid", "section", "notes")
        hints = [f"{key}:{metadata[key]}" for key in interesting_keys if metadata.get(key)]
        return ", ".join(hints) if hints else "n/a"

    def _build_profile_context(self, stats: list[dict[str, Any]], window_hours: int) -> str:
        lines: list[str] = []
        for entry in stats:
            topics = ", ".join(entry.get("topics") or []) or "unknown"
            last_seen = entry.get("last_activity") or entry.get("last_seen")
            last_seen_str = last_seen.isoformat() if isinstance(last_seen, datetime) else str(last_seen or "unknown")
            metadata_notes = self._summarize_callsign_metadata(entry.get("metadata") or {})
            bias = entry.get("bias_score")
            lines.append(
                "\n".join(
                    [
                        f"### CALLSIGN {entry['callsign']}",
                        f"WindowHours: {window_hours}",
                        f"SegmentsWindow: {entry.get('segments', 0)} | DistinctNets: {entry.get('net_count', 0)} | NCSCount: {entry.get('ncs_count', 0)}",
                        f"BiasScore: {bias}",
                        f"LastActivity: {last_seen_str}",
                        f"RecentTopics: {topics}",
                        f"MetadataHints: {metadata_notes}",
                    ]
                )
            )
        return "\n\n".join(lines)

    def _build_profile_prompt(self, profile_block: str, window_hours: int) -> str:
        return (
            "You maintain extended operator profiles for amateur radio callsigns. "
            f"Use the ~{window_hours}-hour activity summaries to refresh biographies, favorite topics, "
            "and engagement metrics. Respond with STRICT JSON of the form "
            '{"profiles": [{"callsign": "...", "summary": "...", "favorite_topics": [], '
            '"activity_score": 0.0-1.0, "engagement_score": 0.0-1.0, "recommended_actions": []}]}. '
            "Reflect notable clubs, NCS history, and topical focus."
            f"\nDATA START\n{profile_block}\nDATA END"
        )

    async def _run_profile_refresh(self) -> bool:
        stats = await self._select_profile_candidates()
        if not stats:
            return False

        window_hours = max(1, self.analysis_settings.profile_context_hours)
        profile_context = self._build_profile_context(stats, window_hours)
        prompt = self._build_profile_prompt(profile_context, window_hours)

        try:
            response_text, response_meta = await self.call_vllm(
                prompt,
                pass_label="profile_refresh",
                extra_metadata={"profile_count": len(stats)},
            )
        except Exception as exc:  # pragma: no cover - vLLM connectivity
            logger.warning("profile_refresh_call_failed", error=str(exc))
            return False

        try:
            payload = json.loads(response_text)
        except json.JSONDecodeError:
            logger.warning("profile_refresh_bad_json", preview=response_text[:200])
            return False

        updated = await self._persist_profile_payload(payload)
        latency_ms = (response_meta or {}).get("latency_ms") if isinstance(response_meta, dict) else None
        logger.info(
            "profile_refresh_completed",
            requested=len(stats),
            updated=updated,
            latency_ms=latency_ms,
        )
        return True

    async def _persist_profile_payload(self, payload: dict[str, Any]) -> int:
        profiles = payload.get("profiles") if isinstance(payload, dict) else None
        if not profiles:
            return 0

        updated = 0
        async with get_session() as session:
            now = datetime.now(timezone.utc)
            for entry in profiles:
                callsign_raw = (entry or {}).get("callsign")
                normalized = normalize_callsign(callsign_raw)
                if not normalized:
                    continue

                stmt = (
                    select(Callsign, CallsignProfile)
                    .outerjoin(CallsignProfile, CallsignProfile.callsign_id == Callsign.id)
                    .where(Callsign.callsign == normalized)
                    .with_for_update(skip_locked=True)
                )
                result = await session.execute(stmt)
                row = result.first()
                if not row:
                    continue

                callsign, profile = row
                if profile is None:
                    profile = CallsignProfile(callsign_id=callsign.id)
                    session.add(profile)

                summary = (entry.get("summary") or entry.get("profile_summary") or "").strip() or None
                topics = entry.get("favorite_topics") or entry.get("primary_topics") or entry.get("topics") or []
                activity_score = self._coerce_float(entry.get("activity_score"))
                engagement_score = self._coerce_float(entry.get("engagement_score"))

                profile.profile_summary = summary
                profile.primary_topics = topics if topics else None
                profile.activity_score = activity_score
                profile.engagement_score = engagement_score
                profile.last_analyzed_at = now

                metadata = dict(profile.metadata_ or {})
                metadata["last_refresh"] = {
                    "recommended_actions": entry.get("recommended_actions"),
                    "source": "vllm_profile_refresh",
                }
                profile.metadata_ = metadata
                updated += 1

            await session.flush()
        return updated

    @staticmethod
    def _coerce_float(value: Any) -> float | None:
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _parse_iso_datetime(value: str | None) -> datetime | None:
        if not value:
            return None
        try:
            parsed = datetime.fromisoformat(value)
        except ValueError:
            return None
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed

    async def _get_or_create_profile(self, session, callsign_id: uuid.UUID) -> CallsignProfile:
        result = await session.execute(
            select(CallsignProfile).where(CallsignProfile.callsign_id == callsign_id)
        )
        profile = result.scalar_one_or_none()
        if profile:
            return profile

        profile = CallsignProfile(callsign_id=callsign_id)
        session.add(profile)
        await session.flush()
        return profile

    async def _get_or_create_club(self, session, name: str) -> ClubProfile:
        result = await session.execute(select(ClubProfile).where(ClubProfile.name == name))
        club = result.scalar_one_or_none()
        if club:
            return club

        club = ClubProfile(name=name)
        session.add(club)
        await session.flush()
        return club

    async def _get_or_create_membership(self, session, club_id: uuid.UUID, callsign_id: uuid.UUID) -> ClubMembership:
        result = await session.execute(
            select(ClubMembership).where(
                ClubMembership.club_id == club_id,
                ClubMembership.callsign_id == callsign_id,
            )
        )
        membership = result.scalar_one_or_none()
        if membership:
            return membership

        now = datetime.now(timezone.utc)
        membership = ClubMembership(
            club_id=club_id,
            callsign_id=callsign_id,
            role=ClubRole.MEMBER,
            first_seen=now,
            last_seen=now,
        )
        session.add(membership)
        await session.flush()
        return membership

    def _resolve_context_map(self, contexts: Sequence[ContextEntry]) -> dict[int, ContextEntry]:
        return {ctx.index: ctx for ctx in contexts}

    def _resolve_transcription_id(self, sources: Sequence[int] | None, context_map: dict[int, ContextEntry]) -> uuid.UUID | None:
        if sources:
            for source in sources:
                ctx = context_map.get(source)
                if ctx:
                    return ctx.transcription_id
        if context_map:
            return next(iter(context_map.values())).transcription_id
        return None

    def _effective_duration(self, ctx_list: Sequence[ContextEntry]) -> int:
        duration = 0
        for ctx in ctx_list:
            if ctx.audio_duration_sec:
                duration += int(ctx.audio_duration_sec)
            else:
                # Fallback: ~3.2 words/sec assumption
                duration += max(30, int(len(ctx.snippet.split()) / 3.2))
        return max(duration, 1)

    def _safe_checkin_type(self, raw: str | None) -> CheckinType:
        if not raw:
            return CheckinType.UNKNOWN
        normalized = raw.strip().upper()
        for member in CheckinType:
            if member.name == normalized or member.value.upper() == normalized:
                return member
        return CheckinType.UNKNOWN

    def _safe_club_role(self, raw: str | None) -> ClubRole:
        if not raw:
            return ClubRole.MEMBER
        normalized = raw.strip().upper()
        for member in ClubRole:
            if member.name == normalized or member.value.upper() == normalized:
                return member
        return ClubRole.MEMBER

    async def _persist_nets(self, nets: list[dict[str, Any]] | None, context_map: dict[int, ContextEntry]) -> None:
        if not nets:
            return

        async with get_session() as session:
            existing_cache: dict[uuid.UUID, bool] = {}
            for net in nets:
                sources_ctx = [context_map[idx] for idx in net.get("sources", []) if idx in context_map]
                if not sources_ctx:
                    continue

                participant_entries = [entry for entry in (net.get("participants") or []) if entry.get("callsign")]
                # Deduplicate participants by callsign (case-insensitive)
                deduped: dict[str, dict[str, Any]] = {}
                for entry in participant_entries:
                    callsign = entry.get("callsign", "").upper()
                    if not callsign:
                        continue
                    if callsign in deduped:
                        continue
                    deduped[callsign] = entry
                participant_entries = list(deduped.values())

                if len(sources_ctx) < _MIN_NET_TRANSCRIPTS:
                    self._log_net_skip(
                        "insufficient_transcripts",
                        net,
                        transcript_count=len(sources_ctx),
                        min_required=_MIN_NET_TRANSCRIPTS,
                    )
                    continue
                if len(participant_entries) < _MIN_NET_PARTICIPANTS:
                    self._log_net_skip(
                        "insufficient_participants",
                        net,
                        participant_count=len(participant_entries),
                        min_required=_MIN_NET_PARTICIPANTS,
                    )
                    continue

                ncs_callsign = (net.get("ncs") or "").upper()
                non_ncs = [p for p in participant_entries if p.get("callsign", "").upper() != ncs_callsign]
                if len(non_ncs) < _MIN_NON_NCS_PARTICIPANTS:
                    self._log_net_skip(
                        "insufficient_non_ncs",
                        net,
                        non_ncs=len(non_ncs),
                        min_required=_MIN_NON_NCS_PARTICIPANTS,
                    )
                    continue

                confidence = float((net.get("stats") or {}).get("confidence") or 0.0)
                if confidence < _MIN_NET_CONFIDENCE:
                    self._log_net_skip(
                        "low_confidence",
                        net,
                        confidence=confidence,
                        threshold=_MIN_NET_CONFIDENCE,
                    )
                    continue

                summary = (net.get("summary") or "").strip()
                if len(summary) < _MIN_NET_SUMMARY_LEN:
                    self._log_net_skip(
                        "summary_too_short",
                        net,
                        length=len(summary),
                        min_required=_MIN_NET_SUMMARY_LEN,
                    )
                    continue

                start_time = min(ctx.audio_created_at for ctx in sources_ctx)
                duration_sec = int(net.get("stats", {}).get("duration_minutes", 0) * 60) or self._effective_duration(sources_ctx)
                if duration_sec < _MIN_NET_DURATION_SEC:
                    self._log_net_skip(
                        "duration_too_short",
                        net,
                        duration=duration_sec,
                        min_required=_MIN_NET_DURATION_SEC,
                    )
                    continue
                end_time = start_time + timedelta(seconds=duration_sec)
                topics = net.get("topics") or None
                statistics = net.get("stats") or {}
                source_segments = [
                    {
                        "transcription_id": str(ctx.transcription_id),
                        "filename": ctx.audio_filename,
                        "captured": ctx.audio_created_at.isoformat(),
                    }
                    for ctx in sources_ctx
                ]

                primary_audio_id = sources_ctx[0].audio_id
                if primary_audio_id not in existing_cache:
                    result = await session.execute(
                        select(NetSession.id).where(NetSession.audio_file_id == primary_audio_id)
                    )
                    existing_cache[primary_audio_id] = result.scalar_one_or_none() is not None
                if existing_cache[primary_audio_id]:
                    logger.debug(
                        "analysis_skip_duplicate_net",
                        audio_file_id=str(primary_audio_id),
                        net_name=net.get("name"),
                    )
                    continue

                # Extract formal structure data
                formal_structure = net.get("formal_structure")
                ncs_script = net.get("ncs_script")
                checkin_sequence = net.get("checkin_sequence")

                net_session = NetSession(
                    net_name=net.get("name", "Unnamed Net"),
                    net_type=net.get("net_type"),
                    club_name=net.get("club"),
                    audio_file_id=sources_ctx[0].audio_id,
                    start_time=start_time,
                    end_time=end_time,
                    duration_sec=duration_sec,
                    participant_count=len(participant_entries),
                    confidence=confidence,
                    summary=summary,
                    topics=topics,
                    statistics=statistics,
                    source_segments=source_segments,
                    formal_structure=formal_structure,
                    ncs_script=ncs_script,
                    checkin_sequence=checkin_sequence,
                )
                session.add(net_session)
                await session.flush()
                
                # Log net creation to database
                await self._log_net_detection(
                    net_session_id=net_session.id,
                    net_name=net.get("name", "Unnamed Net"),
                    confidence=confidence,
                    participant_count=len(participant_entries),
                    duration_sec=duration_sec,
                    audio_file_ids=[ctx.audio_id for ctx in sources_ctx],
                    pass_type=sources_ctx[0].pass_type if sources_ctx else "unknown",
                )

                ncs_callsign = net.get("ncs")
                if ncs_callsign:
                    callsign_record = await self._get_or_create_callsign(session, ncs_callsign)
                    if callsign_record:
                        net_session.ncs_callsign_id = callsign_record.id

                start = start_time
                end = end_time
                for participant in participant_entries:
                    participant_callsign = participant.get("callsign")
                    callsign_record = await self._get_or_create_callsign(session, participant_callsign)
                    if not callsign_record:
                        continue

                    participation = NetParticipation(
                        net_session_id=net_session.id,
                        callsign_id=callsign_record.id,
                        first_seen=start,
                        last_seen=end,
                        transmission_count=max(1, int(participant.get("talk_turns", 1))),
                        estimated_talk_seconds=int(participant.get("talk_time_sec", 0)),
                        checkin_type=self._safe_checkin_type(participant.get("checkin_type")),
                    )
                    session.add(participation)

    async def _persist_callsign_profiles(self, callsigns: list[dict[str, Any]] | None, context_map: dict[int, ContextEntry]) -> None:
        if not callsigns:
            return

        now = datetime.now(timezone.utc)
        async with get_session() as session:
            for entry in callsigns:
                callsign_record = await self._get_or_create_callsign(session, entry.get("callsign", ""))
                if not callsign_record:
                    continue

                profile = await self._get_or_create_profile(session, callsign_record.id)
                if entry.get("profile_summary"):
                    profile.profile_summary = entry["profile_summary"]
                if entry.get("favorite_topics"):
                    profile.primary_topics = entry["favorite_topics"]
                stats = entry.get("stats") or {}
                profile.activity_score = float(stats.get("net_checkins", profile.activity_score or 0))
                profile.engagement_score = float(stats.get("nets_as_ncs", profile.engagement_score or 0))
                profile.last_analyzed_at = now

                metadata = dict(profile.metadata_ or {})
                metadata.update(
                    {
                        "net_roles": entry.get("net_roles"),
                        "open_qso_refs": entry.get("open_qso_refs"),
                        "stats": stats,
                    }
                )
                profile.metadata_ = metadata

                transcription_id = self._resolve_transcription_id(entry.get("sources"), context_map)
                if transcription_id and entry.get("favorite_topics"):
                    for topic in entry["favorite_topics"]:
                        session.add(
                            CallsignTopic(
                                callsign_id=callsign_record.id,
                                transcription_id=transcription_id,
                                topic=topic[:100],
                                confidence=0.9,
                                excerpt=entry.get("profile_summary"),
                                detected_at=now,
                            )
                        )

    async def _persist_clubs(self, clubs: list[dict[str, Any]] | None, context_map: dict[int, ContextEntry]) -> None:
        if not clubs:
            return

        now = datetime.now(timezone.utc)
        async with get_session() as session:
            for club_entry in clubs:
                name = club_entry.get("name")
                if not name:
                    continue
                club = await self._get_or_create_club(session, name)
                if club_entry.get("summary"):
                    club.summary = club_entry["summary"]
                if club_entry.get("schedule"):
                    club.schedule = club_entry["schedule"][:255]
                metadata = dict(club.metadata_ or {})
                metadata.update(
                    {
                        "related_nets": club_entry.get("related_nets"),
                        "ncs_callsigns": club_entry.get("ncs_callsigns"),
                    }
                )
                club.metadata_ = metadata
                club.last_analyzed_at = now

                for member in club_entry.get("members", []):
                    callsign_record = await self._get_or_create_callsign(session, member.get("callsign", ""))
                    if not callsign_record:
                        continue
                    membership = await self._get_or_create_membership(session, club.id, callsign_record.id)
                    membership.role = self._safe_club_role(member.get("role"))
                    membership.notes = member.get("notes")
                    membership.last_seen = now

    async def _persist_trends(self, trends: dict[str, Any] | None) -> None:
        if not trends:
            return

        now = datetime.now(timezone.utc)
        window_hours = int(trends.get("window_hours", 4))
        window_end = now
        window_start = now - timedelta(hours=window_hours)

        async with get_session() as session:
            result = await session.execute(
                select(TrendSnapshot).order_by(TrendSnapshot.window_end.desc()).limit(1)
            )
            latest = result.scalar_one_or_none()
            if latest:
                latest_end = latest.window_end
                if latest_end.tzinfo is None:
                    latest_end = latest_end.replace(tzinfo=timezone.utc)
                if (now - latest_end).total_seconds() < self.analysis_settings.trend_refresh_minutes * 60:
                    return

            snapshot = TrendSnapshot(
                window_start=window_start,
                window_end=window_end,
                trending_topics=trends.get("topics"),
                trending_callsigns=trends.get("callsigns"),
                notable_nets=trends.get("notable_nets"),
                notes=trends.get("notes"),
                metadata_={"source": "vllm"},
            )
            session.add(snapshot)

    async def _persist_merge_suggestions(self, suggestions: list[dict[str, Any]] | None) -> None:
        if not suggestions or not self.analysis_settings.merge_suggestion_enabled:
            return

        async with get_session() as session:
            for suggestion in suggestions:
                suggestion_type = (suggestion.get("type") or "").lower()
                entity = suggestion.get("entity") or suggestion.get("alias")
                canonical = suggestion.get("canonical") or entity
                confidence = float(suggestion.get("confidence") or 0.0)
                reason = suggestion.get("reason") or ""

                if not entity or not canonical:
                    continue

                payload = {
                    "alias": entity,
                    "canonical": canonical,
                    "confidence": confidence,
                    "reason": reason,
                }

                if suggestion_type == "club":
                    club = await self._get_or_create_club(session, canonical)
                    metadata = dict(club.metadata_ or {})
                    alias_list = list(metadata.get("alias_suggestions") or [])
                    if payload not in alias_list:
                        alias_list.append(payload)
                        metadata["alias_suggestions"] = alias_list
                        club.metadata_ = metadata
                elif suggestion_type == "callsign":
                    callsign_record = await self._get_or_create_callsign(session, canonical)
                    if not callsign_record:
                        continue
                    profile = await self._get_or_create_profile(session, callsign_record.id)
                    metadata = dict(profile.metadata_ or {})
                    alias_list = list(metadata.get("alias_suggestions") or [])
                    if payload not in alias_list:
                        alias_list.append(payload)
                        metadata["alias_suggestions"] = alias_list
                        profile.metadata_ = metadata
                else:
                    logger.debug(
                        "merge_suggestion_skipped_unknown_type",
                        suggestion_type=suggestion_type,
                        alias=entity,
                    )

    async def _validate_detected_nets(
        self,
        contexts: list[ContextEntry],
        nets: list[dict[str, Any]] | None,
    ) -> list[dict[str, Any]]:
        if not nets or not self.analysis_settings.net_validation_enabled:
            return nets or []

        digest_parts: list[str] = []
        for ctx in contexts:
            snippet = (ctx.snippet or ctx.transcript_text)[:320].replace("\n", " ")
            digest_parts.append(
                f"Transcript {ctx.index} at {ctx.audio_created_at.isoformat()} (duration {ctx.audio_duration_sec or 'unknown'}s): {snippet}"
            )

        nets_outline = []
        for net in nets:
            nets_outline.append(
                {
                    "name": net.get("name"),
                    "ncs": net.get("ncs"),
                    "participant_count": len(net.get("participants") or []),
                    "summary": net.get("summary"),
                    "confidence": (net.get("stats") or {}).get("confidence"),
                }
            )

        # Add formal structure info to validation
        for i, net in enumerate(nets):
            if i < len(nets_outline):
                formal = net.get("formal_structure", {})
                nets_outline[i]["has_formal_structure"] = all([
                    formal.get("has_opening"),
                    formal.get("has_checkins"),
                    formal.get("has_closing")
                ])
                nets_outline[i]["formal_checkins"] = len(net.get("checkin_sequence", []))

        validator_prompt = json.dumps(
            {
                "transcript_digests": digest_parts,
                "claimed_nets": nets_outline,
            },
            ensure_ascii=False,
        )

        prompt = (
            "You are the validation pass for amateur radio net detection. "
            "FORMAL NETS must have: opening statement, formal check-ins, and closing. "
            "Validate each net checking: (1) Has complete formal structure? "
            "(2) Contains formal check-in statements? (3) Confidence >=0.85? "
            "Mark as valid=true|false with confidence 0-1 and specific reason. "
            "Reject if structure is incomplete or check-ins are vague. "
            "Provide JSON: {\"validations\": [{\"name\": \"...\", \"valid\": true, \"confidence\": 0.9, \"reason\": \"...\"}]}."
            f"\nINPUT:\n{validator_prompt}"
        )

        audio_ids = [ctx.audio_id for ctx in contexts]

        try:
            response_text, _ = await self.call_vllm(
                prompt,
                pass_label="validator",
                audio_file_ids=audio_ids,
                extra_metadata={
                    "claimed_nets": len(nets or []),
                    "context_digest_count": len(contexts),
                },
            )
            result = json.loads(response_text)
        except Exception as exc:  # pragma: no cover - validation best-effort
            logger.warning("analysis_net_validation_failed", error=str(exc))
            return nets

        validations = {item.get("name"): item for item in (result.get("validations") or [])}
        threshold = max(0.0, min(1.0, self.analysis_settings.net_validation_min_confidence))
        filtered: list[dict[str, Any]] = []
        for net in nets:
            name = net.get("name")
            verdict = validations.get(name)
            if not verdict:
                self._log_net_skip("validator_missing_verdict", net)
                continue
            if not verdict.get("valid"):
                self._log_net_skip(
                    "validator_rejected",
                    net,
                    validator_reason=verdict.get("reason"),
                )
                continue
            if float(verdict.get("confidence") or 0.0) < threshold:
                self._log_net_skip(
                    "validator_low_confidence",
                    net,
                    validator_confidence=verdict.get("confidence"),
                    threshold=threshold,
                )
                continue
            filtered.append(net)

        return filtered

    def _log_net_skip(self, reason: str, net: dict[str, Any] | None, **details: Any) -> None:
        payload: dict[str, Any] = {"reason": reason}
        if net:
            payload.update(
                {
                    "net_name": net.get("name"),
                    "confidence": (net.get("stats") or {}).get("confidence"),
                    "transcript_sources": len(net.get("sources") or []),
                }
            )
        payload.update(details)
        logger.info("analysis_skip_net", **payload)

    async def _persist_analysis(self, contexts: list[ContextEntry], payload: dict[str, Any]) -> None:
        context_map = self._resolve_context_map(contexts)
        await self._persist_nets(payload.get("nets"), context_map)
        await self._persist_callsign_profiles(payload.get("callsigns"), context_map)
        await self._persist_clubs(payload.get("clubs"), context_map)
        await self._persist_trends(payload.get("trends"))
        await self._persist_merge_suggestions(payload.get("merge_suggestions"))

    async def _finalize_audio_metadata(self, contexts: list[ContextEntry]) -> None:
        if not contexts:
            return

        async with get_session() as session:
            now = datetime.now(timezone.utc)
            for ctx in contexts:
                record = await session.get(AudioFile, ctx.audio_id)
                if not record:
                    continue
                metadata = dict(record.metadata_ or {})
                history = list(metadata.get("analysis_history") or [])
                history.append({
                    "pass_type": ctx.pass_type,
                    "completed_at": now.isoformat(),
                })
                metadata["analysis_history"] = history[-10:]
                metadata["analysis_passes"] = int(metadata.get("analysis_passes", 0)) + 1
                metadata["last_analysis_pass_type"] = ctx.pass_type
                metadata["last_analysis_completed_at"] = now.isoformat()
                metadata.pop("analysis_mode", None)
                metadata.pop("pending_refinement", None)
                metadata.pop("refinement_reason", None)
                record.metadata_ = metadata
            await session.flush()

    async def _record_analysis_audit(
        self,
        contexts: list[ContextEntry],
        *,
        backend: str,
        pass_type: str,
        latency_ms: int | None = None,
        response_tokens: int | None = None,
        observations: dict[str, Any] | None = None,
    ) -> None:
        if not contexts:
            return

        obs = observations or {}
        async with get_session() as session:
            for ctx in contexts:
                session.add(
                    AnalysisAudit(
                        audio_file_id=ctx.audio_id,
                        transcription_id=ctx.transcription_id,
                        pass_type=pass_type,
                        backend=backend,
                        latency_ms=latency_ms,
                        prompt_characters=len(ctx.snippet or ctx.transcript_text),
                        response_tokens=response_tokens,
                        observations={**obs, "callsigns": ctx.callsigns},
                    )
                )
            await session.flush()

    async def _record_ai_pass_metric(
        self,
        *,
        backend: str,
        pass_label: str,
        latency_ms: int | None,
        success: bool,
        error: str | None = None,
        prompt_tokens: int | None = None,
        completion_tokens: int | None = None,
        total_tokens: int | None = None,
    ) -> None:
        async with get_session() as session:
            metric = ProcessingMetric(
                stage=f"ai_pass_{backend}",
                processing_time_ms=max(0, latency_ms or 0),
                success=success,
                error_message=error,
                timestamp=datetime.now(timezone.utc),
                metadata_={
                    "pass_label": pass_label,
                    "backend": backend,
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": total_tokens,
                },
            )
            session.add(metric)
            await session.flush()

    async def _log_ai_run(
        self,
        *,
        backend: str,
        model: str | None,
        pass_label: str,
        prompt_text: str,
        response_text: str | None,
        success: bool,
        latency_ms: int | None,
        prompt_tokens: int | None = None,
        completion_tokens: int | None = None,
        total_tokens: int | None = None,
        audio_file_ids: Sequence[uuid.UUID] | None = None,
        metadata: dict[str, Any] | None = None,
        error_message: str | None = None,
        gpu_utilization_pct: float | None = None,
    ) -> None:
        async with get_session() as session:
            entry = AiRunLog(
                backend=backend,
                model=model,
                pass_label=pass_label,
                success=success,
                error_message=error_message,
                prompt_text=prompt_text,
                response_text=response_text,
                prompt_characters=len(prompt_text or ""),
                response_characters=len(response_text) if response_text else None,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                latency_ms=latency_ms,
                gpu_utilization_pct=gpu_utilization_pct,
                audio_file_ids=[str(aid) for aid in audio_file_ids] if audio_file_ids else None,
                metadata_=metadata or {},
            )
            session.add(entry)
            await session.flush()

    async def _log_net_detection(
        self,
        *,
        net_session_id: uuid.UUID,
        net_name: str,
        confidence: float,
        participant_count: int,
        duration_sec: int,
        audio_file_ids: list[uuid.UUID],
        pass_type: str,
    ) -> None:
        """Log net detection event to processing_metrics for monitoring."""
        async with get_session() as session:
            metric = ProcessingMetric(
                stage="net_detected",
                processing_time_ms=0,  # Not applicable for detection events
                success=True,
                timestamp=datetime.now(timezone.utc),
                metadata_={
                    "net_session_id": str(net_session_id),
                    "net_name": net_name,
                    "confidence": confidence,
                    "participant_count": participant_count,
                    "duration_sec": duration_sec,
                    "audio_file_count": len(audio_file_ids),
                    "audio_file_ids": [str(aid) for aid in audio_file_ids],
                    "pass_type": pass_type,
                    "detection_source": "vllm_analysis",
                },
            )
            session.add(metric)
            await session.flush()
        
        logger.info(
            "net_detected_and_logged",
            net_session_id=str(net_session_id),
            net_name=net_name,
            confidence=confidence,
            participants=participant_count,
            duration_sec=duration_sec,
            audio_files=len(audio_file_ids),
            pass_type=pass_type,
        )

    @staticmethod
    def _extract_usage_value(usage: dict[str, Any] | None, keys: Sequence[str]) -> int | None:
        if not usage:
            return None
        for key in keys:
            value = usage.get(key)
            if value is None:
                continue
            try:
                return int(value)
            except (TypeError, ValueError):
                continue
        return None

    def _resolve_openai_client(self) -> AsyncOpenAI | None:
        api_key = (
            self.vllm_settings.openai_api_key.get_secret_value().strip()
            if self.vllm_settings.openai_api_key
            else ""
        )
        if not api_key:
            return None
        if self._openai_client is None:
            self._openai_client = AsyncOpenAI(api_key=api_key)
        return self._openai_client

    def _extract_responses_text(self, response: Any) -> str:
        if response is None:
            return ""
        chunks: list[str] = []
        output_text = getattr(response, "output_text", None)
        if output_text:
            chunks.extend(output_text)
        else:  # Fallback to iterating raw output blocks
            for item in getattr(response, "output", []) or []:
                for content in getattr(item, "content", []) or []:
                    text_value = getattr(content, "text", None)
                    if text_value:
                        chunks.append(text_value)
        return "".join(chunks).strip()

    async def _maybe_crosscheck_with_openai(
        self,
        contexts: list[ContextEntry],
        *,
        context_block: str,
        original_payload: dict[str, Any],
    ) -> dict[str, Any] | None:
        if not contexts:
            return None
        if not self.analysis_settings.crosscheck_enabled:
            return None
        if all(ctx.pass_type == "primary" for ctx in contexts):
            return None
        if random.random() > max(0.0, min(1.0, self.analysis_settings.crosscheck_probability)):
            return None

        client = self._resolve_openai_client()
        if client is None:
            logger.debug("analysis_crosscheck_skipped_no_openai")
            return None

        prompt = (
            "You are validating previously analyzed amateur radio nets. Review the context and"
            " confirm whether organized nets exist, catching any that may have been missed."
            " Respond STRICTLY with the JSON schema used earlier (nets, callsigns, clubs, trends)."
            " Focus on accuracy over creativity."
            f"\nCONTEXT START\n{context_block}\nCONTEXT END"
        )

        audio_ids = [ctx.audio_id for ctx in contexts]

        try:
            start = time.perf_counter()
            response = await client.responses.create(
                model=self.analysis_settings.openai_responses_model,
                input=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": prompt,
                            }
                        ],
                    }
                ],
                max_output_tokens=self.analysis_settings.max_response_tokens,
                temperature=0.1,
                reasoning={"effort": "medium"},
                response_format={"type": "json_object"},
            )
            latency_ms = int((time.perf_counter() - start) * 1000)
            usage = getattr(response, "usage", None) or {}
            prompt_tokens = self._extract_usage_value(usage, ("prompt_tokens", "input_tokens"))
            completion_tokens = self._extract_usage_value(usage, ("completion_tokens", "output_tokens"))
            total_tokens = self._extract_usage_value(usage, ("total_tokens",))
            await self._record_ai_pass_metric(
                backend="openai",
                pass_label="crosscheck",
                latency_ms=latency_ms,
                success=True,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
            )
        except Exception as exc:  # pragma: no cover - network variability
            error_msg = str(exc)
            logger.warning("analysis_crosscheck_failed", error=error_msg)
            await self._log_ai_run(
                backend="openai",
                model=self.analysis_settings.openai_responses_model,
                pass_label="crosscheck",
                prompt_text=prompt,
                response_text=None,
                success=False,
                latency_ms=None,
                audio_file_ids=audio_ids,
                metadata={"stage": "crosscheck"},
                error_message=error_msg,
            )
            return None

        text_output = self._extract_responses_text(response)
        if not text_output:
            return None

        try:
            payload = json.loads(text_output)
        except json.JSONDecodeError:
            logger.warning("analysis_crosscheck_bad_json", preview=text_output[:120])
            return None

        await self._record_analysis_audit(
            contexts,
            backend="openai_responses",
            pass_type="crosscheck",
            latency_ms=latency_ms,
            response_tokens=total_tokens or usage.get("output_tokens"),
            observations={
                "nets_detected": len(payload.get("nets") or []),
                "original_nets": len(original_payload.get("nets") or []),
            },
        )
        await self._log_ai_run(
            backend="openai",
            model=self.analysis_settings.openai_responses_model,
            pass_label="crosscheck",
            prompt_text=prompt,
            response_text=text_output,
            success=True,
            latency_ms=latency_ms,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            audio_file_ids=audio_ids,
            metadata={
                "stage": "crosscheck",
                "usage": usage,
            },
        )
        return payload

    async def process_one(self) -> bool:
        if await self._should_pause_for_transcription():
            await asyncio.sleep(self.analysis_settings.idle_poll_interval_sec)
            await self._track_idle_time()
            return False

        contexts = await self._reserve_batch()
        if not contexts:
            # Entering idle mode - track it
            await self._track_idle_time()

            # Aggressive backfill mode: chain multiple work types
            if self.analysis_settings.aggressive_backfill_enabled:
                work_completed = await self._aggressive_backfill_work()
                if work_completed > 0:
                    # Work was queued, try to grab it immediately
                    contexts = await self._reserve_batch()
                    if contexts:
                        self._reset_idle_tracking()
                    else:
                        # Work queued but not yet available, that's OK
                        return True
                else:
                    # No work available anywhere - true idle
                    return False
            else:
                # Legacy single-task idle mode
                retried = await self._retry_failed_analysis()
                if retried:
                    logger.info("analysis_requeued_failed", count=retried)
                    return True
                refinement = await self._queue_refinement_work()
                if refinement:
                    contexts = await self._reserve_batch()
                if not contexts:
                    gpu_job = await self._maybe_queue_gpu_backfill()
                    if gpu_job == "profile_refresh":
                        return True
                    if gpu_job in {"audio_requeued", "overdrive_requeue"}:
                        contexts = await self._reserve_batch()
                    if not contexts:
                        return False
        else:
            # Found work immediately - reset idle tracking
            self._reset_idle_tracking()

        if contexts and self.analysis_settings.transcript_smoothing_enabled:
            await self._smooth_context_transcripts(contexts)

        extension_attempts = 0
        payload: dict[str, Any] | None = None
        context_block = ""
        vllm_latency_ms: int | None = None
        response_tokens: int | None = None

        while True:
            overdrive_context = await self._gpu_is_underutilized()
            included, context_block = self._build_context_block(
                contexts,
                overdrive=overdrive_context,
            )
            if not included:
                await self._release_unprocessed([ctx.audio_id for ctx in contexts])
                logger.warning("analysis_batch_skipped_no_context")
                return False

            prompt = self._build_prompt(context_block)

            try:
                pass_label = contexts[0].pass_type if contexts else "analysis"
                response_text, response_meta = await self.call_vllm(
                    prompt,
                    pass_label=pass_label,
                    audio_file_ids=[ctx.audio_id for ctx in included],
                    extra_metadata={
                        "context_count": len(included),
                        "extension_attempts": extension_attempts,
                    },
                )
                vllm_latency_ms = (response_meta or {}).get("latency_ms")
                payload = json.loads(response_text)
                usage = (response_meta or {}).get("usage") or {}
                response_tokens = usage.get("total_tokens") or usage.get("completion_tokens")
            except Exception as exc:
                unused_audio_ids = {ctx.audio_id for ctx in contexts} - {ctx.audio_id for ctx in included}
                if unused_audio_ids:
                    await self._release_unprocessed(list(unused_audio_ids))
                logger.error("analysis_batch_failed", error=str(exc), exc_info=True)
                await self._mark_audio_files([ctx.audio_id for ctx in included], AudioFileState.FAILED_ANALYSIS, failed=True)
                return False

            context_request = payload.get("context_request") if isinstance(payload, dict) else None
            if context_request and extension_attempts < self.analysis_settings.max_context_extensions:
                extra_contexts = await self._extend_context_window(contexts, context_request)
                if extra_contexts:
                    contexts.extend(extra_contexts)
                    contexts.sort(key=lambda ctx: ctx.audio_created_at)
                    extension_attempts += 1
                    logger.info(
                        "analysis_extending_context",
                        added=len(extra_contexts),
                        attempts=extension_attempts,
                        request=context_request,
                    )
                    continue
                logger.warning("analysis_context_request_unfulfilled", request=context_request)
            break

        unused_audio_ids = {ctx.audio_id for ctx in contexts} - {ctx.audio_id for ctx in included}
        if unused_audio_ids:
            await self._release_unprocessed(list(unused_audio_ids))

        batch_pass_type = included[0].pass_type if included else "primary"
        payload = payload or {}
        
        # Count nets before validation
        nets_proposed = len(payload.get("nets") or [])
        
        payload["nets"] = await self._validate_detected_nets(included, payload.get("nets"))
        
        # Count nets after validation
        nets_validated = len(payload.get("nets") or [])
        nets_rejected = nets_proposed - nets_validated
        
        await self._persist_analysis(included, payload)
        await self._mark_audio_files([ctx.audio_id for ctx in included], AudioFileState.COMPLETE)
        await self._finalize_audio_metadata(included)
        await self._record_analysis_audit(
            included,
            backend="vllm",
            pass_type=batch_pass_type,
            latency_ms=vllm_latency_ms,
            response_tokens=response_tokens,
            observations={
                "context_extensions": extension_attempts,
                "nets_proposed": nets_proposed,
                "nets_validated": nets_validated,
                "nets_rejected": nets_rejected,
                "callsigns_extracted": len(payload.get("callsigns") or []),
                "clubs_identified": len(payload.get("clubs") or []),
            },
        )

        cross_payload = await self._maybe_crosscheck_with_openai(
            included,
            context_block=context_block,
            original_payload=payload,
        )
        if cross_payload:
            cross_nets = len(cross_payload.get("nets") or [])
            cross_callsigns = len(cross_payload.get("callsigns") or [])
            await self._persist_analysis(included, cross_payload)
            logger.info(
                "analysis_crosscheck_applied",
                nets=cross_nets,
                callsigns=cross_callsigns,
            )

        logger.info(
            "analysis_batch_completed",
            batch_size=len(included),
            pass_type=batch_pass_type,
            nets_detected=nets_validated,
            nets_proposed=nets_proposed,
            nets_rejected=nets_rejected,
            callsigns=len(payload.get("callsigns") or []),
            clubs=len(payload.get("clubs") or []),
            sources=[ctx.audio_filename for ctx in included],
            latency_ms=vllm_latency_ms,
            gpu_utilization=await self._sample_gpu_utilization(),
        )
        return True

    async def run_worker(self, worker_id: int = 0) -> None:
        logger.info("analysis_worker_started", worker_id=worker_id)
        
        # Immediately log configuration
        logger.info(
            "analysis_worker_config",
            worker_id=worker_id,
            idle_poll_sec=self.analysis_settings.idle_poll_interval_sec,
            aggressive_backfill=self.analysis_settings.aggressive_backfill_enabled,
            chain_limit=self.analysis_settings.idle_work_chain_limit,
            gpu_watch=self.analysis_settings.gpu_watch_enabled,
        )
        
        # Worker statistics
        iterations = 0
        work_cycles = 0
        idle_cycles = 0
        start_time = time.monotonic()
        last_stats_log = start_time
        
        try:
            logger.info("analysis_worker_entering_main_loop", worker_id=worker_id)
            while True:
                iterations += 1
                logger.debug("analysis_worker_starting_iteration", worker_id=worker_id, iteration=iterations)
                
                try:
                    processed = await self.process_one()
                except Exception as process_exc:
                    logger.error(
                        "analysis_worker_process_one_exception",
                        worker_id=worker_id,
                        iteration=iterations,
                        error=str(process_exc),
                        exc_info=True,
                    )
                    raise
                
                # Log after EVERY iteration to see if loop continues
                logger.info(
                    "analysis_worker_iteration_complete",
                    worker_id=worker_id,
                    iteration=iterations,
                    processed=processed,
                )
                
                if processed:
                    work_cycles += 1
                else:
                    idle_cycles += 1
                    await asyncio.sleep(self.analysis_settings.idle_poll_interval_sec)
                
                logger.debug("analysis_worker_about_to_loop", worker_id=worker_id, iteration=iterations)
                
                # Log worker statistics every 5 minutes
                now = time.monotonic()
                if now - last_stats_log >= 300:
                    elapsed_sec = now - start_time
                    work_pct = (work_cycles / iterations * 100) if iterations > 0 else 0
                    idle_pct = (idle_cycles / iterations * 100) if iterations > 0 else 0
                    
                    logger.info(
                        "analysis_worker_statistics",
                        worker_id=worker_id,
                        iterations=iterations,
                        work_cycles=work_cycles,
                        idle_cycles=idle_cycles,
                        work_percent=round(work_pct, 2),
                        idle_percent=round(idle_pct, 2),
                        elapsed_seconds=int(elapsed_sec),
                        total_idle_ms=self._total_idle_ms,
                        idle_streaks=self._idle_streak_count,
                        gpu_utilization=await self._sample_gpu_utilization(),
                    )
                    
                    # Persist to database
                    await self._record_worker_statistics(
                        worker_id=worker_id,
                        iterations=iterations,
                        work_cycles=work_cycles,
                        idle_cycles=idle_cycles,
                        elapsed_sec=int(elapsed_sec),
                    )
                    
                    last_stats_log = now
                
                # Log every 100 iterations for visibility during startup
                if iterations % 100 == 0:
                    logger.debug(
                        "analysis_worker_heartbeat",
                        worker_id=worker_id,
                        iterations=iterations,
                        work_cycles=work_cycles,
                        idle_cycles=idle_cycles,
                    )
                    
        except asyncio.CancelledError:
            logger.info("analysis_worker_cancelled", worker_id=worker_id)
        finally:
            # Final statistics
            elapsed_sec = time.monotonic() - start_time
            work_pct = (work_cycles / iterations * 100) if iterations > 0 else 0
            idle_pct = (idle_cycles / iterations * 100) if iterations > 0 else 0
            
            logger.info(
                "analysis_worker_stopped",
                worker_id=worker_id,
                final_iterations=iterations,
                final_work_cycles=work_cycles,
                final_idle_cycles=idle_cycles,
                final_work_percent=round(work_pct, 2),
                final_idle_percent=round(idle_pct, 2),
                total_elapsed_seconds=int(elapsed_sec),
                total_idle_ms=self._total_idle_ms,
            )
            
            await self.close()

    async def _record_worker_statistics(
        self,
        worker_id: int,
        iterations: int,
        work_cycles: int,
        idle_cycles: int,
        elapsed_sec: int,
    ) -> None:
        """Persist worker statistics to database for long-term analysis."""
        async with get_session() as session:
            metric = ProcessingMetric(
                stage=f"analysis_worker_{worker_id}",
                processing_time_ms=elapsed_sec * 1000,
                success=True,
                timestamp=datetime.now(timezone.utc),
                metadata_={
                    "worker_id": worker_id,
                    "iterations": iterations,
                    "work_cycles": work_cycles,
                    "idle_cycles": idle_cycles,
                    "work_percent": round((work_cycles / iterations * 100) if iterations > 0 else 0, 2),
                    "idle_percent": round((idle_cycles / iterations * 100) if iterations > 0 else 0, 2),
                    "total_idle_ms": self._total_idle_ms,
                    "idle_streaks": self._idle_streak_count,
                    "gpu_utilization": await self._sample_gpu_utilization(),
                    "aggressive_backfill": self.analysis_settings.aggressive_backfill_enabled,
                },
            )
            session.add(metric)
            await session.flush()


async def main() -> None:
    from tsn_common import setup_logging

    settings = get_settings()
    setup_logging(settings.logging)

    analyzer = TranscriptAnalyzer(settings.vllm, settings.analysis)
    workers = [
        asyncio.create_task(analyzer.run_worker(i))
        for i in range(settings.analysis.worker_count)
    ]

    await asyncio.gather(*workers)


if __name__ == "__main__":
    asyncio.run(main())
