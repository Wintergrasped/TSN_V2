"""
Analyzer - batches transcripts and drives deep analysis via vLLM.
Uses the full 32k-context window to detect nets, build profiles, surface clubs,

"""

from __future__ import annotations

import asyncio
import json
import random
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
from tsn_common.models import (
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


class TranscriptAnalyzer:
    """Analyzes batches of transcripts using vLLM with prioritization controls."""

    def __init__(self, vllm_settings: VLLMSettings, analysis_settings: AnalysisSettings):
        self.vllm_settings = vllm_settings
        self.analysis_settings = analysis_settings
        self.http_client = httpx.AsyncClient(timeout=vllm_settings.timeout_sec)
        self._openai_client: AsyncOpenAI | None = None

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

        if backlog >= threshold:
            logger.debug(
                "analysis_paused_for_transcription",
                backlog=backlog,
                threshold=threshold,
            )
            return True
        return False

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
                )
            )

        return extra_contexts

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

    async def _retry_failed_analysis(self, limit: int | None = None) -> int:
        """Move failed analysis files back into the queue when idle."""

        limit = limit or self.analysis_settings.max_batch_size
        async with get_session() as session:
            result = await session.execute(
                select(AudioFile)
                .where(AudioFile.state == AudioFileState.FAILED_ANALYSIS)
                .order_by(AudioFile.updated_at)
                .limit(limit)
                .with_for_update(skip_locked=True)
            )
            records = result.scalars().all()
            if not records:
                return 0

            for record in records:
                record.state = AudioFileState.QUEUED_ANALYSIS

            await session.flush()
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

    def _build_context_block(self, contexts: list[ContextEntry]) -> tuple[list[ContextEntry], str]:
        """Trim transcripts into the configured context budget and build prompt text."""
        if not contexts:
            return [], ""

        remaining = max(self.analysis_settings.context_char_budget, 2000)
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
            body = ctx.transcript_text.strip()
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

    async def call_vllm(self, prompt: str, *, pass_label: str = "analysis") -> tuple[str, dict[str, Any]]:
        """Call the vLLM server with automatic loopback fallback."""

        def candidate_urls() -> list[str]:
            urls = [self.vllm_settings.base_url.rstrip("/")]
            # Try loopback as a safety net when the primary IP is unreachable.
            fallback = "http://127.0.0.1:8001"
            if fallback not in urls:
                urls.append(fallback)

            def _endpoint(url: str) -> str:
                url = url.rstrip("/")
                if url.endswith("/v1"):
                    return f"{url}/chat/completions"
                return f"{url}/v1/chat/completions"

            return [_endpoint(url) for url in urls]

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

        errors: list[tuple[str, str]] = []
        for endpoint in candidate_urls():
            try:
                start_time = time.perf_counter()
                response = await self.http_client.post(endpoint, headers=headers, json=payload)
                response.raise_for_status()
                data = response.json()
                latency_ms = int((time.perf_counter() - start_time) * 1000)
                meta = {
                    "model": data.get("model"),
                    "usage": data.get("usage"),
                    "latency_ms": latency_ms,
                }
                await self._record_ai_pass_metric(
                    backend="vllm",
                    pass_label=pass_label,
                    latency_ms=latency_ms,
                    success=True,
                )
                return data["choices"][0]["message"]["content"], meta
            except httpx.ConnectError as exc:  # pragma: no cover - network heavy
                errors.append((endpoint, str(exc)))
                logger.warning(
                    "vllm_call_attempt_failed",
                    base_url=endpoint,
                    error=str(exc),
                    prompt_size=len(prompt),
                )
                continue
            except Exception as exc:  # pragma: no cover - network heavy
                logger.error("vllm_call_failed", base_url=endpoint, error=str(exc))
                raise

        await self._record_ai_pass_metric(
            backend="vllm",
            pass_label=pass_label,
            latency_ms=None,
            success=False,
            error="all_endpoints_failed",
        )
        logger.error("vllm_call_failed_all_endpoints", errors=errors, prompt_size=len(prompt))
        raise RuntimeError("All vLLM endpoints failed")

    def _build_prompt(self, context_block: str) -> str:
        """Build the instruction payload referencing the attached context."""
        return f"""You have a 32,768-token context window. Consume as much of the
provided context as needed to keep the analysis GPU saturated.

CONTEXT START
{context_block}
CONTEXT END

Tasks (minimum requirements):
1. Net Identification (Primary Pass): Decide whether the provided transcripts
    form one cohesive net or multiple nets. Only emit a net when you have
    >=3 transcript sources spanning >=180 seconds and >=2 non-NCS participants.
    If you cannot meet those constraints, request more context instead of
    emitting low-confidence fragments.
2. Net Validation (Self-check): For each proposed net, justify why the evidence
    proves it is a legitimate organized session. Include a confidence 0-1.
3. Callsign Topic History & Profiles: summarize favorite topics, recent roles,
    and notable transcript references for each callsign.
4. Club/Organization Detection & Alias Insight: identify clubs, schedules, and
    when two names are likely the same group (e.g., "PSRG" vs "Puget Sound").
5. Trend Analysis: describe topic/callsign trends or anomalies even when nets
    are quiet so GPUs stay busy.

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
      "stats": {{
         "duration_minutes": 0,
         "checkins": 0,
         "avg_checkin_length_sec": 0,
         "confidence": 0.0
      }},
      "participants": [
        {{"callsign": "W1ABC", "checkin_type": "regular|late|io|proxy|relay",
          "talk_time_sec": 45, "notes": "..."}}
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
    },
    "merge_suggestions": [
        {
            "type": "club|callsign",
            "entity": "PSRG",
            "canonical": "Puget Sound Repeater Group",
            "confidence": 0.85,
            "reason": "context evidence"
        }
    ]
}}

If you need more context to finish a net, include a top-level
"context_request" object such as
{"after_context": 4, "min_segments": 2, "reason": "Net still in progress"}.
"""

    async def _mark_audio_files(self, audio_ids: Sequence[uuid.UUID], state: AudioFileState, failed: bool = False) -> None:
        if not audio_ids:
            return
        async with get_session() as session:
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

    async def _release_unprocessed(self, audio_ids: Sequence[uuid.UUID]) -> None:
        if not audio_ids:
            return
        async with get_session() as session:
            for audio_id in audio_ids:
                record = await session.get(AudioFile, audio_id)
                if record:
                    record.state = AudioFileState.QUEUED_ANALYSIS

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
        session.add(record)
        await session.flush()
        return record

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
                )
                session.add(net_session)
                await session.flush()

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

        validator_prompt = json.dumps(
            {
                "transcript_digests": digest_parts,
                "claimed_nets": nets_outline,
            },
            ensure_ascii=False,
        )

        prompt = (
            "You are the validation pass for amateur radio net detection. "
            "Given transcript digests and proposed nets, mark each net as valid=true|false, "
            "include a confidence score 0-1, and a reason. Provide JSON of the form "
            "{\"validations\": [{\"name\": \"...\", \"valid\": true, \"confidence\": 0.9, \"reason\": \"...\"}]}"."
            f"\nINPUT:\n{validator_prompt}"
        )

        try:
            response_text, _ = await self.call_vllm(prompt, pass_label="validator")
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
                },
            )
            session.add(metric)
            await session.flush()

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
            await self._record_ai_pass_metric(
                backend="openai",
                pass_label="crosscheck",
                latency_ms=latency_ms,
                success=True,
            )
        except Exception as exc:  # pragma: no cover - network variability
            logger.warning("analysis_crosscheck_failed", error=str(exc))
            return None

        text_output = self._extract_responses_text(response)
        if not text_output:
            return None

        try:
            payload = json.loads(text_output)
        except json.JSONDecodeError:
            logger.warning("analysis_crosscheck_bad_json", preview=text_output[:120])
            return None

        usage = getattr(response, "usage", None) or {}
        tokens = usage.get("total_tokens") or usage.get("output_tokens")
        await self._record_analysis_audit(
            contexts,
            backend="openai_responses",
            pass_type="crosscheck",
            latency_ms=latency_ms,
            response_tokens=tokens,
            observations={
                "nets_detected": len(payload.get("nets") or []),
                "original_nets": len(original_payload.get("nets") or []),
            },
        )
        return payload

    async def process_one(self) -> bool:
        if await self._should_pause_for_transcription():
            await asyncio.sleep(self.analysis_settings.idle_poll_interval_sec)
            return False

        contexts = await self._reserve_batch()
        if not contexts:
            retried = await self._retry_failed_analysis()
            if retried:
                logger.info("analysis_requeued_failed", count=retried)
                return True
            refinement = await self._queue_refinement_work()
            if refinement:
                contexts = await self._reserve_batch()
            if not contexts:
                return False

        extension_attempts = 0
        payload: dict[str, Any] | None = None
        context_block = ""
        vllm_latency_ms: int | None = None
        response_tokens: int | None = None

        while True:
            included, context_block = self._build_context_block(contexts)
            if not included:
                await self._release_unprocessed([ctx.audio_id for ctx in contexts])
                logger.warning("analysis_batch_skipped_no_context")
                return False

            prompt = self._build_prompt(context_block)

            try:
                pass_label = contexts[0].pass_type if contexts else "analysis"
                response_text, response_meta = await self.call_vllm(prompt, pass_label=pass_label)
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
        payload["nets"] = await self._validate_detected_nets(included, payload.get("nets"))
        await self._persist_analysis(included, payload)
        await self._mark_audio_files([ctx.audio_id for ctx in included], AudioFileState.COMPLETE)
        await self._finalize_audio_metadata(included)
        await self._record_analysis_audit(
            included,
            backend="vllm",
            pass_type=batch_pass_type,
            latency_ms=vllm_latency_ms,
            response_tokens=response_tokens,
            observations={"context_extensions": extension_attempts},
        )

        cross_payload = await self._maybe_crosscheck_with_openai(
            included,
            context_block=context_block,
            original_payload=payload,
        )
        if cross_payload:
            await self._persist_analysis(included, cross_payload)
            logger.info(
                "analysis_crosscheck_applied",
                nets=len(cross_payload.get("nets") or []),
                callsigns=len(cross_payload.get("callsigns") or []),
            )

        logger.info(
            "analysis_batch_completed",
            batch_size=len(included),
            pass_type=batch_pass_type,
            sources=[ctx.audio_filename for ctx in included],
        )
        return True

    async def run_worker(self, worker_id: int = 0) -> None:
        logger.info("analysis_worker_started", worker_id=worker_id)
        try:
            while True:
                processed = await self.process_one()
                if not processed:
                    await asyncio.sleep(self.analysis_settings.idle_poll_interval_sec)
        except asyncio.CancelledError:
            logger.info("analysis_worker_cancelled", worker_id=worker_id)
        finally:
            await self.close()
            logger.info("analysis_worker_stopped", worker_id=worker_id)


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
