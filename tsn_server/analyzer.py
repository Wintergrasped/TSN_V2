"""
Analyzer - batches transcripts and drives deep analysis via vLLM.
Uses the full 32k-context window to detect nets, build profiles, surface clubs,

"""

from __future__ import annotations

import asyncio
import json
import uuid
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Sequence

import httpx
from sqlalchemy import func, select

from tsn_common.config import AnalysisSettings, VLLMSettings, get_settings
from tsn_common.db import get_session
from tsn_common.logging import get_logger
from tsn_common.models import (
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
    TrendSnapshot,
    Transcription,
)
from tsn_common.utils import normalize_callsign

logger = get_logger(__name__)


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


class TranscriptAnalyzer:
    """Analyzes batches of transcripts using vLLM with prioritization controls."""

    def __init__(self, vllm_settings: VLLMSettings, analysis_settings: AnalysisSettings):
        self.vllm_settings = vllm_settings
        self.analysis_settings = analysis_settings
        self.http_client = httpx.AsyncClient(timeout=vllm_settings.timeout_sec)

        logger.info(
            "analyzer_initialized",
            vllm_url=vllm_settings.base_url,
            model=vllm_settings.model,
            analysis_workers=analysis_settings.worker_count,
            batch_size=analysis_settings.max_batch_size,
        )

    async def close(self) -> None:
        await self.http_client.aclose()

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
                )
            )

        return contexts

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

    async def call_vllm(self, prompt: str) -> str:
        """Call the vLLM server with the constructed prompt."""
        try:
            response = await self.http_client.post(
                f"{self.vllm_settings.base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.vllm_settings.api_key.get_secret_value()}",
                    "Content-Type": "application/json",
                },
                json={
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
                },
            )
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]["content"]
        except Exception as exc:  # pragma: no cover - network heavy
            logger.error("vllm_call_failed", error=str(exc), prompt_size=len(prompt))
            raise

    def _build_prompt(self, context_block: str) -> str:
        """Build the instruction payload referencing the attached context."""
        return f"""You have a 32,768-token context window. Consume as much of the
provided context as needed to keep the analysis GPU saturated.

CONTEXT START
{context_block}
CONTEXT END

Tasks (minimum requirements):
1. Net Identification: find organized nets, determine NCS, club ties, roster
   (normal, late, short time/IO, relay), net type, duration, topics, stats.
2. Callsign Topic History & Profiles: for every callsign mentioned, summarize
   favorite topics, recent behavior, whether they were NCS, late, or open-QSO,
   and surface transcripts where notable traffic occurred.
3. Club/Organization Detection: identify clubs or groups, their NCS ops, net
   schedules, and member roles.
4. Trend Analysis: report topic/callsign trends, anomalies, and noteworthy
   changes to keep the silicon hot even when live nets are quiet.

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
  }}
}}
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
            for net in nets:
                sources_ctx = [context_map[idx] for idx in net.get("sources", []) if idx in context_map]
                if not sources_ctx:
                    continue

                start_time = min(ctx.audio_created_at for ctx in sources_ctx)
                duration_sec = int(net.get("stats", {}).get("duration_minutes", 0) * 60) or self._effective_duration(sources_ctx)
                end_time = start_time + timedelta(seconds=duration_sec)
                topics = net.get("topics") or None
                summary = net.get("summary")
                statistics = net.get("stats") or {}
                source_segments = [
                    {
                        "transcription_id": str(ctx.transcription_id),
                        "filename": ctx.audio_filename,
                        "captured": ctx.audio_created_at.isoformat(),
                    }
                    for ctx in sources_ctx
                ]

                net_session = NetSession(
                    net_name=net.get("name", "Unnamed Net"),
                    net_type=net.get("net_type"),
                    club_name=net.get("club"),
                    audio_file_id=sources_ctx[0].audio_id,
                    start_time=start_time,
                    end_time=end_time,
                    duration_sec=duration_sec,
                    participant_count=len(net.get("participants", [])),
                    confidence=float(net.get("stats", {}).get("confidence", 0.75)),
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
                for participant in net.get("participants", []):
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
            if latest and (now - latest.window_end).total_seconds() < self.analysis_settings.trend_refresh_minutes * 60:
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

    async def _persist_analysis(self, contexts: list[ContextEntry], payload: dict[str, Any]) -> None:
        context_map = self._resolve_context_map(contexts)
        await self._persist_nets(payload.get("nets"), context_map)
        await self._persist_callsign_profiles(payload.get("callsigns"), context_map)
        await self._persist_clubs(payload.get("clubs"), context_map)
        await self._persist_trends(payload.get("trends"))

    async def process_one(self) -> bool:
        if await self._should_pause_for_transcription():
            await asyncio.sleep(self.analysis_settings.idle_poll_interval_sec)
            return False

        contexts = await self._reserve_batch()
        if not contexts:
            return False

        included, context_block = self._build_context_block(contexts)
        if not included:
            await self._release_unprocessed([ctx.audio_id for ctx in contexts])
            logger.warning("analysis_batch_skipped_no_context")
            return False

        unused_audio_ids = {ctx.audio_id for ctx in contexts} - {ctx.audio_id for ctx in included}
        if unused_audio_ids:
            await self._release_unprocessed(list(unused_audio_ids))

        prompt = self._build_prompt(context_block)

        try:
            response = await self.call_vllm(prompt)
            payload = json.loads(response)
            await self._persist_analysis(included, payload)
            await self._mark_audio_files([ctx.audio_id for ctx in included], AudioFileState.COMPLETE)

            logger.info(
                "analysis_batch_completed",
                batch_size=len(included),
                sources=[ctx.audio_filename for ctx in included],
            )
            return True
        except Exception as exc:
            logger.error("analysis_batch_failed", error=str(exc), exc_info=True)
            await self._mark_audio_files([ctx.audio_id for ctx in included], AudioFileState.FAILED_ANALYSIS, failed=True)
            return False

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
