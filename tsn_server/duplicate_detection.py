"""Duplicate audio detection service - identifies and removes duplicate transcripts."""

import asyncio
import json
import time
from datetime import datetime, timedelta, timezone
from typing import Any

from sqlalchemy import select, and_
from sqlalchemy.ext.asyncio import AsyncSession

from tsn_common.config import get_settings
from tsn_common.db import get_session
from tsn_common.logging import get_logger
from tsn_common.models import AudioFile, AudioFileState, Transcription
from tsn_common.resource_lock import get_resource_lock
from tsn_server.analyzer import TranscriptAnalyzer

logger = get_logger(__name__)


class DuplicateDetectionService:
    """
    Detects and removes duplicate audio files across multiple nodes.
    
    Uses vLLM to intelligently determine if two transcripts are duplicates,
    accounting for transcription variations.
    """
    
    def __init__(self):
        settings = get_settings()
        self.vllm_settings = settings.vllm
        self.analysis_settings = settings.analysis
        
        # Create analyzer for vLLM client
        self.analyzer = TranscriptAnalyzer(
            vllm_settings=self.vllm_settings,
            analysis_settings=self.analysis_settings,
        )
        
        # Detection window - check files within this time range
        self.detection_window_minutes = 10
        
        # Min similarity for duplicate (0-100)
        self.similarity_threshold = 85
        
        logger.info(
            "duplicate_detection_service_initialized",
            window_minutes=self.detection_window_minutes,
            similarity_threshold=self.similarity_threshold,
        )
    
    async def check_for_duplicates(
        self,
        audio_file: AudioFile,
        transcript: Transcription,
        session: AsyncSession,
    ) -> tuple[bool, str | None]:
        """
        Check if this audio file is a duplicate of an existing file.
        
        Args:
            audio_file: The audio file to check
            transcript: Its transcription
            session: Database session
            
        Returns:
            (is_duplicate, duplicate_audio_file_id)
        """
        # Get recently transcribed files (excluding self)
        time_window_start = audio_file.created_at - timedelta(
            minutes=self.detection_window_minutes
        )
        time_window_end = audio_file.created_at + timedelta(
            minutes=self.detection_window_minutes
        )
        
        stmt = (
            select(AudioFile, Transcription)
            .join(Transcription, AudioFile.id == Transcription.audio_file_id)
            .where(
                and_(
                    AudioFile.id != audio_file.id,
                    AudioFile.created_at >= time_window_start,
                    AudioFile.created_at <= time_window_end,
                    AudioFile.state.in_([
                        AudioFileState.QUEUED_EXTRACTION,
                        AudioFileState.EXTRACTING,
                        AudioFileState.QUEUED_ANALYSIS,
                        AudioFileState.ANALYZING,
                        AudioFileState.ANALYZED,
                        AudioFileState.COMPLETE,
                    ]),
                )
            )
            .order_by(AudioFile.created_at)
        )
        
        result = await session.execute(stmt)
        candidates = result.all()
        
        if not candidates:
            return False, None
        
        # Check each candidate
        for candidate_audio, candidate_transcript in candidates:
            is_dup = await self._compare_transcripts(
                transcript.transcript_text,
                candidate_transcript.transcript_text,
                audio_file,
                candidate_audio,
            )
            
            if is_dup:
                # Keep the older one, mark newer as duplicate
                if audio_file.created_at < candidate_audio.created_at:
                    # This file is older, mark candidate as duplicate
                    logger.info(
                        "duplicate_detected_marking_newer",
                        kept_file=str(audio_file.id),
                        kept_node=audio_file.node_id,
                        duplicate_file=str(candidate_audio.id),
                        duplicate_node=candidate_audio.node_id,
                    )
                    await self._mark_as_duplicate(session, candidate_audio, audio_file.id)
                    # Don't return - keep checking for other duplicates
                else:
                    # Candidate is older, mark this file as duplicate
                    logger.info(
                        "duplicate_detected_marking_self",
                        kept_file=str(candidate_audio.id),
                        kept_node=candidate_audio.node_id,
                        duplicate_file=str(audio_file.id),
                        duplicate_node=audio_file.node_id,
                    )
                    return True, str(candidate_audio.id)
        
        return False, None
    
    async def _compare_transcripts(
        self,
        transcript_a: str,
        transcript_b: str,
        audio_a: AudioFile,
        audio_b: AudioFile,
    ) -> bool:
        """
        Use vLLM to determine if two transcripts are duplicates.
        
        Args:
            transcript_a: First transcript text
            transcript_b: Second transcript text
            audio_a: First audio file metadata
            audio_b: Second audio file metadata
            
        Returns:
            True if duplicates, False otherwise
        """
        # Quick check: if transcripts are identical, it's a duplicate
        if transcript_a.strip() == transcript_b.strip():
            logger.info(
                "duplicate_exact_match",
                file_a=str(audio_a.id),
                file_b=str(audio_b.id),
            )
            return True
        
        # Quick check: if length difference is too large, not a duplicate
        len_a = len(transcript_a)
        len_b = len(transcript_b)
        if len_a > 0 and len_b > 0:
            ratio = min(len_a, len_b) / max(len_a, len_b)
            if ratio < 0.7:  # Less than 70% similar in length
                return False
        
        # Use vLLM for intelligent comparison
        prompt = f"""You are analyzing two audio transcripts from different radio receiver nodes to determine if they are duplicates of the same transmission.

**TRANSCRIPT A** (Node: {audio_a.node_id}, Time: {audio_a.created_at.isoformat()}):
{transcript_a}

**TRANSCRIPT B** (Node: {audio_b.node_id}, Time: {audio_b.created_at.isoformat()}):
{transcript_b}

**TASK**: Determine if these are duplicates of the SAME transmission.

Duplicates will have:
- Same or very similar content
- Same callsigns mentioned
- Same conversation flow
- Minor differences from transcription variations (e.g., "K7ABC" vs "K7 ABC", "check-in" vs "checking")
- Similar duration

Different transmissions will have:
- Different speakers/callsigns
- Different topics
- Different timestamps (more than 2 minutes apart)
- Completely different content

Respond STRICTLY with JSON:
{{
  "is_duplicate": true|false,
  "confidence": 0-100,
  "similarity_score": 0-100,
  "reasoning": "Brief explanation",
  "key_differences": ["If not duplicate, list differences"]
}}
"""
        
        try:
            start = time.perf_counter()
            response_text, _ = await self.analyzer.call_vllm(
                prompt,
                pass_label="duplicate_detection",
                audio_file_ids=[audio_a.id, audio_b.id],
                extra_metadata={
                    "file_a_node": audio_a.node_id,
                    "file_b_node": audio_b.node_id,
                },
            )
            latency_ms = int((time.perf_counter() - start) * 1000)
            
            # Parse response
            result = json.loads(response_text)
            is_duplicate = result.get("is_duplicate", False)
            confidence = result.get("confidence", 0)
            similarity = result.get("similarity_score", 0)
            reasoning = result.get("reasoning", "")
            
            logger.info(
                "duplicate_check_vllm_result",
                file_a=str(audio_a.id),
                file_b=str(audio_b.id),
                is_duplicate=is_duplicate,
                confidence=confidence,
                similarity=similarity,
                reasoning=reasoning,
                latency_ms=latency_ms,
            )
            
            # Consider duplicate if high confidence and above threshold
            if is_duplicate and confidence >= 80 and similarity >= self.similarity_threshold:
                return True
            
            return False
            
        except Exception as e:
            logger.error(
                "duplicate_check_vllm_failed",
                error=str(e),
                file_a=str(audio_a.id),
                file_b=str(audio_b.id),
            )
            # On error, don't mark as duplicate (conservative)
            return False
    
    async def _mark_as_duplicate(
        self,
        session: AsyncSession,
        duplicate_file: AudioFile,
        original_file_id: str,
    ) -> None:
        """
        Mark a file as duplicate and prevent further processing.
        
        Args:
            session: Database session
            duplicate_file: The file to mark as duplicate
            original_file_id: ID of the original file it duplicates
        """
        # Update state to skip further processing
        duplicate_file.state = AudioFileState.COMPLETE
        
        # Add metadata
        metadata = dict(duplicate_file.metadata_ or {})
        metadata["is_duplicate"] = True
        metadata["duplicate_of"] = original_file_id
        metadata["duplicate_detected_at"] = datetime.now(timezone.utc).isoformat()
        duplicate_file.metadata_ = metadata
        
        await session.flush()
        
        logger.info(
            "duplicate_marked",
            duplicate_file=str(duplicate_file.id),
            original_file=original_file_id,
        )
    
    async def scan_for_duplicates(self) -> None:
        """
        Scan recent transcriptions for duplicates (background task).
        """
        # Check if vLLM is blocked before scanning
        resource_lock = get_resource_lock()
        if resource_lock.is_vllm_blocked():
            logger.debug(
                "duplicate_scan_skipped_vllm_blocked",
                cooldown_remaining=resource_lock.get_ingestion_cooldown_remaining(),
                pause_reason=resource_lock.get_system_pause_reason(),
            )
            return
        
        try:
            async with get_session() as session:
                # Get recently transcribed files that haven't been checked
                cutoff = datetime.now(timezone.utc) - timedelta(
                    minutes=self.detection_window_minutes * 2
                )
                
                stmt = (
                    select(AudioFile, Transcription)
                    .join(Transcription, AudioFile.id == Transcription.audio_file_id)
                    .where(
                        and_(
                            AudioFile.created_at >= cutoff,
                            AudioFile.state.in_([
                                AudioFileState.QUEUED_EXTRACTION,
                                AudioFileState.QUEUED_ANALYSIS,
                            ]),
                        )
                    )
                    .order_by(AudioFile.created_at)
                    .limit(20)  # Process max 20 per cycle
                )
                
                result = await session.execute(stmt)
                files_to_check = result.all()
                
                for audio_file, transcript in files_to_check:
                    # Check for duplicates
                    is_dup, original_id = await self.check_for_duplicates(
                        audio_file,
                        transcript,
                        session,
                    )
                    
                    if is_dup and original_id:
                        await self._mark_as_duplicate(session, audio_file, original_id)
                
                if files_to_check:
                    logger.info(
                        "duplicate_scan_completed",
                        files_checked=len(files_to_check),
                    )
                    
        except Exception as e:
            logger.error(
                "duplicate_scan_failed",
                error=str(e),
                exc_info=True,
            )
    
    async def run(self) -> None:
        """Run duplicate detection loop."""
        logger.info("duplicate_detection_service_started")
        
        try:
            while True:
                try:
                    await self.scan_for_duplicates()
                except Exception as e:
                    logger.error(
                        "duplicate_detection_cycle_failed",
                        error=str(e),
                        exc_info=True,
                    )
                
                # Wait before next scan (run every 30 seconds)
                await asyncio.sleep(30.0)
                
        except asyncio.CancelledError:
            logger.info("duplicate_detection_service_cancelled")
            raise
        except Exception as e:
            logger.error(
                "duplicate_detection_service_fatal",
                error=str(e),
                exc_info=True,
            )
            raise
