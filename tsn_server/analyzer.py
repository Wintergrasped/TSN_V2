"""
Analyzer - extracts topics, detects nets, and generates profiles using vLLM.
"""

import asyncio
import json
from datetime import datetime, timezone
from typing import Optional

import httpx
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from tsn_common.config import VLLMSettings, get_settings
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
    NetParticipation,
    NetSession,
    Transcription,
)

logger = get_logger(__name__)


class TranscriptAnalyzer:
    """
    Analyzes transcripts using vLLM for topics, nets, and profiles.
    """

    def __init__(self, settings: VLLMSettings):
        self.settings = settings
        self.http_client = httpx.AsyncClient(timeout=settings.timeout_sec)
        
        logger.info(
            "analyzer_initialized",
            vllm_url=settings.base_url,
            model=settings.model,
        )

    async def close(self) -> None:
        """Close HTTP client."""
        await self.http_client.aclose()

    async def call_vllm(self, prompt: str, max_tokens: int = 1000) -> str:
        """
        Call vLLM API.
        
        Args:
            prompt: User prompt
            max_tokens: Maximum response tokens
            
        Returns:
            Response content
        """
        try:
            response = await self.http_client.post(
                f"{self.settings.base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.settings.api_key.get_secret_value()}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self.settings.model,
                    "messages": [
                        {
                            "role": "system",
                            "content": "You are an expert at analyzing amateur radio conversations.",
                        },
                        {"role": "user", "content": prompt},
                    ],
                    "temperature": 0.3,
                    "max_tokens": max_tokens,
                    "response_format": {"type": "json_object"},
                },
            )
            
            response.raise_for_status()
            data = response.json()
            
            return data["choices"][0]["message"]["content"]
            
        except Exception as e:
            logger.error(
                "vllm_call_failed",
                error=str(e),
                prompt_length=len(prompt),
            )
            raise

    async def extract_topics(
        self,
        transcription: Transcription,
        callsigns: list[Callsign],
    ) -> list[str]:
        """
        Extract discussion topics from transcript.
        
        Args:
            transcription: Transcription record
            callsigns: List of callsigns mentioned
            
        Returns:
            List of topic strings
        """
        callsign_list = ", ".join(cs.callsign for cs in callsigns[:10])
        
        prompt = f"""Analyze this amateur radio conversation transcript and extract the main topics discussed.

Callsigns present: {callsign_list}

Transcript:
{transcription.transcript_text[:2000]}

Extract 3-5 main topics. Be specific but concise.

Respond with JSON only:
{{"topics": ["topic1", "topic2", "topic3"]}}"""

        try:
            response = await self.call_vllm(prompt, max_tokens=500)
            result = json.loads(response)
            
            topics = result.get("topics", [])
            
            logger.info(
                "topics_extracted",
                transcription_id=str(transcription.id),
                topic_count=len(topics),
                topics=topics,
            )
            
            return topics
            
        except Exception as e:
            logger.error(
                "topic_extraction_failed",
                transcription_id=str(transcription.id),
                error=str(e),
            )
            return []

    async def detect_net(
        self,
        transcription: Transcription,
        callsigns: list[Callsign],
    ) -> Optional[dict]:
        """
        Detect if transcript is a net session.
        
        Args:
            transcription: Transcription record
            callsigns: List of callsigns mentioned
            
        Returns:
            Net information dict or None
        """
        # Heuristic check first
        text_lower = transcription.transcript_text.lower()
        net_keywords = [
            "check in",
            "checking in",
            "net control",
            "this is the",
            "traffic",
            "rag chew",
        ]
        
        has_keywords = any(kw in text_lower for kw in net_keywords)
        has_multiple_callsigns = len(callsigns) >= 3
        
        if not (has_keywords and has_multiple_callsigns):
            return None
        
        # AI confirmation
        callsign_list = ", ".join(cs.callsign for cs in callsigns[:20])
        
        prompt = f"""Analyze if this amateur radio conversation is a net session (organized check-in).

Callsigns present: {callsign_list}

Transcript:
{transcription.transcript_text[:2000]}

Determine:
1. Is this a net session? (true/false)
2. If yes, what is the net name?
3. Who is the NCS (Net Control Station)?
4. What type of net? (traffic, rag_chew, emergency, technical, etc.)

Respond with JSON only:
{{
  "is_net": true,
  "net_name": "Evening Rag Chew Net",
  "ncs_callsign": "W1ABC",
  "net_type": "rag_chew"
}}"""

        try:
            response = await self.call_vllm(prompt, max_tokens=300)
            result = json.loads(response)
            
            if result.get("is_net"):
                logger.info(
                    "net_detected",
                    transcription_id=str(transcription.id),
                    net_name=result.get("net_name"),
                    ncs=result.get("ncs_callsign"),
                )
                return result
            
            return None
            
        except Exception as e:
            logger.error(
                "net_detection_failed",
                transcription_id=str(transcription.id),
                error=str(e),
            )
            return None

    async def extract_checkins(
        self,
        transcription: Transcription,
        callsigns: list[Callsign],
        ncs_callsign: Optional[str],
    ) -> list[tuple[str, CheckinType]]:
        """
        Extract check-in roster with types.
        
        Args:
            transcription: Transcription record
            callsigns: List of callsigns mentioned
            ncs_callsign: NCS callsign to exclude
            
        Returns:
            List of (callsign, checkin_type) tuples
        """
        callsign_list = ", ".join(cs.callsign for cs in callsigns[:20])
        
        prompt = f"""Extract the check-in roster from this net transcript.

Callsigns present: {callsign_list}
NCS: {ncs_callsign or "unknown"}

Transcript:
{transcription.transcript_text[:2000]}

For each participant (excluding NCS), determine check-in type:
- regular: Standard check-in
- relay: Relay check-in through another station
- late: Late check-in after roster call

Respond with JSON only:
{{
  "checkins": [
    {{"callsign": "W1ABC", "type": "regular"}},
    {{"callsign": "K2XYZ", "type": "late"}}
  ]
}}"""

        try:
            response = await self.call_vllm(prompt, max_tokens=800)
            result = json.loads(response)
            
            checkins = []
            for item in result.get("checkins", []):
                callsign = item.get("callsign", "").upper()
                checkin_type_str = item.get("type", "regular")
                
                try:
                    checkin_type = CheckinType[checkin_type_str.upper()]
                except KeyError:
                    checkin_type = CheckinType.REGULAR
                
                if callsign and callsign != ncs_callsign:
                    checkins.append((callsign, checkin_type))
            
            logger.info(
                "checkins_extracted",
                transcription_id=str(transcription.id),
                checkin_count=len(checkins),
            )
            
            return checkins
            
        except Exception as e:
            logger.error(
                "checkin_extraction_failed",
                transcription_id=str(transcription.id),
                error=str(e),
            )
            return []

    async def generate_profile_summary(
        self,
        callsign: Callsign,
        recent_transcripts: list[str],
    ) -> Optional[str]:
        """
        Generate AI summary for callsign profile.
        
        Args:
            callsign: Callsign record
            recent_transcripts: List of recent transcript excerpts
            
        Returns:
            Profile summary text
        """
        combined_text = "\n\n".join(recent_transcripts[:5])
        
        prompt = f"""Generate a brief profile summary for amateur radio operator {callsign.callsign}.

Recent conversation excerpts:
{combined_text[:3000]}

Create a 2-3 sentence summary covering:
- Topics of interest
- Operating style/personality
- Technical expertise areas

Respond with JSON only:
{{"summary": "Brief profile summary here"}}"""

        try:
            response = await self.call_vllm(prompt, max_tokens=300)
            result = json.loads(response)
            
            summary = result.get("summary", "")
            
            logger.info(
                "profile_generated",
                callsign=callsign.callsign,
                summary_length=len(summary),
            )
            
            return summary
            
        except Exception as e:
            logger.error(
                "profile_generation_failed",
                callsign=callsign.callsign,
                error=str(e),
            )
            return None

    async def analyze_transcription(
        self,
        audio_file: AudioFile,
        transcription: Transcription,
    ) -> None:
        """
        Perform complete analysis on a transcription.
        
        Args:
            audio_file: AudioFile record
            transcription: Transcription record
        """
        async with get_session() as session:
            # Get callsigns from logs
            result = await session.execute(
                select(Callsign)
                .join(CallsignLog, CallsignLog.callsign_id == Callsign.id)
                .where(CallsignLog.transcription_id == transcription.id)
                .distinct()
            )
            callsigns = list(result.scalars().all())
            
            logger.info(
                "analysis_started",
                audio_file_id=str(audio_file.id),
                callsign_count=len(callsigns),
            )
        
        # Extract topics
        topics = await self.extract_topics(transcription, callsigns)
        
        # Store topics
        async with get_session() as session:
            for topic in topics:
                for callsign in callsigns:
                    topic_entry = CallsignTopic(
                        callsign_id=callsign.id,
                        transcription_id=transcription.id,
                        topic=topic,
                        detected_at=datetime.now(timezone.utc),
                    )
                    session.add(topic_entry)
        
        # Detect net
        net_info = await self.detect_net(transcription, callsigns)
        
        if net_info:
            async with get_session() as session:
                # Create net session
                net_session = NetSession(
                    audio_file_id=audio_file.id,
                    net_name=net_info.get("net_name", "Unknown Net"),
                    net_type=net_info.get("net_type", "rag_chew"),
                    start_time=audio_file.created_at,
                )
                session.add(net_session)
                await session.flush()
                
                # Get NCS callsign
                ncs_callsign_str = net_info.get("ncs_callsign", "").upper()
                ncs_callsign = None
                
                if ncs_callsign_str:
                    result = await session.execute(
                        select(Callsign).where(Callsign.callsign == ncs_callsign_str)
                    )
                    ncs_callsign = result.scalar_one_or_none()
                
                if ncs_callsign:
                    net_session.ncs_callsign_id = ncs_callsign.id
                
                # Extract checkins
                checkins = await self.extract_checkins(
                    transcription,
                    callsigns,
                    ncs_callsign_str,
                )
                
                # Store participations
                for callsign_str, checkin_type in checkins:
                    result = await session.execute(
                        select(Callsign).where(Callsign.callsign == callsign_str)
                    )
                    callsign = result.scalar_one_or_none()
                    
                    if callsign:
                        participation = NetParticipation(
                            net_session_id=net_session.id,
                            callsign_id=callsign.id,
                            checkin_type=checkin_type,
                            checkin_time=audio_file.created_at,
                        )
                        session.add(participation)
        
        # Update profiles for active callsigns
        for callsign in callsigns[:10]:  # Limit to top 10
            async with get_session() as session:
                # Get recent transcripts for this callsign
                result = await session.execute(
                    select(Transcription.transcript_text)
                    .join(CallsignLog, CallsignLog.transcription_id == Transcription.id)
                    .where(CallsignLog.callsign_id == callsign.id)
                    .order_by(CallsignLog.detected_at.desc())
                    .limit(5)
                )
                recent_texts = [row[0][:500] for row in result.all()]
                
                if recent_texts:
                    summary = await self.generate_profile_summary(callsign, recent_texts)
                    
                    if summary:
                        # Get or create profile
                        result = await session.execute(
                            select(CallsignProfile).where(
                                CallsignProfile.callsign_id == callsign.id
                            )
                        )
                        profile = result.scalar_one_or_none()
                        
                        if profile:
                            profile.ai_summary = summary
                            profile.last_updated = datetime.now(timezone.utc)
                        else:
                            profile = CallsignProfile(
                                callsign_id=callsign.id,
                                ai_summary=summary,
                            )
                            session.add(profile)

    async def get_next_audio_file(
        self,
        session: AsyncSession,
    ) -> Optional[tuple[AudioFile, Transcription]]:
        """Get next file to analyze."""
        result = await session.execute(
            select(AudioFile, Transcription)
            .join(Transcription, AudioFile.id == Transcription.audio_file_id)
            .where(AudioFile.state == AudioFileState.QUEUED_ANALYSIS)
            .order_by(AudioFile.created_at)
            .limit(1)
            .with_for_update(skip_locked=True)
        )
        
        row = result.first()
        return row if row else None

    async def process_one(self) -> bool:
        """Process one file from queue."""
        async with get_session() as session:
            result = await self.get_next_audio_file(session)
            
            if result is None:
                return False
            
            audio_file, transcription = result
            
            # Update state
            audio_file.state = AudioFileState.ANALYZING
            await session.flush()
        
        # Analyze
        try:
            await self.analyze_transcription(audio_file, transcription)
            
            # Update state
            async with get_session() as session:
                audio_file = await session.get(AudioFile, audio_file.id)
                audio_file.state = AudioFileState.ANALYZED
                audio_file.state = AudioFileState.COMPLETE
                
                logger.info(
                    "analysis_completed",
                    audio_file_id=str(audio_file.id),
                )
        except Exception as e:
            logger.error(
                "analysis_failed",
                audio_file_id=str(audio_file.id),
                error=str(e),
                exc_info=True,
            )
            
            async with get_session() as session:
                audio_file = await session.get(AudioFile, audio_file.id)
                audio_file.state = AudioFileState.FAILED_ANALYSIS
                audio_file.retry_count += 1
        
        return True

    async def run_worker(self, worker_id: int = 0) -> None:
        """Run analysis worker loop."""
        logger.info("analysis_worker_started", worker_id=worker_id)
        
        try:
            while True:
                try:
                    processed = await self.process_one()
                    
                    if not processed:
                        await asyncio.sleep(2.0)
                        
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.error(
                        "analysis_worker_error",
                        worker_id=worker_id,
                        error=str(e),
                        exc_info=True,
                    )
                    await asyncio.sleep(5.0)
        finally:
            await self.close()
            logger.info("analysis_worker_stopped", worker_id=worker_id)


async def main() -> None:
    """Main entry point."""
    from tsn_common import setup_logging
    
    settings = get_settings()
    setup_logging(settings.logging)
    
    analyzer = TranscriptAnalyzer(settings.vllm)
    
    # Run multiple workers
    workers = [
        asyncio.create_task(analyzer.run_worker(i))
        for i in range(2)  # Analysis is heavier, use fewer workers
    ]
    
    await asyncio.gather(*workers)


if __name__ == "__main__":
    asyncio.run(main())
