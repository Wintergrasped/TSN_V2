"""
Callsign extractor - extracts and validates callsigns using regex + vLLM.
"""

import asyncio
import re
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
    Transcription,
    ValidationMethod,
)
from tsn_common.utils import normalize_callsign

logger = get_logger(__name__)

# Regex pattern for amateur radio callsigns
CALLSIGN_PATTERN = re.compile(
    r"\b([A-Z]{1,2}\d[A-Z]{1,4})\b",
    re.IGNORECASE,
)


class CallsignExtractor:
    """
    Extracts callsigns from transcripts using regex + vLLM validation.
    """

    def __init__(self, settings: VLLMSettings):
        self.settings = settings
        self.http_client = httpx.AsyncClient(timeout=settings.timeout_sec)
        self.validated_cache: set[str] = set()
        
        logger.info(
            "callsign_extractor_initialized",
            vllm_url=settings.base_url,
            model=settings.model,
        )

    async def close(self) -> None:
        """Close HTTP client."""
        await self.http_client.aclose()

    def extract_candidates(self, text: str) -> list[str]:
        """
        Extract callsign candidates using regex.
        
        Args:
            text: Transcript text
            
        Returns:
            List of candidate callsigns (normalized)
        """
        candidates = set()
        
        for match in CALLSIGN_PATTERN.finditer(text):
            callsign = normalize_callsign(match.group(1))
            if len(callsign) >= 4:  # Minimum valid callsign length
                candidates.add(callsign)
        
        return list(candidates)

    async def validate_with_vllm(self, callsigns: list[str]) -> dict[str, bool]:
        """
        Validate callsigns using vLLM.
        
        Args:
            callsigns: List of candidate callsigns
            
        Returns:
            Dict of callsign -> is_valid
        """
        if not callsigns:
            return {}
        
        # Filter already cached
        to_validate = [cs for cs in callsigns if cs not in self.validated_cache]
        
        if not to_validate:
            return {cs: True for cs in callsigns}
        
        try:
            # Build prompt
            callsign_list = ", ".join(to_validate)
            prompt = f"""You are validating amateur radio callsigns.

Callsigns to validate: {callsign_list}

For each callsign, determine if it's a valid amateur radio callsign format.
Valid formats follow ITU regulations: 1-2 letter prefix + digit + 1-4 letter suffix.

Respond with JSON only:
{{"valid": ["CALL1", "CALL2"], "invalid": ["INVALID1"]}}"""

            # Call vLLM
            response = await self.http_client.post(
                f"{self.settings.base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.settings.api_key.get_secret_value()}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self.settings.model,
                    "messages": [
                        {"role": "system", "content": "You are a callsign validator."},
                        {"role": "user", "content": prompt},
                    ],
                    "temperature": 0.0,
                    "max_tokens": 500,
                    "response_format": {"type": "json_object"},
                },
            )
            
            response.raise_for_status()
            data = response.json()
            
            content = data["choices"][0]["message"]["content"]
            
            # Parse JSON response
            import json
            result = json.loads(content)
            
            valid_set = set(result.get("valid", []))
            
            # Cache valid callsigns
            self.validated_cache.update(valid_set)
            
            # Build result dict
            validation_result = {}
            for cs in to_validate:
                validation_result[cs] = cs in valid_set
            
            # Add cached callsigns
            for cs in callsigns:
                if cs not in validation_result:
                    validation_result[cs] = cs in self.validated_cache
            
            logger.info(
                "vllm_validation_completed",
                total_callsigns=len(callsigns),
                validated=sum(validation_result.values()),
                rejected=len(validation_result) - sum(validation_result.values()),
            )
            
            return validation_result
            
        except Exception as e:
            logger.error(
                "vllm_validation_failed",
                callsigns=callsigns,
                error=str(e),
            )
            # Fall back to regex-only validation
            return {cs: True for cs in callsigns}

    async def get_or_create_callsign(
        self,
        session: AsyncSession,
        callsign: str,
        validated: bool,
    ) -> Callsign:
        """
        Get existing callsign or create new one.
        
        Args:
            session: Database session
            callsign: Normalized callsign
            validated: Whether validated by vLLM
            
        Returns:
            Callsign record
        """
        # Check if exists
        result = await session.execute(
            select(Callsign).where(Callsign.callsign == callsign)
        )
        existing = result.scalar_one_or_none()
        
        if existing:
            # Update activity
            existing.last_seen = datetime.now(timezone.utc)
            existing.seen_count += 1
            
            # Update validation if newly validated
            if validated and not existing.validated:
                existing.validated = True
                existing.validation_method = ValidationMethod.VLLM
            
            return existing
        else:
            # Create new
            now = datetime.now(timezone.utc)
            new_callsign = Callsign(
                callsign=callsign,
                validated=validated,
                validation_method=ValidationMethod.VLLM if validated else ValidationMethod.REGEX,
                first_seen=now,
                last_seen=now,
                seen_count=1,
            )
            session.add(new_callsign)
            await session.flush()
            
            return new_callsign

    async def extract_from_transcription(
        self,
        transcription: Transcription,
    ) -> list[Callsign]:
        """
        Extract callsigns from a transcription.
        
        Args:
            transcription: Transcription record
            
        Returns:
            List of extracted Callsign records
        """
        # Extract candidates
        candidates = self.extract_candidates(transcription.transcript_text)
        
        if not candidates:
            logger.info(
                "no_callsigns_found",
                transcription_id=str(transcription.id),
            )
            return []
        
        logger.info(
            "callsign_candidates_extracted",
            transcription_id=str(transcription.id),
            candidate_count=len(candidates),
            candidates=candidates[:10],  # Log first 10
        )
        
        # Validate with vLLM
        validation_results = await self.validate_with_vllm(candidates)
        
        # Store in database
        callsigns = []
        
        async with get_session() as session:
            for candidate, is_valid in validation_results.items():
                callsign = await self.get_or_create_callsign(session, candidate, is_valid)
                callsigns.append(callsign)
                
                # Create log entry
                log_entry = CallsignLog(
                    callsign_id=callsign.id,
                    transcription_id=transcription.id,
                    detected_at=datetime.now(timezone.utc),
                    confidence=1.0 if is_valid else 0.5,
                )
                session.add(log_entry)
        
        logger.info(
            "callsigns_extracted",
            transcription_id=str(transcription.id),
            total_extracted=len(callsigns),
            validated=sum(1 for cs in callsigns if cs.validated),
        )
        
        return callsigns

    async def get_next_transcription(
        self,
        session: AsyncSession,
    ) -> Optional[tuple[AudioFile, Transcription]]:
        """Get next transcription to process."""
        result = await session.execute(
            select(AudioFile, Transcription)
            .join(Transcription, AudioFile.id == Transcription.audio_file_id)
            .where(AudioFile.state == AudioFileState.QUEUED_EXTRACTION)
            .order_by(AudioFile.created_at)
            .limit(1)
            .with_for_update(skip_locked=True)
        )
        
        row = result.first()
        return row if row else None

    async def process_one(self) -> bool:
        """Process one transcription from queue."""
        async with get_session() as session:
            result = await self.get_next_transcription(session)
            
            if result is None:
                return False
            
            audio_file, transcription = result
            
            # Update state
            audio_file.state = AudioFileState.EXTRACTING
            await session.flush()
        
        # Extract callsigns
        try:
            callsigns = await self.extract_from_transcription(transcription)
            
            # Update state
            async with get_session() as session:
                audio_file = await session.get(AudioFile, audio_file.id)
                audio_file.state = AudioFileState.CALLSIGNS_EXTRACTED
                audio_file.state = AudioFileState.QUEUED_ANALYSIS
                
                logger.info(
                    "extraction_completed",
                    audio_file_id=str(audio_file.id),
                    callsign_count=len(callsigns),
                )
        except Exception as e:
            logger.error(
                "extraction_failed",
                audio_file_id=str(audio_file.id),
                error=str(e),
                exc_info=True,
            )
            
            async with get_session() as session:
                audio_file = await session.get(AudioFile, audio_file.id)
                audio_file.state = AudioFileState.FAILED_EXTRACTION
                audio_file.retry_count += 1
        
        return True

    async def run_worker(self, worker_id: int = 0) -> None:
        """Run extraction worker loop."""
        logger.info("extraction_worker_started", worker_id=worker_id)
        
        try:
            while True:
                try:
                    processed = await self.process_one()
                    
                    if not processed:
                        await asyncio.sleep(1.0)
                        
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.error(
                        "extraction_worker_error",
                        worker_id=worker_id,
                        error=str(e),
                        exc_info=True,
                    )
                    await asyncio.sleep(5.0)
        finally:
            await self.close()
            logger.info("extraction_worker_stopped", worker_id=worker_id)


async def main() -> None:
    """Main entry point."""
    from tsn_common import setup_logging
    
    settings = get_settings()
    setup_logging(settings.logging)
    
    extractor = CallsignExtractor(settings.vllm)
    
    # Run multiple workers
    workers = [
        asyncio.create_task(extractor.run_worker(i))
        for i in range(settings.vllm.max_concurrent)
    ]
    
    await asyncio.gather(*workers)


if __name__ == "__main__":
    asyncio.run(main())
