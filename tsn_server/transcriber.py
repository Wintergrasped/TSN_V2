"""
Transcription pipeline - processes audio files with Whisper.
"""

import asyncio
import time
from pathlib import Path
from typing import Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from tsn_common.config import TranscriptionSettings, get_settings
from tsn_common.db import get_session
from tsn_common.logging import get_logger
from tsn_common.models import (
    AudioFile,
    AudioFileState,
    Transcription,
    TranscriptionBackend,
)

logger = get_logger(__name__)


class TranscriptionPipeline:
    """
    Transcription pipeline using faster-whisper.
    
    Processes audio files from the queue and stores transcripts.
    """

    def __init__(
        self,
        settings: TranscriptionSettings,
        storage_dir: Path,
    ):
        self.settings = settings
        self.storage_dir = storage_dir
        self.model = None
        
        logger.info(
            "transcription_pipeline_initialized",
            backend=settings.backend,
            model=settings.model,
            device=settings.device,
        )

    def _load_model(self) -> None:
        """Load Whisper model (lazy initialization)."""
        if self.model is not None:
            return
        
        if self.settings.backend == "faster-whisper":
            from faster_whisper import WhisperModel
            
            self.model = WhisperModel(
                self.settings.model,
                device=self.settings.device,
                compute_type=self.settings.compute_type,
            )
            
            logger.info(
                "whisper_model_loaded",
                backend="faster-whisper",
                model=self.settings.model,
                device=self.settings.device,
            )
        else:
            raise NotImplementedError(f"Backend {self.settings.backend} not implemented")

    async def transcribe_file(
        self,
        audio_file: AudioFile,
    ) -> Optional[Transcription]:
        """
        Transcribe an audio file.
        
        Args:
            audio_file: AudioFile database record
            
        Returns:
            Transcription record if successful, None otherwise
        """
        file_path = self.storage_dir / audio_file.filename
        
        if not file_path.exists():
            logger.error(
                "transcribe_file_not_found",
                audio_file_id=str(audio_file.id),
                filename=audio_file.filename,
            )
            return None
        
        try:
            logger.info(
                "transcription_starting",
                audio_file_id=str(audio_file.id),
                filename=audio_file.filename,
            )
            
            start_time = time.time()
            
            # Load model if needed
            self._load_model()
            
            # Run transcription in executor (blocking I/O)
            loop = asyncio.get_event_loop()
            transcript_text = await loop.run_in_executor(
                None,
                self._transcribe_sync,
                str(file_path),
            )
            
            processing_time_ms = int((time.time() - start_time) * 1000)
            
            # Create transcription record
            transcription = Transcription(
                audio_file_id=audio_file.id,
                transcript_text=transcript_text,
                language=self.settings.language,
                backend=TranscriptionBackend.FASTER_WHISPER,
                processing_time_ms=processing_time_ms,
                word_count=len(transcript_text.split()) if transcript_text else 0,
                char_count=len(transcript_text) if transcript_text else 0,
            )
            
            logger.info(
                "transcription_completed",
                audio_file_id=str(audio_file.id),
                filename=audio_file.filename,
                processing_time_ms=processing_time_ms,
                word_count=transcription.word_count,
                char_count=transcription.char_count,
            )
            
            return transcription
            
        except Exception as e:
            logger.error(
                "transcription_failed",
                audio_file_id=str(audio_file.id),
                filename=audio_file.filename,
                error=str(e),
                exc_info=True,
            )
            return None

    def _transcribe_sync(self, file_path: str) -> str:
        """
        Synchronous transcription (runs in executor).
        
        Args:
            file_path: Path to audio file
            
        Returns:
            Transcript text
        """
        if self.model is None:
            raise RuntimeError("Model not loaded")
        
        segments, _ = self.model.transcribe(
            file_path,
            beam_size=self.settings.beam_size,
            vad_filter=self.settings.vad_filter,
            language=self.settings.language,
            temperature=self.settings.temperature,
            condition_on_previous_text=False,
        )
        
        # Collect all segments
        text = " ".join(seg.text.strip() for seg in segments).strip()
        
        return text

    async def get_next_file(
        self,
        session: AsyncSession,
    ) -> Optional[AudioFile]:
        """
        Get next file to transcribe from queue.
        
        Uses SELECT FOR UPDATE SKIP LOCKED for concurrent workers.
        
        Args:
            session: Database session
            
        Returns:
            AudioFile to process, or None if queue empty
        """
        result = await session.execute(
            select(AudioFile)
            .where(AudioFile.state == AudioFileState.QUEUED_TRANSCRIPTION)
            .order_by(AudioFile.created_at)
            .limit(1)
            .with_for_update(skip_locked=True)
        )
        
        return result.scalar_one_or_none()

    async def process_one(self) -> bool:
        """
        Process one file from the queue.
        
        Returns:
            True if file was processed, False if queue empty
        """
        async with get_session() as session:
            # Get next file
            audio_file = await self.get_next_file(session)
            
            if audio_file is None:
                return False
            
            # Update state to transcribing
            audio_file.state = AudioFileState.TRANSCRIBING
            await session.flush()
        
        # Transcribe (outside transaction for long-running operation)
        transcription = await self.transcribe_file(audio_file)
        
        # Update database
        async with get_session() as session:
            # Re-fetch audio file
            audio_file = await session.get(AudioFile, audio_file.id)
            
            if transcription:
                # Save transcription
                session.add(transcription)
                
                # Update state
                audio_file.state = AudioFileState.TRANSCRIBED
                audio_file.state = AudioFileState.QUEUED_EXTRACTION
                
                logger.info(
                    "file_transcribed_queued_extraction",
                    audio_file_id=str(audio_file.id),
                    transcription_id=str(transcription.id),
                )
            else:
                # Mark as failed
                audio_file.state = AudioFileState.FAILED_TRANSCRIPTION
                audio_file.retry_count += 1
                
                logger.error(
                    "file_transcription_failed",
                    audio_file_id=str(audio_file.id),
                    retry_count=audio_file.retry_count,
                )
        
        return True

    async def run_worker(self, worker_id: int = 0) -> None:
        """
        Run transcription worker loop.
        
        Args:
            worker_id: Worker identifier for logging
        """
        logger.info("transcription_worker_started", worker_id=worker_id)
        
        while True:
            try:
                processed = await self.process_one()
                
                if not processed:
                    # Queue empty, wait before polling again
                    await asyncio.sleep(1.0)
                    
            except asyncio.CancelledError:
                logger.info("transcription_worker_cancelled", worker_id=worker_id)
                break
            except Exception as e:
                logger.error(
                    "transcription_worker_error",
                    worker_id=worker_id,
                    error=str(e),
                    exc_info=True,
                )
                await asyncio.sleep(5.0)


async def main() -> None:
    """Main entry point for transcription pipeline."""
    from tsn_common import setup_logging
    
    settings = get_settings()
    setup_logging(settings.logging)
    
    storage_dir = Path("/path/to/storage")  # TODO: from config
    
    # Create pipeline
    pipeline = TranscriptionPipeline(settings.transcription, storage_dir)
    
    # Run multiple workers
    workers = [
        asyncio.create_task(pipeline.run_worker(i))
        for i in range(settings.transcription.max_concurrent)
    ]
    
    # Wait for all workers
    await asyncio.gather(*workers)


if __name__ == "__main__":
    asyncio.run(main())
