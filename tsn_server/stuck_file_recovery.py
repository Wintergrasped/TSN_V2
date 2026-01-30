"""Recovery service for files stuck in processing states."""

import asyncio
from datetime import datetime, timedelta, timezone

from sqlalchemy import select, and_

from tsn_common.config import get_settings
from tsn_common.db import get_session
from tsn_common.logging import get_logger
from tsn_common.models import AudioFile, AudioFileState

logger = get_logger(__name__)


class StuckFileRecovery:
    """
    Monitors and recovers files stuck in processing states.
    
    Files stuck in TRANSCRIBING or ANALYZING for >15 minutes are reset
    to their respective queue states.
    """
    
    def __init__(self):
        self.stuck_timeout_minutes = 15
        self.check_interval_seconds = 60  # Check every minute
        
        logger.info(
            "stuck_file_recovery_initialized",
            timeout_minutes=self.stuck_timeout_minutes,
            check_interval_seconds=self.check_interval_seconds,
        )
    
    async def recover_stuck_files(self) -> None:
        """Find and recover stuck files."""
        try:
            async with get_session() as session:
                cutoff = datetime.now(timezone.utc) - timedelta(
                    minutes=self.stuck_timeout_minutes
                )
                
                # Find files stuck in TRANSCRIBING
                result = await session.execute(
                    select(AudioFile).where(
                        and_(
                            AudioFile.state == AudioFileState.TRANSCRIBING,
                            AudioFile.updated_at < cutoff,
                        )
                    )
                )
                stuck_transcribing = result.scalars().all()
                
                for audio_file in stuck_transcribing:
                    stuck_duration = datetime.now(timezone.utc) - audio_file.updated_at
                    stuck_minutes = int(stuck_duration.total_seconds() / 60)
                    
                    logger.warning(
                        "stuck_file_recovered_transcribing",
                        audio_file_id=str(audio_file.id),
                        filename=audio_file.filename,
                        stuck_minutes=stuck_minutes,
                        node_id=audio_file.node_id,
                    )
                    
                    audio_file.state = AudioFileState.QUEUED_TRANSCRIPTION
                    audio_file.retry_count += 1
                
                # Find files stuck in ANALYZING
                result = await session.execute(
                    select(AudioFile).where(
                        and_(
                            AudioFile.state == AudioFileState.ANALYZING,
                            AudioFile.updated_at < cutoff,
                        )
                    )
                )
                stuck_analyzing = result.scalars().all()
                
                for audio_file in stuck_analyzing:
                    stuck_duration = datetime.now(timezone.utc) - audio_file.updated_at
                    stuck_minutes = int(stuck_duration.total_seconds() / 60)
                    
                    logger.warning(
                        "stuck_file_recovered_analyzing",
                        audio_file_id=str(audio_file.id),
                        filename=audio_file.filename,
                        stuck_minutes=stuck_minutes,
                        node_id=audio_file.node_id,
                    )
                    
                    audio_file.state = AudioFileState.QUEUED_ANALYSIS
                    audio_file.retry_count += 1
                
                # Find files stuck in EXTRACTING
                result = await session.execute(
                    select(AudioFile).where(
                        and_(
                            AudioFile.state == AudioFileState.EXTRACTING,
                            AudioFile.updated_at < cutoff,
                        )
                    )
                )
                stuck_extracting = result.scalars().all()
                
                for audio_file in stuck_extracting:
                    stuck_duration = datetime.now(timezone.utc) - audio_file.updated_at
                    stuck_minutes = int(stuck_duration.total_seconds() / 60)
                    
                    logger.warning(
                        "stuck_file_recovered_extracting",
                        audio_file_id=str(audio_file.id),
                        filename=audio_file.filename,
                        stuck_minutes=stuck_minutes,
                        node_id=audio_file.node_id,
                    )
                    
                    audio_file.state = AudioFileState.QUEUED_EXTRACTION
                    audio_file.retry_count += 1
                
                await session.flush()
                
                total_recovered = len(stuck_transcribing) + len(stuck_analyzing) + len(stuck_extracting)
                
                if total_recovered > 0:
                    logger.info(
                        "stuck_files_recovery_completed",
                        transcribing_recovered=len(stuck_transcribing),
                        analyzing_recovered=len(stuck_analyzing),
                        extracting_recovered=len(stuck_extracting),
                        total_recovered=total_recovered,
                    )
                    
        except Exception as e:
            logger.error(
                "stuck_file_recovery_failed",
                error=str(e),
                exc_info=True,
            )
    
    async def run(self) -> None:
        """Run stuck file recovery loop."""
        logger.info("stuck_file_recovery_service_started")
        
        try:
            while True:
                try:
                    await self.recover_stuck_files()
                except Exception as e:
                    logger.error(
                        "stuck_file_recovery_cycle_failed",
                        error=str(e),
                        exc_info=True,
                    )
                
                # Wait before next check
                await asyncio.sleep(self.check_interval_seconds)
                
        except asyncio.CancelledError:
            logger.info("stuck_file_recovery_service_cancelled")
            raise
        except Exception as e:
            logger.error(
                "stuck_file_recovery_service_fatal",
                error=str(e),
                exc_info=True,
            )
            raise
