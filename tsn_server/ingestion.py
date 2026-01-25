"""
Ingestion service - receives files and queues them for processing.
"""

import asyncio
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from tsn_common.config import ServerSettings, get_settings
from tsn_common.db import get_session
from tsn_common.logging import get_logger
from tsn_common.models import AudioFile, AudioFileState
from tsn_common.utils import compute_sha256, get_audio_metadata

logger = get_logger(__name__)


class IngestionService:
    """
    Ingestion service - processes incoming audio files.
    
    Responsibilities:
    1. Detect new files in incoming directory
    2. Validate file format and size
    3. Compute SHA256 for deduplication
    4. Create database record
    5. Update state to queued_transcription
    """

    def __init__(self, server_settings: ServerSettings, storage_dir: Path):
        self.settings = server_settings
        self.incoming_dir = server_settings.incoming_dir
        self.storage_dir = storage_dir
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.poll_interval = server_settings.poll_interval_sec
        
        logger.info(
            "ingestion_service_initialized",
            incoming_dir=str(self.incoming_dir),
            storage_dir=str(self.storage_dir),
            poll_interval_sec=self.poll_interval,
        )

    async def check_duplicate(
        self,
        session: AsyncSession,
        sha256: str,
    ) -> Optional[AudioFile]:
        """
        Check if file already exists in database.
        
        Args:
            session: Database session
            sha256: File hash
            
        Returns:
            Existing AudioFile if duplicate, None otherwise
        """
        result = await session.execute(
            select(AudioFile).where(AudioFile.sha256 == sha256)
        )
        return result.scalar_one_or_none()

    async def ingest_file(
        self,
        file_path: Path,
        node_id: str = "unknown",
    ) -> Optional[AudioFile]:
        """
        Ingest a single file.
        
        Args:
            file_path: Path to the audio file
            node_id: Source node identifier
            
        Returns:
            AudioFile record if successful, None otherwise
        """
        if not file_path.exists():
            logger.error("ingest_file_not_found", path=str(file_path))
            return None
        
        try:
            # Compute hash
            sha256 = compute_sha256(file_path)
            file_size = file_path.stat().st_size
            
            logger.info(
                "ingesting_file",
                filename=file_path.name,
                size_bytes=file_size,
                sha256=sha256[:16] + "...",
            )
            
            # Check for duplicate
            async with get_session() as session:
                existing = await self.check_duplicate(session, sha256)
                
                if existing:
                    logger.warning(
                        "duplicate_file_detected",
                        filename=file_path.name,
                        existing_id=str(existing.id),
                        sha256=sha256,
                    )
                    # Delete duplicate
                    file_path.unlink()
                    return existing
                
                # Get audio metadata
                metadata = get_audio_metadata(file_path)
                
                # Move to storage
                storage_path = self.storage_dir / file_path.name
                file_path.rename(storage_path)
                
                # Create database record
                audio_file = AudioFile(
                    filename=file_path.name,
                    sha256=sha256,
                    file_size=file_size,
                    duration_sec=metadata.get("duration_sec"),
                    sample_rate=metadata.get("sample_rate"),
                    channels=metadata.get("channels"),
                    node_id=node_id,
                    uploaded_at=datetime.now(timezone.utc),
                    state=AudioFileState.RECEIVED,
                    metadata_=metadata,
                )
                
                session.add(audio_file)
                await session.flush()
                
                # Update state to queued
                audio_file.state = AudioFileState.QUEUED_TRANSCRIPTION
                
                logger.info(
                    "file_ingested",
                    filename=file_path.name,
                    audio_file_id=str(audio_file.id),
                    duration_sec=audio_file.duration_sec,
                )
                
                return audio_file
                
        except Exception as e:
            logger.error(
                "ingest_file_failed",
                filename=file_path.name,
                error=str(e),
                exc_info=True,
            )
            return None

    async def scan_incoming_directory(self) -> list[Path]:
        """
        Scan incoming directory for new files.
        
        Returns:
            List of file paths to process
        """
        files = []
        
        try:
            for file_path in self.incoming_dir.glob("*.wav"):
                if file_path.is_file():
                    files.append(file_path)
            
            logger.info(
                "incoming_directory_scanned",
                file_count=len(files),
            )
            
        except Exception as e:
            logger.error("directory_scan_failed", error=str(e))
        
        return files

    async def process_incoming_files(self) -> int:
        """
        Process all files in incoming directory.
        
        Returns:
            Number of files processed
        """
        files = await self.scan_incoming_directory()
        processed = 0
        
        for file_path in files:
            result = await self.ingest_file(file_path)
            if result:
                processed += 1
        
        logger.info(
            "incoming_files_processed",
            total=len(files),
            processed=processed,
        )
        
        return processed

    async def run_loop(self, interval_sec: float | None = None) -> None:
        """
        Run continuous ingestion loop.
        
        Args:
            interval_sec: Polling interval in seconds
        """
        interval = interval_sec or self.poll_interval
        logger.info("ingestion_loop_started", interval_sec=interval)
        
        while True:
            try:
                await self.process_incoming_files()
                await asyncio.sleep(interval)
            except asyncio.CancelledError:
                logger.info("ingestion_loop_cancelled")
                break
            except Exception as e:
                logger.error("ingestion_loop_error", error=str(e), exc_info=True)
                await asyncio.sleep(interval)


async def main() -> None:
    """Main entry point for ingestion service."""
    from tsn_common import setup_logging
    
    settings = get_settings()
    setup_logging(settings.logging)
    
    # Create ingestion service
    service = IngestionService(settings.server, settings.storage.base_path)
    await service.run_loop()


if __name__ == "__main__":
    asyncio.run(main())
