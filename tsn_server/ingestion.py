"""
Ingestion service - receives files and queues them for processing.
"""

import asyncio
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from tsn_common.config import ServerSettings, get_settings
from tsn_common.db import get_session
from tsn_common.logging import get_logger
from tsn_common.models import AudioFile, AudioFileState, Node
from tsn_common.resource_lock import get_resource_lock
from tsn_common.utils import (
    compute_sha256,
    get_audio_metadata,
    parse_audio_filename_metadata,
)

try:
    from scripts.fix_audio_node_ids import repair_node_ids
    REPAIR_AVAILABLE = True
except ImportError:
    REPAIR_AVAILABLE = False

from tsn_server.storage_guard import StorageGuard

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

    def __init__(
        self,
        server_settings: ServerSettings,
        storage_dir: Path,
        storage_guard: StorageGuard | None = None,
    ):
        self.settings = server_settings
        self.incoming_dir = server_settings.incoming_dir
        self.storage_dir = storage_dir
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.poll_interval = server_settings.poll_interval_sec
        self.storage_guard = storage_guard
        self._startup_repair_done = False
        
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
            filename_meta = parse_audio_filename_metadata(file_path.name)
            inferred_node_id = filename_meta.get("node_id") or node_id

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
                if filename_meta.get("recorded_at"):
                    metadata["source_timestamp"] = filename_meta["recorded_at"].isoformat()
                if filename_meta.get("node_id"):
                    metadata["filename_node_id"] = filename_meta["node_id"]
                metadata["filename_format"] = filename_meta.get("format", "unknown")
                metadata["archive_ingest"] = filename_meta.get("is_archive", False)

                filename_node_id = metadata.get("filename_node_id")
                if filename_node_id and filename_node_id != inferred_node_id:
                    if node_id not in {"unknown", filename_node_id}:
                        logger.info(
                            "node_id_overridden_by_filename",
                            filename=file_path.name,
                            requested_node=node_id,
                            filename_node=filename_node_id,
                        )
                    inferred_node_id = filename_node_id
                
                # Log node_id extraction for multi-node deployments
                if filename_node_id and filename_node_id != "unknown":
                    logger.info(
                        "node_id_extracted_from_filename",
                        filename=file_path.name,
                        node_id=filename_node_id,
                        format=filename_meta.get("format"),
                    )
                
                # Move to storage (handle cross-device moves)
                storage_path = self.storage_dir / file_path.name
                try:
                    file_path.rename(storage_path)
                except OSError as exc:
                    logger.warning(
                        "rename_failed_falling_back_to_copy",
                        filename=file_path.name,
                        error=str(exc),
                    )
                    try:
                        shutil.move(str(file_path), str(storage_path))
                    except Exception as copy_exc:
                        logger.error(
                            "storage_copy_failed",
                            filename=file_path.name,
                            destination=str(storage_path),
                            error=str(copy_exc),
                        )
                        if self.storage_guard:
                            self.storage_guard.mark_unavailable(
                                source="ingestion_move",
                                error=str(copy_exc),
                            )
                        return None
                
                # Create database record - set to QUEUED_TRANSCRIPTION immediately
                # to avoid race condition where workers grab RECEIVED state before update
                audio_file = AudioFile(
                    filename=file_path.name,
                    sha256=sha256,
                    file_size=file_size,
                    duration_sec=metadata.get("duration_sec"),
                    sample_rate=metadata.get("sample_rate"),
                    channels=metadata.get("channels"),
                    node_id=inferred_node_id,
                    uploaded_at=datetime.now(timezone.utc),
                    state=AudioFileState.QUEUED_TRANSCRIPTION,
                    metadata_=metadata,
                )
                
                session.add(audio_file)
                await session.flush()
                
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
            for file_path in self.incoming_dir.iterdir():
                if not file_path.is_file():
                    continue

                if file_path.suffix.lower() not in {".wav", ".mp3"}:
                    continue

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
        
        # Run node_id repair on first loop to fix any historical records
        if not self._startup_repair_done and REPAIR_AVAILABLE:
            try:
                logger.info("running_startup_node_id_repair")
                await repair_node_ids(limit=1000)
                logger.info("startup_node_id_repair_complete")
            except Exception as exc:
                logger.warning(
                    "startup_node_id_repair_failed",
                    error=str(exc),
                    exc_info=True,
                )
            finally:
                self._startup_repair_done = True
        
        while True:
            try:
                if self.storage_guard and not self.storage_guard.is_available():
                    await self.storage_guard.wait_until_available("ingestion_loop")
                    continue

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
