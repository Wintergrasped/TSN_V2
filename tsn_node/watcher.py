"""
File watcher - monitors incoming directory for new WAV files.
"""

import asyncio
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Set

from watchfiles import awatch

from tsn_common.config import NodeSettings, get_settings
from tsn_common.logging import get_logger
from tsn_common.utils import compute_sha256

logger = get_logger(__name__)


class FileWatcher:
    """
    Watches incoming directory for new WAV files.
    Queues them for transfer when they're stable.
    """

    def __init__(self, settings: NodeSettings):
        self.settings = settings
        self.incoming_dir = settings.audio_incoming_dir
        self.pending_files: dict[Path, float] = {}  # file -> detected timestamp
        self.processing_files: Set[Path] = set()
        self.transfer_queue: asyncio.Queue[Path] = asyncio.Queue()
        
        logger.info(
            "file_watcher_initialized",
            incoming_dir=str(self.incoming_dir),
            min_file_age=settings.min_file_age_sec,
            min_file_size=settings.min_file_size,
        )

    def _is_wav_file(self, path: Path) -> bool:
        """Check if file is a WAV file."""
        return path.suffix.lower() == ".wav"

    def _is_file_stable(self, path: Path) -> bool:
        """
        Check if file is stable (not currently being written).
        
        A file is considered stable if:
        1. It's older than min_file_age_sec
        2. It's larger than min_file_size
        """
        try:
            if not path.exists():
                return False
            
            stat = path.stat()
            
            # Check minimum size
            if stat.st_size < self.settings.min_file_size:
                return False
            
            # Check age
            age = time.time() - stat.st_mtime
            if age < self.settings.min_file_age_sec:
                return False
            
            return True
        except Exception as e:
            logger.warning("file_stability_check_failed", path=str(path), error=str(e))
            return False

    async def scan_existing_files(self) -> None:
        """Scan for existing files on startup."""
        try:
            for file_path in self.incoming_dir.glob("*.wav"):
                if file_path not in self.pending_files and file_path not in self.processing_files:
                    self.pending_files[file_path] = time.time()
                    logger.info("existing_file_detected", filename=file_path.name)
        except Exception as e:
            logger.error("existing_file_scan_failed", error=str(e))

    async def check_pending_files(self) -> None:
        """Check pending files for stability and queue them."""
        ready_files = []
        
        for file_path, detected_at in list(self.pending_files.items()):
            if file_path in self.processing_files:
                continue
            
            if not file_path.exists():
                # File disappeared
                del self.pending_files[file_path]
                logger.warning("file_disappeared", filename=file_path.name)
                continue
            
            if self._is_file_stable(file_path):
                ready_files.append(file_path)
        
        # Queue ready files
        for file_path in ready_files:
            del self.pending_files[file_path]
            self.processing_files.add(file_path)
            await self.transfer_queue.put(file_path)
            logger.info(
                "file_queued_for_transfer",
                filename=file_path.name,
                pending_time_sec=time.time() - self.pending_files.get(file_path, time.time()),
            )

    async def watch_loop(self) -> None:
        """Main watch loop using watchfiles."""
        logger.info("watch_loop_started", directory=str(self.incoming_dir))
        
        # Initial scan
        await self.scan_existing_files()
        
        # Start periodic stability check
        stability_task = asyncio.create_task(self._periodic_stability_check())
        
        try:
            async for changes in awatch(self.incoming_dir):
                for change_type, path_str in changes:
                    path = Path(path_str)
                    
                    # Only care about WAV files
                    if not self._is_wav_file(path):
                        continue
                    
                    # File created or modified
                    if change_type in (1, 2):  # Created or Modified
                        if path not in self.pending_files and path not in self.processing_files:
                            self.pending_files[path] = time.time()
                            logger.info(
                                "new_file_detected",
                                filename=path.name,
                                change_type=change_type,
                            )
                    
                    # File deleted
                    elif change_type == 3:  # Deleted
                        if path in self.pending_files:
                            del self.pending_files[path]
                        if path in self.processing_files:
                            self.processing_files.discard(path)
                        logger.info("file_deleted", filename=path.name)
        finally:
            stability_task.cancel()
            try:
                await stability_task
            except asyncio.CancelledError:
                pass

    async def _periodic_stability_check(self) -> None:
        """Periodically check pending files for stability."""
        while True:
            try:
                await asyncio.sleep(self.settings.watch_interval_sec)
                await self.check_pending_files()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("stability_check_error", error=str(e))

    def mark_file_processed(self, file_path: Path) -> None:
        """Mark file as processed (remove from tracking)."""
        self.processing_files.discard(file_path)
        logger.debug("file_marked_processed", filename=file_path.name)

    def mark_file_failed(self, file_path: Path) -> None:
        """Mark file as failed (can be retried later)."""
        self.processing_files.discard(file_path)
        # Re-add to pending for retry
        self.pending_files[file_path] = time.time()
        logger.warning("file_marked_failed_will_retry", filename=file_path.name)

    async def run(self) -> None:
        """Run the file watcher."""
        logger.info("file_watcher_starting", node_id=self.settings.node_id)
        await self.watch_loop()


async def main() -> None:
    """Main entry point for file watcher."""
    from tsn_common import setup_logging
    
    settings = get_settings()
    if settings.node is None:
        logger.error("node_settings_not_configured")
        return
    
    setup_logging(settings.logging)
    
    watcher = FileWatcher(settings.node)
    await watcher.run()


if __name__ == "__main__":
    asyncio.run(main())
