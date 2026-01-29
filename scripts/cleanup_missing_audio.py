#!/usr/bin/env python3
"""
Cleanup script for audio files that exist in database but not on disk.

This script identifies audio files in QUEUED_TRANSCRIPTION state that don't
have corresponding files in the storage directory and marks them as FAILED_TRANSCRIPTION.
"""

import asyncio
from pathlib import Path
import sys

from sqlalchemy import select, update

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from tsn_common.config import get_settings
from tsn_common.db import get_session
from tsn_common.logging import setup_logging, get_logger
from tsn_common.models import AudioFile, AudioFileState

logger = get_logger(__name__)


async def cleanup_missing_audio_files(storage_dir: Path, dry_run: bool = True) -> None:
    """
    Find and mark audio files as failed if they don't exist on disk.
    
    Args:
        storage_dir: Path to storage directory containing audio files
        dry_run: If True, only report what would be done without making changes
    """
    logger.info("cleanup_starting", storage_dir=str(storage_dir), dry_run=dry_run)
    
    # Get all files in QUEUED_TRANSCRIPTION state
    async with get_session() as session:
        result = await session.execute(
            select(AudioFile)
            .where(AudioFile.state == AudioFileState.QUEUED_TRANSCRIPTION)
            .order_by(AudioFile.created_at)
        )
        queued_files = result.scalars().all()
    
    logger.info("found_queued_files", count=len(queued_files))
    
    missing_files = []
    existing_files = []
    
    # Check which files actually exist on disk
    for audio_file in queued_files:
        file_path = storage_dir / audio_file.filename
        if file_path.exists():
            existing_files.append(audio_file)
            logger.debug(
                "file_exists",
                filename=audio_file.filename,
                size_bytes=audio_file.size_bytes,
            )
        else:
            missing_files.append(audio_file)
            logger.warning(
                "file_missing",
                audio_file_id=str(audio_file.id),
                filename=audio_file.filename,
                created_at=str(audio_file.created_at),
            )
    
    logger.info(
        "scan_complete",
        total_queued=len(queued_files),
        existing=len(existing_files),
        missing=len(missing_files),
    )
    
    if not missing_files:
        logger.info("no_missing_files_found")
        return
    
    if dry_run:
        logger.info(
            "dry_run_summary",
            would_mark_failed=len(missing_files),
            sample_files=[f.filename for f in missing_files[:10]],
        )
        return
    
    # Mark missing files as FAILED_TRANSCRIPTION
    logger.info("marking_files_as_failed", count=len(missing_files))
    
    async with get_session() as session:
        for audio_file in missing_files:
            audio_file_db = await session.get(AudioFile, audio_file.id)
            if audio_file_db:
                audio_file_db.state = AudioFileState.FAILED_TRANSCRIPTION
                audio_file_db.retry_count += 1
                logger.info(
                    "marked_file_as_failed",
                    audio_file_id=str(audio_file.id),
                    filename=audio_file.filename,
                )
        
        await session.commit()
    
    logger.info("cleanup_complete", marked_failed=len(missing_files))


async def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Clean up audio files missing from storage"
    )
    parser.add_argument(
        "--storage-dir",
        type=Path,
        default=Path("/storage"),
        help="Path to storage directory (default: /storage for container)",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually mark files as failed (default is dry-run)",
    )
    
    args = parser.parse_args()
    
    # Setup logging
    settings = get_settings()
    setup_logging(settings.logging)
    
    # Run cleanup
    await cleanup_missing_audio_files(
        storage_dir=args.storage_dir,
        dry_run=not args.execute,
    )


if __name__ == "__main__":
    asyncio.run(main())
