"""Utility to backfill audio_files.node_id from filename prefixes."""

from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timezone

from sqlalchemy import select

from tsn_common import setup_logging
from tsn_common.config import get_settings
from tsn_common.db import get_session
from tsn_common.logging import get_logger
from tsn_common.models import AudioFile
from tsn_common.utils import parse_audio_filename_metadata

logger = get_logger(__name__)


async def repair_node_ids(limit: int | None = None) -> None:
    """Ensure every audio file carries the node ID encoded in its filename."""

    processed = 0
    updated = 0
    async with get_session() as session:
        stream = await session.stream_scalars(
            select(AudioFile).order_by(AudioFile.created_at)
        )
        try:
            async for audio in stream:
                if limit is not None and processed >= limit:
                    break
                processed += 1
                parsed = parse_audio_filename_metadata(audio.filename)
                filename_node = parsed.get("node_id")
                recorded_at = parsed.get("recorded_at")
                metadata = dict(audio.metadata_ or {})
                changed = False

                if filename_node and audio.node_id != filename_node:
                    logger.info(
                        "audio_node_id_repaired",
                        audio_file_id=str(audio.id),
                        filename=audio.filename,
                        old_node=audio.node_id,
                        new_node=filename_node,
                    )
                    audio.node_id = filename_node
                    metadata["filename_node_id"] = filename_node
                    changed = True

                if recorded_at and not metadata.get("source_timestamp"):
                    metadata["source_timestamp"] = recorded_at.isoformat()
                    changed = True

                if changed:
                    metadata["node_id_repaired_at"] = datetime.now(timezone.utc).isoformat()
                    audio.metadata_ = metadata
                    updated += 1

            await session.commit()
        finally:
            await stream.close()

    logger.info("node_id_repair_complete", processed=processed, updated=updated)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Stop after processing this many audio files (useful for dry runs)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = get_settings()
    setup_logging(settings.logging)
    asyncio.run(repair_node_ids(limit=args.limit))


if __name__ == "__main__":
    main()
