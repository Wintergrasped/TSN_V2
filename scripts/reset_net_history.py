"""Utility script to purge invalid net data and requeue audio for re-analysis."""

from __future__ import annotations

import asyncio
from typing import Iterable

from sqlalchemy import delete, select

from tsn_common import get_settings, setup_logging
from tsn_common.db import get_session
from tsn_common.logging import get_logger
from tsn_common.models import (
    AudioFile,
    AudioFileState,
    NetParticipation,
    NetSession,
    TrendSnapshot,
)

logger = get_logger(__name__)

_ANALYSIS_KEYS_TO_DROP = {
    "analysis_history",
    "analysis_passes",
    "last_analysis_pass_type",
    "last_analysis_completed_at",
    "pending_refinement",
    "refinement_reason",
    "refinement_requested_at",
}


async def _delete_table_rows(model) -> int:
    async with get_session() as session:
        result = await session.execute(delete(model))
        deleted = result.rowcount or 0
        logger.info("reset_deleted_rows", table=model.__tablename__, count=deleted)
        return deleted


def _prune_metadata(metadata: dict | None) -> dict | None:
    if not metadata:
        return metadata
    clone = dict(metadata)
    for key in _ANALYSIS_KEYS_TO_DROP:
        clone.pop(key, None)
    return clone


async def _requeue_audio() -> tuple[int, int]:
    inspected = 0
    changed = 0
    async with get_session() as session:
        result = await session.stream_scalars(select(AudioFile))
        async for audio in result:
            original_state = audio.state
            metadata = _prune_metadata(audio.metadata_)
            metadata_changed = metadata is not None and metadata != audio.metadata_
            if metadata_changed:
                audio.metadata_ = metadata
            state_changed = False
            if original_state in {
                AudioFileState.COMPLETE,
                AudioFileState.ANALYZED,
                AudioFileState.FAILED_ANALYSIS,
            }:
                if audio.state != AudioFileState.QUEUED_ANALYSIS:
                    audio.state = AudioFileState.QUEUED_ANALYSIS
                    state_changed = True
            if metadata_changed or state_changed:
                changed += 1
            inspected += 1
        await session.flush()
    logger.info("reset_audio_requeued", inspected=inspected, changed=changed)
    return inspected, changed


async def reset_net_history() -> None:
    await _delete_table_rows(NetParticipation)
    await _delete_table_rows(NetSession)
    await _delete_table_rows(TrendSnapshot)
    await _requeue_audio()
    logger.info("reset_net_history_complete")


async def main() -> None:
    settings = get_settings()
    setup_logging(settings.logging)
    await reset_net_history()


if __name__ == "__main__":
    asyncio.run(main())
