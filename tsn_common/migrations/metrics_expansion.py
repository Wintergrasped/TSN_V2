"""Schema tweaks for AI metrics expansion."""

from sqlalchemy import text

from tsn_common.config import get_settings
from tsn_common.db import get_engine
from tsn_common.logging import get_logger

logger = get_logger(__name__)


class MetricsExpansionMigrator:
    """Ensure smoothing columns exist for transcriptions."""

    def __init__(self) -> None:
        self.engine = get_engine()
        self.schema = get_settings().database.name

    async def run(self) -> None:
        try:
            await self._ensure_transcription_columns()
        except Exception as exc:  # pragma: no cover - direct SQL migrations
            logger.error("metrics_expansion_failed", error=str(exc))
            raise

    async def _ensure_transcription_columns(self) -> None:
        async with self.engine.begin() as conn:
            logger.info("metrics_expansion_checking_transcriptions")
            await conn.execute(
                text(
                    "ALTER TABLE `transcriptions` "
                    "ADD COLUMN IF NOT EXISTS `smoothed_text` LONGTEXT NULL"
                )
            )
            await conn.execute(
                text(
                    "ALTER TABLE `transcriptions` "
                    "ADD COLUMN IF NOT EXISTS `smoothed_metadata` JSON NULL"
                )
            )
            await conn.execute(
                text(
                    "ALTER TABLE `transcriptions` "
                    "ADD COLUMN IF NOT EXISTS `smoothed_at` DATETIME(6) NULL"
                )
            )
            logger.info("metrics_expansion_transcriptions_ready")

