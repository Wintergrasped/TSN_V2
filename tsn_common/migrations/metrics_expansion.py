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
            await self._ensure_net_formal_structure_columns()
        except Exception as exc:  # pragma: no cover - direct SQL migrations
            logger.error("metrics_expansion_failed", error=str(exc))
            raise

    async def _ensure_transcription_columns(self) -> None:
        async with self.engine.begin() as conn:
            logger.info("metrics_expansion_checking_transcriptions")
            
            # Check which columns exist
            result = await conn.execute(
                text(
                    "SELECT COLUMN_NAME FROM INFORMATION_SCHEMA.COLUMNS "
                    "WHERE TABLE_SCHEMA = DATABASE() "
                    "AND TABLE_NAME = 'transcriptions' "
                    "AND COLUMN_NAME IN ('smoothed_text', 'smoothed_metadata', 'smoothed_at')"
                )
            )
            existing = {row[0] for row in result.fetchall()}
            
            # Only add columns that don't exist
            if 'smoothed_text' not in existing:
                await conn.execute(
                    text("ALTER TABLE `transcriptions` ADD COLUMN `smoothed_text` LONGTEXT NULL")
                )
            if 'smoothed_metadata' not in existing:
                await conn.execute(
                    text("ALTER TABLE `transcriptions` ADD COLUMN `smoothed_metadata` JSON NULL")
                )
            if 'smoothed_at' not in existing:
                await conn.execute(
                    text("ALTER TABLE `transcriptions` ADD COLUMN `smoothed_at` DATETIME(6) NULL")
                )
            
            logger.info("metrics_expansion_transcriptions_ready")

    async def _ensure_net_formal_structure_columns(self) -> None:
        async with self.engine.begin() as conn:
            logger.info("metrics_expansion_checking_net_sessions")
            
            # Check which columns exist
            result = await conn.execute(
                text(
                    "SELECT COLUMN_NAME FROM INFORMATION_SCHEMA.COLUMNS "
                    "WHERE TABLE_SCHEMA = DATABASE() "
                    "AND TABLE_NAME = 'net_sessions' "
                    "AND COLUMN_NAME IN ('formal_structure', 'ncs_script', 'checkin_sequence')"
                )
            )
            existing = {row[0] for row in result.fetchall()}
            
            # Only add columns that don't exist
            if 'formal_structure' not in existing:
                await conn.execute(
                    text("ALTER TABLE `net_sessions` ADD COLUMN `formal_structure` JSON NULL")
                )
            if 'ncs_script' not in existing:
                await conn.execute(
                    text("ALTER TABLE `net_sessions` ADD COLUMN `ncs_script` JSON NULL")
                )
            if 'checkin_sequence' not in existing:
                await conn.execute(
                    text("ALTER TABLE `net_sessions` ADD COLUMN `checkin_sequence` JSON NULL")
                )
            
            logger.info("metrics_expansion_net_sessions_ready")
