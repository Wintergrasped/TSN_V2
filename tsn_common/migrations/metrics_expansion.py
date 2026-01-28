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
            await self._ensure_ai_tables()
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

    async def _ensure_ai_tables(self) -> None:
        async with self.engine.begin() as conn:
            logger.info("metrics_expansion_checking_ai_tables")
            await conn.execute(
                text(
                    "CREATE TABLE IF NOT EXISTS `ai_run_logs` ("
                    "`id` CHAR(36) PRIMARY KEY,"
                    "`created_at` DATETIME(6) NOT NULL,"
                    "`updated_at` DATETIME(6) NOT NULL,"
                    "`backend` VARCHAR(32) NOT NULL,"
                    "`model` VARCHAR(128) NULL,"
                    "`pass_label` VARCHAR(64) NOT NULL,"
                    "`success` TINYINT(1) NOT NULL DEFAULT 1,"
                    "`error_message` LONGTEXT NULL,"
                    "`prompt_text` LONGTEXT NOT NULL,"
                    "`response_text` LONGTEXT NULL,"
                    "`prompt_characters` INT NOT NULL DEFAULT 0,"
                    "`response_characters` INT NULL,"
                    "`prompt_tokens` INT NULL,"
                    "`completion_tokens` INT NULL,"
                    "`total_tokens` INT NULL,"
                    "`latency_ms` INT NULL,"
                    "`gpu_utilization_pct` FLOAT NULL,"
                    "`audio_file_ids` JSON NULL,"
                    "`metadata` JSON NOT NULL,"
                    "INDEX `ix_ai_run_logs_pass_label` (`pass_label`, `created_at`)")
                )
            )
            await conn.execute(
                text(
                    "CREATE TABLE IF NOT EXISTS `gpu_utilization_samples` ("
                    "`id` CHAR(36) PRIMARY KEY,"
                    "`created_at` DATETIME(6) NOT NULL,"
                    "`updated_at` DATETIME(6) NOT NULL,"
                    "`utilization_pct` FLOAT NOT NULL,"
                    "`sample_source` VARCHAR(32) NOT NULL,"
                    "`is_saturated` TINYINT(1) NOT NULL DEFAULT 0,"
                    "`notes` LONGTEXT NULL,"
                    "INDEX `ix_gpu_samples_created_at` (`created_at`)")
                )
            )
            logger.info("metrics_expansion_ai_tables_ready")
