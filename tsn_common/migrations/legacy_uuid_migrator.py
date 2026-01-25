"""Automated in-place migration for legacy INT primary keys to UUIDs."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Dict, List, Sequence

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncConnection, AsyncEngine

from tsn_common import setup_logging
from tsn_common.config import get_settings
from tsn_common.db import close_engine, get_engine
from tsn_common.logging import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class ForeignKeySpec:
    """Describes a foreign key relationship that must be migrated."""

    table: str
    column: str
    ref_table: str
    nullable: bool
    ondelete: str | None


@dataclass
class IndexDefinition:
    """Stores index metadata so we can recreate it after column swaps."""

    table: str
    name: str
    columns: List[str]
    unique: bool


class LegacyUUIDMigrator:
    """Performs the INT -> UUID conversion without dropping user data."""

    TABLES: Sequence[str] = (
        "audio_files",
        "transcriptions",
        "callsigns",
        "callsign_log",
        "callsign_topics",
        "net_sessions",
        "net_participations",
        "callsign_profiles",
        "phonetic_corrections",
        "processing_metrics",
        "system_health",
    )

    FOREIGN_KEYS: Sequence[ForeignKeySpec] = (
        ForeignKeySpec("transcriptions", "audio_file_id", "audio_files", False, "CASCADE"),
        ForeignKeySpec("callsign_log", "callsign_id", "callsigns", False, "CASCADE"),
        ForeignKeySpec("callsign_log", "transcription_id", "transcriptions", False, "CASCADE"),
        ForeignKeySpec("callsign_topics", "callsign_id", "callsigns", False, "CASCADE"),
        ForeignKeySpec("callsign_topics", "transcription_id", "transcriptions", False, "CASCADE"),
        ForeignKeySpec("net_sessions", "ncs_callsign_id", "callsigns", True, "SET NULL"),
        ForeignKeySpec("net_participations", "net_session_id", "net_sessions", False, "CASCADE"),
        ForeignKeySpec("net_participations", "callsign_id", "callsigns", False, "CASCADE"),
        ForeignKeySpec("callsign_profiles", "callsign_id", "callsigns", False, "CASCADE"),
    )

    def __init__(self) -> None:
        settings = get_settings()
        self.schema = settings.database.name
        self.engine: AsyncEngine = get_engine()
        self.index_definitions: Dict[tuple[str, str], IndexDefinition] = {}

    async def run(self) -> None:
        if not await self._needs_migration():
            logger.info("legacy_uuid_migration_skipped", reason="schema already matches models")
            return

        logger.info("legacy_uuid_migration_started", schema=self.schema)
        await self._prepare_uuid_columns()
        await self._prepare_foreign_key_columns()
        await self._drop_legacy_foreign_keys()
        await self._swap_foreign_key_columns()
        await self._swap_primary_keys()
        await self._recreate_indexes()
        await self._add_foreign_keys()
        await self._record_migration()
        logger.info("legacy_uuid_migration_complete", schema=self.schema)

    # --- high-level helpers -------------------------------------------------

    async def _needs_migration(self) -> bool:
        async with self.engine.begin() as conn:
            column = await self._get_column_type(conn, "callsigns", "id")
        if column is None:
            return False
        return "char" not in column.lower()

    async def _prepare_uuid_columns(self) -> None:
        async with self.engine.begin() as conn:
            for table in self.TABLES:
                if not await self._table_exists(conn, table):
                    continue
                if await self._column_is_uuid(conn, table, "id"):
                    continue
                await self._ensure_column(conn, table, "id_uuid", "CHAR(36) NULL")
                await conn.execute(text(f"UPDATE `{table}` SET id_uuid = UUID() WHERE id_uuid IS NULL"))
                await conn.execute(text(f"ALTER TABLE `{table}` MODIFY COLUMN id_uuid CHAR(36) NOT NULL"))
                logger.info("uuid_column_populated", table=table)

    async def _prepare_foreign_key_columns(self) -> None:
        async with self.engine.begin() as conn:
            for fk in self.FOREIGN_KEYS:
                if not await self._table_exists(conn, fk.table):
                    continue
                if not await self._column_exists(conn, fk.table, fk.column):
                    logger.info(
                        "foreign_helper_skipped",
                        table=fk.table,
                        column=fk.column,
                        reason="column_missing",
                    )
                    continue
                if await self._column_is_uuid(conn, fk.table, fk.column):
                    continue
                parent_uuid_column = await self._resolve_parent_uuid_column(conn, fk.ref_table)
                if parent_uuid_column is None:
                    logger.error("parent_uuid_missing", table=fk.ref_table)
                    raise RuntimeError(
                        f"Parent table {fk.ref_table} does not have legacy ids to map or UUID primary keys"
                    )
                helper_column = f"{fk.column}_uuid"
                await self._ensure_column(conn, fk.table, helper_column, "CHAR(36) NULL")
                update_sql = text(
                    f"""
                    UPDATE `{fk.table}` child
                    JOIN `{fk.ref_table}` parent ON child.`{fk.column}` = parent.`id`
                    SET child.`{helper_column}` = parent.`{parent_uuid_column}`
                    WHERE child.`{fk.column}` IS NOT NULL
                      AND child.`{helper_column}` IS NULL
                    """
                )
                await conn.execute(update_sql)
                null_sql = "NULL" if fk.nullable else "NOT NULL"
                await conn.execute(
                    text(
                        f"ALTER TABLE `{fk.table}` MODIFY COLUMN `{helper_column}` CHAR(36) {null_sql}"
                    )
                )
                logger.info("foreign_helper_populated", table=fk.table, column=fk.column)

    async def _drop_legacy_foreign_keys(self) -> None:
        async with self.engine.begin() as conn:
            for fk in self.FOREIGN_KEYS:
                await self._drop_foreign_keys_for_column(conn, fk.table, fk.column)

    async def _swap_foreign_key_columns(self) -> None:
        async with self.engine.begin() as conn:
            for fk in self.FOREIGN_KEYS:
                if not await self._column_exists(conn, fk.table, fk.column):
                    logger.info(
                        "foreign_swap_skipped",
                        table=fk.table,
                        column=fk.column,
                        reason="column_missing",
                    )
                    continue
                if await self._column_is_uuid(conn, fk.table, fk.column):
                    continue
                helper_column = f"{fk.column}_uuid"
                if not await self._column_exists(conn, fk.table, helper_column):
                    raise RuntimeError(f"Helper column {helper_column} missing on {fk.table}")
                await self._record_and_drop_indexes(conn, fk.table, fk.column)
                await conn.execute(text(f"ALTER TABLE `{fk.table}` DROP COLUMN `{fk.column}`"))
                null_sql = "NULL" if fk.nullable else "NOT NULL"
                await conn.execute(
                    text(
                        f"ALTER TABLE `{fk.table}` CHANGE COLUMN `{helper_column}` `{fk.column}` CHAR(36) {null_sql}"
                    )
                )
                logger.info("foreign_column_swapped", table=fk.table, column=fk.column)

    async def _swap_primary_keys(self) -> None:
        async with self.engine.begin() as conn:
            for table in self.TABLES:
                if not await self._table_exists(conn, table):
                    continue
                if not await self._column_exists(conn, table, "id_uuid"):
                    continue
                await self._record_and_drop_indexes(conn, table, "id")
                await conn.execute(text(f"ALTER TABLE `{table}` DROP PRIMARY KEY"))
                await conn.execute(text(f"ALTER TABLE `{table}` DROP COLUMN id"))
                await conn.execute(
                    text(
                        f"ALTER TABLE `{table}` CHANGE COLUMN id_uuid id CHAR(36) NOT NULL"
                    )
                )
                await conn.execute(text(f"ALTER TABLE `{table}` ADD PRIMARY KEY (id)"))
                logger.info("primary_key_swapped", table=table)

    async def _recreate_indexes(self) -> None:
        if not self.index_definitions:
            return
        async with self.engine.begin() as conn:
            for definition in self.index_definitions.values():
                columns_sql = ", ".join(f"`{col}`" for col in definition.columns)
                unique_sql = "UNIQUE " if definition.unique else ""
                sql = text(
                    f"CREATE {unique_sql}INDEX `{definition.name}` ON `{definition.table}` ({columns_sql})"
                )
                await conn.execute(sql)
                logger.info("index_recreated", table=definition.table, index=definition.name)

    async def _add_foreign_keys(self) -> None:
        async with self.engine.begin() as conn:
            for fk in self.FOREIGN_KEYS:
                if not await self._table_exists(conn, fk.table):
                    continue
                if not await self._column_is_uuid(conn, fk.table, fk.column):
                    continue
                constraint_name = f"fk_{fk.table}_{fk.column}_{fk.ref_table}"
                if await self._foreign_key_exists(conn, fk.table, constraint_name):
                    continue
                ondelete_sql = f" ON DELETE {fk.ondelete}" if fk.ondelete else ""
                sql = text(
                    f"""
                    ALTER TABLE `{fk.table}`
                    ADD CONSTRAINT `{constraint_name}`
                    FOREIGN KEY (`{fk.column}`)
                    REFERENCES `{fk.ref_table}`(id)
                    {ondelete_sql}
                    """
                )
                await conn.execute(sql)
                logger.info("foreign_recreated", table=fk.table, column=fk.column)

    async def _record_migration(self) -> None:
        async with self.engine.begin() as conn:
            await conn.execute(
                text(
                    """
                    CREATE TABLE IF NOT EXISTS tsn_schema_migrations (
                        version VARCHAR(64) PRIMARY KEY,
                        applied_at DATETIME(6) NOT NULL
                    )
                    """
                )
            )
            await conn.execute(
                text(
                    """
                    INSERT INTO tsn_schema_migrations(version, applied_at)
                    VALUES (:version, UTC_TIMESTAMP(6))
                    ON DUPLICATE KEY UPDATE applied_at = VALUES(applied_at)
                    """
                ),
                {"version": "tsn_v2_uuid_migration"},
            )

    # --- metadata helpers ---------------------------------------------------

    async def _table_exists(self, conn: AsyncConnection, table: str) -> bool:
        result = await conn.execute(
            text(
                """
                SELECT 1 FROM INFORMATION_SCHEMA.TABLES
                WHERE TABLE_SCHEMA = :schema AND TABLE_NAME = :table
                """
            ),
            {"schema": self.schema, "table": table},
        )
        return bool(result.scalar())

    async def _column_exists(self, conn: AsyncConnection, table: str, column: str) -> bool:
        result = await conn.execute(
            text(
                """
                SELECT 1 FROM INFORMATION_SCHEMA.COLUMNS
                WHERE TABLE_SCHEMA = :schema
                  AND TABLE_NAME = :table
                  AND COLUMN_NAME = :column
                """
            ),
            {"schema": self.schema, "table": table, "column": column},
        )
        return bool(result.scalar())

    async def _get_column_type(self, conn: AsyncConnection, table: str, column: str) -> str | None:
        if not await self._table_exists(conn, table):
            return None
        result = await conn.execute(
            text(
                """
                SELECT COLUMN_TYPE
                FROM INFORMATION_SCHEMA.COLUMNS
                WHERE TABLE_SCHEMA = :schema
                  AND TABLE_NAME = :table
                  AND COLUMN_NAME = :column
                """
            ),
            {"schema": self.schema, "table": table, "column": column},
        )
        return result.scalar_one_or_none()

    async def _column_is_uuid(self, conn: AsyncConnection, table: str, column: str) -> bool:
        if not await self._column_exists(conn, table, column):
            return False
        result = await conn.execute(
            text(
                """
                SELECT COLUMN_TYPE
                FROM INFORMATION_SCHEMA.COLUMNS
                WHERE TABLE_SCHEMA = :schema
                  AND TABLE_NAME = :table
                  AND COLUMN_NAME = :column
                """
            ),
            {"schema": self.schema, "table": table, "column": column},
        )
        column_type = result.scalar()
        return bool(column_type and "char(36" in column_type.lower())

    async def _resolve_parent_uuid_column(self, conn: AsyncConnection, table: str) -> str | None:
        if await self._column_exists(conn, table, "id_uuid"):
            return "id_uuid"
        if await self._column_is_uuid(conn, table, "id"):
            return "id"
        return None

    async def _ensure_column(self, conn: AsyncConnection, table: str, column: str, definition: str) -> None:
        if await self._column_exists(conn, table, column):
            return
        await conn.execute(text(f"ALTER TABLE `{table}` ADD COLUMN `{column}` {definition}"))

    async def _drop_foreign_keys_for_column(self, conn: AsyncConnection, table: str, column: str) -> None:
        result = await conn.execute(
            text(
                """
                SELECT CONSTRAINT_NAME
                FROM INFORMATION_SCHEMA.KEY_COLUMN_USAGE
                WHERE TABLE_SCHEMA = :schema
                  AND TABLE_NAME = :table
                  AND COLUMN_NAME = :column
                  AND REFERENCED_TABLE_NAME IS NOT NULL
                """
            ),
            {"schema": self.schema, "table": table, "column": column},
        )
        for (constraint_name,) in result.fetchall():
            await conn.execute(text(f"ALTER TABLE `{table}` DROP FOREIGN KEY `{constraint_name}`"))
            logger.info("foreign_dropped", table=table, constraint=constraint_name)

    async def _record_and_drop_indexes(self, conn: AsyncConnection, table: str, column: str) -> None:
        result = await conn.execute(
            text(
                """
                SELECT INDEX_NAME, NON_UNIQUE, SEQ_IN_INDEX, COLUMN_NAME
                FROM INFORMATION_SCHEMA.STATISTICS
                WHERE TABLE_SCHEMA = :schema
                  AND TABLE_NAME = :table
                  AND COLUMN_NAME = :column
                ORDER BY INDEX_NAME, SEQ_IN_INDEX
                """
            ),
            {"schema": self.schema, "table": table, "column": column},
        )
        rows = result.fetchall()
        if not rows:
            return
        grouped: Dict[str, dict] = {}
        for index_name, non_unique, seq, col in rows:
            if index_name == "PRIMARY":
                continue
            entry = grouped.setdefault(
                index_name, {"columns": [], "unique": not bool(non_unique)}
            )
            entry["columns"].append((seq, col))
        for index_name, meta in grouped.items():
            ordered_columns = [col for _, col in sorted(meta["columns"], key=lambda item: item[0])]
            key = (table, index_name)
            if key not in self.index_definitions:
                self.index_definitions[key] = IndexDefinition(
                    table=table,
                    name=index_name,
                    columns=ordered_columns,
                    unique=meta["unique"],
                )
            await conn.execute(text(f"ALTER TABLE `{table}` DROP INDEX `{index_name}`"))
            logger.info("index_dropped", table=table, index=index_name)

    async def _foreign_key_exists(self, conn: AsyncConnection, table: str, constraint: str) -> bool:
        result = await conn.execute(
            text(
                """
                SELECT 1 FROM INFORMATION_SCHEMA.TABLE_CONSTRAINTS
                WHERE TABLE_SCHEMA = :schema
                  AND TABLE_NAME = :table
                  AND CONSTRAINT_NAME = :constraint
                  AND CONSTRAINT_TYPE = 'FOREIGN KEY'
                """
            ),
            {"schema": self.schema, "table": table, "constraint": constraint},
        )
        return bool(result.scalar())


async def async_main() -> None:
    settings = get_settings()
    setup_logging(settings.logging)
    migrator = LegacyUUIDMigrator()
    await migrator.run()
    await close_engine()


if __name__ == "__main__":
    asyncio.run(async_main())
