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

MIGRATION_VERSION = "tsn_v2_uuid_migration"


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
    sub_parts: List[int | None]
    unique: bool
    index_type: str | None
    visible: bool | None


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
        self._supports_index_visibility: bool | None = None

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
            if await self._migration_already_recorded(conn):
                return False
            for table in self.TABLES:
                if not await self._table_exists(conn, table):
                    continue
                if not await self._column_is_uuid(conn, table, "id"):
                    return True
            for fk in self.FOREIGN_KEYS:
                if not await self._table_exists(conn, fk.table):
                    continue
                if not await self._column_exists(conn, fk.table, fk.column):
                    continue
                if not await self._column_is_uuid(conn, fk.table, fk.column):
                    return True
        return False

    async def _prepare_uuid_columns(self) -> None:
        async with self.engine.begin() as conn:
            # Set lock wait timeout to 10 seconds to fail fast if tables are locked
            await conn.execute(text("SET SESSION lock_wait_timeout = 10"))
            
            for table in self.TABLES:
                if not await self._table_exists(conn, table):
                    continue
                if not await self._column_exists(conn, table, "id"):
                    logger.info("uuid_column_skipped", table=table, reason="id_missing")
                    continue
                if await self._column_is_uuid(conn, table, "id"):
                    continue
                await self._ensure_column(conn, table, "id_uuid", "CHAR(36) NULL")
                await conn.execute(text(f"UPDATE `{table}` SET id_uuid = UUID() WHERE id_uuid IS NULL"))
                remaining = await self._count_null_values(conn, table, "id_uuid")
                null_sql = "NULL" if remaining else "NOT NULL"
                await conn.execute(
                    text(f"ALTER TABLE `{table}` MODIFY COLUMN id_uuid CHAR(36) {null_sql}")
                )
                if remaining:
                    logger.warning(
                        "uuid_column_partial",
                        table=table,
                        rows_without_uuid=remaining,
                    )
                else:
                    logger.info("uuid_column_populated", table=table)

    async def _prepare_foreign_key_columns(self) -> None:
        async with self.engine.begin() as conn:
            for fk in self.FOREIGN_KEYS:
                await self._prepare_foreign_key_helper(conn, fk)

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
                    prepared = await self._prepare_foreign_key_helper(conn, fk)
                    if not prepared:
                        logger.info(
                            "foreign_swap_skipped",
                            table=fk.table,
                            column=fk.column,
                            reason="helper_missing",
                        )
                        continue
                await self._record_and_drop_indexes(conn, fk.table, fk.column)
                await conn.execute(text(f"ALTER TABLE `{fk.table}` DROP COLUMN `{fk.column}`"))
                await conn.execute(
                    text(
                        f"ALTER TABLE `{fk.table}` CHANGE COLUMN `{helper_column}` `{fk.column}` CHAR(36) NULL"
                    )
                )
                if not fk.nullable:
                    null_rows = await self._count_null_values(conn, fk.table, fk.column)
                    if null_rows == 0:
                        await conn.execute(
                            text(
                                f"ALTER TABLE `{fk.table}` MODIFY COLUMN `{fk.column}` CHAR(36) NOT NULL"
                            )
                        )
                    else:
                        logger.warning(
                            "foreign_column_nullable_due_to_orphans",
                            table=fk.table,
                            column=fk.column,
                            orphan_rows=null_rows,
                        )
                logger.info("foreign_column_swapped", table=fk.table, column=fk.column)

    async def _swap_primary_keys(self) -> None:
        async with self.engine.begin() as conn:
            for table in self.TABLES:
                if not await self._table_exists(conn, table):
                    continue
                if not await self._column_exists(conn, table, "id_uuid"):
                    continue
                if not await self._column_exists(conn, table, "id"):
                    logger.info("primary_swap_skipped", table=table, reason="id_missing")
                    continue
                if await self._column_is_uuid(conn, table, "id"):
                    logger.info(
                        "primary_swap_skipped",
                        table=table,
                        reason="already_uuid",
                    )
                    continue
                await conn.execute(text(f"UPDATE `{table}` SET id_uuid = UUID() WHERE id_uuid IS NULL"))
                remaining = await self._count_null_values(conn, table, "id_uuid")
                if remaining:
                    logger.warning(
                        "primary_swap_deferred",
                        table=table,
                        rows_without_uuid=remaining,
                    )
                    continue
                await self._drop_referencing_foreign_keys(conn, table)
                await self._record_and_drop_indexes(conn, table, "id")
                if await self._column_has_auto_increment(conn, table, "id"):
                    column_type = await self._get_column_type(conn, table, "id")
                    if column_type is None:
                        logger.error("primary_swap_type_unknown", table=table)
                        continue
                    await conn.execute(
                        text(
                            f"ALTER TABLE `{table}` MODIFY COLUMN id {column_type} NOT NULL"
                        )
                    )
                if await self._table_has_primary_key(conn, table):
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
                column_clauses: List[str] = []
                for column, sub_part in zip(definition.columns, definition.sub_parts):
                    if sub_part:
                        column_clauses.append(f"`{column}`({sub_part})")
                    else:
                        column_clauses.append(f"`{column}`")
                columns_sql = ", ".join(column_clauses)
                unique_sql = "UNIQUE " if definition.unique else ""
                using_sql = f" USING {definition.index_type}" if definition.index_type else ""
                visible_sql = " INVISIBLE" if definition.visible is False else ""
                sql = text(
                    f"CREATE {unique_sql}INDEX `{definition.name}` ON `{definition.table}`{using_sql} ({columns_sql}){visible_sql}"
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
                if not await self._table_exists(conn, fk.ref_table):
                    continue
                if not await self._column_is_uuid(conn, fk.ref_table, "id"):
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
                {"version": MIGRATION_VERSION},
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

    async def _column_has_auto_increment(self, conn: AsyncConnection, table: str, column: str) -> bool:
        result = await conn.execute(
            text(
                """
                SELECT EXTRA
                FROM INFORMATION_SCHEMA.COLUMNS
                WHERE TABLE_SCHEMA = :schema
                  AND TABLE_NAME = :table
                  AND COLUMN_NAME = :column
                """
            ),
            {"schema": self.schema, "table": table, "column": column},
        )
        extra = result.scalar()
        return bool(extra and "auto_increment" in extra.lower())

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

    async def _drop_referencing_foreign_keys(self, conn: AsyncConnection, table: str) -> None:
        result = await conn.execute(
            text(
                """
                SELECT k.TABLE_NAME, k.CONSTRAINT_NAME
                FROM INFORMATION_SCHEMA.KEY_COLUMN_USAGE k
                JOIN INFORMATION_SCHEMA.TABLE_CONSTRAINTS c
                  ON c.CONSTRAINT_NAME = k.CONSTRAINT_NAME
                 AND c.TABLE_NAME = k.TABLE_NAME
                 AND c.TABLE_SCHEMA = k.TABLE_SCHEMA
                WHERE k.TABLE_SCHEMA = :schema
                  AND k.REFERENCED_TABLE_NAME = :table
                  AND c.CONSTRAINT_TYPE = 'FOREIGN KEY'
                GROUP BY k.TABLE_NAME, k.CONSTRAINT_NAME
                """
            ),
            {"schema": self.schema, "table": table},
        )
        rows = result.fetchall()
        if not rows:
            return
        for child_table, constraint_name in rows:
            await conn.execute(text(f"ALTER TABLE `{child_table}` DROP FOREIGN KEY `{constraint_name}`"))
            logger.info(
                "foreign_dropped_parent",
                child_table=child_table,
                constraint=constraint_name,
                referenced_table=table,
            )

    async def _record_and_drop_indexes(self, conn: AsyncConnection, table: str, column: str) -> None:
        supports_visibility = await self._database_supports_index_visibility(conn)
        visibility_select = ", IS_VISIBLE" if supports_visibility else ""
        result = await conn.execute(
            text(
                f"""
                SELECT INDEX_NAME, NON_UNIQUE, SEQ_IN_INDEX, COLUMN_NAME, SUB_PART, INDEX_TYPE{visibility_select}
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
        for row in rows:
            if supports_visibility:
                index_name, non_unique, seq, col, sub_part, index_type, is_visible = row
            else:
                index_name, non_unique, seq, col, sub_part, index_type = row
                is_visible = None
            if index_name == "PRIMARY":
                continue
            if index_name not in grouped:
                grouped[index_name] = {
                    "columns": [],
                    "unique": not bool(non_unique),
                    "index_type": index_type,
                    "visible": None if is_visible is None else str(is_visible).upper() == "YES",
                }
            grouped[index_name]["columns"].append((seq, col, sub_part))
        for index_name, meta in grouped.items():
            ordered = sorted(meta["columns"], key=lambda item: item[0])
            column_names = [value for _, value, _ in ordered]
            sub_parts = [value for _, _, value in ordered]
            key = (table, index_name)
            if key not in self.index_definitions:
                self.index_definitions[key] = IndexDefinition(
                    table=table,
                    name=index_name,
                    columns=column_names,
                    sub_parts=sub_parts,
                    unique=meta["unique"],
                    index_type=meta["index_type"],
                    visible=meta["visible"],
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

    async def _prepare_foreign_key_helper(self, conn: AsyncConnection, fk: ForeignKeySpec) -> bool:
        if not await self._table_exists(conn, fk.table):
            return False
        if not await self._column_exists(conn, fk.table, fk.column):
            logger.info(
                "foreign_helper_skipped",
                table=fk.table,
                column=fk.column,
                reason="column_missing",
            )
            return False
        if await self._column_is_uuid(conn, fk.table, fk.column):
            return False
        parent_uuid_column = await self._resolve_parent_uuid_column(conn, fk.ref_table)
        if parent_uuid_column is None:
            logger.warning(
                "foreign_helper_skipped",
                table=fk.table,
                column=fk.column,
                reason="parent_uuid_missing",
            )
            return False
        helper_column = f"{fk.column}_uuid"
        await self._ensure_column(conn, fk.table, helper_column, "CHAR(36) NULL")
        
        # Batch the UPDATE to avoid locking entire table for large tables
        batch_size = 1000
        total_updated = 0
        while True:
            result = await conn.execute(
                text(
                    f"""
                    UPDATE `{fk.table}` child
                    JOIN `{fk.ref_table}` parent ON child.`{fk.column}` = parent.`id`
                    SET child.`{helper_column}` = parent.`{parent_uuid_column}`
                    WHERE child.`{fk.column}` IS NOT NULL
                      AND child.`{helper_column}` IS NULL
                    LIMIT {batch_size}
                    """
                )
            )
            rows_updated = result.rowcount
            total_updated += rows_updated
            if rows_updated == 0:
                break
            # Log progress for large tables
            if total_updated % 5000 == 0:
                logger.info(
                    "foreign_helper_progress",
                    table=fk.table,
                    column=fk.column,
                    rows_updated=total_updated,
                )
        
        if total_updated > 0:
            logger.info(
                "foreign_helper_completed",
                table=fk.table,
                column=fk.column,
                total_rows=total_updated,
            )
        
        orphan_count = await self._count_unmapped_helper_rows(conn, fk.table, helper_column, fk.column)
        nullable = fk.nullable or orphan_count > 0
        null_sql = "NULL" if nullable else "NOT NULL"
        await conn.execute(
            text(
                f"ALTER TABLE `{fk.table}` MODIFY COLUMN `{helper_column}` CHAR(36) {null_sql}"
            )
        )
        if orphan_count and not fk.nullable:
            logger.warning(
                "foreign_helper_orphans",
                table=fk.table,
                column=fk.column,
                orphan_rows=orphan_count,
            )
        logger.info("foreign_helper_prepared", table=fk.table, column=fk.column)
        return True

    async def _count_unmapped_helper_rows(
        self,
        conn: AsyncConnection,
        table: str,
        helper_column: str,
        legacy_column: str,
    ) -> int:
        result = await conn.execute(
            text(
                f"""
                SELECT COUNT(*)
                FROM `{table}`
                WHERE `{legacy_column}` IS NOT NULL
                  AND `{helper_column}` IS NULL
                """
            )
        )
        count = result.scalar()
        return int(count or 0)

    async def _count_null_values(self, conn: AsyncConnection, table: str, column: str) -> int:
        result = await conn.execute(
            text(
                f"SELECT COUNT(*) FROM `{table}` WHERE `{column}` IS NULL"
            )
        )
        count = result.scalar()
        return int(count or 0)

    async def _table_has_primary_key(self, conn: AsyncConnection, table: str) -> bool:
        result = await conn.execute(
            text(
                """
                SELECT 1
                FROM INFORMATION_SCHEMA.TABLE_CONSTRAINTS
                WHERE TABLE_SCHEMA = :schema
                  AND TABLE_NAME = :table
                  AND CONSTRAINT_TYPE = 'PRIMARY KEY'
                """
            ),
            {"schema": self.schema, "table": table},
        )
        return bool(result.scalar())

    async def _database_supports_index_visibility(self, conn: AsyncConnection) -> bool:
        if self._supports_index_visibility is not None:
            return self._supports_index_visibility
        result = await conn.execute(
            text(
                """
                SELECT 1
                FROM INFORMATION_SCHEMA.COLUMNS
                WHERE TABLE_SCHEMA = 'information_schema'
                  AND TABLE_NAME = 'STATISTICS'
                  AND COLUMN_NAME = 'IS_VISIBLE'
                """
            )
        )
        self._supports_index_visibility = bool(result.scalar())
        return self._supports_index_visibility

    async def _migration_already_recorded(self, conn: AsyncConnection) -> bool:
        if not await self._table_exists(conn, "tsn_schema_migrations"):
            return False
        result = await conn.execute(
            text(
                """
                SELECT 1
                FROM tsn_schema_migrations
                WHERE version = :version
                """
            ),
            {"version": MIGRATION_VERSION},
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
