"""Automated in-place migration for legacy INT primary keys to UUIDs."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Dict, List, Sequence

from sqlalchemy import text
from sqlalchemy.engine import Connection

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
        self.engine = get_engine()
        self.sync_engine = self.engine.sync_engine
        self.index_definitions: Dict[tuple[str, str], IndexDefinition] = {}

    def run(self) -> None:
        if not self._needs_migration():
            logger.info("legacy_uuid_migration_skipped", reason="schema already matches models")
            return

        logger.info("legacy_uuid_migration_started", schema=self.schema)
        self._prepare_uuid_columns()
        self._prepare_foreign_key_columns()
        self._drop_legacy_foreign_keys()
        self._swap_foreign_key_columns()
        self._swap_primary_keys()
        self._recreate_indexes()
        self._add_foreign_keys()
        self._record_migration()
        logger.info("legacy_uuid_migration_complete", schema=self.schema)

    # --- high-level helpers -------------------------------------------------

    def _needs_migration(self) -> bool:
        column = self._get_column_type("callsigns", "id")
        if column is None:
            return False
        return "char" not in column.lower()

    def _prepare_uuid_columns(self) -> None:
        with self.sync_engine.begin() as conn:
            for table in self.TABLES:
                if not self._table_exists(conn, table):
                    continue
                if self._column_is_uuid(conn, table, "id"):
                    continue
                self._ensure_column(conn, table, "id_uuid", "CHAR(36) NULL")
                conn.execute(
                    text(f"UPDATE `{table}` SET id_uuid = UUID() WHERE id_uuid IS NULL")
                )
                conn.execute(
                    text(f"ALTER TABLE `{table}` MODIFY COLUMN id_uuid CHAR(36) NOT NULL")
                )
                logger.info("uuid_column_populated", table=table)

    def _prepare_foreign_key_columns(self) -> None:
        with self.sync_engine.begin() as conn:
            for fk in self.FOREIGN_KEYS:
                if not self._table_exists(conn, fk.table):
                    continue
                if self._column_is_uuid(conn, fk.table, fk.column):
                    continue
                parent_has_uuid = self._column_exists(conn, fk.ref_table, "id_uuid")
                if not parent_has_uuid:
                    logger.error("parent_uuid_missing", table=fk.ref_table)
                    raise RuntimeError(f"Parent table {fk.ref_table} missing id_uuid helper column")
                helper_column = f"{fk.column}_uuid"
                self._ensure_column(conn, fk.table, helper_column, "CHAR(36) NULL")
                update_sql = text(
                    f"""
                    UPDATE `{fk.table}` child
                    JOIN `{fk.ref_table}` parent ON child.`{fk.column}` = parent.`id`
                    SET child.`{helper_column}` = parent.`id_uuid`
                    WHERE child.`{fk.column}` IS NOT NULL
                      AND child.`{helper_column}` IS NULL
                    """
                )
                conn.execute(update_sql)
                null_sql = "NULL" if fk.nullable else "NOT NULL"
                conn.execute(
                    text(
                        f"ALTER TABLE `{fk.table}` MODIFY COLUMN `{helper_column}` CHAR(36) {null_sql}"
                    )
                )
                logger.info("foreign_helper_populated", table=fk.table, column=fk.column)

    def _drop_legacy_foreign_keys(self) -> None:
        with self.sync_engine.begin() as conn:
            for fk in self.FOREIGN_KEYS:
                self._drop_foreign_keys_for_column(conn, fk.table, fk.column)

    def _swap_foreign_key_columns(self) -> None:
        with self.sync_engine.begin() as conn:
            for fk in self.FOREIGN_KEYS:
                if self._column_is_uuid(conn, fk.table, fk.column):
                    continue
                helper_column = f"{fk.column}_uuid"
                if not self._column_exists(conn, fk.table, helper_column):
                    raise RuntimeError(f"Helper column {helper_column} missing on {fk.table}")
                self._record_and_drop_indexes(conn, fk.table, fk.column)
                conn.execute(text(f"ALTER TABLE `{fk.table}` DROP COLUMN `{fk.column}`"))
                null_sql = "NULL" if fk.nullable else "NOT NULL"
                conn.execute(
                    text(
                        f"ALTER TABLE `{fk.table}` CHANGE COLUMN `{helper_column}` `{fk.column}` CHAR(36) {null_sql}"
                    )
                )
                logger.info("foreign_column_swapped", table=fk.table, column=fk.column)

    def _swap_primary_keys(self) -> None:
        with self.sync_engine.begin() as conn:
            for table in self.TABLES:
                if not self._table_exists(conn, table):
                    continue
                if not self._column_exists(conn, table, "id_uuid"):
                    continue
                self._record_and_drop_indexes(conn, table, "id")
                conn.execute(text(f"ALTER TABLE `{table}` DROP PRIMARY KEY"))
                conn.execute(text(f"ALTER TABLE `{table}` DROP COLUMN id"))
                conn.execute(
                    text(
                        f"ALTER TABLE `{table}` CHANGE COLUMN id_uuid id CHAR(36) NOT NULL"
                    )
                )
                conn.execute(text(f"ALTER TABLE `{table}` ADD PRIMARY KEY (id)"))
                logger.info("primary_key_swapped", table=table)

    def _recreate_indexes(self) -> None:
        if not self.index_definitions:
            return
        with self.sync_engine.begin() as conn:
            for definition in self.index_definitions.values():
                columns_sql = ", ".join(f"`{col}`" for col in definition.columns)
                unique_sql = "UNIQUE " if definition.unique else ""
                sql = text(
                    f"CREATE {unique_sql}INDEX `{definition.name}` ON `{definition.table}` ({columns_sql})"
                )
                conn.execute(sql)
                logger.info("index_recreated", table=definition.table, index=definition.name)

    def _add_foreign_keys(self) -> None:
        with self.sync_engine.begin() as conn:
            for fk in self.FOREIGN_KEYS:
                if not self._table_exists(conn, fk.table):
                    continue
                if not self._column_is_uuid(conn, fk.table, fk.column):
                    continue
                constraint_name = f"fk_{fk.table}_{fk.column}_{fk.ref_table}"
                if self._foreign_key_exists(conn, fk.table, constraint_name):
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
                conn.execute(sql)
                logger.info("foreign_recreated", table=fk.table, column=fk.column)

    def _record_migration(self) -> None:
        with self.sync_engine.begin() as conn:
            conn.execute(
                text(
                    """
                    CREATE TABLE IF NOT EXISTS tsn_schema_migrations (
                        version VARCHAR(64) PRIMARY KEY,
                        applied_at DATETIME(6) NOT NULL
                    )
                    """
                )
            )
            conn.execute(
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

    def _table_exists(self, conn: Connection, table: str) -> bool:
        result = conn.execute(
            text(
                """
                SELECT 1 FROM INFORMATION_SCHEMA.TABLES
                WHERE TABLE_SCHEMA = :schema AND TABLE_NAME = :table
                """
            ),
            {"schema": self.schema, "table": table},
        ).scalar()
        return bool(result)

    def _column_exists(self, conn: Connection, table: str, column: str) -> bool:
        result = conn.execute(
            text(
                """
                SELECT 1 FROM INFORMATION_SCHEMA.COLUMNS
                WHERE TABLE_SCHEMA = :schema
                  AND TABLE_NAME = :table
                  AND COLUMN_NAME = :column
                """
            ),
            {"schema": self.schema, "table": table, "column": column},
        ).scalar()
        return bool(result)

    def _get_column_type(self, table: str, column: str) -> str | None:
        with self.sync_engine.begin() as conn:
            row = conn.execute(
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
            ).scalar()
            return row

    def _column_is_uuid(self, conn: Connection, table: str, column: str) -> bool:
        if not self._column_exists(conn, table, column):
            return False
        column_type = conn.execute(
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
        ).scalar()
        return bool(column_type and "char(36" in column_type.lower())

    def _ensure_column(self, conn: Connection, table: str, column: str, definition: str) -> None:
        if self._column_exists(conn, table, column):
            return
        conn.execute(text(f"ALTER TABLE `{table}` ADD COLUMN `{column}` {definition}"))

    def _drop_foreign_keys_for_column(self, conn: Connection, table: str, column: str) -> None:
        rows = conn.execute(
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
        ).fetchall()
        for (constraint_name,) in rows:
            conn.execute(text(f"ALTER TABLE `{table}` DROP FOREIGN KEY `{constraint_name}`"))
            logger.info("foreign_dropped", table=table, constraint=constraint_name)

    def _record_and_drop_indexes(self, conn: Connection, table: str, column: str) -> None:
        rows = conn.execute(
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
        ).fetchall()
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
            conn.execute(text(f"ALTER TABLE `{table}` DROP INDEX `{index_name}`"))
            logger.info("index_dropped", table=table, index=index_name)

    def _foreign_key_exists(self, conn: Connection, table: str, constraint: str) -> bool:
        row = conn.execute(
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
        ).scalar()
        return bool(row)


async def async_main() -> None:
    settings = get_settings()
    setup_logging(settings.logging)
    migrator = LegacyUUIDMigrator()
    migrator.run()
    await close_engine()


if __name__ == "__main__":
    asyncio.run(async_main())
