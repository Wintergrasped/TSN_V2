"""Legacy schema migrator for TSN v2.

This utility detects the pre-v2 MySQL schema (int primary keys, legacy tables)
and performs an in-place migration that:

1. Creates a one-time backup of the entire `repeater` database under a new
   schema name (e.g. `repeater_legacy_20260125_0103`).
2. Drops all legacy tables/views from `repeater` (after the backup succeeds).
3. Recreates the TSN v2 schema using SQLAlchemy metadata.
4. Seeds built-in reference data (phonetic corrections) and records the
   migration in a `tsn_schema_migrations` table to prevent accidental re-runs.

Usage:
    poetry run python scripts/migrate_legacy_schema.py

The script relies on the same settings (.env) as the rest of the services, so
ensure environment variables are loaded/authenticated before running.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Sequence

from sqlalchemy import inspect, text
from sqlalchemy.engine import Connection

from tsn_common import setup_logging
from tsn_common.config import get_settings
from tsn_common.db import close_engine, get_engine
from tsn_common.db_init import seed_phonetic_corrections
from tsn_common.logging import get_logger
from tsn_common.models.base import Base

logger = get_logger(__name__)


@dataclass
class DatabaseInventory:
    """Simple container for database object listings."""

    tables: list[str]
    views: list[str]


class LegacySchemaMigrator:
    """Handles legacy schema detection, backup, and recreation."""

    def __init__(self) -> None:
        self.settings = get_settings()
        self.engine = get_engine()
        self.sync_engine = self.engine.sync_engine
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        self.backup_schema = f"{self.settings.database.name}_legacy_{timestamp}"
        self._inspector = inspect(self.sync_engine)

    async def run(self) -> None:
        if not self._is_legacy_schema():
            logger.info("tsn_schema_migrations_detected", message="Schema already migrated; nothing to do")
            return

        logger.info(
            "legacy_schema_detected",
            target_database=self.settings.database.name,
            backup_schema=self.backup_schema,
        )

        with self.sync_engine.begin() as conn:
            inventory = self._snapshot_inventory(conn)
            self._create_backup_schema(conn)
            self._copy_objects_to_backup(conn, inventory)
            self._drop_views(conn, inventory.views)
            self._drop_tables(conn, inventory.tables)

        await self._create_tsn_schema()
        await seed_phonetic_corrections()
        await self._record_migration()
        logger.info("legacy_schema_migration_complete", backup_schema=self.backup_schema)

    def _is_legacy_schema(self) -> bool:
        """Heuristic to determine whether migration is required."""

        if "tsn_schema_migrations" in self._inspector.get_table_names():
            return False

        try:
            callsign_columns = {
                col["name"].lower(): col for col in self._inspector.get_columns("callsigns")
            }
        except Exception:  # Table missing entirely
            return True

        id_column = callsign_columns.get("id")
        if id_column is None:
            return True

        col_type = id_column["type"].__class__.__name__.lower()
        return "char" not in col_type and "uuid" not in col_type

    def _snapshot_inventory(self, conn: Connection) -> DatabaseInventory:
        """List all tables and views in the active schema."""

        rows = conn.execute(text(f"SHOW FULL TABLES FROM `{self.settings.database.name}`")).fetchall()
        tables: list[str] = []
        views: list[str] = []
        for name, obj_type in rows:
            if obj_type.upper() == "BASE TABLE":
                tables.append(name)
            else:
                views.append(name)
        logger.info(
            "database_inventory_captured",
            tables=len(tables),
            views=len(views),
            backup_schema=self.backup_schema,
        )
        return DatabaseInventory(tables=tables, views=views)

    def _create_backup_schema(self, conn: Connection) -> None:
        conn.execute(text(f"CREATE DATABASE IF NOT EXISTS `{self.backup_schema}`"))
        logger.info("backup_schema_created", schema=self.backup_schema)

    def _copy_objects_to_backup(self, conn: Connection, inventory: DatabaseInventory) -> None:
        logger.info("legacy_copy_started", schema=self.backup_schema)
        for table_name in inventory.tables:
            qualified_src = f"`{self.settings.database.name}`.`{table_name}`"
            qualified_dst = f"`{self.backup_schema}`.`{table_name}`"
            conn.execute(text(f"CREATE TABLE {qualified_dst} LIKE {qualified_src}"))
            conn.execute(text(f"INSERT INTO {qualified_dst} SELECT * FROM {qualified_src}"))
        logger.info("legacy_copy_completed", tables=len(inventory.tables))

        if inventory.views:
            logger.warning(
                "legacy_views_preserved",
                count=len(inventory.views),
                artifact="tsn_legacy_views",
                note="View definitions stored as SQL text inside backup schema",
            )
            conn.execute(
                text(
                    f"CREATE TABLE IF NOT EXISTS `{self.backup_schema}`.`tsn_legacy_views` ("
                    " view_name VARCHAR(255) PRIMARY KEY,"
                    " definition LONGTEXT NOT NULL"
                    ")"
                )
            )
            for view_name in inventory.views:
                result = conn.execute(
                    text(f"SHOW CREATE VIEW `{self.settings.database.name}`.`{view_name}`")
                ).fetchone()
                create_sql = result[1] if result and len(result) > 1 else None
                if not create_sql:
                    continue
                conn.execute(
                    text(
                        f"REPLACE INTO `{self.backup_schema}`.`tsn_legacy_views`(view_name, definition)"
                        " VALUES (:view, :definition)"
                    ),
                    {"view": view_name, "definition": create_sql},
                )

    def _drop_views(self, conn: Connection, views: Sequence[str]) -> None:
        if not views:
            return
        conn.execute(text("SET FOREIGN_KEY_CHECKS=0"))
        for view_name in views:
            conn.execute(text(f"DROP VIEW IF EXISTS `{self.settings.database.name}`.`{view_name}`"))
        conn.execute(text("SET FOREIGN_KEY_CHECKS=1"))
        logger.info("legacy_views_dropped", count=len(views))

    def _drop_tables(self, conn: Connection, tables: Sequence[str]) -> None:
        if not tables:
            return
        conn.execute(text("SET FOREIGN_KEY_CHECKS=0"))
        for table_name in tables:
            conn.execute(text(f"DROP TABLE IF EXISTS `{self.settings.database.name}`.`{table_name}`"))
        conn.execute(text("SET FOREIGN_KEY_CHECKS=1"))
        logger.info("legacy_tables_dropped", count=len(tables))

    async def _create_tsn_schema(self) -> None:
        logger.info("tsn_schema_recreation_started")
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        logger.info("tsn_schema_recreation_complete")

    async def _record_migration(self) -> None:
        async with self.engine.begin() as conn:
            await conn.execute(
                text(
                    "CREATE TABLE IF NOT EXISTS tsn_schema_migrations ("
                    " version VARCHAR(64) PRIMARY KEY,"
                    " applied_at DATETIME(6) NOT NULL"
                    ")"
                )
            )
            await conn.execute(
                text(
                    "INSERT INTO tsn_schema_migrations(version, applied_at)"
                    " VALUES (:version, UTC_TIMESTAMP(6))"
                    " ON DUPLICATE KEY UPDATE applied_at = VALUES(applied_at)"
                ),
                {"version": "tsn_v2_initial"},
            )
        logger.info("tsn_schema_version_recorded", version="tsn_v2_initial")


async def async_main() -> None:
    settings = get_settings()
    setup_logging(settings.logging)

    migrator = LegacySchemaMigrator()
    await migrator.run()
    await close_engine()


if __name__ == "__main__":
    asyncio.run(async_main())
