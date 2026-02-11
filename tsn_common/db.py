"""Database connection and session management."""

import asyncio
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from tsn_common.config import DatabaseSettings, get_settings
from tsn_common.logging import get_logger
from tsn_common.models.base import Base

logger = get_logger(__name__)

_engine: AsyncEngine | None = None
_session_factory: async_sessionmaker[AsyncSession] | None = None
_schema_ready = False
_schema_lock = asyncio.Lock()


def get_engine() -> AsyncEngine:
    """Get or create the database engine."""
    global _engine
    if _engine is None:
        settings = get_settings()
        
        # Add connection arguments to prevent infinite hangs on table locks
        connect_args = {
            "connect_timeout": 10,  # Connection timeout in seconds
            "read_timeout": 30,     # Query read timeout in seconds
            "write_timeout": 30,    # Query write timeout in seconds
        }
        
        _engine = create_async_engine(
            settings.database.url,
            pool_size=settings.database.pool_size,
            max_overflow=settings.database.max_overflow,
            pool_timeout=settings.database.pool_timeout,
            connect_args=connect_args,
            echo=settings.debug,
        )
        logger.info(
            "database_engine_created",
            host=settings.database.host,
            database=settings.database.name,
            pool_size=settings.database.pool_size,
            connect_timeout=10,
            query_timeout=30,
        )
    return _engine


def get_session_factory() -> async_sessionmaker[AsyncSession]:
    """Get or create the session factory."""
    global _session_factory
    if _session_factory is None:
        engine = get_engine()
        _session_factory = async_sessionmaker(
            engine,
            class_=AsyncSession,
            expire_on_commit=False,
        )
    return _session_factory


@asynccontextmanager
async def get_session() -> AsyncGenerator[AsyncSession, None]:
    """
    Get a database session.
    
    Usage:
        async with get_session() as session:
            result = await session.execute(select(AudioFile))
    """
    factory = get_session_factory()
    await _ensure_schema_ready()
    async with factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


async def close_engine() -> None:
    """Close the database engine (cleanup)."""
    global _engine, _session_factory
    if _engine is not None:
        await _engine.dispose()
        _engine = None
        _session_factory = None
        logger.info("database_engine_closed")


async def _run_bootstrap_migrations() -> None:
    """Run idempotent migrations that keep legacy installs current."""
    
    # MIGRATIONS DISABLED - causing table locks on startup
    # The UUID migration is taking too long and locking transcriptions table
    # Re-enable after migration is fully debugged or run manually offline
    logger.warning("bootstrap_migrations_disabled", reason="table_lock_prevention")
    return

    # from tsn_common.migrations.legacy_uuid_migrator import LegacyUUIDMigrator
    # from tsn_common.migrations.metrics_expansion import MetricsExpansionMigrator

    # migrators = [LegacyUUIDMigrator(), MetricsExpansionMigrator()]
    # for migrator in migrators:
    #     await migrator.run()


async def _ensure_schema_ready() -> None:
    """Create missing tables and execute migrations exactly once per process."""

    global _schema_ready
    if _schema_ready:
        return
    async with _schema_lock:
        if _schema_ready:
            return

        engine = get_engine()
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all, checkfirst=True)

        await _run_bootstrap_migrations()

        _schema_ready = True
        logger.info("database_schema_verified")
