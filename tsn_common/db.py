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
        _engine = create_async_engine(
            settings.database.url,
            pool_size=settings.database.pool_size,
            max_overflow=settings.database.max_overflow,
            pool_timeout=settings.database.pool_timeout,
            echo=settings.debug,
        )
        logger.info(
            "database_engine_created",
            host=settings.database.host,
            database=settings.database.name,
            pool_size=settings.database.pool_size,
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

    from tsn_common.migrations.legacy_uuid_migrator import LegacyUUIDMigrator

    migrator = LegacyUUIDMigrator()
    await migrator.run()


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
