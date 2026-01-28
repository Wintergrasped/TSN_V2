"""FastAPI entrypoint for the TSN Web Portal."""

import asyncio
import re
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from sqlalchemy import select, update
from starlette.middleware.sessions import SessionMiddleware

from tsn_common.db import get_engine, async_session_maker
from tsn_common.logging import get_logger
from tsn_common.models.audio import AudioFile
from tsn_common.models.base import Base
from web import models  # noqa: F401 ensures PortalUser is registered
from web.config import get_web_settings
from web.routes import api, auth, dashboard, profiles

BASE_DIR = Path(__file__).resolve().parent
logger = get_logger(__name__)

# Pattern to match node_id from filename: NODEID_timestamp.wav (e.g., 66296_2026012520595573.WAV)
NODE_FILENAME_PATTERN = re.compile(r'^(\d+)_\d+\.wav$', re.IGNORECASE)


async def repair_node_ids_periodic() -> None:
    """
    Background task that periodically checks audio_files table for missing node_ids.
    Extracts node_id from filename pattern: node_X_timestamp.wav
    Runs every 5 minutes.
    """
    while True:
        try:
            await asyncio.sleep(300)  # Wait 5 minutes between checks
            
            async with async_session_maker() as session:
                # Find audio files with NULL or 'unknown' node_id
                stmt = select(AudioFile).where(
                    (AudioFile.node_id.is_(None)) | (AudioFile.node_id == "unknown")
                )
                result = await session.execute(stmt)
                files = result.scalars().all()
                
                if not files:
                    continue
                
                repaired_count = 0
                for audio_file in files:
                    match = NODE_FILENAME_PATTERN.match(audio_file.filename)
                    if match:
                        node_id = match.group(1)
                        audio_file.node_id = node_id
                        repaired_count += 1
                
                if repaired_count > 0:
                    await session.commit()
                    logger.info(
                        "node_id_repair_completed",
                        repaired=repaired_count,
                        total_checked=len(files)
                    )
                    
        except asyncio.CancelledError:
            logger.info("node_id_repair_task_cancelled")
            raise
        except Exception as exc:
            logger.error("node_id_repair_task_error", error=str(exc))
            # Continue running despite errors
            await asyncio.sleep(60)  # Wait 1 minute before retry on error


def create_app() -> FastAPI:
    settings = get_web_settings()

    app = FastAPI(
        title=f"{settings.brand_name} Network Dashboard",
        description="Modern portal for TSN data",
        version="2.0.0",
        docs_url="/api/docs",
        redoc_url="/api/redoc",
    )

    app.include_router(dashboard.router)
    app.include_router(profiles.router)
    app.include_router(api.router)
    app.include_router(auth.router)

    app.add_middleware(
        SessionMiddleware,
        secret_key=settings.session_secret,
        https_only=False,
        max_age=60 * 60 * 24 * 7,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    static_dir = BASE_DIR / "static"
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    # Background task for node_id repair
    background_task = None

    @app.on_event("startup")
    async def _startup() -> None:  # pragma: no cover - app bootstrap
        nonlocal background_task
        engine = get_engine()
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all, checkfirst=True)
        
        # Start background node_id repair task
        background_task = asyncio.create_task(repair_node_ids_periodic())
        logger.info("node_id_repair_task_started")

    @app.on_event("shutdown")
    async def _shutdown() -> None:  # pragma: no cover - app teardown
        if background_task:
            background_task.cancel()
            try:
                await background_task
            except asyncio.CancelledError:
                pass
            logger.info("node_id_repair_task_stopped")

    @app.get("/healthz")
    async def healthcheck() -> dict:
        return {"status": "ok"}

    return app


app = create_app()
