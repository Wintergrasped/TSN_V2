"""FastAPI entrypoint for the TSN Web Portal."""

from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from starlette.middleware.sessions import SessionMiddleware

from tsn_common.db import get_engine
from tsn_common.models.base import Base
from web import models  # noqa: F401 ensures PortalUser is registered
from web.config import get_web_settings
from web.routes import api, auth, dashboard, profiles

BASE_DIR = Path(__file__).resolve().parent


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

    @app.on_event("startup")
    async def _startup() -> None:  # pragma: no cover - app bootstrap
        engine = get_engine()
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all, checkfirst=True)

    @app.get("/healthz")
    async def healthcheck() -> dict:
        return {"status": "ok"}

    return app


app = create_app()
