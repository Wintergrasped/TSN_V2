"""Common FastAPI dependencies for the portal."""

from collections.abc import AsyncGenerator
from pathlib import Path

from fastapi import Depends, HTTPException, Request, status
from fastapi.templating import Jinja2Templates
from sqlalchemy.ext.asyncio import AsyncSession

from tsn_common.db import get_session
from web.config import get_web_settings
from web.services.users import get_user_by_id

BASE_DIR = Path(__file__).resolve().parent

templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

def _inject_globals(template_env: Jinja2Templates) -> None:
    """Attach shared globals once."""

    template_env.env.globals.setdefault("brand_name", get_web_settings().brand_name)
    template_env.env.globals.setdefault("support_email", get_web_settings().support_email)


_inject_globals(templates)


async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    """Yield an async session tied to the shared TSN database."""

    async with get_session() as session:
        yield session


async def get_current_user(
    request: Request,
    session: AsyncSession = Depends(get_db_session),
):
    """Fetch the PortalUser stored in the session cookie."""

    user_id = request.session.get("user_id")
    if not user_id:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Login required")

    user = await get_user_by_id(session, user_id)
    if user is None:
        request.session.clear()
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid session")
    return user


async def maybe_current_user(
    request: Request,
    session: AsyncSession = Depends(get_db_session),
):
    """Async-friendly optional user resolver."""

    user_id = request.session.get("user_id")
    if not user_id:
        return None
    return await get_user_by_id(session, user_id)
