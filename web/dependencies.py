"""Common FastAPI dependencies for the portal."""

from collections.abc import AsyncGenerator
from pathlib import Path
from urllib.parse import quote_plus

from fastapi import Depends, HTTPException, Request, status
from fastapi.templating import Jinja2Templates
from markupsafe import Markup, escape
from sqlalchemy.ext.asyncio import AsyncSession

from tsn_common.db import get_session
from web.config import get_web_settings
from web.services.users import get_user_by_id

BASE_DIR = Path(__file__).resolve().parent

templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))


def _callsign_link(value: str | None, css_class: str | None = None) -> Markup:
    if not value:
        return Markup("")
    href = f"/callsigns/{quote_plus(value.strip().upper())}"
    class_attr = f' class="{css_class}"' if css_class else ""
    return Markup(f"<a{class_attr} href=\"{href}\">{escape(value)}</a>")


def _inject_globals(template_env: Jinja2Templates) -> None:
    """Attach shared globals once."""

    template_env.env.globals.setdefault("brand_name", get_web_settings().brand_name)
    template_env.env.globals.setdefault("support_email", get_web_settings().support_email)
    template_env.env.globals.setdefault("callsign_link", _callsign_link)


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
