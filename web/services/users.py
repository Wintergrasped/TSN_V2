"""User management helpers."""

from datetime import datetime, timezone
import uuid

from passlib.context import CryptContext
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from web.models.user import PortalUser

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def hash_password(password: str) -> str:
    """Return a salted hash suitable for storage."""

    return pwd_context.hash(password)


def verify_password(password: str, hashed: str) -> bool:
    """Compare raw password against stored hash."""

    return pwd_context.verify(password, hashed)


async def get_user_by_email(session: AsyncSession, email: str) -> PortalUser | None:
    result = await session.execute(select(PortalUser).where(PortalUser.email == email.lower()))
    return result.scalar_one_or_none()


async def get_user_by_id(session: AsyncSession, user_id: str | uuid.UUID) -> PortalUser | None:
    identifier = uuid.UUID(str(user_id))
    result = await session.execute(select(PortalUser).where(PortalUser.id == identifier))
    return result.scalar_one_or_none()


async def create_user(
    session: AsyncSession,
    *,
    email: str,
    password: str,
    display_name: str,
    callsign: str | None = None,
) -> PortalUser:
    """Create and persist a new PortalUser."""

    normalized_email = email.lower()
    user = PortalUser(
        email=normalized_email,
        display_name=display_name,
        callsign=callsign,
        password_hash=hash_password(password),
    )
    session.add(user)
    await session.flush()
    return user


async def authenticate_user(
    session: AsyncSession,
    *,
    email: str,
    password: str,
) -> PortalUser | None:
    """Validate credentials returning a PortalUser on success."""

    user = await get_user_by_email(session, email)
    if user is None:
        return None
    if not verify_password(password, user.password_hash):
        return None
    return user


async def record_login(session: AsyncSession, user: PortalUser) -> None:
    """Update last_login timestamp."""

    user.last_login_at = datetime.now(timezone.utc)
    await session.flush()
