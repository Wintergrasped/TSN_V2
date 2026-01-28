"""User management helpers."""

from datetime import datetime, timezone
import uuid

from passlib.context import CryptContext
from passlib.exc import PasswordSizeError
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from web.models.user import PortalUser

pwd_context = CryptContext(
    schemes=["argon2", "bcrypt_sha256"],
    deprecated="auto",
)

_MIN_PASSWORD_LEN = 8
_MAX_PASSWORD_LEN = 256


def _validate_password(password: str) -> str:
    if not isinstance(password, str):
        raise ValueError("Password must be a string")
    if len(password) < _MIN_PASSWORD_LEN:
        raise ValueError("Password must be at least 8 characters long")
    if len(password) > _MAX_PASSWORD_LEN:
        raise ValueError("Password must be 256 characters or fewer")
    return password


def hash_password(password: str) -> str:
    """Return a salted hash suitable for storage."""

    normalized = _validate_password(password)
    try:
        return pwd_context.hash(normalized)
    except PasswordSizeError as exc:  # pragma: no cover - defensive guard
        raise ValueError("Password must be 256 characters or fewer") from exc


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
    safe_display = display_name.strip()
    if not safe_display:
        raise ValueError("Display name is required")
    safe_callsign = callsign.strip().upper() if callsign else None
    user = PortalUser(
        email=normalized_email,
        display_name=safe_display,
        callsign=safe_callsign,
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
