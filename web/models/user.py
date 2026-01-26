"""Portal-specific user table for authentication."""

from datetime import datetime

from sqlalchemy import Boolean, DateTime, Index, JSON, String
from sqlalchemy.orm import Mapped, mapped_column

from tsn_common.models.base import Base


class PortalUser(Base):
    """Simple credential store for the web experience."""

    __tablename__ = "portal_users"

    email: Mapped[str] = mapped_column(String(255), nullable=False, unique=True, index=True)
    callsign: Mapped[str | None] = mapped_column(String(32), nullable=True, index=True)
    display_name: Mapped[str] = mapped_column(String(100), nullable=False)
    password_hash: Mapped[str] = mapped_column(String(255), nullable=False)
    is_admin: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    last_login_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    preferences: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)

    __table_args__ = (
        Index("ix_portal_users_callsign", "callsign"),
    )

    def __repr__(self) -> str:  # pragma: no cover - debugging helper
        return f"<PortalUser(email={self.email!r}, callsign={self.callsign!r})>"
