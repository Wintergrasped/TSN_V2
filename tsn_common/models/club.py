"""Club models - identifies organizations mentioned in nets and transcripts."""

import enum
import uuid
from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import DateTime, Enum, ForeignKey, Index, JSON, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from tsn_common.models.base import Base, GUID

if TYPE_CHECKING:
    from tsn_common.models.callsign import Callsign
    from tsn_common.models.transcription import Transcription


class ClubRole(str, enum.Enum):
    """Role a callsign plays inside a club/net organization."""

    MEMBER = "member"
    NCS = "ncs"
    GUEST = "guest"


class ClubProfile(Base):
    """Aggregated profile for a club or organization detected in transcripts."""

    __tablename__ = "club_profiles"

    name: Mapped[str] = mapped_column(String(255), nullable=False, unique=True, index=True)
    summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    schedule: Mapped[str | None] = mapped_column(String(255), nullable=True)
    metadata_: Mapped[dict] = mapped_column("metadata", JSON, nullable=False, default=dict)
    last_analyzed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    memberships: Mapped[list["ClubMembership"]] = relationship(
        back_populates="club", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return f"<ClubProfile(name={self.name!r})>"


class ClubMembership(Base):
    """Links callsigns to clubs with roles inferred from analysis."""

    __tablename__ = "club_memberships"

    club_id: Mapped[uuid.UUID] = mapped_column(
        GUID(), ForeignKey("club_profiles.id", ondelete="CASCADE"), nullable=False, index=True
    )
    callsign_id: Mapped[uuid.UUID] = mapped_column(
        GUID(), ForeignKey("callsigns.id", ondelete="CASCADE"), nullable=False, index=True
    )
    role: Mapped[ClubRole] = mapped_column(
        Enum(ClubRole, native_enum=False), nullable=False, default=ClubRole.MEMBER
    )
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)
    evidence_transcription_id: Mapped[uuid.UUID | None] = mapped_column(
        GUID(), ForeignKey("transcriptions.id", ondelete="SET NULL"), nullable=True
    )
    first_seen: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    last_seen: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)

    club: Mapped["ClubProfile"] = relationship(back_populates="memberships")
    callsign: Mapped["Callsign"] = relationship()

    __table_args__ = (
        Index(
            "ix_club_memberships_club_callsign",
            "club_id",
            "callsign_id",
            unique=True,
        ),
    )

    def __repr__(self) -> str:
        return f"<ClubMembership(club_id={self.club_id}, callsign_id={self.callsign_id})>"
