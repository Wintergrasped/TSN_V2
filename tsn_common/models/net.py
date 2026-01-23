"""
Net session models - tracks organized radio nets and participation.
"""

import enum
from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import DateTime, Enum, Float, ForeignKey, Index, Integer, String, Text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from tsn_common.models.base import Base

if TYPE_CHECKING:
    from tsn_common.models.callsign import Callsign


class CheckinType(str, enum.Enum):
    """Type of net check-in."""

    REGULAR = "regular"
    LATE = "late"
    IO = "io"  # Information only
    PROXY = "proxy"
    ECHOLINK = "echolink"
    ALLSTAR = "allstar"
    RECHECK = "recheck"
    UNKNOWN = "unknown"


class NetSession(Base):
    """
    Represents a detected radio net session.
    A net is an organized on-air meeting with check-ins and structure.
    """

    __tablename__ = "net_sessions"

    # Net identification
    net_name: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    club_name: Mapped[str | None] = mapped_column(String(255), nullable=True)

    # Net Control Station
    ncs_callsign_id: Mapped[UUID | None] = mapped_column(
        ForeignKey("callsigns.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )

    # Timing
    start_time: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, index=True
    )
    end_time: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    duration_sec: Mapped[int] = mapped_column(Integer, nullable=False)

    # Participation
    participant_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    # Detection confidence
    confidence: Mapped[float] = mapped_column(Float, nullable=False)

    # AI-generated summary
    summary: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Relationships
    ncs_callsign: Mapped["Callsign | None"] = relationship(foreign_keys=[ncs_callsign_id])
    participations: Mapped[list["NetParticipation"]] = relationship(
        back_populates="net_session", cascade="all, delete-orphan"
    )

    # Indexes
    __table_args__ = (
        Index("ix_net_sessions_name_start", "net_name", "start_time"),
        Index("ix_net_sessions_start_time", "start_time"),
    )

    def __repr__(self) -> str:
        return f"<NetSession(net_name={self.net_name!r}, start_time={self.start_time})>"

    def to_dict(self) -> dict:
        return {
            **super().to_dict(),
            "net_name": self.net_name,
            "club_name": self.club_name,
            "ncs_callsign_id": str(self.ncs_callsign_id) if self.ncs_callsign_id else None,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat(),
            "duration_sec": self.duration_sec,
            "participant_count": self.participant_count,
            "confidence": self.confidence,
        }


class NetParticipation(Base):
    """
    Tracks individual callsign participation in a net session.
    """

    __tablename__ = "net_participations"

    # Foreign keys
    net_session_id: Mapped[UUID] = mapped_column(
        ForeignKey("net_sessions.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    callsign_id: Mapped[UUID] = mapped_column(
        ForeignKey("callsigns.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    # Participation details
    first_seen: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    last_seen: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    transmission_count: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    estimated_talk_seconds: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    checkin_type: Mapped[CheckinType] = mapped_column(
        Enum(CheckinType, native_enum=False),
        nullable=False,
        default=CheckinType.UNKNOWN,
    )

    # Relationships
    net_session: Mapped["NetSession"] = relationship(back_populates="participations")
    callsign: Mapped["Callsign"] = relationship(back_populates="net_participations")

    # Indexes
    __table_args__ = (
        Index("ix_net_participations_net_callsign", "net_session_id", "callsign_id", unique=True),
        Index("ix_net_participations_callsign_first", "callsign_id", "first_seen"),
    )

    def __repr__(self) -> str:
        return (
            f"<NetParticipation(net_session_id={self.net_session_id}, "
            f"callsign_id={self.callsign_id})>"
        )
