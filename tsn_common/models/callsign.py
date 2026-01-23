"""
Callsign models - tracks amateur radio callsigns and their occurrences.
"""

import enum
from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import (
    DateTime,
    Enum,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from tsn_common.models.base import Base

if TYPE_CHECKING:
    from tsn_common.models.transcription import Transcription
    from tsn_common.models.net import NetParticipation
    from tsn_common.models.profile import CallsignProfile


class ValidationMethod(str, enum.Enum):
    """Method used to validate a callsign."""

    QRZ = "qrz"  # QRZ XML API lookup
    VLLM = "vllm"  # vLLM AI validation
    REGEX = "regex"  # Regex pattern match only
    MANUAL = "manual"  # Manually verified


class Callsign(Base):
    """
    Master table of all callsigns detected in the system.
    Tracks validation status and activity.
    """

    __tablename__ = "callsigns"

    # Callsign (normalized: uppercase, no suffix)
    callsign: Mapped[str] = mapped_column(
        String(20), unique=True, nullable=False, index=True
    )

    # Validation
    validated: Mapped[bool] = mapped_column(nullable=False, default=False)
    validation_method: Mapped[ValidationMethod | None] = mapped_column(
        Enum(ValidationMethod, native_enum=False),
        nullable=True,
    )

    # Activity tracking
    first_seen: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    last_seen: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    seen_count: Mapped[int] = mapped_column(Integer, nullable=False, default=1)

    # External data (from QRZ, etc.)
    metadata_: Mapped[dict] = mapped_column("metadata", JSONB, nullable=False, default=dict)

    # Relationships
    callsign_logs: Mapped[list["CallsignLog"]] = relationship(
        back_populates="callsign", cascade="all, delete-orphan"
    )
    callsign_topics: Mapped[list["CallsignTopic"]] = relationship(
        back_populates="callsign", cascade="all, delete-orphan"
    )
    net_participations: Mapped[list["NetParticipation"]] = relationship(
        back_populates="callsign", cascade="all, delete-orphan"
    )
    profile: Mapped["CallsignProfile | None"] = relationship(
        back_populates="callsign", uselist=False, cascade="all, delete-orphan"
    )

    # Indexes
    __table_args__ = (
        Index("ix_callsigns_validated", "validated"),
        Index("ix_callsigns_last_seen", "last_seen"),
    )

    def __repr__(self) -> str:
        return f"<Callsign(callsign={self.callsign}, validated={self.validated})>"

    def to_dict(self) -> dict:
        return {
            **super().to_dict(),
            "callsign": self.callsign,
            "validated": self.validated,
            "validation_method": self.validation_method.value if self.validation_method else None,
            "first_seen": self.first_seen.isoformat(),
            "last_seen": self.last_seen.isoformat(),
            "seen_count": self.seen_count,
        }


class CallsignLog(Base):
    """
    Append-only log of every callsign mention in transcripts.
    Used for timeline reconstruction and correlation.
    """

    __tablename__ = "callsign_log"

    # Foreign keys
    callsign_id: Mapped[UUID] = mapped_column(
        ForeignKey("callsigns.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    transcription_id: Mapped[UUID] = mapped_column(
        ForeignKey("transcriptions.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    # Detection metadata
    detected_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, index=True
    )
    confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    context_snippet: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Relationships
    callsign: Mapped["Callsign"] = relationship(back_populates="callsign_logs")
    transcription: Mapped["Transcription"] = relationship(back_populates="callsign_logs")

    # Indexes
    __table_args__ = (
        Index("ix_callsign_log_detected_at", "detected_at"),
        Index("ix_callsign_log_callsign_detected", "callsign_id", "detected_at"),
    )

    def __repr__(self) -> str:
        return f"<CallsignLog(callsign_id={self.callsign_id}, transcription_id={self.transcription_id})>"


class CallsignTopic(Base):
    """
    Topic events extracted from transcripts - what was each callsign discussing?
    """

    __tablename__ = "callsign_topics"

    # Foreign keys
    callsign_id: Mapped[UUID] = mapped_column(
        ForeignKey("callsigns.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    transcription_id: Mapped[UUID] = mapped_column(
        ForeignKey("transcriptions.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    # Topic data
    topic: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    excerpt: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Timestamp
    detected_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, index=True
    )

    # Relationships
    callsign: Mapped["Callsign"] = relationship(back_populates="callsign_topics")
    transcription: Mapped["Transcription"] = relationship(back_populates="callsign_topics")

    # Indexes
    __table_args__ = (
        Index("ix_callsign_topics_topic_detected", "topic", "detected_at"),
        Index("ix_callsign_topics_callsign_topic", "callsign_id", "topic"),
    )

    def __repr__(self) -> str:
        return f"<CallsignTopic(callsign_id={self.callsign_id}, topic={self.topic})>"
