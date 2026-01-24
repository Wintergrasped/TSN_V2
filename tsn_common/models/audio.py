"""
Audio file models - tracks uploaded WAV files and their processing state.
"""

import enum
from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import BigInteger, Enum, Index, Integer, JSON, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from tsn_common.models.base import Base

if TYPE_CHECKING:
    from tsn_common.models.transcription import Transcription


class AudioFileState(str, enum.Enum):
    """Processing state of an audio file."""

    # Node side
    PENDING = "pending"  # Detected, not yet uploaded
    UPLOADING = "uploading"  # Transfer in progress
    UPLOADED = "uploaded"  # Successfully transferred

    # Server side
    RECEIVED = "received"  # Server received file
    QUEUED_TRANSCRIPTION = "queued_transcription"  # Waiting for transcription
    TRANSCRIBING = "transcribing"  # Whisper processing
    TRANSCRIBED = "transcribed"  # Transcription complete
    QUEUED_EXTRACTION = "queued_extraction"  # Waiting for callsign extraction
    EXTRACTING = "extracting"  # Extracting callsigns
    CALLSIGNS_EXTRACTED = "callsigns_extracted"  # Callsigns extracted
    QUEUED_ANALYSIS = "queued_analysis"  # Waiting for deep analysis
    ANALYZING = "analyzing"  # Running analysis
    ANALYZED = "analyzed"  # Analysis complete
    COMPLETE = "complete"  # All processing done

    # Error states
    FAILED_UPLOAD = "failed_upload"
    FAILED_TRANSCRIPTION = "failed_transcription"
    FAILED_EXTRACTION = "failed_extraction"
    FAILED_ANALYSIS = "failed_analysis"


class AudioFile(Base):
    """
    Represents a single audio file (WAV) captured from the repeater.
    Tracks the file through the entire processing pipeline.
    """

    __tablename__ = "audio_files"

    # File identification
    filename: Mapped[str] = mapped_column(String(255), unique=True, nullable=False, index=True)
    sha256: Mapped[str] = mapped_column(String(64), unique=True, nullable=False, index=True)

    # File metadata
    file_size: Mapped[int] = mapped_column(BigInteger, nullable=False)
    duration_sec: Mapped[float | None] = mapped_column(nullable=True)
    sample_rate: Mapped[int | None] = mapped_column(Integer, nullable=True)
    channels: Mapped[int | None] = mapped_column(Integer, nullable=True)

    # Source tracking
    node_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    uploaded_at: Mapped[datetime | None] = mapped_column(nullable=True)

    # Processing state
    state: Mapped[AudioFileState] = mapped_column(
        Enum(AudioFileState, native_enum=False),
        nullable=False,
        default=AudioFileState.PENDING,
        index=True,
    )
    retry_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    # Extensibility
    metadata_: Mapped[dict] = mapped_column("metadata", JSON, nullable=False, default=dict)

    # Relationships
    transcriptions: Mapped[list["Transcription"]] = relationship(
        back_populates="audio_file", cascade="all, delete-orphan"
    )

    # Indexes
    __table_args__ = (
        Index("ix_audio_files_state_created", "state", "created_at"),
        Index("ix_audio_files_node_created", "node_id", "created_at"),
    )

    def __repr__(self) -> str:
        return f"<AudioFile(filename={self.filename}, state={self.state.value})>"

    def to_dict(self) -> dict:
        return {
            **super().to_dict(),
            "filename": self.filename,
            "sha256": self.sha256,
            "file_size": self.file_size,
            "duration_sec": self.duration_sec,
            "node_id": self.node_id,
            "state": self.state.value,
            "retry_count": self.retry_count,
        }
