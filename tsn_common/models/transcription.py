"""Transcription models - stores Whisper output and metadata."""

import enum
import uuid
from typing import TYPE_CHECKING

from sqlalchemy import Enum, Float, ForeignKey, Index, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from tsn_common.models.base import Base, GUID

if TYPE_CHECKING:
    from tsn_common.models.audio import AudioFile
    from tsn_common.models.callsign import CallsignLog, CallsignTopic


class TranscriptionBackend(str, enum.Enum):
    """Backend used for transcription."""

    FASTER_WHISPER = "faster-whisper"
    WHISPER_CPP = "whisper.cpp"
    OPENAI_WHISPER = "openai-whisper"


class Transcription(Base):
    """
    Stores the transcribed text from an audio file.
    One-to-one with AudioFile in most cases, but allows for retries.
    """

    __tablename__ = "transcriptions"

    # Foreign key to audio file
    audio_file_id: Mapped[uuid.UUID] = mapped_column(
        GUID(),
        ForeignKey("audio_files.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    # Transcription content
    transcript_text: Mapped[str] = mapped_column(Text, nullable=False)
    language: Mapped[str] = mapped_column(String(10), nullable=False, default="en")

    # Processing metadata
    backend: Mapped[TranscriptionBackend] = mapped_column(
        Enum(TranscriptionBackend, native_enum=False),
        nullable=False,
    )
    confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    processing_time_ms: Mapped[int] = mapped_column(Integer, nullable=False)

    # Quality flags
    word_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    char_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    quality_score: Mapped[float | None] = mapped_column(Float, nullable=True)

    # Relationships
    audio_file: Mapped["AudioFile"] = relationship(back_populates="transcriptions")
    callsign_logs: Mapped[list["CallsignLog"]] = relationship(
        back_populates="transcription", cascade="all, delete-orphan"
    )
    callsign_topics: Mapped[list["CallsignTopic"]] = relationship(
        back_populates="transcription", cascade="all, delete-orphan"
    )

    # Indexes
    __table_args__ = (
        Index("ix_transcriptions_audio_file_created", "audio_file_id", "created_at"),
        Index("ix_transcriptions_backend_created", "backend", "created_at"),
    )

    def __repr__(self) -> str:
        preview = self.transcript_text[:50] if self.transcript_text else ""
        return f"<Transcription(audio_file_id={self.audio_file_id}, text={preview!r}...)>"

    def to_dict(self) -> dict:
        return {
            **super().to_dict(),
            "audio_file_id": str(self.audio_file_id),
            "transcript_text": self.transcript_text,
            "language": self.language,
            "backend": self.backend.value,
            "confidence": self.confidence,
            "processing_time_ms": self.processing_time_ms,
            "word_count": self.word_count,
            "char_count": self.char_count,
        }
