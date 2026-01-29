"""Net candidate models - streaming autodetect with vLLM micro-windows."""

import enum
import uuid
from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import (
    DateTime,
    Enum,
    Float,
    ForeignKey,
    Index,
    Integer,
    JSON,
    String,
    Text,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from tsn_common.models.base import Base, GUID

if TYPE_CHECKING:
    pass


class CandidateStatus(str, enum.Enum):
    """Net candidate lifecycle status."""

    WARMUP = "warmup"  # net_likelihood trending up
    ACTIVE = "active"  # accumulating evidence
    COOLING = "cooling"  # likelihood dropping
    ENDED = "ended"  # candidate ended, awaiting final verification
    VERIFIED = "verified"  # OpenAI confirmed as net
    REJECTED = "rejected"  # OpenAI rejected
    DISMISSED = "dismissed"  # manually dismissed by user
    PROMOTED = "promoted"  # manually promoted to NetSession


class NetCandidate(Base):
    """
    Represents a potential net detected through streaming vLLM analysis.
    
    Candidates are built incrementally through micro-window evaluations
    and finalized with OpenAI verification.
    """

    __tablename__ = "net_candidates"

    # Status
    status: Mapped[CandidateStatus] = mapped_column(
        Enum(CandidateStatus, native_enum=False),
        nullable=False,
        default=CandidateStatus.WARMUP,
        index=True,
    )

    # Time window
    start_ts: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, index=True
    )
    end_ts: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True, index=True
    )

    # Node scope (which repeater/node detected this)
    node_id: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)

    # vLLM confidence metrics
    vllm_confidence_avg: Mapped[float | None] = mapped_column(Float, nullable=True)
    vllm_confidence_peak: Mapped[float | None] = mapped_column(Float, nullable=True)
    vllm_evaluation_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    # OpenAI verdict
    openai_verdict_json: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    openai_verified_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    # Extracted features (from vLLM passes)
    features_json: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    # Structure: {
    #   "control_station_callsign": "K7XYZ",
    #   "unique_callsigns": ["K7A", "W1B", ...],
    #   "checkin_activity_avg": 75.0,
    #   "directed_net_style_avg": 80.0,
    #   ...
    # }

    # Evidence (short excerpt strings + transcript IDs)
    evidence_json: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    # Structure: {
    #   "excerpts": ["Opening statement...", "Check-in: K7XYZ..."],
    #   "transcript_ids": ["uuid1", "uuid2", ...],
    #   "audio_file_ids": ["uuid1", "uuid2", ...]
    # }

    # For promoted candidates: link to created NetSession
    promoted_net_session_id: Mapped[uuid.UUID | None] = mapped_column(
        GUID(),
        ForeignKey("net_sessions.id", ondelete="SET NULL"),
        nullable=True,
    )

    # Indexes
    __table_args__ = (
        Index("ix_net_candidates_status_updated", "status", "updated_at"),
        Index("ix_net_candidates_node_start", "node_id", "start_ts"),
        Index("ix_net_candidates_active_lookup", "status", "node_id", "start_ts"),
    )

    def __repr__(self) -> str:
        return f"<NetCandidate(status={self.status}, start_ts={self.start_ts}, node_id={self.node_id})>"

    def to_dict(self) -> dict:
        return {
            **super().to_dict(),
            "status": self.status.value if isinstance(self.status, enum.Enum) else self.status,
            "start_ts": self.start_ts.isoformat(),
            "end_ts": self.end_ts.isoformat() if self.end_ts else None,
            "node_id": self.node_id,
            "vllm_confidence_avg": self.vllm_confidence_avg,
            "vllm_confidence_peak": self.vllm_confidence_peak,
            "vllm_evaluation_count": self.vllm_evaluation_count,
            "openai_verdict_json": self.openai_verdict_json,
            "openai_verified_at": self.openai_verified_at.isoformat() if self.openai_verified_at else None,
            "features_json": self.features_json,
            "evidence_json": self.evidence_json,
            "promoted_net_session_id": (
                str(self.promoted_net_session_id) if self.promoted_net_session_id else None
            ),
        }


class NetCandidateWindow(Base):
    """
    Records individual vLLM micro-window evaluations for a candidate.
    
    Each window represents a 3-5 minute analysis with vLLM likelihood score
    and extracted signals.
    """

    __tablename__ = "net_candidate_windows"

    # Foreign key to candidate
    candidate_id: Mapped[uuid.UUID] = mapped_column(
        GUID(),
        ForeignKey("net_candidates.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    # Window time range
    window_start: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, index=True
    )
    window_end: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )

    # vLLM output (full JSON response)
    vllm_output_json: Mapped[dict] = mapped_column(JSON, nullable=False)
    # Structure: {
    #   "net_likelihood": 75,
    #   "signals": {
    #     "control_station_callsign": "K7XYZ",
    #     "checkin_activity": 80,
    #     "directed_net_style": 70,
    #     ...
    #   },
    #   "evidence": ["excerpt1", "excerpt2", ...],
    #   "suggested_action": "extend_candidate"
    # }

    # Performance tracking
    vllm_latency_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)

    # Relationship
    candidate: Mapped["NetCandidate"] = relationship()

    # Indexes
    __table_args__ = (
        Index("ix_net_candidate_windows_candidate_start", "candidate_id", "window_start"),
        Index("ix_net_candidate_windows_time_lookup", "window_start", "window_end"),
    )

    def __repr__(self) -> str:
        return f"<NetCandidateWindow(candidate_id={self.candidate_id}, window_start={self.window_start})>"

    def to_dict(self) -> dict:
        return {
            **super().to_dict(),
            "candidate_id": str(self.candidate_id),
            "window_start": self.window_start.isoformat(),
            "window_end": self.window_end.isoformat(),
            "vllm_output_json": self.vllm_output_json,
            "vllm_latency_ms": self.vllm_latency_ms,
        }
