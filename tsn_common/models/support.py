"""
Support models - corrections, metrics, health checks.
"""

from datetime import datetime
import uuid

from sqlalchemy import Boolean, DateTime, Float, Index, Integer, JSON, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from tsn_common.models.base import Base, GUID


class PhoneticCorrection(Base):
    """
    Phonetic correction map for Whisper ASR errors.
    Example: "kilo kilo seven november quebec" → "KK7NQN"
    """

    __tablename__ = "phonetic_corrections"

    detect: Mapped[str] = mapped_column(String(100), nullable=False, unique=True, index=True)
    correct: Mapped[str] = mapped_column(String(100), nullable=False)

    def __repr__(self) -> str:
        return f"<PhoneticCorrection(detect={self.detect!r}, correct={self.correct!r})>"


class ProcessingMetric(Base):
    """
    Records processing metrics for each pipeline stage.
    Used for performance monitoring and optimization.
    """

    __tablename__ = "processing_metrics"

    stage: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    processing_time_ms: Mapped[int] = mapped_column(Integer, nullable=False)
    success: Mapped[bool] = mapped_column(Boolean, nullable=False)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, index=True
    )

    # Additional metadata
    metadata_: Mapped[dict] = mapped_column("metadata", JSON, nullable=False, default=dict)

    __table_args__ = (Index("ix_processing_metrics_stage_timestamp", "stage", "timestamp"),)

    def __repr__(self) -> str:
        return f"<ProcessingMetric(stage={self.stage}, success={self.success})>"


class SystemHealth(Base):
    """
    System health status for each component.
    Used for monitoring and alerting.
    """

    __tablename__ = "system_health"

    component: Mapped[str] = mapped_column(String(100), nullable=False, unique=True, index=True)
    status: Mapped[str] = mapped_column(String(20), nullable=False)  # healthy, degraded, down
    last_heartbeat: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)

    # Resource metrics
    cpu_percent: Mapped[float | None] = mapped_column(Float, nullable=True)
    memory_mb: Mapped[int | None] = mapped_column(Integer, nullable=True)
    disk_free_gb: Mapped[int | None] = mapped_column(Integer, nullable=True)

    # Additional metrics
    metrics: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)

    def __repr__(self) -> str:
        return f"<SystemHealth(component={self.component!r}, status={self.status})>"


class AnalysisAudit(Base):
    """Per-audio analysis pass tracking for stats and tuning."""

    __tablename__ = "analysis_audits"

    audio_file_id: Mapped[uuid.UUID] = mapped_column(GUID(), nullable=False, index=True)
    transcription_id: Mapped[uuid.UUID] = mapped_column(GUID(), nullable=False, index=True)
    pass_type: Mapped[str] = mapped_column(String(32), nullable=False)
    backend: Mapped[str] = mapped_column(String(32), nullable=False)
    latency_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    prompt_characters: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    response_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    observations: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)

    __table_args__ = (Index("ix_analysis_audits_audio_backend", "audio_file_id", "backend"),)

    def __repr__(self) -> str:
        return f"<AnalysisAudit(audio_file_id={self.audio_file_id}, pass={self.pass_type})>"


class AiRunLog(Base):
    """Detailed record of every AI request/response cycle."""

    __tablename__ = "ai_run_logs"

    backend: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    model: Mapped[str | None] = mapped_column(String(128), nullable=True)
    pass_label: Mapped[str] = mapped_column(String(64), nullable=False)
    success: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    prompt_text: Mapped[str] = mapped_column(Text, nullable=False)
    response_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    prompt_characters: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    response_characters: Mapped[int | None] = mapped_column(Integer, nullable=True)
    prompt_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    completion_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    total_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    latency_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    gpu_utilization_pct: Mapped[float | None] = mapped_column(Float, nullable=True)
    audio_file_ids: Mapped[list[str] | None] = mapped_column(JSON, nullable=True)
    metadata_: Mapped[dict] = mapped_column("metadata", JSON, nullable=False, default=dict)

    __table_args__ = (
        Index("ix_ai_run_logs_pass_created", "pass_label", "created_at"),
    )

    def __repr__(self) -> str:
        return f"<AiRunLog(pass_label={self.pass_label}, backend={self.backend})>"


class GpuUtilizationSample(Base):
    """Periodic GPU utilization samples for capacity planning."""

    __tablename__ = "gpu_utilization_samples"

    utilization_pct: Mapped[float] = mapped_column(Float, nullable=False)
    sample_source: Mapped[str] = mapped_column(String(32), nullable=False, default="nvidia-smi")
    is_saturated: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)

    def __repr__(self) -> str:
        return f"<GpuUtilizationSample(source={self.sample_source}, utilization={self.utilization_pct})>"
