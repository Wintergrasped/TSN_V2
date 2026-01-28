"""
Configuration management using Pydantic Settings.
Hierarchical: Environment variables → .env file → Defaults
"""

from functools import lru_cache
from pathlib import Path
from typing import Literal
from urllib.parse import quote_plus

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class DatabaseSettings(BaseSettings):
    """Database connection settings (MySQL/MariaDB only)."""

    engine: Literal["mysql"] = Field(
        default="mysql",
        description="Database engine to use",
    )
    host: str = Field(default="localhost", description="Database host")
    port: int = Field(default=5432, description="Database port")
    name: str = Field(default="tsn", description="Database name")
    user: str = Field(default="tsn_user", description="Database user")
    password: SecretStr = Field(description="Database password")

    pool_size: int = Field(default=20, description="Connection pool size")
    max_overflow: int = Field(default=10, description="Max overflow connections")
    pool_timeout: int = Field(default=30, description="Pool timeout seconds")

    driver: str | None = Field(
        default=None,
        description="Optional SQLAlchemy async driver override",
    )

    @property
    def resolved_driver(self) -> str:
        """Return the async driver for the configured engine."""
        return self.driver or "asyncmy"

    @property
    def url(self) -> str:
        """Get database URL for SQLAlchemy."""
        user = quote_plus(self.user)
        password = quote_plus(self.password.get_secret_value())
        base = f"mysql+{self.resolved_driver}://{user}:{password}"
        url = f"{base}@{self.host}:{self.port}/{self.name}"
        return f"{url}?charset=utf8mb4"

    model_config = SettingsConfigDict(env_prefix="TSN_DB_")


class NodeSettings(BaseSettings):
    """Node-side (repeater) settings."""

    enabled: bool = Field(default=False, description="Enable node services")
    node_id: str = Field(default="node-disabled", description="Unique node identifier")
    audio_incoming_dir: Path = Field(
        default=Path("/tmp/tsn_node/incoming"), description="Directory to watch for new WAV files"
    )
    audio_archive_dir: Path = Field(
        default=Path("/tmp/tsn_node/archive"), description="Directory for archived audio"
    )

    # Transfer settings
    sftp_host: str = Field(default="localhost", description="SFTP server hostname")
    sftp_port: int = Field(default=22)
    sftp_username: str = Field(default="tsn", description="SFTP username")
    sftp_key_path: Path | None = Field(default=None, description="Path to SSH private key")
    sftp_password: SecretStr | None = Field(default=None, description="SFTP password")
    sftp_remote_dir: str = Field(default="/incoming", description="Remote upload directory")

    # Watcher settings
    watch_interval_sec: float = Field(default=1.0, description="File watch poll interval")
    min_file_age_sec: float = Field(default=2.0, description="Minimum age before processing")
    min_file_size: int = Field(default=1000, description="Minimum file size (bytes)")
    transfer_workers: int = Field(default=2, description="Number of transfer workers")

    @field_validator("audio_incoming_dir", "audio_archive_dir")
    @classmethod
    def validate_directory(cls, v: Path) -> Path:
        if not v.exists():
            v.mkdir(parents=True, exist_ok=True)
        return v

    model_config = SettingsConfigDict(env_prefix="TSN_NODE_")


class TranscriptionSettings(BaseSettings):
    """Whisper transcription settings."""

    backend: Literal["faster-whisper", "whisper.cpp", "openai"] = "faster-whisper"
    model: str = Field(default="medium.en", description="Whisper model name")
    device: Literal["cuda", "cpu", "auto"] = Field(
        default="auto",
        description="Device preference: cuda, cpu, or auto-detect",
    )
    compute_type: str = Field(default="float16", description="Compute type for faster-whisper")
    language: str = Field(default="en", description="Language code")

    beam_size: int = Field(default=5, description="Beam size for decoding")
    vad_filter: bool = Field(default=True, description="Enable VAD filtering")
    temperature: float = Field(default=0.0, description="Temperature for sampling")

    max_concurrent: int = Field(default=4, description="Max concurrent transcriptions")
    timeout_sec: int = Field(default=300, description="Timeout per file")

    model_config = SettingsConfigDict(env_prefix="TSN_WHISPER_")


class VLLMSettings(BaseSettings):
    """vLLM API settings."""

    base_url: str = Field(
        default="http://192.168.0.104:8001/v1", description="vLLM base URL (OpenAI-compatible)"
    )
    model: str = Field(
        default="Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4", description="Model identifier"
    )
    api_key: SecretStr = Field(
        default=SecretStr(""),
        description="API key (set via TSN_VLLM_API_KEY environment variable)",
    )

    timeout_sec: int = Field(default=120, description="Request timeout")
    max_retries: int = Field(default=4, description="Max retry attempts")
    max_concurrent: int = Field(default=10, description="Max concurrent requests")

    # Fallback to OpenAI
    fallback_enabled: bool = Field(default=False, description="Enable OpenAI fallback")
    openai_api_key: SecretStr | None = Field(default=None, description="OpenAI API key")
    openai_model: str = Field(default="gpt-4o-mini", description="OpenAI model")

    model_config = SettingsConfigDict(env_prefix="TSN_VLLM_")


class AnalysisSettings(BaseSettings):
    """Deep analysis settings using vLLM output."""

    worker_count: int = Field(default=2, description="Concurrent analysis workers")
    max_batch_size: int = Field(default=4, description="Max transcripts per analysis batch")
    max_context_extensions: int = Field(
        default=4,
        description="How many times analysis can append extra transcripts when LLM requests more context",
    )
    context_char_budget: int = Field(
        default=28000,
        description="Approximate character budget to stay within the 32k token window",
    )
    max_response_tokens: int = Field(default=2500, description="Max tokens requested from vLLM")
    transcription_backlog_pause: int = Field(
        default=20,
        description="Pause analysis when >= this many files await transcription/extraction",
    )
    analysis_queue_priority_floor: int = Field(
        default=400,
        description="Minimum queued analysis items required to ignore transcription pauses",
    )
    idle_poll_interval_sec: float = Field(
        default=2.0,
        description="Sleep interval when no analysis work is available",
    )
    trend_refresh_minutes: int = Field(
        default=30,
        description="Minimum minutes between trend snapshots to avoid churn",
    )
    refinement_window_hours: int = Field(
        default=38,
        description="How far back (hours) to look when backfilling idle refinements",
    )
    refinement_batch_size: int = Field(
        default=8,
        description="How many completed files to requeue per idle refinement cycle",
    )
    max_refinement_passes: int = Field(
        default=10,
        description="Maximum number of analysis passes (primary + refinements) per audio file",
    )
    crosscheck_enabled: bool = Field(
        default=True,
        description="Enable opportunistic OpenAI Responses cross-checks",
    )
    crosscheck_probability: float = Field(
        default=0.15,
        description="Probability (0-1) that an idle refinement batch will trigger an OpenAI cross-check",
    )
    openai_responses_model: str = Field(
        default="gpt-4o-mini",
        description="Model used for OpenAI Responses cross-checks",
    )
    net_validation_enabled: bool = Field(
        default=True,
        description="Perform a second vLLM pass to validate detected nets before persisting",
    )
    net_validation_min_confidence: float = Field(
        default=0.6,
        description="Minimum validator confidence to keep a detected net",
    )
    merge_suggestion_enabled: bool = Field(
        default=True,
        description="Capture merge/alias suggestions from analyzer output",
    )
    gpu_watch_enabled: bool = Field(
        default=True,
        description="Monitor GPU utilization from the analyzer container",
    )
    gpu_low_utilization_pct: float = Field(
        default=65.0,
        description="Below this percent the GPU is considered idle and backfill work is triggered",
    )
    gpu_check_interval_sec: float = Field(
        default=15.0,
        description="Seconds between GPU utilization samples to avoid hammering nvidia-smi",
    )
    gpu_overdrive_budget: int = Field(
        default=31500,
        description="Character budget to target when GPU is idle (keeps 32k context saturated)",
    )
    overdrive_window_hours: int = Field(
        default=168,
        description="How far back (hours) to pull completed audio for GPU overdrive re-analysis",
    )
    overdrive_batch_size: int = Field(
        default=6,
        description="How many completed files to requeue during GPU overdrive cycles",
    )
    overdrive_cooldown_hours: int = Field(
        default=12,
        description="Minimum hours between overdrive re-analysis passes per audio file",
    )
    profile_refresh_hours: int = Field(
        default=12,
        description="Minimum hours before a callsign profile becomes eligible for refresh",
    )
    profile_context_hours: int = Field(
        default=640,
        description="Historical window (hours) of data summarized for profile refresh prompts",
    )
    profile_batch_size: int = Field(
        default=3,
        description="How many callsign profiles to refresh in a single idle GPU pass",
    )
    profile_min_seen_count: int = Field(
        default=5,
        description="Minimum segment count before a callsign is considered for profile refresh",
    )
    transcript_smoothing_enabled: bool = Field(
        default=True,
        description="Use vLLM to produce cleaned transcript variants",
    )
    transcript_smoothing_batch_size: int = Field(
        default=4,
        description="How many transcripts to smooth per AI call",
    )
    failed_analysis_rescue_minutes: int = Field(
        default=10,
        description="Minutes to wait before automatically re-queuing failed analyses",
    )
    failed_analysis_rescue_batch: int = Field(
        default=25,
        description="How many failed analyses to rescue per sweep",
    )
    failed_analysis_retry_limit: int = Field(
        default=6,
        description="Maximum failed attempts before leaving a file in failed_analysis",
    )
    gpu_saturation_threshold_pct: float = Field(
        default=95.0,
        description="Utilization percent considered \"full GPU\" for metrics",
    )
    purge_net_history_on_boot: bool = Field(
        default=False,
        description="When true, purge all existing net history once during orchestrator startup",
    )

    model_config = SettingsConfigDict(env_prefix="TSN_ANALYSIS_")


class QRZSettings(BaseSettings):
    """QRZ XML API settings for callsign validation."""

    username: str | None = Field(default=None, description="QRZ username")
    password: SecretStr | None = Field(default=None, description="QRZ password")
    cache_ttl_sec: int = Field(default=86400, description="Cache TTL (24 hours)")
    enabled: bool = Field(default=False, description="Enable QRZ validation")

    model_config = SettingsConfigDict(env_prefix="TSN_QRZ_")


class ProcessingSettings(BaseSettings):
    """General processing settings."""

    max_retries: int = Field(default=3, description="Max retry attempts for failed tasks")
    retry_backoff_base: float = Field(default=2.0, description="Exponential backoff base")
    retry_max_wait_sec: int = Field(default=300, description="Max wait between retries")

    batch_size: int = Field(default=100, description="Batch size for bulk operations")
    queue_poll_interval_sec: float = Field(default=1.0, description="Queue polling interval")

    # Stage timeouts
    ingestion_timeout_sec: int = Field(default=60)
    transcription_timeout_sec: int = Field(default=300)
    extraction_timeout_sec: int = Field(default=60)
    analysis_timeout_sec: int = Field(default=180)

    model_config = SettingsConfigDict(env_prefix="TSN_PROC_")


class LoggingSettings(BaseSettings):
    """Logging configuration."""

    level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    format: Literal["json", "console"] = "json"
    output: Literal["stdout", "file"] = "stdout"
    file_path: Path | None = Field(default=None, description="Log file path if output=file")
    rotate_mb: int = Field(default=100, description="Log rotation size (MB)")

    model_config = SettingsConfigDict(env_prefix="TSN_LOG_")


class MetricsSettings(BaseSettings):
    """Prometheus metrics settings."""

    enabled: bool = Field(default=True, description="Enable metrics export")
    port: int = Field(default=9090, description="Metrics HTTP server port")
    path: str = Field(default="/metrics", description="Metrics endpoint path")

    model_config = SettingsConfigDict(env_prefix="TSN_METRICS_")


class ServerSettings(BaseSettings):
    """Server-side settings."""

    enabled: bool = Field(default=False, description="Enable server services")
    incoming_dir: Path | None = Field(
        default=None,
        description="Directory for incoming files from nodes",
    )
    poll_interval_sec: float = Field(default=5.0, description="Ingestion poll interval")

    @field_validator("incoming_dir")
    @classmethod
    def validate_directory(cls, v: Path | None) -> Path | None:
        if v is None:
            return None
        if not v.exists():
            v.mkdir(parents=True, exist_ok=True)
        return v

    @model_validator(mode="after")
    def ensure_required_fields(self) -> "ServerSettings":
        if self.enabled and self.incoming_dir is None:
            default_dir = Path("/incoming")
            default_dir.mkdir(parents=True, exist_ok=True)
            self.incoming_dir = default_dir
        return self

    model_config = SettingsConfigDict(env_prefix="TSN_SERVER_")


class StorageSettings(BaseSettings):
    """Storage settings."""

    base_path: Path = Field(
        default=Path("/tmp/tsn_storage"),
        description="Base storage directory for audio files",
    )

    @field_validator("base_path")
    @classmethod
    def validate_directory(cls, v: Path) -> Path:
        if not v.exists():
            v.mkdir(parents=True, exist_ok=True)
        return v

    model_config = SettingsConfigDict(env_prefix="TSN_STORAGE_")


class MonitoringSettings(BaseSettings):
    """Health check and monitoring settings."""

    enabled: bool = Field(default=True, description="Enable health check server")
    host: str = Field(default="0.0.0.0", description="Health server host")
    port: int = Field(default=8080, description="Health server port")

    model_config = SettingsConfigDict(env_prefix="TSN_MONITORING_")


class Settings(BaseSettings):
    """Root settings object."""

    # Component settings
    database: DatabaseSettings = Field(default_factory=DatabaseSettings)
    node: NodeSettings = Field(default_factory=NodeSettings)
    server: ServerSettings = Field(default_factory=ServerSettings)
    storage: StorageSettings = Field(default_factory=StorageSettings)
    transcription: TranscriptionSettings = Field(default_factory=TranscriptionSettings)
    vllm: VLLMSettings = Field(default_factory=VLLMSettings)
    analysis: AnalysisSettings = Field(default_factory=AnalysisSettings)
    qrz: QRZSettings = Field(default_factory=QRZSettings)
    processing: ProcessingSettings = Field(default_factory=ProcessingSettings)
    logging: LoggingSettings = Field(default_factory=LoggingSettings)
    metrics: MetricsSettings = Field(default_factory=MetricsSettings)
    monitoring: MonitoringSettings = Field(default_factory=MonitoringSettings)

    # Environment
    environment: Literal["development", "staging", "production"] = "development"
    debug: bool = Field(default=False, description="Debug mode")

    model_config = SettingsConfigDict(
        env_prefix="TSN_",
        env_file=".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        case_sensitive=False,
    )


@lru_cache()
def get_settings() -> Settings:
    """
    Get cached settings instance.
    Uses LRU cache to avoid re-reading environment on every call.
    """
    return Settings()
