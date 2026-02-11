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
    archive_cleanup_enabled: bool = Field(
        default=True,
        description="Enable background archive compaction (WAV→MP3→zip)",
    )
    archive_cleanup_interval_hours: int = Field(
        default=24,
        description="How often (hours) to compact archived WAV files",
    )
    archive_cleanup_min_age_minutes: int = Field(
        default=60,
        description="Minimum file age (minutes) before eligible for compaction",
    )
    archive_mp3_bitrate_kbps: int = Field(
        default=160,
        description="Bitrate (kbps) for MP3 conversions",
    )
    archive_ffmpeg_path: str = Field(
        default="ffmpeg",
        description="Path to ffmpeg binary used for WAV→MP3 conversions",
    )

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
    allow_cpu_fallback: bool = Field(
        default=True,
        description="Allow automatic fallback to CPU when CUDA is unavailable",
    )

    beam_size: int = Field(default=5, description="Beam size for decoding")
    vad_filter: bool = Field(default=True, description="Enable VAD filtering")
    temperature: float = Field(default=0.0, description="Temperature for sampling")

    max_concurrent: int = Field(default=4, description="Max concurrent transcriptions")
    timeout_sec: int = Field(default=300, description="Timeout per file")
    hf_cache_dir: Path | None = Field(
        default=None,
        description="Optional HuggingFace cache directory override for Whisper models",
    )
    missing_file_error_threshold: int = Field(
        default=25,
        description="Consecutive missing-file errors before backing off",
    )
    missing_file_backoff_sec: int = Field(
        default=120,
        description="How long to pause transcription when storage appears missing",
    )

    model_config = SettingsConfigDict(env_prefix="TSN_WHISPER_")


class VLLMSettings(BaseSettings):
    """vLLM API settings."""

    base_url: str = Field(
        default="http://host.docker.internal:8001/v1", description="vLLM base URL (OpenAI-compatible)"
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
        default=0.1,
        description="Sleep interval when no analysis work is available (reduced to 0.1s for aggressive keep-hot)",
    )
    aggressive_backfill_enabled: bool = Field(
        default=True,
        description="Aggressively chain multiple background tasks to keep vLLM continuously loaded",
    )
    idle_work_chain_limit: int = Field(
        default=10,
        description="Maximum number of consecutive idle work tasks before yielding (prevents infinite loops)",
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
    archive_dirs: tuple[Path, ...] | str = Field(
        default="",
        description="Optional archive directories to search when restoring missing files (comma-separated)",
    )
    health_check_enabled: bool = Field(
        default=True,
        description="Enable periodic availability checks for the storage mount",
    )
    health_check_interval_sec: float = Field(
        default=5.0,
        description="Seconds between storage availability probes",
    )
    health_probe_filename: str = Field(
        default=".tsn_storage_probe",
        description="Sentinel filename used to detect if the storage mount disappeared",
    )
    health_missing_backoff_sec: int = Field(
        default=300,
        description="How long to pause heavy processing when storage is unavailable",
    )

    @field_validator("base_path")
    @classmethod
    def validate_directory(cls, v: Path) -> Path:
        if not v.exists():
            v.mkdir(parents=True, exist_ok=True)
        return v

    @field_validator("archive_dirs", mode="before")
    @classmethod
    def parse_archive_dirs(cls, value) -> tuple[Path, ...]:
        # Handle empty/None values before Pydantic tries JSON parsing
        if value is None or value == "" or value == ():
            return tuple()
        if isinstance(value, str):
            # Handle whitespace-only strings
            value = value.strip()
            if not value:
                return tuple()
            parts = [part.strip() for part in value.split(",")]
            return tuple(Path(part) for part in parts if part)
        if isinstance(value, Path):
            return (value,)
        if isinstance(value, (list, tuple, set)):
            return tuple(Path(part) if not isinstance(part, Path) else part for part in value)
        return tuple()

    @field_validator("archive_dirs", mode="after")
    @classmethod
    def normalize_archive_dirs(cls, value: tuple[Path, ...]) -> tuple[Path, ...]:
        normalized: list[Path] = []
        for path in value:
            if path in normalized:
                continue
            normalized.append(path)
        return tuple(normalized)

    model_config = SettingsConfigDict(env_prefix="TSN_STORAGE_")


class NetAutoDetectSettings(BaseSettings):
    """Net auto-detection settings for streaming vLLM-heavy detection."""

    enabled: bool = Field(default=True, description="Enable net auto-detection system")

    # Micro-window evaluation
    window_size_minutes: int = Field(default=4, description="Size of micro-windows for vLLM evaluation")
    window_step_seconds: int = Field(default=45, description="Step size between windows (30-60s recommended)")
    max_excerpts_per_window: int = Field(default=20, description="Max transcript excerpts per vLLM call")

    # vLLM calling frequency (AGGRESSIVE)
    vllm_call_interval_sec: float = Field(default=60.0, description="Target: 1 vLLM call per node per minute")
    vllm_max_concurrent_per_node: int = Field(default=3, description="Max concurrent vLLM calls per node")
    vllm_backpressure_threshold: int = Field(default=10, description="Coarsen step if backlog exceeds this")

    # Candidate state machine thresholds
    candidate_start_likelihood: int = Field(default=65, description="Min likelihood to start candidate")
    candidate_start_consecutive_windows: int = Field(default=3, description="Windows above threshold to start")
    candidate_extend_likelihood: int = Field(default=55, description="Min likelihood to extend candidate")
    candidate_end_likelihood: int = Field(default=40, description="Max likelihood before ending candidate")
    candidate_end_consecutive_windows: int = Field(default=4, description="Windows below threshold to end")
    candidate_min_unique_callsigns: int = Field(default=6, description="Min callsigns for candidate promotion")

    # Multi-pass vLLM use
    boundary_refinement_interval_minutes: int = Field(
        default=7, description="Minutes between boundary refinement vLLM passes"
    )
    roster_assist_interval_minutes: int = Field(
        default=10, description="Minutes between roster assist vLLM passes"
    )

    # OpenAI verification (FINAL adjudication only)
    openai_verify_enabled: bool = Field(default=True, description="Enable OpenAI final verification")
    openai_min_confidence: int = Field(default=80, description="Min OpenAI confidence for VERIFIED status")
    openai_model: str = Field(default="gpt-4o-mini", description="OpenAI model for verification")
    openai_max_evidence_excerpts: int = Field(
        default=60, description="Max excerpts in OpenAI verification payload"
    )

    # Performance
    orchestrator_poll_interval_sec: float = Field(default=5.0, description="How often orchestrator checks for work")
    node_inactivity_minutes: int = Field(default=15, description="Minutes of silence before pausing node evaluation")

    model_config = SettingsConfigDict(env_prefix="TSN_NET_AUTODETECT_")


class MonitoringSettings(BaseSettings):
    """Health check and monitoring settings."""

    enabled: bool = Field(default=True, description="Enable health check server")
    host: str = Field(default="0.0.0.0", description="Health server host")
    port: int = Field(default=8080, description="Health server port")

    model_config = SettingsConfigDict(env_prefix="TSN_MONITORING_")


class SystemLoadSettings(BaseSettings):
    """System load guard configuration."""

    enabled: bool = Field(default=True, description="Enable system load monitor")
    cpu_percent_threshold: float = Field(
        default=85.0,
        description="CPU percent that triggers load shedding",
    )
    memory_percent_threshold: float = Field(
        default=92.0,
        description="Memory percent that triggers load shedding",
    )
    check_interval_sec: float = Field(default=5.0, description="How often to sample load")
    pause_duration_sec: int = Field(
        default=30,
        description="How long to pause vLLM activity once load is high",
    )
    breach_samples_required: int = Field(
        default=3,
        description="Consecutive high-load samples required before pausing",
    )

    model_config = SettingsConfigDict(env_prefix="TSN_LOAD_")


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
    net_autodetect: NetAutoDetectSettings = Field(default_factory=NetAutoDetectSettings)
    qrz: QRZSettings = Field(default_factory=QRZSettings)
    processing: ProcessingSettings = Field(default_factory=ProcessingSettings)
    logging: LoggingSettings = Field(default_factory=LoggingSettings)
    metrics: MetricsSettings = Field(default_factory=MetricsSettings)
    monitoring: MonitoringSettings = Field(default_factory=MonitoringSettings)
    system_load: SystemLoadSettings = Field(default_factory=SystemLoadSettings)

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
