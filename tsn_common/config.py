"""
Configuration management using Pydantic Settings.
Hierarchical: Environment variables → .env file → Defaults
"""

from functools import lru_cache
from pathlib import Path
from typing import Literal
from urllib.parse import quote_plus

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class DatabaseSettings(BaseSettings):
    """Database connection settings."""

    engine: Literal["postgresql", "mysql"] = Field(
        default="postgresql",
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
        if self.driver:
            return self.driver
        return "asyncpg" if self.engine == "postgresql" else "asyncmy"

    @property
    def url(self) -> str:
        """Get database URL for SQLAlchemy."""
        user = quote_plus(self.user)
        password = quote_plus(self.password.get_secret_value())
        base = f"{self.engine}+{self.resolved_driver}://{user}:{password}"
        url = f"{base}@{self.host}:{self.port}/{self.name}"
        if self.engine == "mysql":
            return f"{url}?charset=utf8mb4"
        return url

    model_config = SettingsConfigDict(env_prefix="TSN_DB_")


class NodeSettings(BaseSettings):
    """Node-side (repeater) settings."""

    enabled: bool = Field(default=False, description="Enable node services")
    node_id: str = Field(description="Unique node identifier")
    audio_incoming_dir: Path = Field(description="Directory to watch for new WAV files")
    audio_archive_dir: Path = Field(description="Directory for archived audio")

    # Transfer settings
    sftp_host: str = Field(description="SFTP server hostname")
    sftp_port: int = Field(default=22)
    sftp_username: str = Field(description="SFTP username")
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
    device: str = Field(default="cuda", description="Device: cuda, cpu, auto")
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
    api_key: SecretStr = Field(default=SecretStr("sk-no-auth"), description="API key")

    timeout_sec: int = Field(default=120, description="Request timeout")
    max_retries: int = Field(default=4, description="Max retry attempts")
    max_concurrent: int = Field(default=10, description="Max concurrent requests")

    # Fallback to OpenAI
    fallback_enabled: bool = Field(default=False, description="Enable OpenAI fallback")
    openai_api_key: SecretStr | None = Field(default=None, description="OpenAI API key")
    openai_model: str = Field(default="gpt-4o-mini", description="OpenAI model")

    model_config = SettingsConfigDict(env_prefix="TSN_VLLM_")


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

    enabled: bool = Field(default=True, description="Enable server services")
    incoming_dir: Path = Field(description="Directory for incoming files from nodes")
    poll_interval_sec: float = Field(default=5.0, description="Ingestion poll interval")

    @field_validator("incoming_dir")
    @classmethod
    def validate_directory(cls, v: Path) -> Path:
        if not v.exists():
            v.mkdir(parents=True, exist_ok=True)
        return v

    model_config = SettingsConfigDict(env_prefix="TSN_SERVER_")


class StorageSettings(BaseSettings):
    """Storage settings."""

    base_path: Path = Field(description="Base storage directory for audio files")

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
