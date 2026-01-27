"""Web-facing configuration helpers."""

from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field


class WebSettings(BaseSettings):
    """Portal specific settings sourced from env/.env."""

    session_secret: str = Field(
        default="replace-me",
        description="Secret used to sign session cookies",
    )
    brand_name: str = Field(default="KK7NQN", description="Display name for headers")
    support_email: str = Field(default="support@example.com")
    allow_registration: bool = Field(default=True, description="Enable self-service signup")
    public_dashboard: bool = Field(default=True, description="Allow anonymous dashboard access")
    vllm_api_key: str | None = Field(default=None, description="Override vLLM API key for the portal")
    openai_api_key: str | None = Field(default=None, description="Override OpenAI fallback key for the portal")

    model_config = SettingsConfigDict(env_prefix="TSN_WEB_", env_file=".env", env_file_encoding="utf-8")


@lru_cache()
def get_web_settings() -> WebSettings:
    """Return cached settings."""

    return WebSettings()
