"""Pydantic settings for local and container runtime configuration."""

from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime settings loaded from environment variables and `.env`."""

    langchain_chat_model: str = "gpt-4o-mini"
    langchain_model_provider: str = "openai"
    openai_api_key: str = ""
    anthropic_api_key: str = ""
    run_live_llm_tests: bool = False
    log_level: str = "INFO"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    """Return cached application settings.

    Returns:
        Settings loaded from environment variables and `.env`.
    """
    return Settings()
