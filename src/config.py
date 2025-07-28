"""Application configuration module.

This module defines a unified settings object using Pydantic's `BaseSettings`.
All environment variables are read once into a single `Settings` instance which
is then injected into other modules as needed.  Modules should **not** access
environment variables directly.
"""

from __future__ import annotations

from functools import lru_cache

# Try to import BaseSettings from pydantic_settings (Pydantic v2) and fall back to pydantic (v1)
try:
    from pydantic_settings import BaseSettings
except ImportError:
    from pydantic import BaseSettings

from pydantic import Field, validator


class Settings(BaseSettings):
    """Unified configuration loaded from environment variables.

    Attributes mirror the `.env.example` file.  Default values are provided for
    development but should be overridden in production via environment
    variables.
    """

    # Application
    app_name: str = Field("PipelineBuilder", env="APP_NAME")
    log_level: str = Field("INFO", env="LOG_LEVEL")

    # Databases
    database_url: str = Field(
        "sqlite+aiosqlite:///./dev.db", env="DATABASE_URL"
    )
    redis_url: str = Field("redis://localhost:6379/0", env="REDIS_URL")

    # OpenAI
    openai_api_key: str = Field("", env="OPENAI_API_KEY")
    openai_model: str = Field("gpt-4o", env="OPENAI_MODEL")
    embedding_model: str = Field(
        "text-embedding-3-large", env="EMBEDDING_MODEL"
    )

    # Prompt budgets
    prompt_budget_system: int = Field(500, env="PROMPT_BUDGET_SYSTEM")
    prompt_budget_user_ctx: int = Field(1000, env="PROMPT_BUDGET_USER_CTX")
    prompt_budget_snippets: int = Field(500, env="PROMPT_BUDGET_SNIPPETS")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False

    @validator("log_level")
    def validate_log_level(cls, value: str) -> str:
        levels = {"CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"}
        upper = value.upper()
        if upper not in levels:
            raise ValueError(f"Invalid log level: {value}")
        return upper


@lru_cache()
def get_settings() -> Settings:
    """Return a cached Settings instance.

    Using an LRU cache ensures the environment is read only once per process.
    """
    return Settings()
