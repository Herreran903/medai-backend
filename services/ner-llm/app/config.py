"""
Configuration settings for the LLM NER microservice.

This module defines environment-based configuration for the LLM service,
including API keys, model selection, and service settings.
"""

from functools import lru_cache
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    LLM NER Service configuration settings.

    All settings can be overridden via environment variables.
    Example: GPT_MODEL="gpt-5.2" overrides the gpt_model field.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        case_sensitive=False,
        extra="ignore",
    )

    # Service configuration
    service_name: str = Field(
        default="ner-llm",
        description="Service identifier for logging and monitoring",
    )
    log_level: str = Field(
        default="info",
        description="Logging level (debug, info, warning, error, critical)",
    )

    # API keys
    openai_api_key: Optional[str] = Field(
        default=None,
        description="OpenAI API key for GPT provider",
    )

    # Model selection
    gpt_model: str = Field(
        default="gpt-5.2",
        description="GPT model identifier to use (default: gpt-5.2)",
    )


@lru_cache
def get_settings() -> Settings:
    """
    Get cached settings instance.

    Returns:
        Settings: Singleton settings instance loaded from environment.
    """
    return Settings()
