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
    Example: GPT_MODEL="gpt-5.4" overrides the gpt_model field.
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
        default="gpt-5.4",
        description="GPT model identifier to use (default: gpt-5.4)",
    )
    gpt_temperature: float = Field(
        default=0.0,
        description="Sampling temperature for GPT inference",
    )
    gpt_max_output_tokens: int = Field(
        default=4000,
        description="Maximum completion tokens for GPT responses",
    )
    prompt_mode: str = Field(
        default="few_shot",
        description="Prompting mode: zero_shot or few_shot",
    )
    gpt_max_retries: int = Field(
        default=3,
        description="Max retries for GPT extraction when payload/transport fails",
    )
    gpt_retry_sleep_sec: float = Field(
        default=2.0,
        description="Sleep seconds between GPT retry attempts",
    )


@lru_cache
def get_settings() -> Settings:
    """
    Get cached settings instance.

    Returns:
        Settings: Singleton settings instance loaded from environment.
    """
    return Settings()
