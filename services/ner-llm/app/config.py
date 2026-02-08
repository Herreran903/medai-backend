"""
Configuration settings for the LLM NER microservice.

This module defines environment-based configuration for the LLM service,
including API keys, provider selection, and service settings.
"""

from functools import lru_cache
from typing import Literal, Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    LLM NER Service configuration settings.

    All settings can be overridden via environment variables.
    Example: LLM_PROVIDER="gpt" overrides the llm_provider field.
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

    # LLM provider selection (FIXED: Only GPT for experiments)
    llm_provider: Literal["gpt"] = Field(
        default="gpt",
        description="LLM provider to use (FIXED: only gpt for experiments)",
    )

    # API keys
    anthropic_api_key: Optional[str] = Field(
        default=None,
        description="Legacy Anthropic API key (Claude provider disabled in experiments)",
    )
    openai_api_key: Optional[str] = Field(
        default=None,
        description="OpenAI API key for GPT provider",
    )

    # Model selection
    claude_model: str = Field(
        default="claude-sonnet-4-5-20250929",
        description="Legacy Claude model identifier (disabled in experiments)",
    )
    gpt_model: str = Field(
        default="gpt-5.2",
        description="GPT model to use (FIXED: gpt-5.2 for experiments)",
    )

    # Local LLM settings (for future Ollama integration)
    local_llm_url: str = Field(
        default="http://localhost:11434",
        description="Ollama API endpoint URL",
    )
    local_llm_model: str = Field(
        default="llama3",
        description="Local model name in Ollama",
    )


@lru_cache
def get_settings() -> Settings:
    """
    Get cached settings instance.

    Returns:
        Settings: Singleton settings instance loaded from environment.
    """
    return Settings()
