"""
Configuration settings for the BiLSTM-CRF NER microservice.

This module defines environment-based configuration for the BiLSTM-CRF service,
including model path and service settings.
"""

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    BiLSTM-CRF NER Service configuration settings.

    All settings can be overridden via environment variables.
    Example: MODEL_DIR="/custom/path" overrides the model_dir field.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        case_sensitive=False,
        extra="ignore",
    )

    # Service configuration
    service_name: str = Field(
        default="ner-bilstm-crf",
        description="Service identifier for logging and monitoring",
    )
    log_level: str = Field(
        default="info",
        description="Logging level (debug, info, warning, error, critical)",
    )

    # Model configuration
    model_dir: Path = Field(
        default=Path("/app/models/model"),
        description="Directory containing BiLSTM-CRF model files (model.keras or saved_model/, word2idx.json, tag2idx.json, config.json)",
    )


@lru_cache
def get_settings() -> Settings:
    """
    Get cached settings instance.

    Returns:
        Settings: Singleton settings instance loaded from environment.
    """
    return Settings()
