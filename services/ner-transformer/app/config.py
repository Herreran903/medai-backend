"""
Configuration settings for the Transformer NER microservice.

This module defines environment-based configuration for the Transformer service,
including model selection, Hugging Face model IDs, and service settings.
"""

from functools import lru_cache
from typing import Literal, Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Transformer NER Service configuration settings.

    All settings can be overridden via environment variables.
    Example: MODEL_VARIANT="roberta" overrides the model_variant field.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        case_sensitive=False,
        extra="ignore",
    )

    # Service configuration
    service_name: str = Field(
        default="ner-transformer",
        description="Service identifier for logging and monitoring",
    )
    log_level: str = Field(
        default="info",
        description="Logging level (debug, info, warning, error, critical)",
    )

    # Model variant selection
    model_variant: Literal["beto", "roberta"] = Field(
        default="beto",
        description="Transformer variant to use (beto or roberta)",
    )

    # Hugging Face model IDs
    transformer_beto_model_id: str = Field(
        default="NicolasUnivalle/beto-vm-ner-full",
        description="Hugging Face model ID for BETO-based clinical NER",
    )
    transformer_roberta_model_id: str = Field(
        default="NicolasUnivalle/roberta-vm-ner-full",
        description="Hugging Face model ID for RoBERTa-based clinical NER",
    )

    # Device configuration
    device: Optional[str] = Field(
        default=None,
        description="Device for model inference (cuda, cpu, or None for auto-detect)",
    )


@lru_cache
def get_settings() -> Settings:
    """
    Get cached settings instance.

    Returns:
        Settings: Singleton settings instance loaded from environment.
    """
    return Settings()
