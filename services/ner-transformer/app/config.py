"""
Configuration settings for the Transformer NER microservice.

This module defines environment-based configuration for the Transformer service,
including model selection, Hugging Face model IDs, and service settings.
"""

from functools import lru_cache
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Transformer NER Service configuration settings.

    All settings can be overridden via environment variables.
    Example: TRANSFORMER_STRIDE=128 overrides the transformer_stride field.
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

    # Hugging Face model ID for RoBERTa NER.
    # The service loads weights and tokenizer from the HF Hub at startup
    # (cached in /root/.cache/huggingface via the docker volume).
    transformer_roberta_model_id: str = Field(
        default="NicolasUnivalle/roberta-vm-ner",
        description="Hugging Face model ID for RoBERTa NER",
    )

    # Device configuration
    device: Optional[str] = Field(
        default=None,
        description="Device for model inference (cuda, cpu, or None for auto-detect)",
    )

    # Tokenization windowing
    transformer_max_len: int = Field(
        default=512,
        description="Maximum tokenized sequence length per window (RoBERTa default is 512).",
    )
    transformer_stride: int = Field(
        default=128,
        description=(
            "Sliding-window overlap (tokens) used when processing long notes. "
            "Must be < transformer_max_len."
        ),
    )
    transformer_window_batch_size: int = Field(
        default=8,
        description=(
            "Number of windows processed per forward pass. Lower values reduce peak memory "
            "usage for very long notes."
        ),
    )


@lru_cache
def get_settings() -> Settings:
    """
    Get cached settings instance.

    Returns:
        Settings: Singleton settings instance loaded from environment.
    """
    return Settings()
