"""
MedAI Backend Configuration Module.

This module defines the centralized configuration for the MedAI backend application,
providing environment-based settings management with type validation through Pydantic.

The configuration system supports:
    - Environment variable overrides for all settings
    - Default values suitable for development environments
    - Type validation and coercion via Pydantic
    - Singleton pattern through LRU caching for consistent access

Architecture Context:
    This module serves as the single source of truth for application configuration,
    consumed by dependency injection in FastAPI routes, service initialization,
    and infrastructure components (MongoDB, CORS, model loading).

Usage:
    Configuration is accessed via the cached :func:`get_settings` function, which
    ensures a single :class:`Settings` instance throughout the application lifecycle.

Example:
    >>> from app.config import get_settings
    >>> settings = get_settings()
    >>> print(settings.mongodb_uri)
    'mongodb://mongo:27017'

See Also:
    - :mod:`app.deps` for dependency injection using these settings
    - :mod:`app.services.registry` for model configuration consumption
"""

from __future__ import annotations

from functools import lru_cache
from typing import List

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application-wide configuration settings for MedAI Backend.

    This class centralizes all configurable parameters for the MedAI backend,
    including server configuration, database connections, CORS policies,
    model selection, and external API integrations.

    All settings can be overridden via environment variables. The environment
    variable name matches the attribute name in uppercase (e.g., `app_name` -> `APP_NAME`).

    Attributes:
        app_name: Human-readable application name used in OpenAPI documentation
            and logging. Defaults to "MedAI Backend".
        environment: Deployment environment identifier (e.g., "dev", "prod", "staging").
            Used for conditional behavior and logging context.
        host: Network interface to bind the server. Use "0.0.0.0" for container
            deployments to accept external connections.
        port: TCP port for the HTTP server. Standard FastAPI/Uvicorn port.
        log_level: Logging verbosity level (e.g., "debug", "info", "warning", "error").
            Applied to the root logger configuration.
        cors_origins: Allowed origins for Cross-Origin Resource Sharing (CORS).
            Must include the frontend application URL for browser-based access.
        mongodb_uri: MongoDB connection string. Supports replica sets and
            authentication parameters.
        mongodb_db: Target database name within the MongoDB instance.
        save_results: Global flag to enable/disable persistence of extraction results.
            When False, the API operates in stateless mode.
        default_model: Fallback model identifier when no model is specified in requests.
            Must be a key in :data:`app.services.registry.MODEL_REGISTRY`.

    Configuration Loading:
        Settings are loaded from environment variables with fallback to `.env.dev`.
        The loading order is: environment variables > .env file > defaults.

    Example:
        >>> settings = Settings()
        >>> settings.mongodb_uri
        'mongodb://mongo:27017'
    """

    app_name: str = Field(
        default="MedAI Backend",
        description="Application name displayed in OpenAPI documentation and logs.",
    )

    environment: str = Field(
        default="dev",
        description="Deployment environment identifier (dev, staging, prod).",
    )

    host: str = Field(
        default="0.0.0.0",
        description="Network interface for server binding. Use 0.0.0.0 for containers.",
    )

    port: int = Field(
        default=8000,
        description="HTTP server port number.",
    )

    log_level: str = Field(
        default="info",
        description="Logging verbosity level (debug, info, warning, error, critical).",
    )

    cors_origins: List[str] = [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "https://medai-frontend-seven.vercel.app",
    ]
    """
    Allowed CORS origins for cross-origin requests.
    
    This list must include all frontend application URLs that will access
    the API. For production, ensure only trusted origins are included.
    """

    mongodb_uri: str = Field(
        default="mongodb://mongo:27017",
        description="MongoDB connection URI with optional authentication.",
    )

    mongodb_db: str = Field(
        default="medai",
        description="Target MongoDB database name for episode and note storage.",
    )

    save_results: bool = Field(
        default=True,
        description="Enable persistence of extraction results to MongoDB.",
    )

    default_model: str = "transformer"
    """
    Default extraction model when not specified in API requests.
    
    Must correspond to a key in :data:`app.services.registry.MODEL_REGISTRY`.
    The transformer model provides the best balance of accuracy and performance
    for clinical NER tasks.
    """

    # Microservices configuration (always remote - gateway delegates to NER services)
    ner_transformer_url: str = Field(
        default="http://ner-transformer:8001",
        description="Transformer NER service URL.",
    )

    ner_lstm_url: str = Field(
        default="http://ner-bilstm:8002",
        description="BiLSTM NER service URL.",
    )

    ner_llm_url: str = Field(
        default="http://ner-llm:8003",
        description="LLM NER service URL.",
    )

    ner_request_timeout: float = Field(
        default=120.0,
        description="Timeout for HTTP requests to NER services in seconds.",
    )

    @field_validator("cors_origins", mode="before")
    @classmethod
    def parse_string_list(cls, v):
        """Parse list fields from comma-separated string or list."""
        if isinstance(v, str):
            return [item.strip() for item in v.split(",") if item.strip()]
        return v

    model_config = SettingsConfigDict(
        env_file=".env.dev",
        env_file_encoding="utf-8",
        case_sensitive=False,
        env_ignore_empty=True,
        extra="ignore",
    )
    """
    Pydantic Settings configuration.
    
    - ``env_file``: Default environment file for local development
    - ``env_file_encoding``: UTF-8 encoding for environment file parsing
    - ``case_sensitive``: Environment variables are case-insensitive
    - ``env_ignore_empty``: Empty environment variables are treated as unset
    """


@lru_cache
def get_settings() -> Settings:
    """
    Retrieve the singleton Settings instance.

    This function provides cached access to the application configuration,
    ensuring consistent settings across all components and avoiding
    repeated environment variable parsing.

    The LRU cache guarantees that only one Settings instance exists
    throughout the application lifecycle, implementing the singleton pattern.

    Returns:
        Settings: The cached application configuration instance.

    Example:
        >>> from app.config import get_settings
        >>> settings = get_settings()
        >>> settings.app_name
        'MedAI Backend'

    Note:
        For testing, the cache can be cleared with ``get_settings.cache_clear()``
        to allow configuration changes between test cases.
    """
    return Settings()
