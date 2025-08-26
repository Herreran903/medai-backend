from __future__ import annotations

from typing import List

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # App
    app_name: str = "MedAI Backend"
    environment: str = "dev"
    host: str = "0.0.0.0"
    port: int = 8000
    log_level: str = "info"

    # CORS (lista separada por comas en .env)
    cors_origins: List[str] = Field(default_factory=lambda: ["http://localhost:3000"])

    # Persistencia
    mongodb_uri: str = "mongodb://localhost:27017"
    mongodb_db: str = "medai"
    save_results: bool = True

    # Modelos
    default_model: str = "transformer"
    models_enabled: List[str] = Field(
        default_factory=lambda: ["lstm", "transformer", "llm"]
    )

    model_config = SettingsConfigDict(
        env_file=".env.dev",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )


from functools import lru_cache


@lru_cache
def get_settings() -> Settings:
    return Settings()
