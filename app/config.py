# app/config.py
from __future__ import annotations

from functools import lru_cache
from typing import List, Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # App
    app_name: str = Field(default="MedAI Backend", env="APP_NAME")
    environment: str = Field(default="dev", env="ENVIRONMENT")
    host: str = Field(default="0.0.0.0", env="HOST")
    port: int = Field(default=8000, env="PORT")
    log_level: str = Field(default="info", env="LOG_LEVEL")

    # CORS
    cors_origins: List[str] = [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ]

    # Persistencia
    mongodb_uri: str = Field(default="mongodb://mongo:27017", env="MONGODB_URI")
    mongodb_db: str = Field(default="medai", env="MONGODB_DB")
    save_results: bool = Field(default=True, env="SAVE_RESULTS")

    # Modelos
    default_model: str = "transformer"
    models_enabled: List[str] = ["lstm", "transformer", "llm"]

    # 🔑 API Key de UMLS
    umls_apikey: Optional[str] = Field(default=None, env="UMLS_APIKEY")

    model_config = SettingsConfigDict(
        env_file=".env.dev",
        env_file_encoding="utf-8",
        case_sensitive=False,
        env_ignore_empty=True,
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()
