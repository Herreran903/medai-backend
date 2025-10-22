# config.py
# Este archivo define la configuración principal de la aplicación MedAI Backend.
# Utiliza Pydantic para manejar la configuración basada en variables de entorno,
# con soporte para valores predeterminados y validación de tipos.

from __future__ import annotations

from functools import lru_cache
from typing import List, Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


# Clase principal para manejar la configuración de la aplicación.
# Utiliza Pydantic para definir y validar los parámetros de configuración.
class Settings(BaseSettings):
    # Nombre de la aplicación, configurable mediante la variable de entorno APP_NAME.
    app_name: str = Field(default="MedAI Backend", env="APP_NAME")

    # Entorno de ejecución (e.g., desarrollo, producción), configurable mediante ENVIRONMENT.
    environment: str = Field(default="dev", env="ENVIRONMENT")

    # Dirección del host donde se ejecutará la aplicación, configurable mediante HOST.
    host: str = Field(default="0.0.0.0", env="HOST")

    # Puerto donde se ejecutará la aplicación, configurable mediante PORT.
    port: int = Field(default=8000, env="PORT")

    # Nivel de registro para la aplicación (e.g., info, debug), configurable mediante LOG_LEVEL.
    log_level: str = Field(default="info", env="LOG_LEVEL")

    # Lista de orígenes permitidos para solicitudes CORS.
    # Esto es útil para habilitar el acceso desde aplicaciones frontend específicas.
    cors_origins: List[str] = [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ]

    # URI de conexión a la base de datos MongoDB, configurable mediante MONGODB_URI.
    mongodb_uri: str = Field(default="mongodb://mongo:27017", env="MONGODB_URI")

    # Nombre de la base de datos en MongoDB, configurable mediante MONGODB_DB.
    mongodb_db: str = Field(default="medai", env="MONGODB_DB")

    # Bandera para habilitar o deshabilitar el guardado de resultados, configurable mediante SAVE_RESULTS.
    save_results: bool = Field(default=True, env="SAVE_RESULTS")

    # Modelo predeterminado que se utilizará en la aplicación.
    default_model: str = "transformer"

    # Lista de modelos habilitados para su uso en la aplicación.
    models_enabled: List[str] = ["lstm", "transformer", "llm"]

    # API Key opcional para interactuar con UMLS (Unified Medical Language System).
    umls_apikey: Optional[str] = Field(default=None, env="UMLS_APIKEY")

    # Configuración adicional para Pydantic Settings.
    model_config = SettingsConfigDict(
        env_file=".env.dev",  # Archivo de entorno predeterminado.
        env_file_encoding="utf-8",  # Codificación del archivo de entorno.
        case_sensitive=False,  # Las variables de entorno no son sensibles a mayúsculas/minúsculas.
        env_ignore_empty=True,  # Ignora variables de entorno vacías.
    )


# Función para obtener una instancia única de la configuración.
# Utiliza lru_cache para almacenar en caché la instancia y evitar múltiples inicializaciones.
@lru_cache
def get_settings() -> Settings:
    # Retorna una instancia de la clase Settings.
    return Settings()
