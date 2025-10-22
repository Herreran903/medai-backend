# Este archivo define las dependencias principales para la aplicación FastAPI,
# incluyendo la configuración, la conexión a la base de datos MongoDB y el manejo de CORS.

from __future__ import annotations

from functools import lru_cache
from typing import Generator, Iterable

from fastapi import Depends
from pymongo import MongoClient
from pymongo.database import Database

from app.config import Settings, get_settings


# Devuelve la configuración de la aplicación.
# Utiliza la función `get_settings` para obtener los valores de configuración.
def settings_dep() -> Settings:
    return get_settings()


# Crea y cachea una instancia de MongoClient para conectarse a MongoDB.
# Esto evita crear múltiples conexiones innecesarias al reutilizar la misma instancia.
# - `uri`: URI de conexión a MongoDB.
# Devuelve un cliente de MongoDB configurado.
@lru_cache
def _get_mongo_client_cached(uri: str) -> MongoClient:
    return MongoClient(uri, tz_aware=True, uuidRepresentation="standard")


# Proporciona un cliente de MongoDB como dependencia para FastAPI.
# Utiliza la configuración de la aplicación para obtener el URI de conexión.
# - `settings`: Configuración de la aplicación, inyectada como dependencia.
# Devuelve un cliente de MongoDB reutilizable.
def get_mongo_client(settings: Settings = Depends(settings_dep)) -> MongoClient:
    return _get_mongo_client_cached(settings.mongodb_uri)


# Proporciona una instancia de la base de datos MongoDB como dependencia para FastAPI.
# - `client`: Cliente de MongoDB, inyectado como dependencia.
# - `settings`: Configuración de la aplicación, inyectada como dependencia.
# Devuelve un generador que produce la base de datos especificada en la configuración.
def get_db(
    client: MongoClient = Depends(get_mongo_client),
    settings: Settings = Depends(settings_dep),
) -> Generator[Database, None, None]:
    # Obtiene la base de datos especificada en la configuración.
    db = client[settings.mongodb_db]
    yield db


# Proporciona los orígenes permitidos para CORS como dependencia para FastAPI.
# - `settings`: Configuración de la aplicación, inyectada como dependencia.
# Devuelve un iterable con los orígenes permitidos.
def get_cors_origins(settings: Settings = Depends(settings_dep)) -> Iterable[str]:
    return settings.cors_origins
