from __future__ import annotations

from functools import lru_cache
from typing import Generator, Iterable

from fastapi import Depends
from pymongo import MongoClient
from pymongo.database import Database

from app.config import Settings, get_settings


def settings_dep() -> Settings:
    return get_settings()


@lru_cache
def _get_mongo_client_cached(uri: str) -> MongoClient:
    return MongoClient(uri, tz_aware=True, uuidRepresentation="standard")


def get_mongo_client(settings: Settings = Depends(settings_dep)) -> MongoClient:
    return _get_mongo_client_cached(settings.mongodb_uri)


def get_db(
    client: MongoClient = Depends(get_mongo_client),
    settings: Settings = Depends(settings_dep),
) -> Generator[Database, None, None]:
    db = client[settings.mongodb_db]
    yield db


def get_cors_origins(settings: Settings = Depends(settings_dep)) -> Iterable[str]:
    return settings.cors_origins
