"""
FastAPI Dependency Injection Module for MedAI Backend.

This module provides dependency injection functions for FastAPI routes,
implementing the Dependency Injection pattern to decouple route handlers
from infrastructure concerns such as database connections and configuration.

Architecture Context:
    MedAI uses FastAPI's ``Depends()`` mechanism to inject shared resources
    into route handlers. This module centralizes all injectable dependencies,
    ensuring consistent resource management and enabling easy testing through
    dependency overrides.

Key Dependencies:
    - :func:`settings_dep`: Application configuration access
    - :func:`get_mongo_client`: MongoDB client connection (cached)
    - :func:`get_db`: MongoDB database instance (per-request)
    - :func:`get_episode_repository`: Episode repository for data access
    - :func:`get_extraction_service`: Extraction service for business logic
    - :func:`get_cors_origins`: CORS allowed origins list

Lifecycle Management:
    - MongoDB client is cached at the module level via LRU cache
    - Database instances are yielded per-request for proper resource cleanup
    - Settings are cached via :func:`app.config.get_settings`
    - Repository and Service instances are created per-request

Usage in Routes:
    >>> from fastapi import Depends
    >>> from app.deps import get_extraction_service
    >>>
    >>> @router.post("/extract")
    >>> async def extract(
    ...     service: ExtractionService = Depends(get_extraction_service),
    ... ):
    ...     # service is automatically injected with all dependencies
    ...     return await service.extract_single(...)

See Also:
    - :mod:`app.config` for Settings class definition
    - :mod:`app.routers.extract` for dependency usage examples
    - :mod:`app.repositories` for repository pattern implementation
    - :mod:`app.services.extraction_service` for service layer
"""

from __future__ import annotations

from functools import lru_cache
from typing import Generator, Iterable

from fastapi import Depends
from pymongo import MongoClient
from pymongo.database import Database

from app.config import Settings, get_settings


def settings_dep() -> Settings:
    """
    Provide application settings as a FastAPI dependency.

    This function wraps :func:`app.config.get_settings` for use with
    FastAPI's dependency injection system, enabling route handlers to
    declare settings as a parameter.

    Returns:
        Settings: The cached application configuration instance.

    Example:
        >>> @router.get("/config")
        >>> def get_config(settings: Settings = Depends(settings_dep)):
        ...     return {"app_name": settings.app_name}

    Note:
        Settings are cached at the application level, so this dependency
        is effectively a singleton across all requests.
    """
    return get_settings()


@lru_cache
def _get_mongo_client_cached(uri: str) -> MongoClient:
    """
    Create and cache a MongoDB client connection.

    This internal function implements connection pooling by caching
    the MongoClient instance based on the connection URI. The LRU cache
    ensures that only one client exists per unique URI.

    Args:
        uri: MongoDB connection string (e.g., "mongodb://localhost:27017").
            Supports authentication, replica sets, and connection options.

    Returns:
        MongoClient: A configured MongoDB client with timezone awareness
            and standard UUID representation.

    Configuration:
        - ``tz_aware=True``: All datetime objects are timezone-aware (UTC)
        - ``uuidRepresentation="standard"``: Uses standard UUID binary format

    Warning:
        This function should not be called directly. Use :func:`get_mongo_client`
        as a FastAPI dependency instead.

    Note:
        The client connection pool is managed by PyMongo and persists
        for the application lifetime. Connection cleanup occurs on
        application shutdown.
    """
    return MongoClient(uri, tz_aware=True, uuidRepresentation="standard")


def get_mongo_client(settings: Settings = Depends(settings_dep)) -> MongoClient:
    """
    Provide a MongoDB client as a FastAPI dependency.

    This dependency injects a cached MongoDB client into route handlers,
    using the connection URI from application settings.

    Args:
        settings: Application settings (auto-injected via Depends).

    Returns:
        MongoClient: A reusable MongoDB client instance.

    Example:
        >>> @router.get("/health/db")
        >>> def check_db(client: MongoClient = Depends(get_mongo_client)):
        ...     return {"connected": client.server_info() is not None}

    Note:
        The client is cached per URI, so multiple requests share the same
        connection pool. This is the recommended pattern for MongoDB in
        web applications.
    """
    return _get_mongo_client_cached(settings.mongodb_uri)


def get_db(
    client: MongoClient = Depends(get_mongo_client),
    settings: Settings = Depends(settings_dep),
) -> Generator[Database, None, None]:
    """
    Provide a MongoDB database instance as a FastAPI dependency.

    This is the primary database dependency for route handlers. It yields
    the configured database from the MongoDB client, enabling access to
    collections for episode and note storage.

    Args:
        client: MongoDB client (auto-injected via Depends).
        settings: Application settings (auto-injected via Depends).

    Yields:
        Database: The MongoDB database instance specified in settings.

    Example:
        >>> @router.post("/notes")
        >>> async def create_note(db: Database = Depends(get_db)):
        ...     result = db.episodes.insert_one({"notes": []})
        ...     return {"id": str(result.inserted_id)}

    Collections:
        The MedAI database contains the following collections:

        - ``episodes``: Clinical episode documents containing patient notes
          and extracted entities. See :mod:`app.repositories.episode_repository` for schema.

    Note:
        This dependency uses a generator pattern for potential future
        cleanup logic, though MongoDB connections are pooled and don't
        require per-request cleanup.
    """
    db = client[settings.mongodb_db]
    yield db


def get_cors_origins(settings: Settings = Depends(settings_dep)) -> Iterable[str]:
    """
    Provide CORS allowed origins as a FastAPI dependency.

    This dependency exposes the configured CORS origins list for use
    in middleware configuration or dynamic CORS handling.

    Args:
        settings: Application settings (auto-injected via Depends).

    Returns:
        Iterable[str]: List of allowed origin URLs for CORS.

    Example:
        >>> @router.get("/cors-config")
        >>> def get_cors(origins: Iterable[str] = Depends(get_cors_origins)):
        ...     return {"allowed_origins": list(origins)}

    Note:
        CORS middleware is typically configured at application startup
        in :mod:`app.main`. This dependency is provided for cases where
        dynamic CORS handling is needed.
    """
    return settings.cors_origins


def get_episode_repository(db: Database = Depends(get_db)):
    """
    Provide an EpisodeRepository instance as a FastAPI dependency.

    This dependency creates a repository instance for data access operations,
    implementing the Repository pattern to abstract database operations.

    Args:
        db: MongoDB database (auto-injected via Depends).

    Returns:
        EpisodeRepository: Repository instance for episode/note operations.

    Example:
        >>> @router.get("/episodes/{episode_id}")
        >>> async def get_episode(
        ...     episode_id: str,
        ...     repo: EpisodeRepository = Depends(get_episode_repository)
        ... ):
        ...     return await repo.get_episode(episode_id)

    Note:
        A new repository instance is created per request, but it shares
        the same database connection from the connection pool.
    """
    from app.repositories import EpisodeRepository

    return EpisodeRepository(db)


def get_extraction_service(
    repository=Depends(get_episode_repository), settings: Settings = Depends(settings_dep)
):
    """
    Provide an ExtractionService instance as a FastAPI dependency.

    This dependency creates a service layer instance that orchestrates
    extraction operations, implementing the Service Layer pattern to
    separate business logic from API routing.

    Args:
        repository: Episode repository (auto-injected via Depends).
        settings: Application settings (auto-injected via Depends).

    Returns:
        ExtractionService: Service instance for extraction operations.

    Example:
        >>> @router.post("/extract")
        >>> async def extract(
        ...     service: ExtractionService = Depends(get_extraction_service)
        ... ):
        ...     return await service.extract_single(
        ...         text="...",
        ...         model="transformer",
        ...         episode_id="EP-001",
        ...         note_date="2024-01-15"
        ...     )

    Note:
        The service is created per request with injected dependencies,
        providing a clean interface for business logic without direct
        database or configuration access in route handlers.
    """
    from app.services.extraction_service import ExtractionService

    return ExtractionService(repository, settings)
