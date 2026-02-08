"""
MedAI Backend Application Entry Point.

This module defines the FastAPI application factory and global application instance
for the MedAI clinical entity extraction backend.

Architecture Context:
    MedAI Backend is a RESTful API service that provides:

    - Clinical Named Entity Recognition (NER) from medical notes
    - Multiple extraction models (LSTM, Transformer, LLM)
    - Entity normalization to standard medical terminologies (SNOMED-CT, ICD-10),
      currently disabled in the gateway flow
    - Persistent storage of extraction results in MongoDB

    The application follows the Factory pattern for testability and supports
    both development and production deployment configurations.

Application Structure:
    - ``/extract``: Single note entity extraction endpoint
    - ``/extract-batch``: Batch processing for multiple files
    - ``/notes/{note_id}``: Retrieve stored extraction results
    - ``/health``: Health check endpoint for container orchestration

Startup Sequence:
    1. Load configuration from environment via :mod:`app.config`
    2. Initialize FastAPI with OpenAPI metadata
    3. Configure CORS middleware for frontend access
    4. Establish MongoDB connection and ensure indexes
    5. Register API routers

Usage:
    Development server::

        uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

    Production deployment::

        gunicorn app.main:app -w 4 -k uvicorn.workers.UvicornWorker

See Also:
    - :mod:`app.config` for configuration settings
    - :mod:`app.routers.extract` for API endpoint definitions
    - :mod:`app.deps` for dependency injection
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import get_settings
from app.deps import _get_mongo_client_cached
from app.indexes import ensure_indexes
from app.routers.extract import router as extract_router


def create_app() -> FastAPI:
    """
    Create and configure the FastAPI application instance.

    This factory function implements the application factory pattern,
    enabling multiple application instances for testing and supporting
    different configurations per environment.

    The function performs the following initialization:

    1. **Configuration Loading**: Retrieves settings from environment
    2. **FastAPI Initialization**: Creates app with OpenAPI metadata
    3. **CORS Configuration**: Enables cross-origin requests from frontend
    4. **Startup Events**: Registers database initialization handlers
    5. **Router Registration**: Mounts API endpoint routers

    Returns:
        FastAPI: A fully configured FastAPI application instance.

    Configuration:
        The application title and version are derived from :class:`app.config.Settings`.
        CORS origins are configured to allow the MedAI frontend application.

    Middleware:
        - **CORSMiddleware**: Configured with:
            - ``allow_origins``: From settings.cors_origins
            - ``allow_credentials``: True (for cookie-based auth if needed)
            - ``allow_methods``: All HTTP methods (*)
            - ``allow_headers``: All headers (*)

    Events:
        - **startup**: Initializes MongoDB connection and creates indexes

    Example:
        >>> app = create_app()
        >>> # Use with TestClient for testing
        >>> from fastapi.testclient import TestClient
        >>> client = TestClient(app)
        >>> response = client.get("/health")
        >>> assert response.status_code == 200

    Note:
        For production deployments, ensure environment variables are set
        for MongoDB connection and any required API keys.
    """
    settings = get_settings()

    app = FastAPI(
        title=settings.app_name,
        version="0.1.0",
        description=(
            "MedAI Backend API for clinical entity extraction from medical notes. "
            "Supports multiple NER models including LSTM, Transformer (RoBERTa), "
            "and LLM-based extraction."
        ),
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
    )

    # Configure CORS middleware for frontend access
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.on_event("startup")
    def on_startup():
        """
        Application startup event handler.

        Performs initialization tasks that require the application context:

        1. Establishes MongoDB connection using cached client
        2. Ensures required database indexes exist

        This handler runs once when the application starts, before
        accepting any requests.

        Note:
            Model loading is deferred to first request to reduce startup time.
            See :mod:`app.services.registry` for lazy model initialization.
        """
        client = _get_mongo_client_cached(settings.mongodb_uri)
        db = client[settings.mongodb_db]
        ensure_indexes(db)

    @app.get(
        "/health",
        tags=["Health"],
        summary="Health check endpoint",
        description=(
            "Returns the application health status. Standard health check endpoint "
            "compatible with load balancers and orchestration systems."
        ),
        response_description="Health status object with 'ok' status on success.",
    )
    def health():
        """
        Health check endpoint.

        Returns:
            dict: Health status object with ``status`` field.

        Response Example:
            .. code-block:: json

                {"status": "ok"}
        """
        return {"status": "ok"}

    # Register API routers
    app.include_router(extract_router)

    return app


# Global application instance for ASGI servers
app = create_app()
"""
Global FastAPI application instance.

This instance is used by ASGI servers (Uvicorn, Gunicorn) to serve
the application. It is created at module import time.

Usage with Uvicorn::

    uvicorn app.main:app --host 0.0.0.0 --port 8000

Usage with Gunicorn::

    gunicorn app.main:app -w 4 -k uvicorn.workers.UvicornWorker
"""
