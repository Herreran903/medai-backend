"""
CRF NER Microservice - FastAPI Application.

This microservice provides clinical Named Entity Recognition using
an sklearn-crfsuite CRF model via a REST API.

Endpoints:
    - POST /predict: Extract entities from clinical text
    - GET /health: Liveness probe
    - GET /readyz: Readiness probe

The service implements the standardized NER contract defined in shared.schemas,
ensuring compatibility with the API gateway.
"""

import logging
import os
import sys
import time
from typing import Optional

from fastapi import FastAPI, HTTPException, status
from fastapi.responses import JSONResponse

# Add parent directory to path for shared module access
sys.path.insert(0, "/app")

from shared.schemas import Entity, ErrorResponse, NERRequest, NERResponse

from app.config import get_settings

DOCS_BUILD = os.getenv("DOCS_BUILD") == "1"

if DOCS_BUILD:
    class CRFExtractor:  # type: ignore
        """Placeholder to allow OpenAPI export without heavy deps."""

        pass
else:
    from app.extractor import CRFExtractor

# Configure logging
settings = get_settings()
logging.basicConfig(
    level=settings.log_level.upper(),
    format="[%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# FastAPI application
app = FastAPI(
    title="MedAI CRF NER Service",
    version="1.0.0",
    description=(
        "Clinical Named Entity Recognition microservice using a pre-trained "
        "sklearn-crfsuite CRF model. Extracts structured medical entities "
        "from Spanish clinical notes with focus on mechanical ventilation parameters."
    ),
)

# Global extractor instance
_extractor: Optional[CRFExtractor] = None
_model_loaded = False


# ==============================================================================
# Lifecycle Events
# ==============================================================================


@app.on_event("startup")
async def startup_event():
    """
    Initialize CRF extractor at startup.

    Loads the CRF model from local files (crf_model.pkl, tag2idx.json, config.json).
    """
    global _extractor, _model_loaded

    if DOCS_BUILD:
        logger.info("DOCS_BUILD=1; skipping CRF model load")
        _model_loaded = False
        return

    logger.info("Starting CRF NER service")

    try:
        logger.info("Loading CRF model from: %s", settings.model_dir)
        _extractor = CRFExtractor(model_dir=settings.model_dir.as_posix())
        _model_loaded = True
        logger.info("CRF model loaded successfully")
    except Exception as e:
        logger.error("Failed to load CRF model: %s", e, exc_info=True)
        _model_loaded = False


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on service shutdown."""
    logger.info("Shutting down CRF NER service")


# ==============================================================================
# Health Check Endpoints
# ==============================================================================


@app.get(
    "/health",
    tags=["Health"],
    summary="Health check endpoint",
    description="Returns 200 OK if service is running.",
)
def health():
    """Liveness probe - returns OK if service is alive."""
    return {"status": "ok"}


@app.get(
    "/readyz",
    tags=["Health"],
    summary="Readiness probe",
    description=(
        "Returns 200 OK if service is ready to accept requests (model loaded). "
        "Returns 503 Service Unavailable if model still loading."
    ),
)
def readyz():
    if _model_loaded and _extractor is not None:
        return {
            "status": "ready",
            "model_loaded": True,
            "model_dir": str(settings.model_dir),
        }
    raise HTTPException(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        detail={
            "status": "loading",
            "model_loaded": False,
            "model_dir": str(settings.model_dir),
        },
    )


# ==============================================================================
# Extraction Endpoint
# ==============================================================================


@app.post(
    "/predict",
    response_model=NERResponse,
    responses={
        200: {
            "description": "Successful extraction",
            "model": NERResponse,
        },
        503: {
            "description": "Service not ready (model still loading)",
            "model": ErrorResponse,
        },
        500: {
            "description": "Extraction failed",
            "model": ErrorResponse,
        },
    },
    tags=["Extraction"],
    summary="Extract clinical entities from text",
    description=(
        "Processes clinical text using a CRF model and returns structured "
        "medical entities with character offsets."
    ),
)
def predict(request: NERRequest):
    if not _model_loaded or _extractor is None:
        logger.error("Extraction request received but model not ready")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "error": {
                    "code": "MODEL_NOT_LOADED",
                    "message": "CRF model not initialized",
                    "detail": f"Model directory: {settings.model_dir}",
                }
            },
        )

    if not request.text or not request.text.strip():
        logger.warning("Empty text received in extraction request")
        return NERResponse(
            entities=[],
            meta={
                "model": "crf",
                "count": 0,
            },
        )

    try:
        start_time = time.time()

        raw_entities = _extractor.predict(request.text)

        inference_time_ms = (time.time() - start_time) * 1000

        entities = []
        for e in raw_entities:
            try:
                entities.append(Entity(**e))
            except Exception as entity_error:
                logger.warning("Invalid entity format, skipping: %s (error: %s)", e, entity_error)
                continue

        meta = {
            "model": "crf",
            "count": len(entities),
            "inference_time_ms": round(inference_time_ms, 2),
        }

        if hasattr(_extractor, "meta"):
            try:
                meta.update(_extractor.meta())
            except Exception:
                pass

        return NERResponse(entities=entities, meta=meta)

    except Exception as e:
        logger.error("Extraction failed: %s", e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": {
                    "code": "EXTRACTION_FAILED",
                    "message": "CRF extraction error",
                    "detail": str(e),
                }
            },
        )


# ==============================================================================
# Exception Handlers
# ==============================================================================


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler for unhandled errors."""
    logger.error("Unhandled exception: %s", exc, exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": {
                "code": "INTERNAL_ERROR",
                "message": "An unexpected error occurred",
                "detail": str(exc),
            }
        },
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8004, log_level=settings.log_level.lower())

