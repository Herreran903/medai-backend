from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import get_settings
from app.deps import _get_mongo_client_cached
from app.indexes import ensure_indexes
from app.routers.extract import router as extract_router


def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(title=settings.app_name, version="0.1.0")

    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.on_event("startup")
    def on_startup():
        client = _get_mongo_client_cached(settings.mongodb_uri)
        db = client[settings.mongodb_db]
        ensure_indexes(db)

    @app.get("/healthz")
    def healthz():
        return {"status": "ok"}

    app.include_router(extract_router)
    return app


app = create_app()
