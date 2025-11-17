# Este archivo define la aplicación principal de FastAPI para el backend del proyecto.
# Configura la aplicación, incluyendo middleware, rutas y eventos de inicio.

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import get_settings
from app.deps import _get_mongo_client_cached
from app.indexes import ensure_indexes
from app.routers.extract import router as extract_router


def create_app() -> FastAPI:
    """
    Crea y configura una instancia de la aplicación FastAPI.

    - Configura el middleware de CORS con los orígenes permitidos.
    - Define eventos de inicio para inicializar la base de datos y los índices.
    - Incluye rutas específicas para la funcionalidad de la aplicación.

    Retorna:
        FastAPI: La instancia configurada de la aplicación.
    """
    settings = get_settings()  # Obtiene la configuración global de la aplicación.

    app = FastAPI(
        title=settings.app_name, version="0.1.0"
    )  # Inicializa la aplicación con metadatos básicos.

    # Configuración del middleware de CORS para permitir solicitudes desde orígenes específicos.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,  # Lista de orígenes permitidos, definida en la configuración.
        allow_credentials=True,  # Permite el envío de credenciales en las solicitudes.
        allow_methods=["*"],  # Permite todos los métodos HTTP.
        allow_headers=["*"],  # Permite todos los encabezados HTTP.
    )

    @app.on_event("startup")
    def on_startup():
        """
        Evento de inicio de la aplicación.

        - Establece la conexión con la base de datos MongoDB.
        - Asegura que los índices necesarios estén creados en la base de datos.
        """
        client = _get_mongo_client_cached(
            settings.mongodb_uri
        )  # Obtiene un cliente de MongoDB con caché.
        db = client[
            settings.mongodb_db
        ]  # Selecciona la base de datos especificada en la configuración.
        ensure_indexes(
            db
        )  # Crea los índices necesarios en las colecciones de la base de datos.

    @app.get("/healthz")
    def healthz():
        """
        Endpoint de salud para verificar el estado de la aplicación.

        Retorna:
            dict: Un objeto JSON con el estado de la aplicación.
        """
        return {
            "status": "ok"
        }  # Respuesta estándar indicando que la aplicación está operativa.

    # Incluye el router para las rutas relacionadas con la funcionalidad de extracción.
    app.include_router(extract_router)
    return app


# Crea una instancia global de la aplicación para que sea utilizada por el servidor.
app = create_app()
