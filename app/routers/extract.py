# Este archivo define un conjunto de endpoints para un servicio de extracción de entidades
# desde texto o archivos. Incluye funciones para procesar texto, guardar resultados en una base
# de datos y manejar lotes de archivos. Utiliza FastAPI para la creación de rutas y MongoDB
# como base de datos.

import json
from datetime import datetime
from typing import Dict, List, Optional

from bson import ObjectId
from fastapi import (
    APIRouter,
    Depends,
    File,
    Form,
    HTTPException,
    Path,
    UploadFile,
    status,
)
from pymongo.database import Database

from app.config import Settings
from app.deps import get_db, settings_dep
from app.schemas import (
    BatchAckItem,
    BatchAckResponse,
    BatchItem,
    Entity,
    ExtractAck,
    ExtractResponse,
)
from app.services.pipeline import extract_from_text
from app.services.store import (
    save_result,  # Función para guardar resultados en la base de datos
)
from app.services.text_utils import read_any_to_text  # Convierte archivos a texto

# Se define un enrutador para los endpoints de extracción
router = APIRouter()


# Función auxiliar para analizar fechas en formato ISO 8601
def _parse_iso8601(dt: Optional[str]) -> Optional[datetime]:
    """
    Convierte una cadena en formato ISO 8601 a un objeto datetime.
    Si el formato es inválido, lanza una excepción HTTP 400.
    """
    if not dt:
        return None
    s = dt.strip()
    if s.endswith("Z"):  # Ajusta el formato UTC si termina en 'Z'
        s = s[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(s)
    except ValueError:
        try:
            # Intenta agregar una hora por defecto si solo se proporciona la fecha
            return datetime.fromisoformat(s + "T00:00:00")
        except Exception:
            raise HTTPException(status_code=400, detail="Formato de note_date inválido")


# Función auxiliar para convertir una cadena CSV en una lista de cadenas
def _parse_csv(v: Optional[str]) -> List[str]:
    """
    Convierte una cadena CSV en una lista de cadenas, eliminando espacios en blanco.
    """
    if not v:
        return []
    return [p.strip() for p in v.split(",") if p.strip()]


# Endpoint para extraer entidades desde texto o archivos
@router.post("/extract", response_model=ExtractAck, status_code=status.HTTP_201_CREATED)
async def extract(
    text: Optional[str] = Form(None),  # Texto proporcionado directamente
    file: Optional[UploadFile] = File(None),  # Archivo cargado por el usuario
    model: str = Form(...),  # Modelo de extracción a utilizar (lstm, transformer, llm)
    model_variant: Optional[str] = Form(
        None
    ),  # Variante del modelo (claude/gpt/local para llm, beto/roberta para transformer)
    episode_id: Optional[str] = Form(None),  # ID del episodio asociado
    note_date: Optional[str] = Form(None),  # Fecha de la nota en formato ISO 8601
    save: Optional[bool] = Form(True),  # Indica si se debe guardar el resultado
    normalize: Optional[bool] = Form(False),  # Normalización de entidades
    systems_csv: Optional[str] = Form(None),  # Sistemas específicos en formato CSV
    restrict_types_csv: Optional[str] = Form(None),  # Tipos de entidades a restringir
    expand: Optional[bool] = Form(False),  # Expande el resultado completo
    db: Database = Depends(get_db),  # Dependencia para obtener la base de datos
    settings: Settings = Depends(settings_dep),  # Configuración global
):
    """
    Procesa texto o archivos para extraer entidades usando un modelo específico.
    Guarda los resultados si está habilitado y devuelve un resumen de la operación.
    """
    if not text and not file:
        # Verifica que se proporcione al menos texto o un archivo
        raise HTTPException(status_code=400, detail="Proporciona 'text' o 'file'")

    if file:
        # Convierte el contenido del archivo a texto
        content = await file.read()
        text = read_any_to_text(file.filename, content)

    # Convierte las cadenas CSV en listas
    systems = _parse_csv(systems_csv)
    restrict_types = _parse_csv(restrict_types_csv)

    # Analiza y valida la fecha de la nota
    note_dt = _parse_iso8601(note_date)
    note_date_iso = note_dt.isoformat() if note_dt else None

    # Llama al servicio de extracción de texto con los parámetros proporcionados
    res = extract_from_text(
        text or "",
        model=model or settings.default_model,
        model_variant=model_variant,  # Pasa la variante del modelo
        normalize=bool(normalize),
        systems=systems,
        restrict_types=restrict_types or None,
    )

    # Verifica que se proporcione un ID de episodio
    if not episode_id:
        raise HTTPException(status_code=400, detail="Falta 'episode_id'")
    # Verifica que la fecha sea válida
    if not note_date_iso:
        raise HTTPException(
            status_code=400, detail="Falta 'note_date' o formato inválido"
        )

    # Inicializa variables para el almacenamiento
    note_id = None
    stored = False
    if save and settings.save_results:
        # Guarda el resultado en la base de datos si está habilitado
        note_id = save_result(
            db=db,
            payload=text or "",
            result=res,
            model=model,
            episode_id=episode_id,
            note_date_iso=note_date_iso,
            filename=getattr(file, "filename", None),
            source_system="api.extract",
            dedupe_by_hash=True,  # Evita duplicados basados en hash
        )
        stored = True

    # Construye la respuesta con los datos procesados
    ack = ExtractAck(
        id=note_id or "",
        stored=stored,
        url=(f"/notes/{note_id}" if note_id else None),
        filename=getattr(file, "filename", None),
        episode_id=episode_id,
        note_date=note_date_iso,
        entity_count=len(res.entities) if hasattr(res, "entities") else None,
        result=(
            res if expand else None
        ),  # Incluye el resultado completo si se solicita
    )
    return ack


# Endpoint para procesar múltiples archivos en un solo lote
@router.post("/extract-batch", response_model=BatchAckResponse)
async def extract_batch(
    files: List[UploadFile] = File(...),  # Lista de archivos cargados
    model: str = Form(...),  # Modelo de extracción a utilizar
    model_variant: Optional[str] = Form(None),  # Variante del modelo
    save: Optional[bool] = Form(True),  # Indica si se deben guardar los resultados
    normalize: Optional[bool] = Form(False),  # Normalización de entidades
    systems_csv: Optional[str] = Form(None),  # Sistemas específicos en formato CSV
    restrict_types_csv: Optional[str] = Form(None),  # Tipos de entidades a restringir
    notes_meta: Optional[str] = Form(None),  # JSON con metadatos por archivo
    db: Database = Depends(get_db),  # Dependencia para obtener la base de datos
    settings: Settings = Depends(settings_dep),  # Configuración global
):
    """
    Procesa un lote de archivos para extraer entidades usando un modelo específico.
    Devuelve un resumen de la operación para cada archivo.
    """
    # Convierte las cadenas CSV en listas
    systems = _parse_csv(systems_csv)
    restrict_types = _parse_csv(restrict_types_csv)

    # Parseo de metadatos por archivo (notes_meta: JSON con filename, episode_id, note_date)
    meta_by_filename: Dict[str, Dict[str, Optional[str]]] = {}
    if notes_meta:
        try:
            raw = json.loads(notes_meta)
            if isinstance(raw, list):
                for m in raw:
                    if isinstance(m, dict):
                        fn = str(m.get("filename") or "").strip()
                        if fn:
                            meta_by_filename[fn] = {
                                "episode_id": m.get("episode_id") or None,
                                "note_date": m.get("note_date") or None,
                            }
        except Exception:
            # Si el JSON viene inválido, continuamos sin metadatos
            meta_by_filename = {}

    items: List[BatchAckItem] = (
        []
    )  # Lista para almacenar los resultados de cada archivo
    for f in files:
        try:
            # Convierte el contenido del archivo a texto
            content = await f.read()
            text = read_any_to_text(f.filename, content)

            # Metadatos por archivo (obligatorios para comportarse como /extract)
            meta = meta_by_filename.get(f.filename, {})
            episode_id = meta.get("episode_id")
            note_date_raw = meta.get("note_date")
            note_dt = _parse_iso8601(note_date_raw) if note_date_raw else None
            note_date_iso = note_dt.isoformat() if note_dt else None

            if not episode_id:
                raise ValueError(f"Falta 'episode_id' para '{f.filename}'")
            if not note_date_iso:
                raise ValueError(
                    f"Falta 'note_date' o formato inválido para '{f.filename}'"
                )

            # Llama al servicio de extracción de texto
            res = extract_from_text(
                text,
                model=model or settings.default_model,
                model_variant=model_variant,
                normalize=bool(normalize),
                systems=systems,
                restrict_types=restrict_types or None,
            )

            note_id = None
            stored = False
            if save and settings.save_results:
                # Guarda el resultado en la base de datos si está habilitado
                note_id = save_result(
                    db=db,
                    payload=text,
                    result=res,
                    model=model,
                    episode_id=episode_id,
                    note_date_iso=note_date_iso,
                    filename=f.filename,
                    source_system="api.extract-batch",
                    dedupe_by_hash=True,
                )
                stored = True

            # Agrega el resultado exitoso a la lista
            items.append(
                BatchAckItem(
                    filename=f.filename,
                    id=note_id,
                    stored=stored,
                    entity_count=(
                        len(res.entities) if hasattr(res, "entities") else None
                    ),
                    url=(f"/notes/{note_id}" if note_id else None),
                )
            )
        except Exception as e:
            # Maneja errores y agrega el resultado fallido a la lista
            items.append(
                BatchAckItem(
                    filename=f.filename,
                    stored=False,
                    error=str(e),
                )
            )
    return BatchAckResponse(items=items)


# Endpoint para recuperar una nota específica desde la base de datos
@router.get("/notes/{note_id}", response_model=ExtractResponse)
async def get_note(
    note_id: str = Path(
        ..., description="UUID de la nota (note_id)"
    ),  # ID único de la nota
    db: Database = Depends(get_db),  # Dependencia para obtener la base de datos
):
    """
    Recupera una nota específica desde la base de datos usando su ID único.
    Devuelve el texto, las entidades y los metadatos asociados.
    """
    # Busca la nota en la colección de episodios
    doc = db.episodes.find_one(
        {"notes.note_id": note_id},
        {
            "_id": 0,
            "notes": {"$elemMatch": {"note_id": note_id}},
        },  # Filtra solo la nota requerida
    )

    if not doc or not doc.get("notes"):
        # Lanza una excepción si la nota no se encuentra
        raise HTTPException(status_code=404, detail="Nota no encontrada")

    # Extrae la nota y construye la respuesta
    note = doc["notes"][0]
    return ExtractResponse(
        text=note.get("text") or "",
        entities=[
            Entity(**e) for e in note.get("entities", [])
        ],  # Convierte las entidades a objetos
        meta=note.get("meta") or {},  # Incluye metadatos si están disponibles
    )
