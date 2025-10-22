# Este script define funciones para almacenar resultados procesados en una base de datos MongoDB.
# Incluye lógica para evitar duplicados basados en un hash de contenido y permite asociar notas a episodios.

import hashlib
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from pymongo.database import Database


# Función auxiliar para calcular el hash SHA-256 de una cadena.
# Esto se utiliza para identificar de manera única el contenido de las notas.
def _sha256(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


# Función principal para guardar un resultado en la base de datos.
# Parámetros clave:
# - `db`: Conexión a la base de datos MongoDB.
# - `payload`: Texto o contenido de la nota.
# - `result`: Resultado procesado que contiene entidades y metadatos.
# - `model`: Nombre del modelo que generó el resultado.
# - `episode_id`: Identificador del episodio al que se asocia la nota.
# - `dedupe_by_hash`: Si es True, evita duplicados basados en el hash del contenido.
# Retorna el `note_id` de la nota almacenada.
def save_result(
    db: Database,
    payload: str,
    result: Any,
    model: str,
    *,
    episode_id: str,
    note_date_iso: Optional[str] = None,
    filename: Optional[str] = None,
    source_system: Optional[str] = None,
    dedupe_by_hash: bool = True,
) -> str:
    # Obtiene la fecha y hora actual en formato UTC.
    created_at = datetime.now(timezone.utc)

    # Calcula un hash único para el contenido de la nota.
    content_hash = _sha256(payload or "")
    # Genera un identificador único para la nota.
    note_id = str(uuid.uuid4())

    # Construye el diccionario que representa la nota a almacenar.
    note: Dict[str, Any] = {
        "note_id": note_id,
        "filename": filename,
        "source_system": source_system,
        "text": payload,
        "entities": [
            e.model_dump() for e in getattr(result, "entities", [])
        ],  # Serializa las entidades del resultado.
        "meta": getattr(
            result, "meta", None
        ),  # Extrae metadatos del resultado si existen.
        "model": model,
        "note_date": note_date_iso,
        "created_at": created_at,
        "content_hash": content_hash,
    }

    # Obtiene la colección de episodios desde la base de datos.
    episodes = db.episodes

    if dedupe_by_hash:
        # Intenta insertar la nota solo si no existe otra con el mismo hash en el episodio.
        upd = episodes.update_one(
            {
                "_id": episode_id,
                "notes": {
                    "$not": {"$elemMatch": {"content_hash": content_hash}}
                },  # Verifica que no exista el hash.
            },
            {
                "$setOnInsert": {
                    "_id": episode_id,
                    "created_at": created_at,
                },  # Crea el episodio si no existe.
                "$push": {"notes": note},  # Agrega la nueva nota al array de notas.
                "$set": {
                    "updated_at": created_at
                },  # Actualiza la fecha de modificación del episodio.
            },
            upsert=True,  # Permite insertar el episodio si no existe.
        )

        # Si se modificó el documento, significa que la nota fue insertada.
        if upd.modified_count == 1:
            return note_id

        # Si no se insertó, busca una nota existente con el mismo hash.
        existing = episodes.find_one(
            {"_id": episode_id, "notes.content_hash": content_hash},
            {"notes.$": 1},  # Proyecta solo la nota coincidente.
        )
        if existing and "notes" in existing and existing["notes"]:
            return existing["notes"][0][
                "note_id"
            ]  # Retorna el `note_id` de la nota existente.

        # Si no se encuentra una nota existente, realiza un intento de inserción como fallback.
        fallback = episodes.update_one(
            {"_id": episode_id},
            {
                "$setOnInsert": {"_id": episode_id, "created_at": created_at},
                "$push": {"notes": note},
                "$set": {"updated_at": created_at},
            },
            upsert=True,
        )
        return note_id
    else:
        # Si no se requiere deduplicación, simplemente inserta la nota en el episodio.
        episodes.update_one(
            {"_id": episode_id},
            {
                "$setOnInsert": {"_id": episode_id, "created_at": created_at},
                "$push": {"notes": note},
                "$set": {"updated_at": created_at},
            },
            upsert=True,
        )
        return note_id
