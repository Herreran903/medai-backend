# Este archivo define los esquemas de datos utilizados en la aplicación backend de "medai".
# Los esquemas están diseñados con Pydantic para validar y estructurar datos relacionados con
# entidades extraídas, respuestas de extracción, y acuses de recibo (ACKs) en operaciones batch.

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field

# ------------------------------
# Esquemas para entidades y códigos
# ------------------------------


# Representa un código asociado a una entidad, con información opcional como puntaje y fuente.
class Code(BaseModel):
    system: str  # Sistema de codificación (ej. SNOMED, ICD-10).
    code: str  # Código específico dentro del sistema.
    display: Optional[str] = None  # Descripción legible del código.
    score: Optional[float] = None  # Confianza asociada al código (si aplica).
    source: Optional[str] = None  # Fuente del código (ej. modelo, usuario).


# Representa una entidad extraída de un texto, con información como tipo, ubicación y códigos asociados.
class Entity(BaseModel):
    type: str  # Tipo de la entidad (ej. "diagnóstico", "medicación").
    text: str  # Texto exacto que corresponde a la entidad.
    score: Optional[float] = None  # Confianza en la extracción de la entidad.
    code: Optional[str] = None  # Código principal asociado a la entidad (si aplica).
    start: Optional[int] = None  # Índice de inicio del texto en el documento original.
    end: Optional[int] = None  # Índice de fin del texto en el documento original.
    codes: List[Code] = Field(
        default_factory=list
    )  # Lista de códigos adicionales asociados.


# ------------------------------
# Esquemas para respuestas de extracción
# ------------------------------


# Representa la respuesta de una operación de extracción, incluyendo el texto procesado y las entidades detectadas.
class ExtractResponse(BaseModel):
    text: str  # Texto completo procesado durante la extracción.
    entities: List[Entity] = Field(
        default_factory=list
    )  # Lista de entidades extraídas.
    meta: Dict[str, Any] = Field(
        default_factory=dict
    )  # Metadatos adicionales sobre la extracción.


# Representa un ítem individual en un procesamiento batch, con entidades y metadatos asociados.
class BatchItem(BaseModel):
    filename: str  # Nombre del archivo procesado.
    entities: List[Entity] = Field(
        default_factory=list
    )  # Entidades extraídas del archivo.
    meta: Dict[str, Any] = Field(
        default_factory=dict
    )  # Metadatos adicionales sobre el archivo.


# ------------------------------
# Esquemas para acuses de recibo (ACKs)
# ------------------------------


# Representa un acuse de recibo para una operación de extracción, con información sobre el estado y resultado.
class ExtractAck(BaseModel):
    id: str  # Identificador único del acuse de recibo.
    stored: bool  # Indica si el resultado fue almacenado exitosamente.
    url: Optional[str] = None  # URL donde se almacenó el resultado (si aplica).
    filename: Optional[str] = None  # Nombre del archivo asociado al acuse.
    episode_id: Optional[str] = None  # Identificador del episodio clínico (si aplica).
    note_date: Optional[str] = None  # Fecha de la nota procesada (si aplica).
    entity_count: Optional[int] = None  # Número total de entidades extraídas.
    result: Optional[ExtractResponse] = (
        None  # Resultado completo de la extracción (si aplica).
    )


# Representa un ítem individual en un acuse de recibo batch, con información sobre errores o estado.
class BatchAckItem(BaseModel):
    filename: str  # Nombre del archivo procesado.
    id: Optional[str] = None  # Identificador único del ítem (si aplica).
    stored: bool  # Indica si el ítem fue almacenado exitosamente.
    entity_count: Optional[int] = None  # Número de entidades extraídas en el archivo.
    url: Optional[str] = None  # URL donde se almacenó el resultado (si aplica).
    error: Optional[str] = None  # Mensaje de error en caso de fallo.


# Representa la respuesta completa de un acuse de recibo batch, con una lista de ítems procesados.
class BatchAckResponse(BaseModel):
    items: List[BatchAckItem]  # Lista de ítems procesados en el batch.
