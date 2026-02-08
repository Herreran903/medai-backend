"""
Pydantic Schema Definitions for MedAI Backend API.

This module defines the data transfer objects (DTOs) used throughout the MedAI
backend for request validation, response serialization, and OpenAPI documentation.

Architecture Context:
    MedAI processes clinical notes to extract medical entities using NLP models.
    The schemas in this module represent:

    - **Domain Objects**: Clinical entities and their normalized values
    - **API Responses**: Extraction results and acknowledgments
    - **Batch Operations**: Multi-file processing results

Schema Hierarchy:
    .. code-block:: text

        Entity
          └── ExtractResponse (contains List[Entity])
                └── ExtractAck (contains Optional[ExtractResponse])

        BatchAckItem
          └── BatchAckResponse (contains List[BatchAckItem])

Domain Model:
    The entity extraction domain follows this conceptual model:

    - **Entity**: A span of text identified as a clinical concept (e.g., diagnosis,
      medication, vital sign) with optional character offsets and a normalized value.

Usage:
    These schemas are used in FastAPI route definitions for automatic
    request validation and OpenAPI schema generation::

        @router.post("/extract", response_model=ExtractAck)
        async def extract(...) -> ExtractAck:
            ...

See Also:
    - :mod:`app.routers.extract` for API endpoint usage
"""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

# ==============================================================================
# Entity Schemas
# ==============================================================================


class Entity(BaseModel):
    """
    Clinical entity extracted from medical text.

    Represents a named entity identified by the NER models, including its
    type classification, text span, optional normalized value, and offsets.

    Entity Types:
        MedAI recognizes entities specific to mechanical ventilation clinical notes:

        **Ventilation Configuration:**
            - ``MODO``: Ventilator mode (AC/VC, PC, SIMV, PSV, CPAP)
            - ``FIO2``: Fraction of inspired oxygen (%)
            - ``PEEP``: Positive end-expiratory pressure (cmH2O)
            - ``FR``: Respiratory rate (breaths/min)
            - ``VT``: Tidal volume (mL)
            - ``FLUJO``: Inspiratory flow (L/min)
            - ``I_E``: Inspiration:expiration ratio
            - ``SENS``: Trigger sensitivity

        **Ventilation Response:**
            - ``SAO2``: Oxygen saturation (%)
            - ``PP``: Peak airway pressure (cmH2O)
            - ``PMES``: Plateau pressure (cmH2O)
            - ``PM``: Mechanical power

        **Anthropometrics:**
            - ``EDAD``: Patient age (years)
            - ``PESO``: Body weight (kg)
            - ``TALLA``: Height (m)

        **Vital Signs:**
            - ``TEMP``: Body temperature (°C)
            - ``PA``: Blood pressure (mmHg)
            - ``FC``: Heart rate (bpm)
            - ``GLICEMIA``: Blood glucose (mg/dL)

        **Arterial Blood Gases:**
            - ``PH``: Arterial pH
            - ``PACO2``: Partial pressure of CO2 (mmHg)
            - ``PAO2``: Partial pressure of O2 (mmHg)
            - ``PAFI``: PaO2/FiO2 ratio

        **Clinical:**
            - ``DX``: Diagnosis (normalizable to SNOMED-CT/ICD-10)

    Attributes:
        type: Entity type classification from the NER model.
        text: Exact text span from the source document.
        code: Normalized value extracted from the entity text.
        start: Character offset where entity begins in source text.
        end: Character offset where entity ends in source text.

    Example:
        >>> entity = Entity(
        ...     type="FIO2",
        ...     text="FiO2 60%",
        ...     code="60",
        ...     start=13,
        ...     end=21
        ... )
    """

    type: str = Field(
        ...,
        description="Entity type classification (e.g., DX, FIO2, PEEP, TEMP).",
        json_schema_extra={"example": "DX"},
    )
    text: str = Field(
        ...,
        description="Exact text span extracted from the source document.",
        json_schema_extra={"example": "neumonía adquirida en comunidad"},
    )
    code: Optional[str] = Field(
        default=None,
        description="Normalized value extracted from the entity text.",
        json_schema_extra={"example": "40"},
    )
    start: Optional[int] = Field(
        default=None,
        ge=0,
        description="Character offset where entity begins in source text.",
        json_schema_extra={"example": 45},
    )
    end: Optional[int] = Field(
        default=None,
        ge=0,
        description="Character offset where entity ends in source text.",
        json_schema_extra={"example": 76},
    )


# ==============================================================================
# Extraction Response Schemas
# ==============================================================================


class ExtractResponse(BaseModel):
    """
    Complete extraction result from a clinical note.

    Contains the processed text, all extracted entities, and metadata
    about the extraction process. This is the primary response format
    for the ``GET /notes/{note_id}`` endpoint.

    Attributes:
        text: The complete source text that was processed.
        entities: List of extracted clinical entities with their normalized values.
        meta: Extraction metadata including model info and statistics.

    Metadata Fields:
        The ``meta`` dictionary contains:

        - ``model``: Identifier of the model used (e.g., "lstm", "transformer", "llm")
        - ``inference_time_ms``: Inference time in milliseconds
        - ``entity_count``: Number of entities extracted

    Example:
        >>> response = ExtractResponse(
        ...     text="Paciente con FiO2 60%, PEEP 8 cmH2O",
        ...     entities=[
        ...         Entity(type="FIO2", text="FiO2 60%", code="60", start=14, end=22),
        ...         Entity(type="PEEP", text="PEEP 8 cmH2O", code="8", start=24, end=37),
        ...     ],
        ...     meta={"model": "transformer", "inference_time_ms": 120.5, "entity_count": 2}
        ... )
    """

    text: str = Field(
        ...,
        description="Complete source text that was processed for entity extraction.",
        json_schema_extra={"example": "Paciente con FiO2 60%, PEEP 8 cmH2O"},
    )
    entities: List[Entity] = Field(
        default_factory=list,
        description="List of clinical entities extracted from the text.",
    )
    meta: Dict[str, Any] = Field(
        default_factory=dict,
        description="Extraction metadata including model info and statistics.",
        json_schema_extra={"example": {"model": "transformer", "inference_time_ms": 120.5, "entity_count": 2}},
    )


# ==============================================================================
# Acknowledgment Schemas
# ==============================================================================


class ExtractAck(BaseModel):
    """
    Acknowledgment response for single extraction requests.

    Returned by the ``POST /extract`` endpoint to confirm processing
    and provide access information for the stored result.

    This schema provides:

    - Confirmation of successful processing
    - Storage status and location
    - Summary statistics
    - Optional full result (when ``expand=true``)

    Attributes:
        id: Unique identifier for the stored note (UUID format).
        stored: Whether the result was persisted to MongoDB.
        url: Relative URL to retrieve the full extraction result.
        filename: Original filename if a file was uploaded.
        episode_id: Clinical episode identifier for grouping notes.
        note_date: ISO 8601 date of the clinical note.
        entity_count: Number of entities extracted.
        result: Full extraction result (only when expand=true).

    Example:
        >>> ack = ExtractAck(
        ...     id="550e8400-e29b-41d4-a716-446655440000",
        ...     stored=True,
        ...     url="/notes/550e8400-e29b-41d4-a716-446655440000",
        ...     episode_id="EP-2024-001",
        ...     note_date="2024-01-15T10:30:00",
        ...     entity_count=12
        ... )
    """

    id: str = Field(
        ...,
        description="Unique identifier for the stored note (UUID format).",
        json_schema_extra={"example": "550e8400-e29b-41d4-a716-446655440000"},
    )
    stored: bool = Field(
        ...,
        description="Whether the extraction result was persisted to storage.",
        json_schema_extra={"example": True},
    )
    url: Optional[str] = Field(
        default=None,
        description="Relative URL to retrieve the full extraction result.",
        json_schema_extra={"example": "/notes/550e8400-e29b-41d4-a716-446655440000"},
    )
    filename: Optional[str] = Field(
        default=None,
        description="Original filename if a file was uploaded.",
        json_schema_extra={"example": "nota_clinica.pdf"},
    )
    episode_id: Optional[str] = Field(
        default=None,
        description="Clinical episode identifier for grouping related notes.",
        json_schema_extra={"example": "EP-2024-001"},
    )
    note_date: Optional[str] = Field(
        default=None,
        description="ISO 8601 formatted date of the clinical note.",
        json_schema_extra={"example": "2024-01-15T10:30:00"},
    )
    entity_count: Optional[int] = Field(
        default=None,
        ge=0,
        description="Total number of entities extracted from the note.",
        json_schema_extra={"example": 12},
    )
    result: Optional[ExtractResponse] = Field(
        default=None,
        description="Full extraction result (included when expand=true parameter is set).",
    )


class BatchAckItem(BaseModel):
    """
    Acknowledgment for a single file in batch processing.

    Represents the processing status for one file in a batch extraction
    request, including success/failure status and error details.

    Attributes:
        filename: Original filename of the processed document.
        id: Unique note identifier if successfully stored.
        stored: Whether this file's result was persisted.
        entity_count: Number of entities extracted (if successful).
        url: Relative URL to retrieve the result (if stored).
        error: Error message if processing failed.

    Example:
        >>> # Successful item
        >>> item = BatchAckItem(
        ...     filename="nota_001.pdf",
        ...     id="uuid-here",
        ...     stored=True,
        ...     entity_count=8,
        ...     url="/notes/uuid-here"
        ... )
        >>> # Failed item
        >>> item = BatchAckItem(
        ...     filename="nota_002.pdf",
        ...     stored=False,
        ...     error="Missing episode_id for 'nota_002.pdf'"
        ... )
    """

    filename: str = Field(
        ...,
        description="Original filename of the processed document.",
        json_schema_extra={"example": "nota_clinica_001.pdf"},
    )
    id: Optional[str] = Field(
        default=None,
        description="Unique note identifier if successfully stored.",
        json_schema_extra={"example": "550e8400-e29b-41d4-a716-446655440000"},
    )
    stored: bool = Field(
        ...,
        description="Whether this file's extraction result was persisted.",
        json_schema_extra={"example": True},
    )
    entity_count: Optional[int] = Field(
        default=None,
        ge=0,
        description="Number of entities extracted from this file.",
        json_schema_extra={"example": 8},
    )
    url: Optional[str] = Field(
        default=None,
        description="Relative URL to retrieve the extraction result.",
        json_schema_extra={"example": "/notes/550e8400-e29b-41d4-a716-446655440000"},
    )
    error: Optional[str] = Field(
        default=None,
        description="Error message if processing failed for this file.",
        json_schema_extra={"example": "Missing episode_id for 'nota_002.pdf'"},
    )


class BatchAckResponse(BaseModel):
    """
    Complete response for batch extraction requests.

    Returned by the ``POST /extract-batch`` endpoint with processing
    status for all files in the batch.

    Attributes:
        items: List of acknowledgments for each processed file.

    Example:
        >>> response = BatchAckResponse(
        ...     items=[
        ...         BatchAckItem(filename="nota_001.pdf", stored=True, entity_count=8),
        ...         BatchAckItem(filename="nota_002.pdf", stored=False, error="Invalid format"),
        ...     ]
        ... )
    """

    items: List[BatchAckItem] = Field(
        ...,
        description="List of processing acknowledgments for each file in the batch.",
    )
