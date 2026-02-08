"""
Clinical Entity Extraction API Router.

This module defines the FastAPI router for clinical Named Entity Recognition (NER)
endpoints, providing the primary API interface for the MedAI backend.

Architecture Context:
    The extraction router serves as the thin API layer that delegates to the
    ExtractionService for all business logic:

    - Single note extraction (``POST /extract``)
    - Batch file processing (``POST /extract-batch``)
    - Result retrieval (``GET /notes/{note_id}``)

    Following the Service Layer pattern from fastapi-templates skill, routers
    are kept minimal and only handle:
    - Request validation (via Pydantic)
    - Dependency injection
    - Delegation to service layer
    - Response formatting (handled by Pydantic)

API Design:
    The API follows REST conventions with multipart/form-data for file uploads:

    - Form fields for parameters (model, episode_id, note_date, etc.)
    - File upload for document processing
    - JSON responses with extraction results or acknowledgments

Supported Models:
    - ``lstm``: BiLSTM-CRF model for fast inference
    - ``transformer``: Fine-tuned RoBERTa (FIXED: only roberta)
    - ``llm``: LLM-based extraction (variants: claude, gpt)

Integration Points:
    - :mod:`app.services.extraction_service`: Business logic orchestration
    - :mod:`app.deps`: Dependency injection for service layer

Usage:
    The router is mounted in :mod:`app.main` and exposes endpoints at:

    - ``POST /extract`` - Single note extraction
    - ``POST /extract-batch`` - Batch processing
    - ``GET /notes/{note_id}`` - Retrieve stored result

See Also:
    - :mod:`app.schemas` for request/response models
    - :mod:`app.services.extraction_service` for business logic
"""

from typing import List, Optional

from fastapi import (
    APIRouter,
    Depends,
    File,
    Form,
    Path,
    UploadFile,
    status,
)

from app.deps import get_extraction_service
from app.schemas import (
    BatchAckResponse,
    ExtractAck,
    ExtractResponse,
)
from app.services.extraction_service import ExtractionService

router = APIRouter(
    tags=["Extraction"],
    responses={
        400: {"description": "Invalid request parameters or missing required fields"},
        404: {"description": "Requested resource not found"},
        500: {"description": "Internal server error during extraction"},
    },
)
"""
FastAPI router for extraction endpoints.

All endpoints are tagged with "Extraction" for OpenAPI grouping.
Common error responses are defined at the router level.
"""


@router.post(
    "/extract",
    response_model=ExtractAck,
    status_code=status.HTTP_201_CREATED,
    summary="Extract clinical entities from text or file",
    description="""
Extract clinical Named Entities from a single clinical note using the specified NER model.

**Business Purpose:**
This endpoint processes individual clinical notes (text or uploaded files) to identify
and extract medical entities such as diagnoses, vital signs, ventilation parameters,
and laboratory values. Results are persisted to MongoDB for later retrieval.

**Usage Context:**
- Frontend single-note analysis workflow
- Real-time clinical decision support integration
- Manual note processing by clinical staff

**Supported File Formats:**
- Plain text (.txt)
- PDF documents (.pdf)
- Word documents (.docx)

**Model Selection:**
- `lstm`: Fast inference, moderate accuracy
- `transformer`: Best accuracy (FIXED: `roberta` only)
- `llm`: Highest flexibility (variants: `claude`, `gpt`)

**Normalization:**
Normalization is currently disabled in the gateway; the `normalize` flag is
accepted but ignored.
""",
    response_description="Acknowledgment with note ID and optional full result",
)
async def extract(
    text: Optional[str] = Form(
        None,
        description="Clinical note text content. Either `text` or `file` must be provided.",
    ),
    file: Optional[UploadFile] = File(
        None,
        description="Clinical note file (PDF, DOCX, or TXT). Either `text` or `file` must be provided.",
    ),
    model: str = Form(
        ...,
        description="Extraction model identifier: `lstm`, `transformer`, or `llm`.",
    ),
    model_variant: Optional[str] = Form(
        None,
        description="Model variant: `roberta` for transformer (fixed), `gpt` for llm (fixed).",
    ),
    episode_id: str = Form(
        ...,
        description="Clinical episode identifier for grouping related notes. Required (API returns 400 if missing).",
    ),
    note_date: str = Form(
        ...,
        description="ISO 8601 date of the clinical note (e.g., `2024-01-15T10:30:00`). Required (API returns 400 if missing).",
    ),
    save: Optional[bool] = Form(
        True,
        description="Whether to persist extraction results to MongoDB.",
    ),
    normalize: Optional[bool] = Form(
        False,
        description="Whether to normalize DX entities (currently ignored by gateway).",
    ),
    systems_csv: Optional[str] = Form(
        None,
        description="Comma-separated target coding systems for normalization (e.g., `SNOMEDCT_US,ICD10CM`).",
    ),
    restrict_types_csv: Optional[str] = Form(
        None,
        description="Comma-separated entity types to include in normalization (e.g., `DX`).",
    ),
    expand: Optional[bool] = Form(
        False,
        description="Whether to include full extraction result in response.",
    ),
    service: ExtractionService = Depends(get_extraction_service),
):
    """
    Extract clinical entities from a single note.

    This endpoint delegates all business logic to the ExtractionService,
    following the Service Layer pattern. The route handler only handles
    request validation and service coordination.

    Args:
        text: Raw clinical note text.
        file: Uploaded file (PDF, DOCX, TXT).
        model: Extraction model (lstm, transformer, llm).
        model_variant: Model variant for transformer/llm.
        episode_id: Clinical episode identifier (required).
        note_date: Note date in ISO 8601 format (required).
        save: Whether to persist results.
        normalize: Whether to apply UMLS normalization (currently ignored by gateway).
        systems_csv: Target coding systems for normalization.
        restrict_types_csv: Entity types to normalize.
        expand: Include full result in response.
        service: ExtractionService instance (injected).

    Returns:
        ExtractAck: Acknowledgment with note ID and metadata.

    Raises:
        HTTPException: 400 if validation fails or required fields missing.
    """
    return await service.extract_single(
        text=text,
        file=file,
        model=model,
        model_variant=model_variant,
        episode_id=episode_id or "",
        note_date=note_date or "",
        save=save or True,
        normalize=normalize or False,
        systems_csv=systems_csv,
        restrict_types_csv=restrict_types_csv,
        expand=expand or False,
    )


@router.post(
    "/extract-batch",
    response_model=BatchAckResponse,
    summary="Extract entities from multiple files",
    description="""
Process multiple clinical note files in a single batch request.

**Business Purpose:**
Enables bulk processing of clinical notes for retrospective analysis,
data migration, or batch import workflows. Each file is processed
independently with individual success/failure tracking.

**Usage Context:**
- Bulk import of historical clinical notes
- Batch processing jobs from external systems
- Data migration and ETL pipelines

**Metadata Format:**
The `notes_meta` parameter accepts a JSON array mapping filenames to metadata:

```json
[
    {"filename": "nota_001.pdf", "episode_id": "EP-001", "note_date": "2024-01-15"},
    {"filename": "nota_002.pdf", "episode_id": "EP-002", "note_date": "2024-01-16"}
]
```

**Error Handling:**
Files that fail processing are included in the response with error details.
Successfully processed files are stored independently.
""",
    response_description="Batch acknowledgment with per-file status",
)
async def extract_batch(
    files: List[UploadFile] = File(
        ...,
        description="List of clinical note files to process (PDF, DOCX, TXT).",
    ),
    model: str = Form(
        ...,
        description="Extraction model identifier: `lstm`, `transformer`, or `llm`.",
    ),
    model_variant: Optional[str] = Form(
        None,
        description="Model variant: `roberta` for transformer (fixed), `gpt` for llm (fixed).",
    ),
    save: Optional[bool] = Form(
        True,
        description="Whether to persist extraction results to MongoDB.",
    ),
    normalize: Optional[bool] = Form(
        False,
        description="Whether to normalize DX entities (currently ignored by gateway).",
    ),
    systems_csv: Optional[str] = Form(
        None,
        description="Comma-separated target coding systems for normalization.",
    ),
    restrict_types_csv: Optional[str] = Form(
        None,
        description="Comma-separated entity types to include in normalization.",
    ),
    notes_meta: Optional[str] = Form(
        None,
        description="JSON array with per-file metadata (filename, episode_id, note_date). Required; files without metadata will fail.",
    ),
    service: ExtractionService = Depends(get_extraction_service),
):
    """
    Process multiple files for entity extraction.

    This endpoint delegates all batch processing logic to the ExtractionService.
    Each file is processed independently with individual error handling.

    Args:
        files: List of uploaded files.
        model: Extraction model identifier.
        model_variant: Model variant for transformer/llm.
        save: Whether to persist results.
        normalize: Whether to apply UMLS normalization (currently ignored by gateway).
        systems_csv: Target coding systems.
        restrict_types_csv: Entity types to normalize.
        notes_meta: JSON array with per-file metadata (required).
        service: ExtractionService instance (injected).

    Returns:
        BatchAckResponse: List of per-file acknowledgments.

    Note:
        Files without matching metadata in notes_meta will fail with an error.
    """
    return await service.extract_batch(
        files=files,
        model=model,
        model_variant=model_variant,
        save=save or True,
        normalize=normalize or False,
        systems_csv=systems_csv,
        restrict_types_csv=restrict_types_csv,
        notes_meta_json=notes_meta,
    )


@router.get(
    "/notes/{note_id}",
    response_model=ExtractResponse,
    summary="Retrieve stored extraction result",
    description="""
Retrieve a previously stored extraction result by its unique note identifier.

**Business Purpose:**
Enables retrieval of extraction results for display in the frontend,
integration with downstream systems, or audit purposes.

**Usage Context:**
- Frontend note detail view
- API integration for external systems
- Audit and compliance verification

**Response Content:**
Returns the complete extraction result including:
- Original text content
- All extracted entities with types and spans
- Extraction metadata (model, normalization status)
""",
    response_description="Complete extraction result with text, entities, and metadata",
    responses={
        404: {"description": "Note not found with the specified ID"},
    },
)
async def get_note(
    note_id: str = Path(
        ...,
        description="Unique note identifier (UUID format) returned from extraction endpoints.",
    ),
    service: ExtractionService = Depends(get_extraction_service),
):
    """
    Retrieve a stored extraction result.

    This endpoint delegates note retrieval to the ExtractionService,
    which uses the repository layer for database access.

    Args:
        note_id: UUID of the note to retrieve.
        service: ExtractionService instance (injected).

    Returns:
        ExtractResponse: Complete extraction result.

    Raises:
        HTTPException: 404 if note not found.
    """
    return await service.get_note(note_id)
