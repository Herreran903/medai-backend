"""
Clinical Entity Extraction API Router.

This module defines the FastAPI router for clinical Named Entity Recognition (NER)
endpoints, providing the primary API interface for the MedAI backend.

Architecture Context:
    The extraction router is the main entry point for clinical NLP operations:

    - Single note extraction (``POST /extract``)
    - Batch file processing (``POST /extract-batch``)
    - Result retrieval (``GET /notes/{note_id}``)

    All endpoints integrate with the extraction pipeline, storage layer,
    and optional UMLS normalization service.

API Design:
    The API follows REST conventions with multipart/form-data for file uploads:

    - Form fields for parameters (model, episode_id, note_date, etc.)
    - File upload for document processing
    - JSON responses with extraction results or acknowledgments

Supported Models:
    - ``lstm``: BiLSTM-CRF model for fast inference
    - ``transformer``: Fine-tuned BETO/RoBERTa (variants: beto, roberta)
    - ``llm``: LLM-based extraction (variants: claude, gpt)

Integration Points:
    - :mod:`app.services.pipeline`: Extraction orchestration
    - :mod:`app.services.store`: MongoDB persistence
    - :mod:`app.services.text_utils`: File format conversion
    - :mod:`app.deps`: Database and settings injection

Usage:
    The router is mounted in :mod:`app.main` and exposes endpoints at:

    - ``POST /extract`` - Single note extraction
    - ``POST /extract-batch`` - Batch processing
    - ``GET /notes/{note_id}`` - Retrieve stored result

See Also:
    - :mod:`app.schemas` for request/response models
    - :mod:`app.services.pipeline` for extraction logic
"""

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
from app.services.store import save_result
from app.services.text_utils import read_any_to_text

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


def _extract_from_text(*args, **kwargs):
    """
    Lazy import wrapper for extraction pipeline.

    Defers import of the pipeline module to avoid circular dependencies
    and reduce startup time by delaying model loading.

    Returns:
        Result from :func:`app.services.pipeline.extract_from_text`.
    """
    from app.services.pipeline import extract_from_text

    return extract_from_text(*args, **kwargs)


def _parse_iso8601(dt: Optional[str]) -> Optional[datetime]:
    """
    Parse ISO 8601 date string to datetime object.

    Handles various ISO 8601 formats including:

    - Full datetime: ``2024-01-15T10:30:00``
    - With timezone: ``2024-01-15T10:30:00Z`` or ``2024-01-15T10:30:00+00:00``
    - Date only: ``2024-01-15`` (assumes midnight)

    Args:
        dt: ISO 8601 formatted date string, or None.

    Returns:
        Parsed datetime object, or None if input is None/empty.

    Raises:
        HTTPException: 400 error if date format is invalid.
    """
    if not dt:
        return None
    s = dt.strip()
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(s)
    except ValueError:
        try:
            return datetime.fromisoformat(s + "T00:00:00")
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid note_date format")


def _parse_csv(v: Optional[str]) -> List[str]:
    """
    Parse comma-separated string to list.

    Args:
        v: Comma-separated string (e.g., "SNOMEDCT_US,ICD10CM").

    Returns:
        List of trimmed non-empty strings.

    Example:
        >>> _parse_csv("SNOMEDCT_US, ICD10CM")
        ['SNOMEDCT_US', 'ICD10CM']
    """
    if not v:
        return []
    return [p.strip() for p in v.split(",") if p.strip()]


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
- `transformer`: Best accuracy (variants: `beto`, `roberta`)
- `llm`: Highest flexibility (variants: `claude`, `gpt`)

**Normalization:**
When `normalize=true`, extracted diagnosis entities (DX) are linked to
SNOMED-CT and ICD-10 codes via UMLS lookup.
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
        description="Model variant: `beto`/`roberta` for transformer, `claude`/`gpt` for llm.",
    ),
    episode_id: Optional[str] = Form(
        None,
        description="Clinical episode identifier for grouping related notes. Required for storage.",
    ),
    note_date: Optional[str] = Form(
        None,
        description="ISO 8601 date of the clinical note (e.g., `2024-01-15T10:30:00`). Required for storage.",
    ),
    save: Optional[bool] = Form(
        True,
        description="Whether to persist extraction results to MongoDB.",
    ),
    normalize: Optional[bool] = Form(
        False,
        description="Whether to normalize DX entities to SNOMED-CT/ICD-10 codes via UMLS.",
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
    db: Database = Depends(get_db),
    settings: Settings = Depends(settings_dep),
):
    """
    Extract clinical entities from a single note.

    This endpoint accepts either raw text or an uploaded file, processes it
    through the specified NER model, optionally normalizes entities to
    standard medical codes, and persists the result to MongoDB.

    Args:
        text: Raw clinical note text.
        file: Uploaded file (PDF, DOCX, TXT).
        model: Extraction model (lstm, transformer, llm).
        model_variant: Model variant for transformer/llm.
        episode_id: Clinical episode identifier (required).
        note_date: Note date in ISO 8601 format (required).
        save: Whether to persist results.
        normalize: Whether to apply UMLS normalization.
        systems_csv: Target coding systems for normalization.
        restrict_types_csv: Entity types to normalize.
        expand: Include full result in response.
        db: MongoDB database (injected).
        settings: Application settings (injected).

    Returns:
        ExtractAck: Acknowledgment with note ID and metadata.

    Raises:
        HTTPException: 400 if neither text nor file provided, or missing required fields.
    """
    if not text and not file:
        raise HTTPException(status_code=400, detail="Provide either 'text' or 'file'")

    if file:
        content = await file.read()
        text = read_any_to_text(file.filename, content)

    systems = _parse_csv(systems_csv)
    restrict_types = _parse_csv(restrict_types_csv)

    note_dt = _parse_iso8601(note_date)
    note_date_iso = note_dt.isoformat() if note_dt else None

    res = _extract_from_text(
        text or "",
        model=model or settings.default_model,
        model_variant=model_variant,
        normalize=bool(normalize),
        systems=systems,
        restrict_types=restrict_types or None,
    )

    if not episode_id:
        raise HTTPException(status_code=400, detail="Missing 'episode_id'")
    if not note_date_iso:
        raise HTTPException(
            status_code=400, detail="Missing 'note_date' or invalid format"
        )

    note_id = None
    stored = False
    if save and settings.save_results:
        note_id = save_result(
            db=db,
            payload=text or "",
            result=res,
            model=model,
            episode_id=episode_id,
            note_date_iso=note_date_iso,
            filename=getattr(file, "filename", None),
            source_system="api.extract",
            dedupe_by_hash=True,
        )
        stored = True

    ack = ExtractAck(
        id=note_id or "",
        stored=stored,
        url=(f"/notes/{note_id}" if note_id else None),
        filename=getattr(file, "filename", None),
        episode_id=episode_id,
        note_date=note_date_iso,
        entity_count=len(res.entities) if hasattr(res, "entities") else None,
        result=(res if expand else None),
    )
    return ack


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
        description="Model variant: `beto`/`roberta` for transformer, `claude`/`gpt` for llm.",
    ),
    save: Optional[bool] = Form(
        True,
        description="Whether to persist extraction results to MongoDB.",
    ),
    normalize: Optional[bool] = Form(
        False,
        description="Whether to normalize DX entities to SNOMED-CT/ICD-10 codes.",
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
        description="JSON array with per-file metadata (filename, episode_id, note_date).",
    ),
    db: Database = Depends(get_db),
    settings: Settings = Depends(settings_dep),
):
    """
    Process multiple files for entity extraction.

    Each file is processed independently through the extraction pipeline.
    Metadata (episode_id, note_date) must be provided via the notes_meta
    JSON parameter for each file.

    Args:
        files: List of uploaded files.
        model: Extraction model identifier.
        model_variant: Model variant for transformer/llm.
        save: Whether to persist results.
        normalize: Whether to apply UMLS normalization.
        systems_csv: Target coding systems.
        restrict_types_csv: Entity types to normalize.
        notes_meta: JSON array with per-file metadata.
        db: MongoDB database (injected).
        settings: Application settings (injected).

    Returns:
        BatchAckResponse: List of per-file acknowledgments.

    Note:
        Files without matching metadata in notes_meta will fail with an error.
    """
    systems = _parse_csv(systems_csv)
    restrict_types = _parse_csv(restrict_types_csv)

    # Parse per-file metadata
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
            meta_by_filename = {}

    items: List[BatchAckItem] = []
    for f in files:
        try:
            content = await f.read()
            text = read_any_to_text(f.filename, content)

            meta = meta_by_filename.get(f.filename, {})
            episode_id = meta.get("episode_id")
            note_date_raw = meta.get("note_date")
            note_dt = _parse_iso8601(note_date_raw) if note_date_raw else None
            note_date_iso = note_dt.isoformat() if note_dt else None

            if not episode_id:
                raise ValueError(f"Missing 'episode_id' for '{f.filename}'")
            if not note_date_iso:
                raise ValueError(
                    f"Missing 'note_date' or invalid format for '{f.filename}'"
                )

            res = _extract_from_text(
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
            items.append(
                BatchAckItem(
                    filename=f.filename,
                    stored=False,
                    error=str(e),
                )
            )
    return BatchAckResponse(items=items)


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
    db: Database = Depends(get_db),
):
    """
    Retrieve a stored extraction result.

    Queries MongoDB for the note with the specified ID and returns
    the complete extraction result including text, entities, and metadata.

    Args:
        note_id: UUID of the note to retrieve.
        db: MongoDB database (injected).

    Returns:
        ExtractResponse: Complete extraction result.

    Raises:
        HTTPException: 404 if note not found.
    """
    doc = db.episodes.find_one(
        {"notes.note_id": note_id},
        {
            "_id": 0,
            "notes": {"$elemMatch": {"note_id": note_id}},
        },
    )

    if not doc or not doc.get("notes"):
        raise HTTPException(status_code=404, detail="Note not found")

    note = doc["notes"][0]
    return ExtractResponse(
        text=note.get("text") or "",
        entities=[Entity(**e) for e in note.get("entities", [])],
        meta=note.get("meta") or {},
    )
