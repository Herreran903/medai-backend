"""
MongoDB Storage Service for Extraction Results.

This module provides persistence functionality for clinical entity extraction
results, implementing content-based deduplication and episode-based organization.

Architecture Context:
    The storage service is responsible for persisting extraction results to
    MongoDB. It implements a document model where:

    - **Episodes** are top-level documents representing clinical encounters
    - **Notes** are embedded documents within episodes containing extraction results
    - **Deduplication** prevents duplicate notes based on content hash

Data Model:
    .. code-block:: text

        episodes (collection)
        ├── _id: str (episode_id)
        ├── created_at: datetime
        ├── updated_at: datetime
        └── notes: array
            ├── note_id: str (UUID)
            ├── filename: str (optional)
            ├── source_system: str
            ├── text: str
            ├── entities: array[Entity]
            ├── meta: dict
            ├── model: str
            ├── note_date: str (ISO 8601)
            ├── created_at: datetime
            └── content_hash: str (SHA-256)

Deduplication Strategy:
    When ``dedupe_by_hash=True`` (default), the service:

    1. Computes SHA-256 hash of the note content
    2. Checks if a note with the same hash exists in the episode
    3. Returns existing note_id if duplicate found
    4. Creates new note only if no duplicate exists

    This prevents duplicate processing of the same clinical note content.

Usage:
    >>> from app.services.store import save_result
    >>> note_id = save_result(
    ...     db=db,
    ...     payload="Clinical note text...",
    ...     result=extraction_result,
    ...     model="transformer",
    ...     episode_id="EP-2024-001",
    ...     note_date_iso="2024-01-15T10:30:00"
    ... )

See Also:
    - :mod:`app.routers.extract` for API integration
    - :mod:`app.indexes` for database index definitions
    - :mod:`app.deps` for database connection management
"""

import hashlib
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from pymongo.database import Database


def _sha256(s: str) -> str:
    """
    Compute SHA-256 hash of a string.

    Used for content-based deduplication of clinical notes.

    Args:
        s: Input string to hash.

    Returns:
        Hexadecimal string representation of SHA-256 hash.

    Example:
        >>> _sha256("Hello, World!")
        'dffd6021bb2bd5b0af676290809ec3a53191dd81c7f70a4b28688a362182986f'
    """
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


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
    """
    Save extraction result to MongoDB.

    Persists the extraction result as a note document within an episode,
    with optional content-based deduplication.

    Args:
        db: MongoDB database instance from :func:`app.deps.get_db`.
        payload: Original clinical note text content.
        result: Extraction result object with ``entities`` and ``meta`` attributes.
            Typically an :class:`app.schemas.ExtractResponse` instance.
        model: Extraction model identifier used (e.g., "transformer", "lstm").
        episode_id: Clinical episode identifier for grouping related notes.
            Used as the document ``_id`` in the episodes collection.
        note_date_iso: ISO 8601 formatted date of the clinical note.
            Represents when the note was created clinically (not when processed).
        filename: Original filename if the note was uploaded as a file.
        source_system: Identifier of the system that submitted the note
            (e.g., "api.extract", "api.extract-batch").
        dedupe_by_hash: Whether to check for duplicate content before inserting.
            When True, returns existing note_id if content hash matches.

    Returns:
        str: UUID of the stored note. If deduplication found an existing note,
            returns the existing note's ID.

    Document Structure:
        The note document contains:

        - ``note_id``: Unique identifier (UUID v4)
        - ``filename``: Original filename (if applicable)
        - ``source_system``: Source system identifier
        - ``text``: Original note content
        - ``entities``: List of extracted entities (serialized)
        - ``meta``: Extraction metadata
        - ``model``: Model used for extraction
        - ``note_date``: Clinical note date (ISO 8601)
        - ``created_at``: Processing timestamp (UTC)
        - ``content_hash``: SHA-256 hash for deduplication

    Deduplication Behavior:
        When ``dedupe_by_hash=True``:

        1. Compute SHA-256 hash of ``payload``
        2. Query for existing note with same hash in episode
        3. If found, return existing ``note_id`` without inserting
        4. If not found, insert new note document

    Upsert Behavior:
        The function uses MongoDB upsert to handle both:

        - New episodes: Creates document with ``_id=episode_id``
        - Existing episodes: Appends note to ``notes`` array

    Example:
        >>> from pymongo import MongoClient
        >>> client = MongoClient("mongodb://localhost:27017")
        >>> db = client["medai"]
        >>>
        >>> # Save extraction result
        >>> note_id = save_result(
        ...     db=db,
        ...     payload="Paciente con FiO2 60%",
        ...     result=ExtractResponse(text="...", entities=[...], meta={}),
        ...     model="transformer",
        ...     episode_id="EP-001",
        ...     note_date_iso="2024-01-15T10:30:00",
        ...     source_system="api.extract"
        ... )
        >>> print(note_id)
        '550e8400-e29b-41d4-a716-446655440000'

        >>> # Duplicate content returns same ID
        >>> note_id_2 = save_result(
        ...     db=db,
        ...     payload="Paciente con FiO2 60%",  # Same content
        ...     result=ExtractResponse(...),
        ...     model="transformer",
        ...     episode_id="EP-001",
        ...     note_date_iso="2024-01-15T10:30:00"
        ... )
        >>> note_id == note_id_2
        True

    Note:
        The function assumes the ``episodes`` collection exists and has
        appropriate indexes. See :mod:`app.indexes` for index definitions.
    """
    created_at = datetime.now(timezone.utc)

    # Compute content hash for deduplication
    content_hash = _sha256(payload or "")

    # Generate unique note identifier
    note_id = str(uuid.uuid4())

    # Build note document
    note: Dict[str, Any] = {
        "note_id": note_id,
        "filename": filename,
        "source_system": source_system,
        "text": payload,
        "entities": [e.model_dump() for e in getattr(result, "entities", [])],
        "meta": getattr(result, "meta", None),
        "model": model,
        "note_date": note_date_iso,
        "created_at": created_at,
        "content_hash": content_hash,
    }

    episodes = db.episodes

    if dedupe_by_hash:
        # Check for existing note with same content hash
        existing = episodes.find_one(
            {"_id": episode_id, "notes.content_hash": content_hash},
            {"notes.$": 1},
        )
        if existing and existing.get("notes"):
            return existing["notes"][0]["note_id"]

        # Insert new note (upsert episode if needed)
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
    else:
        # Insert without deduplication check
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
