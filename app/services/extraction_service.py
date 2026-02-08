"""
Extraction Service - Business Logic Layer.

This module implements the Service Layer pattern, encapsulating all business logic
for clinical entity extraction. It coordinates between the pipeline (extraction),
repository (persistence), and external services.

Architecture:
    Following the fastapi-templates skill pattern, this service layer:
    - Keeps business logic separate from API routing
    - Uses repository pattern for data access
    - Provides a clean, testable interface
    - Handles orchestration of multiple operations

The service layer sits between routers and repositories:
    Router -> Service -> Repository/Pipeline
           -> Service -> External APIs

Usage:
    >>> from app.services.extraction_service import ExtractionService
    >>> from app.repositories import EpisodeRepository
    >>>
    >>> service = ExtractionService(repository, settings)
    >>> result = await service.extract_single(
    ...     text="Paciente con FiO2 60%",
    ...     model="transformer",
    ...     episode_id="EP-001",
    ...     note_date="2024-01-15T10:30:00"
    ... )
"""

import logging
import uuid
from datetime import datetime
from typing import Dict, List, Optional

from fastapi import HTTPException, UploadFile, status

from app.config import Settings
from app.repositories import EpisodeRepository
from app.schemas import (
    BatchAckItem,
    BatchAckResponse,
    Entity,
    ExtractAck,
    ExtractResponse,
)
from app.services.text_utils import read_any_to_text

logger = logging.getLogger(__name__)


class ExtractionService:
    """
    Service for clinical entity extraction operations.

    This service encapsulates all business logic for extraction workflows,
    coordinating between extraction pipeline, data persistence, and response
    formatting.

    Attributes:
        repository: Episode repository for data access
        settings: Application settings
    """

    def __init__(self, repository: EpisodeRepository, settings: Settings):
        """
        Initialize extraction service.

        Args:
            repository: Episode repository instance
            settings: Application settings
        """
        self.repository = repository
        self.settings = settings

    @staticmethod
    def _parse_iso8601(dt: Optional[str]) -> Optional[datetime]:
        """
        Parse ISO 8601 date string to datetime object.

        Handles various ISO 8601 formats including:
        - Full datetime: 2024-01-15T10:30:00
        - With timezone: 2024-01-15T10:30:00Z
        - Date only: 2024-01-15

        Args:
            dt: ISO 8601 formatted date string

        Returns:
            Parsed datetime object, or None if input is None

        Raises:
            HTTPException: 400 if date format is invalid
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
                raise HTTPException(
                    status_code=400, detail="Invalid note_date format"
                )

    @staticmethod
    def _parse_csv(v: Optional[str]) -> List[str]:
        """
        Parse comma-separated string to list.

        Args:
            v: Comma-separated string

        Returns:
            List of trimmed non-empty strings
        """
        if not v:
            return []
        return [p.strip() for p in v.split(",") if p.strip()]

    async def _extract_from_text(self, *args, **kwargs):
        """
        Lazy import wrapper for extraction pipeline.

        Defers import to avoid circular dependencies.
        """
        from app.services.pipeline import extract_from_text

        return await extract_from_text(*args, **kwargs)

    async def extract_single(
        self,
        *,
        text: Optional[str] = None,
        file: Optional[UploadFile] = None,
        model: str,
        model_variant: Optional[str] = None,
        episode_id: str,
        note_date: str,
        save: bool = True,
        normalize: bool = False,
        systems_csv: Optional[str] = None,
        restrict_types_csv: Optional[str] = None,
        expand: bool = False,
        filename_override: Optional[str] = None,
    ) -> ExtractAck:
        """
        Extract entities from a single clinical note.

        This method handles the complete extraction workflow:
        1. Validate inputs
        2. Process file if provided
        3. Call extraction pipeline
        4. Save results to database
        5. Return acknowledgment

        Args:
            text: Raw clinical note text
            file: Uploaded file (PDF, DOCX, TXT)
            model: Extraction model identifier
            model_variant: Model variant (roberta for transformer, gpt for llm; both fixed)
            episode_id: Clinical episode identifier
            note_date: Note date in ISO 8601 format
            save: Whether to persist results
            normalize: Whether to apply UMLS normalization (currently disabled)
            systems_csv: Comma-separated target coding systems
            restrict_types_csv: Comma-separated entity types to normalize
            expand: Include full result in response
            filename_override: Override filename (for batch processing)

        Returns:
            ExtractAck with note ID and metadata

        Raises:
            HTTPException: 400 if validation fails or required fields missing
        """
        # Validate input: either text or file must be provided
        if not text and not file:
            raise HTTPException(
                status_code=400, detail="Provide either 'text' or 'file'"
            )

        # Validate required fields for persistence
        if not episode_id:
            raise HTTPException(status_code=400, detail="Missing 'episode_id'")

        # Process file if provided
        actual_filename = filename_override or (file.filename if file else None)
        if file:
            content = await file.read()
            text = read_any_to_text(file.filename, content)

        # Parse and validate date
        note_dt = self._parse_iso8601(note_date)
        if not note_dt:
            raise HTTPException(
                status_code=400, detail="Missing 'note_date' or invalid format"
            )
        note_date_iso = note_dt.isoformat()

        # Parse normalization parameters
        systems = self._parse_csv(systems_csv)
        restrict_types = self._parse_csv(restrict_types_csv)

        # Normalization temporarily disabled to avoid extra model loads
        normalize = False

        # Call extraction pipeline
        logger.info(
            f"Extracting entities: model={model}, variant={model_variant}, "
            f"episode={episode_id}, text_length={len(text or '')}"
        )

        result = await self._extract_from_text(
            text or "",
            model=model or self.settings.default_model,
            model_variant=model_variant,
            normalize=bool(normalize),
            systems=systems,
            restrict_types=restrict_types or None,
        )

        # Persist results if requested
        note_id = None
        stored = False

        if save and self.settings.save_results:
            note_id = await self._save_extraction_result(
                text=text or "",
                result=result,
                model=model,
                episode_id=episode_id,
                note_date_iso=note_date_iso,
                filename=actual_filename,
                source_system="api.extract",
            )
            stored = True
            logger.info(f"Saved extraction result: note_id={note_id}")

        # Build acknowledgment response
        return ExtractAck(
            id=note_id or "",
            stored=stored,
            url=(f"/notes/{note_id}" if note_id else None),
            filename=actual_filename,
            episode_id=episode_id,
            note_date=note_date_iso,
            entity_count=len(result.entities) if hasattr(result, "entities") else None,
            result=(result if expand else None),
        )

    async def extract_batch(
        self,
        *,
        files: List[UploadFile],
        model: str,
        model_variant: Optional[str] = None,
        save: bool = True,
        normalize: bool = False,
        systems_csv: Optional[str] = None,
        restrict_types_csv: Optional[str] = None,
        notes_meta_json: Optional[str] = None,
    ) -> BatchAckResponse:
        """
        Process multiple files for entity extraction.

        Each file is processed independently through extract_single.
        Metadata must be provided via notes_meta_json for each file.

        Args:
            files: List of uploaded files
            model: Extraction model identifier
            model_variant: Model variant
            save: Whether to persist results
            normalize: Whether to apply UMLS normalization
            systems_csv: Target coding systems
            restrict_types_csv: Entity types to normalize
            notes_meta_json: JSON array with per-file metadata

        Returns:
            BatchAckResponse with per-file status

        Note:
            Files without matching metadata will fail with an error
        """
        import json

        # Parse per-file metadata
        meta_by_filename: Dict[str, Dict[str, Optional[str]]] = {}
        if notes_meta_json:
            try:
                raw = json.loads(notes_meta_json)
                if isinstance(raw, list):
                    for m in raw:
                        if isinstance(m, dict):
                            fn = str(m.get("filename") or "").strip()
                            if fn:
                                meta_by_filename[fn] = {
                                    "episode_id": m.get("episode_id") or None,
                                    "note_date": m.get("note_date") or None,
                                }
            except Exception as e:
                logger.warning(f"Failed to parse notes_meta JSON: {e}")
                meta_by_filename = {}

        # Process each file
        items: List[BatchAckItem] = []
        for f in files:
            try:
                # Get metadata for this file
                meta = meta_by_filename.get(f.filename, {})
                episode_id = meta.get("episode_id")
                note_date = meta.get("note_date")

                if not episode_id:
                    raise ValueError(f"Missing 'episode_id' for '{f.filename}'")
                if not note_date:
                    raise ValueError(f"Missing 'note_date' for '{f.filename}'")

                # Extract using single extraction workflow
                ack = await self.extract_single(
                    file=f,
                    model=model,
                    model_variant=model_variant,
                    episode_id=episode_id,
                    note_date=note_date,
                    save=save,
                    normalize=normalize,
                    systems_csv=systems_csv,
                    restrict_types_csv=restrict_types_csv,
                    expand=False,
                    filename_override=f.filename,
                )

                items.append(
                    BatchAckItem(
                        filename=f.filename,
                        id=ack.id if ack.id else None,
                        stored=ack.stored,
                        entity_count=ack.entity_count,
                        url=ack.url,
                    )
                )

            except Exception as e:
                logger.error(f"Failed to process file {f.filename}: {e}")
                items.append(
                    BatchAckItem(
                        filename=f.filename,
                        stored=False,
                        error=str(e),
                    )
                )

        return BatchAckResponse(items=items)

    async def get_note(self, note_id: str) -> ExtractResponse:
        """
        Retrieve a stored extraction result by note ID.

        Args:
            note_id: UUID of the note to retrieve

        Returns:
            ExtractResponse with text, entities, and metadata

        Raises:
            HTTPException: 404 if note not found
        """
        note = await self.repository.find_note_by_id(note_id)

        if not note:
            raise HTTPException(status_code=404, detail="Note not found")

        return ExtractResponse(
            text=note.get("text") or "",
            entities=[Entity(**e) for e in note.get("entities", [])],
            meta=note.get("meta") or {},
        )

    async def _save_extraction_result(
        self,
        text: str,
        result: ExtractResponse,
        model: str,
        episode_id: str,
        note_date_iso: str,
        filename: Optional[str] = None,
        source_system: Optional[str] = None,
    ) -> str:
        """
        Save extraction result using repository.

        Args:
            text: Original note text
            result: Extraction result
            model: Model identifier
            episode_id: Episode ID
            note_date_iso: ISO 8601 date string
            filename: Optional filename
            source_system: Optional source system

        Returns:
            Note ID (UUID)
        """
        note_id = str(uuid.uuid4())

        # Build complete note document
        note_doc = self.repository.build_note_document(
            note_id=note_id,
            text=text,
            entities=getattr(result, "entities", []),
            meta=getattr(result, "meta", {}),
            model=model,
            note_date_iso=note_date_iso,
            filename=filename,
            source_system=source_system,
        )

        # Save via repository with deduplication
        return await self.repository.save_note(
            episode_id=episode_id,
            note_data=note_doc,
            dedupe_by_hash=True,
        )
