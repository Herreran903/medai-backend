"""
Episode Repository for MongoDB Operations.

This module implements the Repository pattern for episode and note data access,
providing a clean abstraction over MongoDB operations.

The repository encapsulates all database logic for episodes and notes,
making it easier to:
- Test business logic with mock repositories
- Change database implementation if needed
- Maintain consistent data access patterns
- Apply database optimizations in one place

Architecture:
    This follows the Repository pattern from the fastapi-templates skill,
    separating data access concerns from business logic and API routing.

Usage:
    >>> from app.repositories import EpisodeRepository
    >>> from pymongo.database import Database
    >>>
    >>> repository = EpisodeRepository(db)
    >>> note_id = await repository.save_note(
    ...     episode_id="EP-001",
    ...     note_data={...},
    ...     dedupe_by_hash=True
    ... )
"""

import hashlib
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from pymongo.database import Database


class EpisodeRepository:
    """
    Repository for episode and note data access.

    This class encapsulates all MongoDB operations related to episodes and notes,
    providing a clean interface for the service layer.

    Attributes:
        db: MongoDB database instance
        collection: Episodes collection reference
    """

    def __init__(self, db: Database):
        """
        Initialize repository with database connection.

        Args:
            db: MongoDB database instance from get_db dependency
        """
        self.collection = db.episodes

    @staticmethod
    def _compute_hash(content: str) -> str:
        """
        Compute SHA-256 hash of content for deduplication.

        Args:
            content: Text content to hash

        Returns:
            Hexadecimal hash string
        """
        return hashlib.sha256(content.encode("utf-8")).hexdigest()

    async def find_note_by_id(self, note_id: str) -> Optional[Dict[str, Any]]:
        """
        Find a note by its unique ID.

        Args:
            note_id: UUID of the note to find

        Returns:
            Note document if found, None otherwise
        """
        doc = self.collection.find_one(
            {"notes.note_id": note_id},
            {"_id": 0, "notes": {"$elemMatch": {"note_id": note_id}}},
        )

        if not doc or not doc.get("notes"):
            return None

        return doc["notes"][0]

    async def find_note_by_hash(
        self, episode_id: str, content_hash: str
    ) -> Optional[str]:
        """
        Find existing note ID with matching content hash in an episode.

        Args:
            episode_id: Episode identifier
            content_hash: SHA-256 hash of note content

        Returns:
            Note ID if duplicate found, None otherwise
        """
        existing = self.collection.find_one(
            {"_id": episode_id, "notes.content_hash": content_hash},
            {"notes.$": 1},
        )

        if existing and existing.get("notes"):
            return existing["notes"][0]["note_id"]

        return None

    async def save_note(
        self,
        episode_id: str,
        note_data: Dict[str, Any],
        *,
        dedupe_by_hash: bool = True,
    ) -> str:
        """
        Save a note to an episode with optional deduplication.

        This method handles both creating new episodes and appending notes
        to existing episodes via MongoDB upsert.

        Args:
            episode_id: Episode identifier (used as document _id)
            note_data: Complete note document to insert (must include all required fields)
            dedupe_by_hash: Whether to check for duplicate content before inserting

        Returns:
            str: Note ID (UUID). Returns existing note ID if duplicate found.

        Note Structure:
            note_data should contain:
            - note_id: UUID string
            - text: Original note content
            - entities: List of extracted entities
            - meta: Extraction metadata
            - model: Model used
            - note_date: ISO 8601 date string
            - created_at: datetime object
            - content_hash: SHA-256 hash
            - filename: Optional filename
            - source_system: Optional source identifier
        """
        created_at = datetime.now(timezone.utc)
        note_id = note_data.get("note_id")

        if not note_id:
            raise ValueError("note_data must include 'note_id'")

        # Check for duplicates if requested
        if dedupe_by_hash:
            content_hash = note_data.get("content_hash")
            if content_hash:
                existing_id = await self.find_note_by_hash(episode_id, content_hash)
                if existing_id:
                    return existing_id

        # Upsert episode with new note
        self.collection.update_one(
            {"_id": episode_id},
            {
                "$setOnInsert": {"_id": episode_id, "created_at": created_at},
                "$push": {"notes": note_data},
                "$set": {"updated_at": created_at},
            },
            upsert=True,
        )

        return note_id

    def build_note_document(
        self,
        note_id: str,
        text: str,
        entities: list,
        meta: dict,
        model: str,
        note_date_iso: str,
        *,
        filename: Optional[str] = None,
        source_system: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Build a complete note document for insertion.

        This helper method constructs a properly formatted note document
        with all required fields, including content hash computation.

        Args:
            note_id: Unique note identifier (UUID)
            text: Original note text content
            entities: List of extracted Entity objects
            meta: Extraction metadata dict
            model: Model identifier used for extraction
            note_date_iso: ISO 8601 formatted date string
            filename: Optional original filename
            source_system: Optional source system identifier

        Returns:
            Complete note document ready for insertion
        """
        created_at = datetime.now(timezone.utc)
        content_hash = self._compute_hash(text or "")

        return {
            "note_id": note_id,
            "filename": filename,
            "source_system": source_system,
            "text": text,
            "entities": [e.model_dump() for e in entities],
            "meta": meta,
            "model": model,
            "note_date": note_date_iso,
            "created_at": created_at,
            "content_hash": content_hash,
        }
