"""
MongoDB Index Management for MedAI Backend.

This module defines database indexes for the MedAI MongoDB collections,
optimizing query performance for common access patterns in the clinical
note extraction workflow.

Architecture Context:
    MedAI stores clinical episodes and their associated notes in MongoDB.
    Each episode contains an array of notes with extracted entities.
    The indexes defined here support efficient querying by:

    - Temporal ordering (updated_at, note_date)
    - Content deduplication (content_hash)

Index Strategy:
    Indexes are created idempotently on application startup via
    :func:`ensure_indexes`. MongoDB's ``create_index`` is safe to call
    multiple times; existing indexes are not recreated.

Data Model:
    The ``episodes`` collection follows this structure::

        {
            "_id": "episode-uuid",
            "created_at": ISODate,
            "updated_at": ISODate,
            "notes": [
                {
                    "note_id": "note-uuid",
                    "note_date": ISODate,
                    "content_hash": "sha256-hex",
                    "text": "clinical note content",
                    "entities": [...],
                    "meta": {...}
                }
            ]
        }

Usage:
    This module is invoked during application startup in :mod:`app.main`::

        from app.indexes import ensure_indexes
        ensure_indexes(db)

See Also:
    - :mod:`app.repositories.episode_repository` for data access layer
    - :mod:`app.main` for startup initialization
"""

from pymongo.database import Database


def ensure_indexes(db: Database) -> None:
    """
    Create required indexes on the episodes collection.

    This function ensures that all necessary indexes exist for optimal
    query performance. It is designed to be called on application startup
    and is idempotent (safe to call multiple times).

    Args:
        db: MongoDB database instance from :func:`app.deps.get_db`.

    Indexes Created:
        ``episodes.updated_at``:
            Supports sorting and filtering episodes by last modification time.
            Used for listing recently updated episodes and cache invalidation.

        ``episodes.notes.note_date``:
            Supports querying notes by clinical date within episodes.
            Enables temporal filtering for clinical timeline views.

        ``episodes.notes.content_hash``:
            Supports content-based deduplication of notes.
            Used by :meth:`app.repositories.episode_repository.EpisodeRepository.add_note`
            to prevent duplicate note insertion based on SHA-256 content hash.

    Performance Considerations:
        - All indexes are single-field ascending indexes
        - Array field indexes (notes.*) create multikey indexes
        - Index creation is non-blocking for existing collections

    Example:
        >>> from pymongo import MongoClient
        >>> client = MongoClient("mongodb://localhost:27017")
        >>> db = client["medai"]
        >>> ensure_indexes(db)
        >>> # Indexes are now available for queries

    Note:
        Index creation on large collections may take time on first run.
        Subsequent calls return immediately as indexes already exist.
    """
    eps = db.episodes

    # Temporal index for episode ordering and filtering
    eps.create_index("updated_at")

    # Temporal index for note-level date queries
    eps.create_index("notes.note_date")

    # Content hash index for deduplication lookups
    eps.create_index("notes.content_hash")
