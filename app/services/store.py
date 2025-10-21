import hashlib
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from pymongo.database import Database


def _sha256(s: str) -> str:
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
    created_at = datetime.now(timezone.utc)

    content_hash = _sha256(payload or "")
    note_id = str(uuid.uuid4())

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
        upd = episodes.update_one(
            {
                "_id": episode_id,
                "notes": {"$not": {"$elemMatch": {"content_hash": content_hash}}},
            },
            {
                "$setOnInsert": {"_id": episode_id, "created_at": created_at},
                "$push": {"notes": note},
                "$set": {"updated_at": created_at},
            },
            upsert=True,
        )

        if upd.modified_count == 1:
            return note_id

        existing = episodes.find_one(
            {"_id": episode_id, "notes.content_hash": content_hash},
            {"notes.$": 1},
        )
        if existing and "notes" in existing and existing["notes"]:
            return existing["notes"][0]["note_id"]

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
