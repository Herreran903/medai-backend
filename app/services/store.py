from datetime import datetime

from pymongo.database import Database


def save_result(
    db: Database,
    payload: str,
    result,
    model: str,
    filename: str | None = None,
):
    doc = {
        "filename": filename,
        "text": payload,
        "entities": [e.model_dump() for e in result.entities],
        "meta": result.meta,
        "model": model,
        "created_at": datetime.utcnow(),
    }
    db.extractions.insert_one(doc)
