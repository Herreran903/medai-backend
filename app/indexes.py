from pymongo.database import Database


def ensure_indexes(db: Database) -> None:
    eps = db.episodes
    eps.create_index("updated_at")
    eps.create_index("notes.note_date")
    eps.create_index("notes.content_hash")
