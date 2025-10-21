from datetime import datetime
from typing import List, Optional

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
from app.services.pipeline import extract_from_text
from app.services.store import save_result
from app.services.text_utils import read_any_to_text

router = APIRouter()


def _parse_iso8601(dt: Optional[str]) -> Optional[datetime]:
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
            raise HTTPException(status_code=400, detail="Formato de note_date inválido")


def _parse_csv(v: Optional[str]) -> List[str]:
    if not v:
        return []
    return [p.strip() for p in v.split(",") if p.strip()]


@router.post("/extract", response_model=ExtractAck, status_code=status.HTTP_201_CREATED)
async def extract(
    text: Optional[str] = Form(None),
    file: Optional[UploadFile] = File(None),
    model: str = Form(...),
    episode_id: Optional[str] = Form(None),
    note_date: Optional[str] = Form(None),
    save: Optional[bool] = Form(True),
    normalize: Optional[bool] = Form(False),
    systems_csv: Optional[str] = Form(None),
    restrict_types_csv: Optional[str] = Form(None),
    expand: Optional[bool] = Form(False),
    db: Database = Depends(get_db),
    settings: Settings = Depends(settings_dep),
):
    if not text and not file:
        raise HTTPException(status_code=400, detail="Proporciona 'text' o 'file'")

    if file:
        content = await file.read()
        text = read_any_to_text(file.filename, content)

    systems = _parse_csv(systems_csv)
    restrict_types = _parse_csv(restrict_types_csv)

    note_dt = _parse_iso8601(note_date)
    note_date_iso = note_dt.isoformat() if note_dt else None

    res = extract_from_text(
        text or "",
        model=model or settings.default_model,
        normalize=bool(normalize),
        systems=systems,
        restrict_types=restrict_types or None,
    )

    if not episode_id:
        raise HTTPException(status_code=400, detail="Falta 'episode_id'")
    if not note_date_iso:
        raise HTTPException(
            status_code=400, detail="Falta 'note_date' o formato inválido"
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


@router.post("/extract-batch", response_model=BatchAckResponse)
async def extract_batch(
    files: List[UploadFile] = File(...),
    model: str = Form(...),
    save: Optional[bool] = Form(True),
    normalize: Optional[bool] = Form(False),
    systems_csv: Optional[str] = Form(None),
    restrict_types_csv: Optional[str] = Form(None),
    db: Database = Depends(get_db),
    settings: Settings = Depends(settings_dep),
):
    systems = _parse_csv(systems_csv)
    restrict_types = _parse_csv(restrict_types_csv)

    items: List[BatchAckItem] = []
    for f in files:
        try:
            content = await f.read()
            text = read_any_to_text(f.filename, content)
            res = extract_from_text(
                text,
                model=model or settings.default_model,
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


@router.get("/notes/{note_id}", response_model=ExtractResponse)
async def get_note(
    note_id: str = Path(..., description="UUID de la nota (note_id)"),
    db: Database = Depends(get_db),
):
    doc = db.episodes.find_one(
        {"notes.note_id": note_id},
        {"_id": 0, "notes": {"$elemMatch": {"note_id": note_id}}},
    )

    if not doc or not doc.get("notes"):
        raise HTTPException(status_code=404, detail="Nota no encontrada")

    note = doc["notes"][0]
    return ExtractResponse(
        text=note.get("text") or "",
        entities=[Entity(**e) for e in note.get("entities", [])],
        meta=note.get("meta") or {},
    )
