# app/routers/extract.py
from typing import List, Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from pymongo.database import Database

from app.config import Settings
from app.deps import get_db, settings_dep
from app.schemas import BatchItem, ExtractResponse
from app.services.pipeline import extract_from_text
from app.services.store import save_result
from app.services.text_utils import read_any_to_text

router = APIRouter()


@router.post("/extract", response_model=ExtractResponse)
async def extract(
    text: Optional[str] = Form(None),
    file: Optional[UploadFile] = File(None),
    model: str = Form(...),
    save: Optional[bool] = Form(True),
    db: Database = Depends(get_db),
    settings: Settings = Depends(settings_dep),
):
    if not text and not file:
        raise HTTPException(400, "Proporciona 'text' o 'file'")

    if file:
        content = await file.read()
        text = read_any_to_text(file.filename, content)

    res = extract_from_text(text or "", model=model or settings.default_model)

    if save and settings.save_results:
        save_result(db=db, payload=text or "", result=res, model=model)

    return res


@router.post("/extract-batch", response_model=List[BatchItem])
async def extract_batch(
    files: List[UploadFile] = File(...),
    model: str = Form(...),
    save: Optional[bool] = Form(True),
    db: Database = Depends(get_db),
    settings: Settings = Depends(settings_dep),
):
    items: List[BatchItem] = []
    for f in files:
        content = await f.read()
        text = read_any_to_text(f.filename, content)
        res = extract_from_text(text, model=model or settings.default_model)
        if save and settings.save_results:
            save_result(
                db=db, payload=text, result=res, model=model, filename=f.filename
            )
        items.append(
            BatchItem(filename=f.filename, entities=res.entities, meta=res.meta)
        )
    return items
