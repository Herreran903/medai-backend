from __future__ import annotations

from inspect import signature
from typing import Any, Dict, List

from app.schemas import Entity, ExtractResponse
from app.services.registry import MODEL_REGISTRY


def extract_from_text(text: str, model: str) -> ExtractResponse:
    if model not in MODEL_REGISTRY:
        raise ValueError(f"Modelo no soportado: {model}")

    extractor = MODEL_REGISTRY[model]
    if callable(extractor) and not hasattr(extractor, "predict"):
        extractor = extractor()
        MODEL_REGISTRY[model] = extractor

    try:
        raw_entities = extractor.predict(text)
    except TypeError:
        raw_entities = extractor.predict(text)
    except Exception as e:
        raise RuntimeError(f"Error ejecutando el extractor '{model}': {e}") from e

    entities: List[Entity] = []
    for e in raw_entities or []:
        s, end = e.get("start"), e.get("end")
        if s is not None and end is not None:
            if not (0 <= s < end <= len(text or "")):
                e = {k: v for k, v in e.items() if k not in ("start", "end")}
        entities.append(Entity(**e))

    meta: Dict[str, Any] = {"model": model, "count": len(entities)}
    if hasattr(extractor, "meta"):
        try:
            ext_meta = extractor.meta()
            if isinstance(ext_meta, dict):
                meta.update(ext_meta)
        except Exception:
            pass

    return ExtractResponse(text=text or "", entities=entities, meta=meta)
