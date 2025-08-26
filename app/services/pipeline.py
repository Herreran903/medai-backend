from app.schemas import Entity, ExtractResponse
from app.services.registry import MODEL_REGISTRY


def extract_from_text(text: str, model: str, threshold: float = 0.5) -> ExtractResponse:
    if model not in MODEL_REGISTRY:
        raise ValueError(f"Modelo no soportado: {model}")
    extractor = MODEL_REGISTRY[model]
    raw_entities = extractor.predict(text)

    ent = []
    for e in raw_entities:
        if e.get("score", 1.0) >= threshold:
            ent.append(Entity(**e))
    meta = {"model": model, "count": len(ent)}
    return ExtractResponse(entities=ent, meta=meta)
