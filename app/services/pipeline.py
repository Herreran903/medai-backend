from typing import Any, Dict, List

from app.schemas import Entity, ExtractResponse
from app.services.normalizer import NormOptions, normalize_entities
from app.services.registry import MODEL_REGISTRY


def extract_from_text(
    text: str,
    model: str,
    *,
    normalize: bool = False,
    systems: List[str] | None = None,
    restrict_types: List[str] | None = None,
) -> ExtractResponse:
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

        if normalize and entities:
            opts = NormOptions(
                enabled=True,
                systems=systems or None,
                restrict_types=restrict_types,
                min_link_score=0.60,
                max_candidates=25,
                vsac_whitelists=None,
            )
            ents_dicts = [e.model_dump() for e in entities]
            ents_norm = normalize_entities(ents_dicts, opts)
            entities = [Entity(**d) for d in ents_norm]

    meta: Dict[str, Any] = {
        "model": model,
        "count": len(entities),
        "normalized": bool(normalize),
    }
    if hasattr(extractor, "meta"):
        try:
            ext_meta = extractor.meta()
            if isinstance(ext_meta, dict):
                meta.update(ext_meta)
        except Exception:
            pass

    return ExtractResponse(text=text or "", entities=entities, meta=meta)
