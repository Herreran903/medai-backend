"""pipeline.py
Servicio de alto nivel para la extracción de entidades desde texto.

Responsabilidades principales:
 1. Resolver y construir el extractor adecuado a partir de `MODEL_REGISTRY`.
 2. Para modelos Transformer, seleccionar la variante correcta (BETO / RoBERTa).
 3. Estandarizar y validar las entidades devueltas por los modelos base.
 4. Aplicar la capa de normalización semántica (opcional).
 5. Construir un objeto [`ExtractResponse()`](app/schemas.py:1) consistente para el API.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from app.config import get_settings
from app.models.transformer import TransformerExtractor
from app.schemas import Entity, ExtractResponse
from app.services.normalizer import NormOptions, normalize_entities
from app.services.registry import MODEL_REGISTRY

logger = logging.getLogger(__name__)

# Caché en memoria para instancias de extractores Transformer diferenciadas por variante.
_TRANSFORMER_CACHE: Dict[str, TransformerExtractor] = {}


def _resolve_transformer_extractor(
    model_variant: Optional[str],
) -> TransformerExtractor:
    """Devuelve una instancia de [`TransformerExtractor()`](app/models/transformer.py:291) configurada.

    - Usa `model_variant` para decidir entre BETO o RoBERTa.
    - Lee los `model_id` concretos desde [`Settings`](app/config.py:17).
    - Reutiliza instancias ya inicializadas para evitar recargar pesos.
    """

    # Normalizamos la variante; BETO es el valor por defecto (backward compatible).
    variant = (model_variant or "beto").strip().lower()
    if variant not in {"beto", "roberta"}:
        logger.warning(
            "Variante de transformer desconocida '%s', usando 'beto' por defecto",
            variant,
        )
        variant = "beto"

    cache_key = f"transformer:{variant}"
    if cache_key in _TRANSFORMER_CACHE:
        return _TRANSFORMER_CACHE[cache_key]

    settings = get_settings()
    if variant == "roberta":
        model_id = settings.transformer_roberta_model_id
    else:
        model_id = settings.transformer_beto_model_id

    logger.info(
        "Inicializando TransformerExtractor: variant=%s model_id=%s",
        variant,
        model_id,
    )
    extractor = TransformerExtractor(model_id=model_id)
    _TRANSFORMER_CACHE[cache_key] = extractor
    return extractor


def extract_from_text(
    text: str,
    model: str,
    *,
    model_variant: Optional[str] = None,
    normalize: bool = False,
    systems: Optional[List[str]] = None,
    restrict_types: Optional[List[str]] = None,
) -> ExtractResponse:
    """Punto único de entrada para extraer entidades desde texto plano.

    Parámetros
    ----------
    text:
        Texto clínico fuente (cadena arbitraria; se normaliza a `str`).
    model:
        Clave del modelo registrada en [`MODEL_REGISTRY`](app/services/registry.py:15)
        (por ejemplo: "lstm", "transformer", "llm").
    model_variant:
        Variante concreta cuando `model == "transformer"` ("beto" o "roberta").
    normalize:
        Si es `True`, aplica normalización semántica sobre las entidades extraídas.
    systems:
        Lista de sistemas terminológicos objetivo para la normalización.
    restrict_types:
        Tipos de entidades a conservar durante la normalización.
    """

    # Aseguramos que el texto siempre sea una cadena (defensivo ante integraciones externas).
    if text is None:
        text = ""
    elif not isinstance(text, str):
        text = str(text)

    if model not in MODEL_REGISTRY:
        raise ValueError(f"Modelo no soportado: {model}")

    logger.info("Usando modelo de extracción: %s (variant=%s)", model, model_variant)

    # Selección explícita de variante para Transformer (BETO / RoBERTa)
    if model == "transformer":
        extractor = _resolve_transformer_extractor(model_variant)
    else:
        # Para otros modelos, usamos directamente el registro global.
        extractor = MODEL_REGISTRY[model]
        # Permitimos tanto instancias como fábricas (callables sin .predict inicial).
        if callable(extractor) and not hasattr(extractor, "predict"):
            extractor = extractor()
            MODEL_REGISTRY[model] = extractor

    # Llamada al modelo base con manejo de errores controlado.
    try:
        raw_entities = extractor.predict(text)
    except Exception as exc:  # pragma: no cover - defensivo
        logger.exception("Error al ejecutar predict() en el extractor '%s'", model)
        raise RuntimeError("Fallo interno del modelo de extracción") from exc

    # Normalizamos estructura de entidades a objetos [`Entity`](app/schemas.py:1).
    entities: List[Entity] = []
    for item in raw_entities or []:
        if not isinstance(item, dict):
            logger.warning("Entidad cruda ignorada por tipo inesperado: %r", type(item))
            continue

        e = dict(item)  # copiamos para no mutar el original
        s, end = e.get("start"), e.get("end")
        if s is not None and end is not None:
            # Validamos que el span esté dentro de los límites del texto.
            if not (0 <= s < end <= len(text)):
                logger.warning(
                    "Span inválido descartando offsets (start=%s, end=%s, len=%s)",
                    s,
                    end,
                    len(text),
                )
                e = {k: v for k, v in e.items() if k not in ("start", "end")}

        try:
            entities.append(Entity(**e))
        except Exception as exc:  # pragma: no cover - defensivo
            logger.warning(
                "Entidad cruda no válida, se descarta: %r (error=%s)", e, exc
            )

    # Capa opcional de normalización semántica (UMLS / terminologías).
    if normalize and entities:
        opts = NormOptions(
            enabled=True,
            systems=systems,
            restrict_types=restrict_types,
            min_link_score=0.60,
            max_candidates=25,
        )
        ents_dicts = [e.model_dump() for e in entities]
        ents_norm = normalize_entities(ents_dicts, opts)
        entities = [Entity(**d) for d in ents_norm]

    meta: Dict[str, Any] = {
        "model": model,
        "count": len(entities),
        "normalized": bool(normalize),
    }

    # Enriquecemos con metadatos del extractor si los expone.
    if hasattr(extractor, "meta"):
        try:
            meta.update(extractor.meta())
        except Exception:  # pragma: no cover - best-effort
            logger.debug(
                "Fallo al leer meta() del extractor '%s'", model, exc_info=True
            )

    return ExtractResponse(text=text or "", entities=entities, meta=meta)
