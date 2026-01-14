"""
Clinical Entity Extraction Pipeline Service.

This module provides the high-level orchestration layer for clinical Named Entity
Recognition (NER), coordinating model selection, extraction, validation, and
optional normalization.

Architecture Context:
    The pipeline service is the central coordination point for all extraction
    operations in MedAI. It abstracts the complexity of:

    - Model selection and initialization
    - Transformer variant resolution (BETO/RoBERTa)
    - Entity validation and standardization
    - Optional UMLS normalization

    All API endpoints delegate to this service via :func:`extract_from_text`.

Pipeline Flow:
    .. code-block:: text

        Input Text
            │
            ▼
        ┌─────────────────┐
        │ Model Selection │ ← MODEL_REGISTRY / Transformer Cache
        └────────┬────────┘
                 │
                 ▼
        ┌─────────────────┐
        │   Extraction    │ ← LSTM / Transformer / LLM
        └────────┬────────┘
                 │
                 ▼
        ┌─────────────────┐
        │   Validation    │ ← Span bounds, type checking
        └────────┬────────┘
                 │
                 ▼
        ┌─────────────────┐
        │ Normalization   │ ← Optional UMLS lookup
        └────────┬────────┘
                 │
                 ▼
        ExtractResponse

Model Caching:
    Transformer models are cached in memory to avoid repeated weight loading:

    - Cache key: ``transformer:{variant}`` (e.g., "transformer:beto")
    - Cache is module-level, persists for application lifetime
    - LSTM and LLM models use :data:`app.services.registry.MODEL_REGISTRY`

Usage:
    >>> from app.services.pipeline import extract_from_text
    >>> result = extract_from_text(
    ...     "Paciente con FiO2 60%, PEEP 8 cmH2O",
    ...     model="transformer",
    ...     model_variant="beto",
    ...     normalize=False
    ... )
    >>> print(result.entities[0].type)
    'FIO2'

See Also:
    - :mod:`app.routers.extract` for API endpoint integration
    - :mod:`app.services.registry` for model registration
    - :mod:`app.services.normalizer` for UMLS normalization
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from app.config import get_settings
from app.models.transformer import TransformerExtractor
from app.schemas import Entity, ExtractResponse

# from app.services.normalizer import NormOptions, normalize_entities
from app.services.registry import MODEL_REGISTRY

logger = logging.getLogger(__name__)

# In-memory cache for Transformer extractors by variant
_TRANSFORMER_CACHE: Dict[str, TransformerExtractor] = {}
"""
Cache for initialized Transformer extractors.

Keys are formatted as ``transformer:{variant}`` where variant is
"beto" or "roberta". This avoids reloading model weights on each request.
"""


def _resolve_transformer_extractor(
    model_variant: Optional[str],
) -> TransformerExtractor:
    """
    Resolve and cache a Transformer extractor for the specified variant.

    This function implements lazy initialization with caching for Transformer
    models, ensuring that model weights are loaded only once per variant.

    Args:
        model_variant: Transformer variant ("beto" or "roberta").
            Defaults to "beto" if not specified or invalid.

    Returns:
        TransformerExtractor: Cached or newly initialized extractor.

    Caching:
        Extractors are cached by variant in :data:`_TRANSFORMER_CACHE`.
        The cache persists for the application lifetime.

    Configuration:
        Model IDs are read from :class:`app.config.Settings`:

        - ``transformer_beto_model_id``: BETO model Hugging Face ID
        - ``transformer_roberta_model_id``: RoBERTa model Hugging Face ID

    Example:
        >>> extractor = _resolve_transformer_extractor("roberta")
        >>> extractor.variant
        'roberta'
    """
    # Normalize variant with BETO as default (backward compatible)
    variant = (model_variant or "beto").strip().lower()
    if variant not in {"beto", "roberta"}:
        logger.warning(
            "Unknown transformer variant '%s', defaulting to 'beto'",
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
        "Initializing TransformerExtractor: variant=%s model_id=%s",
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
    """
    Extract clinical entities from text using the specified model.

    This is the primary entry point for all extraction operations in MedAI.
    It coordinates model selection, extraction, validation, and optional
    normalization into a single unified interface.

    Args:
        text: Clinical note text to process. Accepts any string input;
            None and non-string values are coerced to empty string.
        model: Extraction model identifier. Must be a key in
            :data:`app.services.registry.MODEL_REGISTRY`:

            - ``lstm``: BiLSTM-CRF model
            - ``transformer``: Fine-tuned BETO/RoBERTa
            - ``llm``: LLM-based extraction (Claude/GPT)

        model_variant: Model-specific variant selection:

            - For ``transformer``: "beto" (default) or "roberta"
            - For ``llm``: "claude" (default) or "gpt"
            - Ignored for ``lstm``

        normalize: Whether to apply UMLS normalization to DX entities.
            Requires ``UMLS_APIKEY`` environment variable.
        systems: Target terminology systems for normalization
            (e.g., ["SNOMEDCT_US", "ICD10CM"]).
        restrict_types: Entity types to include in normalization
            (e.g., ["DX"]). None means all supported types.

    Returns:
        ExtractResponse: Extraction result containing:

        - ``text``: Original input text
        - ``entities``: List of extracted :class:`Entity` objects
        - ``meta``: Extraction metadata (model, count, normalized flag)

    Raises:
        ValueError: If the specified model is not in MODEL_REGISTRY.
        RuntimeError: If model prediction fails.

    Pipeline Steps:
        1. **Input Validation**: Coerce text to string
        2. **Model Resolution**: Select extractor from registry or cache
        3. **Extraction**: Call model's ``predict()`` method
        4. **Entity Validation**: Validate spans and construct Entity objects
        5. **Normalization**: Optional UMLS code assignment
        6. **Response Construction**: Build ExtractResponse with metadata

    Example:
        >>> # Basic extraction with transformer
        >>> result = extract_from_text(
        ...     "FiO2 60%, PEEP 8 cmH2O",
        ...     model="transformer"
        ... )
        >>> len(result.entities)
        2

        >>> # Extraction with normalization
        >>> result = extract_from_text(
        ...     "Diagnóstico: neumonía",
        ...     model="transformer",
        ...     normalize=True,
        ...     systems=["SNOMEDCT_US"]
        ... )
        >>> result.entities[0].codes[0].system
        'SNOMEDCT_US'

    Note:
        Model loading occurs on first use. Transformer models are cached
        for subsequent requests. LSTM and LLM models are initialized
        at application startup via MODEL_REGISTRY.
    """
    # Input validation: ensure text is always a string
    if text is None:
        text = ""
    elif not isinstance(text, str):
        text = str(text)

    if model not in MODEL_REGISTRY:
        raise ValueError(f"Unsupported model: {model}")

    logger.info("Using extraction model: %s (variant=%s)", model, model_variant)

    # Model selection with variant handling for Transformer
    if model == "transformer":
        extractor = _resolve_transformer_extractor(model_variant)
    else:
        # Use global registry for LSTM and LLM
        extractor = MODEL_REGISTRY[model]
        # Support both instances and factory callables
        if callable(extractor) and not hasattr(extractor, "predict"):
            extractor = extractor()
            MODEL_REGISTRY[model] = extractor

    # Execute extraction with error handling
    try:
        raw_entities = extractor.predict(text)
    except Exception as exc:
        logger.exception("Error executing predict() on extractor '%s'", model)
        raise RuntimeError("Internal model extraction failure") from exc

    # Validate and construct Entity objects
    entities: List[Entity] = []
    for item in raw_entities or []:
        if not isinstance(item, dict):
            logger.warning("Ignoring non-dict entity: %r", type(item))
            continue

        e = dict(item)  # Copy to avoid mutation
        s, end = e.get("start"), e.get("end")
        if s is not None and end is not None:
            # Validate span bounds
            if not (0 <= s < end <= len(text)):
                logger.warning(
                    "Invalid span, discarding offsets (start=%s, end=%s, len=%s)",
                    s,
                    end,
                    len(text),
                )
                e = {k: v for k, v in e.items() if k not in ("start", "end")}

        try:
            entities.append(Entity(**e))
        except Exception as exc:
            logger.warning("Invalid entity discarded: %r (error=%s)", e, exc)

    # Optional UMLS normalization (temporarily disabled to avoid extra model loads)
    # if normalize and entities:
    #     opts = NormOptions(
    #         enabled=True,
    #         systems=systems,
    #         restrict_types=restrict_types,
    #         min_link_score=0.60,
    #         max_candidates=25,
    #     )
    #     ents_dicts = [e.model_dump() for e in entities]
    #     ents_norm = normalize_entities(ents_dicts, opts)
    #     entities = [Entity(**d) for d in ents_norm]
    normalize = False

    # Build metadata
    meta: Dict[str, Any] = {
        "model": model,
        "count": len(entities),
        "normalized": bool(normalize),
    }

    # Include extractor metadata if available
    if hasattr(extractor, "meta"):
        try:
            meta.update(extractor.meta())
        except Exception:
            logger.debug(
                "Failed to read meta() from extractor '%s'", model, exc_info=True
            )

    return ExtractResponse(text=text or "", entities=entities, meta=meta)
