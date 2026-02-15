"""
Clinical Entity Extraction Pipeline Service.

This module provides the high-level orchestration layer for clinical Named Entity
Recognition (NER), coordinating model selection, extraction, validation, and
optional normalization.

Architecture Context:
    The pipeline service is the central coordination point for all extraction
    operations in MedAI. It abstracts the complexity of:

    - Model selection and initialization
    - Transformer variant resolution (RoBERTa fixed)
    - Entity validation and standardization
    - Optional terminology normalization (handled downstream; disabled in gateway path)

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
        │   Extraction    │ ← BiLSTM / CRF / Transformer / LLM
        └────────┬────────┘
                 │
                 ▼
        ┌─────────────────┐
        │   Validation    │ ← Span bounds, type checking
        └────────┬────────┘
                 │
                 ▼
        ┌─────────────────┐
        │ Normalization   │ ← Regex value normalization; terminology optional (downstream)
        └────────┬────────┘
                 │
                 ▼
        ExtractResponse

Model Caching:
    Transformer models are cached in memory to avoid repeated weight loading:

    - Cache key: ``transformer:{variant}`` (e.g., "transformer:roberta")
    - Cache is module-level, persists for application lifetime
    - BiLSTM and LLM models use :data:`app.services.registry.MODEL_REGISTRY`

Usage:
    >>> from app.services.pipeline import extract_from_text
    >>> result = extract_from_text(
    ...     "Paciente con FiO2 60%, PEEP 8 cmH2O",
    ...     model="transformer",
    ...     model_variant="roberta",
    ...     normalize=False
    ... )
    >>> print(result.entities[0].type)
    'FIO2'

See Also:
    - :mod:`app.routers.extract` for API endpoint integration
    - :mod:`app.services.registry` for model registration
"""

from __future__ import annotations

import logging
from typing import List, Optional

from app.config import get_settings
from app.schemas import Entity, ExtractResponse
from app.services.ner_client import NERClient, NERServiceError
from app.services.registry import MODEL_REGISTRY
from app.services.utils import extract_normalized_value

logger = logging.getLogger(__name__)


async def extract_from_text(
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

            - ``lstm``: BiLSTM model (no CRF layer)
            - ``lstm_crf``: BiLSTM-CRF model (Viterbi decoding)
            - ``crf``: CRF model (sklearn-crfsuite)
            - ``transformer``: Fine-tuned RoBERTa (fixed)
            - ``llm``: LLM-based extraction (GPT fixed)

        model_variant: Model-specific variant selection:

            - For ``transformer``: "roberta" (fixed for experiments)
            - For ``llm``: "gpt" (fixed for experiments)
            - Ignored for ``lstm``, ``lstm_crf``, and ``crf``

        normalize: Whether to request terminology normalization from the NER service.
            Note: the gateway service layer currently forces this to False.
        systems: Target terminology systems for normalization
            (e.g., ["SNOMEDCT_US", "ICD10CM"]).
        restrict_types: Entity types to include in normalization
            (e.g., ["DX"]). None means all supported types.

    Returns:
        ExtractResponse: Extraction result containing:

        - ``text``: Original input text
        - ``entities``: List of extracted :class:`Entity` objects
        - ``meta``: Extraction metadata (model, inference_time_ms, entity_count)

    Raises:
        ValueError: If the specified model is not in MODEL_REGISTRY.
        RuntimeError: If model prediction fails.

    Pipeline Steps:
        1. **Input Validation**: Coerce text to string
        2. **Model Resolution**: Select extractor from registry or cache
        3. **Extraction**: Call model's ``predict()`` method
        4. **Entity Validation**: Validate spans and construct Entity objects
        5. **Normalization**: Regex value normalization; optional terminology normalization downstream
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

    Note:
        Model loading occurs on first use. Transformer models are cached
        for subsequent requests. BiLSTM and LLM models are initialized
        at application startup via MODEL_REGISTRY.
    """
    # Input validation: ensure text is always a string
    if text is None:
        text = ""
    elif not isinstance(text, str):
        text = str(text)

    if model not in MODEL_REGISTRY:
        raise ValueError(f"Unsupported model: {model}")

    logger.info("Using extraction model: %s (variant=%s) [microservices mode]", model, model_variant)

    settings = get_settings()

    # Microservices mode: Call HTTP service
    try:
        async with NERClient(settings) as client:
            result = await client.predict(
                model=model,
                text=text,
                model_variant=model_variant,
                normalize=normalize,
                systems=systems,
                restrict_types=restrict_types,
            )

        # Parse response from NER service
        entities = [Entity(**e) for e in result.get("entities", [])]
        meta = result.get("meta", {})

        # Apply regex-based normalization to fill missing code values
        for entity in entities:
            if not entity.code or entity.code is None:
                normalized = extract_normalized_value(entity.text, entity.type)
                entity.code = normalized

        # Simplify meta to only essential fields for thesis
        simplified_meta = {
            "model": meta.get("model"),
            "inference_time_ms": meta.get("inference_time_ms"),
            "entity_count": len(entities),
        }

        return ExtractResponse(text=text, entities=entities, meta=simplified_meta)

    except NERServiceError as exc:
        logger.exception("NER service call failed for model '%s'", model)
        raise RuntimeError(f"NER service unavailable: {exc}") from exc
