"""
Clinical Entity Extraction Pipeline Service.

This module provides the high-level orchestration layer for clinical Named Entity
Recognition (NER), coordinating model selection, extraction, validation, and
value-code enrichment.

Architecture Context:
    The pipeline service is the central coordination point for all extraction
    operations in MedAI. It abstracts the complexity of:

    - Model selection and initialization
    - Transformer variant resolution (RoBERTa fixed)
    - Entity validation and standardization
    - Regex-based value normalization for the ``code`` field

    All API endpoints delegate to this service via :func:`extract_from_text`.

Pipeline Flow:
    Input Text -> Model Selection -> Extraction -> Validation -> Value Coding -> ExtractResponse

Microservices Routing:
    The gateway does not load model weights locally. It validates model names
    against :data:`app.services.registry.MODEL_REGISTRY` and routes requests
    to dedicated NER services via HTTP.

Usage:
    >>> from app.services.pipeline import extract_from_text
    >>> result = extract_from_text(
    ...     "Paciente con FiO2 60%, PEEP 8 cmH2O",
    ...     model="transformer",
    ...     model_variant="roberta"
    ... )
    >>> print(result.entities[0].type)
    'FIO2'

See Also:
    - :mod:`app.routers.extract` for API endpoint integration
    - :mod:`app.services.registry` for model registration
"""

from __future__ import annotations

import logging
from typing import Optional

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
) -> ExtractResponse:
    """
    Extract clinical entities from text using the specified model.

    This is the primary entry point for all extraction operations in MedAI.
    It coordinates model selection, extraction, validation, and value-code
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

            - For ``transformer``: "roberta" (fixed)
            - For ``llm``: "gpt" (fixed)
            - Ignored for ``lstm``, ``lstm_crf``, and ``crf``

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
        2. **Model Resolution**: Validate model against registry
        3. **Extraction**: Route request to NER microservice via HTTP
        4. **Entity Validation**: Validate spans and construct Entity objects
        5. **Value Coding**: Regex value normalization for missing ``code`` fields
        6. **Response Construction**: Build ExtractResponse with metadata

    Example:
        >>> # Basic extraction with transformer
        >>> result = extract_from_text(
        ...     "FiO2 60%, PEEP 8 cmH2O",
        ...     model="transformer"
        ... )
        >>> len(result.entities)
        2
    Note:
        Model lifecycle is handled by the NER microservices. The gateway only
        validates model identifiers and orchestrates HTTP calls.
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
            "provider": meta.get("provider"),
            "inference_time_ms": meta.get("inference_time_ms"),
            "entity_count": len(entities),
        }

        return ExtractResponse(text=text, entities=entities, meta=simplified_meta)

    except NERServiceError as exc:
        logger.exception("NER service call failed for model '%s'", model)
        raise RuntimeError(f"NER service unavailable: {exc}") from exc

