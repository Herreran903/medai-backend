"""
Model Registry for Clinical Entity Extraction (Microservices Mode).

This module provides a centralized registry of available NER extraction models
for validation purposes. In microservices mode, models are NOT loaded locally;
instead, the registry serves as a lookup table for valid model identifiers.

Architecture Context:
    In microservices architecture, the gateway does NOT load model instances.
    This registry is used only for:

    - Model name validation (checking if a model identifier is valid)
    - Service discovery (mapping model names to service URLs)

    Actual model instances run in separate NER microservices:
    - ``lstm``: NER BiLSTM Service (port 8002)
    - ``transformer``: NER Transformer Service (port 8001)
    - ``llm``: NER LLM Service (port 8003)

Registered Models:
    - ``lstm``: BiLSTM-CRF model for fast inference on clinical text
    - ``transformer``: Fine-tuned Spanish RoBERTa (fixed)
    - ``llm``: Large Language Model extraction (GPT fixed)

Usage:
    >>> from app.services.registry import MODEL_REGISTRY
    >>> model_name = "transformer"
    >>> if model_name in MODEL_REGISTRY:
    ...     # Valid model, proceed with HTTP request to service
    ...     pass

See Also:
    - :mod:`app.services.pipeline` for extraction orchestration
    - :mod:`app.services.ner_client` for HTTP communication with services
    - :mod:`app.config` for service URL configuration
"""

from typing import Dict

MODEL_REGISTRY: Dict[str, None] = {
    "lstm": None,
    "transformer": None,
    "llm": None,
}
"""
Central registry of valid model identifiers (microservices mode).

This dictionary maps model identifiers to None (no local instances loaded).
Used for validation only - actual inference is performed by NER microservices.

Registered Models:
    ``lstm``:
        BiLSTM-CRF model - runs in NER BiLSTM Service (port 8002)

    ``transformer``:
        Fine-tuned Spanish RoBERTa - runs in NER Transformer Service (port 8001)
        Variants: "roberta" (fixed)

    ``llm``:
        Large Language Model extraction (GPT) - runs in NER LLM Service (port 8003)
        Variants: "gpt" (fixed)

Example:
    >>> # Validate model name
    >>> if "transformer" in MODEL_REGISTRY:
    ...     # Valid model - route to service
    ...     pass

    >>> # Get all available models
    >>> available_models = list(MODEL_REGISTRY.keys())
    >>> print(available_models)
    ['lstm', 'transformer', 'llm']
"""
