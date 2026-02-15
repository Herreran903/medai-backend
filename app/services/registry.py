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
    - ``lstm_crf``: NER BiLSTM-CRF Service (port 8005)
    - ``transformer``: NER Transformer Service (port 8001)
    - ``llm``: NER LLM Service (port 8003)
    - ``crf``: NER CRF Service (port 8004)

Registered Models:
    - ``lstm``: BiLSTM model for fast inference on clinical text
    - ``lstm_crf``: BiLSTM-CRF model (Viterbi decoding)
    - ``transformer``: Fine-tuned Spanish RoBERTa (fixed)
    - ``llm``: Large Language Model extraction (GPT fixed)
    - ``crf``: sklearn-crfsuite CRF model (pre-trained)

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
    "lstm_crf": None,
    "transformer": None,
    "llm": None,
    "crf": None,
}
"""
Central registry of valid model identifiers (microservices mode).

This dictionary maps model identifiers to None (no local instances loaded).
Used for validation only - actual inference is performed by NER microservices.

Registered Models:
    ``lstm``:
        BiLSTM model - runs in NER BiLSTM Service (port 8002)

    ``lstm_crf``:
        BiLSTM-CRF model - runs in NER BiLSTM-CRF Service (port 8005)

    ``transformer``:
        Fine-tuned Spanish RoBERTa - runs in NER Transformer Service (port 8001)
        Variants: "roberta" (fixed)

    ``llm``:
        Large Language Model extraction (GPT) - runs in NER LLM Service (port 8003)
        Variants: "gpt" (fixed)

    ``crf``:
        CRF model - runs in NER CRF Service (port 8004)

Example:
    >>> # Validate model name
    >>> if "transformer" in MODEL_REGISTRY:
    ...     # Valid model - route to service
    ...     pass

    >>> # Get all available models
    >>> available_models = list(MODEL_REGISTRY.keys())
    >>> print(available_models)
    ['lstm', 'lstm_crf', 'transformer', 'llm', 'crf']
"""
