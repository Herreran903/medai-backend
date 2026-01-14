"""
Model Registry for Clinical Entity Extraction.

This module provides a centralized registry of available NER extraction models,
implementing the Service Locator pattern for model access throughout the application.

Architecture Context:
    The model registry serves as the single source of truth for available
    extraction models in MedAI. It provides:

    - Centralized model instantiation
    - Consistent model access interface
    - Lazy initialization support (via callable factories)

    The registry is consumed by :mod:`app.services.pipeline` for model
    resolution during extraction requests.

Registered Models:
    - ``lstm``: BiLSTM-CRF model for fast inference on clinical text
    - ``transformer``: Fine-tuned Spanish Transformer (BETO/RoBERTa)
    - ``llm``: Large Language Model extraction (Claude/GPT)

Initialization Behavior:
    Models are instantiated at module import time. This means:

    - Model weights are loaded when the application starts
    - First request latency is reduced
    - Memory is allocated upfront

    For Transformer models with variant selection, the pipeline uses
    a separate cache (:data:`app.services.pipeline._TRANSFORMER_CACHE`)
    to support multiple variants.

Usage:
    >>> from app.services.registry import MODEL_REGISTRY
    >>> extractor = MODEL_REGISTRY["lstm"]
    >>> entities = extractor.predict("FiO2 60%")

    The registry can also be used for model validation:

    >>> model_name = "transformer"
    >>> if model_name in MODEL_REGISTRY:
    ...     extractor = MODEL_REGISTRY[model_name]

Extension:
    To add a new model:

    1. Implement a class with ``predict(text: str) -> List[Dict]`` method
    2. Optionally implement ``meta() -> Dict`` for metadata
    3. Add instance or factory to MODEL_REGISTRY

See Also:
    - :mod:`app.services.pipeline` for model usage in extraction
    - :mod:`app.models.lstm` for LSTM implementation
    - :mod:`app.models.transformer` for Transformer implementation
    - :mod:`app.models.llm` for LLM implementation
"""

from typing import Dict

from app.models.llm import LLMExtractor
from app.models.lstm import LSTMExtractor
from app.models.transformer import TransformerExtractor

MODEL_REGISTRY: Dict[str, object] = {
    "lstm": LSTMExtractor(),
    "transformer": TransformerExtractor(),
    "llm": LLMExtractor(),
}
"""
Central registry of available extraction models.

This dictionary maps model identifiers to extractor instances. Each extractor
must implement the following interface:

- ``predict(text: str) -> List[Dict]``: Extract entities from text
- ``meta() -> Dict`` (optional): Return extractor metadata

Registered Models:
    ``lstm``:
        BiLSTM-CRF model trained on mechanical ventilation clinical notes.
        Provides fast inference with moderate accuracy. Best for high-throughput
        scenarios where latency is critical.
        
        - Implementation: :class:`app.models.lstm.LSTMExtractor`
        - Model files: ``app/models/utils/lstm_model_vmi/``
        
    ``transformer``:
        Fine-tuned Spanish Transformer model (BETO or RoBERTa) for clinical NER.
        Provides best accuracy for entity extraction. Default model for most
        use cases.
        
        - Implementation: :class:`app.models.transformer.TransformerExtractor`
        - Variants: "beto" (default), "roberta"
        - Model source: Hugging Face Hub
        
    ``llm``:
        Large Language Model-based extraction using Claude or GPT APIs.
        Provides highest flexibility and can handle complex entity types.
        Requires API keys for external services.
        
        - Implementation: :class:`app.models.llm.LLMExtractor`
        - Variants: "claude" (default), "gpt", "local" (stub)
        - Requires: ANTHROPIC_API_KEY or OPENAI_API_KEY

Example:
    >>> # Direct registry access
    >>> extractor = MODEL_REGISTRY["lstm"]
    >>> result = extractor.predict("PEEP 8 cmH2O")
    >>> print(result)
    [{'type': 'PEEP', 'text': '8 cmH2O', 'start': 5, 'end': 12, 'score': 0.92}]
    
    >>> # Check model availability
    >>> available_models = list(MODEL_REGISTRY.keys())
    >>> print(available_models)
    ['lstm', 'transformer', 'llm']

Note:
    For Transformer variant selection (BETO vs RoBERTa), use the pipeline
    service with ``model_variant`` parameter rather than direct registry access.
    The registry's transformer entry uses the default variant (BETO).

Warning:
    Modifying the registry at runtime may cause inconsistent behavior.
    The registry is designed to be populated at module import time.
"""
