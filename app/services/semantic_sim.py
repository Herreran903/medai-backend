"""
Semantic Similarity Service for Multilingual Text Comparison.

This module provides semantic similarity computation using multilingual
sentence embeddings, enabling cross-lingual comparison between Spanish
clinical text and English medical terminology.

Architecture Context:
    The semantic similarity service is a core component of the entity
    normalization pipeline. It enables matching Spanish clinical entity
    text to English UMLS concept names without explicit translation.

    The service uses the ``intfloat/multilingual-e5-base`` model, which
    provides high-quality multilingual embeddings suitable for:

    - Cross-lingual semantic search
    - Text similarity computation
    - Concept matching across languages

Model Details:
    - **Model**: intfloat/multilingual-e5-base
    - **Embedding Dimension**: 768
    - **Languages**: 100+ languages including Spanish and English
    - **Device**: CPU (configurable)

Caching Strategy:
    - Model is loaded lazily on first use
    - Individual text embeddings are cached (LRU, 8192 entries)
    - Embeddings are normalized for efficient cosine similarity

Performance Considerations:
    - First call incurs model loading latency (~2-5 seconds)
    - Subsequent calls benefit from embedding cache
    - Batch embedding available for multiple texts

Usage:
    >>> from app.services.semantic_sim import sim_texts
    >>> score = sim_texts("neumonía", "pneumonia")
    >>> print(f"Similarity: {score:.3f}")
    Similarity: 0.847

    For batch processing:

    >>> from app.services.semantic_sim import _embed_many, cosine_sim
    >>> embeddings = _embed_many(["text1", "text2", "text3"])

See Also:
    - :mod:`app.services.normalizer` for usage in entity normalization
    - :mod:`app.services.translator` for explicit translation alternative
"""

from __future__ import annotations

from functools import lru_cache
from typing import Iterable, List

import numpy as np

# Global model instance (lazy-loaded)
_MODEL = None
"""
Cached SentenceTransformer model instance.

Loaded on first use via :func:`_load_model`. Persists for application lifetime.
"""


def _load_model():
    """
    Load the multilingual sentence embedding model.

    Implements lazy loading pattern - the model is loaded only when
    first needed and cached for subsequent calls.

    Returns:
        SentenceTransformer: Loaded model instance.

    Model Configuration:
        - Name: intfloat/multilingual-e5-base
        - Device: CPU (for consistent behavior across environments)
        - Normalization: Enabled for cosine similarity optimization

    Note:
        First call may take 2-5 seconds for model download/loading.
        Subsequent calls return the cached instance immediately.
    """
    global _MODEL
    if _MODEL is None:
        from sentence_transformers import SentenceTransformer

        name = "intfloat/multilingual-e5-base"
        _MODEL = SentenceTransformer(name, device="cpu")
    return _MODEL


@lru_cache(maxsize=8192)
def _embed_one(text: str) -> np.ndarray:
    """
    Generate embedding for a single text string.

    Computes a normalized embedding vector for the input text using
    the multilingual sentence transformer model.

    Args:
        text: Input text to embed. Empty strings return zero vector.

    Returns:
        np.ndarray: Normalized embedding vector (768 dimensions).

    Caching:
        Results are cached using LRU cache with 8192 entry limit.
        Cache key is the exact input string.

    Example:
        >>> embedding = _embed_one("neumonía")
        >>> embedding.shape
        (768,)
        >>> np.linalg.norm(embedding)  # Normalized
        1.0
    """
    m = _load_model()
    if not text:
        return np.zeros(m.get_sentence_embedding_dimension(), dtype=np.float32)
    prompt = text
    return m.encode(prompt, normalize_embeddings=True)


def _embed_many(texts: Iterable[str]) -> np.ndarray:
    """
    Generate embeddings for multiple texts in batch.

    More efficient than calling :func:`_embed_one` repeatedly for
    large numbers of texts, as it leverages batch processing.

    Args:
        texts: Iterable of text strings to embed.

    Returns:
        np.ndarray: Matrix of embeddings, shape (n_texts, 768).

    Note:
        This function does not use caching. For repeated queries,
        use :func:`_embed_one` which caches individual results.

    Example:
        >>> embeddings = _embed_many(["text1", "text2", "text3"])
        >>> embeddings.shape
        (3, 768)
    """
    m = _load_model()
    arr: List[str] = [t or "" for t in texts]
    return m.encode(arr, normalize_embeddings=True)


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """
    Compute cosine similarity between two vectors.

    For normalized vectors (as produced by this module), cosine similarity
    reduces to the dot product.

    Args:
        a: First embedding vector.
        b: Second embedding vector.

    Returns:
        float: Cosine similarity score in range [-1.0, 1.0].
            Returns 0.0 if either vector is None.

    Note:
        Since embeddings from this module are normalized, the result
        is equivalent to ``np.dot(a, b)``.

    Example:
        >>> a = _embed_one("pneumonia")
        >>> b = _embed_one("lung infection")
        >>> score = cosine_sim(a, b)
        >>> print(f"Similarity: {score:.3f}")
        Similarity: 0.723
    """
    if a is None or b is None:
        return 0.0
    return float(np.clip(np.dot(a, b), -1.0, 1.0))


def sim_texts(a: str, b: str) -> float:
    """
    Compute semantic similarity between two text strings.

    This is the primary interface for text similarity computation.
    It handles embedding generation and similarity calculation in
    a single call.

    Args:
        a: First text string (typically Spanish clinical text).
        b: Second text string (typically English concept name).

    Returns:
        float: Semantic similarity score in range [-1.0, 1.0].
            Higher values indicate greater semantic similarity.

    Algorithm:
        1. Generate normalized embedding for text ``a``
        2. Generate normalized embedding for text ``b``
        3. Compute cosine similarity (dot product for normalized vectors)

    Example:
        >>> # Cross-lingual similarity
        >>> score = sim_texts("neumonía", "pneumonia")
        >>> print(f"Score: {score:.3f}")
        Score: 0.847

        >>> # Same-language similarity
        >>> score = sim_texts("heart attack", "myocardial infarction")
        >>> print(f"Score: {score:.3f}")
        Score: 0.812

    Use Cases:
        - Entity normalization: Match Spanish entities to English concepts
        - Duplicate detection: Find semantically similar notes
        - Concept search: Find related medical terms

    Note:
        The multilingual model handles cross-lingual comparison natively,
        so explicit translation is not required (though it may improve
        results in some cases).
    """
    va = _embed_one(a)
    vb = _embed_one(b)
    return cosine_sim(va, vb)
