"""
Spanish-English Translation Service for Entity Normalization.

This module provides translation capabilities for converting Spanish clinical
text to English, supporting the entity normalization workflow that matches
Spanish entities to English UMLS concept names.

Architecture Context:
    The translation service is used by :mod:`app.services.normalizer` to
    improve entity-to-concept matching. While the semantic similarity service
    can handle cross-lingual comparison directly, explicit translation can
    improve matching accuracy for certain entity types.

Translation Providers:
    The service supports multiple translation backends with automatic fallback:

    - **Google Translate** (via deep_translator): Primary online provider
    - **MyMemory** (via deep_translator): Fallback online provider
    - **MarianMT** (via transformers): Offline neural machine translation

Provider Selection:
    Controlled by ``TRANSLATOR_PROVIDER`` environment variable:

    - ``chain`` (default): Try Google → MyMemory → MarianMT
    - ``google``: Google Translate only
    - ``deep``: Same as chain
    - ``offline``: MarianMT only (no network required)
    - ``none``/``off``/``false``: Disable translation (return input)

Fault Tolerance:
    The service implements automatic failover:

    1. Track consecutive failures for web providers
    2. After ``TRANSLATOR_FAIL_LIMIT`` failures (default: 3), disable web
    3. Fall back to offline MarianMT translation
    4. If all fail, return original text

Caching:
    Translation results are cached using LRU cache (4096 entries) to
    reduce API calls and improve performance for repeated queries.

Usage:
    >>> from app.services.translator import translate_es_to_en
    >>> english = translate_es_to_en("neumonía")
    >>> print(english)
    'pneumonia'

Configuration:
    Environment variables:

    - ``TRANSLATOR_PROVIDER``: Provider selection (default: "chain")
    - ``TRANSLATOR_FAIL_LIMIT``: Failures before disabling web (default: 3)

See Also:
    - :mod:`app.services.normalizer` for translation usage
    - :mod:`app.services.semantic_sim` for cross-lingual similarity
"""

from __future__ import annotations

import os
from functools import lru_cache

# Configuration from environment
PROVIDER = os.getenv("TRANSLATOR_PROVIDER", "chain").lower()
"""
Translation provider selection.

Values: "chain", "google", "deep", "offline", "none", "off", "false"
"""

FAIL_LIMIT = int(os.getenv("TRANSLATOR_FAIL_LIMIT", "3"))
"""
Maximum consecutive web provider failures before fallback to offline.
"""


class _State:
    """
    Internal state tracking for translation service.

    Tracks web provider health and warning status to implement
    automatic failover behavior.

    Attributes:
        down_web: Whether web providers are disabled due to failures.
        fails_web: Count of consecutive web provider failures.
        warned: Whether a warning has been logged (prevents spam).
    """

    down_web = False
    fails_web = 0
    warned = False


def _warn_once(msg: str):
    """
    Log a warning message only once.

    Prevents log spam by tracking whether a warning has already been issued.

    Args:
        msg: Warning message to log.
    """
    if not _State.warned:
        print(f"[translator] {msg}")
        _State.warned = True


# MarianMT model instances (lazy-loaded)
_MARIAN = None
"""Cached MarianMT model instance."""

_MARIAN_TOK = None
"""Cached MarianMT tokenizer instance."""


def _offline_marian_es_en(text: str) -> str:
    """
    Translate Spanish to English using offline MarianMT model.

    Uses the Helsinki-NLP/opus-mt-es-en model for neural machine translation
    without requiring network access.

    Args:
        text: Spanish text to translate.

    Returns:
        English translation, or original text on failure.

    Model Details:
        - Model: Helsinki-NLP/opus-mt-es-en
        - Framework: Hugging Face Transformers
        - Max length: 256 tokens

    Note:
        First call loads the model (~500MB download on first use).
        Subsequent calls use the cached model instance.
    """
    global _MARIAN, _MARIAN_TOK
    try:
        if _MARIAN is None:
            from transformers import MarianMTModel, MarianTokenizer

            model_name = "Helsinki-NLP/opus-mt-es-en"
            _MARIAN_TOK = MarianTokenizer.from_pretrained(model_name)
            _MARIAN = MarianMTModel.from_pretrained(model_name)

        # Tokenize and translate
        inputs = _MARIAN_TOK(
            [text], return_tensors="pt", padding=True, truncation=True, max_length=256
        )
        gen = _MARIAN.generate(**inputs, max_length=256, num_beams=1)
        return _MARIAN_TOK.decode(gen[0], skip_special_tokens=True)
    except Exception as e:
        _warn_once(f"offline MarianMT failed: {type(e).__name__}: {e}")
        return text


@lru_cache(maxsize=4096)
def translate_es_to_en(text: str) -> str:
    """
    Translate Spanish text to English.

    This is the primary interface for Spanish-English translation.
    It automatically selects the appropriate provider based on
    configuration and availability.

    Args:
        text: Spanish text to translate.

    Returns:
        English translation of the input text.
        Returns original text if translation fails or is disabled.

    Provider Chain:
        When ``TRANSLATOR_PROVIDER="chain"`` (default):

        1. Try Google Translate (via deep_translator)
        2. On failure, try MyMemory Translator
        3. On failure, fall back to offline MarianMT
        4. If all fail, return original text

    Caching:
        Results are cached using LRU cache with 4096 entry limit.
        Cache key is the exact input string.

    Example:
        >>> translate_es_to_en("neumonía adquirida en comunidad")
        'community acquired pneumonia'

        >>> translate_es_to_en("insuficiencia respiratoria")
        'respiratory failure'

    Configuration:
        Set ``TRANSLATOR_PROVIDER`` environment variable:

        - ``chain``: Full fallback chain (default)
        - ``google``: Google Translate only
        - ``offline``: MarianMT only (no network)
        - ``none``: Disable translation

    Note:
        Empty strings are returned unchanged without API calls.
    """
    if not text:
        return text

    # Check if translation is disabled
    if PROVIDER in ("none", "off", "false"):
        return text

    # Use offline provider directly if configured
    if PROVIDER == "offline":
        return _offline_marian_es_en(text)

    # Check if web providers are disabled due to failures
    if _State.down_web and PROVIDER in ("chain", "deep", "google"):
        return _offline_marian_es_en(text)

    try:
        # Chain provider: try Google, then MyMemory
        if PROVIDER in ("chain", "deep"):
            from deep_translator import GoogleTranslator, MyMemoryTranslator

            try:
                return GoogleTranslator(source="es", target="en").translate(text)
            except Exception:
                return MyMemoryTranslator(source="es", target="en").translate(text)

        # Google-only provider
        elif PROVIDER == "google":
            from deep_translator import GoogleTranslator

            return GoogleTranslator(source="es", target="en").translate(text)

        # Unknown provider: return original
        return text

    except Exception as e:
        # Track failures and disable web if threshold exceeded
        _State.fails_web += 1
        if _State.fails_web >= FAIL_LIMIT:
            _State.down_web = True
            _warn_once(
                f"web provider disabled after {FAIL_LIMIT} fails: {type(e).__name__}: {e}"
            )
        # Fall back to offline translation
        return _offline_marian_es_en(text) or text
