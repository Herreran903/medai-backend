"""
LLM-Based Clinical Entity Extraction Module.

This module implements clinical Named Entity Recognition (NER) using Large Language
Models (LLMs) with structured output capabilities. It uses OpenAI GPT as the
active provider.

Architecture Context:
    The LLM extractor serves as an alternative to traditional NER models (LSTM,
    Transformer) when higher accuracy or flexibility is needed. It leverages
    the reasoning capabilities of LLMs to identify clinical entities from
    mechanical ventilation notes.

    The module follows the Strategy pattern with a Facade:

    - :class:`GPTLLMExtractor`: OpenAI GPT implementation (active)
    - :class:`LLMExtractor`: Facade (GPT-only)

Supported Providers:
    - **GPT** (active): Uses GPT with structured outputs (JSON schema mode)

Entity Categories:
    The LLM is prompted to extract entities in these clinical categories:

    - Ventilation configuration (MODO, FIO2, PEEP, FR, VT, etc.)
    - Ventilation response (SAO2)
    - Anthropometrics (EDAD, PESO, TALLA)
    - Vital signs (TEMP, PA, PAM, FC, GLICEMIA, POSTURA)
    - Arterial blood gases (PH, PACO2, PAO2, PAFI)
    - Diagnoses (DX)

Output Format:
    All extractors return entities in a format compatible with the pipeline::

        [
            {
                "type": "FIO2",
                "text": "60%",
                "start": 18,
                "end": 21,
                "code": "60"  # Normalized value
            }
        ]

Usage:
    >>> from app.extractor import LLMExtractor
    >>> extractor = LLMExtractor(api_key="OPENAI_API_KEY")
    >>> entities = extractor.predict("Paciente con FiO2 60%, PEEP 8")
    >>> print(entities)
    [{"type": "FIO2", "text": "60%", ...}, {"type": "PEEP", "text": "8", ...}]

Configuration:
    API keys are read from environment variables:

    - ``OPENAI_API_KEY``: Required for GPT provider (active)

See Also:
    - :mod:`app.services.pipeline` for extraction orchestration
    - :mod:`app.services.registry` for model registration
    - :class:`app.schemas.Entity` for output schema
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from .prompts import SYSTEM_PROMPT, build_user_prompt

# Logger configuration
logger = logging.getLogger(__name__)


# ==============================================================================
# JSON Schema Definitions for Structured Output
# ==============================================================================


class LLMEntity(BaseModel):
    """Pydantic model for LLM-extracted clinical entities (aligned with notebook)."""

    model_config = ConfigDict(extra="ignore")

    type: str = Field(..., description="Tipo de entidad clinica (sin prefijos BIO)")
    text: str = Field(..., description="Texto exacto de la entidad en el input")
    start: int = Field(..., ge=0, description="Offset inicial (char)")
    end: int = Field(..., ge=0, description="Offset final (char)")
    score: Optional[float] = Field(
        0.0, ge=0.0, le=1.0, description="Confianza normalizada [0,1]"
    )
    code: Optional[str] = Field(None, description="Codigo clinico opcional")


class LLMOutput(BaseModel):
    """Structured response containing all extracted clinical entities."""

    model_config = ConfigDict(extra="forbid")

    entities: List[LLMEntity] = Field(default_factory=list)


# ==============================================================================
# Entity Category Mappings
# ==============================================================================

ENTITY_CATEGORIES: Dict[str, List[str]] = {
    "ventilacion": ["MODO", "FIO2", "PEEP", "FR", "VT", "I_E"],
    "respuesta_ventilacion": ["SAO2"],
    "antropometricos": ["EDAD", "PESO", "TALLA"],
    "signos_vitales": ["TEMP", "PA", "PAM", "FC", "GLICEMIA", "POSTURA"],
    "observaciones": ["DX"],
    "gases_arteriales": ["PH", "PACO2", "HCO3", "BE", "PAO2", "PAFI"],
}
"""
Mapping of semantic categories to entity labels.

Used for:
- Organizing extraction prompts by clinical domain
- Validating extracted entity types
- Grouping entities in output
"""


# ==============================================================================
# Helper Functions (aligned with notebook)
# ==============================================================================


def _fix_schema_for_openai(obj: Any) -> None:
    """Adjust schema recursively for OpenAI strict mode requirements."""
    if isinstance(obj, dict):
        if obj.get("type") == "object":
            if "additionalProperties" not in obj:
                obj["additionalProperties"] = False
            if "properties" in obj:
                all_props = list(obj["properties"].keys())
                if "required" not in obj:
                    obj["required"] = all_props
                else:
                    for prop in all_props:
                        if prop not in obj["required"]:
                            obj["required"].append(prop)
        for value in obj.values():
            _fix_schema_for_openai(value)
    elif isinstance(obj, list):
        for item in obj:
            _fix_schema_for_openai(item)


def _extract_json_block(text: str) -> str:
    """Extract valid JSON from a text response (strips markdown fences, etc.)."""
    if text.startswith("```"):
        text = text.replace("```json", "", 1).replace("```", "").strip()
    decoder = json.JSONDecoder()
    for idx, ch in enumerate(text):
        if ch not in "{[":
            continue
        try:
            _, end = decoder.raw_decode(text[idx:])
            return text[idx : idx + end]
        except json.JSONDecodeError:
            continue
    return text


def _normalize_llm_payload(payload: Any) -> dict:
    """Normalize various LLM response shapes into {"entities": [...]}."""
    if isinstance(payload, list):
        return {"entities": payload}
    if isinstance(payload, dict):
        if "entities" in payload:
            return payload
        if any(k in payload for k in ("type", "text", "start", "end", "score", "code")):
            return {"entities": [payload]}
        if "$defs" in payload or "properties" in payload:
            return {"entities": []}
    return {"entities": []}


@dataclass
class ResolvedSpan:
    start: int
    end: int
    method: str


def _find_spans(text: str, query: str, *, case_sensitive: bool) -> List[int]:
    if not query:
        return []
    hay = text if case_sensitive else text.lower()
    needle = query if case_sensitive else query.lower()
    positions: List[int] = []
    start = 0
    while True:
        idx = hay.find(needle, start)
        if idx == -1:
            break
        positions.append(idx)
        start = idx + 1
    return positions


def _normalize_entity_type(entity_type: str, valid_types: set) -> Optional[str]:
    """
    Normalize and validate an LLM-returned entity type label.

    Mirrors ``normalize_entity_type`` in LLM.ipynb:
        - strip whitespace
        - uppercase
        - return None if not in valid_types (entity is dropped by caller)

    Args:
        entity_type: Raw label as returned by the LLM.
        valid_types: Allowed entity labels (typically ``self.entity_types``).

    Returns:
        Canonical label if recognized, else ``None``.
    """
    if not entity_type:
        return None
    norm = entity_type.strip().upper()
    if norm in valid_types:
        return norm
    return None


def _resolve_entity_offsets(
    text: str, entity: LLMEntity
) -> Tuple[Optional[ResolvedSpan], str]:
    """
    Resolve entity offsets against the real input text.

    Resolution strategy mirrors the notebook:
    1. Accept provided offsets if exact (or case-insensitive exact).
    2. Search exact text occurrence(s), optionally using provided offset as hint.
    3. Search case-insensitive occurrence(s).
    4. Fallback to partial contained match.
    5. Fallback to normalized search without spaces.
    """
    raw_text = (entity.text or "").strip()
    if not raw_text:
        return None, "empty_text"

    provided_start = entity.start
    provided_end = entity.end

    # 1. Verificar si el offset proporcionado es exacto
    if provided_start is not None and provided_end is not None:
        if 0 <= provided_start < provided_end <= len(text):
            extracted = text[provided_start:provided_end]
            if extracted == raw_text:
                return (
                    ResolvedSpan(provided_start, provided_end, "provided_exact"),
                    "ok",
                )
            if extracted.lower() == raw_text.lower():
                return (
                    ResolvedSpan(
                        provided_start, provided_end, "provided_case_insensitive"
                    ),
                    "ok",
                )

    # 2. Buscar todas las ocurrencias exactas
    hits = _find_spans(text, raw_text, case_sensitive=True)

    if len(hits) == 1:
        start = hits[0]
        return ResolvedSpan(start, start + len(raw_text), "exact_unique"), "ok"

    if len(hits) > 1:
        if provided_start is not None:
            closest = min(hits, key=lambda h: abs(h - provided_start))
            distance = abs(closest - provided_start)
            if distance <= 50:
                return (
                    ResolvedSpan(closest, closest + len(raw_text), "exact_closest"),
                    "ok",
                )
            for hit in hits:
                if abs(hit - provided_start) <= 100:
                    return (
                        ResolvedSpan(hit, hit + len(raw_text), "exact_nearby"),
                        "ok",
                    )
        start = hits[0]
        return ResolvedSpan(start, start + len(raw_text), "exact_first"), "ok"

    # 3. Buscar sin distinguir mayúsculas/minúsculas
    hits = _find_spans(text, raw_text, case_sensitive=False)

    if len(hits) == 1:
        start = hits[0]
        return (
            ResolvedSpan(start, start + len(raw_text), "case_insensitive_unique"),
            "ok",
        )

    if len(hits) > 1:
        if provided_start is not None:
            closest = min(hits, key=lambda h: abs(h - provided_start))
            distance = abs(closest - provided_start)
            if distance <= 50:
                return (
                    ResolvedSpan(
                        closest, closest + len(raw_text), "case_insensitive_closest"
                    ),
                    "ok",
                )
            for hit in hits:
                if abs(hit - provided_start) <= 100:
                    return (
                        ResolvedSpan(
                            hit, hit + len(raw_text), "case_insensitive_nearby"
                        ),
                        "ok",
                    )
        start = hits[0]
        return (
            ResolvedSpan(start, start + len(raw_text), "case_insensitive_first"),
            "ok",
        )

    # 4. Buscar coincidencia parcial (raw_text contenido en el texto)
    text_lower = text.lower()
    raw_lower = raw_text.lower()

    idx = text_lower.find(raw_lower)
    if idx != -1:
        return ResolvedSpan(idx, idx + len(raw_text), "partial_contained"), "ok"

    # 5. Buscar con tokens normalizados (sin espacios)
    raw_normalized = raw_text.replace(" ", "").lower()
    if len(raw_normalized) >= 2:
        for j in range(len(text) - len(raw_normalized) + 1):
            window = text[j : j + len(raw_normalized) + 5].replace(" ", "").lower()
            if window.startswith(raw_normalized):
                end_idx = j
                chars_matched = 0
                for k in range(j, min(j + len(raw_text) + 10, len(text))):
                    if text[k] != " ":
                        chars_matched += 1
                    if chars_matched >= len(raw_normalized):
                        end_idx = k + 1
                        break
                return ResolvedSpan(j, end_idx, "normalized_match"), "ok"

    return None, "not_found"


# ==============================================================================
# GPT LLM Extractor
# ==============================================================================


class GPTLLMExtractor:
    """
    Clinical entity extractor using OpenAI GPT (aligned with LLM.ipynb notebook).

    Uses OpenAI Chat Completions with structured outputs (JSON schema mode) to
    return valid JSON that matches :class:`LLMOutput`. Prompts are loaded from
    ``prompts.py`` (zero-shot only).

    Architecture:
        Uses ``response_format={"type": "json_schema", ...}`` in strict mode
        when available, and falls back to prompt-based JSON parsing if needed.
        Prompts are zero-shot (matches the canonical experiment
        ``exp-20260424-010034-llm-gpt-structured``).

    Attributes:
        api_key: OpenAI API key for authentication.
        model: GPT model identifier (default: gpt-5.4).
        client: OpenAI client instance (lazy-initialized).
        entity_types: Sorted list of entity type labels from ENTITY_CATEGORIES.

    Example:
        >>> extractor = GPTLLMExtractor()
        >>> entities = extractor.predict("Temperatura 38.5°C, FC 92 lpm")
        >>> print(entities[0]["type"])
        'TEMP'

    Note:
        Requires the ``openai`` package and valid API key.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-5.4",
        *,
        temperature: float = 0.0,
        max_output_tokens: int = 4000,
        max_retries: int = 3,
        retry_sleep: float = 2.0,
    ) -> None:
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model
        self.temperature = float(temperature)
        self.max_output_tokens = int(max_output_tokens)
        self.max_retries = max(1, int(max_retries))
        self.retry_sleep = max(0.0, float(retry_sleep))
        self.client = None
        self.last_raw: Optional[str] = None

        # Derive entity types from category mapping
        self.entity_types = sorted(
            {label for labels in ENTITY_CATEGORIES.values() for label in labels}
        )

        if self.api_key:
            try:
                from openai import OpenAI

                self.client = OpenAI(api_key=self.api_key)
                logger.info("GPT LLM Extractor initialized with model: %s", self.model)
            except ImportError:
                logger.warning(
                    "openai package not installed. Install with: pip install openai"
                )
            except Exception as e:
                logger.error("Error initializing OpenAI client: %s", e)
        else:
            logger.warning(
                "OPENAI_API_KEY not configured. Extractor will not function."
            )

    def _build_prompt_args(
        self, text: str, include_schema: bool, schema: dict
    ) -> Dict[str, Any]:
        return {
            "text": text,
            "entity_types": self.entity_types,
            "schema": schema,
            "include_schema": include_schema,
        }

    def _extract_payload_once(self, text: str, schema: dict) -> Any:
        # Deep copy avoids mutating the canonical schema structure.
        openai_schema = json.loads(json.dumps(schema))
        _fix_schema_for_openai(openai_schema)

        try:
            user_prompt = build_user_prompt(
                **self._build_prompt_args(
                    text=text, include_schema=False, schema=schema
                )
            )
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=self.temperature,
                max_completion_tokens=self.max_output_tokens,
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "ClinicalEntities",
                        "strict": True,
                        "schema": openai_schema,
                    },
                },
            )
            raw = response.choices[0].message.content or ""
            self.last_raw = raw
            return json.loads(raw)
        except Exception as exc:
            # Notebook behavior: only fallback to prompt-only mode when
            # response_format is unsupported for the selected model/provider.
            if "response_format" not in str(exc):
                raise

            logger.warning(
                "Structured outputs unavailable, using prompt-only fallback: %s", exc
            )
            user_prompt = build_user_prompt(
                **self._build_prompt_args(text=text, include_schema=True, schema=schema)
            )
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=self.temperature,
                max_completion_tokens=self.max_output_tokens,
            )
            raw = response.choices[0].message.content or ""
            self.last_raw = raw
            return json.loads(_extract_json_block(raw))

    def predict(self, text: str) -> List[Dict[str, Any]]:
        """Extract clinical entities from text using GPT (aligned with notebook)."""
        if not self.client:
            logger.error("OpenAI client not initialized. Returning empty list.")
            return []

        if not text or not text.strip():
            return []

        schema = LLMOutput.model_json_schema()
        last_exc: Optional[Exception] = None

        for attempt in range(1, self.max_retries + 1):
            try:
                payload = self._extract_payload_once(text=text, schema=schema)
                payload = _normalize_llm_payload(payload)
                llm_output = LLMOutput.model_validate(payload)

                result = []
                dropped_offset = 0
                dropped_type = 0
                valid_types = set(self.entity_types)
                for entity in llm_output.entities:
                    canon_type = _normalize_entity_type(entity.type, valid_types)
                    if canon_type is None:
                        dropped_type += 1
                        logger.warning(
                            "Dropping entity with unknown/invalid type %r (valid: %s)",
                            entity.type,
                            sorted(valid_types),
                        )
                        continue

                    resolved, status = _resolve_entity_offsets(text, entity)
                    if not resolved:
                        dropped_offset += 1
                        logger.warning(
                            "Dropping entity due to unresolved offsets (%s): %s",
                            status,
                            entity.model_dump(),
                        )
                        continue

                    span_text = text[resolved.start : resolved.end]
                    result.append(
                        {
                            "type": canon_type,
                            "text": span_text,
                            "start": resolved.start,
                            "end": resolved.end,
                            "code": entity.code,
                        }
                    )

                if dropped_type:
                    logger.warning(
                        "Dropped %d entities with invalid type", dropped_type
                    )
                if dropped_offset:
                    logger.warning(
                        "Dropped %d entities after offset reconciliation",
                        dropped_offset,
                    )
                logger.info("Extracted %d clinical entities with GPT", len(result))
                return result
            except (ValidationError, json.JSONDecodeError, KeyError) as exc:
                last_exc = exc
                raw_preview = (self.last_raw or "")[:300].replace("\n", " ")
                logger.warning(
                    "LLM payload validation failed (attempt %d/%d): %s | raw_preview=%s",
                    attempt,
                    self.max_retries,
                    exc,
                    raw_preview or "N/A",
                )
            except Exception as exc:
                last_exc = exc
                logger.warning(
                    "LLM extraction attempt failed (%d/%d): %s",
                    attempt,
                    self.max_retries,
                    exc,
                )

            if attempt < self.max_retries and self.retry_sleep > 0:
                time.sleep(self.retry_sleep)

        raw_preview = (self.last_raw or "")[:400].replace("\n", " ")
        logger.error(
            "Error in GPT extraction after %d attempts: %s | last_raw_preview=%s",
            self.max_retries,
            last_exc,
            raw_preview or "N/A",
        )
        return []

    def meta(self) -> Dict[str, Any]:
        """
        Return extractor metadata for logging and debugging.

        Returns:
            Dictionary containing extractor configuration.
        """
        return {
            "model": self.model,
            "provider": "gpt",
            "temperature": self.temperature,
            "max_output_tokens": self.max_output_tokens,
        }


# ==============================================================================
# LLM Extractor Facade
# ==============================================================================


class LLMExtractor:
    """
    Facade class for LLM-based clinical entity extraction.

    This facade runs GPT-only.

    Configuration:
        - Provider: GPT (forced)
        - Model: gpt-5.4 (default)
        - Temperature: 0.0 (default)
        - Prompting: zero-shot (matches canonical experiment)

    Attributes:
        provider: Always "gpt" (fixed).
        extractor: GPTLLMExtractor instance.

    Example:
        >>> extractor = LLMExtractor()
        >>> entities = extractor.predict("FiO2 60%, PEEP 8")

    See Also:
        - :class:`GPTLLMExtractor` for implementation details
    """

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-5.4",
        *,
        temperature: float = 0.0,
        max_output_tokens: int = 4000,
        max_retries: int = 3,
        retry_sleep: float = 2.0,
    ) -> None:
        """
        Initialize the LLM extractor (GPT-only).

        Args:
            api_key: OpenAI API key. Required.
            model: Model identifier (defaults to gpt-5.4).

        Raises:
            ValueError: If OpenAI API key is not provided.
        """
        if not api_key:
            raise ValueError("OpenAI API key required for GPT extractor")

        # Always use GPT with deterministic configuration (zero-shot)
        self.extractor = GPTLLMExtractor(
            api_key=api_key,
            model=model,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            max_retries=max_retries,
            retry_sleep=retry_sleep,
        )

        logger.info("LLMExtractor initialized with provider: gpt (model: %s)", model)

    def predict(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract clinical entities from text.

        Delegates to the underlying provider-specific extractor.

        Args:
            text: Clinical note text to process.

        Returns:
            List of entity dictionaries in pipeline-compatible format.
        """
        return self.extractor.predict(text)

    def meta(self) -> Dict[str, Any]:
        """
        Return metadata from the underlying extractor.

        Returns:
            Dictionary with extractor metadata.
        """
        return self.extractor.meta()
