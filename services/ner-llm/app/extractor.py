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
    - Ventilation response (SAO2, PP, PMES, PM)
    - Anthropometrics (EDAD, PESO, TALLA)
    - Vital signs (TEMP, PA, FC, GLICEMIA)
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
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field

# Logger configuration
logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    fmt = logging.Formatter("[%(levelname)s] %(name)s: %(message)s")
    handler.setFormatter(fmt)
    logger.addHandler(handler)
logger.setLevel(logging.INFO)


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
    score: Optional[float] = Field(0.0, ge=0.0, le=1.0, description="Confianza normalizada [0,1]")
    code: Optional[str] = Field(None, description="Codigo clinico opcional")


class LLMOutput(BaseModel):
    """Structured response containing all extracted clinical entities."""

    model_config = ConfigDict(extra="forbid")

    entities: List[LLMEntity] = Field(default_factory=list)


# ==============================================================================
# Entity Category Mappings
# ==============================================================================

ENTITY_CATEGORIES: Dict[str, List[str]] = {
    "ventilacion": ["MODO", "FIO2", "PEEP", "FR", "VT", "FLUJO", "I_E", "SENS"],
    "respuesta_ventilacion": ["SAO2", "PP", "PMES", "PM"],
    "antropometricos": ["EDAD", "PESO", "TALLA"],
    "signos_vitales": ["TEMP", "PA", "PAS", "PAD", "PAM", "FC", "GLICEMIA", "POSTURA"],
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
# Prompts (imported from prompts.py — single source of truth)
# ==============================================================================

from .prompts import SYSTEM_PROMPT, build_user_prompt, load_few_shot_examples


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


# ==============================================================================
# GPT LLM Extractor
# ==============================================================================


class GPTLLMExtractor:
    """
    Clinical entity extractor using OpenAI GPT (aligned with LLM.ipynb notebook).

    Uses OpenAI Chat Completions with structured outputs (JSON schema mode) to
    return valid JSON that matches :class:`LLMOutput`. Prompts and few-shot
    examples are loaded from ``prompts.py`` and ``examples/`` respectively.

    Architecture:
        Uses ``response_format={"type": "json_schema", ...}`` in strict mode
        when available, and falls back to prompt-based JSON parsing if needed.
        Few-shot prompting is hardcoded (prompt_mode="few_shot").

    Attributes:
        api_key: OpenAI API key for authentication.
        model: GPT model identifier (default: gpt-5.4).
        client: OpenAI client instance (lazy-initialized).
        entity_types: Sorted list of entity type labels from ENTITY_CATEGORIES.
        few_shot_examples: Few-shot examples loaded at init.

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
    ) -> None:
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model
        # Keep provider default temperature to preserve compatibility with
        # models that do not accept custom temperature values.
        self.temperature: Optional[float] = None
        self.client = None

        # Derive entity types from category mapping
        self.entity_types = sorted(
            {label for labels in ENTITY_CATEGORIES.values() for label in labels}
        )

        # Load few-shot examples (empty list if file not found)
        self.few_shot_examples = load_few_shot_examples()

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

    def predict(self, text: str) -> List[Dict[str, Any]]:
        """Extract clinical entities from text using GPT (aligned with notebook)."""
        if not self.client:
            logger.error("OpenAI client not initialized. Returning empty list.")
            return []

        if not text or not text.strip():
            return []

        try:
            schema = LLMOutput.model_json_schema()

            # Build user prompt with few-shot (hardcoded)
            user_prompt = build_user_prompt(
                text,
                self.entity_types,
                schema,
                prompt_mode="few_shot",
                few_shot_examples=self.few_shot_examples,
            )

            # Prepare schema for OpenAI strict mode
            openai_schema = schema.copy()
            _fix_schema_for_openai(openai_schema)

            # API call with structured outputs
            try:
                request_kwargs = {
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt},
                    ],
                    "response_format": {
                        "type": "json_schema",
                        "json_schema": {
                            "name": "ClinicalEntities",
                            "strict": True,
                            "schema": openai_schema,
                        },
                    },
                    "max_completion_tokens": 4000,
                }
                if self.temperature is not None:
                    request_kwargs["temperature"] = self.temperature

                response = self.client.chat.completions.create(**request_kwargs)
                raw = response.choices[0].message.content
                payload = json.loads(raw)
            except Exception as e:
                # Fallback without structured outputs
                logger.warning("Structured outputs failed, using fallback: %s", e)
                fallback_kwargs = {
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt},
                    ],
                    "max_completion_tokens": 4000,
                }
                if self.temperature is not None:
                    fallback_kwargs["temperature"] = self.temperature

                response = self.client.chat.completions.create(**fallback_kwargs)
                raw = response.choices[0].message.content
                raw_json = _extract_json_block(raw)
                payload = json.loads(raw_json)

            # Normalize and validate
            payload = _normalize_llm_payload(payload)
            llm_output = LLMOutput.model_validate(payload)

            # Convert to pipeline-compatible format
            result = []
            for entity in llm_output.entities:
                result.append({
                    "type": entity.type,
                    "text": entity.text,
                    "start": entity.start,
                    "end": entity.end,
                    "code": entity.code,
                })

            logger.info("Extracted %d clinical entities with GPT", len(result))
            return result

        except Exception as e:
            logger.error("Error in GPT extraction: %s", e)
            return []

    def meta(self) -> Dict[str, Any]:
        """
        Return extractor metadata for logging and debugging.

        Returns:
            Dictionary containing extractor configuration.
        """
        return {
            "model": self.model,
            "provider": "openai",
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
        - Temperature: provider default
        - Prompting: few-shot (hardcoded)

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
    ) -> None:
        """
        Initialize the LLM extractor (GPT-only).

        Args:
            api_key: OpenAI API key. Required.
            model: Model identifier (defaults to gpt-5.4).

        Raises:
            ValueError: If OpenAI API key is not provided.
        """
        self.provider = "gpt"

        if not api_key:
            raise ValueError("OpenAI API key required for GPT extractor")

        # Always use GPT with deterministic configuration
        self.extractor = GPTLLMExtractor(
            api_key=api_key,
            model=model,
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
