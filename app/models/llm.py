"""
LLM-Based Clinical Entity Extraction Module.

This module implements clinical Named Entity Recognition (NER) using Large Language
Models (LLMs) with structured output capabilities. It provides extractors for
Claude (Anthropic) and GPT (OpenAI) with JSON schema validation.

Architecture Context:
    The LLM extractor serves as an alternative to traditional NER models (LSTM,
    Transformer) when higher accuracy or flexibility is needed. It leverages
    the reasoning capabilities of LLMs to identify clinical entities from
    mechanical ventilation notes.

    The module follows the Strategy pattern with a Facade:

    - :class:`ClaudeLLMExtractor`: Anthropic Claude implementation
    - :class:`GPTLLMExtractor`: OpenAI GPT implementation
    - :class:`LocalLLMExtractor`: Stub for future local model support
    - :class:`LLMExtractor`: Facade that selects the appropriate extractor

Supported Providers:
    - **Claude** (default): Uses Claude Sonnet 4.5 with system prompt for JSON output
    - **GPT**: Uses GPT-4o with native structured outputs (JSON schema mode)
    - **Local**: Placeholder for Ollama/local model integration

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
                "score": 0.95,
                "code": "60"  # Normalized value
            }
        ]

Usage:
    >>> from app.models.llm import LLMExtractor
    >>> extractor = LLMExtractor(provider="claude")
    >>> entities = extractor.predict("Paciente con FiO2 60%, PEEP 8")
    >>> print(entities)
    [{"type": "FIO2", "text": "60%", ...}, {"type": "PEEP", "text": "8", ...}]

Configuration:
    API keys are read from environment variables:

    - ``ANTHROPIC_API_KEY``: Required for Claude provider
    - ``OPENAI_API_KEY``: Required for GPT provider

See Also:
    - :mod:`app.services.pipeline` for extraction orchestration
    - :mod:`app.services.registry` for model registration
    - :class:`app.schemas.Entity` for output schema
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

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


class ClinicalEntity(BaseModel):
    """
    Pydantic model for LLM-extracted clinical entities.

    This schema is used to validate and structure the JSON output from LLMs,
    ensuring consistent entity format regardless of the underlying model.

    Attributes:
        label: Entity type classification matching MedAI taxonomy.
        value_raw: Exact text as it appears in the source document.
        value_norm: Normalized numeric value with standard units.
        units: Standardized unit of measurement.
        confidence: Model confidence score (0.0 to 1.0).
        category: Semantic category for entity grouping.

    Note:
        This schema is converted to JSON Schema for LLM prompting and
        response validation.
    """

    label: str = Field(
        ...,
        description="Clinical entity type (e.g., FiO2, PEEP, TEMP, DX).",
    )
    value_raw: str = Field(
        ...,
        description="Exact value as it appears in the source text.",
    )
    value_norm: str = Field(
        default="",
        description="Normalized value in standard numeric format.",
    )
    units: str = Field(
        default="",
        description="Standardized unit of measurement (%, cmH2O, mL, etc.).",
    )
    confidence: float = Field(
        default=0.95,
        ge=0.0,
        le=1.0,
        description="Extraction confidence score (0.0-1.0).",
    )
    category: str = Field(
        ...,
        description="Semantic category (ventilacion, signos_vitales, etc.).",
    )


class ClinicalEntitiesResponse(BaseModel):
    """
    Structured response containing all extracted clinical entities.

    This is the top-level schema for LLM JSON output validation.

    Attributes:
        entities: List of extracted clinical entities.
    """

    entities: List[ClinicalEntity] = Field(
        default_factory=list,
        description="List of clinical entities extracted from the text.",
    )


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

LABEL_TO_CATEGORY: Dict[str, str] = {}
"""Reverse mapping from entity label to category."""

for category, labels in ENTITY_CATEGORIES.items():
    for label in labels:
        LABEL_TO_CATEGORY[label] = category


# ==============================================================================
# Extraction Prompt Template
# ==============================================================================

EXTRACTION_PROMPT = """Eres un experto en extracción de entidades clínicas de notas médicas de pacientes en ventilación mecánica.

Tu tarea es extraer ÚNICAMENTE las entidades clínicas mencionadas en el texto, organizadas por las siguientes categorías:

**CONFIGURACIÓN DE VENTILACIÓN:**
- MODO: Modo de operación del ventilador (AC/VC, VC+, PC, SIMV, PSV, CPAP, etc.)
- FIO2: Fracción inspirada de oxígeno (%)
- PEEP: Presión positiva al final de la espiración (cmH2O)
- FR: Frecuencia respiratoria (rpm o respiraciones/min) - pueden ser 2 valores como 14/20
- VT: Volumen tidal o corriente (mL) - pueden ser 2 valores como 380/397
- FLUJO: Flujo inspiratorio (L/min)
- I_E: Relación inspiración:espiración (I:E)
- SENS: Sensibilidad del trigger

**RESPUESTA A LA VENTILACIÓN:**
- SAO2: Saturación de oxígeno (SaO2 %)
- PP: Presión pico de vía aérea (cmH2O)
- PMES: Presión meseta (cmH2O)
- PM: Poder mecánico del ventilador

**ANTROPOMÉTRICOS:**
- EDAD: Edad del paciente (años) - solo número
- PESO: Peso corporal (kg) - solo número
- TALLA: Estatura (m) - solo número

**SIGNOS VITALES:**
- TEMP: Temperatura corporal (°C)
- PA: Presión arterial completa (ej: 120/80 mmHg)
- PAS: Presión arterial sistólica (mmHg)
- PAD: Presión arterial diastólica (mmHg)
- PAM: Presión arterial media / TAM (mmHg)
- FC: Frecuencia cardiaca (lpm o latidos/min)
- GLICEMIA: Glucosa en sangre / Glucometría (mg/dL)
- POSTURA: Postura o posicionamiento del paciente

**OBSERVACIONES:**
- DX: Diagnósticos / impresión diagnóstica

**GASES ARTERIALES:**
- PH: pH arterial
- PACO2: Presión parcial de CO2 (PaCO2 mmHg)
- HCO3: Bicarbonato (HCO3⁻ mEq/L)
- BE: Exceso de base (Base excess)
- PAO2: Presión parcial de O2 (PaO2 mmHg)
- PAFI: Relación PaO2/FiO2 (PaFi)

**INSTRUCCIONES CRÍTICAS:**
1. Extrae SOLO las entidades que aparecen explícitamente en el texto
2. Para value_raw: copia el valor exacto como aparece en el texto
3. Para value_norm: normaliza a formato numérico estándar (ej: "60%" -> "60", "420 ml" -> "420")
4. Para units: usa unidades estándar (%, cmH2O, mL, L/min, kg, cm, °C, mmHg, mg/dL, lpm, mEq/L)
5. Asigna confidence entre 0.8-1.0 según la claridad de la mención
6. NO inventes valores que no estén en el texto
7. NO incluyas texto explicativo, SOLO el JSON

Texto a analizar:
{text}

Extrae las entidades en formato JSON estructurado."""
"""
Prompt template for clinical entity extraction.

The prompt instructs the LLM to:
1. Extract only explicitly mentioned entities
2. Preserve raw text values
3. Normalize values to standard formats
4. Assign confidence scores
5. Return structured JSON output

The ``{text}`` placeholder is replaced with the input clinical note.
"""


# ==============================================================================
# Claude LLM Extractor
# ==============================================================================


class ClaudeLLMExtractor:
    """
    Clinical entity extractor using Anthropic Claude.

    This extractor uses Claude Sonnet 4.5 with a system prompt that enforces
    JSON output conforming to the :class:`ClinicalEntitiesResponse` schema.

    Architecture:
        Claude does not natively support JSON schema mode in the same way as
        OpenAI, so this implementation uses a detailed system prompt with
        the JSON schema embedded to guide output format.

    Attributes:
        api_key: Anthropic API key for authentication.
        model: Claude model identifier (default: claude-sonnet-4-20250514).
        client: Anthropic client instance (lazy-initialized).

    Example:
        >>> extractor = ClaudeLLMExtractor()
        >>> entities = extractor.predict("FiO2 60%, PEEP 8 cmH2O")
        >>> print(entities[0]["type"])
        'FIO2'

    Note:
        Requires the ``anthropic`` package and valid API key.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "claude-sonnet-4-20250514",
    ) -> None:
        """
        Initialize the Claude extractor.

        Args:
            api_key: Anthropic API key. If not provided, reads from
                ``ANTHROPIC_API_KEY`` environment variable.
            model: Claude model identifier. Defaults to Claude Sonnet 4.5.

        Raises:
            Warning logged if API key is not configured.
        """
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        self.model = model
        self.client = None

        if self.api_key:
            try:
                import anthropic

                self.client = anthropic.Anthropic(api_key=self.api_key)
                logger.info(
                    "Claude LLM Extractor initialized with model: %s", self.model
                )
            except ImportError:
                logger.warning(
                    "anthropic package not installed. Install with: pip install anthropic"
                )
            except Exception as e:
                logger.error("Error initializing Anthropic client: %s", e)
        else:
            logger.warning(
                "ANTHROPIC_API_KEY not configured. Extractor will not function."
            )

    def _calculate_offsets(self, text: str, entity_text: str) -> tuple[int, int]:
        """
        Calculate character offsets for an entity in the source text.

        Performs exact match first, then falls back to case-insensitive
        regex search if exact match fails.

        Args:
            text: Complete source text.
            entity_text: Entity text span to locate.

        Returns:
            Tuple of (start, end) character positions, or (-1, -1) if not found.
        """
        start = text.find(entity_text)
        if start == -1:
            pattern = re.escape(entity_text)
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                start = match.start()
                end = match.end()
            else:
                return (-1, -1)
        else:
            end = start + len(entity_text)

        return (start, end)

    def predict(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract clinical entities from text using Claude.

        Args:
            text: Clinical note text to process.

        Returns:
            List of entity dictionaries compatible with the pipeline format::

                [
                    {
                        "type": str,      # Entity label
                        "text": str,      # Raw text span
                        "start": int,     # Character offset start
                        "end": int,       # Character offset end
                        "score": float,   # Confidence score
                        "code": str       # Normalized value
                    }
                ]

        Note:
            Returns empty list if client is not initialized or on error.
        """
        if not self.client:
            logger.error("Claude client not initialized. Returning empty list.")
            return []

        if not text or not text.strip():
            return []

        try:
            prompt = EXTRACTION_PROMPT.format(text=text)
            schema = ClinicalEntitiesResponse.model_json_schema()
            response = None
            content = None

            # Use system prompt with JSON schema (Claude SDK doesn't support response_format)
            try:
                system_msg = (
                    "Return exclusively valid JSON that strictly conforms to this Pydantic JSON Schema. "
                    "Do not include ANY text outside the JSON:\n" + json.dumps(schema)
                )
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=4096,
                    temperature=0.0,
                    system=system_msg,
                    messages=[{"role": "user", "content": prompt}],
                )
                content = response.content[0].text
            except Exception as e:
                logger.error("Error calling Claude API: %s", e)
                return []

            # Validate response against schema
            try:
                entities_response = ClinicalEntitiesResponse.model_validate_json(
                    content
                )
            except Exception:
                try:
                    # Attempt to extract JSON from response text
                    m = re.search(r"\{.*\}|\[.*\]", content, re.DOTALL)
                    if not m:
                        raise ValueError("No JSON found in model output")
                    raw_json = m.group(0)
                    if raw_json.lstrip().startswith("["):
                        wrapped = json.dumps({"entities": json.loads(raw_json)})
                        entities_response = (
                            ClinicalEntitiesResponse.model_validate_json(wrapped)
                        )
                    else:
                        entities_response = (
                            ClinicalEntitiesResponse.model_validate_json(raw_json)
                        )
                except Exception as e2:
                    logger.error("Error validating Claude JSON output: %s", e2)
                    return []

            # Convert to pipeline-compatible format
            result = []
            for entity in entities_response.entities:
                start, end = self._calculate_offsets(text, entity.value_raw)

                entity_dict = {
                    "type": entity.label,
                    "text": entity.value_raw,
                    "start": start if start >= 0 else None,
                    "end": end if end >= 0 else None,
                    "score": entity.confidence,
                    "code": entity.value_norm,
                }
                result.append(entity_dict)

            logger.info("Extracted %d clinical entities with Claude", len(result))
            return result

        except Exception as e:
            logger.error("Error in Claude extraction: %s", e)
            return []

    def meta(self) -> Dict[str, Any]:
        """
        Return extractor metadata for logging and debugging.

        Returns:
            Dictionary containing extractor configuration and capabilities.
        """
        return {
            "extractor": "claude",
            "model": self.model,
            "provider": "anthropic",
            "structured_outputs": False,
            "categories": list(ENTITY_CATEGORIES.keys()),
            "total_labels": sum(len(labels) for labels in ENTITY_CATEGORIES.values()),
        }


# ==============================================================================
# GPT LLM Extractor
# ==============================================================================


class GPTLLMExtractor:
    """
    Clinical entity extractor using OpenAI GPT.

    This extractor uses GPT-4o with native structured outputs (JSON schema mode)
    for guaranteed valid JSON responses.

    Architecture:
        GPT-4o supports ``response_format`` with JSON schema validation,
        ensuring the output always conforms to the expected structure.
        This provides more reliable parsing than prompt-based approaches.

    Attributes:
        api_key: OpenAI API key for authentication.
        model: GPT model identifier (default: gpt-4o-2024-08-06).
        client: OpenAI client instance (lazy-initialized).

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
        model: str = "gpt-4o-2024-08-06",
    ) -> None:
        """
        Initialize the GPT extractor.

        Args:
            api_key: OpenAI API key. If not provided, reads from
                ``OPENAI_API_KEY`` environment variable.
            model: GPT model identifier. Defaults to GPT-4o.

        Raises:
            Warning logged if API key is not configured.
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model
        self.client = None

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

    def _calculate_offsets(self, text: str, entity_text: str) -> tuple[int, int]:
        """
        Calculate character offsets for an entity in the source text.

        Args:
            text: Complete source text.
            entity_text: Entity text span to locate.

        Returns:
            Tuple of (start, end) character positions, or (-1, -1) if not found.
        """
        start = text.find(entity_text)
        if start == -1:
            pattern = re.escape(entity_text)
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                start = match.start()
                end = match.end()
            else:
                return (-1, -1)
        else:
            end = start + len(entity_text)

        return (start, end)

    def predict(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract clinical entities from text using GPT.

        Uses OpenAI's structured outputs feature for guaranteed JSON schema
        compliance.

        Args:
            text: Clinical note text to process.

        Returns:
            List of entity dictionaries compatible with the pipeline format.

        Note:
            Falls back to prompt-based JSON if structured outputs fail.
        """
        if not self.client:
            logger.error("OpenAI client not initialized. Returning empty list.")
            return []

        if not text or not text.strip():
            return []

        try:
            prompt = EXTRACTION_PROMPT.format(text=text)
            schema = ClinicalEntitiesResponse.model_json_schema()

            # Prepare schema for OpenAI strict mode
            openai_schema = schema.copy()

            def fix_schema_for_openai(obj):
                """Adjust schema for OpenAI strict mode requirements."""
                if isinstance(obj, dict):
                    if "type" in obj and obj["type"] == "object":
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
                    for key, value in obj.items():
                        fix_schema_for_openai(value)
                elif isinstance(obj, list):
                    for item in obj:
                        fix_schema_for_openai(item)

            fix_schema_for_openai(openai_schema)

            # Use structured outputs
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {
                            "role": "system",
                            "content": "You are an expert in clinical entity extraction. Return ONLY valid JSON.",
                        },
                        {"role": "user", "content": prompt},
                    ],
                    response_format={
                        "type": "json_schema",
                        "json_schema": {
                            "name": "clinical_entities_extraction",
                            "strict": True,
                            "schema": openai_schema,
                        },
                    },
                    temperature=0.0,
                    max_tokens=4096,
                )
                content = response.choices[0].message.content
            except Exception as e:
                # Fallback without structured outputs
                logger.warning("Structured outputs failed, using fallback: %s", e)
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {
                            "role": "system",
                            "content": f"You are an expert in clinical entity extraction. Return ONLY valid JSON conforming to this schema:\n{json.dumps(schema)}",
                        },
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.0,
                    max_tokens=4096,
                )
                content = response.choices[0].message.content

            # Validate response
            try:
                entities_response = ClinicalEntitiesResponse.model_validate_json(
                    content
                )
            except Exception:
                try:
                    m = re.search(r"\{.*\}|\[.*\]", content, re.DOTALL)
                    if not m:
                        raise ValueError("No JSON found in model output")
                    raw_json = m.group(0)
                    if raw_json.lstrip().startswith("["):
                        wrapped = json.dumps({"entities": json.loads(raw_json)})
                        entities_response = (
                            ClinicalEntitiesResponse.model_validate_json(wrapped)
                        )
                    else:
                        entities_response = (
                            ClinicalEntitiesResponse.model_validate_json(raw_json)
                        )
                except Exception as e2:
                    logger.error("Error validating GPT JSON output: %s", e2)
                    return []

            # Convert to pipeline-compatible format
            result = []
            for entity in entities_response.entities:
                start, end = self._calculate_offsets(text, entity.value_raw)

                entity_dict = {
                    "type": entity.label,
                    "text": entity.value_raw,
                    "start": start if start >= 0 else None,
                    "end": end if end >= 0 else None,
                    "score": entity.confidence,
                    "code": entity.value_norm,
                }
                result.append(entity_dict)

            logger.info("Extracted %d clinical entities with GPT", len(result))
            return result

        except Exception as e:
            logger.error("Error in GPT extraction: %s", e)
            return []

    def meta(self) -> Dict[str, Any]:
        """
        Return extractor metadata for logging and debugging.

        Returns:
            Dictionary containing extractor configuration and capabilities.
        """
        return {
            "extractor": "gpt",
            "model": self.model,
            "provider": "openai",
            "structured_outputs": True,
            "categories": list(ENTITY_CATEGORIES.keys()),
            "total_labels": sum(len(labels) for labels in ENTITY_CATEGORIES.values()),
        }


# ==============================================================================
# Local LLM Extractor (Stub)
# ==============================================================================


class LocalLLMExtractor:
    """
    Stub extractor for local LLM models (e.g., Llama, Mistral via Ollama).

    This class provides a placeholder for future local model integration,
    allowing the system to be extended without API dependencies.

    Attributes:
        model: Local model identifier.
        base_url: Ollama or compatible server URL.

    Note:
        This is a stub implementation. The :meth:`predict` method returns
        an empty list until full implementation is completed.
    """

    def __init__(
        self,
        model: str = "llama3.1:8b",
        base_url: str = "http://localhost:11434",
    ) -> None:
        """
        Initialize the local LLM extractor stub.

        Args:
            model: Local model name (e.g., "llama3.1:8b").
            base_url: Ollama server URL.
        """
        self.model = model
        self.base_url = base_url

        logger.info("Local LLM Extractor (STUB) initialized with model: %s", self.model)
        logger.warning("Local LLM extractor is a stub. Implementation pending.")

    def predict(self, text: str) -> List[Dict[str, Any]]:
        """
        Stub prediction method.

        Args:
            text: Clinical note text (unused in stub).

        Returns:
            Empty list (implementation pending).
        """
        logger.warning("Local LLM extractor not implemented. Returning empty list.")
        return []

    def meta(self) -> Dict[str, Any]:
        """
        Return extractor metadata.

        Returns:
            Dictionary indicating stub status.
        """
        return {
            "extractor": "local_llm",
            "model": self.model,
            "provider": "local",
            "base_url": self.base_url,
            "status": "stub",
        }


# ==============================================================================
# LLM Extractor Facade
# ==============================================================================


class LLMExtractor:
    """
    Facade class for LLM-based clinical entity extraction.

    This class provides a unified interface to different LLM providers,
    selecting the appropriate extractor based on the specified provider.

    The facade pattern simplifies client code by hiding the complexity
    of provider selection and initialization.

    Supported Providers:
        - ``claude``: Anthropic Claude (default)
        - ``gpt``: OpenAI GPT-4o
        - ``local``: Local models via Ollama (stub)

    Attributes:
        provider: Selected provider identifier.
        extractor: Underlying provider-specific extractor instance.

    Example:
        >>> # Default Claude extractor
        >>> extractor = LLMExtractor()
        >>> entities = extractor.predict("FiO2 60%")
        >>>
        >>> # Explicit GPT extractor
        >>> extractor = LLMExtractor(provider="gpt")
        >>> entities = extractor.predict("PEEP 8 cmH2O")

    See Also:
        - :class:`ClaudeLLMExtractor` for Claude implementation
        - :class:`GPTLLMExtractor` for GPT implementation
    """

    def __init__(
        self,
        provider: str = "claude",
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        **kwargs,
    ) -> None:
        """
        Initialize the LLM extractor with the specified provider.

        Args:
            provider: LLM provider ("claude", "gpt", "local").
            api_key: API key for the selected provider.
            model: Specific model identifier to use.
            **kwargs: Additional arguments passed to the provider extractor.

        Note:
            Unknown providers default to Claude with a warning.
        """
        self.provider = provider.lower()

        if self.provider == "claude":
            self.extractor = ClaudeLLMExtractor(
                api_key=api_key, model=model or "claude-sonnet-4-20250514"
            )
        elif self.provider == "gpt":
            self.extractor = GPTLLMExtractor(
                api_key=api_key, model=model or "gpt-4o-2024-08-06"
            )
        elif self.provider == "local":
            self.extractor = LocalLLMExtractor(model=model or "llama3.1:8b", **kwargs)
        else:
            logger.warning("Unknown provider: %s. Using Claude as default.", provider)
            self.extractor = ClaudeLLMExtractor(api_key=api_key)

        logger.info("LLMExtractor initialized with provider: %s", self.provider)

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
            Dictionary with extractor metadata including provider info.
        """
        base_meta = self.extractor.meta()
        base_meta["facade_provider"] = self.provider
        return base_meta
