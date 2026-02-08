"""
LLM-Based Clinical Entity Extraction Module.

This module implements clinical Named Entity Recognition (NER) using Large Language
Models (LLMs) with structured output capabilities. It includes GPT (OpenAI) as the
active provider and retains a Claude (Anthropic) extractor as legacy code.

Architecture Context:
    The LLM extractor serves as an alternative to traditional NER models (LSTM,
    Transformer) when higher accuracy or flexibility is needed. It leverages
    the reasoning capabilities of LLMs to identify clinical entities from
    mechanical ventilation notes.

    The module follows the Strategy pattern with a Facade:

    - :class:`ClaudeLLMExtractor`: Anthropic Claude implementation (legacy)
    - :class:`GPTLLMExtractor`: OpenAI GPT implementation (active)
    - :class:`LocalLLMExtractor`: Stub for future local model support
    - :class:`LLMExtractor`: Facade fixed to GPT for experiments

Supported Providers:
    - **GPT** (active/fixed): Uses GPT with structured outputs (JSON schema mode)
    - **Claude** (legacy): Claude Sonnet support retained but disabled in experiments
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
    - ``ANTHROPIC_API_KEY``: Legacy (Claude extractor retained but disabled in experiments)

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
        value_norm: Normalized value extracted from the entity text.

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
        description="Exact text span including label/abbreviation and value (e.g., 'FiO2 60%', 'PEEP 8 cmH2O'). Include the clinical abbreviation when present in the text.",
    )
    value_norm: str = Field(
        default="",
        description="Normalized value in standard numeric format.",
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
# System and Extraction Prompts
# ==============================================================================

SYSTEM_PROMPT = (
    "Actuas exclusivamente como un extractor de entidades para evaluacion experimental de NER. "
    "Recibiras textos ya preprocesados dentro de un pipeline comparativo que incluye BiLSTM y modelos fine-tuned; "
    "por tanto, debes producir unicamente una salida estructurada estrictamente valida conforme al esquema Pydantic/JSON Schema proporcionado, "
    "sin texto libre, sin explicaciones y sin encabezados. "
    "La inferencia asume decodificacion guiada estricta (JSON Schema / FSM / XGrammar), por lo que solo estan permitidos tokens compatibles con el esquema. "
    "Extrae todas las entidades presentes en el texto, indicando texto exacto, etiqueta y offsets de caracteres (start, end) alineados con el texto de entrada. "
    "No inventes entidades, no cambies etiquetas, no omitas campos y no agregues claves adicionales. "
    "Si existe ambiguedad, prioriza recall sin violar el esquema."
)

EXTRACTION_PROMPT = """Eres un experto en extraccion de entidades clinicas de notas medicas de pacientes en ventilacion mecanica.

Tu tarea es extraer UNICAMENTE las entidades clinicas mencionadas en el texto.

IMPORTANTE - REGLAS DE EXTRACCION:
1. SIEMPRE incluir la etiqueta/sigla cuando esté presente en el texto junto al valor
2. Ejemplo: "FiO2 60%" -> extraer "FiO2 60%" (incluir la etiqueta "FiO2")
3. Ejemplo: "PEEP 8 cmH2O" -> extraer "PEEP 8 cmH2O" (incluir la etiqueta "PEEP")
4. Ejemplo: "VT 420 ml" -> extraer "VT 420 ml" (incluir la etiqueta "VT")
5. Ejemplo: "modo AC VC" -> extraer "AC VC" (NO incluir la palabra descriptiva "modo")
6. Si SOLO aparece el valor sin etiqueta en el texto, extraer solo el valor

**CONFIGURACION DE VENTILACION:**
- MODO: Modo del ventilador (sin palabras descriptivas como "modo", "en modo", "VMI")
  Variantes: AC, VC, PC, PC+, VC+, SIMV, PSV, CPAP, PRVC, ACV, VCV, PCV, BiPAP
  Ejemplos: "modo AC VC" -> extraer "AC VC", "en modo PC+" -> extraer "PC+", "VMI modo SIMV" -> extraer "SIMV"

- FIO2: Fraccion inspirada de oxigeno (incluir etiqueta + valor)
  Variantes: FiO2, FIO2, fio2, Fi02, fraccion inspirada de oxigeno
  Ejemplos: "FiO2 40%", "fio2: 0.5", "FIO2 80%", "Fi02 100"

- PEEP: Presion positiva al final de la espiracion
  Variantes: PEEP, peep, Peep
  Ejemplos: "PEEP 8", "peep: 10", "PEEP 12 cmH2O"

- FR: Frecuencia respiratoria
  Variantes: FR, fr, Freq resp, frecuencia respiratoria, rpm
  Ejemplos: "FR 14/20", "fr: 18 rpm", "FR 20", "frecuencia respiratoria 22"

- VT: Volumen tidal/corriente
  Variantes: VT, Vt, vt, Vol, vol, VC (volumen corriente), volumen tidal
  Ejemplos: "VT 380", "Vt: 450 mL", "Vol 480/490", "vol corriente 420"

- FLUJO: Flujo inspiratorio
  Variantes: Flujo, flujo, flow
  Ejemplos: "Flujo 45", "flujo 50 L/min"

- I_E: Relacion inspiracion:espiracion
  Variantes: I:E, I/E, relacion I:E, Rel I:E, RIE
  Ejemplos: "I:E 1:2", "1:2", "relacion 1:3", "Rel. 1:1.5"

- SENS: Sensibilidad del trigger
  Variantes: Sens, sens, sensibilidad, trigger
  Ejemplos: "Sens 2", "sensibilidad: 3", "sens: 1.5"

**RESPUESTA A LA VENTILACION:**
- SAO2: Saturacion de oxigeno
  Variantes: SaO2, SAO2, Sat, sat, saturacion, SO2, SpO2, SatO2
  Ejemplos: "SaO2 95%", "Sat 92%", "saturacion 97%", "SO2 94%", "SpO2 98"

- PP: Presion pico de via aerea
  Variantes: PP, Ppico, P pico, presion pico, peak pressure
  Ejemplos: "PP 25", "Ppico: 22", "presion pico 28", "P pico 30"

- PMES: Presion meseta/plateau
  Variantes: Pmes, Pmeseta, P meseta, plateau, presion meseta, P plateau
  Ejemplos: "Pmeseta 18", "plateau 20", "P meseta: 22", "presion plateau 19"

- PM: Poder mecanico
  Variantes: PM, poder mecanico, mechanical power

**ANTROPOMETRICOS:**
- EDAD: Edad del paciente - SOLO valor y unidad
  Variantes: anos, anos de edad, a, years
  Correcto: "78 anos", "65 anos", "42 a"
  Incorrecto: "edad 78 anos", "paciente de 65 anos de edad", "Edad: 70"

- PESO: Peso corporal - SOLO valor y unidad, NO la palabra "peso"
  Variantes: kg, kilos, kilogramos, Kg, KG
  Correcto: "65 kg", "72 kg", "95", "80 kilos"
  Incorrecto: "Peso 65 kg", "peso: 72 kg", "Peso(Kg): 70"

**SIGNOS VITALES:**
- TEMP: Temperatura corporal
  Variantes: T, Temp, temperatura, temp max, temp min, celsius
  Ejemplos: "T 36.5", "Temp: 37.2", "36.8 C", "T 37C", "temperatura 38.5"

- PA: Presion arterial completa (sistolica/diastolica)
  Variantes: PA, TA, presion arterial, tension arterial, PA(mmHg)
  Ejemplos: "PA 120/80", "TA: 130/85", "TA 110/70 mmHg", "presion arterial 125/80"

- PAS: Presion arterial sistolica SOLAMENTE
  Variantes: PAS, TAS, presion sistolica, tension sistolica, sistolica
  Ejemplos: "TAS 125", "PAS 140", "TAS menor 150", "sistolica 130"

- PAM: Presion arterial media
  Variantes: PAM, TAM, presion arterial media, tension arterial media, MAP
  Ejemplos: "PAM 85", "TAM: 78", "PAM 65 mmHg", "TAM 70"

- FC: Frecuencia cardiaca
  Variantes: FC, fc, frecuencia cardiaca, pulso, lpm, lat/min, latidos
  Ejemplos: "FC 82", "FC: 75 lpm", "fc 90", "frecuencia cardiaca 88"

- GLICEMIA: Glucosa/glucometria en sangre
  Variantes: Glicemia, glicemia, glucometria, glucosa, HGT, dextrostix, mg/dL
  Ejemplos: "Glicemia 120", "glucometria: 145", "125-130-140 mg/dL", "HGT 110"

- POSTURA: Posicion/posicionamiento del paciente
  Variantes: posicion, postura, decubito, cabecera, fowler, semifowler, prono, supino
  Ejemplos: "Cabecera 30", "supino", "prono", "decubito lateral", "semifowler", "Cabecera 45 grados", "posicion fowler"

**DIAGNOSTICOS (DX):**
- DX: Diagnosticos clinicos - extraer SOLO el nucleo diagnostico (2-5 palabras max)
  Variantes: diagnostico, Dx, dx, impresion diagnostica
  Correcto: "SDRA severo", "Falla ventilatoria", "Shock septico", "NAV", "Neumonia asociada a ventilador"
  Correcto: "HTA", "DM2", "EPOC", "IRC", "ERC", "IAM" (siglas de diagnosticos)
  Incorrecto: "paciente con antecedente de falla ventilatoria tipo 2" (muy largo)
  Incorrecto: "se considera SDRA severo por criterios de Berlin" (incluye explicacion)

**GASES ARTERIALES:**
- PH: pH sanguineo
  Variantes: pH, ph, PH
  Ejemplos: "pH 7.35", "ph: 7.42", "PH 7.28"

- PACO2: Presion parcial de CO2
  Variantes: PaCO2, PCO2, pCO2, CO2, paco2
  Ejemplos: "PaCO2 42", "pCO2: 38", "PCO2 45", "CO2 40"

- HCO3: Bicarbonato
  Variantes: HCO3, HCO3-, bicarbonato, Bic
  Ejemplos: "HCO3 24", "bicarbonato: 22", "HCO3- 26", "Bic 23"

- BE: Exceso de base
  Variantes: BE, EB, exceso de base, base excess
  Ejemplos: "BE -2", "EB: +1", "BE 3.5", "exceso de base -4"

- PAO2: Presion parcial de O2
  Variantes: PaO2, PO2, pO2, pao2
  Ejemplos: "PaO2 85", "pO2: 92", "PO2 78"

- PAFI: Relacion PaO2/FiO2
  Variantes: PAFI, PaFi, PaFiO2, pafi, Pa/Fi, relacion PaO2/FiO2, indice de Kirby
  Ejemplos: "PaFi 180", "PAFI: 250", "pafi 120", "Pa/Fi 200"

**REGLAS CRITICAS:**
1. NO incluir palabras descriptivas como "modo", "peso", "edad", "temperatura" antes del valor
2. SI incluir siglas medicas (FiO2, PEEP, FC, TAM, etc.) cuando estan junto al valor numerico
3. Para DX: extraer solo el diagnostico, no la oracion completa
4. Buscar TODAS las variantes de cada entidad listadas arriba
5. Extraer TODAS las entidades que encuentres, priorizar recall sobre precision
6. OBLIGATORIO: Calcular start y end (offsets de caracteres) para cada entidad
   - start: posicion del primer caracter de la entidad en el texto (0-indexed)
   - end: posicion del caracter DESPUES del ultimo caracter de la entidad
   - Ejemplo: en "Peso 65 kg", si extraes "65 kg", start=5, end=10

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
# Claude LLM Extractor (NOT USED - Only using GPT for experiments)
# ==============================================================================


class ClaudeLLMExtractor:
    """
    Clinical entity extractor using Anthropic Claude.

    NOTE: This extractor is NOT USED in current experimental setup.
    Only GPT is used for experiments. This class is kept for reference.

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
            "model": self.model,
            "provider": "anthropic",
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
        model: str = "gpt-5.2",  # Fixed: GPT-5.2 with temperature 0
    ) -> None:
        """
        Initialize the GPT extractor with fixed configuration for experiments.

        Configuration:
            - Model: gpt-5.2
            - Temperature: 0.0 (deterministic outputs)
            - System Prompt: Experimental NER extraction instructions

        Args:
            api_key: OpenAI API key. If not provided, reads from
                ``OPENAI_API_KEY`` environment variable.
            model: GPT model identifier. Defaults to gpt-5.2 (fixed for experiments).

        Raises:
            Warning logged if API key is not configured.
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model
        self.temperature = 0.0  # Fixed for reproducibility
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

            # Use structured outputs with experimental system prompt
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {
                            "role": "system",
                            "content": SYSTEM_PROMPT,
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
                    temperature=self.temperature,
                    max_completion_tokens=4096,
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
                            "content": SYSTEM_PROMPT,
                        },
                        {"role": "user", "content": prompt},
                    ],
                    temperature=self.temperature,
                    max_completion_tokens=4096,
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
            Dictionary containing extractor configuration.
        """
        return {
            "model": self.model,
            "provider": "openai",
        }


# ==============================================================================
# Local LLM Extractor (NOT USED - Only using GPT for experiments)
# ==============================================================================


class LocalLLMExtractor:
    """
    Stub extractor for local LLM models (e.g., Llama, Mistral via Ollama).

    NOTE: This extractor is NOT USED in current experimental setup.
    Only GPT is used for experiments. This class is kept for reference.

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

    SIMPLIFIED FOR EXPERIMENTS: This facade now ONLY uses GPT-5.2 with fixed configuration.
    Claude and Local providers are disabled.

    Configuration:
        - Provider: GPT (forced)
        - Model: gpt-5.2 (fixed)
        - Temperature: 0.0 (deterministic)

    Attributes:
        provider: Always "gpt" (fixed for experiments).
        extractor: GPTLLMExtractor instance.

    Example:
        >>> extractor = LLMExtractor()
        >>> entities = extractor.predict("FiO2 60%, PEEP 8")

    See Also:
        - :class:`GPTLLMExtractor` for implementation details
    """

    def __init__(
        self,
        provider: str = "gpt",
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        **kwargs,
    ) -> None:
        """
        Initialize the LLM extractor (GPT only, fixed for experiments).

        Args:
            provider: Ignored - always uses "gpt".
            api_key: OpenAI API key. Required.
            model: Model identifier (defaults to gpt-5.2).
            **kwargs: Ignored.

        Raises:
            ValueError: If OpenAI API key is not provided.
        """
        # Force GPT provider regardless of input
        self.provider = "gpt"

        if not api_key:
            raise ValueError("OpenAI API key required for GPT extractor")

        # Always use GPT with fixed configuration
        self.extractor = GPTLLMExtractor(
            api_key=api_key,
            model=model or "gpt-5.2",
        )

        logger.info("LLMExtractor initialized with provider: gpt (model: %s)", model or "gpt-5.2")

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
