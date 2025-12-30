"""
LLM Extractor for Clinical Entity Extraction
Uses Claude Sonnet 4.5 with Structured Outputs (JSON Schema) to extract clinical entities
from medical text with guaranteed JSON format and comprehensive entity coverage.
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

# Configuración del logger
logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    fmt = logging.Formatter("[%(levelname)s] %(name)s: %(message)s")
    handler.setFormatter(fmt)
    logger.addHandler(handler)
logger.setLevel(logging.INFO)


# ============================================================================
# JSON Schema Definitions for Clinical Entities
# ============================================================================


class ClinicalEntity(BaseModel):
    """
    Representa una entidad clínica extraída del texto.
    Compatible con el formato esperado por el pipeline LSTM/Transformer.
    """

    label: str = Field(
        description="Tipo de entidad clínica (ej: FiO2, PEEP, temperatura)"
    )
    value_raw: str = Field(description="Valor extraído tal como aparece en el texto")
    value_norm: str = Field(
        default="", description="Valor normalizado (número + unidad estándar)"
    )
    units: str = Field(default="", description="Unidades de medida normalizadas")
    confidence: float = Field(
        default=0.95, ge=0.0, le=1.0, description="Confianza de la extracción (0-1)"
    )
    category: str = Field(
        description="Categoría de la entidad (ej: ventilacion, signos_vitales)"
    )


class ClinicalEntitiesResponse(BaseModel):
    """
    Respuesta estructurada que contiene todas las entidades clínicas extraídas.
    """

    entities: List[ClinicalEntity] = Field(
        default_factory=list, description="Lista de entidades clínicas extraídas"
    )


# ============================================================================
# Entity Categories and Labels Mapping
# ============================================================================

ENTITY_CATEGORIES = {
    "ventilacion": ["MODO", "FIO2", "PEEP", "FR", "VT", "FLUJO", "I_E", "SENS"],
    "respuesta_ventilacion": ["SAO2", "PP", "PMES", "PM"],
    "antropometricos": ["EDAD", "PESO", "TALLA"],
    "signos_vitales": ["TEMP", "PA", "PAS", "PAD", "PAM", "FC", "GLICEMIA", "POSTURA"],
    "observaciones": ["DX"],
    "gases_arteriales": ["PH", "PACO2", "HCO3", "BE", "PAO2", "PAFI"],
}

# Mapeo inverso: label -> category
LABEL_TO_CATEGORY = {}
for category, labels in ENTITY_CATEGORIES.items():
    for label in labels:
        LABEL_TO_CATEGORY[label] = category


# ============================================================================
# Prompt Template for Claude
# ============================================================================

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


# ============================================================================
# LLM Extractors
# ============================================================================


class ClaudeLLMExtractor:
    """
    Extractor de entidades clínicas usando Claude Sonnet 4.5 con Structured Outputs.
    Garantiza respuestas en formato JSON válido con schema validation.
    """

    def __init__(
        self, api_key: Optional[str] = None, model: str = "claude-sonnet-4-20250514"
    ):
        """
        Inicializa el extractor de Claude.

        Args:
            api_key: API key de Anthropic (si no se proporciona, se lee de ANTHROPIC_API_KEY)
            model: Modelo de Claude a usar (por defecto claude-sonnet-4-20250514)
        """
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        self.model = model
        self.client = None

        if self.api_key:
            try:
                import anthropic

                self.client = anthropic.Anthropic(api_key=self.api_key)
                logger.info(
                    f"Claude LLM Extractor inicializado con modelo: {self.model}"
                )
            except ImportError:
                logger.warning(
                    "anthropic package no instalado. Instala con: pip install anthropic"
                )
            except Exception as e:
                logger.error(f"Error inicializando cliente de Anthropic: {e}")
        else:
            logger.warning(
                "ANTHROPIC_API_KEY no configurada. El extractor no funcionará."
            )

    def _calculate_offsets(self, text: str, entity_text: str) -> tuple[int, int]:
        """
        Calcula los offsets (start, end) de una entidad en el texto original.

        Args:
            text: Texto completo
            entity_text: Texto de la entidad a buscar

        Returns:
            Tupla (start, end) con las posiciones
        """
        # Busca la primera ocurrencia del texto de la entidad
        start = text.find(entity_text)
        if start == -1:
            # Si no se encuentra exactamente, intenta búsqueda case-insensitive
            pattern = re.escape(entity_text)
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                start = match.start()
                end = match.end()
            else:
                # Si aún no se encuentra, retorna posiciones inválidas
                return (-1, -1)
        else:
            end = start + len(entity_text)

        return (start, end)

    def predict(self, text: str) -> List[Dict[str, Any]]:
        """
        Extrae entidades clínicas del texto usando Claude con structured outputs.

        Args:
            text: Texto médico del cual extraer entidades

        Returns:
            Lista de diccionarios con formato compatible con pipeline LSTM/Transformer:
            [{"type": str, "text": str, "start": int, "end": int, "score": float, "code": str}]
        """
        if not self.client:
            logger.error("Cliente de Claude no inicializado. Retornando lista vacía.")
            return []

        if not text or not text.strip():
            return []

        try:
            # Construye el prompt con el texto
            prompt = EXTRACTION_PROMPT.format(text=text)
            schema = ClinicalEntitiesResponse.model_json_schema()
            response = None
            content = None

            # Usa fallback JSON-only por prompt (Anthropic SDK no soporta response_format)
            try:
                system_msg = (
                    "Devuelve exclusivamente un JSON válido que cumpla estrictamente con este JSON Schema de Pydantic. "
                    "No incluyas NINGÚN texto fuera del JSON:\n" + json.dumps(schema)
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
                logger.error(f"Error llamando a Claude: {e}")
                return []

            # Intenta validar directamente contra el schema; si falla, intenta recuperar el JSON del texto
            try:
                entities_response = ClinicalEntitiesResponse.model_validate_json(
                    content
                )
            except Exception:
                try:
                    import re as _re

                    # Extrae el primer objeto o arreglo JSON
                    m = _re.search(r"\{.*\}|\[.*\]", content, _re.DOTALL)
                    if not m:
                        raise ValueError("No se encontró JSON en la salida del modelo")
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
                    logger.error(f"Error validando JSON de Claude: {e2}")
                    return []

            # Convierte al formato esperado por el pipeline
            result = []
            for entity in entities_response.entities:
                # Calcula offsets en el texto original
                start, end = self._calculate_offsets(text, entity.value_raw)

                # Construye el diccionario en formato compatible
                entity_dict = {
                    "type": entity.label,
                    "text": entity.value_raw,
                    "start": start if start >= 0 else None,
                    "end": end if end >= 0 else None,
                    "score": entity.confidence,
                    "code": entity.value_norm,  # Usa value_norm como código
                }
                result.append(entity_dict)

            logger.info(f"Extraídas {len(result)} entidades clínicas con Claude")
            return result

        except Exception as e:
            logger.error(f"Error en extracción con Claude: {e}")
            return []

    def meta(self) -> Dict[str, Any]:
        """Retorna metadatos del extractor."""
        return {
            "extractor": "claude",
            "model": self.model,
            "provider": "anthropic",
            "structured_outputs": False,  # SDK no soporta response_format, usa system prompt
            "categories": list(ENTITY_CATEGORIES.keys()),
            "total_labels": sum(len(labels) for labels in ENTITY_CATEGORIES.values()),
        }


class GPTLLMExtractor:
    """
    Extractor de entidades clínicas usando OpenAI GPT con Structured Outputs.
    Garantiza respuestas en formato JSON válido con schema validation.
    """

    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4o-2024-08-06"):
        """
        Inicializa el extractor de GPT.

        Args:
            api_key: API key de OpenAI (si no se proporciona, se lee de OPENAI_API_KEY)
            model: Modelo de GPT a usar (por defecto gpt-4o-2024-08-06)
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model
        self.client = None

        if self.api_key:
            try:
                from openai import OpenAI

                self.client = OpenAI(api_key=self.api_key)
                logger.info(f"GPT LLM Extractor inicializado con modelo: {self.model}")
            except ImportError:
                logger.warning(
                    "openai package no instalado. Instala con: pip install openai"
                )
            except Exception as e:
                logger.error(f"Error inicializando cliente de OpenAI: {e}")
        else:
            logger.warning("OPENAI_API_KEY no configurada. El extractor no funcionará.")

    def _calculate_offsets(self, text: str, entity_text: str) -> tuple[int, int]:
        """
        Calcula los offsets (start, end) de una entidad en el texto original.

        Args:
            text: Texto completo
            entity_text: Texto de la entidad a buscar

        Returns:
            Tupla (start, end) con las posiciones
        """
        # Busca la primera ocurrencia del texto de la entidad
        start = text.find(entity_text)
        if start == -1:
            # Si no se encuentra exactamente, intenta búsqueda case-insensitive
            pattern = re.escape(entity_text)
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                start = match.start()
                end = match.end()
            else:
                # Si aún no se encuentra, retorna posiciones inválidas
                return (-1, -1)
        else:
            end = start + len(entity_text)

        return (start, end)

    def predict(self, text: str) -> List[Dict[str, Any]]:
        """
        Extrae entidades clínicas del texto usando GPT con structured outputs.

        Args:
            text: Texto médico del cual extraer entidades

        Returns:
            Lista de diccionarios con formato compatible con pipeline LSTM/Transformer:
            [{"type": str, "text": str, "start": int, "end": int, "score": float, "code": str}]
        """
        if not self.client:
            logger.error("Cliente de OpenAI no inicializado. Retornando lista vacía.")
            return []

        if not text or not text.strip():
            return []

        try:
            # Construye el prompt con el texto
            prompt = EXTRACTION_PROMPT.format(text=text)
            schema = ClinicalEntitiesResponse.model_json_schema()

            # Prepara el schema para OpenAI (requiere additionalProperties: false y required completo)
            openai_schema = schema.copy()

            def fix_schema_for_openai(obj):
                """Ajusta el schema para cumplir con OpenAI strict mode"""
                if isinstance(obj, dict):
                    # Agrega additionalProperties: false para objetos
                    if "type" in obj and obj["type"] == "object":
                        if "additionalProperties" not in obj:
                            obj["additionalProperties"] = False
                        # Asegura que required incluya todas las propiedades
                        if "properties" in obj:
                            all_props = list(obj["properties"].keys())
                            if "required" not in obj:
                                obj["required"] = all_props
                            else:
                                # Agrega propiedades faltantes a required
                                for prop in all_props:
                                    if prop not in obj["required"]:
                                        obj["required"].append(prop)
                    # Recursión
                    for key, value in obj.items():
                        fix_schema_for_openai(value)
                elif isinstance(obj, list):
                    for item in obj:
                        fix_schema_for_openai(item)

            fix_schema_for_openai(openai_schema)

            # Usa structured outputs de OpenAI
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {
                            "role": "system",
                            "content": "Eres un experto en extracción de entidades clínicas. Devuelve SOLO JSON válido.",
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
                # Fallback sin structured outputs
                logger.warning(f"Error con structured outputs, usando fallback: {e}")
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {
                            "role": "system",
                            "content": f"Eres un experto en extracción de entidades clínicas. Devuelve SOLO JSON válido que cumpla con este schema:\n{json.dumps(schema)}",
                        },
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.0,
                    max_tokens=4096,
                )
                content = response.choices[0].message.content

            # Intenta validar directamente contra el schema
            try:
                entities_response = ClinicalEntitiesResponse.model_validate_json(
                    content
                )
            except Exception:
                try:
                    # Extrae el primer objeto o arreglo JSON
                    m = re.search(r"\{.*\}|\[.*\]", content, re.DOTALL)
                    if not m:
                        raise ValueError("No se encontró JSON en la salida del modelo")
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
                    logger.error(f"Error validando JSON de GPT: {e2}")
                    return []

            # Convierte al formato esperado por el pipeline
            result = []
            for entity in entities_response.entities:
                # Calcula offsets en el texto original
                start, end = self._calculate_offsets(text, entity.value_raw)

                # Construye el diccionario en formato compatible
                entity_dict = {
                    "type": entity.label,
                    "text": entity.value_raw,
                    "start": start if start >= 0 else None,
                    "end": end if end >= 0 else None,
                    "score": entity.confidence,
                    "code": entity.value_norm,  # Usa value_norm como código
                }
                result.append(entity_dict)

            logger.info(f"Extraídas {len(result)} entidades clínicas con GPT")
            return result

        except Exception as e:
            logger.error(f"Error en extracción con GPT: {e}")
            return []

    def meta(self) -> Dict[str, Any]:
        """Retorna metadatos del extractor."""
        return {
            "extractor": "gpt",
            "model": self.model,
            "provider": "openai",
            "structured_outputs": True,
            "categories": list(ENTITY_CATEGORIES.keys()),
            "total_labels": sum(len(labels) for labels in ENTITY_CATEGORIES.values()),
        }


class LocalLLMExtractor:
    """
    Stub para extractor usando LLM local (ej: Llama, Mistral via Ollama).
    Implementación futura para modelos locales con structured outputs.
    """

    def __init__(
        self, model: str = "llama3.1:8b", base_url: str = "http://localhost:11434"
    ):
        """
        Inicializa el extractor de LLM local (stub).

        Args:
            model: Nombre del modelo local
            base_url: URL base del servidor (ej: Ollama)
        """
        self.model = model
        self.base_url = base_url

        logger.info(f"Local LLM Extractor (STUB) inicializado con modelo: {self.model}")
        logger.warning("Local LLM extractor es un stub. Implementación pendiente.")

    def predict(self, text: str) -> List[Dict[str, Any]]:
        """
        Stub para extracción con LLM local.

        Args:
            text: Texto médico

        Returns:
            Lista vacía (implementación pendiente)
        """
        logger.warning("Local LLM extractor no implementado. Retornando lista vacía.")
        return []

    def meta(self) -> Dict[str, Any]:
        """Retorna metadatos del extractor."""
        return {
            "extractor": "local_llm",
            "model": self.model,
            "provider": "local",
            "base_url": self.base_url,
            "status": "stub",
        }


# ============================================================================
# Main LLMExtractor Class (Facade)
# ============================================================================


class LLMExtractor:
    """
    Clase principal que actúa como facade para diferentes extractores LLM.
    Por defecto usa Claude Sonnet 4.5, pero puede configurarse para usar GPT o LLM local.
    """

    def __init__(
        self,
        provider: str = "claude",
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        **kwargs,
    ):
        """
        Inicializa el extractor LLM según el proveedor especificado.

        Args:
            provider: Proveedor del LLM ("claude", "gpt", "local")
            api_key: API key (si aplica)
            model: Modelo específico a usar
            **kwargs: Argumentos adicionales para el extractor
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
            logger.warning(
                f"Proveedor desconocido: {provider}. Usando Claude por defecto."
            )
            self.extractor = ClaudeLLMExtractor(api_key=api_key)

        logger.info(f"LLMExtractor inicializado con proveedor: {self.provider}")

    def predict(self, text: str) -> List[Dict[str, Any]]:
        """
        Extrae entidades clínicas del texto.

        Args:
            text: Texto médico del cual extraer entidades

        Returns:
            Lista de diccionarios con entidades en formato compatible con pipeline
        """
        return self.extractor.predict(text)

    def meta(self) -> Dict[str, Any]:
        """
        Retorna metadatos del extractor actual.

        Returns:
            Diccionario con información del extractor
        """
        base_meta = self.extractor.meta()
        base_meta["facade_provider"] = self.provider
        return base_meta
