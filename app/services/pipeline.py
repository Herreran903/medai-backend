# pipeline.py
# Este módulo define una función principal para extraer entidades desde texto utilizando modelos registrados.
# Incluye soporte para normalización opcional de entidades y manejo de metadatos del modelo.

from typing import Any, Dict, List

from app.schemas import Entity, ExtractResponse
from app.services.normalizer import NormOptions, normalize_entities
from app.services.registry import MODEL_REGISTRY


# Función principal para extraer entidades desde texto.
# text: texto de entrada del cual se extraerán entidades.
# model: nombre del modelo registrado que se usará para la extracción.
# normalize: indica si las entidades extraídas deben ser normalizadas.
# systems: lista opcional de sistemas para normalización.
# restrict_types: lista opcional de tipos de entidades a restringir durante la normalización.
# Retorna un ExtractResponse con las entidades extraídas y metadatos.
def extract_from_text(
    text: str,
    model: str,
    *,
    model_variant: str | None = None,
    normalize: bool = False,
    systems: List[str] | None = None,
    restrict_types: List[str] | None = None,
) -> ExtractResponse:
    # Verifica si el modelo solicitado está registrado; lanza un error si no lo está.
    if model not in MODEL_REGISTRY:
        raise ValueError(f"Modelo no soportado: {model}")

    # Obtiene el extractor del registro de modelos. Si es una función callable sin atributo 'predict',
    # se inicializa y se actualiza en el registro.
    extractor = MODEL_REGISTRY[model]
    if callable(extractor) and not hasattr(extractor, "predict"):
        extractor = extractor()
        MODEL_REGISTRY[model] = extractor
    
    # Si se especifica una variante del modelo, reinicializa el extractor con esa variante
    if model_variant:
        if model == "llm":
            # Para LLM: claude, gpt, local
            from app.models.llm import LLMExtractor
            extractor = LLMExtractor(provider=model_variant)
        elif model == "transformer":
            # Para Transformer: permite especificar model_id diferente (beto, roberta, etc.)
            from app.models.transformer import TransformerExtractor
            import os
            # Mapeo de variantes a model_ids - lee primero de variables de entorno
            variant_map = {
                "beto": os.getenv("TRANSFORMER_BETO_MODEL_ID", "NicolasUnivalle/beto-vm-ner-full"),
                "beto_peft": os.getenv("TRANSFORMER_BETO_PEFT_MODEL_ID", "NicolasUnivalle/beto-vm-ner-peft"),
                "roberta": os.getenv("TRANSFORMER_ROBERTA_MODEL_ID", "NicolasUnivalle/roberta-vm-ner-full"),
                "roberta_peft": os.getenv("TRANSFORMER_ROBERTA_PEFT_MODEL_ID", "NicolasUnivalle/roberta-vm-ner-peft"),
            }
            model_id = variant_map.get(model_variant, model_variant)
            extractor = TransformerExtractor(model_id=model_id)

    # Intenta predecir entidades desde el texto usando el extractor.
    # Maneja errores específicos y generales para asegurar robustez.
    try:
        raw_entities = extractor.predict(text)
    except TypeError:
        # Manejo especial para casos donde el extractor requiere un segundo intento.
        raw_entities = extractor.predict(text)
    except Exception as e:
        # Propaga errores inesperados con un mensaje descriptivo.
        raise RuntimeError(f"Error ejecutando el extractor '{model}': {e}") from e

    # Inicializa la lista de entidades procesadas.
    entities: List[Entity] = []
    for e in raw_entities or []:
        # Valida los offsets 'start' y 'end' de cada entidad para asegurar que estén dentro del rango del texto.
        s, end = e.get("start"), e.get("end")
        if s is not None and end is not None:
            if not (0 <= s < end <= len(text or "")):
                # Si los offsets son inválidos, se eliminan para evitar inconsistencias.
                e = {k: v for k, v in e.items() if k not in ("start", "end")}
        # Crea una instancia de Entity con los datos procesados.
        entities.append(Entity(**e))

        # Si se solicita normalización y hay entidades, aplica el proceso de normalización.
        if normalize and entities:
            opts = NormOptions(
                enabled=True,
                systems=systems
                or None,  # Sistemas específicos para normalización, si se proporcionan.
                restrict_types=restrict_types,  # Tipos de entidades a restringir, si se especifican.
                min_link_score=0.60,  # Puntaje mínimo para considerar un enlace válido.
                max_candidates=25,  # Número máximo de candidatos para normalización.
                vsac_whitelists=None,  # Lista blanca opcional para normalización (no usada aquí).
            )
            # Convierte las entidades a diccionarios para el proceso de normalización.
            ents_dicts = [e.model_dump() for e in entities]
            # Aplica la normalización y reconstruye las entidades normalizadas.
            ents_norm = normalize_entities(ents_dicts, opts)
            entities = [Entity(**d) for d in ents_norm]

    # Construye los metadatos de la respuesta, incluyendo información del modelo y normalización.
    meta: Dict[str, Any] = {
        "model": model,
        "count": len(entities),  # Número total de entidades extraídas.
        "normalized": bool(normalize),  # Indica si se aplicó normalización.
    }
    # Si el extractor tiene metadatos adicionales, los agrega al diccionario de metadatos.
    if hasattr(extractor, "meta"):
        try:
            ext_meta = extractor.meta()
            if isinstance(ext_meta, dict):
                meta.update(ext_meta)
        except Exception:
            # Ignora errores al obtener metadatos adicionales del extractor.
            pass

    # Retorna la respuesta con el texto original, las entidades procesadas y los metadatos.
    return ExtractResponse(text=text or "", entities=entities, meta=meta)
