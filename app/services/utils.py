"""
Utility Functions for MedAI Services.

This module provides helper functions used across various services,
including value normalization for entity extraction.
"""

import re
from typing import Optional


def extract_normalized_value(text: str, entity_type: str) -> Optional[str]:
    """
    Extrae el valor normalizado de una entidad usando regex.

    Para valores numéricos: extrae números con unidades
    Para diagnósticos/modos: extrae texto limpio

    Args:
        text: Texto de la entidad ("FiO2 40%", "modo AC")
        entity_type: Tipo de entidad ("FIO2", "MODO", "DX")

    Returns:
        Valor normalizado o None

    Examples:
        >>> extract_normalized_value("FiO2 40%", "FIO2")
        "40"
        >>> extract_normalized_value("PEEP 8 cmH2O", "PEEP")
        "8"
        >>> extract_normalized_value("PA 120/80", "PA")
        "120/80"
        >>> extract_normalized_value("modo AC", "MODO")
        "AC"
        >>> extract_normalized_value("I:E 1:2", "I_E")
        "1:2"
        >>> extract_normalized_value("pH 7.35", "PH")
        "7.35"
        >>> extract_normalized_value("Peso 65 kg", "PESO")
        "65"
        >>> extract_normalized_value("edad 78 años", "EDAD")
        "78"
    """
    if not text:
        return None

    text = text.strip()

    # Patrones numéricos (orden importa: de más específico a más general)
    numeric_patterns = [
        (r'(\d+)/(\d+)', lambda m: m.group(0)),              # "120/80" → "120/80"
        (r'(\d+):(\d+(?:\.\d+)?)', lambda m: m.group(0)),    # "1:2" → "1:2"
        (r'(\d+(?:\.\d+)?)\s*%', lambda m: m.group(1)),      # "40%" → "40"
        (r'(-?\d+(?:\.\d+)?)', lambda m: m.group(1)),        # Número general
    ]

    for pattern, extractor in numeric_patterns:
        match = re.search(pattern, text)
        if match:
            return extractor(match)

    # Si no hay número, limpiar texto descriptivo
    clean = text

    # Para MODO: remover palabras como "modo", "en modo", "VMI"
    if entity_type == "MODO":
        clean = re.sub(r'^(modo|en\s+modo|VMI)\s+', '', clean, flags=re.IGNORECASE)

    # Para PESO/EDAD: remover palabras descriptivas
    if entity_type in ["PESO", "EDAD"]:
        clean = re.sub(r'^(peso|edad|Peso|Edad)[\s:]+', '', clean, flags=re.IGNORECASE)

    return clean.strip() if clean else None
