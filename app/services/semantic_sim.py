# Este script implementa funciones para calcular la similitud semántica entre textos
# utilizando embeddings generados por un modelo preentrenado de Sentence Transformers.
# Incluye funciones para cargar el modelo, generar embeddings y calcular similitudes.

from __future__ import annotations

from functools import lru_cache
from typing import Iterable, List

import numpy as np

_MODEL = None  # Variable global para almacenar el modelo cargado


def _load_model():
    """
    Carga el modelo preentrenado de Sentence Transformers si no está ya cargado.
    Utiliza un modelo específico para embeddings multilingües.
    """
    global _MODEL
    if _MODEL is None:
        from sentence_transformers import SentenceTransformer

        # Carga el modelo 'intfloat/multilingual-e5-base' en CPU
        name = "intfloat/multilingual-e5-base"
        _MODEL = SentenceTransformer(name, device="cpu")
    return _MODEL


@lru_cache(maxsize=8192)
def _embed_one(text: str) -> np.ndarray:
    """
    Genera el embedding de un texto individual utilizando el modelo cargado.

    Parámetros:
    - text: Cadena de texto a procesar.

    Retorna:
    - Un vector numpy que representa el embedding del texto.
    """
    m = _load_model()  # Asegura que el modelo esté cargado
    if not text:
        # Retorna un vector de ceros si el texto está vacío
        return np.zeros(m.get_sentence_embedding_dimension(), dtype=np.float32)
    prompt = text  # Define el texto a procesar
    # Genera el embedding normalizado del texto
    return m.encode(prompt, normalize_embeddings=True)


def _embed_many(texts: Iterable[str]) -> np.ndarray:
    """
    Genera embeddings para múltiples textos simultáneamente.

    Parámetros:
    - texts: Iterable de cadenas de texto.

    Retorna:
    - Una matriz numpy donde cada fila es el embedding de un texto.
    """
    m = _load_model()  # Asegura que el modelo esté cargado
    # Reemplaza textos vacíos con cadenas vacías para evitar errores
    arr: List[str] = [t or "" for t in texts]
    # Genera embeddings normalizados para todos los textos
    return m.encode(arr, normalize_embeddings=True)


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """
    Calcula la similitud coseno entre dos vectores.

    Parámetros:
    - a: Primer vector numpy.
    - b: Segundo vector numpy.

    Retorna:
    - Un valor flotante entre -1.0 y 1.0 que representa la similitud.
    """
    if a is None or b is None:
        # Retorna 0.0 si alguno de los vectores es None
        return 0.0
    # Calcula el producto punto y lo limita al rango [-1.0, 1.0]
    return float(np.clip(np.dot(a, b), -1.0, 1.0))


def sim_texts(a: str, b: str) -> float:
    """
    Calcula la similitud semántica entre dos textos.

    Parámetros:
    - a: Primer texto.
    - b: Segundo texto.

    Retorna:
    - Un valor flotante entre -1.0 y 1.0 que representa la similitud.
    """
    # Genera los embeddings de ambos textos
    va = _embed_one(a)
    vb = _embed_one(b)
    # Calcula la similitud coseno entre los embeddings
    return cosine_sim(va, vb)
