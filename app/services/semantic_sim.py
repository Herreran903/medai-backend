from __future__ import annotations

from functools import lru_cache
from typing import Iterable, List

import numpy as np

_MODEL = None


def _load_model():
    global _MODEL
    if _MODEL is None:
        from sentence_transformers import SentenceTransformer

        name = "intfloat/multilingual-e5-base"
        _MODEL = SentenceTransformer(name, device="cpu")
    return _MODEL


@lru_cache(maxsize=8192)
def _embed_one(text: str) -> np.ndarray:
    m = _load_model()
    if not text:
        return np.zeros(m.get_sentence_embedding_dimension(), dtype=np.float32)
    prompt = text
    return m.encode(prompt, normalize_embeddings=True)


def _embed_many(texts: Iterable[str]) -> np.ndarray:
    m = _load_model()
    arr: List[str] = [t or "" for t in texts]
    return m.encode(arr, normalize_embeddings=True)


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    if a is None or b is None:
        return 0.0
    return float(np.clip(np.dot(a, b), -1.0, 1.0))


def sim_texts(a: str, b: str) -> float:
    va = _embed_one(a)
    vb = _embed_one(b)
    return cosine_sim(va, vb)
