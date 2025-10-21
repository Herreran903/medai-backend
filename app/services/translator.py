from __future__ import annotations

import os
from functools import lru_cache

PROVIDER = os.getenv("TRANSLATOR_PROVIDER", "chain").lower()
FAIL_LIMIT = int(os.getenv("TRANSLATOR_FAIL_LIMIT", "3"))


class _State:
    down_web = False
    fails_web = 0
    warned = False


def _warn_once(msg: str):
    if not _State.warned:
        print(f"[translator] {msg}")
        _State.warned = True


_MARIAN = None
_MARIAN_TOK = None


def _offline_marian_es_en(text: str) -> str:
    global _MARIAN, _MARIAN_TOK
    try:
        if _MARIAN is None:
            from transformers import MarianMTModel, MarianTokenizer

            model_name = "Helsinki-NLP/opus-mt-es-en"
            _MARIAN_TOK = MarianTokenizer.from_pretrained(model_name)
            _MARIAN = MarianMTModel.from_pretrained(model_name)
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
    if not text:
        return text

    if PROVIDER in ("none", "off", "false"):
        return text

    if PROVIDER == "offline":
        return _offline_marian_es_en(text)

    if _State.down_web and PROVIDER in ("chain", "deep", "google"):
        return _offline_marian_es_en(text)

    try:
        if PROVIDER in ("chain", "deep"):
            from deep_translator import GoogleTranslator, MyMemoryTranslator

            try:
                return GoogleTranslator(source="es", target="en").translate(text)
            except Exception:
                return MyMemoryTranslator(source="es", target="en").translate(text)

        elif PROVIDER == "google":
            from deep_translator import GoogleTranslator

            return GoogleTranslator(source="es", target="en").translate(text)

        return text

    except Exception as e:
        _State.fails_web += 1
        if _State.fails_web >= FAIL_LIMIT:
            _State.down_web = True
            _warn_once(
                f"web provider disabled after {FAIL_LIMIT} fails: {type(e).__name__}: {e}"
            )
        return _offline_marian_es_en(text) or text
