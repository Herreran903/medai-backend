# Este archivo define un servicio de traducción de texto de español a inglés.
# Soporta múltiples proveedores de traducción, incluyendo opciones en línea y offline,
# con manejo de fallos y un sistema de caché para optimizar el rendimiento.

from __future__ import annotations

import os
from functools import lru_cache

# Obtiene el proveedor de traducción desde las variables de entorno, por defecto "chain".
PROVIDER = os.getenv("TRANSLATOR_PROVIDER", "chain").lower()
# Define el límite de fallos consecutivos permitidos antes de deshabilitar el proveedor web.
FAIL_LIMIT = int(os.getenv("TRANSLATOR_FAIL_LIMIT", "3"))


# Clase interna para mantener el estado global del servicio de traducción.
class _State:
    down_web = False  # Indica si el proveedor web está deshabilitado.
    fails_web = 0  # Contador de fallos consecutivos del proveedor web.
    warned = False  # Indica si ya se emitió una advertencia.


# Función auxiliar para emitir una advertencia solo una vez.
# Esto evita inundar los logs con mensajes repetitivos.
def _warn_once(msg: str):
    if not _State.warned:
        print(f"[translator] {msg}")
        _State.warned = True


# Variables globales para almacenar el modelo y el tokenizador de MarianMT.
_MARIAN = None
_MARIAN_TOK = None


# Función para realizar traducción offline usando el modelo MarianMT.
# Carga el modelo y el tokenizador si no están inicializados, y traduce el texto.
def _offline_marian_es_en(text: str) -> str:
    global _MARIAN, _MARIAN_TOK
    try:
        if _MARIAN is None:
            # Carga el modelo y el tokenizador de Helsinki-NLP para traducción de español a inglés.
            from transformers import MarianMTModel, MarianTokenizer

            model_name = "Helsinki-NLP/opus-mt-es-en"
            _MARIAN_TOK = MarianTokenizer.from_pretrained(model_name)
            _MARIAN = MarianMTModel.from_pretrained(model_name)
        # Tokeniza el texto de entrada y genera la traducción.
        inputs = _MARIAN_TOK(
            [text], return_tensors="pt", padding=True, truncation=True, max_length=256
        )
        gen = _MARIAN.generate(**inputs, max_length=256, num_beams=1)
        # Decodifica la salida generada y elimina tokens especiales.
        return _MARIAN_TOK.decode(gen[0], skip_special_tokens=True)
    except Exception as e:
        # Si ocurre un error, emite una advertencia y devuelve el texto original.
        _warn_once(f"offline MarianMT failed: {type(e).__name__}: {e}")
        return text


# Función principal para traducir texto de español a inglés.
# Utiliza un sistema de caché para evitar traducciones redundantes.
@lru_cache(maxsize=4096)
def translate_es_to_en(text: str) -> str:
    # Si el texto está vacío, lo devuelve directamente.
    if not text:
        return text

    # Si el proveedor está deshabilitado explícitamente, devuelve el texto sin traducir.
    if PROVIDER in ("none", "off", "false"):
        return text

    # Si el proveedor es "offline", utiliza el modelo MarianMT.
    if PROVIDER == "offline":
        return _offline_marian_es_en(text)

    # Si el proveedor web está deshabilitado, utiliza la traducción offline como respaldo.
    if _State.down_web and PROVIDER in ("chain", "deep", "google"):
        return _offline_marian_es_en(text)

    try:
        # Si el proveedor es "chain" o "deep", intenta traducir con GoogleTranslator.
        # Si falla, utiliza MyMemoryTranslator como respaldo.
        if PROVIDER in ("chain", "deep"):
            from deep_translator import GoogleTranslator, MyMemoryTranslator

            try:
                return GoogleTranslator(source="es", target="en").translate(text)
            except Exception:
                return MyMemoryTranslator(source="es", target="en").translate(text)

        # Si el proveedor es "google", utiliza exclusivamente GoogleTranslator.
        elif PROVIDER == "google":
            from deep_translator import GoogleTranslator

            return GoogleTranslator(source="es", target="en").translate(text)

        # Si no se reconoce el proveedor, devuelve el texto sin traducir.
        return text

    except Exception as e:
        # Incrementa el contador de fallos y deshabilita el proveedor web si se supera el límite.
        _State.fails_web += 1
        if _State.fails_web >= FAIL_LIMIT:
            _State.down_web = True
            _warn_once(
                f"web provider disabled after {FAIL_LIMIT} fails: {type(e).__name__}: {e}"
            )
        # Como respaldo, utiliza la traducción offline o devuelve el texto original.
        return _offline_marian_es_en(text) or text
