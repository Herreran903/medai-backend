# Este archivo define una clase `LSTMExtractor` que utiliza un modelo LSTM para extraer entidades
# nombradas de texto en formato BIO. Incluye métodos para tokenización, codificación, predicción
# y decodificación de etiquetas BIO a entidades estructuradas.

from __future__ import annotations

import json
import logging
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple
from xml.dom.minidom import Entity

import numpy as np
import tensorflow as tf

# Configuración del logger para registrar mensajes de advertencia e información
logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    fmt = logging.Formatter("[%(levelname)s] %(name)s: %(message)s")
    handler.setFormatter(fmt)
    logger.addHandler(handler)
logger.setLevel(logging.WARNING)


# Expresión regular para tokenizar texto en palabras y caracteres especiales
_TOKEN_REGEX = re.compile(r"\w+|[^\w\s]", re.UNICODE)


def _join_tokens(tokens: Sequence[str]) -> str:
    """
    Une una lista de tokens en una cadena, insertando espacios donde sea necesario.
    """
    parts: List[str] = []
    for i, tok in enumerate(tokens):
        # Inserta un espacio entre tokens alfanuméricos consecutivos
        if i > 0 and tokens[i - 1].isalnum() and tok.isalnum():
            parts.append(" ")
        parts.append(tok)
    return "".join(parts)


class LSTMExtractor:
    # Clase principal para la extracción de entidades usando un modelo LSTM.
    # Gestiona la carga del modelo, la tokenización, la predicción y la decodificación.

    TOKEN_REGEX = _TOKEN_REGEX  # Expresión regular para tokenización

    def __init__(self, model_dir: str | None = None) -> None:
        """
        Inicializa el extractor cargando el modelo y sus configuraciones.
        - model_dir: Ruta al directorio del modelo. Si no se especifica, usa una ruta predeterminada.
        """
        base_dir = Path(__file__).resolve().parent

        # Ruta predeterminada para el modelo, con soporte para rutas absolutas y relativas
        HARD_PATH = Path("/app/app/models/utils/vmi_enterizacion_ep30_bs32")

        default_dir = (
            HARD_PATH
            if HARD_PATH.exists()
            else (base_dir / "models" / "utils" / "vmi_enterizacion_ep30_bs32")
        )
        resolved = Path(model_dir or default_dir).expanduser().resolve()
        if not resolved.exists():
            raise FileNotFoundError(f"No existe el directorio del modelo: {resolved}")

        self.model_dir = resolved

        # Carga los diccionarios de mapeo para palabras y etiquetas
        self.word2idx: Dict[str, int] = json.loads(
            (self.model_dir / "word2idx.json").read_text(encoding="utf-8")
        )
        self.tag2idx: Dict[str, int] = json.loads(
            (self.model_dir / "tag2idx.json").read_text(encoding="utf-8")
        )
        self.idx2tag: Dict[int, str] = {int(v): k for k, v in self.tag2idx.items()}

        # Carga la configuración del modelo
        cfg = json.loads((self.model_dir / "config.json").read_text(encoding="utf-8"))
        self.max_len: int = int(cfg["max_len"])  # Longitud máxima de secuencias
        self.pad_id: int = int(cfg.get("pad_id", 0))  # ID para padding
        self.unk_id: int = int(
            self.word2idx.get("<UNK>", self.word2idx.get("[UNK]", 1))
        )  # ID para tokens desconocidos

        # Carga el modelo (Keras o SavedModel)
        keras_path = self.model_dir / "model.keras"
        if keras_path.exists():
            self.model = tf.keras.models.load_model(
                keras_path.as_posix(), compile=False
            )
            self._is_keras = True
            logger.info("Modelo Keras cargado desde %s", keras_path)
        else:
            sm_path = self.model_dir / "saved_model"
            self.model = tf.saved_model.load(sm_path.as_posix())
            self._is_keras = False
            logger.info("SavedModel cargado desde %s", sm_path)

    @staticmethod
    def _lower(text: str) -> str:
        # Convierte el texto a minúsculas
        return text.lower()

    def _tokenize(self, text: str) -> Tuple[List[str], List[Tuple[int, int]]]:
        """
        Tokeniza el texto y calcula los spans de cada token.
        - Retorna una lista de tokens y sus posiciones (inicio, fin) en el texto original.
        """
        tokens: List[str] = []
        spans: List[Tuple[int, int]] = []
        for m in self.TOKEN_REGEX.finditer(text):
            tokens.append(m.group(0))
            spans.append((m.start(), m.end()))
        return tokens, spans

    def _encode(self, tokens: Sequence[str]) -> np.ndarray:
        """
        Convierte una secuencia de tokens en IDs numéricos según el vocabulario.
        - Aplica padding o truncamiento según `max_len`.
        """
        ids = [self.word2idx.get(self._lower(t), self.unk_id) for t in tokens]
        if len(ids) < self.max_len:
            # Aplica padding si la secuencia es más corta que `max_len`
            ids = ids + [self.pad_id] * (self.max_len - len(ids))
        else:
            # Trunca la secuencia si excede `max_len`
            ids = ids[: self.max_len]
        return np.asarray([ids], dtype=np.int32)

    def _predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Calcula las probabilidades de las etiquetas para una secuencia de entrada.
        - Retorna un arreglo de forma (T, num_tags).
        """
        if isinstance(self.model, tf.keras.Model):
            # Predicción usando un modelo Keras
            probs = self.model.predict(X, verbose=0)[0]
        else:
            # Predicción usando un modelo SavedModel
            fn = self.model.signatures.get("serve")
            out = fn(tf.convert_to_tensor(X))
            probs = out["outputs"].numpy()[0]
        return probs

    def _bio_decode(
        self,
        tags: Sequence[str],
        tokens: Sequence[str],
        spans: Sequence[Tuple[int, int]],
        probs: np.ndarray,
    ) -> List[Entity]:
        """
        Convierte una secuencia de etiquetas BIO en entidades estructuradas.
        - Une etiquetas B-XXX e I-XXX consecutivas en una sola entidad.
        - Calcula el puntaje promedio de probabilidad para cada entidad.
        """
        ents: List[Entity] = []
        i = 0
        T = len(tokens)

        def _collect_run(start_idx: int, ent_type: str) -> Tuple[int, int]:
            """Identifica el rango de tokens que forman una entidad."""
            j = start_idx + 1
            while j < T and tags[j] == f"I-{ent_type}":
                j += 1
            return start_idx, j

        def _span_entity(i0: int, j0: int, ent_type: str) -> Entity:
            # Construye una entidad a partir de los tokens y sus spans
            start_char = spans[i0][0]
            end_char = spans[min(j0 - 1, T - 1)][1]
            text_str = _join_tokens(tokens[i0:j0])

            # Calcula el puntaje promedio de la entidad
            scores: List[float] = []
            for k in range(i0, j0):
                tag_name = f"I-{ent_type}" if k > i0 else f"B-{ent_type}"
                tag_id = self.tag2idx.get(tag_name)
                if tag_id is not None:
                    scores.append(float(probs[k, tag_id]))
            score = float(np.mean(scores)) if scores else 0.0

            return Entity(
                type=ent_type,
                text=text_str,
                score=score,
                start=int(start_char),
                end=int(end_char),
            )

        while i < T:
            tag = tags[i]
            if not tag or tag == "O":
                # Ignora etiquetas vacías o fuera de entidad
                i += 1
                continue

            if tag.startswith("B-"):
                # Inicia una nueva entidad
                ent_type = tag[2:]
                i0, j0 = _collect_run(i, ent_type)
                ents.append(_span_entity(i0, j0, ent_type))
                i = j0
                continue

            if tag.startswith("I-"):
                # Maneja etiquetas I-XXX fuera de contexto (caso raro)
                ent_type = tag[2:]
                i0, j0 = _collect_run(i, ent_type)
                ents.append(_span_entity(i0, j0, ent_type))
                i = j0
                continue

            i += 1

        return ents

    def predict(self, text: str) -> List[Dict]:
        """
        Predice entidades en un texto dado.
        - Retorna una lista de diccionarios con las entidades detectadas.
        """
        tokens, spans = self._tokenize(text)  # Tokeniza el texto y calcula spans
        X = self._encode(tokens)  # Convierte los tokens en IDs

        probs = self._predict_proba(X)  # Calcula las probabilidades de las etiquetas
        real_T = min(len(tokens), self.max_len)  # Ajusta el número de tokens procesados
        probs = probs[:real_T, :]

        # Decodifica las etiquetas predichas
        pred_ids = probs.argmax(axis=-1).astype(int).tolist()
        pred_tags = [self.idx2tag.get(i, "O") for i in pred_ids]

        # Convierte las etiquetas BIO en entidades estructuradas
        entities = self._bio_decode(pred_tags, tokens[:real_T], spans[:real_T], probs)

        return [ent.to_dict() for ent in entities]
