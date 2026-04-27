"""
CRF-Based Named Entity Recognition for Clinical Text.

This module implements inference for a pre-trained sklearn-crfsuite model
using BIO tagging. It mirrors the MedAI NER output contract (entity spans with
character offsets) and enforces strict compatibility with notebook_v2 features.
"""

from __future__ import annotations

import json
import logging
import pickle
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import spacy
from spacy.tokens import Doc

try:
    import joblib
except Exception:  # pragma: no cover
    joblib = None  # type: ignore


logger = logging.getLogger(__name__)

EXPECTED_FEATURE_SCHEMA = "notebook_v2"
EXPECTED_TOKENIZER_SCHEMA = "regex_word_punct_v1"
_NOTEBOOK_V2_SIGNATURE_PREFIX = "shape.compact:"

# Equivalent token sequence to notebook's re.split(r"(\W)") with empty filtering,
# but preserves exact character offsets via finditer for entity span reconstruction.
_TOKEN_REGEX = re.compile(r"\w+|[^\w\s]", re.UNICODE)

# ---------------------------------------------------------------------------
# UNIT_ANCHORS — must match notebook_v2 exactly
# ---------------------------------------------------------------------------
UNIT_ANCHORS = {
    "pressure": {"mmhg", "cmh2o", "cmh20"},
    "volume": {
        "ml",
        "cc",
        "l",
        "lt",
        "lts",
        "litro",
        "litros",
        "ml/kg",
        "mlkg",
        "ml/h",
        "l/h",
    },
    "saturation": {"%", "porcentaje", "sat", "spo2", "sao2"},
    "resp_rate": {"rpm", "resp/min", "/min", "min-1"},
    "heart_rate": {"lpm", "bpm", "lat/min", "pm"},
    "temperature": {"oc", "co", "grados", "celsius"},
    "chemistry": {"mmol/l", "meq/l", "mg/dl", "g/dl"},
    "weight": {"kg", "kgs", "kilo", "kilos"},
    "height": {"cm", "mts", "metro", "metros"},
    "age": {"años"},
}

# ---------------------------------------------------------------------------
# CLINICAL_GAZETTEERS — must match notebook_v2 exactly
# ---------------------------------------------------------------------------
CLINICAL_GAZETTEERS: Dict[str, set] = {
    "vent_config": {
        "ac",
        "vc",
        "pc",
        "simv",
        "cpap",
        "bipap",
        "psv",
        "ps",
        "aprv",
        "hfov",
        "nava",
        "prvc",
        "vcrp",
        "acv",
        "vcv",
        "pav",
        "prcv",
        "blv",
        "peep",
        "fio2",
        "fi02",
        "vt",
        "fr",
        "vol",
        "volt",
        "ti",
        "te",
        "ie",
    },
    "vent_response": {
        "sao2",
        "so2",
        "sat",
        "spo2",
        "saturacion",
    },
    "anthropometric": {
        "peso",
        "talla",
        "imc",
        "edad",
    },
    "vital_sign": {
        "fc",
        "pa",
        "pas",
        "pad",
        "pam",
        "tam",
        "ta",
        "temp",
        "temperatura",
        "glucometria",
        "glucometrias",
        "glicemia",
        "glucosa",
    },
    "observation": {
        "cabecera",
        "supino",
        "supina",
        "supinacion",
        "prono",
        "pronacion",
        "decubito",
        "semifowler",
        "trendelenburg",
        "sedente",
        "postura",
        "posicion",
        "posicionamiento",
    },
    "blood_gas": {
        "ph",
        "pco2",
        "paco2",
        "po2",
        "pao2",
        "hco3",
        "be",
        "eb",
        "pafi",
        "lactato",
    },
}

# ---------------------------------------------------------------------------
# Numeric regex patterns — must match notebook_v2 exactly
# ---------------------------------------------------------------------------
_NUMERIC_RE = re.compile(r"^[+-]?\d+(?:[\.,]\d+)?$")
_DECIMAL_RE = re.compile(r"^[+-]?\d+[\.,]\d+$")
_BP_RATIO_RE = re.compile(r"^\d{2,3}/\d{2,3}$")
_IE_RATIO_RE = re.compile(r"^\d+(?:[\.,]\d+)?:\d+(?:[\.,]\d+)?$")
_PERCENT_RE = re.compile(r"^[+-]?\d+(?:[\.,]\d+)?%$")

# ---------------------------------------------------------------------------
# spaCy POS tagger (loaded once at module level, same config as notebook_v2)
# ---------------------------------------------------------------------------
_NLP_POS = spacy.load(
    "es_core_news_sm",
    disable=["ner", "parser", "lemmatizer", "senter", "attribute_ruler"],
)
logger.info("spaCy POS tagger cargado: es_core_news_sm (solo morphologizer)")


# ===========================================================================
# Helper functions
# ===========================================================================


def _pos_tag_sentence(sent: List[str]) -> List[str]:
    """Obtiene POS tags para una oración pre-tokenizada usando spaCy."""
    doc = Doc(_NLP_POS.vocab, words=sent)
    for pipe_name in ["tok2vec", "morphologizer"]:
        _NLP_POS.get_pipe(pipe_name)(doc)
    return [token.pos_ for token in doc]


def _tokenize(text: str) -> Tuple[List[str], List[Tuple[int, int]]]:
    tokens: List[str] = []
    spans: List[Tuple[int, int]] = []
    for m in _TOKEN_REGEX.finditer(text):
        tokens.append(m.group(0))
        spans.append((m.start(), m.end()))
    return tokens, spans


def _normalize_token(token: str) -> str:
    token = token.lower().strip()
    token = token.replace("²", "2").replace("°", "o")
    token = token.strip(".,;()[]{}\\")
    return token


def _word_shape(token: str) -> str:
    """Representación compacta de forma de palabra para patrones clínicos."""
    tags = []
    for ch in token:
        if ch.isdigit():
            t = "d"
        elif ch.isupper():
            t = "X"
        elif ch.islower():
            t = "x"
        elif ch in "/:.,%+-":  # matches notebook_v2
            t = ch
        else:
            t = "o"
        if not tags or tags[-1] != t:
            tags.append(t)
    return "".join(tags)[:10]


def _gazetteer_lookup(token: str) -> Dict[str, bool]:
    """Busca un token en todos los gazetteers clínicos."""
    tok = _normalize_token(token)
    return {f"gaz.{cat}": tok in vocab for cat, vocab in CLINICAL_GAZETTEERS.items()}


def _classify_anchor_unit(token: str) -> Optional[str]:
    tok = _normalize_token(token)

    for category, vocab in UNIT_ANCHORS.items():
        if tok in vocab:
            return category

    # Patrones adicionales no cubiertos por el diccionario (notebook_v2)
    if tok.endswith("%"):
        return "saturation"
    if _IE_RATIO_RE.fullmatch(tok):
        return "ratio_ie"

    return None


def _scan_next_anchors(
    sent: Sequence[str], i: int, max_offset: int = 3
) -> Tuple[Dict[str, bool], Optional[int]]:
    categories = {
        "pressure": False,
        "volume": False,
        "saturation": False,
        "resp_rate": False,
        "heart_rate": False,
        "temperature": False,
        "chemistry": False,
        "weight": False,
        "height": False,
        "age": False,
        "ratio_ie": False,
    }

    first_distance: Optional[int] = None

    for offset in range(1, max_offset + 1):
        j = i + offset
        if j >= len(sent):
            break
        unit_cat = _classify_anchor_unit(sent[j])
        if unit_cat is None:
            continue

        categories[unit_cat] = True
        if first_distance is None:
            first_distance = offset

    return categories, first_distance


def word2features(
    tokens: Sequence[str],
    i: int,
    pos_tags: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Extrae features para un token en una oración.

    Incluye forma de palabra, contexto local, patrones simbólicos,
    anclaje semántico por unidades, POS tags y gazetteers clínicos.
    Debe ser idéntica a word2features() del notebook_v2.
    """
    word = tokens[i]

    # --- Core token features (notebook_v2 SIMPLE: 11 claves) ---
    # Eliminadas por redundancia: isalpha, isalnum, mixed_case, len,
    # shape.has_slash/colon/dot/comma/decimal_sep (contenidas en shape.compact),
    # shape.bp_ratio_like/ie_ratio_like/percent_like (contenidas en shape.compact),
    # num.is_candidate (subsumido por has_digit), num.is_decimal (contenido en shape.compact).
    features: Dict[str, Any] = {
        "bias": 1.0,
        "word.lower()": word.lower(),
        "word[-3:]": word[-3:],
        "word[-2:]": word[-2:],
        "word[:3]": word[:3],
        "word[:2]": word[:2],
        "word.isupper()": word.isupper(),
        "word.isdigit()": word.isdigit(),
        "word.has_digit": any(c.isdigit() for c in word),
        "word.has_hyphen": "-" in word,
        "shape.compact": _word_shape(word),
    }

    # --- POS tags (spaCy es_core_news_sm, morphologizer only) ---
    if pos_tags is not None:
        features["pos"] = pos_tags[i]
        if i > 0:
            features["-1:pos"] = pos_tags[i - 1]
        if i > 1:
            features["-2:pos"] = pos_tags[i - 2]
        if i < len(tokens) - 1:
            features["+1:pos"] = pos_tags[i + 1]
        if i < len(tokens) - 2:
            features["+2:pos"] = pos_tags[i + 2]

    # --- Gazetteers clínicos (token actual) ---
    features.update(_gazetteer_lookup(word))

    # --- Contexto: token anterior i-1 ---
    if i > 0:
        word1 = tokens[i - 1]
        features.update(
            {
                "-1:word.lower()": word1.lower(),
                "-1:word.isupper()": word1.isupper(),
                "-1:word.isdigit()": word1.isdigit(),
                "-1:word[-3:]": word1[-3:],
                "-1:word[-2:]": word1[-2:],
            }
        )
        features.update({f"-1:{k}": v for k, v in _gazetteer_lookup(word1).items()})
    else:
        features["BOS"] = True

    # --- Contexto: dos tokens anteriores i-2 (solo lower + sufijo 3 chars) ---
    if i > 1:
        word2 = tokens[i - 2]
        features.update(
            {
                "-2:word.lower()": word2.lower(),
                "-2:word[-3:]": word2[-3:],
            }
        )

    # --- Contexto: token siguiente i+1 ---
    if i < len(tokens) - 1:
        word1 = tokens[i + 1]
        features.update(
            {
                "+1:word.lower()": word1.lower(),
                "+1:word.isupper()": word1.isupper(),
                "+1:word.isdigit()": word1.isdigit(),
                "+1:word[-3:]": word1[-3:],
                "+1:word[-2:]": word1[-2:],
            }
        )
        features.update({f"+1:{k}": v for k, v in _gazetteer_lookup(word1).items()})
    else:
        features["EOS"] = True

    # --- Contexto: dos tokens siguientes i+2 (simétrico a -2) ---
    if i < len(tokens) - 2:
        word2 = tokens[i + 2]
        features.update(
            {
                "+2:word.lower()": word2.lower(),
                "+2:word[-3:]": word2[-3:],
            }
        )

    # --- Anclaje semántico por unidades en ventana t+1..t+3 ---
    # Disparo: cualquier token con al menos un dígito (notebook_v2 SIMPLE).
    if any(c.isdigit() for c in word):
        anchor_flags, anchor_distance = _scan_next_anchors(tokens, i, max_offset=3)
        features.update(
            {
                "num.anchor.next.pressure": anchor_flags["pressure"],
                "num.anchor.next.volume": anchor_flags["volume"],
                "num.anchor.next.saturation": anchor_flags["saturation"],
                "num.anchor.next.resp_rate": anchor_flags["resp_rate"],
                "num.anchor.next.heart_rate": anchor_flags["heart_rate"],
                "num.anchor.next.temperature": anchor_flags["temperature"],
                "num.anchor.next.chemistry": anchor_flags["chemistry"],
                "num.anchor.next.weight": anchor_flags["weight"],
                "num.anchor.next.height": anchor_flags["height"],
                "num.anchor.next.age": anchor_flags["age"],
                "num.anchor.next.ratio_ie": anchor_flags["ratio_ie"],
                "num.anchor.next.d1": anchor_distance == 1,
                "num.anchor.next.d2": anchor_distance == 2,
                "num.anchor.next.d3": anchor_distance == 3,
            }
        )

    return features


def _collect_run(tags: Sequence[str], start_idx: int, ent_type: str) -> Tuple[int, int]:
    j = start_idx + 1
    while j < len(tags) and tags[j] == f"I-{ent_type}":
        j += 1
    return start_idx, j


def _bio_to_entities(
    text: str,
    tags: Sequence[str],
    spans: Sequence[Tuple[int, int]],
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    i = 0
    T = min(len(tags), len(spans))

    while i < T:
        tag = tags[i]
        if not tag or tag == "O":
            i += 1
            continue

        if tag.startswith(("B-", "I-")):
            ent_type = tag[2:]
            i0, j0 = _collect_run(tags, i, ent_type)
            start_char = spans[i0][0]
            end_char = spans[min(j0 - 1, T - 1)][1]

            if 0 <= start_char < end_char <= len(text):
                out.append(
                    {
                        "type": ent_type,
                        "text": text[start_char:end_char],
                        "start": int(start_char),
                        "end": int(end_char),
                        "code": None,
                    }
                )
            i = j0
            continue

        i += 1

    return out


class CRFExtractor:
    """
    sklearn-crfsuite CRF extractor for clinical NER.

    It validates that model artifacts match the supported feature/tokenizer
    schemas before serving inference requests.
    """

    def __init__(self, model_dir: str | None = None) -> None:
        base_dir = Path(__file__).resolve().parent
        hard_path = Path("/app/models/model")
        default_dir = (
            hard_path if hard_path.exists() else (base_dir.parent / "models" / "model")
        )
        resolved = Path(model_dir or default_dir).expanduser().resolve()
        if not resolved.exists():
            raise FileNotFoundError(f"Model directory not found: {resolved}")

        self.model_dir = resolved

        cfg_path = self.model_dir / "config.json"
        model_path = self.model_dir / "crf_model.pkl"

        if cfg_path.exists():
            self.config = json.loads(cfg_path.read_text(encoding="utf-8"))
        else:
            self.config = {}

        self.feature_schema = self._required_schema_field("feature_schema")
        self.tokenizer_schema = self._required_schema_field("tokenizer_schema")
        self._validate_schema_compatibility()

        if not model_path.exists():
            raise FileNotFoundError(f"CRF model file not found: {model_path}")

        self.model = self._load_pickle(model_path)

        if not hasattr(self.model, "predict"):
            raise TypeError(
                "Loaded object does not implement predict(). "
                "Expected a sklearn-crfsuite CRF estimator."
            )

        self._validate_model_signature(self.model)

    def _required_schema_field(self, key: str) -> str:
        value = self.config.get(key)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(
                f"Invalid model config: missing required field '{key}' with non-empty string value"
            )
        return value

    def _validate_schema_compatibility(self) -> None:
        if self.feature_schema != EXPECTED_FEATURE_SCHEMA:
            raise ValueError(
                "Unsupported feature schema in model config. "
                f"Expected '{EXPECTED_FEATURE_SCHEMA}', got '{self.feature_schema}'"
            )
        if self.tokenizer_schema != EXPECTED_TOKENIZER_SCHEMA:
            raise ValueError(
                "Unsupported tokenizer schema in model config. "
                f"Expected '{EXPECTED_TOKENIZER_SCHEMA}', got '{self.tokenizer_schema}'"
            )

    @staticmethod
    def _validate_model_signature(model: Any) -> None:
        state_features = getattr(model, "state_features_", None)
        if state_features is None or not hasattr(state_features, "keys"):
            raise ValueError(
                "Loaded model does not expose state_features_ required for schema validation"
            )

        for key in state_features.keys():
            if not isinstance(key, tuple) or not key:
                continue
            attr = key[0]
            if isinstance(attr, str) and attr.startswith(_NOTEBOOK_V2_SIGNATURE_PREFIX):
                return

        raise ValueError(
            "Model artifact signature mismatch for feature schema "
            f"'{EXPECTED_FEATURE_SCHEMA}': expected CRFsuite attributes with prefix "
            f"'{_NOTEBOOK_V2_SIGNATURE_PREFIX}'"
        )

    @staticmethod
    def _load_pickle(path: Path):
        if joblib is not None:
            try:
                return joblib.load(path)
            except Exception:
                pass
        with path.open("rb") as f:
            return pickle.load(f)

    def predict(
        self,
        text: str,
    ) -> List[Dict[str, Any]]:
        if not text:
            return []

        tokens, spans = _tokenize(text)
        if not tokens:
            return []

        pos_tags = _pos_tag_sentence(tokens)
        X = [word2features(tokens, i, pos_tags=pos_tags) for i in range(len(tokens))]

        try:
            y_pred = self.model.predict([X])[0]
        except Exception as e:
            logger.error("CRF prediction failed: %s", e, exc_info=True)
            raise

        return _bio_to_entities(text, y_pred, spans)

    def meta(self) -> Dict[str, Any]:
        return {
            "extractor": "crf",
            "model_dir": str(self.model_dir),
            "n_tags": self.config.get("n_tags"),
            "feature_schema": self.feature_schema,
            "tokenizer_schema": self.tokenizer_schema,
        }
