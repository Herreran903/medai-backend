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

try:
    import joblib
except Exception:  # pragma: no cover
    joblib = None  # type: ignore


logger = logging.getLogger(__name__)

EXPECTED_FEATURE_SCHEMA = "notebook_v2"
EXPECTED_TOKENIZER_SCHEMA = "regex_word_punct_v1"
_NOTEBOOK_V2_SIGNATURE_PREFIX = "shape.compact:"

# Equivalent token sequence to notebook's re.split(r"(\\W)") with empty filtering,
# but preserves exact character offsets via finditer for entity span reconstruction.
_TOKEN_REGEX = re.compile(r"\w+|[^\w\s]", re.UNICODE)

UNIT_ANCHORS = {
    "pressure": {"mmhg", "cmh2o", "cmh20"},
    "volume": {"ml", "cc", "l", "lt", "lts", "litro", "litros", "ml/kg", "mlkg"},
    "saturation": {"%", "porcentaje", "sat", "spo2", "sao2"},
    "oxygen_fraction": {"fio2"},
    "resp_rate": {"rpm", "resp/min", "/min", "min-1"},
    "heart_rate": {"lpm", "bpm", "lat/min"},
    "temperature": {"oc", "co", "grados", "celsius"},
    "chemistry": {"mmol/l", "meq/l", "mg/dl", "g/dl"},
    "weight": {"kg", "kgs", "kilo", "kilos"},
    "height": {"cm", "mts", "metro", "metros"},
}

_NUMERIC_RE = re.compile(r"^[+-]?\d+(?:[\.,]\d+)?$")
_DECIMAL_RE = re.compile(r"^[+-]?\d+[\.,]\d+$")
_BP_RATIO_RE = re.compile(r"^\d{2,3}/\d{2,3}$")
_IE_RATIO_RE = re.compile(r"^\d+(?:[\.,]\d+)?:\d+(?:[\.,]\d+)?$")
_PERCENT_RE = re.compile(r"^[+-]?\d+(?:[\.,]\d+)?%$")


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
    token = token.strip(".,;()[]{}")
    return token


def _word_shape(token: str) -> str:
    tags = []
    for ch in token:
        if ch.isdigit():
            t = "d"
        elif ch.isupper():
            t = "X"
        elif ch.islower():
            t = "x"
        elif ch in "/:.,%+-":
            t = ch
        else:
            t = "o"
        if not tags or tags[-1] != t:
            tags.append(t)
    return "".join(tags)[:24]


def _classify_anchor_unit(token: str) -> Optional[str]:
    tok = _normalize_token(token)

    for category, vocab in UNIT_ANCHORS.items():
        if tok in vocab:
            return category

    if re.fullmatch(r"(mmhg|cmh2o|cmh20)", tok):
        return "pressure"
    if re.fullmatch(r"(ml|min|l|min|ml/h|l/h)", tok):
        return "volume"
    if tok == "%" or tok.endswith("%"):
        return "saturation"
    if re.fullmatch(r"(rpm|bpm|lpm)", tok):
        return "resp_rate"
    if _IE_RATIO_RE.fullmatch(tok):
        return "ratio_ie"

    return None


def _scan_next_anchors(
    sent: Sequence[str], i: int, max_offset: int = 3
) -> Tuple[Dict[str, bool], Optional[str], Optional[int]]:
    categories = {
        "pressure": False,
        "volume": False,
        "saturation": False,
        "oxygen_fraction": False,
        "resp_rate": False,
        "heart_rate": False,
        "temperature": False,
        "chemistry": False,
        "weight": False,
        "height": False,
        "ratio_ie": False,
    }

    first_unit = None
    first_distance = None

    for offset in range(1, max_offset + 1):
        j = i + offset
        if j >= len(sent):
            break
        unit_cat = _classify_anchor_unit(sent[j])
        if unit_cat is None:
            continue

        categories[unit_cat] = True
        if first_unit is None:
            first_unit = _normalize_token(sent[j])
            first_distance = offset

    return categories, first_unit, first_distance


def _token2features(tokens: Sequence[str], i: int) -> Dict[str, Any]:
    word = tokens[i]
    word_norm = _normalize_token(word)

    is_numeric_candidate = (
        bool(_NUMERIC_RE.fullmatch(word_norm))
        or bool(_BP_RATIO_RE.fullmatch(word_norm))
        or bool(_IE_RATIO_RE.fullmatch(word_norm))
        or bool(_PERCENT_RE.fullmatch(word_norm))
        or any(c.isdigit() for c in word)
    )

    features: Dict[str, Any] = {
        "bias": 1.0,
        "word.lower()": word.lower(),
        "word[-3:]": word[-3:],
        "word[-2:]": word[-2:],
        "word[:3]": word[:3],
        "word[:2]": word[:2],
        "word.isupper()": word.isupper(),
        "word.istitle()": word.istitle(),
        "word.isdigit()": word.isdigit(),
        "word.isalpha()": word.isalpha(),
        "word.isalnum()": word.isalnum(),
        "word.len": len(word),
        "word.has_hyphen": "-" in word,
        "word.has_digit": any(c.isdigit() for c in word),
        "word.has_upper": any(c.isupper() for c in word),
        "word.all_caps": word.isupper() and len(word) > 1,
        "word.mixed_case": (
            not word.islower() and not word.isupper() and word.isalpha()
        ),
        "shape.compact": _word_shape(word),
        "shape.has_slash": "/" in word,
        "shape.has_colon": ":" in word,
        "shape.has_dot": "." in word,
        "shape.has_comma": "," in word,
        "shape.has_decimal_sep": bool(re.search(r"\d[\.,]\d", word)),
        "shape.bp_ratio_like": bool(_BP_RATIO_RE.fullmatch(word_norm)),
        "shape.ie_ratio_like": bool(_IE_RATIO_RE.fullmatch(word_norm)),
        "shape.percent_like": bool(_PERCENT_RE.fullmatch(word_norm)),
        "num.is_candidate": is_numeric_candidate,
        "num.is_decimal": bool(_DECIMAL_RE.fullmatch(word_norm)),
    }

    if i > 0:
        word1 = tokens[i - 1]
        features.update(
            {
                "-1:word.lower()": word1.lower(),
                "-1:word.istitle()": word1.istitle(),
                "-1:word.isupper()": word1.isupper(),
                "-1:word.isdigit()": word1.isdigit(),
                "-1:word[-3:]": word1[-3:],
                "-1:word[-2:]": word1[-2:],
            }
        )
    else:
        features["BOS"] = True

    if i > 1:
        word2 = tokens[i - 2]
        features.update(
            {
                "-2:word.lower()": word2.lower(),
                "-2:word.istitle()": word2.istitle(),
                "-2:word.isupper()": word2.isupper(),
            }
        )

    if i < len(tokens) - 1:
        word1 = tokens[i + 1]
        features.update(
            {
                "+1:word.lower()": word1.lower(),
                "+1:word.istitle()": word1.istitle(),
                "+1:word.isupper()": word1.isupper(),
                "+1:word.isdigit()": word1.isdigit(),
                "+1:word[-3:]": word1[-3:],
                "+1:word[-2:]": word1[-2:],
            }
        )
    else:
        features["EOS"] = True

    if i < len(tokens) - 2:
        word2 = tokens[i + 2]
        features.update(
            {
                "+2:word.lower()": word2.lower(),
                "+2:word.istitle()": word2.istitle(),
                "+2:word.isupper()": word2.isupper(),
            }
        )

    if is_numeric_candidate:
        anchor_flags, anchor_unit, anchor_distance = _scan_next_anchors(
            tokens, i, max_offset=3
        )
        features.update(
            {
                "num.anchor.any_next": any(anchor_flags.values()),
                "num.anchor.next.pressure": anchor_flags["pressure"],
                "num.anchor.next.volume": anchor_flags["volume"],
                "num.anchor.next.saturation": anchor_flags["saturation"],
                "num.anchor.next.oxygen_fraction": anchor_flags["oxygen_fraction"],
                "num.anchor.next.resp_rate": anchor_flags["resp_rate"],
                "num.anchor.next.heart_rate": anchor_flags["heart_rate"],
                "num.anchor.next.temperature": anchor_flags["temperature"],
                "num.anchor.next.chemistry": anchor_flags["chemistry"],
                "num.anchor.next.weight": anchor_flags["weight"],
                "num.anchor.next.height": anchor_flags["height"],
                "num.anchor.next.ratio_ie": anchor_flags["ratio_ie"],
                "num.anchor.next.unit": anchor_unit if anchor_unit else "__NONE__",
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

        X = [_token2features(tokens, i) for i in range(len(tokens))]

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
