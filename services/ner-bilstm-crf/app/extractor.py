"""
BiLSTM-CRF Named Entity Recognition for Clinical Text.

This extractor serves the BiLSTM-CRF notebook_v2 pipeline only.
Legacy artifacts are intentionally rejected at startup.
"""

from __future__ import annotations

import json
import logging
import re
import unicodedata
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import tensorflow as tf

try:
    from tensorflow_addons.text import CRF
except Exception:  # pragma: no cover
    CRF = None  # type: ignore

try:
    from mwrapper import ModelWithCRFLoss, ModelWithCRFLossDSCLoss
except Exception:  # pragma: no cover
    ModelWithCRFLoss = None  # type: ignore
    ModelWithCRFLossDSCLoss = None  # type: ignore

from shared.schemas import Entity

logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    fmt = logging.Formatter("[%(levelname)s] %(name)s: %(message)s")
    handler.setFormatter(fmt)
    logger.addHandler(handler)
logger.setLevel(logging.WARNING)


EXPECTED_FEATURE_SCHEMA = "notebook_v2"
EXPECTED_TOKENIZER_SCHEMA = "regex_word_punct_v1"

CAP_PAD_TOKEN = "<PAD_CAP>"
SPECIAL_VOCAB_TOKENS = {"<PAD>", "<UNK>", "[UNK]"}


def _normalize_text_soft(text: str) -> str:
    text = unicodedata.normalize("NFC", str(text))
    text = text.replace("\u00a0", " ")
    text = re.sub(r"[\u200B-\u200D\uFEFF]", "", text)
    text = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F-\x9F]", " ", text)
    text = re.sub(r"[^\S\r\n]+", " ", text)
    text = re.sub(r" *\n+ *", "\n", text)
    return text.strip()


def _normalize_token_soft(token: str) -> str:
    token = unicodedata.normalize("NFC", str(token))
    token = token.replace("\u00a0", " ")
    token = re.sub(r"[\u200B-\u200D\uFEFF]", "", token)
    token = re.sub(r"[\x00-\x1F\x7F-\x9F]", "", token)
    token = re.sub(r"\s+", " ", token)
    return token.strip()


def _normalize_lexical_token(token: str) -> str:
    if token in SPECIAL_VOCAB_TOKENS:
        return token
    token = _normalize_token_soft(token)
    return token.lower() if token else token


def _tokenize_text(text: str) -> List[str]:
    text = _normalize_text_soft(text)
    tokens: List[str] = []
    for chunk in re.split(r"(\W)", text):
        chunk = _normalize_token_soft(chunk)
        if chunk:
            tokens.append(chunk)
    return tokens


def _compute_offsets(text: str, tokens: Sequence[str]) -> List[Tuple[int, int]]:
    spans: List[Tuple[int, int]] = []
    cursor = 0
    lowered = text.lower()

    for token in tokens:
        start = text.find(token, cursor)
        if start == -1:
            start = lowered.find(token.lower(), cursor)
        if start == -1:
            start = cursor
        end = start + len(token)
        spans.append((start, end))
        cursor = end

    return spans


def _get_capitalization_feature(token: str) -> str:
    token = _normalize_token_soft(token)
    if not token:
        return "other"

    if any(ch.isdigit() for ch in token):
        return "has_digit"

    alpha_chars = [ch for ch in token if ch.isalpha()]
    if alpha_chars:
        if all(ch.islower() for ch in alpha_chars):
            return "all_lower"
        if all(ch.isupper() for ch in alpha_chars):
            return "all_upper"
        if alpha_chars[0].isupper() and all(ch.islower() for ch in alpha_chars[1:]):
            return "initial_upper"
        return "mixed_case"

    if all(not ch.isalnum() for ch in token):
        return "punctuation"
    return "other"


def _join_notebook_tokens(tokens: Sequence[str]) -> str:
    return " ".join(tokens).strip()


class LSTMExtractor:
    """BiLSTM-CRF extractor aligned with notebook_v2 preprocessing."""

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

        self.config: Dict[str, Any] = json.loads(
            (self.model_dir / "config.json").read_text(encoding="utf-8")
        )
        self.feature_schema = self._required_str_field("feature_schema")
        self.tokenizer_schema = self._required_str_field("tokenizer_schema")
        self.use_capitalization_feature = self._optional_bool_field(
            "use_capitalization_feature", default=True
        )
        self._validate_schema_compatibility()

        self.word2idx: Dict[str, int] = json.loads(
            (self.model_dir / "word2idx.json").read_text(encoding="utf-8")
        )
        self.tag2idx: Dict[str, int] = json.loads(
            (self.model_dir / "tag2idx.json").read_text(encoding="utf-8")
        )
        self.idx2tag: Dict[int, str] = {int(v): k for k, v in self.tag2idx.items()}
        self._validate_vocab_signature()

        self.max_len: int = int(self.config["max_len"])
        self.pad_id: int = int(self.config.get("pad_id", self.word2idx.get("<PAD>", 0)))
        self.unk_id: int = int(
            self.word2idx.get("<UNK>", self.word2idx.get("[UNK]", 1))
        )
        self.cap_pad_id: int = int(self.config.get("cap_pad_id", 0))

        self.cap2idx: Dict[str, int] = {}
        if self.use_capitalization_feature:
            cap_path = self.model_dir / "cap2idx.json"
            if not cap_path.exists():
                raise FileNotFoundError(
                    "Capitalization feature is enabled but cap2idx.json is missing"
                )
            self.cap2idx = json.loads(cap_path.read_text(encoding="utf-8"))
            self._validate_cap_config()

        custom_objects: Dict[str, Any] = {}
        if CRF is not None:
            custom_objects["CRF"] = CRF
        if ModelWithCRFLoss is not None:
            custom_objects["ModelWithCRFLoss"] = ModelWithCRFLoss
        if ModelWithCRFLossDSCLoss is not None:
            custom_objects["ModelWithCRFLossDSCLoss"] = ModelWithCRFLossDSCLoss

        keras_path = self.model_dir / "model.keras"
        sm_path = self.model_dir / "saved_model"
        if keras_path.exists():
            model_path = keras_path.as_posix()
        elif sm_path.exists():
            model_path = sm_path.as_posix()
        else:
            raise FileNotFoundError(f"No model found at {keras_path} or {sm_path}")

        logger.info("Loading Keras model from %s", model_path)
        self.model = tf.keras.models.load_model(
            model_path,
            compile=False,
            custom_objects=custom_objects if custom_objects else None,
        )

        self.input_count = self._infer_model_input_count()
        self._validate_model_io_signature()

    def _required_str_field(self, key: str) -> str:
        value = self.config.get(key)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(
                f"Invalid model config: missing required field '{key}' with non-empty string value"
            )
        return value

    def _optional_bool_field(self, key: str, default: bool) -> bool:
        value = self.config.get(key)
        if value is None:
            return default
        if not isinstance(value, bool):
            raise ValueError(
                f"Invalid model config: field '{key}' must be boolean when present"
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

    def _validate_vocab_signature(self) -> None:
        uppercase_examples: List[str] = []
        for token in self.word2idx.keys():
            if token in SPECIAL_VOCAB_TOKENS:
                continue
            if any(ch.isalpha() and ch.isupper() for ch in token):
                uppercase_examples.append(token)
                if len(uppercase_examples) >= 5:
                    break

        if uppercase_examples:
            raise ValueError(
                "Model vocabulary signature mismatch for notebook_v2: expected lowercase "
                "lexical vocabulary built by normalize_lexical_token(). "
                f"Found uppercase tokens, examples={uppercase_examples}"
            )

    def _validate_cap_config(self) -> None:
        required_keys = {
            CAP_PAD_TOKEN,
            "all_lower",
            "all_upper",
            "initial_upper",
            "mixed_case",
            "has_digit",
            "punctuation",
            "other",
        }
        missing = sorted(k for k in required_keys if k not in self.cap2idx)
        if missing:
            raise ValueError(
                "Invalid cap2idx.json for notebook_v2. Missing capitalization keys: "
                f"{missing}"
            )

        cap_pad_id = int(self.cap2idx[CAP_PAD_TOKEN])
        cfg_cap_pad = self.config.get("cap_pad_id")
        if cfg_cap_pad is not None and int(cfg_cap_pad) != cap_pad_id:
            raise ValueError(
                "Invalid config: cap_pad_id does not match cap2idx['<PAD_CAP>']"
            )
        self.cap_pad_id = cap_pad_id

    def _serving_fn(self):
        signatures = getattr(self.model, "signatures", None)
        if signatures:
            return signatures.get("serving_default") or next(iter(signatures.values()))
        return None

    def _infer_model_input_count(self) -> int:
        model_inputs = getattr(self.model, "inputs", None)
        if model_inputs is not None:
            try:
                if len(model_inputs) > 0:
                    return len(model_inputs)
            except Exception:
                pass

        fn = self._serving_fn()
        if fn is not None:
            _, kwargs = fn.structured_input_signature
            if kwargs:
                return len(kwargs)

        return 1

    @staticmethod
    def _first_output(output: Any) -> Any:
        if isinstance(output, dict):
            return next(iter(output.values()))
        if isinstance(output, (list, tuple)):
            return output[0]
        return output

    def _predict_raw(self, X: Any) -> np.ndarray:
        if hasattr(self.model, "predict"):
            raw = self.model.predict(X, verbose=0)
            first = self._first_output(raw)
            arr = np.asarray(first)
        else:
            fn = self._serving_fn()
            if fn is None:
                raise RuntimeError(
                    "Loaded model does not expose predict() or serving signatures"
                )

            if isinstance(X, dict):
                out = fn(**{k: tf.convert_to_tensor(v) for k, v in X.items()})
            elif isinstance(X, (list, tuple)):
                _, kwargs = fn.structured_input_signature
                if kwargs and len(kwargs) == len(X):
                    payload = {
                        name: tf.convert_to_tensor(value)
                        for name, value in zip(kwargs.keys(), X)
                    }
                    out = fn(**payload)
                else:
                    out = fn(tf.convert_to_tensor(X[0]))
            else:
                out = fn(tf.convert_to_tensor(X))

            arr = np.asarray(self._first_output(out))

        if arr.ndim == 3:
            if arr.shape[0] != 1:
                raise ValueError(f"Unexpected batch size in model output: {arr.shape}")
            return arr[0]

        if arr.ndim == 2:
            # CRF decoded output may arrive as (1, max_len)
            if arr.shape[0] == 1 and arr.shape[1] == self.max_len:
                return arr[0]
            return arr

        if arr.ndim == 1:
            return arr

        raise ValueError(f"Unexpected model output shape: {arr.shape}")

    def _validate_model_io_signature(self) -> None:
        expected_count = 2 if self.use_capitalization_feature else 1
        if self.input_count != expected_count:
            raise ValueError(
                "Model input signature mismatch. "
                f"Expected {expected_count} input(s), got {self.input_count}"
            )

        tokens = np.full((1, self.max_len), self.pad_id, dtype=np.int32)
        if self.use_capitalization_feature:
            caps = np.full((1, self.max_len), self.cap_pad_id, dtype=np.int32)
            # SavedModel export for notebook_v2 uses named inputs.
            X: Any = {"tokens": tokens, "caps": caps}
        else:
            X = tokens

        out = self._predict_raw(X)
        if out.ndim == 1:
            if out.shape[0] != self.max_len:
                raise ValueError(
                    "CRF output sequence length mismatch. "
                    f"Expected max_len={self.max_len}, got={out.shape[0]}"
                )
            return

        if out.ndim == 2:
            if out.shape[0] != self.max_len:
                raise ValueError(
                    "Model output sequence length mismatch. "
                    f"Expected max_len={self.max_len}, got={out.shape[0]}"
                )
            return

        raise ValueError(f"Unexpected output rank during model validation: {out.ndim}")

    def _tokenize(self, text: str) -> Tuple[List[str], List[Tuple[int, int]]]:
        tokens = _tokenize_text(text)
        spans = _compute_offsets(text, tokens)
        return tokens, spans

    @staticmethod
    def _pad(ids: List[int], max_len: int, pad_value: int) -> List[int]:
        if len(ids) < max_len:
            return ids + [pad_value] * (max_len - len(ids))
        return ids[:max_len]

    def _encode_tokens(self, tokens: Sequence[str]) -> np.ndarray:
        token_ids = [
            self.word2idx.get(_normalize_lexical_token(token), self.unk_id)
            for token in tokens
        ]
        token_ids = self._pad(token_ids, self.max_len, self.pad_id)
        return np.asarray([token_ids], dtype=np.int32)

    def _encode_caps(self, tokens: Sequence[str]) -> np.ndarray:
        cap_ids = [
            int(self.cap2idx.get(_get_capitalization_feature(token), self.cap_pad_id))
            for token in tokens
        ]
        cap_ids = self._pad(cap_ids, self.max_len, self.cap_pad_id)
        return np.asarray([cap_ids], dtype=np.int32)

    def _build_model_inputs(self, tokens: Sequence[str]) -> Any:
        token_inputs = self._encode_tokens(tokens)
        if not self.use_capitalization_feature:
            return token_inputs
        cap_inputs = self._encode_caps(tokens)
        # Keep named mapping to match SavedModel concrete function signature.
        return {"tokens": token_inputs, "caps": cap_inputs}

    def _bio_decode(
        self,
        tags: Sequence[str],
        tokens: Sequence[str],
        spans: Sequence[Tuple[int, int]],
    ) -> List[Entity]:
        ents: List[Entity] = []
        i = 0
        T = len(spans)

        def _collect_run(start_idx: int, ent_type: str) -> Tuple[int, int]:
            j = start_idx + 1
            while j < T and tags[j] == f"I-{ent_type}":
                j += 1
            return start_idx, j

        def _span_entity(i0: int, j0: int, ent_type: str) -> Entity:
            start_char = spans[i0][0]
            end_char = spans[min(j0 - 1, T - 1)][1]
            return Entity(
                type=ent_type,
                text=_join_notebook_tokens(tokens[i0:j0]),
                start=int(start_char),
                end=int(end_char),
            )

        while i < T:
            tag = tags[i]
            if not tag or tag == "O":
                i += 1
                continue

            if tag.startswith("B-"):
                ent_type = tag[2:]
                i0, j0 = _collect_run(i, ent_type)
                ents.append(_span_entity(i0, j0, ent_type))
                i = j0
                continue

            if tag.startswith("I-"):
                ent_type = tag[2:]
                i0, j0 = _collect_run(i, ent_type)
                ents.append(_span_entity(i0, j0, ent_type))
                i = j0
                continue

            i += 1

        return ents

    def predict(self, text: str) -> List[Dict]:
        if not text:
            return []

        tokens, spans = self._tokenize(text)
        if not tokens:
            return []

        X = self._build_model_inputs(tokens)
        output = self._predict_raw(X)
        real_T = min(len(tokens), self.max_len, output.shape[0])

        if output.ndim == 1:
            pred_ids = output[:real_T].astype(int).tolist()
        else:
            probs = output[:real_T, :]
            pred_ids = probs.argmax(axis=-1).astype(int).tolist()

        pred_tags = [self.idx2tag.get(i, "O") for i in pred_ids]
        entities = self._bio_decode(pred_tags, tokens[:real_T], spans[:real_T])
        return [ent.model_dump() for ent in entities]

    def meta(self) -> Dict:
        return {
            "extractor": "lstm_crf",
            "model_dir": str(self.model_dir),
            "vocab_size": len(self.word2idx),
            "num_tags": len(self.tag2idx),
            "max_len": self.max_len,
            "feature_schema": self.feature_schema,
            "tokenizer_schema": self.tokenizer_schema,
            "use_capitalization_feature": self.use_capitalization_feature,
        }
