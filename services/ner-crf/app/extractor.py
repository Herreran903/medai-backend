"""
CRF-Based Named Entity Recognition for Clinical Text.

This module implements inference for a pre-trained sklearn-crfsuite model
using BIO tagging. It mirrors the MedAI NER output contract (entity spans with
character offsets).

Model Input:
    The CRF expects a sequence of per-token feature dictionaries.
    Tokenization uses a regex aligned with the BiLSTM extractor:
    - Words and punctuation are separated
    - Character offsets are preserved for span reconstruction
"""

from __future__ import annotations

import json
import logging
import pickle
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

try:
    import joblib
except Exception:  # pragma: no cover
    joblib = None  # type: ignore


logger = logging.getLogger(__name__)

# Tokenization regex: matches words and punctuation separately
_TOKEN_REGEX = re.compile(r"\w+|[^\w\s]", re.UNICODE)


def _tokenize(text: str) -> Tuple[List[str], List[Tuple[int, int]]]:
    tokens: List[str] = []
    spans: List[Tuple[int, int]] = []
    for m in _TOKEN_REGEX.finditer(text):
        tokens.append(m.group(0))
        spans.append((m.start(), m.end()))
    return tokens, spans


def _has_any(it: Iterable[bool]) -> bool:
    return any(bool(v) for v in it)


def _has_upper(s: str) -> bool:
    return _has_any(c.isupper() for c in s)


def _has_lower(s: str) -> bool:
    return _has_any(c.islower() for c in s)


def _has_digit(s: str) -> bool:
    return _has_any(c.isdigit() for c in s)


def _word_features(word: str) -> Dict[str, Any]:
    """
    Feature set aligned with the training-time schema embedded in the pickled model.

    Note:
        String-valued features become `key:value` attributes internally (CRFsuite).
    """
    return {
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
        "word.has_digit": _has_digit(word),
        "word.has_upper": _has_upper(word),
        "word.all_caps": word.isupper(),
        "word.mixed_case": _has_upper(word) and _has_lower(word),
    }


def _token2features(tokens: Sequence[str], i: int) -> Dict[str, Any]:
    """
    Build CRF features for token at position i.

    Uses a 2-token context window (-2, -1, +1, +2) and BOS/EOS markers.
    """
    word = tokens[i]
    features: Dict[str, Any] = {"bias": 1.0}
    features.update(_word_features(word))

    if i == 0:
        features["BOS"] = True
    if i == len(tokens) - 1:
        features["EOS"] = True

    for offset in (-2, -1, 1, 2):
        j = i + offset
        if not (0 <= j < len(tokens)):
            continue
        wj = tokens[j]
        sign = f"{offset:+d}"
        for k, v in _word_features(wj).items():
            features[f"{sign}:{k}"] = v
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
    *,
    restrict_types: Optional[set[str]] = None,
) -> List[Dict[str, Any]]:
    """
    Convert BIO tag sequence to entity spans.
    """
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
                if restrict_types is None or ent_type in restrict_types:
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

    Loads a pre-trained CRF model from a pickle file and runs inference over
    regex-tokenized text, returning entity spans aligned to the original input.
    """

    def __init__(self, model_dir: str | None = None) -> None:
        base_dir = Path(__file__).resolve().parent
        hard_path = Path("/app/models/model")
        default_dir = hard_path if hard_path.exists() else (base_dir.parent / "models" / "model")
        resolved = Path(model_dir or default_dir).expanduser().resolve()
        if not resolved.exists():
            raise FileNotFoundError(f"Model directory not found: {resolved}")

        self.model_dir = resolved

        cfg_path = self.model_dir / "config.json"
        tag2idx_path = self.model_dir / "tag2idx.json"
        model_path = self.model_dir / "crf_model.pkl"

        if cfg_path.exists():
            self.config = json.loads(cfg_path.read_text(encoding="utf-8"))
        else:
            self.config = {}

        if tag2idx_path.exists():
            self.tag2idx: Dict[str, int] = json.loads(
                tag2idx_path.read_text(encoding="utf-8")
            )
        else:
            self.tag2idx = {}

        if not model_path.exists():
            raise FileNotFoundError(f"CRF model file not found: {model_path}")

        self.model = self._load_pickle(model_path)

        if not hasattr(self.model, "predict"):
            raise TypeError(
                "Loaded object does not implement predict(). "
                "Expected a sklearn-crfsuite CRF estimator."
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
        *,
        restrict_types: Optional[Sequence[str]] = None,
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

        restrict_set = set(restrict_types) if restrict_types else None
        return _bio_to_entities(text, y_pred, spans, restrict_types=restrict_set)

    def meta(self) -> Dict[str, Any]:
        return {
            "extractor": "crf",
            "model_dir": str(self.model_dir),
            "n_tags": self.config.get("n_tags"),
        }

