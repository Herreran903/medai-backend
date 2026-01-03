"""
BiLSTM-CRF Named Entity Recognition Model for Clinical Text.

This module implements a BiLSTM-based Named Entity Recognition (NER) extractor
for clinical text, specifically trained on mechanical ventilation notes in Spanish.

Architecture Context:
    The LSTM extractor is one of three NER models available in MedAI:

    - **LSTM** (this module): BiLSTM-CRF with BIO tagging, trained on domain data
    - **Transformer**: Fine-tuned BETO/RoBERTa models
    - **LLM**: Claude/GPT-based extraction with structured outputs

    The LSTM model provides fast inference with moderate accuracy, suitable
    for high-throughput scenarios where transformer latency is prohibitive.

Model Architecture:
    The underlying model is a Bidirectional LSTM with:

    - Word embedding layer (vocabulary from training corpus)
    - Bidirectional LSTM layers for sequence encoding
    - Dense layer with softmax for BIO tag prediction
    - Post-processing for BIO to entity span conversion

BIO Tagging Scheme:
    The model uses the BIO (Beginning-Inside-Outside) tagging scheme:

    - ``B-{TYPE}``: Beginning of an entity of type TYPE
    - ``I-{TYPE}``: Inside/continuation of an entity
    - ``O``: Outside any entity (non-entity token)

    Example::

        Token:  "FiO2"  "60"   "%"    "PEEP"  "8"
        Tag:    B-FIO2  I-FIO2 I-FIO2 B-PEEP  I-PEEP

Model Files:
    The model requires the following files in the model directory:

    - ``model.keras`` or ``saved_model/``: TensorFlow model weights
    - ``word2idx.json``: Vocabulary mapping (word -> integer ID)
    - ``tag2idx.json``: Tag mapping (BIO tag -> integer ID)
    - ``config.json``: Model configuration (max_len, pad_id)

Usage:
    >>> from app.models.lstm import LSTMExtractor
    >>> extractor = LSTMExtractor()
    >>> entities = extractor.predict("Paciente con FiO2 60%, PEEP 8 cmH2O")
    >>> print(entities)
    [{"type": "FIO2", "text": "60%", "start": 18, "end": 21, "score": 0.95}]

Configuration:
    The model directory can be specified via constructor or defaults to:
    ``app/models/utils/vmi_enterizacion_ep30_bs32/``

See Also:
    - :mod:`app.services.pipeline` for extraction orchestration
    - :mod:`app.services.registry` for model registration
    - :class:`app.schemas.Entity` for output schema
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import tensorflow as tf

from app.schemas import Entity

# Logger configuration
logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    fmt = logging.Formatter("[%(levelname)s] %(name)s: %(message)s")
    handler.setFormatter(fmt)
    logger.addHandler(handler)
logger.setLevel(logging.WARNING)


# Tokenization regex: matches words and punctuation separately
_TOKEN_REGEX = re.compile(r"\w+|[^\w\s]", re.UNICODE)


def _join_tokens(tokens: Sequence[str]) -> str:
    """
    Reconstruct text from tokens with appropriate spacing.

    Joins tokens into a string, inserting spaces between consecutive
    alphanumeric tokens while preserving punctuation adjacency.

    Args:
        tokens: Sequence of string tokens to join.

    Returns:
        Reconstructed text string.

    Example:
        >>> _join_tokens(["FiO2", "60", "%"])
        'FiO2 60%'
    """
    parts: List[str] = []
    for i, tok in enumerate(tokens):
        if i > 0 and tokens[i - 1].isalnum() and tok.isalnum():
            parts.append(" ")
        parts.append(tok)
    return "".join(parts)


class LSTMExtractor:
    """
    BiLSTM-based clinical Named Entity Recognition extractor.

    This class provides NER extraction using a pre-trained BiLSTM model
    with BIO tagging for mechanical ventilation clinical notes.

    The extractor handles the complete pipeline:

    1. Text tokenization with character offset tracking
    2. Token encoding to vocabulary IDs
    3. Model inference for tag probabilities
    4. BIO decoding to entity spans with confidence scores

    Attributes:
        model_dir: Path to the model directory containing weights and config.
        word2idx: Vocabulary mapping from words to integer IDs.
        tag2idx: Tag mapping from BIO tags to integer IDs.
        idx2tag: Reverse tag mapping from IDs to BIO tags.
        max_len: Maximum sequence length for model input.
        pad_id: Padding token ID for sequences shorter than max_len.
        unk_id: Unknown token ID for out-of-vocabulary words.
        model: Loaded TensorFlow/Keras model instance.

    Example:
        >>> extractor = LSTMExtractor()
        >>> entities = extractor.predict("PEEP 8 cmH2O, FiO2 60%")
        >>> for e in entities:
        ...     print(f"{e['type']}: {e['text']} ({e['score']:.2f})")
        PEEP: 8 cmH2O (0.92)
        FIO2: 60% (0.89)

    Note:
        The model is loaded on instantiation. For lazy loading, use
        :data:`app.services.registry.MODEL_REGISTRY` which defers
        initialization until first use.
    """

    TOKEN_REGEX = _TOKEN_REGEX
    """Compiled regex for tokenization."""

    def __init__(self, model_dir: str | None = None) -> None:
        """
        Initialize the LSTM extractor with model weights and configuration.

        Args:
            model_dir: Path to model directory. If None, uses the default
                path at ``app/models/utils/vmi_enterizacion_ep30_bs32/``.

        Raises:
            FileNotFoundError: If the model directory does not exist.

        Note:
            Model loading may take several seconds on first instantiation.
            The model is loaded into CPU memory by default.
        """
        base_dir = Path(__file__).resolve().parent

        # Support both container and local development paths
        HARD_PATH = Path("/app/app/models/utils/vmi_enterizacion_ep30_bs32")

        default_dir = (
            HARD_PATH
            if HARD_PATH.exists()
            else (base_dir / "models" / "utils" / "vmi_enterizacion_ep30_bs32")
        )
        resolved = Path(model_dir or default_dir).expanduser().resolve()
        if not resolved.exists():
            raise FileNotFoundError(f"Model directory not found: {resolved}")

        self.model_dir = resolved

        # Load vocabulary and tag mappings
        self.word2idx: Dict[str, int] = json.loads(
            (self.model_dir / "word2idx.json").read_text(encoding="utf-8")
        )
        self.tag2idx: Dict[str, int] = json.loads(
            (self.model_dir / "tag2idx.json").read_text(encoding="utf-8")
        )
        self.idx2tag: Dict[int, str] = {int(v): k for k, v in self.tag2idx.items()}

        # Load model configuration
        cfg = json.loads((self.model_dir / "config.json").read_text(encoding="utf-8"))
        self.max_len: int = int(cfg["max_len"])
        self.pad_id: int = int(cfg.get("pad_id", 0))
        self.unk_id: int = int(
            self.word2idx.get("<UNK>", self.word2idx.get("[UNK]", 1))
        )

        # Load model (Keras native or SavedModel format)
        keras_path = self.model_dir / "model.keras"
        if keras_path.exists():
            self.model = tf.keras.models.load_model(
                keras_path.as_posix(), compile=False
            )
            self._is_keras = True
            logger.info("Keras model loaded from %s", keras_path)
        else:
            sm_path = self.model_dir / "saved_model"
            self.model = tf.saved_model.load(sm_path.as_posix())
            self._is_keras = False
            logger.info("SavedModel loaded from %s", sm_path)

    @staticmethod
    def _lower(text: str) -> str:
        """
        Convert text to lowercase for vocabulary lookup.

        Args:
            text: Input text string.

        Returns:
            Lowercase version of the input.
        """
        return text.lower()

    def _tokenize(self, text: str) -> Tuple[List[str], List[Tuple[int, int]]]:
        """
        Tokenize text and compute character spans for each token.

        Uses regex-based tokenization that separates words and punctuation,
        preserving character offsets for entity span reconstruction.

        Args:
            text: Input text to tokenize.

        Returns:
            Tuple of (tokens, spans) where:
            - tokens: List of token strings
            - spans: List of (start, end) character offset tuples

        Example:
            >>> extractor._tokenize("FiO2 60%")
            (['FiO2', '60', '%'], [(0, 4), (5, 7), (7, 8)])
        """
        tokens: List[str] = []
        spans: List[Tuple[int, int]] = []
        for m in self.TOKEN_REGEX.finditer(text):
            tokens.append(m.group(0))
            spans.append((m.start(), m.end()))
        return tokens, spans

    def _encode(self, tokens: Sequence[str]) -> np.ndarray:
        """
        Encode tokens to vocabulary IDs with padding/truncation.

        Converts token strings to integer IDs using the vocabulary mapping.
        Applies padding for short sequences and truncation for long ones.

        Args:
            tokens: Sequence of token strings.

        Returns:
            NumPy array of shape (1, max_len) with token IDs.

        Note:
            Out-of-vocabulary tokens are mapped to the UNK token ID.
        """
        ids = [self.word2idx.get(self._lower(t), self.unk_id) for t in tokens]
        if len(ids) < self.max_len:
            ids = ids + [self.pad_id] * (self.max_len - len(ids))
        else:
            ids = ids[: self.max_len]
        return np.asarray([ids], dtype=np.int32)

    def _predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Compute tag probabilities for encoded input.

        Runs model inference to get probability distributions over
        BIO tags for each token position.

        Args:
            X: Encoded input array of shape (1, max_len).

        Returns:
            Probability array of shape (max_len, num_tags).
        """
        if isinstance(self.model, tf.keras.Model):
            probs = self.model.predict(X, verbose=0)[0]
        else:
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
        Convert BIO tag sequence to entity spans.

        Implements BIO decoding logic to merge consecutive B-/I- tags
        into entity spans with aggregated confidence scores.

        Args:
            tags: Sequence of predicted BIO tags.
            tokens: Original token strings.
            spans: Character offset spans for each token.
            probs: Tag probability matrix for confidence scoring.

        Returns:
            List of Entity objects with type, text, span, and score.

        Algorithm:
            1. Scan for B-{TYPE} tags to start new entities
            2. Extend with consecutive I-{TYPE} tags
            3. Compute mean probability as confidence score
            4. Reconstruct text from token spans
        """
        ents: List[Entity] = []
        i = 0
        T = len(tokens)

        def _collect_run(start_idx: int, ent_type: str) -> Tuple[int, int]:
            """Find the extent of consecutive I-tags for an entity."""
            j = start_idx + 1
            while j < T and tags[j] == f"I-{ent_type}":
                j += 1
            return start_idx, j

        def _span_entity(i0: int, j0: int, ent_type: str) -> Entity:
            """Construct an Entity from token range."""
            start_char = spans[i0][0]
            end_char = spans[min(j0 - 1, T - 1)][1]
            text_str = _join_tokens(tokens[i0:j0])

            # Compute mean probability across entity tokens
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
                i += 1
                continue

            if tag.startswith("B-"):
                ent_type = tag[2:]
                i0, j0 = _collect_run(i, ent_type)
                ents.append(_span_entity(i0, j0, ent_type))
                i = j0
                continue

            if tag.startswith("I-"):
                # Handle orphan I-tags (rare edge case)
                ent_type = tag[2:]
                i0, j0 = _collect_run(i, ent_type)
                ents.append(_span_entity(i0, j0, ent_type))
                i = j0
                continue

            i += 1

        return ents

    def predict(self, text: str) -> List[Dict]:
        """
        Extract clinical entities from text.

        Main entry point for entity extraction. Processes the input text
        through tokenization, encoding, inference, and BIO decoding.

        Args:
            text: Clinical note text to process.

        Returns:
            List of entity dictionaries with keys:

            - ``type``: Entity type (e.g., "FIO2", "PEEP", "DX")
            - ``text``: Extracted text span
            - ``start``: Character offset start position
            - ``end``: Character offset end position
            - ``score``: Confidence score (0.0 to 1.0)

        Example:
            >>> extractor = LSTMExtractor()
            >>> entities = extractor.predict("FiO2 60%, PEEP 8")
            >>> entities[0]
            {'type': 'FIO2', 'text': '60%', 'start': 5, 'end': 8, 'score': 0.91}
        """
        tokens, spans = self._tokenize(text)
        X = self._encode(tokens)

        probs = self._predict_proba(X)
        real_T = min(len(tokens), self.max_len)
        probs = probs[:real_T, :]

        pred_ids = probs.argmax(axis=-1).astype(int).tolist()
        pred_tags = [self.idx2tag.get(i, "O") for i in pred_ids]

        entities = self._bio_decode(pred_tags, tokens[:real_T], spans[:real_T], probs)

        return [ent.model_dump() for ent in entities]

    def meta(self) -> Dict:
        """
        Return extractor metadata for logging and debugging.

        Returns:
            Dictionary containing:

            - ``extractor``: Model type identifier ("lstm")
            - ``model_dir``: Path to model files
            - ``vocab_size``: Number of words in vocabulary
            - ``num_tags``: Number of BIO tags
            - ``max_len``: Maximum sequence length
        """
        return {
            "extractor": "lstm",
            "model_dir": str(self.model_dir),
            "vocab_size": len(self.word2idx),
            "num_tags": len(self.tag2idx),
            "max_len": self.max_len,
        }
