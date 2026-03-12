"""
Transformer Named Entity Recognition Model for Clinical Text.

This module implements Transformer-based NER for Spanish clinical notes using a
fine-tuned RoBERTa token-classification model (Hugging Face Transformers) with
BIO tagging.

Windowing Strategy:
    Long notes are processed with sliding windows: the tokenizer splits the
    input into overlapping windows (``max_len`` with ``stride`` overlap),
    predictions are merged back at word level, and BIO tags are decoded into
    entity spans with character offsets.

Model Details:
    - Base: PlanTL-GOB-ES/roberta-base-bne
    - Fine-tuned: NicolasUnivalle/roberta-vm-ner

Configuration:
    - ``TRANSFORMER_ROBERTA_MODEL_ID`` (default: NicolasUnivalle/roberta-vm-ner)
    - ``TRANSFORMER_MAX_LEN`` (default: 512)
    - ``TRANSFORMER_STRIDE`` (default: 128)
    - ``TRANSFORMER_WINDOW_BATCH_SIZE`` (default: 8)

Usage:
    >>> from app.extractor import TransformerExtractor
    >>> extractor = TransformerExtractor()
    >>> extractor.predict("Paciente con FiO2 60%, PEEP 8 cmH2O")
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

import torch
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

# Logger configuration
logger = logging.getLogger(__name__)


# ==============================================================================
# Base Transformer Extractor
# ==============================================================================


class BaseTransformerExtractor:
    """
    Base class for Transformer-based NER extraction.

    Implements the core extraction pipeline aligned with the training notebook:

    1. Tokenize with offset mapping (sliding windows with stride for long texts)
    2. Predict logits per subtoken
    3. Reconstruct word-level predictions (first subtoken + overlap merge)
    4. Normalize BIO sequence and decode entity spans

    This class is not intended for direct use. Use the variant-specific
    subclasses or the :class:`TransformerExtractor` facade instead.

    Attributes:
        model_id: Hugging Face model identifier or local path.
        MAX_LEN: Token window length for tokenization (subword tokens).
        STRIDE: Sliding-window overlap (subword tokens).
        WINDOW_BATCH_SIZE: Number of windows per forward pass.
        device: PyTorch device (cuda or cpu).
        tokenizer: Hugging Face tokenizer instance.
        model: Loaded classification model.
        id2label: Mapping from tag IDs to BIO tag strings.

    Note:
        Models are loaded in evaluation mode with gradient computation disabled
        for inference efficiency.
    """

    def __init__(
        self,
        model_id: str,
        *,
        max_len: int = 512,
        stride: int = 128,
        window_batch_size: int = 8,
        device: Optional[str] = None,
    ) -> None:
        """
        Initialize the base Transformer extractor.

        Args:
            model_id: Hugging Face model identifier (e.g., "NicolasUnivalle/roberta-vm-ner")
                or path to local model directory.
            max_len: Token window length (subword tokens). Long notes are
                processed with sliding windows instead of being truncated to
                this length.
            stride: Sliding-window overlap (subword tokens) for long notes.
            window_batch_size: Number of windows processed per forward pass.
                Lower values reduce peak memory usage for very long notes.
            device: PyTorch device string ("cuda", "cpu"). If None, automatically
                selects CUDA if available.

        Raises:
            ValueError: If parameters are invalid or the model does not expose
                ``config.id2label`` mapping.

        Note:
            Model download from Hugging Face Hub occurs on first instantiation
            if not cached locally.
        """
        self.model_id: str = model_id
        self.MAX_LEN = int(max_len)
        self.STRIDE = int(stride)
        self.WINDOW_BATCH_SIZE = int(window_batch_size)

        if self.MAX_LEN <= 0:
            raise ValueError("max_len must be > 0")
        if self.STRIDE < 0:
            raise ValueError("stride must be >= 0")
        if self.STRIDE >= self.MAX_LEN:
            raise ValueError("stride must be < max_len")
        if self.WINDOW_BATCH_SIZE <= 0:
            raise ValueError("window_batch_size must be > 0")

        # Device selection with CUDA fallback
        self.device = (
            torch.device(device)
            if device
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )

        # Initialize tokenizer and model
        self.tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(
            model_id, use_fast=True
        )
        self.model: PreTrainedModel = (
            AutoModelForTokenClassification.from_pretrained(model_id)
            .to(self.device)
            .eval()
        )

        # Validate label mapping
        if not getattr(self.model.config, "id2label", None):
            raise ValueError(
                "Model does not expose config.id2label. "
                "Ensure the model was published with id2label/label2id mappings."
            )

        self.id2label: Dict[int, str] = {
            int(k): str(v) for k, v in self.model.config.id2label.items()
        }

    @staticmethod
    def _normalize_bio_sequence(tags: List[str]) -> List[str]:
        """
        Normalize invalid BIO transitions.

        Rules (notebook-aligned):
        - I-X at sequence start -> B-X
        - I-X after O -> B-X
        - I-X after B-Y/I-Y with Y != X -> B-X
        """
        normalized: List[str] = []
        prev_tag = "O"

        for raw_tag in tags:
            tag = str(raw_tag)
            if tag.startswith("I-"):
                ent_type = tag[2:]
                prev_is_entity = prev_tag.startswith(("B-", "I-"))
                prev_type = prev_tag[2:] if prev_is_entity else None
                if (not prev_is_entity) or (prev_type != ent_type):
                    tag = f"B-{ent_type}"

            normalized.append(tag)
            prev_tag = tag

        return normalized

    def predict(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract clinical entities from text.

        Implements the full extraction pipeline:

        1. **Tokenization**: Subword tokenization with offset mapping
        2. **Inference**: Forward pass through classification model
        3. **Word Reconstruction**: Aggregate subtoken predictions per word
        4. **BIO Decoding**: Convert tag sequence to entity spans

        Args:
            text: Clinical note text to process.

        Returns:
            List of entity dictionaries with keys:

            - ``type``: Entity type (e.g., "FIO2", "PEEP", "DX")
            - ``text``: Extracted text span from source
            - ``start``: Character offset start position
            - ``end``: Character offset end position
            - ``code``: Normalized value (None initially, populated by pipeline)

        Example:
            >>> extractor = BaseTransformerExtractor("NicolasUnivalle/roberta-vm-ner")
            >>> entities = extractor.predict("FiO2 60%, PEEP 8 cmH2O")
            >>> entities[0]
            {'type': 'FIO2', 'text': '60%', 'start': 5, 'end': 8, 'code': None}
        """
        if not text:
            return []

        # Tokenize with offset mapping using sliding windows for long texts.
        enc = self.tokenizer(
            text,
            return_offsets_mapping=True,
            truncation=True,
            max_length=self.MAX_LEN,
            stride=self.STRIDE,
            return_overflowing_tokens=True,
            padding="max_length",
            return_tensors="pt",
        )

        input_ids = enc["input_ids"].to(self.device)
        attention_mask = enc["attention_mask"].to(self.device)

        offsets_batch_raw = enc["offset_mapping"]
        offsets_batch = (
            offsets_batch_raw.tolist()
            if hasattr(offsets_batch_raw, "tolist")
            else offsets_batch_raw
        )

        # Merge overlapping windows by averaging logits per word.
        # Voting uses first subtoken per window (notebook-aligned), while
        # character spans are expanded with all subtokens of each word.
        word_logit_sum: Dict[int, torch.Tensor] = {}
        word_logit_count: Dict[int, int] = {}
        word_char_span: Dict[int, List[int]] = {}

        with torch.no_grad():
            num_windows = int(input_ids.shape[0])
            for start in range(0, num_windows, self.WINDOW_BATCH_SIZE):
                end = min(start + self.WINDOW_BATCH_SIZE, num_windows)
                logits_batch = (
                    self.model(
                        input_ids=input_ids[start:end],
                        attention_mask=attention_mask[start:end],
                    )
                    .logits.detach()
                    .cpu()
                )

                for local_idx in range(logits_batch.shape[0]):
                    win_idx = start + local_idx
                    chunk_logits = logits_batch[local_idx]
                    offsets = offsets_batch[win_idx]
                    word_ids = enc.word_ids(batch_index=win_idx)
                    seen_word_ids = set()

                    for tok_i, word_id in enumerate(word_ids):
                        if word_id is None:
                            continue

                        s, e = offsets[tok_i]
                        if s == e:
                            continue

                        word_idx = int(word_id)
                        prev_span = word_char_span.get(word_idx)
                        if prev_span is None:
                            word_char_span[word_idx] = [int(s), int(e)]
                        else:
                            prev_span[0] = min(prev_span[0], int(s))
                            prev_span[1] = max(prev_span[1], int(e))

                        # First subtoken vote per window
                        if word_idx in seen_word_ids:
                            continue
                        seen_word_ids.add(word_idx)

                        logits_row = chunk_logits[tok_i]
                        if word_idx in word_logit_sum:
                            word_logit_sum[word_idx] += logits_row
                            word_logit_count[word_idx] += 1
                        else:
                            word_logit_sum[word_idx] = logits_row.clone()
                            word_logit_count[word_idx] = 1

        merged_words: List[Dict[str, Any]] = []
        for word_idx, logits_sum in word_logit_sum.items():
            votes = word_logit_count[word_idx]
            avg_logits = logits_sum / float(votes)
            pred_id = int(avg_logits.argmax().item())
            span = word_char_span.get(word_idx)
            if not span:
                continue
            s, e = int(span[0]), int(span[1])
            if s >= e:
                continue
            merged_words.append(
                {"start": s, "end": e, "tag": self.id2label.get(pred_id, "O")}
            )
        merged_words.sort(key=lambda w: (w["start"], w["end"]))

        # Normalize invalid BIO transitions before span decoding.
        normalized_tags = self._normalize_bio_sequence(
            [str(w["tag"]) for w in merged_words]
        )
        for w, normalized_tag in zip(merged_words, normalized_tags):
            w["tag"] = normalized_tag

        # BIO decoding (notebook-aligned)
        spans = []
        cur_start, cur_end, cur_label = None, None, None

        for w in merged_words:
            tag = w["tag"]

            if tag.startswith("B-"):
                # Start new entity
                if cur_label:
                    spans.append((cur_start, cur_end, cur_label))
                cur_label = tag[2:]
                cur_start, cur_end = w["start"], w["end"]

            elif tag.startswith("I-") and cur_label == tag[2:]:
                # Continue current entity
                cur_end = w["end"]

            else:
                # End current entity
                if cur_label:
                    spans.append((cur_start, cur_end, cur_label))
                cur_label = None

        if cur_label:
            spans.append((cur_start, cur_end, cur_label))

        # Build output format
        out: List[Dict[str, Any]] = []
        for s, e, t in spans:
            if 0 <= s < e <= len(text):
                out.append(
                    {
                        "type": t,
                        "text": text[s:e],
                        "start": s,
                        "end": e,
                        "code": None,
                    }
                )
        return out

    def meta(self) -> Dict[str, Any]:
        """
        Return extractor metadata for logging and debugging.

        Returns:
            Dictionary containing:

            - ``model_id``: Hugging Face model identifier
            - ``num_labels``: Number of BIO tags
            - ``max_len``: Token window length (subword tokens)
            - ``stride``: Sliding-window overlap (subword tokens)
            - ``window_batch_size``: Windows per forward pass
            - ``device``: Inference device (cuda/cpu)
            - ``labels``: List of unique BIO tags
            - ``decoder``: Decoding strategy description
        """
        return {
            "model_id": self.model_id,
            "num_labels": len(self.id2label),
            "max_len": self.MAX_LEN,
            "stride": self.STRIDE,
            "window_batch_size": self.WINDOW_BATCH_SIZE,
            "device": str(self.device),
            "labels": sorted(set(self.id2label.values())),
            "decoder": (
                "BIO-first-subtoken + sliding-window overlap merge (mean logits) "
                "+ BIO normalization (notebook-aligned)"
            ),
        }


# ==============================================================================
# RoBERTa Transformer Extractor
# ==============================================================================


class RobertaTransformerExtractor(BaseTransformerExtractor):
    """
    RoBERTa-based clinical NER extractor.

    Uses a Spanish RoBERTa model fine-tuned on mechanical ventilation
    clinical notes for entity extraction.

    Model Details:
        - Base: PlanTL-GOB-ES/roberta-base-bne
        - Fine-tuning: NER on mechanical ventilation notes
        - Hugging Face: NicolasUnivalle/roberta-vm-ner

    Attributes:
        Inherits all attributes from :class:`BaseTransformerExtractor`.

    Example:
        >>> extractor = RobertaTransformerExtractor()
        >>> entities = extractor.predict("Temperatura 38.5°C")
        >>> print(entities[0]["type"])
        'TEMP'

    See Also:
        - :class:`TransformerExtractor` for the service-facing extractor API
    """

    def __init__(
        self,
        model_id: Optional[str] = None,
        *,
        max_len: int = 512,
        stride: int = 128,
        window_batch_size: int = 8,
        device: Optional[str] = None,
    ) -> None:
        """
        Initialize the RoBERTa extractor.

        Args:
            model_id: Hugging Face model ID. Defaults to environment variable
                ``TRANSFORMER_ROBERTA_MODEL_ID`` or "NicolasUnivalle/roberta-vm-ner".
            max_len: Token window length (subword tokens).
            stride: Sliding-window overlap (subword tokens) for long notes.
            window_batch_size: Number of windows processed per forward pass.
            device: PyTorch device string.
        """
        model_id = model_id or os.getenv(
            "TRANSFORMER_ROBERTA_MODEL_ID",
            "NicolasUnivalle/roberta-vm-ner",
        )

        super().__init__(
            model_id=model_id,
            max_len=max_len,
            stride=stride,
            window_batch_size=window_batch_size,
            device=device,
        )

        logger.info("RoBERTa Transformer Extractor initialized: %s", model_id)

    def meta(self) -> Dict[str, Any]:
        """
        Return RoBERTa-specific metadata.

        Returns:
            Base metadata extended with variant and architecture information.
        """
        base_meta = super().meta()
        base_meta["variant"] = "roberta"
        base_meta["extractor"] = "roberta_transformer"
        base_meta["architecture"] = "roberta"
        return base_meta


# ==============================================================================
# Transformer Extractor Facade
# ==============================================================================


class TransformerExtractor:
    """
    Service-facing Transformer extractor (RoBERTa).

    This wrapper provides a stable ``predict()`` API and delegates to the
    underlying RoBERTa extractor configured with sliding-window tokenization.

    Attributes:
        model_id: Resolved Hugging Face model identifier.
        max_len: Token window length (subword tokens).
        stride: Sliding-window overlap (subword tokens).
        window_batch_size: Number of windows per forward pass.
        device: PyTorch device string ("cuda", "cpu", or None for auto-detect).
        extractor: Underlying RobertaTransformerExtractor instance.

    Example:
        >>> extractor = TransformerExtractor()
        >>> extractor.variant
        'roberta'

    See Also:
        - :class:`RobertaTransformerExtractor` for RoBERTa-specific usage
    """

    def __init__(
        self,
        model_id: Optional[str] = None,
        *,
        max_len: int = 512,
        stride: int = 128,
        window_batch_size: int = 8,
        device: Optional[str] = None,
    ) -> None:
        """
        Initialize the Transformer extractor (RoBERTa).

        Args:
            model_id: Hugging Face model ID. If None, uses environment variable
                ``TRANSFORMER_ROBERTA_MODEL_ID`` or defaults to the bundled model.
            max_len: Token window length (subword tokens).
            stride: Sliding-window overlap (subword tokens) for long notes.
            window_batch_size: Number of windows processed per forward pass.
            device: PyTorch device string ("cuda" or "cpu"). If None, auto-detects.

        Note:
            Model weights are downloaded on first use if not cached.
        """
        self.model_id = model_id or os.getenv(
            "TRANSFORMER_ROBERTA_MODEL_ID", "NicolasUnivalle/roberta-vm-ner"
        )

        self.max_len = int(max_len)
        self.stride = int(stride)
        self.window_batch_size = int(window_batch_size)
        self.device = device

        self.extractor = RobertaTransformerExtractor(
            model_id=self.model_id,
            max_len=self.max_len,
            stride=self.stride,
            window_batch_size=self.window_batch_size,
            device=self.device,
        )

        logger.info("TransformerExtractor initialized: %s", self.model_id)

    def predict(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract clinical entities from text.

        Delegates to the underlying RoBERTa extractor.

        Args:
            text: Clinical note text to process.

        Returns:
            List of entity dictionaries in pipeline-compatible format.
        """
        return self.extractor.predict(text)

    def meta(self) -> Dict[str, Any]:
        """
        Return metadata from the underlying extractor.

        Returns:
            Dictionary with extractor metadata.
        """
        return self.extractor.meta()
