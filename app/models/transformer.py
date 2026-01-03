"""
Transformer-Based Named Entity Recognition for Clinical Text.

This module implements clinical NER using fine-tuned Spanish Transformer models
(BETO and RoBERTa) with BIO tagging for mechanical ventilation notes.

Architecture Context:
    The Transformer extractor is the primary NER model in MedAI, offering
    the best balance of accuracy and performance for clinical entity extraction.

    Available models:

    - **BETO**: Spanish BERT model fine-tuned on mechanical ventilation notes
    - **RoBERTa**: Spanish RoBERTa model with similar fine-tuning

    Both models are hosted on Hugging Face and loaded via the Transformers library.

Model Architecture:
    The extractors use Hugging Face ``AutoModelForTokenClassification`` with:

    - Pre-trained Spanish language model (BETO or RoBERTa)
    - Token classification head for BIO tag prediction
    - Subword tokenization with offset mapping for span reconstruction

Decoding Strategy:
    The implementation follows a "first subtoken wins" strategy aligned with
    the training notebook:

    1. Tokenize with subword offsets (no sliding windows)
    2. Predict BIO tags per subtoken
    3. Reconstruct word-level tags using first subtoken
    4. Decode BIO sequences to entity spans

Design Pattern:
    The module follows the Strategy pattern with a Facade:

    - :class:`BaseTransformerExtractor`: Core extraction logic
    - :class:`BETOTransformerExtractor`: BETO-specific configuration
    - :class:`RobertaTransformerExtractor`: RoBERTa-specific configuration
    - :class:`TransformerExtractor`: Facade for automatic variant selection

Model Configuration:
    Models are configured via environment variables or :class:`app.config.Settings`:

    - ``TRANSFORMER_BETO_MODEL_ID``: Hugging Face ID for BETO model
    - ``TRANSFORMER_ROBERTA_MODEL_ID``: Hugging Face ID for RoBERTa model

Usage:
    >>> from app.models.transformer import TransformerExtractor
    >>> # Auto-detect variant from model ID
    >>> extractor = TransformerExtractor()
    >>> entities = extractor.predict("Paciente con FiO2 60%, PEEP 8 cmH2O")
    >>>
    >>> # Explicit variant selection
    >>> extractor = TransformerExtractor(model_id="NicolasUnivalle/roberta-vm-ner-full")

See Also:
    - :mod:`app.services.pipeline` for extraction orchestration
    - :mod:`app.services.registry` for model registration
    - :class:`app.schemas.Entity` for output schema
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
logger.setLevel(logging.WARNING)


# ==============================================================================
# Base Transformer Extractor
# ==============================================================================


class BaseTransformerExtractor:
    """
    Base class for Transformer-based NER extraction.

    Implements the core extraction pipeline aligned with the training notebook:

    1. Tokenize with offset mapping (no sliding windows)
    2. Predict BIO tags per subtoken
    3. Reconstruct word-level predictions ("first subtoken wins")
    4. Decode BIO sequences to entity spans

    This class is not intended for direct use. Use the variant-specific
    subclasses or the :class:`TransformerExtractor` facade instead.

    Attributes:
        model_id: Hugging Face model identifier or local path.
        MAX_LEN: Maximum sequence length for tokenization.
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
        device: Optional[str] = None,
    ) -> None:
        """
        Initialize the base Transformer extractor.

        Args:
            model_id: Hugging Face model identifier (e.g., "NicolasUnivalle/beto-vm-ner-full")
                or path to local model directory.
            max_len: Maximum sequence length for tokenization. Longer sequences
                are truncated.
            device: PyTorch device string ("cuda", "cpu"). If None, automatically
                selects CUDA if available.

        Raises:
            ValueError: If the model does not expose ``config.id2label`` mapping.

        Note:
            Model download from Hugging Face Hub occurs on first instantiation
            if not cached locally.
        """
        self.model_id: str = model_id
        self.MAX_LEN = int(max_len)

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
            - ``score``: Confidence score (None for transformer models)
            - ``code``: Normalized code (None, populated by normalizer)

        Example:
            >>> extractor = BaseTransformerExtractor("NicolasUnivalle/beto-vm-ner-full")
            >>> entities = extractor.predict("FiO2 60%, PEEP 8 cmH2O")
            >>> entities[0]
            {'type': 'FIO2', 'text': '60%', 'start': 5, 'end': 8, 'score': None, 'code': None}
        """
        if not text:
            return []

        # Tokenize with offset mapping (no sliding windows)
        enc = self.tokenizer(
            text,
            return_offsets_mapping=True,
            truncation=True,
            max_length=self.MAX_LEN,
            return_tensors="pt",
        )

        input_ids = enc["input_ids"].to(self.device)
        attention_mask = enc["attention_mask"].to(self.device)

        # Offset mapping for span reconstruction
        offsets = enc["offset_mapping"][0].tolist()

        # Word IDs for subtoken-to-word mapping
        word_ids = enc.word_ids(batch_index=0)

        # Model inference
        with torch.no_grad():
            logits = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            ).logits

        preds = logits.argmax(dim=-1)[0].tolist()
        tags = [self.id2label.get(int(p), "O") for p in preds]

        # Word-level reconstruction (first subtoken wins)
        words = []
        current = None

        for i, w_id in enumerate(word_ids):
            if w_id is None:
                continue  # Skip CLS/SEP/PAD tokens

            s, e = offsets[i]
            if s == e:
                continue  # Skip empty/special tokens

            tag = tags[i]

            # New word_id starts a new word
            if current is None or w_id != current["word_id"]:
                if current:
                    words.append(current)
                current = {
                    "word_id": w_id,
                    "start": s,
                    "end": e,
                    "tag": tag,
                }
            else:
                # Same word_id: extend character range
                current["end"] = max(current["end"], e)

        if current:
            words.append(current)

        # BIO decoding (notebook-aligned)
        spans = []
        cur_start, cur_end, cur_label = None, None, None

        for w in words:
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
                        "score": None,  # Transformer models don't provide per-entity scores
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
            - ``max_len``: Maximum sequence length
            - ``device``: Inference device (cuda/cpu)
            - ``labels``: List of unique BIO tags
            - ``decoder``: Decoding strategy description
        """
        return {
            "model_id": self.model_id,
            "num_labels": len(self.id2label),
            "max_len": self.MAX_LEN,
            "device": str(self.device),
            "labels": sorted(set(self.id2label.values())),
            "decoder": "BIO-first-subtoken (notebook-aligned)",
        }


# ==============================================================================
# BETO Transformer Extractor
# ==============================================================================


class BETOTransformerExtractor(BaseTransformerExtractor):
    """
    BETO-based clinical NER extractor.

    BETO (Bidirectional Encoder Representations from Transformers for Spanish)
    is a Spanish BERT model. This extractor uses a version fine-tuned on
    mechanical ventilation clinical notes.

    Model Details:
        - Base: dccuchile/bert-base-spanish-wwm-cased
        - Fine-tuning: NER on mechanical ventilation notes
        - Hugging Face: NicolasUnivalle/beto-vm-ner-full

    Attributes:
        Inherits all attributes from :class:`BaseTransformerExtractor`.

    Example:
        >>> extractor = BETOTransformerExtractor()
        >>> entities = extractor.predict("PEEP 10 cmH2O")
        >>> print(entities[0]["type"])
        'PEEP'

    See Also:
        - :class:`RobertaTransformerExtractor` for RoBERTa variant
        - :class:`TransformerExtractor` for automatic variant selection
    """

    def __init__(
        self,
        model_id: Optional[str] = None,
        *,
        max_len: int = 512,
        device: Optional[str] = None,
    ) -> None:
        """
        Initialize the BETO extractor.

        Args:
            model_id: Hugging Face model ID. Defaults to environment variable
                ``TRANSFORMER_BETO_MODEL_ID`` or "NicolasUnivalle/beto-vm-ner-full".
            max_len: Maximum sequence length.
            device: PyTorch device string.
        """
        model_id = model_id or os.getenv(
            "TRANSFORMER_BETO_MODEL_ID",
            "NicolasUnivalle/beto-vm-ner-full",
        )

        super().__init__(
            model_id=model_id,
            max_len=max_len,
            device=device,
        )

        logger.info("BETO Transformer Extractor initialized: %s", model_id)

    def meta(self) -> Dict[str, Any]:
        """
        Return BETO-specific metadata.

        Returns:
            Base metadata extended with variant information.
        """
        base_meta = super().meta()
        base_meta["variant"] = "beto"
        base_meta["extractor"] = "beto_transformer"
        return base_meta


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
        - Hugging Face: NicolasUnivalle/roberta-vm-ner-full

    Attributes:
        Inherits all attributes from :class:`BaseTransformerExtractor`.

    Example:
        >>> extractor = RobertaTransformerExtractor()
        >>> entities = extractor.predict("Temperatura 38.5°C")
        >>> print(entities[0]["type"])
        'TEMP'

    See Also:
        - :class:`BETOTransformerExtractor` for BETO variant
        - :class:`TransformerExtractor` for automatic variant selection
    """

    def __init__(
        self,
        model_id: Optional[str] = None,
        *,
        max_len: int = 512,
        device: Optional[str] = None,
    ) -> None:
        """
        Initialize the RoBERTa extractor.

        Args:
            model_id: Hugging Face model ID. Defaults to environment variable
                ``TRANSFORMER_ROBERTA_MODEL_ID`` or "NicolasUnivalle/roberta-vm-ner-full".
            max_len: Maximum sequence length.
            device: PyTorch device string.
        """
        model_id = model_id or os.getenv(
            "TRANSFORMER_ROBERTA_MODEL_ID",
            "NicolasUnivalle/roberta-vm-ner-full",
        )

        super().__init__(
            model_id=model_id,
            max_len=max_len,
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
    Facade class for Transformer-based clinical NER extraction.

    This class provides a unified interface to BETO and RoBERTa extractors,
    automatically selecting the appropriate variant based on the model ID.

    The facade pattern simplifies client code by:

    - Auto-detecting model variant from model ID
    - Providing consistent interface regardless of variant
    - Handling default model selection from configuration

    Variant Detection:
        The variant is detected from the model ID string:

        - Contains "roberta" → RoBERTa variant
        - Contains "beto" or "bert" → BETO variant
        - Default → BETO variant

    Attributes:
        model_id: Resolved Hugging Face model identifier.
        max_len: Maximum sequence length.
        device: PyTorch device string.
        variant: Detected variant ("beto" or "roberta").
        extractor: Underlying variant-specific extractor instance.

    Example:
        >>> # Auto-detect BETO from default model
        >>> extractor = TransformerExtractor()
        >>> extractor.variant
        'beto'
        >>>
        >>> # Auto-detect RoBERTa from model ID
        >>> extractor = TransformerExtractor(model_id="NicolasUnivalle/roberta-vm-ner-full")
        >>> extractor.variant
        'roberta'

    See Also:
        - :class:`BETOTransformerExtractor` for BETO-specific usage
        - :class:`RobertaTransformerExtractor` for RoBERTa-specific usage
    """

    def __init__(
        self,
        model_id: Optional[str] = None,
        *,
        max_len: int = 512,
        device: Optional[str] = None,
    ) -> None:
        """
        Initialize the Transformer extractor facade.

        Args:
            model_id: Hugging Face model ID. If None, uses environment variable
                ``HF_MODEL_ID`` or defaults to BETO model.
            max_len: Maximum sequence length for tokenization.
            device: PyTorch device string ("cuda", "cpu").

        Note:
            The underlying extractor is instantiated based on variant detection.
            Model weights are downloaded on first use if not cached.
        """
        self.model_id = model_id or os.getenv(
            "HF_MODEL_ID", "NicolasUnivalle/beto-vm-ner-full"
        )

        self.max_len = int(max_len)
        self.device = device

        # Detect variant from model ID
        self.variant = self._detect_variant(self.model_id)

        # Initialize appropriate extractor
        if self.variant == "roberta":
            self.extractor = RobertaTransformerExtractor(
                model_id=self.model_id,
                max_len=self.max_len,
                device=self.device,
            )
        else:
            self.extractor = BETOTransformerExtractor(
                model_id=self.model_id,
                max_len=self.max_len,
                device=self.device,
            )

        logger.info("TransformerExtractor initialized with variant: %s", self.variant)

    def _detect_variant(self, model_id: str) -> str:
        """
        Detect model variant from model ID string.

        Args:
            model_id: Hugging Face model identifier.

        Returns:
            Variant string: "roberta" or "beto".
        """
        mid = model_id.lower()
        if "roberta" in mid:
            return "roberta"
        if "beto" in mid or "bert" in mid:
            return "beto"
        return "beto"

    def predict(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract clinical entities from text.

        Delegates to the underlying variant-specific extractor.

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
            Dictionary with extractor metadata including facade variant.
        """
        meta = self.extractor.meta()
        meta["facade_variant"] = self.variant
        return meta
