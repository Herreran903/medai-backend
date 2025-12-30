# transformer_extractor.py
# Este archivo implementa extractores de entidades nombradas (NER) basados en modelos Transformers.
# Sigue un patrón de diseño similar a LLM con clases específicas por variante (BETO, RoBERTa)
# y una clase facade principal (TransformerExtractor).
#
# Nota: Esta versión está alineada con el comportamiento del notebook:
# - Decodificación BIO (B-/I-)
# - "First subtoken wins" por palabra
# - Sin sliding windows (sin stride / overflow)

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

# Configuración del logger para registrar eventos y errores
logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)


# ============================================================================
# Base Transformer Extractor (BIO-compatible, notebook-aligned)
# ============================================================================


class BaseTransformerExtractor:
    """
    Clase base para extracción NER usando Transformers (FULL).
    Implementa un decoder alineado con el notebook:

      1) Tokeniza con offsets (sin ventanas deslizantes).
      2) Predice etiquetas por token (argmax).
      3) Reconstruye por palabra con 'first subtoken wins'.
      4) Decodifica spans respetando BIO (B-/I-).
    """

    def __init__(
        self,
        model_id: str,
        *,
        max_len: int = 512,
        device: Optional[str] = None,
    ) -> None:
        # ID del modelo (FULL) en Hugging Face o path local
        self.model_id: str = model_id

        # Longitud máxima de secuencia
        self.MAX_LEN = int(max_len)

        # Determina el dispositivo de ejecución (GPU o CPU)
        self.device = (
            torch.device(device)
            if device
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )

        # Inicializa tokenizer y modelo (FULL)
        self.tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(
            model_id, use_fast=True
        )
        self.model: PreTrainedModel = (
            AutoModelForTokenClassification.from_pretrained(model_id)
            .to(self.device)
            .eval()
        )

        # Mapeo de etiquetas (debe venir en config del modelo)
        if not getattr(self.model.config, "id2label", None):
            raise ValueError(
                "El modelo no expone config.id2label. "
                "Asegúrate de publicar un modelo FULL con id2label/label2id."
            )

        self.id2label: Dict[int, str] = {
            int(k): str(v) for k, v in self.model.config.id2label.items()
        }

    # ------------------------------------------------------------------
    # CORE: Notebook-aligned prediction
    # ------------------------------------------------------------------
    def predict(self, text: str) -> List[Dict[str, Any]]:
        """
        Predice entidades tipo-span en un texto dado.
        Devuelve una lista de diccionarios con {type, text, start, end, score, code}.
        """
        if not text:
            return []

        # Tokeniza con offsets (sin ventanas)
        enc = self.tokenizer(
            text,
            return_offsets_mapping=True,
            truncation=True,
            max_length=self.MAX_LEN,
            return_tensors="pt",
        )

        input_ids = enc["input_ids"].to(self.device)
        attention_mask = enc["attention_mask"].to(self.device)

        # offsets como lista en CPU
        offsets = enc["offset_mapping"][0].tolist()

        # word_ids requiere tokenizer fast
        word_ids = enc.word_ids(batch_index=0)

        # Inferencia
        with torch.no_grad():
            logits = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            ).logits

        preds = logits.argmax(dim=-1)[0].tolist()
        tags = [self.id2label.get(int(p), "O") for p in preds]

        # -------- Reconstrucción por palabra (FIRST SUBTOKEN WINS) --------
        words = []
        current = None

        for i, w_id in enumerate(word_ids):
            if w_id is None:
                continue  # CLS/SEP/PAD

            s, e = offsets[i]
            if s == e:
                continue  # tokens vacíos/especiales

            tag = tags[i]

            # Nuevo word_id => cierra palabra previa
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
                # Mismo word_id => extiende rango de caracteres
                current["end"] = max(current["end"], e)

        if current:
            words.append(current)

        # -------- Decodificación BIO (idéntica al notebook) --------
        spans = []
        cur_start, cur_end, cur_label = None, None, None

        for w in words:
            tag = w["tag"]

            if tag.startswith("B-"):
                if cur_label:
                    spans.append((cur_start, cur_end, cur_label))
                cur_label = tag[2:]
                cur_start, cur_end = w["start"], w["end"]

            elif tag.startswith("I-") and cur_label == tag[2:]:
                cur_end = w["end"]

            else:
                if cur_label:
                    spans.append((cur_start, cur_end, cur_label))
                cur_label = None

        if cur_label:
            spans.append((cur_start, cur_end, cur_label))

        # -------- Formato de salida --------
        out: List[Dict[str, Any]] = []
        for s, e, t in spans:
            if 0 <= s < e <= len(text):
                out.append(
                    {
                        "type": t,
                        "text": text[s:e],
                        "start": s,
                        "end": e,
                        "score": None,  # mantenemos None para compatibilidad con tu API
                        "code": None,
                    }
                )
        return out

    def meta(self) -> Dict[str, Any]:
        """
        Devuelve metadatos útiles para debugging o telemetría.
        """
        return {
            "model_id": self.model_id,
            "num_labels": len(self.id2label),
            "max_len": self.MAX_LEN,
            "device": str(self.device),
            "labels": sorted(set(self.id2label.values())),
            "decoder": "BIO-first-subtoken (notebook-aligned)",
        }


# ============================================================================
# BETO Transformer Extractor
# ============================================================================


class BETOTransformerExtractor(BaseTransformerExtractor):
    """
    Extractor específico para modelos BETO (BERT español).
    Carga un modelo FULL (no PEFT en esta versión).
    """

    def __init__(
        self,
        model_id: Optional[str] = None,
        *,
        max_len: int = 512,
        device: Optional[str] = None,
    ) -> None:
        # Por defecto usa el FULL publicado (o env var)
        model_id = model_id or os.getenv(
            "TRANSFORMER_BETO_MODEL_ID",
            "NicolasUnivalle/beto-vm-ner-full",
        )

        super().__init__(
            model_id=model_id,
            max_len=max_len,
            device=device,
        )

        logger.info("BETO Transformer Extractor inicializado: %s", model_id)

    def meta(self) -> Dict[str, Any]:
        base_meta = super().meta()
        base_meta["variant"] = "beto"
        base_meta["extractor"] = "beto_transformer"
        return base_meta


# ============================================================================
# RoBERTa Transformer Extractor
# ============================================================================


class RobertaTransformerExtractor(BaseTransformerExtractor):
    """
    Extractor específico para modelos RoBERTa español.
    Carga un modelo FULL (no PEFT en esta versión).
    """

    def __init__(
        self,
        model_id: Optional[str] = None,
        *,
        max_len: int = 512,
        device: Optional[str] = None,
    ) -> None:
        # Por defecto usa el FULL publicado (o env var)
        model_id = model_id or os.getenv(
            "TRANSFORMER_ROBERTA_MODEL_ID",
            "NicolasUnivalle/roberta-vm-ner-full",
        )

        super().__init__(
            model_id=model_id,
            max_len=max_len,
            device=device,
        )

        logger.info("RoBERTa Transformer Extractor inicializado: %s", model_id)

    def meta(self) -> Dict[str, Any]:
        base_meta = super().meta()
        base_meta["variant"] = "roberta"
        base_meta["extractor"] = "roberta_transformer"
        base_meta["architecture"] = "roberta"
        return base_meta


# ============================================================================
# Main TransformerExtractor Class (Facade)
# ============================================================================


class TransformerExtractor:
    """
    Clase facade para extractores Transformer.
    Selecciona automáticamente BETO o RoBERTa según el model_id.

    - NO usa base_model_id
    - NO usa stride
    - Decodificación alineada con notebook (BIO, first subtoken)
    """

    def __init__(
        self,
        model_id: Optional[str] = None,
        *,
        max_len: int = 512,
        device: Optional[str] = None,
    ) -> None:
        # Modelo FULL por defecto
        self.model_id = model_id or os.getenv(
            "HF_MODEL_ID", "NicolasUnivalle/beto-vm-ner-full"
        )

        self.max_len = int(max_len)
        self.device = device

        # Detecta variante
        self.variant = self._detect_variant(self.model_id)

        # Inicializa extractor concreto
        if self.variant == "roberta":
            self.extractor = RobertaTransformerExtractor(
                model_id=self.model_id,
                max_len=self.max_len,
                device=self.device,
            )
        else:
            # BETO por defecto
            self.extractor = BETOTransformerExtractor(
                model_id=self.model_id,
                max_len=self.max_len,
                device=self.device,
            )

        logger.info("TransformerExtractor inicializado con variante: %s", self.variant)

    def _detect_variant(self, model_id: str) -> str:
        """
        Detecta la variante del modelo según el model_id.
        """
        mid = model_id.lower()
        if "roberta" in mid:
            return "roberta"
        if "beto" in mid or "bert" in mid:
            return "beto"
        return "beto"

    def predict(self, text: str) -> List[Dict[str, Any]]:
        """
        Extrae entidades del texto usando el extractor seleccionado.
        """
        return self.extractor.predict(text)

    def meta(self) -> Dict[str, Any]:
        """
        Retorna metadatos del extractor actual.
        """
        meta = self.extractor.meta()
        meta["facade_variant"] = self.variant
        return meta
