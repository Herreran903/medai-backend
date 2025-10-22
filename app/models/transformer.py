# transformer_extractor.py
# Este archivo implementa una clase para extracción de entidades nombradas (NER)
# basada en modelos Transformers con soporte para PEFT/LoRA. Incluye tokenización
# con ventanas deslizantes, predicción de etiquetas por token, agregación por palabra
# y colapso de spans solapados.

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import torch
from peft import PeftConfig, PeftModel
from transformers import (
    AutoConfig,
    AutoModelForTokenClassification,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

# Configuración del logger para registrar eventos y errores
logger = logging.getLogger(__name__)
if not logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("[%(levelname)s] %(name)s: %(message)s"))
    logger.addHandler(_h)
logger.setLevel(logging.WARNING)


class TransformerExtractor:
    """
    Clase para extracción de entidades nombradas (NER) usando Transformers.
    Soporta modelos con PEFT/LoRA y realiza predicciones basadas en token classification.

    Flujo principal:
      1) Tokeniza el texto con ventanas deslizantes (truncation + stride).
      2) Predice etiquetas por token.
      3) Agrega predicciones por palabra usando voto mayoritario.
      4) Colapsa palabras consecutivas con la misma etiqueta en spans.
      5) Une spans solapados del mismo tipo.
    """

    def __init__(
        self,
        model_id: Optional[str] = None,
        base_model_id: Optional[str] = None,
        *,
        max_len: int = 512,
        stride: int = 64,
        device: Optional[str] = None,
    ) -> None:
        # Inicializa los identificadores de modelo (adapter y base)
        self.model_id: str = model_id or os.getenv(
            "HF_MODEL_ID", "NicolasUnivalle/beto_vm_peft"
        )
        self.base_model_id: str = base_model_id or os.getenv(
            "HF_BASE_MODEL_ID", "dccuchile/bert-base-spanish-wwm-cased"
        )
        self.hf_token: Optional[str] = os.getenv("HF_TOKEN")

        # Configuración de tokenización con ventanas deslizantes
        self.MAX_LEN = int(max_len)
        self.STRIDE = int(stride)

        # Determina el dispositivo de ejecución (GPU o CPU)
        self.device = (
            torch.device(device)
            if device
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )

        # Inicializa el tokenizer y el modelo
        self.tokenizer: PreTrainedTokenizerBase = self._init_tokenizer()
        self.model: PreTrainedModel = self._init_model()
        self.model.to(self.device).eval()  # Configura el modelo en modo evaluación

        # Verifica y normaliza los mapas de etiquetas (id2label y label2id)
        if not getattr(self.model.config, "id2label", None):
            raise ValueError(
                "El modelo no expone config.id2label. "
                "Incluye el mapeo en el adapter o pasa un modelo completo con id2label/label2id."
            )
        self.id2label: Dict[int, str] = {
            int(k): str(v) for k, v in self.model.config.id2label.items()
        }

    # ---------------------------------------------------------------------
    # Carga
    # ---------------------------------------------------------------------
    def _init_tokenizer(self) -> PreTrainedTokenizerBase:
        """
        Inicializa el tokenizer. Intenta cargarlo desde el adapter y, si falla,
        lo carga desde el modelo base.
        """
        try:
            # Intenta cargar el tokenizer desde el adapter
            tok = AutoTokenizer.from_pretrained(self.model_id, token=self.hf_token)
            logger.info("Tokenizer cargado desde adapter: %s", self.model_id)
            return tok
        except Exception as e:
            logger.info("Fallo tokenizer del adapter (%s). Probando base...", e)

        # Carga el tokenizer desde el modelo base
        tok = AutoTokenizer.from_pretrained(self.base_model_id, token=self.hf_token)
        logger.info("Tokenizer cargado desde base: %s", self.base_model_id)
        return tok

    def _init_model(self) -> PreTrainedModel:
        """
        Inicializa el modelo. Intenta cargar un modelo completo (con pesos fusionados).
        Si falla, construye un modelo base con configuración PEFT.
        """
        logger.info("Intentando cargar modelo completo: %s", self.model_id)
        try:
            # Intenta cargar un modelo completo
            full = AutoModelForTokenClassification.from_pretrained(
                self.model_id, token=self.hf_token
            )
            if getattr(full.config, "id2label", None):
                logger.info("Modelo completo cargado.")
                return full
            logger.info("Modelo completo sin id2label; se continúa con PEFT.")
        except Exception as e:
            logger.info("No se pudo cargar modelo completo (%s). Intentando PEFT...", e)

        # Configuración para cargar un modelo PEFT
        try:
            peft_cfg = PeftConfig.from_pretrained(self.model_id, token=self.hf_token)
            base_name = peft_cfg.base_model_name_or_path or self.base_model_id
        except Exception:
            peft_cfg = None
            base_name = self.base_model_id

        base_cfg = AutoConfig.from_pretrained(base_name, token=self.hf_token)

        # Intenta arrastrar metadatos del adapter (num_labels, id2label, label2id)
        try:
            adapter_cfg = AutoConfig.from_pretrained(self.model_id, token=self.hf_token)
        except Exception:
            adapter_cfg = None

        def _as_int(d: Dict[Any, Any]) -> Dict[int, str]:
            # Convierte claves a enteros
            return {int(k): str(v) for k, v in d.items()}

        def _as_str(d: Dict[Any, Any]) -> Dict[str, int]:
            # Convierte claves a cadenas
            return {str(k): int(v) for k, v in d.items()}

        if adapter_cfg is not None:
            # Propaga configuraciones del adapter al modelo base
            if getattr(adapter_cfg, "num_labels", None):
                base_cfg.num_labels = adapter_cfg.num_labels
            if getattr(adapter_cfg, "id2label", None):
                base_cfg.id2label = _as_int(adapter_cfg.id2label)
            if getattr(adapter_cfg, "label2id", None):
                base_cfg.label2id = _as_str(adapter_cfg.label2id)

        if getattr(base_cfg, "num_labels", None) in (None, 2):
            raise ValueError(
                "No pude determinar num_labels/id2label/label2id.\n"
                "- Si usas PEFT, asegura que el repo del adapter tenga esos campos en config.json, o\n"
                "- establece variables de entorno y reconstruye el config base antes de cargar."
            )

        # Carga el modelo base y el adapter PEFT
        base = AutoModelForTokenClassification.from_pretrained(
            base_name,
            config=base_cfg,
            token=self.hf_token,
            ignore_mismatched_sizes=True,
        )
        model = PeftModel.from_pretrained(base, self.model_id, token=self.hf_token)

        # Propaga mapas de etiquetas al modelo final
        model.config.id2label = base_cfg.id2label
        model.config.label2id = base_cfg.label2id
        logger.info(
            "Modelo PEFT cargado: base=%s, adapter=%s", base_name, self.model_id
        )
        return model

    # ---------------------------------------------------------------------
    # Agregación y spans
    # ---------------------------------------------------------------------
    def _vote_label_per_word(self, token_tags: List[str]) -> str:
        """
        Determina la etiqueta de una palabra usando voto mayoritario entre los tokens.
        Colapsa etiquetas B-/I- al tipo base. Devuelve 'O' si no hay consenso.
        """
        counts: Dict[str, int] = {}
        for t in token_tags:
            if not t or t == "O":
                continue
            grp = t[2:] if (t.startswith("B-") or t.startswith("I-")) else t
            counts[grp] = counts.get(grp, 0) + 1
        return max(counts, key=counts.get) if counts else "O"

    def _aggregate_tokens_to_words(
        self,
        word_ids: List[Optional[int]],
        offsets: List[Tuple[int, int]],
        pred_ids: List[int],
    ) -> List[Tuple[int, int, int, str]]:
        """
        Agrupa tokens por palabra y asigna una etiqueta por palabra usando voto mayoritario.
        Retorna una lista de tuplas con (índice_palabra, inicio, fin, etiqueta).
        """
        by_word: Dict[int, List[Tuple[str, Tuple[int, int]]]] = {}
        for tok_idx, w_id in enumerate(word_ids):
            if w_id is None:
                continue
            s, e = offsets[tok_idx]
            if s == e:
                continue  # Ignora tokens especiales o vacíos
            tag = self.id2label.get(int(pred_ids[tok_idx]), "O")
            by_word.setdefault(w_id, []).append((tag, (s, e)))

        rows: List[Tuple[int, int, int, str]] = []
        for w_id, items in by_word.items():
            # Calcula el rango de caracteres de la palabra
            s_word = min(se[1][0] for se in items)
            e_word = max(se[1][1] for se in items)
            label_group = self._vote_label_per_word([t for t, _ in items])
            rows.append((w_id, s_word, e_word, label_group))

        rows.sort(key=lambda r: r[0])  # Ordena por índice de palabra
        return rows

    @staticmethod
    def _collapse_words_to_spans(
        word_rows: List[Tuple[int, int, int, str]],
    ) -> List[Tuple[int, int, str]]:
        """
        Colapsa palabras consecutivas con la misma etiqueta en un único span.
        """
        if not word_rows:
            return []
        spans: List[Tuple[int, int, str]] = []
        cur_lab: Optional[str] = None
        cur_s = cur_e = None

        for _, s_w, e_w, lbl in word_rows:
            if lbl == "O":
                # Finaliza el span actual si la etiqueta es 'O'
                if cur_lab is not None:
                    spans.append((cur_s, cur_e, cur_lab))
                    cur_lab = None
                continue

            if cur_lab is None:
                # Inicia un nuevo span
                cur_lab = lbl
                cur_s, cur_e = s_w, e_w
            elif lbl == cur_lab and s_w <= (cur_e or s_w) + 1:
                # Extiende el span actual si es del mismo tipo
                cur_e = max(cur_e, e_w)
            else:
                # Finaliza el span actual y comienza uno nuevo
                spans.append((cur_s, cur_e, cur_lab))
                cur_lab = lbl
                cur_s, cur_e = s_w, e_w

        if cur_lab is not None:
            spans.append((cur_s, cur_e, cur_lab))  # Agrega el último span
        return spans

    @staticmethod
    def _merge_overlapping(
        spans: List[Tuple[int, int, str]],
    ) -> List[Tuple[int, int, str]]:
        """
        Une spans solapados del mismo tipo en un único rango.
        """
        if not spans:
            return []
        spans.sort(key=lambda x: (x[0], x[1]))  # Ordena por inicio y fin
        merged = [spans[0]]
        for s, e, t in spans[1:]:
            ps, pe, pt = merged[-1]
            if t == pt and s <= pe:
                # Une spans si son del mismo tipo y se solapan
                merged[-1] = (ps, max(pe, e), pt)
            else:
                merged.append((s, e, t))
        return merged

    # ---------------------------------------------------------------------
    # API pública
    # ---------------------------------------------------------------------
    def predict(self, text: str) -> List[Dict[str, Any]]:
        """
        Predice entidades tipo-span en un texto dado.
        Devuelve una lista de diccionarios con {type, text, start, end, score, code}.
        """
        if not text:
            return []

        # Tokeniza el texto con offsets y ventanas deslizantes
        batch = self.tokenizer(
            text,
            return_offsets_mapping=True,
            return_overflowing_tokens=True,
            truncation=True,
            max_length=self.MAX_LEN,
            stride=self.STRIDE,
        )

        input_ids_list = batch["input_ids"]
        attn_list = batch["attention_mask"]
        offsets_list = batch["offset_mapping"]

        all_spans: List[Tuple[int, int, str]] = []

        with torch.no_grad():
            for i in range(len(input_ids_list)):
                # Prepara los tensores de entrada para el modelo
                input_ids = torch.tensor([input_ids_list[i]], device=self.device)
                attn_mask = torch.tensor([attn_list[i]], device=self.device)

                # Realiza la predicción y obtiene los logits
                logits = self.model(
                    input_ids=input_ids, attention_mask=attn_mask
                ).logits
                pred_ids = logits.argmax(dim=-1)[0].tolist()

                # Agrega predicciones por palabra
                word_ids = batch.word_ids(batch_index=i)
                offsets = offsets_list[i]
                word_rows = self._aggregate_tokens_to_words(word_ids, offsets, pred_ids)
                if not word_rows:
                    continue

                # Colapsa palabras consecutivas en spans
                spans = self._collapse_words_to_spans(word_rows)
                all_spans.extend(spans)

        # Une spans solapados
        merged = self._merge_overlapping(all_spans)

        # Construye la salida final
        out: List[Dict[str, Any]] = []
        seen = set()
        for s, e, t in merged:
            if not (0 <= s < e <= len(text)):
                continue
            key = (s, e, t)
            if key in seen:
                continue
            seen.add(key)
            out.append(
                {
                    "type": t,
                    "text": text[s:e],
                    "start": s,
                    "end": e,
                    "score": None,  # mantenemos None para no cambiar tu API actual
                    "code": None,
                }
            )
        return out

    def meta(self) -> Dict[str, Any]:
        """
        Devuelve metadatos útiles para debugging o telemetría.
        Incluye información sobre el modelo, dispositivo y etiquetas.
        """
        return {
            "model_id": self.model_id,
            "base_model_id": self.base_model_id,
            "num_labels": len(self.id2label),
            "max_len": self.MAX_LEN,
            "stride": self.STRIDE,
            "device": str(self.device),
            "labels": sorted(set(self.id2label.values())),
        }
