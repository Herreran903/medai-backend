from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple

import torch
from peft import PeftModel
from transformers import AutoConfig, AutoModelForTokenClassification, AutoTokenizer


class TransformerExtractor:
    def __init__(
        self,
        model_id: Optional[str] = None,
        base_model_id: Optional[str] = None,
        *,
        max_len: int = 512,
        stride: int = 64,
        device: Optional[str] = None,
    ) -> None:
        self.model_id = model_id or os.getenv(
            "HF_MODEL_ID", "NicolasUnivalle/beto_vm_peft"
        )
        self.base_model_id = base_model_id or os.getenv(
            "HF_BASE_MODEL_ID", "dccuchile/bert-base-spanish-wwm-cased"
        )
        self.hf_token = os.getenv("HF_TOKEN")

        self.MAX_LEN = max_len
        self.STRIDE = stride
        if device:
            self.device = torch.device(device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.tokenizer = self._init_tokenizer()
        self.model = self._init_model()
        self.model.to(self.device).eval()

        if not getattr(self.model.config, "id2label", None):
            raise ValueError(
                "El modelo no expone config.id2label. "
                "Asegúrate de que tu adapter/modelo incluya el mapeo de etiquetas."
            )
        self.id2label: Dict[int, str] = {
            int(k): str(v) for k, v in self.model.config.id2label.items()
        }

    def _init_tokenizer(self):
        try:
            return AutoTokenizer.from_pretrained(self.model_id, token=self.hf_token)
        except Exception:
            return AutoTokenizer.from_pretrained(
                self.base_model_id, token=self.hf_token
            )

    def _init_model(self):
        print("Intentando cargar modelo completo desde:", self.model_id, self.hf_token)
        try:
            full = AutoModelForTokenClassification.from_pretrained(
                self.model_id, token=self.hf_token
            )
            print("Modelo completo cargado correctamente.", full.config)
            if getattr(full.config, "id2label", None):
                return full
        except Exception as e:
            print("No se pudo cargar modelo completo:", e)
            pass

        try:
            peft_cfg = PeftConfig.from_pretrained(self.model_id, token=self.hf_token)
            base_name = peft_cfg.base_model_name_or_path or self.base_model_id
        except Exception:
            peft_cfg = None
            base_name = self.base_model_id

        base_cfg = AutoConfig.from_pretrained(base_name, token=self.hf_token)
        try:
            adapter_cfg = AutoConfig.from_pretrained(self.model_id, token=self.hf_token)
        except Exception:
            adapter_cfg = None

        def _as_int(d):
            return {int(k): str(v) for k, v in d.items()}

        def _as_str(d):
            return {str(k): int(v) for k, v in d.items()}

        if adapter_cfg is not None:
            if getattr(adapter_cfg, "num_labels", None):
                base_cfg.num_labels = adapter_cfg.num_labels
            if getattr(adapter_cfg, "id2label", None):
                base_cfg.id2label = _as_int(adapter_cfg.id2label)
            if getattr(adapter_cfg, "label2id", None):
                base_cfg.label2id = _as_str(adapter_cfg.label2id)

        if getattr(base_cfg, "num_labels", None) in (None, 2):
            raise ValueError(
                "No pude determinar num_labels/id2label/label2id. "
                "Si usas adapter PEFT, asegúrate de que el repo del adapter tenga esos campos en config.json "
                "o fija NUM_LABELS/LABELS_JSON por entorno."
            )

        base = AutoModelForTokenClassification.from_pretrained(
            base_name,
            config=base_cfg,
            token=self.hf_token,
            ignore_mismatched_sizes=True,
        )
        model = PeftModel.from_pretrained(base, self.model_id, token=self.hf_token)

        model.config.id2label = base_cfg.id2label
        model.config.label2id = base_cfg.label2id
        return model

    def _vote_label_per_word(self, token_tags: List[str]) -> str:
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
        by_word: Dict[int, List[Tuple[str, Tuple[int, int]]]] = {}
        for tok_idx, w_id in enumerate(word_ids):
            if w_id is None:
                continue
            s, e = offsets[tok_idx]
            if s == e:
                continue
            tag = self.id2label.get(int(pred_ids[tok_idx]), "O")
            by_word.setdefault(w_id, []).append((tag, (s, e)))

        rows: List[Tuple[int, int, int, str]] = []
        for w_id, items in by_word.items():
            s_word = min(se[1][0] for se in items)
            e_word = max(se[1][1] for se in items)
            label_group = self._vote_label_per_word([t for t, _ in items])
            rows.append((w_id, s_word, e_word, label_group))

        rows.sort(key=lambda r: r[0])
        return rows

    @staticmethod
    def _collapse_words_to_spans(
        word_rows: List[Tuple[int, int, int, str]],
    ) -> List[Tuple[int, int, str]]:
        if not word_rows:
            return []
        spans: List[Tuple[int, int, str]] = []
        cur_lab: Optional[str] = None
        cur_s = cur_e = None

        for _, s_w, e_w, lbl in word_rows:
            if lbl == "O":
                if cur_lab is not None:
                    spans.append((cur_s, cur_e, cur_lab))
                    cur_lab = None
                continue

            if cur_lab is None:
                cur_lab = lbl
                cur_s, cur_e = s_w, e_w
            elif lbl == cur_lab and s_w <= (cur_e or s_w) + 1:
                cur_e = max(cur_e, e_w)
            else:
                spans.append((cur_s, cur_e, cur_lab))
                cur_lab = lbl
                cur_s, cur_e = s_w, e_w

        if cur_lab is not None:
            spans.append((cur_s, cur_e, cur_lab))
        return spans

    @staticmethod
    def _merge_overlapping(
        spans: List[Tuple[int, int, str]],
    ) -> List[Tuple[int, int, str]]:
        if not spans:
            return []
        spans.sort(key=lambda x: (x[0], x[1]))
        merged = [spans[0]]
        for s, e, t in spans[1:]:
            ps, pe, pt = merged[-1]
            if t == pt and s <= pe:
                merged[-1] = (ps, max(pe, e), pt)
            else:
                merged.append((s, e, t))
        return merged

    def predict(self, text: str) -> List[Dict[str, Any]]:
        if not text:
            return []

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
                input_ids = torch.tensor([input_ids_list[i]], device=self.device)
                attn_mask = torch.tensor([attn_list[i]], device=self.device)

                logits = self.model(
                    input_ids=input_ids, attention_mask=attn_mask
                ).logits
                pred_ids = logits.argmax(dim=-1)[0].tolist()

                word_ids = batch.word_ids(batch_index=i)
                offsets = offsets_list[i]

                word_rows = self._aggregate_tokens_to_words(word_ids, offsets, pred_ids)
                if not word_rows:
                    continue

                spans = self._collapse_words_to_spans(word_rows)

                all_spans.extend(spans)

        merged = self._merge_overlapping(all_spans)

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
                    "score": None,
                    "code": None,
                }
            )
        return out

    def meta(self) -> Dict[str, Any]:
        return {
            "model_id": self.model_id,
            "base_model_id": self.base_model_id,
            "num_labels": len(self.id2label),
            "max_len": self.MAX_LEN,
            "stride": self.STRIDE,
            "device": str(self.device),
            "labels": list(sorted(set(self.id2label.values()))),
        }
