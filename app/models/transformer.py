from __future__ import annotations

import os
from typing import Any, Dict, List, Tuple

import torch
from peft import PeftModel
from transformers import AutoConfig, AutoModelForTokenClassification, AutoTokenizer

LABEL2ID: Dict[str, int] = {
    "B-BIOMARCADOR": 0,
    "B-CANCER": 1,
    "B-CIRUGIA": 2,
    "B-DOSIS": 3,
    "B-EDAD": 4,
    "B-FECHA": 5,
    "B-GLEASON": 6,
    "B-MEDICAMENTO": 7,
    "B-TNM": 8,
    "B-TRATAMIENTO": 9,
    "I-BIOMARCADOR": 10,
    "I-CANCER": 11,
    "I-CIRUGIA": 12,
    "I-DOSIS": 13,
    "I-EDAD": 14,
    "I-FECHA": 15,
    "I-GLEASON": 16,
    "I-MEDICAMENTO": 17,
    "I-TNM": 18,
    "I-TRATAMIENTO": 19,
    "O": 20,
}
ID2LABEL: Dict[int, str] = {v: k for k, v in LABEL2ID.items()}


class TransformerExtractor:

    def __init__(
        self,
        model_id: str | None = None,
        base_model_id: str | None = None,
        *,
        max_len: int = 512,
        stride: int = 64,
        device: str | None = None,
    ) -> None:
        self.model_id = model_id or os.getenv(
            "HF_MODEL_ID", "NicolasUnivalle/beto_prostata_peft"
        )
        self.base_model_id = base_model_id or os.getenv(
            "HF_BASE_MODEL_ID", "dccuchile/bert-base-spanish-wwm-cased"
        )
        self.hf_token = os.getenv("HF_TOKEN")

        self.MAX_LEN = max_len
        self.STRIDE = stride
        self.device = torch.device(device or "cpu")

        self.tokenizer = self._init_tokenizer()
        self.model = self._init_model()
        self.model.to(self.device).eval()

        self.id2label: Dict[int, str] = {
            int(k): v for k, v in self.model.config.id2label.items()
        }

    # ---------- Inicialización ----------
    def _init_tokenizer(self):
        try:
            return AutoTokenizer.from_pretrained(self.model_id, token=self.hf_token)
        except Exception:
            return AutoTokenizer.from_pretrained(
                self.base_model_id, token=self.hf_token
            )

    def _init_model(self):
        cfg = AutoConfig.from_pretrained(self.base_model_id, token=self.hf_token)
        cfg.num_labels = len(LABEL2ID)
        cfg.id2label = {int(i): str(lbl) for i, lbl in ID2LABEL.items()}
        cfg.label2id = {str(lbl): int(i) for lbl, i in LABEL2ID.items()}

        base = AutoModelForTokenClassification.from_pretrained(
            self.base_model_id,
            config=cfg,
            token=self.hf_token,
            ignore_mismatched_sizes=True,
        )
        model = PeftModel.from_pretrained(base, self.model_id, token=self.hf_token)
        model.config.id2label = cfg.id2label
        model.config.label2id = cfg.label2id
        return model

    @staticmethod
    def _collapse_bio_to_spans(
        tags: List[str], offsets: List[Tuple[int, int]]
    ) -> List[Tuple[int, int, str]]:
        spans: List[Tuple[int, int, str]] = []
        cur_label: str | None = None
        cur_s = cur_e = None

        for tag, (s, e) in zip(tags, offsets):
            if s == e:
                continue
            if tag == "O" or not tag:
                if cur_label is not None:
                    spans.append((cur_s, cur_e, cur_label))
                    cur_label = None
                continue
            if tag.startswith("B-"):
                if cur_label is not None:
                    spans.append((cur_s, cur_e, cur_label))
                cur_label = tag[2:]
                cur_s, cur_e = s, e
            elif tag.startswith("I-"):
                lab = tag[2:]
                if cur_label == lab and s <= (cur_e or s):
                    cur_e = e
                else:
                    if cur_label is not None:
                        spans.append((cur_s, cur_e, cur_label))
                    cur_label = lab
                    cur_s, cur_e = s, e
            else:
                if cur_label is not None:
                    spans.append((cur_s, cur_e, cur_label))
                cur_label = tag
                cur_s, cur_e = s, e

        if cur_label is not None:
            spans.append((cur_s, cur_e, cur_label))
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

    def _vote_label_per_word(self, token_tags: List[str]) -> str:
        counts: Dict[str, int] = {}
        for t in token_tags:
            if t == "O" or not t:
                continue
            grp = t[2:] if (t.startswith("B-") or t.startswith("I-")) else t
            counts[grp] = counts.get(grp, 0) + 1
        return max(counts, key=counts.get) if counts else "O"

    def _aggregate_tokens_to_words(
        self,
        word_ids: List[int | None],
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

    def _collapse_words_to_spans(
        self, word_rows: List[Tuple[int, int, int, str]]
    ) -> List[Tuple[int, int, str]]:
        if not word_rows:
            return []
        spans: List[Tuple[int, int, str]] = []
        cur_lab: str | None = None
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
                ).logits  # [1, L, C]
                pred_ids = logits.argmax(dim=-1)[0].tolist()

                word_ids = batch.word_ids(
                    batch_index=i
                )  # ids de palabra (None en especiales)
                offsets = offsets_list[i]

                # 1) agregación por palabra
                word_rows = self._aggregate_tokens_to_words(word_ids, offsets, pred_ids)
                if not word_rows:
                    continue

                # 2) colapsar palabras contiguas en spans
                spans = self._collapse_words_to_spans(word_rows)

                # 3) acumular
                all_spans.extend(spans)

        # 4) coser ventanas y deduplicar
        merged = self._merge_overlapping(all_spans)

        # 5) salida final
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
            "num_labels": len(LABEL2ID),
            "max_len": self.MAX_LEN,
            "stride": self.STRIDE,
            "device": str(self.device),
        }
