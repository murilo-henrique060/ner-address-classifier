from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer

MODEL_NAME = "ottema/bert-addresses-brazil"


def align_subwords(
    words: List[str],
    word_ids: List[Optional[int]],
    subtoken_preds: List[str],
    aligned: Optional[List[str]] = None,
    seen_word_ids: Optional[set] = None,
) -> List[str]:
    """Mapeia as predições de subtokens de volta aos tokens originais da palavra.
    
    Garante: len(aligned_preds) == len(words).
    """
    if aligned is None:
        aligned = ["O"] * len(words)
    if seen_word_ids is None:
        seen_word_ids = set()

    for sub_idx, w_id in enumerate(word_ids):
        if w_id is None:
            continue
        if 0 <= w_id < len(words) and w_id not in seen_word_ids:
            if sub_idx < len(subtoken_preds):
                aligned[w_id] = subtoken_preds[sub_idx]
            seen_word_ids.add(w_id)

    return aligned


class AddressNERPredictor:
    """Preditor para o modelo ottema/bert-addresses-brazil com alinhamento de subtokens."""

    def __init__(
        self,
        device: Optional[str] = None,
        batch_size: int = 32,
    ):
        self.batch_size = batch_size
        self.device = torch.device(
            device if device and device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
            self.model = AutoModelForTokenClassification.from_pretrained(MODEL_NAME)
        except Exception:
            self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True, local_files_only=True)
            self.model = AutoModelForTokenClassification.from_pretrained(MODEL_NAME, local_files_only=True)

        self.model.to(self.device)
        self.model.eval()

        raw_id2label = getattr(self.model.config, "id2label", {})
        self.id2label: Dict[int, str] = {int(k): str(v) for k, v in raw_id2label.items()}
        self.max_length: int = getattr(self.model.config, "max_position_embeddings", 512)

    def predict(self, sentences_words: List[List[str]]) -> List[List[str]]:
        """Prediz tags BIO para cada sentença, preservando o número de tokens original."""
        preds, _ = self.predict_with_scores(sentences_words)
        return preds

    def predict_with_scores(self, sentences_words: List[List[str]]) -> Tuple[List[List[str]], List[float]]:
        """Prediz tags BIO e retorna também o score médio de confiança por sentença."""
        all_aligned: List[List[str]] = []
        all_scores: List[float] = []

        for i in range(0, len(sentences_words), self.batch_size):
            batch_words = sentences_words[i : i + self.batch_size]

            encodings = self.tokenizer(
                batch_words,
                is_split_into_words=True,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_overflowing_tokens=True,
            )

            all_pred_ids: List[List[int]] = []
            all_token_probs: List[List[float]] = []
            total_chunks = encodings["input_ids"].size(0)

            for c_start in range(0, total_chunks, self.batch_size):
                c_end = c_start + self.batch_size
                input_ids = encodings["input_ids"][c_start:c_end].to(self.device)
                attention_mask = encodings["attention_mask"][c_start:c_end].to(self.device)

                with torch.no_grad():
                    logits = self.model(input_ids=input_ids, attention_mask=attention_mask).logits
                    probs = torch.softmax(logits, dim=-1)
                    max_probs, pred_ids = torch.max(probs, dim=-1)
                    all_pred_ids.extend(pred_ids.cpu().tolist())
                    all_token_probs.extend(max_probs.cpu().tolist())

            overflow_mapping = encodings["overflow_to_sample_mapping"].tolist()
            batch_aligned: List[List[str]] = [["O"] * len(w) for w in batch_words]
            seen_per_sample: List[set] = [set() for _ in batch_words]
            probs_per_sample: List[List[float]] = [[] for _ in batch_words]

            for chunk_idx, sample_idx in enumerate(overflow_mapping):
                word_ids = encodings.word_ids(batch_index=chunk_idx)
                sub_preds = [self.id2label.get(pid, "O") for pid in all_pred_ids[chunk_idx]]
                sub_probs = all_token_probs[chunk_idx]

                align_subwords(
                    words=batch_words[sample_idx],
                    word_ids=word_ids,
                    subtoken_preds=sub_preds,
                    aligned=batch_aligned[sample_idx],
                    seen_word_ids=seen_per_sample[sample_idx],
                )

                for w_id, prob in zip(word_ids, sub_probs):
                    if w_id is not None:
                        probs_per_sample[sample_idx].append(prob)

            all_aligned.extend(batch_aligned)
            for p_list in probs_per_sample:
                avg_score = round(sum(p_list) / len(p_list), 4) if p_list else 0.0
                all_scores.append(avg_score)

        return all_aligned, all_scores
