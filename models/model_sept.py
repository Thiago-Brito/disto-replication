"""
model_sept.py
-------------

Implementação do modelo SepT (DISTO) baseado em DistilRoBERTa.

Entrada concatenada como texto literal:
    "[QUES] {Q} [ANS] {A} [DIS] {D} [ART] {Ar}"

Sem tokens especiais reais adicionados ao vocabulário: os marcadores são apenas strings no texto.
Saída: sigmoid(Linear(hidden_size, 1)) aplicada ao vetor do token inicial (posição 0 / <s>), similar ao CLS.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer


def build_concat_text(q: str, a: str, d: str, art: str) -> str:
    return f"[QUES] {q} [ANS] {a} [DIS] {d} [ART] {art}"


class SepTDISTO(nn.Module):
    def __init__(self, model_name: str = "distilroberta-base"):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden = self.encoder.config.hidden_size
        self.head = nn.Linear(hidden, 1)
        self.sigmoid = nn.Sigmoid()

    def encode_concat(self, batch: List[Dict[str, str]], max_length: int = 512, device: torch.device | None = None) -> Dict[str, torch.Tensor]:
        texts = [build_concat_text(x["question"], x["answer"], x["distractor"], x["article"]) for x in batch]
        enc = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        if device is not None:
            enc = {k: v.to(device) for k, v in enc.items()}
        return enc

    def forward(self, enc: Dict[str, torch.Tensor]) -> torch.Tensor:
        out = self.encoder(**enc)
        hidden = out.last_hidden_state  # [B, T, H]
        cls = hidden[:, 0, :]  # DistilRoBERTa: primeiro token (<s>) como CLS
        logits = self.head(cls)  # [B, 1]
        probs = self.sigmoid(logits).squeeze(-1)  # [B]
        return probs

# Backwards-compatible alias expected by some scripts
SepT = SepTDISTO
