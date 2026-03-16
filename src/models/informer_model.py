# -*- coding: utf-8 -*-
"""
Минимальная Informer/AutoInformer-подобная модель для ETA на фиксированном окне.

Важно: это учебный пример для сравнения с LSTM.
Мы используем Transformer Encoder с positional encoding и лёгким "distilling"
(Conv1d + stride), что близко по духу Informer (но без ProbSparse attention).
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn

from src.models.dataset import FEATURE_COLS


INPUT_SIZE = len(FEATURE_COLS)


class _SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 2048):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe, persistent=False)  # [max_len, d_model]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, D]
        t = x.size(1)
        return x + self.pe[:t].unsqueeze(0)


class InformerEtaModel(nn.Module):
    """
    Encoder-only модель: (B,T,F)->ETA.
    - Проекция признаков в d_model
    - Positional encoding
    - (опционально) Distilling: Conv1d + stride=2 (уменьшаем T)
    - TransformerEncoder
    - Pooling по времени (mean) + Linear -> scalar
    """

    def __init__(
        self,
        input_size: int = INPUT_SIZE,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        distill: bool = True,
    ):
        super().__init__()
        self.proj = nn.Linear(input_size, d_model)
        self.pos = _SinusoidalPositionalEncoding(d_model=d_model)
        self.distill = distill
        if distill:
            # [B,T,D] -> Conv1d expects [B,D,T]
            self.conv = nn.Sequential(
                nn.Conv1d(d_model, d_model, kernel_size=3, padding=1),
                nn.GELU(),
                nn.Conv1d(d_model, d_model, kernel_size=3, padding=1, stride=2),
                nn.GELU(),
            )
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 1),
        )

    def forward(self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # x: [B, T, F]
        h = self.proj(x)
        h = self.pos(h)

        if self.distill:
            h = self.conv(h.transpose(1, 2)).transpose(1, 2)  # back to [B,T',D]
            if key_padding_mask is not None:
                # Если были маски для паддинга, при stride=2 их надо тоже "сжать".
                # Для учебного случая (фиксированное окно без паддинга) маска обычно None.
                key_padding_mask = key_padding_mask[:, ::2]

        h = self.encoder(h, src_key_padding_mask=key_padding_mask)
        pooled = h.mean(dim=1)  # [B, D]
        return self.head(pooled).squeeze(-1)

