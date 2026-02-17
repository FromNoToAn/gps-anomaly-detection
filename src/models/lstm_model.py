# -*- coding: utf-8 -*-
"""LSTM-модель для предсказания ETA (время до следующей остановки)."""
import torch
import torch.nn as nn

from src.config import WINDOW_SIZE_SEC
from src.models.dataset import FEATURE_COLS

INPUT_SIZE = len(FEATURE_COLS)


class LSTMEtaModel(nn.Module):
    def __init__(
        self,
        input_size: int = INPUT_SIZE,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, F]
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        return self.fc(last).squeeze(-1)
