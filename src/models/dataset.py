# -*- coding: utf-8 -*-
"""
Датасет для LSTM: скользящие окна по предобработанным поездкам.
Признаки: lat, lon, accuracy, bearing, speed (нормализованные).
Цель: eta_sec (время до следующей остановки, сек), масштабированная для обучения.
"""
import os
from typing import List, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.config import PRE_DATASETS_DIR, WINDOW_SIZE_SEC, ETA_SCALE_SEC

FEATURE_COLS = ["lat", "lon", "accuracy", "bearing", "speed"]


def _normalize_features(df: pd.DataFrame) -> np.ndarray:
    """Нормализация признаков: lat/lon центрируем, accuracy/360 для bearing, speed/50."""
    x = np.zeros((len(df), len(FEATURE_COLS)), dtype=np.float32)
    for j, col in enumerate(FEATURE_COLS):
        v = df[col].values
        if col == "lat":
            x[:, j] = (v - 55.0) / 0.5
        elif col == "lon":
            x[:, j] = (v - 83.0) / 0.5
        elif col == "accuracy":
            x[:, j] = np.clip(v / 50.0, 0, 5)
        elif col == "bearing":
            x[:, j] = v / 360.0
        elif col == "speed":
            x[:, j] = np.clip(v / 50.0, 0, 3)
    return x


class ETAWindowDataset(Dataset):
    """
    Скользящее окно длины WINDOW_SIZE_SEC (при 1 сек = 60 шагов).
    Каждый элемент: (X: [T, F], y: scalar) — окно признаков и ETA на последнем шаге.
    Берём только окна, где у последней точки есть валидный eta_sec.
    """

    def __init__(
        self,
        preprocessed_dir: str = PRE_DATASETS_DIR,
        window_size: int = WINDOW_SIZE_SEC,
        eta_scale: float = ETA_SCALE_SEC,
        trip_ids: Optional[List[str]] = None,
    ):
        self.window_size = window_size
        self.eta_scale = eta_scale
        self.samples_x: List[np.ndarray] = []
        self.samples_y: List[float] = []
        csv_files = [
            f for f in os.listdir(preprocessed_dir)
            if f.endswith("_preprocessed.csv")
        ]
        for f in sorted(csv_files):
            trip_id = f.replace("_preprocessed.csv", "")
            if trip_ids is not None and trip_id not in trip_ids:
                continue
            path = os.path.join(preprocessed_dir, f)
            df = pd.read_csv(path)
            if "eta_sec" not in df.columns or df["eta_sec"].isna().all():
                continue
            for c in FEATURE_COLS:
                if c not in df.columns:
                    break
            else:
                x = _normalize_features(df)
                eta = df["eta_sec"].values.astype(np.float32)
                for i in range(len(df) - window_size):
                    if np.isnan(eta[i + window_size - 1]) or eta[i + window_size - 1] < 0:
                        continue
                    self.samples_x.append(x[i : i + window_size])
                    self.samples_y.append(eta[i + window_size - 1] / eta_scale)
        self.samples_y = np.array(self.samples_y, dtype=np.float32)

    def __len__(self) -> int:
        return len(self.samples_x)

    def __getitem__(self, idx: int):
        x = torch.from_numpy(self.samples_x[idx])
        y = torch.tensor(self.samples_y[idx], dtype=torch.float32)
        return x, y
