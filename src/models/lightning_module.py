# -*- coding: utf-8 -*-
"""
Lightning-модуль для обучения LSTM ETA: логирование, чекпоинты, метрики.

Метрики (логируются в TensorBoard и prog_bar):
  - train/val_loss, train/val_rmse_sec  — MSE и RMSE в секундах
  - val_mae_sec   — средняя абсолютная ошибка (сек), устойчивее к выбросам чем RMSE
  - val_median_ae_sec — медианная ошибка (сек), типичная ошибка без влияния хвостов
  - val_bias_sec  — среднее (pred - target) в сек: >0 — переоценка ETA, <0 — недооценка
  - val_mape      — средняя абсолютная процентная ошибка (%), относительная точность
  - val_r2        — R²: 1 = идеально, 0 = как предсказание средним, <0 — хуже среднего
"""
import torch
import pytorch_lightning as pl
from torch.nn import MSELoss

from src.config import ETA_SCALE_SEC, LSTM_HIDDEN_SIZE, LSTM_NUM_LAYERS
from src.models.lstm_model import LSTMEtaModel


def _eta_metrics(pred: torch.Tensor, target: torch.Tensor, scale_sec: float) -> dict:
    """Метрики регрессии ETA: pred и target в масштабированном виде [0, 1]."""
    pred_sec = pred.float() * scale_sec
    target_sec = target.float() * scale_sec
    err_sec = pred_sec - target_sec
    abs_err_sec = err_sec.abs()

    mae_sec = abs_err_sec.mean().item()
    rmse_sec = (err_sec.pow(2).mean().sqrt()).item()
    median_ae_sec = abs_err_sec.median().item()
    bias_sec = err_sec.mean().item()

    # MAPE (%) только где target_sec > 10 сек, иначе деление на ноль
    mask = target_sec > 10.0
    if mask.any():
        mape = (abs_err_sec[mask] / target_sec[mask].clamp(min=1e-3) * 100).mean().item()
    else:
        mape = float("nan")

    # R² (1 - SS_res / SS_tot)
    ss_res = (target - pred).pow(2).sum()
    ss_tot = (target - target.mean()).pow(2).sum().clamp(min=1e-8)
    r2 = (1 - ss_res / ss_tot).item()

    return {
        "mae_sec": mae_sec,
        "rmse_sec": rmse_sec,
        "median_ae_sec": median_ae_sec,
        "bias_sec": bias_sec,
        "mape": mape,
        "r2": r2,
    }


class ETALightningModule(pl.LightningModule):
    """LSTM для предсказания ETA (время до следующей остановки)."""

    def __init__(
        self,
        hidden_size: int = LSTM_HIDDEN_SIZE,
        num_layers: int = LSTM_NUM_LAYERS,
        lr: float = 1e-3,
        eta_scale_sec: float = ETA_SCALE_SEC,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr
        self.eta_scale_sec = eta_scale_sec
        self.model = LSTMEtaModel(hidden_size=hidden_size, num_layers=num_layers)
        self.criterion = MSELoss()
        self._val_preds: list = []
        self._val_targets: list = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def _shared_step(self, batch):
        x, y = batch
        pred = self(x)
        loss = self.criterion(pred, y)
        rmse_sec = (loss.detach().sqrt() * self.eta_scale_sec).item()
        return loss, pred.detach(), y.detach(), rmse_sec

    def training_step(self, batch, batch_idx):
        loss, _pred, _target, rmse_sec = self._shared_step(batch)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train_rmse_sec", rmse_sec, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, pred, target, rmse_sec = self._shared_step(batch)
        self._val_preds.append(pred)
        self._val_targets.append(target)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_rmse_sec", rmse_sec, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def on_validation_epoch_end(self):
        if not self._val_preds:
            return
        pred = torch.cat(self._val_preds)
        target = torch.cat(self._val_targets)
        m = _eta_metrics(pred, target, self.eta_scale_sec)
        self.log("val_mae_sec", m["mae_sec"], prog_bar=True)
        self.log("val_median_ae_sec", m["median_ae_sec"])
        self.log("val_bias_sec", m["bias_sec"])
        self.log("val_mape", m["mape"])
        self.log("val_r2", m["r2"], prog_bar=True)
        self._val_preds.clear()
        self._val_targets.clear()

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
