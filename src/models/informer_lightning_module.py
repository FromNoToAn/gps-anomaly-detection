# -*- coding: utf-8 -*-
"""
Lightning-модуль для Informer/AutoInformer-подобной модели ETA.
Логирование метрик такое же, но префикс в TensorBoard отдельный (задаётся в run_train_informer.py).
"""
from __future__ import annotations

import torch
import pytorch_lightning as pl
from torch.nn import MSELoss

from src.config import ETA_SCALE_SEC
from src.models.lightning_module import _eta_metrics
from src.models.informer_model import InformerEtaModel


class ETAInformerLightningModule(pl.LightningModule):
    def __init__(
        self,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        distill: bool = True,
        lr: float = 1e-3,
        eta_scale_sec: float = ETA_SCALE_SEC,
        scheduler_factor: float = 0.5,
        scheduler_patience: int = 4,
        scheduler_min_lr: float = 1e-6,
        scheduler_enabled: bool = True,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr
        self.eta_scale_sec = eta_scale_sec
        self.scheduler_enabled = scheduler_enabled
        self.scheduler_factor = scheduler_factor
        self.scheduler_patience = scheduler_patience
        self.scheduler_min_lr = scheduler_min_lr

        self.model = InformerEtaModel(
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            distill=distill,
        )
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
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        if not self.scheduler_enabled:
            return optimizer
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=self.scheduler_factor,
            patience=self.scheduler_patience,
            min_lr=self.scheduler_min_lr,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
                "monitor": "val_loss",
            },
        }

