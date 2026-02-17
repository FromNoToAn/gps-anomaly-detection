# -*- coding: utf-8 -*-
"""
Обучение LSTM ETA на PyTorch Lightning.
  Предобработка:  python run_preprocess.py
  Обучение:       python run_train_lstm.py
  С конфигом:     python run_train_lstm.py --config config/train.yaml
  TensorBoard:    tensorboard --logdir logs
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

# корень проекта в path
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import PRE_DATASETS_DIR
from src.models.data_module import ETADataModule
from src.models.lightning_module import ETALightningModule


def load_config(config_path: str | None) -> dict:
    """Загрузить YAML-конфиг или вернуть дефолты."""
    defaults = {
        "trainer": {"max_epochs": 50, "accelerator": "auto", "devices": 1},
        "model": {"hidden_size": 64, "num_layers": 2, "lr": 1e-3},
        "data": {"batch_size": 32, "val_frac": 0.2, "num_workers": 0},
        "logging": {"name": "lstm_eta", "save_dir": "logs", "version": None},
        "checkpoint": {
            "dirpath": "checkpoints",
            "filename": "eta-{epoch:02d}-{val_rmse_sec:.1f}",
            "monitor": "val_loss",
            "mode": "min",
            "save_top_k": 2,
            "save_last": True,
        },
        "early_stopping": {"enabled": False, "monitor": "val_loss", "patience": 10, "mode": "min"},
    }
    if not config_path or not Path(config_path).is_file():
        return defaults
    try:
        import yaml
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        # глубокое слияние не делаем — просто перезаписываем верхний уровень
        for k, v in (cfg or {}).items():
            if k in defaults and isinstance(v, dict) and isinstance(defaults[k], dict):
                defaults[k] = {**defaults[k], **v}
            else:
                defaults[k] = v
    except Exception as e:
        print(f"Warning: could not load config {config_path}: {e}")
    return defaults


def main():
    parser = argparse.ArgumentParser(description="Train LSTM ETA (Lightning)")
    parser.add_argument("--config", "-c", type=str, default=None, help="Path to train.yaml")
    parser.add_argument("--data_dir", type=str, default=PRE_DATASETS_DIR, help="Preprocessed CSV dir")
    args = parser.parse_args()

    config_path = Path(args.config) if args.config else None
    if config_path and not config_path.is_absolute():
        config_path = ROOT / config_path
    cfg = load_config(str(config_path) if config_path else None)
    tcfg = cfg["trainer"]
    mcfg = cfg["model"]
    dcfg = cfg["data"]
    log_cfg = cfg["logging"]
    ckpt_cfg = cfg["checkpoint"]
    es_cfg = cfg["early_stopping"]

    # DataModule
    data_dir = Path(args.data_dir)
    if not data_dir.is_dir():
        raise SystemExit(f"Data dir not found: {data_dir}. Run: python run_preprocess.py")
    dm = ETADataModule(
        preprocessed_dir=str(data_dir),
        batch_size=dcfg["batch_size"],
        val_frac=dcfg["val_frac"],
        num_workers=dcfg["num_workers"],
    )

    # Model
    model = ETALightningModule(
        hidden_size=mcfg["hidden_size"],
        num_layers=mcfg["num_layers"],
        lr=mcfg["lr"],
    )

    # Logger
    log_dir = ROOT / log_cfg["save_dir"]
    logger = TensorBoardLogger(
        save_dir=str(log_dir),
        name=log_cfg["name"],
        version=log_cfg.get("version"),
    )

    # Callbacks
    ckpt_dir = ROOT / ckpt_cfg["dirpath"]
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    callbacks = [
        ModelCheckpoint(
            dirpath=str(ckpt_dir),
            filename=ckpt_cfg["filename"],
            monitor=ckpt_cfg["monitor"],
            mode=ckpt_cfg["mode"],
            save_top_k=ckpt_cfg["save_top_k"],
            save_last=ckpt_cfg["save_last"],
            verbose=True,
        ),
    ]
    if es_cfg.get("enabled"):
        callbacks.append(
            EarlyStopping(
                monitor=es_cfg["monitor"],
                patience=es_cfg["patience"],
                mode=es_cfg["mode"],
                verbose=True,
            )
        )

    # Trainer
    trainer = pl.Trainer(
        max_epochs=tcfg["max_epochs"],
        accelerator=tcfg.get("accelerator", "auto"),
        devices=tcfg.get("devices", 1),
        gradient_clip_val=tcfg.get("gradient_clip_val"),
        logger=logger,
        callbacks=callbacks,
    )

    trainer.fit(model, datamodule=dm)
    print(f"Checkpoints: {ckpt_dir}")
    print(f"TensorBoard: tensorboard --logdir {log_dir}")
    return model


if __name__ == "__main__":
    main()
