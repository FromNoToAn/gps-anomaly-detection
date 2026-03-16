# -*- coding: utf-8 -*-
"""
Обучение градиентного бустинга (XGBoost) для ETA.

Идея: использовать тот же ETAWindowDataset, что и для LSTM, но
превратить каждое окно [T, F] в плоский вектор T*F и обучить
регрессию ETA на последнем шаге.

Запуск:
  python run_train_gbm.py
  python run_train_gbm.py --config config/gbm.yaml
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Tuple

import numpy as np
from torch.utils.data import random_split
from torch.utils.tensorboard import SummaryWriter

from src.config import PRE_DATASETS_DIR, ETA_SCALE_SEC, SEED
from src.models.dataset import ETAWindowDataset


def _load_config(path: Path | None) -> dict:
    defaults = {
        "model": {
            "type": "xgboost",
            "n_estimators": 400,
            "max_depth": 6,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_lambda": 1.0,
            "reg_alpha": 0.0,
            "min_child_weight": 1.0,
        },
        "data": {"batch_size": 2048, "val_frac": 0.2},
        "logging": {"name": "gbm_eta", "save_dir": "logs", "version": None},
        "seed": SEED,
    }
    if not path or not path.is_file():
        return defaults
    try:
        import yaml

        with open(path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        for k, v in cfg.items():
            if k in defaults and isinstance(v, dict) and isinstance(defaults[k], dict):
                defaults[k] = {**defaults[k], **v}
            else:
                defaults[k] = v
    except Exception as e:
        print(f"Warning: could not load GBM config {path}: {e}")
    return defaults


def _make_arrays(ds: ETAWindowDataset, val_frac: float, seed: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Превращает ETAWindowDataset в train/val массивы для XGBoost.
    X: [N, T, F] -> [N, T*F], y: [N].
    """
    import torch

    n_val = int(len(ds) * val_frac)
    n_train = len(ds) - n_val
    gen = torch.Generator().manual_seed(seed)
    train_ds, val_ds = random_split(ds, [n_train, n_val], generator=gen)

    def to_arrays(subset):
        xs, ys = [], []
        for x, y in subset:
            xs.append(x.numpy().reshape(-1))  # [T, F] -> [T*F]
            ys.append(float(y.item()))
        return np.stack(xs, axis=0), np.array(ys, dtype=np.float32)

    x_train, y_train = to_arrays(train_ds)
    x_val, y_val = to_arrays(val_ds)
    return x_train, y_train, x_val, y_val


def main():
    parser = argparse.ArgumentParser(description="Train GBM ETA (XGBoost)")
    parser.add_argument("--config", "-c", type=str, default=None, help="Path to gbm.yaml")
    parser.add_argument("--data_dir", type=str, default=PRE_DATASETS_DIR, help="Preprocessed CSV dir")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    cfg_path = Path(args.config) if args.config else None
    if cfg_path is None:
        default_cfg = root / "config" / "gbm.yaml"
        if default_cfg.is_file():
            cfg_path = default_cfg
    elif not cfg_path.is_absolute():
        cfg_path = root / cfg_path
    cfg = _load_config(cfg_path)
    mcfg = cfg["model"]
    dcfg = cfg["data"]
    log_cfg = cfg["logging"]
    seed = int(cfg.get("seed", SEED))

    np.random.seed(seed)

    data_dir = Path(args.data_dir)
    if not data_dir.is_dir():
        raise SystemExit(f"Data dir not found: {data_dir}. Run: python run_preprocess.py")

    print(f"GBM config: {cfg_path if cfg_path else 'defaults'}, seed={seed}")

    ds = ETAWindowDataset(preprocessed_dir=str(data_dir))
    if len(ds) == 0:
        raise SystemExit("No samples in preprocessed data. Run: python run_preprocess.py")

    x_train, y_train, x_val, y_val = _make_arrays(ds, dcfg["val_frac"], seed=seed)
    print(f"Train samples: {x_train.shape}, Val samples: {x_val.shape}")

    from xgboost import XGBRegressor
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    model = XGBRegressor(
        n_estimators=mcfg["n_estimators"],
        max_depth=mcfg["max_depth"],
        learning_rate=mcfg["learning_rate"],
        subsample=mcfg["subsample"],
        colsample_bytree=mcfg["colsample_bytree"],
        reg_lambda=mcfg["reg_lambda"],
        reg_alpha=mcfg["reg_alpha"],
        min_child_weight=mcfg["min_child_weight"],
        objective="reg:squarederror",
        n_jobs=os.cpu_count() or 4,
        random_state=seed,
    )

    log_dir = root / log_cfg["save_dir"] / log_cfg["name"]
    log_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(log_dir))

    model.fit(
        x_train,
        y_train,
        eval_set=[(x_train, y_train), (x_val, y_val)],
        verbose=False,
    )

    # Оценка на валидации (в секундах)
    pred_val = model.predict(x_val)
    mae = mean_absolute_error(y_val * ETA_SCALE_SEC, pred_val * ETA_SCALE_SEC)
    rmse = mean_squared_error(y_val * ETA_SCALE_SEC, pred_val * ETA_SCALE_SEC, squared=False)
    r2 = r2_score(y_val, pred_val)

    print(f"GBM val_mae_sec={mae:.3f} val_rmse_sec={rmse:.3f} val_r2={r2:.3f}")
    writer.add_scalar("val/mae_sec", mae, 0)
    writer.add_scalar("val/rmse_sec", rmse, 0)
    writer.add_scalar("val/r2", r2, 0)
    writer.flush()
    writer.close()

    # Сохранить модель рядом с чекпоинтами LSTM
    out_dir = root / "checkpoints"
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / "gbm_eta.xgb"
    model.save_model(str(model_path))
    print(f"GBM model saved to: {model_path}")
    print(f"TensorBoard (GBM): tensorboard --logdir {log_dir.parent}")


if __name__ == "__main__":
    main()

