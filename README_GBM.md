# GBM (XGBoost) модель для ETA

Этот документ описывает альтернативную модель ETA на **градиентном бустинге** (XGBoost), чтобы сравнить её с LSTM бейзлайном.

## Идея модели

Мы используем тот же датасет окон, что и для LSTM (`ETAWindowDataset`):

- вход: окно длины `WINDOW_SIZE_SEC` (обычно 60 секунд) с признаками `lat, lon, accuracy, bearing, speed`;
- цель: `eta_sec` на последнем шаге окна (масштабированная на `ETA_SCALE_SEC`).

Для XGBoost каждое окно `[T, F]` преобразуется в плоский вектор `[T*F]` и обучается регрессия.

## Конфиг

- `config/gbm.yaml` — гиперпараметры XGBoost и настройки логирования:
  - `model.n_estimators`, `model.learning_rate`, `model.max_depth`, регуляризация и т.д.
  - `data.val_frac` — доля валидации (как у LSTM).
  - `logging.name` — имя прогона в TensorBoard (по умолчанию `gbm_eta`).
  - `seed` — фиксирует train/val split.

Важно: `batch_size` для XGBoost **не используется** (обучение идёт на массиве целиком).

## Запуск

1) Предобработка (если ещё не делал):

```bash
python run_preprocess.py
```

2) Обучение GBM:

```bash
python run_train_gbm.py
```

Или явно указать конфиг:

```bash
python run_train_gbm.py --config config/gbm.yaml
```

## Логи и артефакты

- **TensorBoard**: `logs/gbm_eta/`
  - `val/mae_sec`
  - `val/rmse_sec`
  - `val/r2`

Запуск просмотра:

```bash
tensorboard --logdir logs
```

- **Модель** сохраняется в: `checkpoints/gbm_eta.xgb`


