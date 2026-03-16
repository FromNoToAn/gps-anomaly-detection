# Informer/AutoInformer‑подобная модель для ETA (пример)

Этот документ описывает альтернативную модель ETA на базе Transformer Encoder, сделанную **по мотивам Informer/AutoInformer** для сравнения с LSTM.

Важно: в проекте это **учебный пример** (без ProbSparse attention), но с идеями, похожими на Informer:
- positional encoding;
- “distilling” по времени (Conv1d со stride=2), чтобы сжимать последовательность.

## Где лежит код

- Модель: `src/models/informer_model.py` (`InformerEtaModel`)
- LightningModule: `src/models/informer_lightning_module.py` (`ETAInformerLightningModule`)
- Конфиг: `config/informer.yaml`
- Скрипт обучения: `run_train_informer.py`

## Вход/цель (как у LSTM)

Используется тот же `ETAWindowDataset`:

- вход: окно `[T, F]`, где \(T = WINDOW_SIZE_SEC\) (обычно 60), признаки:
  - `lat, lon, accuracy, bearing, speed` (нормализованные в датасете);
- цель: `eta_sec` на последнем шаге окна, масштабированная на `ETA_SCALE_SEC`.

## Запуск

1) Предобработка (если ещё не делал):

```bash
python run_preprocess.py
```

2) Обучение Informer‑подобной модели:

```bash
python run_train_informer.py
```

или с конфигом:

```bash
python run_train_informer.py --config config/informer.yaml
```

## Trainer / Scheduler / Логирование

Это обычный PyTorch Lightning training loop:

- `trainer.max_epochs` и прочее — секция `trainer:` в `config/informer.yaml`
- LR scheduler — `ReduceLROnPlateau` по `val_loss` (секция `model.scheduler:`)
- Метрики логируются так же, как в LSTM:
  - `train_loss`, `val_loss`
  - `train_rmse_sec`, `val_rmse_sec`
  - `val_mae_sec`, `val_r2`, и др.

Логи идут в отдельный namespace TensorBoard:
- `logs/informer_eta/`

Просмотр:

```bash
tensorboard --logdir logs
```

## Чекпоинты

Чекпоинты сохраняются в `checkpoints/` (имя задаётся в `config/informer.yaml`, например `informer-eta-...ckpt`).