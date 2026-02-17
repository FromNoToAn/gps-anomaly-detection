# LSTM бейзлайн: предсказание ETA и оценка аномалий

По плану из TODO: предобработка датасета Novosibirsk и обучение LSTM для предсказания времени прибытия на остановки, с порогом аномалий.

## Структура

- **`src/config.py`** — пути (raw, trasses, pre_datasets) и константы (bearing ±20°, speed 7 км/ч, окно 60 сек и т.д.).
- **`src/preprocessing/`**
  - `trass_loader.py` — загрузка маршрутов из JSON, извлечение остановок (id, n).
  - `map_to_route.py` — маппинг GPS-точек на линию маршрута (без «прыжков»).
  - `trip_preprocess.py` — правка bearing/speed, ресэмпл 1 сек, добавление trip_id, vehicle_type, route_direction.
  - `segment_stops.py` — сегменты между остановками, целевая переменная ETA (сек до следующей остановки).
  - `pipeline.py` — запуск предобработки всех CSV из `row_datasets/Novosibirsk` → сохранение в `pre_datasets/Novosibirsk`.
- **`src/models/`**
  - `dataset.py` — скользящие окна (60 шагов по 1 сек), признаки: lat, lon, accuracy, bearing, speed; цель: eta_sec (масштабировано).
  - `lstm_model.py` — LSTM(5, 64, 2 слоя) → линейный слой → предсказание ETA.
  - `lightning_module.py` — PyTorch Lightning: `ETALightningModule` (train/val step, логирование).
  - `data_module.py` — `ETADataModule` (train/val split, DataLoader’ы).
  - `train.py` — функции `train_epoch` / `evaluate` для совместимости (основной запуск — через Lightning).
- **`src/anomaly.py`** — флаг аномалии, если |predicted_ETA − actual_ETA| > порог (по умолчанию 120 сек).

## Установка и запуск (Poetry)

Из **корня проекта** (VKR):

1. Установка зависимостей:
   ```bash
   poetry install
   ```

   **Если Poetry установлен только в Conda:** команду `poetry` нужно вызывать из активированного окружения, где установлен Poetry. Чтобы Poetry **не ставил пакеты в Conda**, а создал свой venv для проекта:
   ```bash
   conda activate env
   cd D:\!NGU!\4_year_2026\VKR
   poetry config virtualenvs.prefer-active-python false
   poetry env remove --all
   poetry install
   ```
   Тогда Poetry создаст отдельное виртуальное окружение (в кэше или в `.venv`), и ошибки с markupsafe в Conda не будет.

   **Если хотите обходиться без Conda при работе с проектом:** установите Poetry глобально (например, [официальный установщик](https://python-poetry.org/docs/#installation) или `pip install --user poetry` в любом Python), затем:
   ```bash
   conda deactivate
   cd D:\!NGU!\4_year_2026\VKR
   poetry install
   ```
   Команды ниже запускайте из **корня проекта** `VKR`, а не из подпапки `datasets`.

2. Предобработка:
   ```bash
   poetry run python run_preprocess.py
   ```
   Читает CSV из `datasets/row_datasets/Novosibirsk/`, для каждого маршрута подгружает `datasets/trasses/trasses_<N>.json`, маппит точки, правит bearing/speed, ресэмплит на 1 сек, считает ETA и сохраняет в `datasets/pre_datasets/Novosibirsk/*_preprocessed.csv`.

3. Обучение LSTM (PyTorch Lightning):
   ```bash
   poetry run python run_train_lstm.py
   ```
   Используется Lightning: TensorBoard-логи в `logs/`, чекпоинты в `checkpoints/` (лучшие по val_loss). Свой конфиг:
   ```bash
   poetry run python run_train_lstm.py --config config/train.yaml
   ```
   В `config/train.yaml` — число эпох, lr, batch_size, early stopping и т.д. Просмотр метрик:
   ```bash
   tensorboard --logdir logs
   ```

4. Аномалии: после получения предсказаний модели используйте `src.anomaly.flag_anomalies(predicted_eta_scaled, actual_eta_scaled, threshold_sec=120)`.

## Зависимости

- Python 3.9+ (задаётся в `pyproject.toml`)
- Управление зависимостями: **Poetry** (см. `pyproject.toml`) или `pip install -r requirements.txt`  
  Нужны: `torch`, `pytorch-lightning`, `tensorboard`, `PyYAML`, `pandas`, `numpy`.
