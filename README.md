# Предсказание ETA и оценка аномалий (ВКР)

Проект для предсказания времени прибытия автобуса на следующую остановку (ETA) по данным GPS и оценки аномалий. Данные — поездки по маршрутам общественного транспорта (Новосибирск). Модель — LSTM по скользящему окну признаков.

## Структура проекта

```
VKR/
├── config/
│   └── train.yaml          # Параметры обучения (эпохи, lr, early stopping и т.д.)
├── datasets/
│   ├── row_datasets/Novosibirsk/   # Исходные CSV поездок (time, lat, lon, accuracy, bearing, speed)
│   ├── trasses/                     # Маршруты: trasses_38.json, trasses_52.json, trasses_72.json
│   └── pre_datasets/Novosibirsk/    # Предобработанные CSV (после run_preprocess.py)
├── logs/                    # TensorBoard (после обучения)
├── checkpoints/             # Чекпоинты модели (лучшие по val_loss)
├── src/
│   ├── config.py            # Пути и константы (окно 60 сек, bearing ±20°, speed 7 км/ч и т.д.)
│   ├── preprocessing/       # Предобработка
│   │   ├── trass_loader.py  # Загрузка маршрутов из JSON, остановки с id из trass
│   │   ├── map_to_route.py  # Маппинг GPS на линию маршрута, route_frac
│   │   ├── trip_preprocess.py # Bearing/speed по TODO, ресэмпл 1 сек, bearing с маршрута
│   │   ├── segment_stops.py # Сегменты по route_frac, current_stop_id/next_stop_id (id из trass), ETA
│   │   └── pipeline.py      # Обработка всех CSV → pre_datasets
│   ├── models/              # LSTM и обучение
│   │   ├── dataset.py       # Скользящие окна 60 сек, признаки: lat, lon, accuracy, bearing, speed
│   │   ├── lstm_model.py    # LSTM → линейный слой → ETA (масштабировано)
│   │   ├── lightning_module.py # PyTorch Lightning (train/val, логирование)
│   │   ├── data_module.py   # DataModule: загрузка, train/val split, DataLoader
│   │   └── train.py         # train_epoch / evaluate (без Lightning)
│   └── anomaly.py           # Флаг аномалии: |pred_ETA − actual_ETA| > порог (120 сек)
├── run_preprocess.py        # Запуск предобработки
├── run_train_lstm.py        # Обучение LSTM (Lightning)
└── requirements.txt         # Зависимости (pip install -r requirements.txt)
```

## Данные

- **Исходные поездки**: CSV в `datasets/row_datasets/Novosibirsk/`. Имя файла: `YYYYMMDD_<маршрут>_<str|rev>.csv` (например `20260123_52_str.csv`). Колонки: `time`, `lat`, `lon`, `accuracy`, `bearing`, `speed`.
- **Маршруты (trass)**: JSON в `datasets/trasses/` — полилиния и остановки с полями `id`, `n` (название). Маршруты могут быть закольцованы (одна и та же остановка в начале и в конце с разными id).
- **После предобработки**: в каждом CSV добавлены `route_frac`, `current_stop_id`, `next_stop_id` (id из trass), `eta_sec` (секунды до следующей остановки), а также `trip_id`, `vehicle_type`, `route_direction`. Точки приведены к сетке 1 сек; координаты и bearing — по геометрии маршрута.

## Предобработка

1. Маппинг GPS на маршрут (без скачков), расчёт `route_frac` (доля пути 0…1).
2. Bearing: проверка ±20° от направления маршрута; при выходе — смесь предыдущего bearing и случайного угла в диапазоне (см. TODO).
3. Speed: при отклонении от скорости по двум точкам > 7 км/ч — замена на расчётную.
4. Ресэмплинг на 1 сек: интерполяция `route_frac`, `speed`, `accuracy` по времени; `lat`, `lon`, `bearing` — по геометрии маршрута в данной точке (повороты не сглаживаются).
5. Сегменты и ETA: текущая/следующая остановка по положению вдоль маршрута (`route_frac`), id остановок из trass (`current_stop_id`, `next_stop_id`). ETA — время до достижения позиции следующей остановки по маршруту.

Запуск (из корня проекта, с активированным conda env):

```bash
python run_preprocess.py
```

Читает все CSV из `datasets/row_datasets/Novosibirsk/`, для каждого ищет `datasets/trasses/trasses_<маршрут>.json`, обрабатывает и сохраняет в `datasets/pre_datasets/Novosibirsk/<имя>_preprocessed.csv`. При отсутствии trass файл пропускается.

## Обучение

LSTM по окну 60 шагов (60 сек): признаки — lat, lon, accuracy, bearing, speed (нормализованные); цель — масштабированный `eta_sec`. Используется PyTorch Lightning, логи в TensorBoard, чекпоинты по val_loss.

Из корня проекта:

```bash
python run_train_lstm.py
```

С конфигом:

```bash
python run_train_lstm.py --config config/train.yaml
```

Просмотр метрик:

```bash
tensorboard --logdir logs
```

В `config/train.yaml` задаются число эпох, lr, batch_size, val_frac, early stopping, путь к чекпоинтам и т.д.

## Установка (Conda + pip)

- Python 3.9+
- Создай окружение и установи зависимости:

```bash
conda create -n vkr python=3.10 -y
conda activate vkr
cd D:\!NGU!\4_year_2026\VKR
pip install -r requirements.txt
```

В `requirements.txt`: torch, pytorch-lightning, tensorboard, PyYAML, pandas, numpy, folium, matplotlib, markupsafe (для ноутбука с картой). Если при `import folium` ошибка про Markup/markupsafe — выполни: `pip install "markupsafe>=2.0,<2.1"`.

## Аномалии

После получения предсказаний модели: `src.anomaly.flag_anomalies(predicted_eta_scaled, actual_eta_scaled, threshold_sec=120)` — флаг аномалии при превышении порога по умолчанию 120 сек.

## Конфигурация

Основные константы в `src/config.py`:

- Препроцессинг: `BEARING_TOLERANCE_DEG`, `SPEED_DEVIATION_KMH`, `RESAMPLE_INTERVAL_SEC`, `STOP_DISTANCE_THRESHOLD_M`.
- LSTM: `WINDOW_SIZE_SEC`, `LSTM_HIDDEN_SIZE`, `LSTM_NUM_LAYERS`, `ETA_SCALE_SEC`, `ANOMALY_ETA_THRESHOLD_SEC`.

Подробности по шагам предобработки и логике bearing — в `TODO.txt`.
