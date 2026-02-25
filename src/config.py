# -*- coding: utf-8 -*-
"""Конфигурация путей и констант для препроцессинга и LSTM бейзлайна."""
import os

# Корень проекта (родитель папки src)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Датасеты
DATASETS_DIR = os.path.join(PROJECT_ROOT, "datasets")
RAW_DIR = os.path.join(DATASETS_DIR, "row_datasets", "Novosibirsk")
TRASSES_DIR = os.path.join(DATASETS_DIR, "trasses")
PRE_DATASETS_DIR = os.path.join(DATASETS_DIR, "pre_datasets", "Novosibirsk")

# Параметры препроцессинга (из TODO)
BEARING_TOLERANCE_DEG = 20       # ±20° от направления маршрута
SPEED_DEVIATION_KMH = 7.0        # замена speed при отклонении > 7 км/ч от расчётного
RESAMPLE_INTERVAL_SEC = 1        # интерполяция на сетку 1 сек
STOP_DISTANCE_THRESHOLD_M = 25   # порог расстояния до остановки для ETA (метры)
VEHICLE_TYPE = "bus"

# Воспроизводимость обучения
SEED = 42

# LSTM
WINDOW_SIZE_SEC = 60             # размер скользящего окна (секунды = число шагов при 1 сек)
LSTM_HIDDEN_SIZE = 64
LSTM_NUM_LAYERS = 2
ETA_SCALE_SEC = 600.0            # масштаб целевой переменной (сек), для стабильности обучения
ANOMALY_ETA_THRESHOLD_SEC = 120  # порог аномалии: отклонение ETA (сек)
