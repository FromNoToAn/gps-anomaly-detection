# -*- coding: utf-8 -*-
"""
Паipeline предобработки: для каждого CSV в row_datasets/Novosibirsk загружаем поездку,
препроцессим (маршрут, bearing, speed, ресэмпл), сегментируем по остановкам, считаем ETA,
сохраняем в pre_datasets/Novosibirsk.
"""
import os
from pathlib import Path

import pandas as pd

from src.config import RAW_DIR, PRE_DATASETS_DIR, TRASSES_DIR
from src.preprocessing.trass_loader import get_route_points_and_stops
from src.preprocessing.trip_preprocess import preprocess_trip, _parse_filename
from src.preprocessing.segment_stops import segment_by_stops, compute_eta_targets


def run_preprocessing(overwrite: bool = False) -> list:
    """
    Обрабатывает все CSV в RAW_DIR и сохраняет в PRE_DATASETS_DIR.
    Возвращает список путей к сохранённым файлам.
    """
    os.makedirs(PRE_DATASETS_DIR, exist_ok=True)
    csv_files = [f for f in os.listdir(RAW_DIR) if f.lower().endswith(".csv")]
    saved = []
    for filename in sorted(csv_files):
        date_part, route, direction = _parse_filename(filename)
        trass_path = os.path.join(TRASSES_DIR, f"trasses_{route}.json")
        if not os.path.isfile(trass_path):
            print(f"Skip {filename}: no trass for route {route}")
            continue
        out_name = Path(filename).stem + "_preprocessed.csv"
        out_path = os.path.join(PRE_DATASETS_DIR, out_name)
        if not overwrite and os.path.isfile(out_path):
            saved.append(out_path)
            continue
        df = pd.read_csv(os.path.join(RAW_DIR, filename))
        trip_id = f"{date_part}_{route}_{direction}"
        try:
            df = preprocess_trip(df, route, direction, trip_id)
        except Exception as e:
            print(f"Error preprocess {filename}: {e}")
            continue
        if df.empty:
            continue
        df = segment_by_stops(df, route, direction)
        df = compute_eta_targets(df, route, direction)
        df.to_csv(out_path, index=False)
        saved.append(out_path)
        print(f"Saved {out_path}")
    return saved
