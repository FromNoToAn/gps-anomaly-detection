# -*- coding: utf-8 -*-
"""
Сегментация поездки по остановкам и целевая переменная ETA (время до следующей остановки).
Сегмент и следующая остановка определяются по положению вдоль маршрута (route_frac),
а не по ближайшей остановке по расстоянию — чтобы на закольцованных маршрутах не прыгать
на «ту же» остановку с обратной стороны кольца.
"""
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd

from src.preprocessing.map_to_route import _cumulative_distances
from src.preprocessing.trass_loader import get_route_points_and_stops


def _stop_fracs(
    route_points: List[Tuple[float, float]],
    stops: List[Tuple],  # (index, (lat, lon), name, stop_id) или (index, (lat, lon), name)
) -> List[float]:
    """
    Доля пути (route_frac) для каждой остановки: cumdist[vertex] / total_length.
    Позволяет определять сегмент по положению вдоль маршрута.
    """
    if not route_points or not stops:
        return []
    cumdist, total_length = _cumulative_distances(route_points)
    if total_length <= 0:
        return [0.0] * len(stops)
    return [cumdist[min(s[0], len(cumdist) - 1)] / total_length for s in stops]


def segment_by_stops(
    df: pd.DataFrame,
    route_number: str,
    direction: str,
) -> pd.DataFrame:
    """
    Добавляет current_stop_id и next_stop_id (id из trass) по положению вдоль маршрута (route_frac).
    Точка в сегменте s: stop_frac[s] <= route_frac < stop_frac[s+1];
    current_stop_id = id остановки s, next_stop_id = id остановки s+1 (или "" в конце маршрута).
    """
    route_points, stops = get_route_points_and_stops(route_number, direction)
    if not stops:
        df = df.copy()
        df["current_stop_id"] = ""
        df["next_stop_id"] = ""
        return df

    if "route_frac" not in df.columns:
        df = df.copy()
        df["current_stop_id"] = ""
        df["next_stop_id"] = ""
        return df

    stop_frac = _stop_fracs(route_points, stops)
    n_stops = len(stops)
    # stops[k] = (index, (lat, lon), name, stop_id)
    get_stop_id = lambda k: stops[k][3] if len(stops[k]) > 3 else str(k)
    route_frac = np.asarray(df["route_frac"], dtype=float)
    n = len(df)
    current_stop_id = []
    next_stop_id = []

    for i in range(n):
        f = route_frac[i]
        if np.isnan(f):
            f = 0.0
        f = max(0.0, min(1.0, f))
        seg = 0
        for s in range(n_stops - 1):
            if stop_frac[s] <= f < stop_frac[s + 1]:
                seg = s
                break
            if f >= stop_frac[s + 1]:
                seg = s + 1
        if f >= stop_frac[-1]:
            seg = n_stops - 1
        current_stop_id.append(get_stop_id(seg))
        next_stop_id.append(get_stop_id(seg + 1) if seg + 1 < n_stops else "")

    df = df.copy()
    df["current_stop_id"] = current_stop_id
    df["next_stop_id"] = next_stop_id
    return df


def compute_eta_targets(
    df: pd.DataFrame,
    route_number: str,
    direction: str,
) -> pd.DataFrame:
    """
    Целевая переменная ETA: время в секундах до прибытия на следующую остановку.
    Прибытие = первый момент, когда route_frac достигает позиции следующей остановки.
    Индекс следующей остановки вычисляется по route_frac (как в segment_by_stops).
    """
    route_points, stops = get_route_points_and_stops(route_number, direction)
    if not stops:
        df = df.copy()
        df["eta_sec"] = np.nan
        return df

    stop_frac = _stop_fracs(route_points, stops)
    n_stops = len(stops)
    times = pd.to_datetime(df["time"])
    n = len(df)
    eta_sec = np.full(n, np.nan, dtype=float)

    if "route_frac" not in df.columns:
        df = df.copy()
        df["eta_sec"] = eta_sec
        return df

    route_frac = np.asarray(df["route_frac"], dtype=float)

    for i in range(n):
        f = route_frac[i]
        if np.isnan(f):
            f = 0.0
        f = max(0.0, min(1.0, f))
        seg = 0
        for s in range(n_stops - 1):
            if stop_frac[s] <= f < stop_frac[s + 1]:
                seg = s
                break
            if f >= stop_frac[s + 1]:
                seg = s + 1
        if f >= stop_frac[-1]:
            seg = n_stops - 1
        next_idx = seg + 1
        if next_idx >= n_stops:
            continue
        target_frac = stop_frac[next_idx]
        t_i = times[i]
        for j in range(i, n):
            fj = route_frac[j]
            if np.isnan(fj):
                continue
            if fj >= target_frac:
                eta_sec[i] = (times[j] - t_i).total_seconds()
                if eta_sec[i] < 0:
                    eta_sec[i] = 0
                break

    df = df.copy()
    df["eta_sec"] = eta_sec
    return df
