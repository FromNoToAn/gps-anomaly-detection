# -*- coding: utf-8 -*-
"""
Предобработка одной поездки: маппинг на маршрут, исправление bearing и speed,
интерполяция на сетку 1 сек. Вход: time, lat, lon, accuracy, bearing, speed.
Выход: те же поля + trip_id, vehicle_type, route_direction; точки каждую 1 сек.
"""
import math
import re
import numpy as np
import pandas as pd
from typing import Optional, Tuple

from src.config import (
    BEARING_TOLERANCE_DEG,
    SPEED_DEVIATION_KMH,
    RESAMPLE_INTERVAL_SEC,
    VEHICLE_TYPE,
)
from src.preprocessing.trass_loader import get_route_points_and_stops
from src.preprocessing.map_to_route import (
    map_gps_to_route,
    route_frac_to_point,
    route_bearing_at_frac,
)

M_PER_DEG_LAT = 111320
M_PER_DEG_LON = 111320 * 0.72


def _parse_filename(filename: str) -> Tuple[str, str, str]:
    """Из имени файла 20260204_38_rev.csv извлекаем date, route, direction."""
    base = re.sub(r"\.csv$", "", filename, flags=re.I)
    parts = base.split("_")
    if len(parts) < 3:
        return base, "", "str"
    date_part = parts[0]
    route = parts[1]
    direction = parts[2].lower() if parts[2].lower() in ("str", "rev") else "str"
    return date_part, route, direction


def _normalize_angle_diff(deg: float) -> float:
    """Разница углов в [-180, 180]."""
    while deg > 180:
        deg -= 360
    while deg < -180:
        deg += 360
    return deg


def _speed_from_points(lat1: float, lon1: float, lat2: float, lon2: float, dt_sec: float) -> float:
    """Скорость км/ч по двум точкам и времени в секундах."""
    if dt_sec <= 0:
        return 0.0
    d = math.sqrt(
        ((lat2 - lat1) * M_PER_DEG_LAT) ** 2 + ((lon2 - lon1) * M_PER_DEG_LON) ** 2
    )
    return (d / 1000.0) / (dt_sec / 3600.0)


def preprocess_trip(
    df: pd.DataFrame,
    route_number: str,
    direction: str,
    trip_id: str,
) -> pd.DataFrame:
    """
    Полная предобработка поездки:
    1) Маппинг lat, lon на маршрут
    2) Bearing: если отклонение от направления маршрута > ±20°, заменяем по правилу из TODO
    3) Speed: если отклонение от расчётной скорости > 7 км/ч — заменяем на расчётную
    4) Ресэмплинг на 1 сек (интерполяция)
    5) Добавляем trip_id, vehicle_type, route_direction
    """
    df = df.dropna(subset=["lat", "lon"]).copy()
    if df.empty:
        return df
    df["time"] = pd.to_datetime(df["time"], utc=True)
    df = df.sort_values("time").reset_index(drop=True)

    route_points, _ = get_route_points_and_stops(route_number, direction)
    if not route_points:
        return df

    lat_m, lon_m, route_frac = map_gps_to_route(
        df["lat"].tolist(), df["lon"].tolist(), route_points
    )
    df["lat"] = lat_m
    df["lon"] = lon_m
    df["route_frac"] = route_frac

    # Bearing (TODO): проверка ±20° от направления маршрута в каждой точке;
    # при выходе — среднее между bearing прошлой точки и случайным углом в ±20° от маршрута.
    route_bearing = [route_bearing_at_frac(f, route_points) for f in route_frac]
    prev_bearing = None
    bearings_new = []
    for i, row in df.iterrows():
        b = row.get("bearing")
        if pd.isna(b) or b == "":
            b = None
        else:
            try:
                b = float(b)
            except (TypeError, ValueError):
                b = None
        ref = route_bearing[i] if i < len(route_bearing) else 0.0
        if b is None:
            if prev_bearing is not None:
                b = prev_bearing
            else:
                b = ref
        diff = _normalize_angle_diff(b - ref)
        if abs(diff) > BEARING_TOLERANCE_DEG:
            # Прошлая точка (или 0 для первой); случайный угол в ±20° от направления маршрута; среднее.
            if prev_bearing is not None:
                prev = prev_bearing
            else:
                prev = ref  # первая точка: условно "прошлая" = направление маршрута
            np.random.seed(i % (2 ** 32))
            random_offset = np.random.uniform(-BEARING_TOLERANCE_DEG, BEARING_TOLERANCE_DEG)
            new_angle = (ref + random_offset) % 360
            # Среднее двух углов с учётом оборота
            b = (prev + new_angle) / 2.0
            if abs(prev - new_angle) > 180:
                b = (b + 180) % 360
            b = b % 360
        else:
            b = (b + 360) % 360
        prev_bearing = b
        bearings_new.append(b)
    df["bearing"] = bearings_new

    # Speed: проверка по соседним точкам
    speeds = df["speed"].fillna(0.0).astype(float).tolist()
    times = df["time"].values
    lats = df["lat"].tolist()
    lons = df["lon"].tolist()
    for i in range(1, len(speeds)):
        dt = (pd.Timestamp(times[i]) - pd.Timestamp(times[i - 1])).total_seconds()
        if dt <= 0:
            continue
        comp = _speed_from_points(lats[i - 1], lons[i - 1], lats[i], lons[i], dt)
        if abs(speeds[i] - comp) > SPEED_DEVIATION_KMH:
            speeds[i] = comp
    df["speed"] = speeds

    # Ресэмплинг на 1 сек: интерполируем по времени route_frac, accuracy, speed (bearing — не интерполируем).
    # lat, lon — строго по полилайну из route_frac.
    # Bearing берём из направления маршрута в каждой точке, чтобы повороты в trass были порезче, без сглаживания углов.
    t0 = df["time"].min()
    t_end = df["time"].max()
    new_index = pd.date_range(start=t0, end=t_end, freq=f"{RESAMPLE_INTERVAL_SEC}s", tz="UTC")
    if len(new_index) == 0:
        new_index = pd.DatetimeIndex([t0], tz="UTC")
    out = pd.DataFrame({"time": new_index})
    for col in ["accuracy", "speed", "route_frac"]:
        if col in df.columns:
            out[col] = np.interp(
                out["time"].astype(np.int64),
                df["time"].astype(np.int64),
                df[col].values,
            )
    # Точки на маршруте строго по полилайну
    out["lat"] = [route_frac_to_point(f, route_points)[0] for f in out["route_frac"]]
    out["lon"] = [route_frac_to_point(f, route_points)[1] for f in out["route_frac"]]
    # Направление — по геометрии маршрута в этой точке (поворот в trass = резкое изменение bearing)
    out["bearing"] = [route_bearing_at_frac(f, route_points) for f in out["route_frac"]]
    if "accuracy" in out.columns:
        out["accuracy"] = out["accuracy"].clip(lower=0)
    out["bearing"] = out["bearing"] % 360
    out["speed"] = out["speed"].clip(lower=0)
    out["trip_id"] = trip_id
    out["vehicle_type"] = VEHICLE_TYPE
    out["route_direction"] = f"{route_number}_{direction}"
    return out
