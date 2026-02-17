# -*- coding: utf-8 -*-
"""Загрузка маршрутов (trasses) из JSON: геометрия пути и остановки (id, n)."""
import json
import os
from typing import List, Tuple, Optional

from src.config import TRASSES_DIR


def load_trass(route_number: str) -> List[dict]:
    """
    Загружает маршрут по номеру (38, 52, 72).
    Возвращает список точек маршрута (каждая: lat, lng, и опционально id, n для остановок).
    """
    path = os.path.join(TRASSES_DIR, f"trasses_{route_number}.json")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Trass not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    # Структура: trasses[0].r[0].u = массив точек
    trasses = data.get("trasses", [])
    if not trasses or not trasses[0].get("r"):
        return []
    points = trasses[0]["r"][0].get("u", [])
    return points


def get_route_points_and_stops(
    route_number: str, direction: str
) -> Tuple[List[Tuple[float, float]], List[Tuple[int, Tuple[float, float], Optional[str], str]]]:
    """
    Возвращает:
    - route_points: список (lat, lon) точек маршрута в порядке движения (с учётом direction).
    - stops: список (index, (lat, lon), name, stop_id) для каждой остановки.
      stop_id — id из trass (строка, напр. "603"); при отсутствии id берётся n или пустая строка.
    direction: 'str' — прямое, 'rev' — обратное (разворачиваем порядок точек).
    """
    raw = load_trass(route_number)
    if not raw:
        return [], []

    route_points = []
    stops = []
    for i, p in enumerate(raw):
        lat = float(p.get("lat", p.get("latitude", 0)))
        lng = float(p.get("lng", p.get("lon", p.get("longitude", 0))))
        route_points.append((lat, lng))
        if p.get("id") or p.get("n"):
            name = p.get("n") or p.get("id") or ""
            stop_id = str(p.get("id")) if p.get("id") is not None else (str(p.get("n")) if p.get("n") else "")
            stops.append((i, (lat, lng), name, stop_id))

    if direction == "rev":
        route_points = list(reversed(route_points))
        n = len(route_points)
        stops = [(n - 1 - idx, (lat, lng), name, stop_id) for (idx, (lat, lng), name, stop_id) in reversed(stops)]

    return route_points, stops
