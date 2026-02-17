# -*- coding: utf-8 -*-
"""
Маппинг GPS-точек на маршрут (trass): проекция на линию пути с сохранением порядка движения.
Траектория не должна "прыгать" туда-сюда — каждая следующая точка проецируется на маршрут
вперёд от предыдущей проекции.
route_frac — доля пути по длине маршрута [0..1], для интерполяции вдоль полилайна (без срезов углов).
"""
import math
from typing import List, Tuple

# Приближение: метры на градус (широта/долгота в районе Новосибирска)
M_PER_DEG_LAT = 111320
M_PER_DEG_LON = 111320 * 0.72  # cos(55°)


def _haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Расстояние между двумя точками в метрах (приближённо)."""
    dlat = (lat2 - lat1) * M_PER_DEG_LAT
    dlon = (lon2 - lon1) * M_PER_DEG_LON
    return math.sqrt(dlat * dlat + dlon * dlon)


def _cumulative_distances(route_points: List[Tuple[float, float]]) -> Tuple[List[float], float]:
    """
    Кумулятивные расстояния от начала маршрута до каждой вершины (в метрах).
    Возвращает (cumdist, total_length). cumdist[0]=0, cumdist[i] = путь до route_points[i].
    """
    if not route_points or len(route_points) < 2:
        return [0.0], 0.0
    cumdist = [0.0]
    for i in range(len(route_points) - 1):
        d = _haversine_m(
            route_points[i][0], route_points[i][1],
            route_points[i + 1][0], route_points[i + 1][1],
        )
        cumdist.append(cumdist[-1] + d)
    return cumdist, cumdist[-1]


def _project_on_segment(
    px: float, py: float,
    ax: float, ay: float,
    bx: float, by: float
) -> Tuple[float, float, float]:
    """
    Проекция точки (px, py) на отрезок (ax,ay)-(bx,by).
    Возвращает (lat_proj, lon_proj, t) где t in [0, 1] — доля от начала отрезка.
    """
    dx = bx - ax
    dy = by - ay
    d2 = dx * dx + dy * dy
    if d2 < 1e-20:
        return ax, ay, 0.0
    t = ((px - ax) * dx + (py - ay) * dy) / d2
    t = max(0.0, min(1.0, t))
    return ax + t * dx, ay + t * dy, t


def map_gps_to_route(
    lats: List[float], lons: List[float],
    route_points: List[Tuple[float, float]]
) -> Tuple[List[float], List[float], List[float]]:
    """
    Смаппить последовательность GPS-точек на маршрут.
    Возвращает (lat_mapped, lon_mapped, route_frac) — координаты на маршруте и доля пути по длине [0..1].
    Ограничение: следующая точка ищется только вперёд по маршруту от предыдущей (без прыжков).
    """
    if not route_points or not lats:
        return list(lats), list(lons), []

    n_route = len(route_points)
    route_lats = [p[0] for p in route_points]
    route_lons = [p[1] for p in route_points]
    cumdist, total_length = _cumulative_distances(route_points)

    out_lat, out_lon, out_frac = [], [], []
    seg_start = 0

    for lat, lon in zip(lats, lons):
        best_dist = 1e30
        best_lat, best_lon = lat, lon
        best_frac = 0.0
        best_next_seg = 0

        for s in range(seg_start, n_route - 1):
            ax, ay = route_lats[s], route_lons[s]
            bx, by = route_lats[s + 1], route_lons[s + 1]
            pl, pn, t = _project_on_segment(lon, lat, ay, ax, by, bx)  # (x,y) = (lon, lat)
            proj_lat, proj_lon = pn, pl
            dist = _haversine_m(lat, lon, proj_lat, proj_lon)
            # Доля пути по длине маршрута (для интерполяции вдоль полилайна)
            seg_len = cumdist[s + 1] - cumdist[s]
            path_dist = cumdist[s] + t * seg_len if seg_len > 0 else cumdist[s]
            seg_frac = path_dist / total_length if total_length > 0 else (s + t) / max(1, n_route - 1)
            seg_frac = max(0.0, min(1.0, seg_frac))
            if dist < best_dist:
                best_dist = dist
                best_lat, best_lon = proj_lat, proj_lon
                best_frac = seg_frac
                best_next_seg = s if t < 0.5 else s + 1
        if best_dist < 1e30:
            out_lat.append(best_lat)
            out_lon.append(best_lon)
            out_frac.append(best_frac)
            seg_start = max(0, best_next_seg - 1)
        else:
            out_lat.append(lat)
            out_lon.append(lon)
            out_frac.append(0.0)

    return out_lat, out_lon, out_frac


def route_frac_to_point(route_frac: float, route_points: List[Tuple[float, float]]) -> Tuple[float, float]:
    """
    Точка на маршруте по доле пути [0..1]. Интерполяция вдоль полилайна — без срезов углов.
    """
    if not route_points:
        return 0.0, 0.0
    if len(route_points) == 1:
        return route_points[0][0], route_points[0][1]
    cumdist, total_length = _cumulative_distances(route_points)
    if total_length <= 0:
        return route_points[0][0], route_points[0][1]
    target_dist = max(0.0, min(1.0, route_frac)) * total_length
    # Найти сегмент: cumdist[s] <= target_dist < cumdist[s+1]
    s = 0
    for i in range(len(cumdist) - 1):
        if target_dist <= cumdist[i + 1]:
            s = i
            break
        s = i
    seg_len = cumdist[s + 1] - cumdist[s]
    t = (target_dist - cumdist[s]) / seg_len if seg_len > 0 else 0.0
    t = max(0.0, min(1.0, t))
    lat = route_points[s][0] + t * (route_points[s + 1][0] - route_points[s][0])
    lon = route_points[s][1] + t * (route_points[s + 1][1] - route_points[s][1])
    return lat, lon


def route_bearing_at_frac(route_frac: float, route_points: List[Tuple[float, float]]) -> float:
    """Направление маршрута (градусы 0=север, 90=восток) в точке с долей пути route_frac."""
    if len(route_points) < 2 or route_frac >= 1.0:
        return 0.0
    cumdist, total_length = _cumulative_distances(route_points)
    if total_length <= 0:
        return 0.0
    target_dist = max(0.0, min(1.0, route_frac)) * total_length
    s = 0
    for i in range(len(cumdist) - 1):
        if target_dist <= cumdist[i + 1]:
            s = i
            break
        s = i
    a, b = route_points[s], route_points[s + 1]
    dlon = (b[1] - a[1]) * M_PER_DEG_LON
    dlat = (b[0] - a[0]) * M_PER_DEG_LAT
    angle_rad = math.atan2(dlon, dlat)
    return math.degrees(angle_rad) % 360
