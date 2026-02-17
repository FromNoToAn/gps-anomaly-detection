# -*- coding: utf-8 -*-
"""
Оценка аномалий по порогу отклонения ETA: если |predicted_ETA - actual_ETA| > threshold,
считаем момент аномалией (например, неожиданная задержка или опережение).
"""
from typing import List, Tuple

from src.config import ANOMALY_ETA_THRESHOLD_SEC, ETA_SCALE_SEC


def flag_anomalies(
    predicted_eta_scaled: List[float],
    actual_eta_scaled: List[float],
    threshold_sec: float = ANOMALY_ETA_THRESHOLD_SEC,
) -> List[Tuple[int, float, float]]:
    """
    Возвращает список (index, pred_sec, actual_sec) для точек, где отклонение > threshold_sec.
    """
    anomalies = []
    for i, (p, a) in enumerate(zip(predicted_eta_scaled, actual_eta_scaled)):
        pred_sec = p * ETA_SCALE_SEC
        actual_sec = a * ETA_SCALE_SEC
        if abs(pred_sec - actual_sec) > threshold_sec:
            anomalies.append((i, pred_sec, actual_sec))
    return anomalies
