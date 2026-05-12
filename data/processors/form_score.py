from __future__ import annotations

import math
from typing import Iterable


def form_score(results: Iterable[int], days_ago: Iterable[int], lambda_decay: float = 0.05) -> float:
    """计算近期战绩分数并标准化到 [-1.0, 1.0]。"""
    results_list = list(results)
    days_list = list(days_ago)
    if len(results_list) != len(days_list):
        raise ValueError("results 与 days_ago 长度必须一致")
    if not results_list:
        return 0.0

    weights = [math.exp(-lambda_decay * d) for d in days_list]
    max_possible = 3.0 * sum(weights)
    raw = sum(float(r) * w for r, w in zip(results_list, weights, strict=True))
    return (raw / max_possible) * 2.0 - 1.0
