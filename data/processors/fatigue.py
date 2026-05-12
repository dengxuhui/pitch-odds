from __future__ import annotations


def fatigue_index(matches_last_30d: int, travel_km: float, minutes_played_key_players: float) -> float:
    """计算疲劳指数，范围 [0.0, 1.0]。"""
    base = min(matches_last_30d / 10.0, 1.0)
    travel = min(travel_km / 5000.0, 0.3)
    load = min(minutes_played_key_players / 900.0, 0.3)
    return min(base + travel + load, 1.0)
