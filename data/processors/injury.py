from __future__ import annotations

from typing import TypedDict


class MissingPlayer(TypedDict):
    importance: float
    position_multiplier: float


def injury_impact(missing_players: list[MissingPlayer]) -> float:
    """计算伤病减损，返回范围 [0.0, -0.30]。"""
    raw = sum(p["importance"] * p["position_multiplier"] for p in missing_players)
    return -min(raw * 0.15, 0.30)
