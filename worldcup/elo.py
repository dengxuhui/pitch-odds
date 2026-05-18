from __future__ import annotations

from collections import defaultdict
from typing import Any


# FIFA/ClubElo 常用 K 因子（世界杯 > 资格赛 > 友谊赛）
K_FACTORS: dict[str, float] = {
    "world_cup":   60.0,
    "qualifier":   40.0,
    "friendly":    20.0,
    "continental": 50.0,
}

_DEFAULT_K = 40.0
_DEFAULT_INITIAL = 1500.0
_DEFAULT_HOME_ADV = 100.0  # 主场优势折算 Elo 点数（中立场地为 0）


class EloRating:
    """FIFA 风格 Elo 评分引擎，面向国家队。

    评分更新公式（标准 Elo）：
        R' = R + K * G * (S - E)
    其中：
        K  = 比赛类型权重
        G  = 进球差倍率（|goal_diff| >= 3 时放大）
        S  = 实际得分（胜1 / 平0.5 / 负0）
        E  = 期望得分 1 / (1 + 10^(-(R_a - R_b) / 400))

    中立场地不计主场优势；若 neutral=False 则为 home 队加 home_advantage 点。
    """

    def __init__(
        self,
        initial_rating: float = _DEFAULT_INITIAL,
        k_factor: float = _DEFAULT_K,
        home_advantage: float = _DEFAULT_HOME_ADV,
    ) -> None:
        self.initial_rating = initial_rating
        self.k_factor = k_factor
        self.home_advantage = home_advantage
        self._ratings: dict[int, float] = defaultdict(lambda: initial_rating)

    # ──────────────────────────────────────────────
    # 公开接口
    # ──────────────────────────────────────────────

    def expected_score(self, team_a: int, team_b: int, *, neutral: bool = True) -> float:
        """返回 team_a 对阵 team_b 的期望得分（胜=1, 平=0.5, 负=0 的加权均值）。"""
        diff = self._ratings[team_a] - self._ratings[team_b]
        if not neutral:
            diff += self.home_advantage
        return 1.0 / (1.0 + 10.0 ** (-diff / 400.0))

    def update(
        self,
        home_team: int,
        away_team: int,
        home_goals: int,
        away_goals: int,
        *,
        neutral: bool = True,
        game_type: str = "world_cup",
    ) -> tuple[float, float]:
        """更新双方评分，返回 (delta_home, delta_away)。

        Args:
            home_team:   主队 ID（中立场地仅作标识）。
            away_team:   客队 ID。
            home_goals:  主队进球数。
            away_goals:  客队进球数。
            neutral:     是否中立场地（世界杯一般为 True）。
            game_type:   比赛类型键，对应 K_FACTORS。

        Returns:
            (Δhome, Δaway) — 本场对各自评分的变化量。
        """
        k = K_FACTORS.get(game_type, self.k_factor)
        goal_diff = abs(home_goals - away_goals)
        g = _goal_multiplier(goal_diff)

        if home_goals > away_goals:
            s_home = 1.0
        elif home_goals == away_goals:
            s_home = 0.5
        else:
            s_home = 0.0
        s_away = 1.0 - s_home

        e_home = self.expected_score(home_team, away_team, neutral=neutral)
        e_away = 1.0 - e_home

        delta_home = k * g * (s_home - e_home)
        delta_away = k * g * (s_away - e_away)

        self._ratings[home_team] += delta_home
        self._ratings[away_team] += delta_away
        return delta_home, delta_away

    def get_rating(self, team_id: int) -> float:
        return self._ratings[team_id]

    def set_rating(self, team_id: int, rating: float) -> None:
        self._ratings[team_id] = rating

    def to_dict(self) -> dict[int, float]:
        return dict(self._ratings)

    def from_dict(self, ratings: dict[Any, float]) -> None:
        for k, v in ratings.items():
            self._ratings[int(k)] = float(v)

    def copy(self) -> "EloRating":
        """返回当前评分状态的深拷贝（用于回测快照）。"""
        clone = EloRating(self.initial_rating, self.k_factor, self.home_advantage)
        clone.from_dict(self._ratings)
        return clone


# ──────────────────────────────────────────────
# 内部辅助
# ──────────────────────────────────────────────

def _goal_multiplier(goal_diff: int) -> float:
    """进球差倍率（来自 FIFA Elo 变体）。

    0–1 球差：1.0
    2   球差：1.5
    3+  球差：(11 + goal_diff) / 8
    """
    if goal_diff <= 1:
        return 1.0
    if goal_diff == 2:
        return 1.5
    return (11.0 + goal_diff) / 8.0
