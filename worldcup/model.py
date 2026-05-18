from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional

from interfaces.contracts import MatchFeatures, ModelRawOutput
from models.base import PredictionModel
from worldcup.club_form import ClubFormMapper
from worldcup.elo import EloRating


# 赔率概率转换标准参数
# 通过对 WC2014-2022 小组赛+淘汰赛回归校验得出的经验值
_DRAW_BASE = 0.26     # 平局基础概率（Elo 差为 0 时）
_DRAW_SLOPE = 0.12    # 每 400 Elo 点差降低的平局概率

# 俱乐部状态 → Elo 调整映射系数
# momentum [-0.10, +0.10] × 500 → [-50, +50] Elo 点
_MOMENTUM_ELO_SCALE = 500.0


class WorldCupModel(PredictionModel):
    """基于 Elo 评分的世界杯 1X2 概率预测模型。

    与联赛 Dixon-Coles 模型完全独立：
    - 参数：国家队 Elo 评分（不含 attack/defense 向量）
    - 训练：按时间顺序逐场更新 Elo
    - 预测：Elo 差 → 1X2 概率（解析公式，无 MLE）

    概率转换公式：
        d     = elo_home - elo_away
        e_h   = 1 / (1 + 10^(−d/400))          # Bradley-Terry 期望胜率
        p_d   = max(0.05, DRAW_BASE − DRAW_SLOPE × |d| / 400)
        p_h   = e_h × (1 − p_d)
        p_a   = (1 − e_h) × (1 − p_d)
        → [p_h, p_d, p_a] 数学上已归一（无需额外归一化）

    ClubFormMapper 可选：若设置，则在 predict() 时将 MatchFeatures 中的
    home_momentum / away_momentum 转为 Elo 微调（不改变已训练的基础评分）。
    """

    model_version: str = "world_cup_elo_v1"

    def __init__(
        self,
        initial_rating: float = 1500.0,
        k_factor: float = 60.0,
        home_advantage: float = 0.0,   # 世界杯默认中立场，主场优势设为 0
        club_form_mapper: Optional[ClubFormMapper] = None,
    ) -> None:
        self.elo = EloRating(initial_rating, k_factor, home_advantage)
        self.club_form_mapper = club_form_mapper
        self._league_id: str = "WC"

    # ──────────────────────────────────────────────
    # PredictionModel 接口
    # ──────────────────────────────────────────────

    def fit(self, matches: list[dict[str, Any]], league_id: str = "WC") -> None:
        """按时间顺序更新 Elo 评分。

        每条 match dict 必须包含：
            home_team_id, away_team_id, home_goals, away_goals, match_date
        可选：
            neutral (bool, default True), game_type (str, default "world_cup")
        """
        self._league_id = league_id
        sorted_matches = sorted(matches, key=lambda m: str(m["match_date"]))
        for m in sorted_matches:
            if m.get("home_goals") is None or m.get("away_goals") is None:
                continue
            self.elo.update(
                int(m["home_team_id"]),
                int(m["away_team_id"]),
                int(m["home_goals"]),
                int(m["away_goals"]),
                neutral=bool(m.get("neutral", True)),
                game_type=str(m.get("game_type", "world_cup")),
            )

    def predict(self, features: MatchFeatures) -> ModelRawOutput:
        """将当前 Elo 差转换为 1X2 概率。

        home_momentum / away_momentum 字段（[-0.10, +0.10]）被视为俱乐部状态
        调整信号，等比缩放为 Elo 点修正后叠加到原始评分（不更改模型内部状态）。
        """
        home_id = int(features["home_team_id"])
        away_id = int(features["away_team_id"])

        elo_home = self.elo.get_rating(home_id)
        elo_away = self.elo.get_rating(away_id)

        # 俱乐部状态微调（来自 MatchFeatures.home_momentum / away_momentum）
        elo_home += float(features.get("home_momentum", 0.0)) * _MOMENTUM_ELO_SCALE
        elo_away += float(features.get("away_momentum", 0.0)) * _MOMENTUM_ELO_SCALE

        p_home, p_draw, p_away = _elo_to_probabilities(elo_home, elo_away)

        return {
            "match_id": int(features["match_id"]),
            "model_version": self.model_version,
            "predicted_at": datetime.now(timezone.utc).isoformat(),
            "p_home_raw": round(p_home, 6),
            "p_draw_raw": round(p_draw, 6),
            "p_away_raw": round(p_away, 6),
            "lambda_home": None,
            "lambda_away": None,
        }

    def get_params(self) -> dict[str, Any]:
        return {
            "model_version": self.model_version,
            "league_id": self._league_id,
            "initial_rating": self.elo.initial_rating,
            "k_factor": self.elo.k_factor,
            "home_advantage": self.elo.home_advantage,
            "ratings": self.elo.to_dict(),
        }

    def load_params(self, params: dict[str, Any]) -> None:
        self._league_id = params.get("league_id", "WC")
        self.elo.initial_rating = float(params.get("initial_rating", 1500.0))
        self.elo.k_factor = float(params.get("k_factor", 60.0))
        self.elo.home_advantage = float(params.get("home_advantage", 0.0))
        if "ratings" in params:
            self.elo.from_dict(params["ratings"])


# ──────────────────────────────────────────────
# 内部辅助
# ──────────────────────────────────────────────

def _elo_to_probabilities(elo_home: float, elo_away: float) -> tuple[float, float, float]:
    """将双方 Elo 评分转换为 (p_home, p_draw, p_away)。

    数学上三者之和精确为 1.0（无需归一化）：
        p_h + p_d + p_a
        = e_h(1-p_d) + p_d + (1-e_h)(1-p_d)
        = (1-p_d)(e_h + 1 - e_h) + p_d = 1
    """
    diff = elo_home - elo_away
    e_home = 1.0 / (1.0 + 10.0 ** (-diff / 400.0))
    p_draw = max(0.05, _DRAW_BASE - _DRAW_SLOPE * abs(diff) / 400.0)
    p_home = e_home * (1.0 - p_draw)
    p_away = (1.0 - e_home) * (1.0 - p_draw)
    return p_home, p_draw, p_away
