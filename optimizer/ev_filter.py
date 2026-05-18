from __future__ import annotations

from interfaces.contracts import CalibratedPrediction, ParlayLeg


_OUTCOME_FIELDS: list[tuple[str, str, str, str, str]] = [
    ("home", "odds_home", "p_home", "ev_home", "edge_home"),
    ("draw", "odds_draw", "p_draw", "ev_draw", "edge_draw"),
    ("away", "odds_away", "p_away", "ev_away", "edge_away"),
]


def filter_positive_ev(
    predictions: list[CalibratedPrediction],
    *,
    safety_margin: float = 1.05,
) -> list[ParlayLeg]:
    """从校准后预测中筛选正期望候选，按 EV 降序返回。

    过滤规则：
    - exclude_flag == False（赔率无高危异常）
    - p_model × odds >= safety_margin（含安全边际，默认 1.05）
    - 同一场比赛保留 EV 最高的单一投注方向

    返回的列表中每个 match_id 唯一，满足串场"一场一注"约束。
    """
    if safety_margin <= 0:
        raise ValueError("safety_margin 必须大于 0")

    best_per_match: dict[int, ParlayLeg] = {}

    for pred in predictions:
        if pred["exclude_flag"]:
            continue
        for outcome, odds_key, p_key, ev_key, edge_key in _OUTCOME_FIELDS:
            ev = float(pred[ev_key])
            if ev < safety_margin:
                continue
            match_id = int(pred["match_id"])
            leg: ParlayLeg = {
                "match_id": match_id,
                "outcome": outcome,
                "odds": float(pred[odds_key]),
                "p_model": float(pred[p_key]),
                "ev": ev,
                "edge": float(pred[edge_key]),
            }
            existing = best_per_match.get(match_id)
            if existing is None or ev > existing["ev"]:
                best_per_match[match_id] = leg

    return sorted(best_per_match.values(), key=lambda x: x["ev"], reverse=True)
