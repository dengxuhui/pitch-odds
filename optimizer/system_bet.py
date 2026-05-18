from __future__ import annotations

from itertools import combinations

from interfaces.contracts import ParlayLeg


def system_bet(candidates: list[ParlayLeg], system_size: int) -> list[list[ParlayLeg]]:
    """生成系统投注组合（容错一场）。

    从 len(candidates) 场里取 system_size 场的所有组合。
    典型用法：5 串取 4（N 取 N-1），任意 4 场全中即盈利。

    Args:
        candidates:   串场候选列表，每个元素对应一场比赛的投注方向。
        system_size:  每注串场的腿数，通常为 len(candidates) - 1。

    Returns:
        所有 C(n, system_size) 种组合，每种组合是一个 ParlayLeg 列表。

    Raises:
        ValueError: system_size < 2 或 system_size >= len(candidates)。
    """
    n = len(candidates)
    if system_size < 2:
        raise ValueError("system_size 不能小于 2")
    if system_size >= n:
        raise ValueError(
            f"system_size ({system_size}) 必须小于候选数量 ({n})，"
            "系统投注至少需要保留一场容错空间"
        )

    return [list(combo) for combo in combinations(candidates, system_size)]


def system_bet_stats(combos: list[list[ParlayLeg]]) -> dict[str, float]:
    """计算系统投注的汇总统计（假设每注注金相等）。

    返回各组合的平均胜率、平均总赔率和平均期望值。
    """
    if not combos:
        raise ValueError("combos 不能为空")

    total_win_rate = 0.0
    total_odds = 0.0
    total_ev = 0.0

    for combo in combos:
        wr = 1.0
        odds = 1.0
        for leg in combo:
            wr *= leg["p_model"]
            odds *= leg["odds"]
        total_win_rate += wr
        total_odds += odds
        total_ev += wr * odds

    n = len(combos)
    return {
        "n_combos": n,
        "avg_win_rate": round(total_win_rate / n, 6),
        "avg_total_odds": round(total_odds / n, 4),
        "avg_expected_ev": round(total_ev / n, 4),
    }
