from __future__ import annotations


def momentum_score(win_streak: int, loss_streak: int, big_loss_flag: bool) -> float:
    """计算心理状态系数，范围 [-0.10, +0.10]。

    win_streak: 当前连胜场次
    loss_streak: 当前连败场次
    big_loss_flag: 最近一场是否大败（失球差 >= 3）
    """
    score = min(win_streak * 0.025, 0.10)
    score -= min(loss_streak * 0.02, 0.08)
    if big_loss_flag:
        score -= 0.05
    return max(-0.10, min(0.10, score))
