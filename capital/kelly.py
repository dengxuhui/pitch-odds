from __future__ import annotations


def half_kelly(p: float, odds: float, *, fraction: float = 0.5) -> float:
    """计算 Half Kelly 注金比例（相对于总资本）。

    Kelly 公式：f = (b*p - q) / b，其中 b = odds - 1，q = 1 - p。
    Half Kelly 将 f 乘以 fraction（默认 0.5）以降低方差。

    Args:
        p:        模型预测的获胜概率。
        odds:     欧洲赔率（含本金返还）。
        fraction: Kelly 缩放系数，默认 0.5（Half Kelly）。

    Returns:
        注金占总资本的比例，最小为 0（负期望值时不投注）。

    Raises:
        ValueError: odds <= 1 或 p 不在 [0, 1] 范围内。
    """
    if odds <= 1.0:
        raise ValueError(f"赔率必须大于 1.0，当前值: {odds}")
    if not (0.0 <= p <= 1.0):
        raise ValueError(f"概率必须在 [0, 1] 范围内，当前值: {p}")
    if not (0.0 < fraction <= 1.0):
        raise ValueError(f"fraction 必须在 (0, 1] 范围内，当前值: {fraction}")

    b = odds - 1.0
    kelly = (b * p - (1.0 - p)) / b
    return max(0.0, kelly * fraction)
