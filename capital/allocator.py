from __future__ import annotations

import uuid

from interfaces.contracts import BetRecord, ParlayPlan, validate_bet_record
from capital.kelly import half_kelly


_TIER_BUDGET_FRACS: dict[str, float] = {
    "hedge":      0.40,
    "core":       0.40,
    "aggressive": 0.20,
}


def allocate_capital(
    plan: ParlayPlan,
    total_capital: float,
    *,
    is_simulation: bool = False,
) -> list[BetRecord]:
    """按三层预算比例与 Half Kelly 为串场方案分配注金。

    每层的最大注金 = total_capital × 层级预算比例。
    实际注金 = min(half_kelly × total_capital, 层级上限)。

    Args:
        plan:          已通过 validate_parlay_plan 的串场方案。
        total_capital: 当前总资本（用于计算注金绝对值）。
        is_simulation: 是否为回测模拟，写入 BetRecord.is_simulation。

    Returns:
        每个 ParlayOption 对应一条 BetRecord，stake 已确定。

    Raises:
        ValueError: total_capital <= 0。
    """
    if total_capital <= 0:
        raise ValueError("total_capital 必须大于 0")

    plan_id = str(uuid.uuid4())
    records: list[BetRecord] = []

    for option in plan["options"]:
        tier = option["tier"]
        budget_frac = _TIER_BUDGET_FRACS.get(tier, 1.0 / 3.0)
        tier_cap = total_capital * budget_frac

        kelly_pct = half_kelly(option["win_rate"], option["total_odds"])
        raw_stake = kelly_pct * total_capital
        stake = min(raw_stake, tier_cap)

        record: BetRecord = {
            "plan_id": plan_id,
            "plan_date": plan["plan_date"],
            "tier": tier,
            "legs": option["legs"],
            "total_odds": option["total_odds"],
            "win_rate": option["win_rate"],
            "stake": round(stake, 2),
            "kelly_pct": round(kelly_pct, 6),
            "is_simulation": is_simulation,
            "won": None,
            "payout": None,
            "profit": None,
            "settled_at": None,
        }
        validate_bet_record(record)
        records.append(record)

    return records
