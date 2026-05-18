from __future__ import annotations

from itertools import combinations

from interfaces.contracts import ParlayLeg, ParlayOption, ParlayPlan, validate_parlay_plan


# 候选数上限：超过此值时组合数量可能过大，截取 EV 最高的前 N 个
_MAX_CANDIDATES_FOR_SEARCH = 20

# 三层预算比例
_TIER_BUDGETS = {"hedge": 0.40, "core": 0.40, "aggressive": 0.20}


def find_optimal_parlay(
    candidates: list[ParlayLeg],
    *,
    min_legs: int = 2,
    max_legs: int = 8,
    min_win_rate: float = 0.20,
) -> ParlayOption | None:
    """在最低胜率约束下，从候选中找期望收益最大的串场组合。

    约束：
    - 每场比赛在同一串场内最多出现一次（candidates 已由 filter_positive_ev 保证唯一）
    - 串场胜率（∏ p_model）>= min_win_rate
    - 腿数在 [min_legs, max_legs] 范围内

    当候选数超过 _MAX_CANDIDATES_FOR_SEARCH 时，只搜索 EV 最高的前 N 个，
    避免组合数爆炸（C(20,8) ≈ 12.6 万，仍可快速完成）。
    """
    if min_legs < 2:
        raise ValueError("min_legs 不能小于 2")
    if max_legs < min_legs:
        raise ValueError("max_legs 不能小于 min_legs")

    search_pool = candidates[:_MAX_CANDIDATES_FOR_SEARCH]
    best_ev = 0.0
    best: ParlayOption | None = None

    for n_legs in range(min_legs, min(max_legs, len(search_pool)) + 1):
        for combo in combinations(search_pool, n_legs):
            win_rate = 1.0
            total_odds = 1.0
            for leg in combo:
                win_rate *= leg["p_model"]
                total_odds *= leg["odds"]
            if win_rate < min_win_rate:
                continue
            expected_ev = win_rate * total_odds
            if expected_ev > best_ev:
                best_ev = expected_ev
                best = {
                    "tier": "optimal",
                    "legs": list(combo),
                    "total_odds": round(total_odds, 4),
                    "win_rate": round(win_rate, 6),
                    "expected_ev": round(expected_ev, 4),
                }

    return best


def build_parlay_plan(
    candidates: list[ParlayLeg],
    plan_date: str,
    total_budget: float,
) -> ParlayPlan:
    """按三层策略生成当日串场方案。

    保底层（hedge）：2~3 腿，胜率最高的稳健组合，占预算 40%
    核心层（core） ：4~5 腿，期望值最高的主力组合，占预算 40%
    冲击层（aggressive）：6~7 腿，高赔率组合，占预算 20%

    各层独立搜索最优组合，无腿数外的最低胜率约束。
    若候选不足以构成某一层的最小腿数，则跳过该层。
    至少需生成一层，否则抛出 ValueError。
    """
    if total_budget <= 0:
        raise ValueError("total_budget 必须大于 0")

    tier_ranges = [
        ("hedge",      2, 3),
        ("core",       4, 5),
        ("aggressive", 6, 7),
    ]

    options: list[ParlayOption] = []
    for tier, min_l, max_l in tier_ranges:
        if len(candidates) < min_l:
            continue
        option = find_optimal_parlay(
            candidates, min_legs=min_l, max_legs=max_l, min_win_rate=0.0
        )
        if option is not None:
            option["tier"] = tier
            options.append(option)

    if not options:
        raise ValueError("候选场次不足，无法生成任何串场方案（至少需要 2 个正期望场次）")

    plan: ParlayPlan = {
        "plan_date": plan_date,
        "options": options,
        "total_budget": total_budget,
    }
    validate_parlay_plan(plan)
    return plan
