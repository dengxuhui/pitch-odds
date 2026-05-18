from __future__ import annotations

import pytest

from interfaces.contracts import CalibratedPrediction, ParlayLeg
from optimizer.ev_filter import filter_positive_ev
from optimizer.parlay_optimizer import build_parlay_plan, find_optimal_parlay
from optimizer.system_bet import system_bet, system_bet_stats


# ──────────────────────────────────────────────
# 测试用数据工厂
# ──────────────────────────────────────────────

def _make_prediction(
    match_id: int,
    p_home: float = 0.5,
    p_draw: float = 0.25,
    p_away: float = 0.25,
    odds_home: float = 2.2,
    odds_draw: float = 3.4,
    odds_away: float = 4.0,
    exclude_flag: bool = False,
) -> CalibratedPrediction:
    overround = 1 / odds_home + 1 / odds_draw + 1 / odds_away
    return {
        "match_id": match_id,
        "model_version": "dixon_coles_v1",
        "p_home": p_home,
        "p_draw": p_draw,
        "p_away": p_away,
        "odds_home": odds_home,
        "odds_draw": odds_draw,
        "odds_away": odds_away,
        "ev_home": p_home * odds_home,
        "ev_draw": p_draw * odds_draw,
        "ev_away": p_away * odds_away,
        "edge_home": p_home - (1 / odds_home) / overround,
        "edge_draw": p_draw - (1 / odds_draw) / overround,
        "edge_away": p_away - (1 / odds_away) / overround,
        "smart_money_flag": False,
        "exclude_flag": exclude_flag,
    }


def _make_leg(match_id: int, p_model: float = 0.55, odds: float = 2.1) -> ParlayLeg:
    return {
        "match_id": match_id,
        "outcome": "home",
        "odds": odds,
        "p_model": p_model,
        "ev": p_model * odds,
        "edge": 0.05,
    }


# ──────────────────────────────────────────────
# ev_filter 测试
# ──────────────────────────────────────────────

def test_filter_excludes_anomaly_flag() -> None:
    preds = [_make_prediction(1, exclude_flag=True)]
    assert filter_positive_ev(preds) == []


def test_filter_requires_safety_margin() -> None:
    # p_home=0.45, odds_home=2.2 → ev=0.99 < 1.05，不应入选
    preds = [_make_prediction(1, p_home=0.45, odds_home=2.2)]
    result = filter_positive_ev(preds)
    assert all(leg["outcome"] != "home" or leg["match_id"] != 1 for leg in result)


def test_filter_returns_positive_ev_candidates() -> None:
    # p_home=0.55, odds_home=2.2 → ev=1.21 >= 1.05
    preds = [_make_prediction(1, p_home=0.55, odds_home=2.2)]
    result = filter_positive_ev(preds)
    assert len(result) == 1
    assert result[0]["match_id"] == 1
    assert result[0]["outcome"] == "home"


def test_filter_keeps_best_ev_per_match() -> None:
    # 同一场比赛 home(EV=1.21) 和 away(EV=1.20)，只保留 EV 更高的
    preds = [_make_prediction(1, p_home=0.55, odds_home=2.2, p_away=0.40, odds_away=3.0)]
    result = filter_positive_ev(preds)
    match_ids = [leg["match_id"] for leg in result]
    assert match_ids.count(1) == 1
    assert result[0]["outcome"] == "home"  # ev=1.21 > ev=1.20


def test_filter_sorted_by_ev_descending() -> None:
    preds = [
        _make_prediction(1, p_home=0.55, odds_home=2.2),   # ev=1.21
        _make_prediction(2, p_home=0.60, odds_home=2.1),   # ev=1.26
        _make_prediction(3, p_home=0.58, odds_home=2.0),   # ev=1.16
    ]
    result = filter_positive_ev(preds)
    evs = [leg["ev"] for leg in result]
    assert evs == sorted(evs, reverse=True)


def test_filter_custom_safety_margin() -> None:
    # EV=1.21，margin=1.25 时不应入选
    preds = [_make_prediction(1, p_home=0.55, odds_home=2.2)]
    assert filter_positive_ev(preds, safety_margin=1.25) == []


def test_filter_invalid_margin_raises() -> None:
    with pytest.raises(ValueError):
        filter_positive_ev([], safety_margin=0)


# ──────────────────────────────────────────────
# find_optimal_parlay 测试
# ──────────────────────────────────────────────

def test_find_optimal_parlay_returns_best_ev() -> None:
    legs = [_make_leg(i, p_model=0.55, odds=2.1) for i in range(1, 6)]
    result = find_optimal_parlay(legs, min_win_rate=0.0)
    assert result is not None
    assert result["expected_ev"] > 0


def test_find_optimal_parlay_respects_min_win_rate() -> None:
    # 5 腿，每腿胜率 0.4 → 2 腿胜率 0.16 < 0.20
    legs = [_make_leg(i, p_model=0.40, odds=2.5) for i in range(1, 6)]
    result = find_optimal_parlay(legs, min_legs=2, max_legs=2, min_win_rate=0.20)
    # 2 腿：0.4^2 = 0.16 < 0.20，不满足
    assert result is None


def test_find_optimal_parlay_leg_count_within_range() -> None:
    legs = [_make_leg(i) for i in range(1, 8)]
    result = find_optimal_parlay(legs, min_legs=3, max_legs=5, min_win_rate=0.0)
    assert result is not None
    assert 3 <= len(result["legs"]) <= 5


def test_find_optimal_parlay_probability_product_correct() -> None:
    legs = [_make_leg(i, p_model=0.6, odds=2.0) for i in range(1, 4)]
    result = find_optimal_parlay(legs, min_legs=2, max_legs=3, min_win_rate=0.0)
    assert result is not None
    n = len(result["legs"])
    expected_wr = 0.6 ** n
    assert abs(result["win_rate"] - expected_wr) < 1e-6


def test_find_optimal_parlay_invalid_legs_raises() -> None:
    legs = [_make_leg(i) for i in range(1, 4)]
    with pytest.raises(ValueError):
        find_optimal_parlay(legs, min_legs=1)
    with pytest.raises(ValueError):
        find_optimal_parlay(legs, min_legs=5, max_legs=3)


def test_find_optimal_parlay_returns_none_when_no_candidates() -> None:
    assert find_optimal_parlay([], min_win_rate=0.0) is None


# ──────────────────────────────────────────────
# build_parlay_plan 测试
# ──────────────────────────────────────────────

def _enough_candidates(n: int = 7) -> list[ParlayLeg]:
    return [_make_leg(i, p_model=0.55, odds=2.1) for i in range(1, n + 1)]


def test_build_parlay_plan_three_tiers() -> None:
    candidates = _enough_candidates(7)
    plan = build_parlay_plan(candidates, "2026-05-18", 1000.0)
    tiers = {opt["tier"] for opt in plan["options"]}
    assert tiers == {"hedge", "core", "aggressive"}


def test_build_parlay_plan_hedge_two_to_three_legs() -> None:
    candidates = _enough_candidates(7)
    plan = build_parlay_plan(candidates, "2026-05-18", 1000.0)
    hedge = next(o for o in plan["options"] if o["tier"] == "hedge")
    assert 2 <= len(hedge["legs"]) <= 3


def test_build_parlay_plan_core_four_to_five_legs() -> None:
    candidates = _enough_candidates(7)
    plan = build_parlay_plan(candidates, "2026-05-18", 1000.0)
    core = next(o for o in plan["options"] if o["tier"] == "core")
    assert 4 <= len(core["legs"]) <= 5


def test_build_parlay_plan_aggressive_six_to_seven_legs() -> None:
    candidates = _enough_candidates(7)
    plan = build_parlay_plan(candidates, "2026-05-18", 1000.0)
    agg = next(o for o in plan["options"] if o["tier"] == "aggressive")
    assert 6 <= len(agg["legs"]) <= 7


def test_build_parlay_plan_skips_tier_when_insufficient_candidates() -> None:
    # 只有 3 个候选，core(4腿) 和 aggressive(6腿) 无法构成
    candidates = _enough_candidates(3)
    plan = build_parlay_plan(candidates, "2026-05-18", 500.0)
    tiers = {opt["tier"] for opt in plan["options"]}
    assert "hedge" in tiers
    assert "core" not in tiers
    assert "aggressive" not in tiers


def test_build_parlay_plan_raises_when_no_candidates() -> None:
    with pytest.raises(ValueError, match="候选场次不足"):
        build_parlay_plan([], "2026-05-18", 1000.0)


def test_build_parlay_plan_raises_when_only_one_candidate() -> None:
    with pytest.raises(ValueError, match="候选场次不足"):
        build_parlay_plan([_make_leg(1)], "2026-05-18", 1000.0)


def test_build_parlay_plan_passes_contract_validation() -> None:
    # validate_parlay_plan 内部已校验，能返回说明通过
    candidates = _enough_candidates(5)
    plan = build_parlay_plan(candidates, "2026-05-18", 1000.0)
    assert plan["plan_date"] == "2026-05-18"
    assert plan["total_budget"] == 1000.0
    assert plan["options"]


def test_build_parlay_plan_invalid_budget_raises() -> None:
    with pytest.raises(ValueError):
        build_parlay_plan(_enough_candidates(5), "2026-05-18", 0.0)


# ──────────────────────────────────────────────
# system_bet 测试
# ──────────────────────────────────────────────

def test_system_bet_combo_count() -> None:
    # C(5,4) = 5
    legs = [_make_leg(i) for i in range(1, 6)]
    combos = system_bet(legs, system_size=4)
    assert len(combos) == 5


def test_system_bet_combo_count_6_choose_5() -> None:
    # C(6,5) = 6
    legs = [_make_leg(i) for i in range(1, 7)]
    combos = system_bet(legs, system_size=5)
    assert len(combos) == 6


def test_system_bet_each_combo_correct_size() -> None:
    legs = [_make_leg(i) for i in range(1, 6)]
    combos = system_bet(legs, system_size=4)
    assert all(len(c) == 4 for c in combos)


def test_system_bet_all_combinations_unique() -> None:
    legs = [_make_leg(i) for i in range(1, 6)]
    combos = system_bet(legs, system_size=4)
    ids_sets = [frozenset(leg["match_id"] for leg in c) for c in combos]
    assert len(set(ids_sets)) == len(ids_sets)


def test_system_bet_invalid_size_raises() -> None:
    legs = [_make_leg(i) for i in range(1, 6)]
    with pytest.raises(ValueError):
        system_bet(legs, system_size=1)    # < 2
    with pytest.raises(ValueError):
        system_bet(legs, system_size=5)    # >= n


def test_system_bet_stats_structure() -> None:
    legs = [_make_leg(i, p_model=0.6, odds=2.0) for i in range(1, 6)]
    combos = system_bet(legs, system_size=4)
    stats = system_bet_stats(combos)
    assert "n_combos" in stats
    assert "avg_win_rate" in stats
    assert "avg_total_odds" in stats
    assert "avg_expected_ev" in stats
    assert stats["n_combos"] == 5
    # 每组 4 腿：胜率 0.6^4 = 0.1296，赔率 2.0^4 = 16.0，EV = 2.0736
    assert abs(stats["avg_win_rate"] - 0.6 ** 4) < 1e-5


# ──────────────────────────────────────────────
# 端到端：filter → optimize → system_bet
# ──────────────────────────────────────────────

def test_e2e_filter_to_plan() -> None:
    preds = [
        _make_prediction(i, p_home=0.55 + i * 0.01, odds_home=2.0 + i * 0.05)
        for i in range(8)
    ]
    candidates = filter_positive_ev(preds)
    assert len(candidates) >= 2
    plan = build_parlay_plan(candidates, "2026-05-18", 1000.0)
    assert plan["options"]


def test_e2e_system_bet_after_filter() -> None:
    preds = [_make_prediction(i, p_home=0.58, odds_home=2.2) for i in range(1, 7)]
    candidates = filter_positive_ev(preds)
    combos = system_bet(candidates, system_size=len(candidates) - 1)
    assert len(combos) == len(candidates)
