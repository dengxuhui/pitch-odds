from __future__ import annotations

import pytest

from capital.kelly import half_kelly
from capital.allocator import allocate_capital
from capital.stop_loss import StopLossTracker
from backtest.capital_sim import simulate_capital, _parlay_won
from backtest.engine import BacktestPrediction, BacktestResult
from interfaces.contracts import ParlayLeg


# ──────────────────────────────────────────────
# half_kelly
# ──────────────────────────────────────────────

def test_half_kelly_positive_ev() -> None:
    # p=0.55, odds=2.2 → b=1.2, kelly=(1.2*0.55-0.45)/1.2=0.175, half=0.0875
    result = half_kelly(0.55, 2.2)
    assert abs(result - 0.0875) < 1e-6


def test_half_kelly_negative_ev_returns_zero() -> None:
    # p=0.3, odds=2.0 → kelly=(1.0*0.3-0.7)/1.0 = -0.4 → clamped to 0
    assert half_kelly(0.3, 2.0) == 0.0


def test_half_kelly_zero_probability_returns_zero() -> None:
    assert half_kelly(0.0, 3.0) == 0.0


def test_half_kelly_fraction_applied() -> None:
    full = half_kelly(0.6, 2.0, fraction=1.0)
    half = half_kelly(0.6, 2.0, fraction=0.5)
    assert abs(half - full / 2) < 1e-9


def test_half_kelly_invalid_odds_raises() -> None:
    with pytest.raises(ValueError):
        half_kelly(0.5, 1.0)
    with pytest.raises(ValueError):
        half_kelly(0.5, 0.9)


def test_half_kelly_invalid_probability_raises() -> None:
    with pytest.raises(ValueError):
        half_kelly(-0.1, 2.0)
    with pytest.raises(ValueError):
        half_kelly(1.1, 2.0)


def test_half_kelly_invalid_fraction_raises() -> None:
    with pytest.raises(ValueError):
        half_kelly(0.5, 2.0, fraction=0.0)
    with pytest.raises(ValueError):
        half_kelly(0.5, 2.0, fraction=1.5)


# ──────────────────────────────────────────────
# allocate_capital
# ──────────────────────────────────────────────

def _make_plan(n_options: int = 2) -> dict:
    tier_names = ["hedge", "core", "aggressive"]
    options = []
    for i in range(n_options):
        options.append({
            "tier": tier_names[i % 3],
            "legs": [{"match_id": i * 10 + j, "outcome": "home", "odds": 2.0,
                      "p_model": 0.55, "ev": 1.1, "edge": 0.05} for j in range(2)],
            "total_odds": 4.0,
            "win_rate": 0.3025,
            "expected_ev": 1.21,
        })
    return {
        "plan_date": "2026-05-18",
        "options": options,
        "total_budget": 1000.0,
    }


def test_allocate_capital_returns_one_record_per_option() -> None:
    plan = _make_plan(2)
    records = allocate_capital(plan, 10_000.0)
    assert len(records) == 2


def test_allocate_capital_stake_within_tier_cap() -> None:
    plan = _make_plan(3)
    records = allocate_capital(plan, 10_000.0)
    caps = {"hedge": 4000.0, "core": 4000.0, "aggressive": 2000.0}
    for rec in records:
        assert rec["stake"] <= caps[rec["tier"]] + 1e-6


def test_allocate_capital_stake_non_negative() -> None:
    plan = _make_plan(3)
    for rec in allocate_capital(plan, 5_000.0):
        assert rec["stake"] >= 0


def test_allocate_capital_invalid_capital_raises() -> None:
    plan = _make_plan(1)
    with pytest.raises(ValueError):
        allocate_capital(plan, 0.0)
    with pytest.raises(ValueError):
        allocate_capital(plan, -100.0)


def test_allocate_capital_records_are_simulations() -> None:
    plan = _make_plan(2)
    records = allocate_capital(plan, 1000.0, is_simulation=True)
    assert all(r["is_simulation"] for r in records)


def test_allocate_capital_unsettled_fields_none() -> None:
    plan = _make_plan(1)
    rec = allocate_capital(plan, 1000.0)[0]
    assert rec["won"] is None
    assert rec["payout"] is None
    assert rec["profit"] is None
    assert rec["settled_at"] is None


def test_allocate_capital_all_same_plan_id() -> None:
    plan = _make_plan(3)
    records = allocate_capital(plan, 1000.0)
    plan_ids = {r["plan_id"] for r in records}
    assert len(plan_ids) == 1


# ──────────────────────────────────────────────
# StopLossTracker
# ──────────────────────────────────────────────

def test_stop_loss_initial_state() -> None:
    t = StopLossTracker(10_000.0)
    assert t.capital == 10_000.0
    assert not t.is_paused
    assert t.consecutive_loss_days == 0


def test_stop_loss_allows_within_daily_limit() -> None:
    t = StopLossTracker(10_000.0, daily_loss_limit_pct=0.10)
    assert t.should_bet(1_000.0)   # 正好 10%，应允许
    assert t.should_bet(999.0)


def test_stop_loss_blocks_over_daily_limit() -> None:
    t = StopLossTracker(10_000.0, daily_loss_limit_pct=0.10)
    assert not t.should_bet(1_001.0)


def test_stop_loss_consecutive_loss_pauses() -> None:
    t = StopLossTracker(10_000.0, max_consecutive_loss_days=3)
    t.record_day(-100.0)
    t.record_day(-100.0)
    assert not t.is_paused
    t.record_day(-100.0)
    assert t.is_paused


def test_stop_loss_win_resets_consecutive() -> None:
    t = StopLossTracker(10_000.0, max_consecutive_loss_days=3)
    t.record_day(-100.0)
    t.record_day(-100.0)
    t.record_day(200.0)   # win
    assert t.consecutive_loss_days == 0
    assert not t.is_paused


def test_stop_loss_drawdown_pauses() -> None:
    t = StopLossTracker(10_000.0, max_drawdown_pct=0.30)
    t.record_day(-3_001.0)
    assert t.is_paused


def test_stop_loss_paused_blocks_all_bets() -> None:
    t = StopLossTracker(10_000.0, max_consecutive_loss_days=2)
    t.record_day(-100.0)
    t.record_day(-100.0)
    assert t.is_paused
    assert not t.should_bet(1.0)


def test_stop_loss_resume_clears_pause() -> None:
    t = StopLossTracker(10_000.0, max_consecutive_loss_days=2)
    t.record_day(-100.0)
    t.record_day(-100.0)
    t.resume()
    assert not t.is_paused
    assert t.consecutive_loss_days == 0


def test_stop_loss_peak_capital_tracks_max() -> None:
    t = StopLossTracker(10_000.0)
    t.record_day(1_000.0)
    t.record_day(-500.0)
    assert t.peak_capital == 11_000.0
    assert t.capital == 10_500.0


def test_stop_loss_invalid_initial_capital_raises() -> None:
    with pytest.raises(ValueError):
        StopLossTracker(0.0)
    with pytest.raises(ValueError):
        StopLossTracker(-1.0)


# ──────────────────────────────────────────────
# _parlay_won
# ──────────────────────────────────────────────

def _leg(match_id: int, outcome: str) -> ParlayLeg:
    return {"match_id": match_id, "outcome": outcome, "odds": 2.0,
            "p_model": 0.5, "ev": 1.0, "edge": 0.0}


def test_parlay_won_all_correct() -> None:
    legs = [_leg(1, "home"), _leg(2, "draw")]
    lookup = {1: "H", 2: "D"}
    assert _parlay_won(legs, lookup)


def test_parlay_won_one_miss() -> None:
    legs = [_leg(1, "home"), _leg(2, "away")]
    lookup = {1: "H", 2: "D"}
    assert not _parlay_won(legs, lookup)


def test_parlay_won_missing_match_id() -> None:
    legs = [_leg(1, "home"), _leg(99, "home")]
    lookup = {1: "H"}
    assert not _parlay_won(legs, lookup)


# ──────────────────────────────────────────────
# simulate_capital 端到端
# ──────────────────────────────────────────────

def _make_backtest_result(n_days: int = 5, preds_per_day: int = 8) -> BacktestResult:
    """构造一个小型 BacktestResult 用于 simulate_capital 测试。

    每天 preds_per_day 场，p_home=0.60, odds_home=2.0（EV=1.20 > 1.05）
    将胜利结果设为 home（H），确保每注串场都会赢。
    """
    predictions: list[BacktestPrediction] = []
    match_id = 1
    for day in range(n_days):
        date_str = f"2023-08-{day + 1:02d}"
        for _ in range(preds_per_day):
            predictions.append(BacktestPrediction(
                match_id=match_id,
                league_id="E0",
                season="2023-24",
                match_date=date_str,
                actual_outcome="H",
                train_until="2023-07-31",
                p_home_raw=0.58,
                p_draw_raw=0.22,
                p_away_raw=0.20,
                p_home=0.60,
                p_draw=0.22,
                p_away=0.18,
                odds_home=2.0,
                odds_draw=3.5,
                odds_away=4.5,
            ))
            match_id += 1
    return BacktestResult(
        league_id="E0",
        model_version="dixon_coles_v1",
        train_seasons=["2018-19"],
        val_season="2022-23",
        test_season="2023-24",
        predictions=predictions,
    )


def test_simulate_capital_returns_result() -> None:
    result = _make_backtest_result()
    sim = simulate_capital(result, initial_capital=10_000.0)
    assert sim.initial_capital == 10_000.0
    assert isinstance(sim.final_capital, float)


def test_simulate_capital_curve_length() -> None:
    n_days = 5
    result = _make_backtest_result(n_days=n_days)
    sim = simulate_capital(result, initial_capital=10_000.0)
    # 每天（含跳过日）各有一个资本快照，加上初始值
    assert len(sim.capital_curve) == n_days + 1


def test_simulate_capital_winning_days_increase_capital() -> None:
    result = _make_backtest_result(n_days=3, preds_per_day=8)
    sim = simulate_capital(result, initial_capital=10_000.0)
    betting_days = [d for d in sim.daily_results if not d.skipped]
    for day in betting_days:
        # 所有场次均为 home 胜，串场应全中
        if day.records:
            assert day.daily_profit >= 0


def test_simulate_capital_invalid_initial_capital_raises() -> None:
    result = _make_backtest_result()
    with pytest.raises(ValueError):
        simulate_capital(result, initial_capital=0.0)


def test_simulate_capital_skips_when_no_candidates() -> None:
    # 所有方向 EV < 1.05：p * odds < 1.05
    # home: 0.40 * 2.0 = 0.80, draw: 0.30 * 3.0 = 0.90, away: 0.30 * 3.0 = 0.90
    predictions = [
        BacktestPrediction(
            match_id=i, league_id="E0", season="2023-24",
            match_date="2023-08-01", actual_outcome="H",
            train_until="2023-07-31",
            p_home_raw=0.4, p_draw_raw=0.3, p_away_raw=0.3,
            p_home=0.4, p_draw=0.3, p_away=0.3,
            odds_home=2.0, odds_draw=3.0, odds_away=3.0,
        )
        for i in range(1, 5)
    ]
    result = BacktestResult(
        league_id="E0", model_version="v1",
        train_seasons=[], val_season="", test_season="",
        predictions=predictions,
    )
    sim = simulate_capital(result, initial_capital=10_000.0)
    assert sim.n_skipped_days >= 1
    assert sim.n_betting_days == 0


def test_simulate_capital_stop_loss_triggers() -> None:
    # 每日亏损：所有结果为 away（A），但预测是 home → 串场全输
    predictions: list[BacktestPrediction] = []
    for day in range(10):
        date_str = f"2023-08-{day + 1:02d}"
        for i in range(8):
            predictions.append(BacktestPrediction(
                match_id=day * 100 + i,
                league_id="E0", season="2023-24",
                match_date=date_str, actual_outcome="A",
                train_until="2023-07-31",
                p_home_raw=0.58, p_draw_raw=0.22, p_away_raw=0.20,
                p_home=0.60, p_draw=0.22, p_away=0.18,
                odds_home=2.0, odds_draw=3.5, odds_away=4.5,
            ))
    result = BacktestResult(
        league_id="E0", model_version="v1",
        train_seasons=[], val_season="", test_season="",
        predictions=predictions,
    )
    # 止损非常严格：连亏 2 天即暂停
    sim = simulate_capital(
        result, initial_capital=10_000.0,
        max_consecutive_loss_days=2,
    )
    # 必然有跳过日（止损触发后）
    assert sim.n_skipped_days > 0
