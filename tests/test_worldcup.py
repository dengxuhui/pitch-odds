from __future__ import annotations

import pytest

from worldcup.elo import EloRating, _goal_multiplier, K_FACTORS
from worldcup.club_form import ClubFormMapper
from worldcup.data import WorldCupMatch, sample_wc_matches, team_id_for
from worldcup.model import WorldCupModel, _elo_to_probabilities
from worldcup.backtest import (
    WCBacktestResult,
    run_wc_backtest,
    compute_wc_metrics,
    _result_label,
)


# ──────────────────────────────────────────────
# EloRating
# ──────────────────────────────────────────────

def test_elo_initial_rating() -> None:
    elo = EloRating(initial_rating=1500)
    assert elo.get_rating(999) == 1500


def test_elo_expected_score_equal_teams() -> None:
    elo = EloRating()
    elo.set_rating(1, 1500)
    elo.set_rating(2, 1500)
    e = elo.expected_score(1, 2)
    assert abs(e - 0.5) < 1e-9


def test_elo_expected_score_stronger_team() -> None:
    elo = EloRating()
    elo.set_rating(1, 1700)
    elo.set_rating(2, 1500)
    e = elo.expected_score(1, 2)
    assert e > 0.5


def test_elo_update_win_increases_winner() -> None:
    elo = EloRating(initial_rating=1500)
    elo.set_rating(1, 1500)
    elo.set_rating(2, 1500)
    delta_h, delta_a = elo.update(1, 2, 2, 0)
    assert delta_h > 0
    assert delta_a < 0
    assert elo.get_rating(1) > 1500
    assert elo.get_rating(2) < 1500


def test_elo_update_draw_equal_teams_no_change() -> None:
    elo = EloRating(initial_rating=1500, k_factor=40)
    elo.set_rating(1, 1500)
    elo.set_rating(2, 1500)
    delta_h, delta_a = elo.update(1, 2, 1, 1)
    assert abs(delta_h) < 1e-9
    assert abs(delta_a) < 1e-9


def test_elo_update_sum_zero() -> None:
    elo = EloRating()
    elo.set_rating(10, 1600)
    elo.set_rating(20, 1400)
    d_h, d_a = elo.update(10, 20, 0, 1)
    assert abs(d_h + d_a) < 1e-9


def test_elo_home_advantage_applied() -> None:
    elo = EloRating(initial_rating=1500, home_advantage=200)
    elo.set_rating(1, 1500)
    elo.set_rating(2, 1500)
    e_home = elo.expected_score(1, 2, neutral=False)
    e_neutral = elo.expected_score(1, 2, neutral=True)
    assert e_home > e_neutral


def test_elo_copy_independence() -> None:
    elo = EloRating()
    elo.set_rating(1, 1600)
    clone = elo.copy()
    elo.set_rating(1, 1700)
    assert clone.get_rating(1) == 1600  # 不受原始修改影响


def test_elo_to_from_dict_roundtrip() -> None:
    elo = EloRating()
    elo.set_rating(5, 1650)
    elo.set_rating(6, 1350)
    d = elo.to_dict()
    elo2 = EloRating()
    elo2.from_dict(d)
    assert elo2.get_rating(5) == 1650
    assert elo2.get_rating(6) == 1350


def test_goal_multiplier_values() -> None:
    assert _goal_multiplier(0) == 1.0
    assert _goal_multiplier(1) == 1.0
    assert _goal_multiplier(2) == 1.5
    assert _goal_multiplier(3) == pytest.approx((11 + 3) / 8)
    assert _goal_multiplier(5) > _goal_multiplier(3)


# ──────────────────────────────────────────────
# ClubFormMapper
# ──────────────────────────────────────────────

def test_club_form_no_players_returns_zero() -> None:
    mapper = ClubFormMapper()
    assert mapper.get_adjustment(999) == 0.0


def test_club_form_positive_form_positive_adj() -> None:
    mapper = ClubFormMapper()
    mapper.add_player(1, 0.8)
    mapper.add_player(1, 0.6)
    adj = mapper.get_adjustment(1)
    assert adj > 0


def test_club_form_negative_form_negative_adj() -> None:
    mapper = ClubFormMapper()
    mapper.add_player(2, -0.9)
    adj = mapper.get_adjustment(2)
    assert adj < 0


def test_club_form_capped_at_50() -> None:
    mapper = ClubFormMapper()
    for _ in range(10):
        mapper.add_player(3, 1.0)
    assert mapper.get_adjustment(3) == 50.0


def test_club_form_capped_at_minus_50() -> None:
    mapper = ClubFormMapper()
    for _ in range(10):
        mapper.add_player(4, -1.0)
    assert mapper.get_adjustment(4) == -50.0


def test_club_form_clear() -> None:
    mapper = ClubFormMapper()
    mapper.add_player(1, 0.9)
    mapper.clear()
    assert mapper.get_adjustment(1) == 0.0


# ──────────────────────────────────────────────
# WorldCupModel 概率转换
# ──────────────────────────────────────────────

def test_elo_to_probs_sum_to_one() -> None:
    for diff in [-300, -100, 0, 100, 300]:
        p_h, p_d, p_a = _elo_to_probabilities(1500 + diff, 1500)
        assert abs(p_h + p_d + p_a - 1.0) < 1e-9


def test_elo_to_probs_equal_teams_symmetric() -> None:
    p_h, p_d, p_a = _elo_to_probabilities(1500, 1500)
    assert abs(p_h - p_a) < 1e-9


def test_elo_to_probs_stronger_home_wins() -> None:
    p_h, _, p_a = _elo_to_probabilities(1800, 1400)
    assert p_h > p_a


def test_elo_to_probs_draw_at_least_5pct() -> None:
    # 即使 Elo 差极大，平局概率不低于 0.05
    _, p_d, _ = _elo_to_probabilities(2500, 1000)
    assert p_d >= 0.05


# ──────────────────────────────────────────────
# WorldCupModel.fit / predict
# ──────────────────────────────────────────────

def _make_model_and_teams() -> tuple[WorldCupModel, int, int]:
    model = WorldCupModel()
    t1 = team_id_for("TestTeamAlpha")
    t2 = team_id_for("TestTeamBeta")
    return model, t1, t2


def test_model_predict_output_keys() -> None:
    model, t1, t2 = _make_model_and_teams()
    features = _dummy_features(1, t1, t2)
    raw = model.predict(features)
    for key in ("match_id", "model_version", "predicted_at",
                "p_home_raw", "p_draw_raw", "p_away_raw"):
        assert key in raw


def test_model_predict_probs_sum_to_one() -> None:
    model, t1, t2 = _make_model_and_teams()
    raw = model.predict(_dummy_features(1, t1, t2))
    total = raw["p_home_raw"] + raw["p_draw_raw"] + raw["p_away_raw"]
    assert abs(total - 1.0) < 1e-6


def test_model_fit_updates_elo() -> None:
    model, t1, t2 = _make_model_and_teams()
    before = model.elo.get_rating(t1)
    model.fit([{
        "match_id": 1, "match_date": "2022-01-01",
        "home_team_id": t1, "away_team_id": t2,
        "home_goals": 3, "away_goals": 0,
        "neutral": True,
    }])
    assert model.elo.get_rating(t1) > before


def test_model_get_set_params_roundtrip() -> None:
    model, t1, t2 = _make_model_and_teams()
    model.elo.set_rating(t1, 1750)
    params = model.get_params()
    model2 = WorldCupModel()
    model2.load_params(params)
    assert abs(model2.elo.get_rating(t1) - 1750) < 1e-6


def test_model_independence_from_league_model() -> None:
    from models.dixon_coles import DixonColesModel
    wc = WorldCupModel()
    dc = DixonColesModel()
    # 确保没有共享状态：各自有独立的 model_version
    assert wc.model_version != dc.model_version
    assert "elo" in wc.model_version
    assert "dixon" in dc.model_version


def test_model_momentum_adjusts_prediction() -> None:
    model, t1, t2 = _make_model_and_teams()
    model.elo.set_rating(t1, 1500)
    model.elo.set_rating(t2, 1500)
    feat_baseline = _dummy_features(1, t1, t2, momentum_home=0.0)
    feat_boost    = _dummy_features(1, t1, t2, momentum_home=0.10)
    p_base  = model.predict(feat_baseline)["p_home_raw"]
    p_boost = model.predict(feat_boost)["p_home_raw"]
    assert p_boost > p_base


# ──────────────────────────────────────────────
# WC Backtest
# ──────────────────────────────────────────────

def test_result_label() -> None:
    assert _result_label(2, 0) == "H"
    assert _result_label(0, 1) == "A"
    assert _result_label(1, 1) == "D"


def test_run_wc_backtest_with_sample_data() -> None:
    matches = sample_wc_matches()
    result = run_wc_backtest(
        matches,
        train_tournaments=[],
        test_tournament="WC2022",
    )
    assert result.test_tournament == "WC2022"
    assert len(result.predictions) == len(matches)


def test_run_wc_backtest_probs_valid() -> None:
    result = run_wc_backtest(
        sample_wc_matches(),
        train_tournaments=[],
        test_tournament="WC2022",
    )
    for p in result.predictions:
        total = p.p_home + p.p_draw + p.p_away
        assert abs(total - 1.0) < 1e-5
        assert 0.0 <= p.p_home <= 1.0
        assert 0.0 <= p.p_draw <= 1.0
        assert 0.0 <= p.p_away <= 1.0


def test_run_wc_backtest_data_leakage_raises() -> None:
    """训练集日期晚于测试集应抛出 ValueError。"""
    # 用同一组数据构造冲突场景：先用 WC2022 训练再用更早的日期测试
    # 通过伪造两个 tournament 且日期重叠模拟
    matches = [
        _make_wc_match(1, "WC_A", "2022-12-18"),   # "训练集"
        _make_wc_match(2, "WC_B", "2022-11-20"),   # "测试集"（日期更早）
    ]
    with pytest.raises(ValueError, match="数据泄漏"):
        run_wc_backtest(matches, train_tournaments=["WC_A"], test_tournament="WC_B")


def test_compute_wc_metrics_structure() -> None:
    result = run_wc_backtest(
        sample_wc_matches(),
        train_tournaments=[],
        test_tournament="WC2022",
    )
    metrics = compute_wc_metrics(result)
    for key in ("total_matches", "brier_score", "hit_rate", "roi",
                "max_drawdown", "sharpe_ratio"):
        assert key in metrics


def test_compute_wc_metrics_brier_range() -> None:
    result = run_wc_backtest(
        sample_wc_matches(),
        train_tournaments=[],
        test_tournament="WC2022",
    )
    metrics = compute_wc_metrics(result)
    assert 0.0 <= metrics["brier_score"] <= 1.0


def test_compute_wc_metrics_hit_rate_range() -> None:
    result = run_wc_backtest(
        sample_wc_matches(),
        train_tournaments=[],
        test_tournament="WC2022",
    )
    metrics = compute_wc_metrics(result)
    assert 0.0 <= metrics["hit_rate"] <= 1.0


def test_compute_wc_metrics_empty_raises() -> None:
    empty = WCBacktestResult(
        model_version="x", train_tournaments=[], val_tournament="",
        test_tournament="WC2022", predictions=[]
    )
    with pytest.raises(ValueError):
        compute_wc_metrics(empty)


def test_wc_model_not_shared_with_league() -> None:
    """世界杯训练不修改联赛 Dixon-Coles 模型的任何模块级状态。"""
    from models.dixon_coles import DixonColesModel
    dc = DixonColesModel()
    initial_version = dc.model_version

    run_wc_backtest(
        sample_wc_matches(),
        train_tournaments=[],
        test_tournament="WC2022",
    )

    assert dc.model_version == initial_version


# ──────────────────────────────────────────────
# 辅助工厂
# ──────────────────────────────────────────────

def _dummy_features(
    match_id: int,
    home_id: int,
    away_id: int,
    momentum_home: float = 0.0,
    momentum_away: float = 0.0,
) -> dict:
    return {
        "match_id": match_id,
        "league_id": "WC",
        "match_date": "2022-12-18",
        "match_week": 0,
        "home_team_id": home_id,
        "away_team_id": away_id,
        "home_form_5": 0.0,   "away_form_5": 0.0,
        "home_form_10": 0.0,  "away_form_10": 0.0,
        "home_goals_scored_avg": 0.0,
        "home_goals_conceded_avg": 0.0,
        "away_goals_scored_avg": 0.0,
        "away_goals_conceded_avg": 0.0,
        "home_fatigue": 0.0,       "away_fatigue": 0.0,
        "home_injury_impact": 0.0, "away_injury_impact": 0.0,
        "home_momentum": momentum_home,
        "away_momentum": momentum_away,
        "days_rest_home": 7, "days_rest_away": 7,
        "odds_home": 2.5, "odds_draw": 3.2, "odds_away": 3.0,
        "p_implied_home": 0.385, "p_implied_draw": 0.300, "p_implied_away": 0.315,
        "odds_drift_home": 0.0,
        "smart_money_flag": False,
        "exclude_flag": False,
    }


def _make_wc_match(match_id: int, tournament: str, date: str) -> WorldCupMatch:
    return WorldCupMatch(
        match_id=match_id,
        tournament=tournament,
        stage="group",
        match_date=date,
        home_team_id=team_id_for(f"TeamX_{match_id}"),
        away_team_id=team_id_for(f"TeamY_{match_id}"),
        home_team_name=f"TeamX_{match_id}",
        away_team_name=f"TeamY_{match_id}",
        home_goals=1,
        away_goals=0,
        neutral=True,
        odds_home=2.0,
        odds_draw=3.5,
        odds_away=4.0,
    )
