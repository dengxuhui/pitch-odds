from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from statistics import mean, pstdev
from typing import Any

from interfaces.contracts import MatchFeatures
from models.calibration import IsotonicThreeWayCalibrator
from worldcup.data import WorldCupMatch
from worldcup.model import WorldCupModel


@dataclass
class WCPrediction:
    match_id: int
    tournament: str
    stage: str
    match_date: str
    home_team_name: str
    away_team_name: str
    actual_outcome: str        # "H" / "D" / "A"
    p_home_raw: float
    p_draw_raw: float
    p_away_raw: float
    p_home: float              # 校准后（若未校准则与 raw 相同）
    p_draw: float
    p_away: float
    odds_home: float
    odds_draw: float
    odds_away: float


@dataclass
class WCBacktestResult:
    model_version: str
    train_tournaments: list[str]
    val_tournament: str        # 校准器拟合所用锦标赛（可为空字符串）
    test_tournament: str
    predictions: list[WCPrediction]


def run_wc_backtest(
    all_matches: list[WorldCupMatch],
    *,
    train_tournaments: list[str],
    test_tournament: str,
    val_tournament: str = "",
) -> WCBacktestResult:
    """在历史世界杯数据上运行回测。

    数据切割：
    - 训练集（train_tournaments）：按时间顺序更新 Elo，不产出预测。
    - 验证集（val_tournament）：若指定，则拟合 Isotonic 校准器后再预测。
    - 测试集（test_tournament）：产出最终预测（Elo 不再更新）。

    严格时间顺序：训练赛事的截止日期 < 测试赛事的开始日期。

    世界杯模型与联赛模型完全隔离：独立 Elo 实例，不共享任何参数。
    """
    train_set = [m for m in all_matches if m.tournament in train_tournaments]
    val_set   = [m for m in all_matches if m.tournament == val_tournament] if val_tournament else []
    test_set  = [m for m in all_matches if m.tournament == test_tournament]

    # 严格时间约束检查
    if train_set and test_set:
        train_max = max(m.match_date for m in train_set)
        test_min  = min(m.match_date for m in test_set)
        if train_max >= test_min:
            raise ValueError(
                f"训练数据最晚日期 {train_max} >= 测试数据最早日期 {test_min}，"
                "存在数据泄漏风险"
            )

    # 训练模型
    model = WorldCupModel()
    train_dicts = [_match_to_dict(m) for m in sorted(train_set, key=lambda x: x.match_date)]
    if train_dicts:
        model.fit(train_dicts, league_id="WC")

    # 可选：拟合校准器
    calibrator: IsotonicThreeWayCalibrator | None = None
    if val_set:
        calibrator = IsotonicThreeWayCalibrator()
        val_raws, val_outcomes = [], []
        for m in sorted(val_set, key=lambda x: x.match_date):
            if m.home_goals is None or m.away_goals is None:
                continue
            features = _match_to_features(m)
            raw = model.predict(features)
            val_raws.append(raw)
            val_outcomes.append(_result_label(m.home_goals, m.away_goals))
            # 验证集同样更新 Elo（时间顺序）
            model.elo.update(
                m.home_team_id, m.away_team_id,
                m.home_goals, m.away_goals,
                neutral=m.neutral,
            )
        if val_raws:
            calibrator.fit(val_raws, val_outcomes)

    # 测试集预测（不更新 Elo）
    predictions: list[WCPrediction] = []
    for m in sorted(test_set, key=lambda x: x.match_date):
        if m.home_goals is None or m.away_goals is None:
            continue
        features = _match_to_features(m)
        raw = model.predict(features)
        actual = _result_label(m.home_goals, m.away_goals)

        if calibrator:
            cal = calibrator.calibrate(raw, features)
            p_home, p_draw, p_away = cal["p_home"], cal["p_draw"], cal["p_away"]
        else:
            p_home = raw["p_home_raw"]
            p_draw = raw["p_draw_raw"]
            p_away = raw["p_away_raw"]

        predictions.append(WCPrediction(
            match_id=m.match_id,
            tournament=m.tournament,
            stage=m.stage,
            match_date=m.match_date,
            home_team_name=m.home_team_name,
            away_team_name=m.away_team_name,
            actual_outcome=actual,
            p_home_raw=raw["p_home_raw"],
            p_draw_raw=raw["p_draw_raw"],
            p_away_raw=raw["p_away_raw"],
            p_home=p_home,
            p_draw=p_draw,
            p_away=p_away,
            odds_home=m.odds_home,
            odds_draw=m.odds_draw,
            odds_away=m.odds_away,
        ))

    return WCBacktestResult(
        model_version=model.model_version,
        train_tournaments=train_tournaments,
        val_tournament=val_tournament,
        test_tournament=test_tournament,
        predictions=predictions,
    )


def compute_wc_metrics(result: WCBacktestResult) -> dict[str, Any]:
    """计算世界杯回测指标（与联赛回测接口一致）。"""
    if not result.predictions:
        raise ValueError("没有可计算指标的预测结果")

    brier_terms, hits, profits = [], 0, []
    cumulative, peak, max_dd = 0.0, 0.0, 0.0

    for p in result.predictions:
        probs = {"H": p.p_home, "D": p.p_draw, "A": p.p_away}
        one_hot = {"H": 0.0, "D": 0.0, "A": 0.0}
        one_hot[p.actual_outcome] = 1.0
        brier = sum((probs[k] - one_hot[k]) ** 2 for k in probs) / 3.0
        brier_terms.append(brier)

        pred = max(probs, key=probs.get)
        if pred == p.actual_outcome:
            hits += 1

        if p.odds_home > 0:
            odds_map = {"H": p.odds_home, "D": p.odds_draw, "A": p.odds_away}
            payout = odds_map[pred] if pred == p.actual_outcome else 0.0
            profit = payout - 1.0
            profits.append(profit)
            cumulative += profit
            peak = max(peak, cumulative)
            max_dd = max(max_dd, peak - cumulative)

    n = len(result.predictions)
    roi = sum(profits) / len(profits) if profits else 0.0
    sharpe = 0.0
    if len(profits) > 1:
        avg_p = mean(profits)
        std_p = pstdev(profits)
        sharpe = (avg_p / std_p) * sqrt(len(profits)) if std_p > 0 else 0.0

    return {
        "total_matches": n,
        "brier_score": round(float(mean(brier_terms)), 6),
        "hit_rate": round(hits / n, 6),
        "roi": round(roi, 6),
        "max_drawdown": round(max_dd, 6),
        "sharpe_ratio": round(sharpe, 6),
    }


# ──────────────────────────────────────────────
# 内部辅助
# ──────────────────────────────────────────────

def _result_label(home_goals: int, away_goals: int) -> str:
    if home_goals > away_goals:
        return "H"
    if home_goals < away_goals:
        return "A"
    return "D"


def _match_to_dict(m: WorldCupMatch) -> dict[str, Any]:
    return {
        "match_id": m.match_id,
        "match_date": m.match_date,
        "home_team_id": m.home_team_id,
        "away_team_id": m.away_team_id,
        "home_goals": m.home_goals,
        "away_goals": m.away_goals,
        "neutral": m.neutral,
        "game_type": "world_cup",
    }


def _match_to_features(m: WorldCupMatch) -> MatchFeatures:
    """构造 WorldCupMatch → MatchFeatures（世界杯不涉及的字段填默认值）。"""
    if m.odds_home > 0:
        overround = 1.0 / m.odds_home + 1.0 / m.odds_draw + 1.0 / m.odds_away
        p_imp_h = (1.0 / m.odds_home) / overround
        p_imp_d = (1.0 / m.odds_draw) / overround
        p_imp_a = (1.0 / m.odds_away) / overround
        odds_h, odds_d, odds_a = m.odds_home, m.odds_draw, m.odds_away
    else:
        p_imp_h, p_imp_d, p_imp_a = 1.0/3, 1.0/3, 1.0/3
        odds_h = odds_d = odds_a = 3.0

    return {
        "match_id": m.match_id,
        "league_id": "WC",
        "match_date": m.match_date,
        "match_week": 0,
        "home_team_id": m.home_team_id,
        "away_team_id": m.away_team_id,
        "home_form_5": 0.0,   "away_form_5": 0.0,
        "home_form_10": 0.0,  "away_form_10": 0.0,
        "home_goals_scored_avg": 0.0,
        "home_goals_conceded_avg": 0.0,
        "away_goals_scored_avg": 0.0,
        "away_goals_conceded_avg": 0.0,
        "home_fatigue": 0.0,       "away_fatigue": 0.0,
        "home_injury_impact": 0.0, "away_injury_impact": 0.0,
        "home_momentum": 0.0,      "away_momentum": 0.0,
        "days_rest_home": 7,       "days_rest_away": 7,
        "odds_home": odds_h,
        "odds_draw": odds_d,
        "odds_away": odds_a,
        "p_implied_home": p_imp_h,
        "p_implied_draw": p_imp_d,
        "p_implied_away": p_imp_a,
        "odds_drift_home": 0.0,
        "smart_money_flag": False,
        "exclude_flag": False,
    }
