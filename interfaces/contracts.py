from __future__ import annotations

from datetime import date, datetime
from typing import Any, Mapping, Optional, TypedDict


PROBABILITY_TOLERANCE = 1e-6


class MatchFeatures(TypedDict):
    match_id: int
    league_id: str
    match_date: str
    match_week: int
    home_team_id: int
    away_team_id: int
    home_form_5: float
    away_form_5: float
    home_form_10: float
    away_form_10: float
    home_goals_scored_avg: float
    home_goals_conceded_avg: float
    away_goals_scored_avg: float
    away_goals_conceded_avg: float
    home_fatigue: float
    away_fatigue: float
    home_injury_impact: float
    away_injury_impact: float
    home_momentum: float
    away_momentum: float
    days_rest_home: int
    days_rest_away: int
    odds_home: float
    odds_draw: float
    odds_away: float
    p_implied_home: float
    p_implied_draw: float
    p_implied_away: float
    odds_drift_home: float
    smart_money_flag: bool
    exclude_flag: bool


class ModelRawOutput(TypedDict):
    match_id: int
    model_version: str
    predicted_at: str
    p_home_raw: float
    p_draw_raw: float
    p_away_raw: float
    lambda_home: Optional[float]
    lambda_away: Optional[float]


class CalibratedPrediction(TypedDict):
    match_id: int
    model_version: str
    p_home: float
    p_draw: float
    p_away: float
    odds_home: float
    odds_draw: float
    odds_away: float
    ev_home: float
    ev_draw: float
    ev_away: float
    edge_home: float
    edge_draw: float
    edge_away: float
    smart_money_flag: bool
    exclude_flag: bool


class ParlayLeg(TypedDict):
    match_id: int
    outcome: str
    odds: float
    p_model: float
    ev: float
    edge: float


class ParlayOption(TypedDict):
    tier: str
    legs: list[ParlayLeg]
    total_odds: float
    win_rate: float
    expected_ev: float


class ParlayPlan(TypedDict):
    plan_date: str
    options: list[ParlayOption]
    total_budget: float


class BetRecord(TypedDict):
    plan_id: str
    plan_date: str
    tier: str
    legs: list[ParlayLeg]
    total_odds: float
    win_rate: float
    stake: float
    kelly_pct: float
    is_simulation: bool
    won: Optional[bool]
    payout: Optional[float]
    profit: Optional[float]
    settled_at: Optional[str]


def _require_keys(data: Mapping[str, Any], keys: list[str], scope: str) -> None:
    missing = [k for k in keys if k not in data]
    if missing:
        raise ValueError(f"{scope} 缺少字段: {missing}")


def _validate_probability(value: Any, field_name: str) -> float:
    if not isinstance(value, (int, float)):
        raise TypeError(f"{field_name} 必须是数字")
    prob = float(value)
    if prob < 0.0 or prob > 1.0:
        raise ValueError(f"{field_name} 必须在 [0, 1] 范围内")
    return prob


def validate_probability_triplet(home: Any, draw: Any, away: Any, *, tolerance: float = PROBABILITY_TOLERANCE) -> None:
    p_home = _validate_probability(home, "p_home")
    p_draw = _validate_probability(draw, "p_draw")
    p_away = _validate_probability(away, "p_away")
    if abs((p_home + p_draw + p_away) - 1.0) > tolerance:
        raise ValueError("概率三元组之和必须等于 1.0")


def _validate_date_string(value: Any, field_name: str) -> None:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} 必须是 YYYY-MM-DD 字符串")
    date.fromisoformat(value)


def _validate_iso_datetime_with_tz(value: Any, field_name: str) -> None:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} 必须是 ISO 8601 字符串")
    dt = datetime.fromisoformat(value)
    if dt.tzinfo is None:
        raise ValueError(f"{field_name} 必须包含时区信息")


def validate_match_features(data: Mapping[str, Any]) -> None:
    _require_keys(
        data,
        [
            "match_id",
            "league_id",
            "match_date",
            "p_implied_home",
            "p_implied_draw",
            "p_implied_away",
            "home_fatigue",
            "away_fatigue",
            "home_injury_impact",
            "away_injury_impact",
            "home_momentum",
            "away_momentum",
            "odds_home",
            "odds_draw",
            "odds_away",
            "exclude_flag",
        ],
        "MatchFeatures",
    )
    _validate_date_string(data["match_date"], "match_date")
    validate_probability_triplet(data["p_implied_home"], data["p_implied_draw"], data["p_implied_away"])
    for key in ("home_fatigue", "away_fatigue"):
        _validate_probability(data[key], key)
    for key in ("home_injury_impact", "away_injury_impact"):
        val = float(data[key])
        if val < -0.30 or val > 0.0:
            raise ValueError(f"{key} 必须在 [-0.30, 0.0] 范围内")
    for key in ("home_momentum", "away_momentum"):
        val = float(data[key])
        if val < -0.10 or val > 0.10:
            raise ValueError(f"{key} 必须在 [-0.10, 0.10] 范围内")
    for key in ("odds_home", "odds_draw", "odds_away"):
        if float(data[key]) < 1.0:
            raise ValueError(f"{key} 必须 >= 1.0")


def validate_model_raw_output(data: Mapping[str, Any]) -> None:
    _require_keys(
        data,
        ["match_id", "model_version", "predicted_at", "p_home_raw", "p_draw_raw", "p_away_raw"],
        "ModelRawOutput",
    )
    _validate_iso_datetime_with_tz(data["predicted_at"], "predicted_at")
    validate_probability_triplet(data["p_home_raw"], data["p_draw_raw"], data["p_away_raw"])


def validate_calibrated_prediction(data: Mapping[str, Any]) -> None:
    _require_keys(
        data,
        [
            "match_id",
            "model_version",
            "p_home",
            "p_draw",
            "p_away",
            "odds_home",
            "odds_draw",
            "odds_away",
            "ev_home",
            "ev_draw",
            "ev_away",
            "edge_home",
            "edge_draw",
            "edge_away",
            "exclude_flag",
        ],
        "CalibratedPrediction",
    )
    validate_probability_triplet(data["p_home"], data["p_draw"], data["p_away"])


def validate_parlay_plan(data: Mapping[str, Any]) -> None:
    _require_keys(data, ["plan_date", "options", "total_budget"], "ParlayPlan")
    _validate_date_string(data["plan_date"], "plan_date")
    if float(data["total_budget"]) < 0:
        raise ValueError("total_budget 不能为负数")
    options = data["options"]
    if not isinstance(options, list) or not options:
        raise ValueError("options 必须是非空列表")
    for option in options:
        _require_keys(option, ["tier", "legs", "total_odds", "win_rate", "expected_ev"], "ParlayOption")
        if float(option["win_rate"]) < 0.0 or float(option["win_rate"]) > 1.0:
            raise ValueError("win_rate 必须在 [0, 1] 范围内")


def validate_bet_record(data: Mapping[str, Any]) -> None:
    _require_keys(
        data,
        ["plan_id", "plan_date", "tier", "legs", "total_odds", "win_rate", "stake", "kelly_pct", "is_simulation"],
        "BetRecord",
    )
    _validate_date_string(data["plan_date"], "plan_date")
    if float(data["win_rate"]) < 0.0 or float(data["win_rate"]) > 1.0:
        raise ValueError("win_rate 必须在 [0, 1] 范围内")
    if float(data["kelly_pct"]) < 0:
        raise ValueError("kelly_pct 不能为负数")
    if data.get("settled_at") is not None:
        _validate_iso_datetime_with_tz(data["settled_at"], "settled_at")
