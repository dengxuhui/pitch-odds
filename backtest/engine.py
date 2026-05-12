from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date, datetime
from typing import Any

from sqlalchemy import Select, and_, select
from sqlalchemy.orm import Session

from data.storage.models import Match, OddsOpening
from interfaces.contracts import MatchFeatures
from models.calibration import IsotonicThreeWayCalibrator
from models.dixon_coles import DixonColesModel


@dataclass
class BacktestPrediction:
    match_id: int
    league_id: str
    season: str
    match_date: str
    actual_outcome: str
    train_until: str
    p_home_raw: float
    p_draw_raw: float
    p_away_raw: float
    p_home: float
    p_draw: float
    p_away: float
    odds_home: float
    odds_draw: float
    odds_away: float


@dataclass
class BacktestResult:
    league_id: str
    model_version: str
    train_seasons: list[str]
    val_season: str
    test_season: str
    predictions: list[BacktestPrediction]


def run_backtest(
    league_id: str,
    train_seasons: list[str],
    val_season: str,
    test_season: str,
    *,
    session: Session,
) -> BacktestResult:
    rows = _load_match_rows(session, league_id, train_seasons + [val_season, test_season])
    return run_backtest_from_rows(
        rows=rows,
        league_id=league_id,
        train_seasons=train_seasons,
        val_season=val_season,
        test_season=test_season,
    )


def run_backtest_from_rows(
    *,
    rows: list[dict[str, Any]],
    league_id: str,
    train_seasons: list[str],
    val_season: str,
    test_season: str,
) -> BacktestResult:
    train_rows = _rows_for_seasons(rows, train_seasons)
    val_rows = _rows_for_seasons(rows, [val_season])
    test_rows = _rows_for_seasons(rows, [test_season])

    if not train_rows or not val_rows or not test_rows:
        raise ValueError("训练/验证/测试数据不完整")

    model = DixonColesModel()
    train_until = max(_to_date(x["match_date"]) for x in train_rows)
    model.fit(_attach_cutoff(train_rows, train_until), league_id)

    calibrator = IsotonicThreeWayCalibrator()
    val_raw = []
    val_outcomes = []
    for row in sorted(val_rows, key=lambda x: _to_date(x["match_date"])):
        if row.get("result") not in {"H", "D", "A"}:
            continue
        features = _build_features(row)
        raw = model.predict(features)
        val_raw.append(raw)
        val_outcomes.append(str(row["result"]))
    if not val_raw:
        raise ValueError("验证集没有可用于校准的完赛样本")
    calibrator.fit(val_raw, val_outcomes)

    predictions: list[BacktestPrediction] = []
    for row in sorted(test_rows, key=lambda x: _to_date(x["match_date"])):
        if row.get("result") not in {"H", "D", "A"}:
            continue
        features = _build_features(row)
        raw = model.predict(features)
        calibrated = calibrator.calibrate(raw, features)
        predictions.append(
            BacktestPrediction(
                match_id=int(row["match_id"]),
                league_id=league_id,
                season=str(row["season"]),
                match_date=str(row["match_date"]),
                actual_outcome=str(row["result"]),
                train_until=train_until.isoformat(),
                p_home_raw=float(raw["p_home_raw"]),
                p_draw_raw=float(raw["p_draw_raw"]),
                p_away_raw=float(raw["p_away_raw"]),
                p_home=float(calibrated["p_home"]),
                p_draw=float(calibrated["p_draw"]),
                p_away=float(calibrated["p_away"]),
                odds_home=float(row["odds_home"]),
                odds_draw=float(row["odds_draw"]),
                odds_away=float(row["odds_away"]),
            )
        )

    return BacktestResult(
        league_id=league_id,
        model_version=model.model_version,
        train_seasons=train_seasons,
        val_season=val_season,
        test_season=test_season,
        predictions=predictions,
    )


def serialize_backtest_result(result: BacktestResult) -> dict[str, Any]:
    return {
        "league_id": result.league_id,
        "model_version": result.model_version,
        "train_seasons": result.train_seasons,
        "val_season": result.val_season,
        "test_season": result.test_season,
        "predictions": [asdict(item) for item in result.predictions],
    }


def _load_match_rows(session: Session, league_id: str, seasons: list[str]) -> list[dict[str, Any]]:
    stmt: Select[tuple[Match, OddsOpening]] = (
        select(Match, OddsOpening)
        .join(OddsOpening, OddsOpening.match_id == Match.id)
        .where(
            and_(
                Match.league_id == league_id,
                Match.season.in_(seasons),
            )
        )
    )
    rows: list[dict[str, Any]] = []
    for match, odds in session.execute(stmt).all():
        rows.append(
            {
                "match_id": match.id,
                "league_id": match.league_id,
                "season": match.season,
                "match_date": match.match_date,
                "home_team_id": match.home_team_id,
                "away_team_id": match.away_team_id,
                "home_goals": match.home_goals,
                "away_goals": match.away_goals,
                "result": match.result,
                "odds_home": float(odds.odds_home),
                "odds_draw": float(odds.odds_draw),
                "odds_away": float(odds.odds_away),
            }
        )
    return rows


def _rows_for_seasons(rows: list[dict[str, Any]], seasons: list[str]) -> list[dict[str, Any]]:
    season_set = set(seasons)
    return [x for x in rows if str(x["season"]) in season_set]


def _attach_cutoff(rows: list[dict[str, Any]], cutoff: date) -> list[dict[str, Any]]:
    enriched = []
    for item in rows:
        data = dict(item)
        data["cutoff_date"] = cutoff.isoformat()
        enriched.append(data)
    return enriched


def _build_features(row: dict[str, Any]) -> MatchFeatures:
    odds_home = float(row["odds_home"])
    odds_draw = float(row["odds_draw"])
    odds_away = float(row["odds_away"])
    overround = (1.0 / odds_home) + (1.0 / odds_draw) + (1.0 / odds_away)
    return {
        "match_id": int(row["match_id"]),
        "league_id": str(row["league_id"]),
        "match_date": str(row["match_date"]),
        "match_week": 0,
        "home_team_id": int(row["home_team_id"]),
        "away_team_id": int(row["away_team_id"]),
        "home_form_5": 0.0,
        "away_form_5": 0.0,
        "home_form_10": 0.0,
        "away_form_10": 0.0,
        "home_goals_scored_avg": 0.0,
        "home_goals_conceded_avg": 0.0,
        "away_goals_scored_avg": 0.0,
        "away_goals_conceded_avg": 0.0,
        "home_fatigue": 0.0,
        "away_fatigue": 0.0,
        "home_injury_impact": 0.0,
        "away_injury_impact": 0.0,
        "home_momentum": 0.0,
        "away_momentum": 0.0,
        "days_rest_home": 7,
        "days_rest_away": 7,
        "odds_home": odds_home,
        "odds_draw": odds_draw,
        "odds_away": odds_away,
        "p_implied_home": (1.0 / odds_home) / overround,
        "p_implied_draw": (1.0 / odds_draw) / overround,
        "p_implied_away": (1.0 / odds_away) / overround,
        "odds_drift_home": 0.0,
        "smart_money_flag": False,
        "exclude_flag": False,
    }


def _to_date(value: Any) -> date:
    if isinstance(value, date):
        return value
    if isinstance(value, datetime):
        return value.date()
    return date.fromisoformat(str(value))
