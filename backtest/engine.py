from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import date, datetime
from typing import Any

from sqlalchemy import Select, and_, select
from sqlalchemy.orm import Session

from data.processors.fatigue import fatigue_index as _calc_fatigue
from data.processors.form_score import form_score as _calc_form
from data.processors.momentum import momentum_score as _calc_momentum
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


def enrich_rows_with_team_features(rows: list[dict[str, Any]]) -> None:
    """为每条比赛行就地注入 form/momentum/fatigue 特征。

    使用当场比赛之前的历史记录动态计算，不引入未来数据。
    疲劳指数仅包含场次密度分量（travel_km / minutes_played 需外部数据，此处为 0）。
    injury_impact 依赖伤病报告，不在此计算，保持 0.0。
    """
    # 按球队 ID 建立比赛索引（升序排列）
    team_index: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        team_index[int(row["home_team_id"])].append(row)
        team_index[int(row["away_team_id"])].append(row)
    for matches in team_index.values():
        matches.sort(key=lambda r: _to_date(r["match_date"]))

    for row in rows:
        match_date = _to_date(row["match_date"])
        home_id = int(row["home_team_id"])
        away_id = int(row["away_team_id"])
        home_feats = _team_status(home_id, match_date, team_index[home_id])
        away_feats = _team_status(away_id, match_date, team_index[away_id])
        row.update({
            "home_form_5":   home_feats["form_5"],
            "home_form_10":  home_feats["form_10"],
            "home_momentum": home_feats["momentum"],
            "home_fatigue":  home_feats["fatigue"],
            "away_form_5":   away_feats["form_5"],
            "away_form_10":  away_feats["form_10"],
            "away_momentum": away_feats["momentum"],
            "away_fatigue":  away_feats["fatigue"],
        })


def _team_status(team_id: int, match_date: date, sorted_matches: list[dict[str, Any]]) -> dict[str, float]:
    """计算单支球队在 match_date 前的状态特征。"""
    # 取 match_date 之前的已完赛记录，最近优先
    prior = [
        r for r in reversed(sorted_matches)
        if _to_date(r["match_date"]) < match_date and r.get("result") in {"H", "D", "A"}
    ]

    if not prior:
        return {"form_5": 0.0, "form_10": 0.0, "momentum": 0.0, "fatigue": 0.0}

    # 积分序列（W=3, D=1, L=0）和距今天数
    pts_list: list[int] = []
    days_list: list[int] = []
    for r in prior[:10]:
        is_home = int(r["home_team_id"]) == team_id
        outcome = r["result"]
        pts = (3 if outcome == "H" else 1 if outcome == "D" else 0) if is_home \
              else (3 if outcome == "A" else 1 if outcome == "D" else 0)
        pts_list.append(pts)
        days_list.append((match_date - _to_date(r["match_date"])).days)

    form_5  = _calc_form(pts_list[:5],  days_list[:5])  if pts_list else 0.0
    form_10 = _calc_form(pts_list[:10], days_list[:10]) if pts_list else 0.0

    # 连胜/连败条纹（pts_list 最近优先）
    win_streak = 0
    for p in pts_list:
        if p == 3:
            win_streak += 1
        else:
            break
    loss_streak = 0
    for p in pts_list:
        if p == 0:
            loss_streak += 1
        else:
            break

    # 大败标记：最近一场失球差 >= 3
    r0 = prior[0]
    hg = r0.get("home_goals") or 0
    ag = r0.get("away_goals") or 0
    is_home_r0 = int(r0["home_team_id"]) == team_id
    big_loss = (ag - hg >= 3) if is_home_r0 else (hg - ag >= 3)

    mom = _calc_momentum(win_streak, loss_streak, big_loss)

    # 疲劳：近30天场次密度（travel_km / minutes_played 暂不可用，置0）
    matches_30d = sum(
        1 for r in sorted_matches
        if 0 < (match_date - _to_date(r["match_date"])).days <= 30
    )
    fat = _calc_fatigue(matches_30d, 0.0, 0.0)

    return {"form_5": form_5, "form_10": form_10, "momentum": mom, "fatigue": fat}


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


def _assert_season_order(train_seasons: list[str], val_season: str, test_season: str) -> None:
    """校验赛季顺序：最后一个训练赛季 < 验证赛季 < 测试赛季，防止数据泄漏。"""
    def _season_key(s: str) -> tuple[int, int]:
        # 格式 "YYYY-YY"，取起始年和终止年
        parts = s.split("-")
        start = int(parts[0])
        end_suffix = int(parts[1]) if len(parts) > 1 else start
        end = (start // 100) * 100 + end_suffix if end_suffix < 100 else end_suffix
        return (start, end)

    last_train = max(train_seasons, key=_season_key)
    if _season_key(last_train) >= _season_key(val_season):
        raise ValueError(
            f"数据隔离校验失败：最后训练赛季 {last_train!r} 不早于验证赛季 {val_season!r}，"
            "存在数据泄漏风险"
        )
    if _season_key(val_season) >= _season_key(test_season):
        raise ValueError(
            f"数据隔离校验失败：验证赛季 {val_season!r} 不早于测试赛季 {test_season!r}，"
            "存在数据泄漏风险"
        )


def run_backtest_from_rows(
    *,
    rows: list[dict[str, Any]],
    league_id: str,
    train_seasons: list[str],
    val_season: str,
    test_season: str,
) -> BacktestResult:
    _assert_season_order(train_seasons, val_season, test_season)
    enrich_rows_with_team_features(rows)
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
    seen: set[int] = set()
    rows: list[dict[str, Any]] = []
    for match, odds in session.execute(stmt).all():
        if match.id in seen:
            continue
        seen.add(match.id)
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
        "home_form_5":  float(row.get("home_form_5",  0.0)),
        "away_form_5":  float(row.get("away_form_5",  0.0)),
        "home_form_10": float(row.get("home_form_10", 0.0)),
        "away_form_10": float(row.get("away_form_10", 0.0)),
        "home_goals_scored_avg": 0.0,
        "home_goals_conceded_avg": 0.0,
        "away_goals_scored_avg": 0.0,
        "away_goals_conceded_avg": 0.0,
        "home_fatigue":       float(row.get("home_fatigue",  0.0)),
        "away_fatigue":       float(row.get("away_fatigue",  0.0)),
        "home_injury_impact": 0.0,
        "away_injury_impact": 0.0,
        "home_momentum": float(row.get("home_momentum", 0.0)),
        "away_momentum": float(row.get("away_momentum", 0.0)),
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
