from __future__ import annotations

import argparse
from datetime import date, datetime, timezone
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from sqlalchemy import and_, delete, select

from backtest.engine import enrich_rows_with_team_features
from data.storage.db import SessionLocal
from data.storage.models import Match, ModelParams, OddsOpening
from models.calibration import IsotonicThreeWayCalibrator
from models.dixon_coles import DixonColesModel


def _parse_csv_items(raw: str) -> list[str]:
    return [part.strip() for part in raw.split(",") if part.strip()]


def _build_features(row: dict) -> dict:
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


def _load_rows(league_id: str, seasons: list[str]) -> list[dict]:
    rows: list[dict] = []
    seen: set[int] = set()
    with SessionLocal() as session:
        stmt = (
            select(Match, OddsOpening)
            .join(OddsOpening, OddsOpening.match_id == Match.id)
            .where(and_(Match.league_id == league_id, Match.season.in_(seasons)))
        )
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


def _assert_season_order(train_seasons: list[str], val_season: str) -> None:
    """校验最后训练赛季早于验证赛季，防止数据泄漏。"""
    def _season_key(s: str) -> tuple[int, int]:
        parts = s.split("-")
        start = int(parts[0])
        end_suffix = int(parts[1]) if len(parts) > 1 else start
        end = (start // 100) * 100 + end_suffix if end_suffix < 100 else end_suffix
        return (start, end)

    last_train = max(train_seasons, key=_season_key)
    if _season_key(last_train) >= _season_key(val_season):
        raise ValueError(
            f"数据隔离校验失败：最后训练赛季 {last_train!r} 不早于验证赛季 {val_season!r}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 2 训练: Dixon-Coles + Isotonic")
    parser.add_argument("--league", default="E0", help="联赛 ID，例如 E0")
    parser.add_argument("--train-seasons", required=True, help="训练赛季，逗号分隔")
    parser.add_argument("--val-season", required=True, help="校准验证赛季")
    parser.add_argument("--train-until", default=None, help="可选训练截止日期 YYYY-MM-DD")
    args = parser.parse_args()

    train_seasons = _parse_csv_items(args.train_seasons)
    _assert_season_order(train_seasons, args.val_season)
    all_rows = _load_rows(args.league, train_seasons + [args.val_season])
    enrich_rows_with_team_features(all_rows)
    train_rows = [x for x in all_rows if x["season"] in train_seasons]
    val_rows = [x for x in all_rows if x["season"] == args.val_season and x.get("result") in {"H", "D", "A"}]
    if not train_rows:
        raise ValueError("训练集为空，请先导入历史数据与赔率")
    if not val_rows:
        raise ValueError("验证集为空，请检查 val-season 数据")

    if args.train_until:
        train_until = date.fromisoformat(args.train_until)
    else:
        train_until = max(x["match_date"] for x in train_rows)

    model = DixonColesModel()
    train_payload = []
    for row in train_rows:
        payload = dict(row)
        payload["cutoff_date"] = train_until.isoformat()
        train_payload.append(payload)
    model.fit(train_payload, args.league)

    calibrator = IsotonicThreeWayCalibrator()
    raw_outputs = []
    outcomes = []
    for row in sorted(val_rows, key=lambda x: x["match_date"]):
        features = _build_features(row)
        raw_outputs.append(model.predict(features))
        outcomes.append(str(row["result"]))
    calibrator.fit(raw_outputs, outcomes)

    params = {
        "model": model.get_params(),
        "calibrator": calibrator.get_params(),
        "train_seasons": train_seasons,
        "val_season": args.val_season,
    }

    with SessionLocal() as session:
        session.execute(
            delete(ModelParams).where(
                and_(
                    ModelParams.league_id == args.league,
                    ModelParams.model_version == model.model_version,
                    ModelParams.train_until == train_until,
                )
            )
        )
        session.add(
            ModelParams(
                league_id=args.league,
                model_version=model.model_version,
                trained_at=datetime.now(timezone.utc),
                train_until=train_until,
                params=params,
                brier_score=None,
                n_matches=len(train_rows),
            )
        )
        session.commit()

    print(
        "训练完成: "
        f"league={args.league}, model={model.model_version}, "
        f"train_matches={len(train_rows)}, val_matches={len(val_rows)}, "
        f"train_until={train_until.isoformat()}"
    )


if __name__ == "__main__":
    main()
