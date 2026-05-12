from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from sqlalchemy.orm import Session


ODDS_COLUMN_CANDIDATES: dict[str, list[str]] = {
    "home": ["B365H", "PSH", "WHH", "VCH", "AvgH", "HomeOdds"],
    "draw": ["B365D", "PSD", "WHD", "VCD", "AvgD", "DrawOdds"],
    "away": ["B365A", "PSA", "WHA", "VCA", "AvgA", "AwayOdds"],
}


@dataclass(frozen=True)
class OddsImportStats:
    league_id: str
    bookmaker: str
    created_odds: int
    skipped_missing_match: int
    skipped_missing_odds: int
    skipped_existing: int


def _parse_match_date(raw: Any) -> datetime.date:
    parsed = pd.to_datetime(raw, dayfirst=True, utc=False, errors="raise")
    return parsed.date()


def _normalize_season(season: str | None, match_date: datetime.date) -> str:
    if season:
        return season
    year = match_date.year
    if match_date.month >= 7:
        return f"{year}-{str((year + 1) % 100).zfill(2)}"
    return f"{year - 1}-{str(year % 100).zfill(2)}"


def _pick_odds_columns(columns: set[str]) -> dict[str, str] | None:
    selected: dict[str, str] = {}
    for outcome, candidates in ODDS_COLUMN_CANDIDATES.items():
        match_col = next((name for name in candidates if name in columns), None)
        if match_col is None:
            return None
        selected[outcome] = match_col
    return selected


def _maybe_float(value: Any) -> float | None:
    if value is None or pd.isna(value):
        return None
    parsed = float(value)
    return parsed if parsed > 1.0 else None


def _calc_overround(home_odds: float, draw_odds: float, away_odds: float) -> float:
    return (1.0 / home_odds) + (1.0 / draw_odds) + (1.0 / away_odds)


def import_opening_odds_from_csv(
    session: Session,
    csv_path: str | Path,
    *,
    league_id: str = "E0",
    bookmaker: str = "bet365",
    season: str | None = None,
) -> OddsImportStats:
    from sqlalchemy import select

    from data.storage.models import Match, OddsOpening, Team

    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"文件不存在: {csv_path}")

    df = pd.read_csv(csv_path)
    required_cols = {"Date", "HomeTeam", "AwayTeam"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"缺少必要列: {sorted(missing)}")

    odds_columns = _pick_odds_columns(set(df.columns))
    if odds_columns is None:
        raise ValueError("未找到可用赔率列，需至少包含主/平/客三列")

    team_rows = session.scalars(select(Team).where(Team.league_id == league_id)).all()
    team_id_by_name = {team.name: team.id for team in team_rows}

    created_odds = 0
    skipped_missing_match = 0
    skipped_missing_odds = 0
    skipped_existing = 0

    for row in df.to_dict(orient="records"):
        match_date = _parse_match_date(row["Date"])
        season_value = _normalize_season(season, match_date)

        home_team_name = str(row["HomeTeam"]).strip()
        away_team_name = str(row["AwayTeam"]).strip()
        home_team_id = team_id_by_name.get(home_team_name)
        away_team_id = team_id_by_name.get(away_team_name)
        if home_team_id is None or away_team_id is None:
            skipped_missing_match += 1
            continue

        match_id = session.scalar(
            select(Match.id).where(
                Match.league_id == league_id,
                Match.season == season_value,
                Match.match_date == match_date,
                Match.home_team_id == home_team_id,
                Match.away_team_id == away_team_id,
            )
        )
        if match_id is None:
            skipped_missing_match += 1
            continue

        home_odds = _maybe_float(row.get(odds_columns["home"]))
        draw_odds = _maybe_float(row.get(odds_columns["draw"]))
        away_odds = _maybe_float(row.get(odds_columns["away"]))
        if home_odds is None or draw_odds is None or away_odds is None:
            skipped_missing_odds += 1
            continue

        exists = session.scalar(
            select(OddsOpening.id).where(
                OddsOpening.match_id == match_id,
                OddsOpening.bookmaker == bookmaker,
            )
        )
        if exists is not None:
            skipped_existing += 1
            continue

        overround = _calc_overround(home_odds, draw_odds, away_odds)
        odds_row = OddsOpening(
            match_id=match_id,
            bookmaker=bookmaker,
            odds_home=home_odds,
            odds_draw=draw_odds,
            odds_away=away_odds,
            overround=overround,
            recorded_at=datetime(match_date.year, match_date.month, match_date.day, tzinfo=timezone.utc),
        )
        session.add(odds_row)
        created_odds += 1

    session.commit()
    return OddsImportStats(
        league_id=league_id,
        bookmaker=bookmaker,
        created_odds=created_odds,
        skipped_missing_match=skipped_missing_match,
        skipped_missing_odds=skipped_missing_odds,
        skipped_existing=skipped_existing,
    )


def main() -> None:
    from data.storage.db import SessionLocal

    parser = argparse.ArgumentParser(description="从 football-data CSV 导入开盘赔率")
    parser.add_argument("csv_path", type=str, help="CSV 文件路径")
    parser.add_argument("--league-id", type=str, default="E0", help="联赛 ID，默认 E0")
    parser.add_argument("--season", type=str, default=None, help="赛季，如 2024-25")
    parser.add_argument("--bookmaker", type=str, default="bet365", help="写入 odds_opening 的 bookmaker 字段")
    args = parser.parse_args()

    with SessionLocal() as session:
        stats = import_opening_odds_from_csv(
            session,
            args.csv_path,
            league_id=args.league_id,
            bookmaker=args.bookmaker,
            season=args.season,
        )

    print(
        "赔率导入完成: "
        f"league={stats.league_id}, bookmaker={stats.bookmaker}, "
        f"新增赔率={stats.created_odds}, "
        f"缺失比赛={stats.skipped_missing_match}, "
        f"缺失赔率={stats.skipped_missing_odds}, "
        f"已存在跳过={stats.skipped_existing}"
    )


if __name__ == "__main__":
    main()
