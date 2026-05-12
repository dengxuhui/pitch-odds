from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
import argparse

import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session

from data.storage.db import SessionLocal
from data.storage.models import League, Match, Team


DEFAULT_LEAGUES: dict[str, tuple[str, str]] = {
    "E0": ("English Premier League", "England"),
    "SP1": ("LaLiga", "Spain"),
    "D1": ("Bundesliga", "Germany"),
    "I1": ("Serie A", "Italy"),
    "F1": ("Ligue 1", "France"),
}


@dataclass(frozen=True)
class ImportStats:
    league_id: str
    season: str
    created_teams: int
    created_matches: int
    skipped_matches: int


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


def _result_from_score(home_goals: int | None, away_goals: int | None) -> str | None:
    if home_goals is None or away_goals is None:
        return None
    if home_goals > away_goals:
        return "H"
    if home_goals < away_goals:
        return "A"
    return "D"


def _maybe_int(value: Any) -> int | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    if pd.isna(value):
        return None
    return int(value)


def _get_or_create_team(session: Session, league_id: str, team_name: str) -> tuple[Team, bool]:
    team = session.scalar(select(Team).where(Team.league_id == league_id, Team.name == team_name))
    if team is not None:
        return team, False
    team = Team(league_id=league_id, name=team_name)
    session.add(team)
    session.flush()
    return team, True


def _ensure_league(session: Session, league_id: str) -> None:
    existing = session.get(League, league_id)
    if existing is not None:
        return
    league_name, country = DEFAULT_LEAGUES.get(league_id, (league_id, "Unknown"))
    session.add(League(id=league_id, name=league_name, country=country))


def import_football_data_csv(
    session: Session,
    csv_path: str | Path,
    *,
    league_id: str = "E0",
    season: str | None = None,
) -> ImportStats:
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"文件不存在: {csv_path}")

    df = pd.read_csv(csv_path)
    required_cols = {"Date", "HomeTeam", "AwayTeam"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"缺少必要列: {sorted(missing)}")

    _ensure_league(session, league_id)
    created_teams = 0
    created_matches = 0
    skipped_matches = 0

    for row in df.to_dict(orient="records"):
        match_date = _parse_match_date(row["Date"])
        season_value = _normalize_season(season, match_date)

        home_team, home_created = _get_or_create_team(session, league_id, str(row["HomeTeam"]).strip())
        away_team, away_created = _get_or_create_team(session, league_id, str(row["AwayTeam"]).strip())
        created_teams += int(home_created) + int(away_created)

        exists_stmt = select(Match.id).where(
            Match.league_id == league_id,
            Match.season == season_value,
            Match.match_date == match_date,
            Match.home_team_id == home_team.id,
            Match.away_team_id == away_team.id,
        )
        if session.scalar(exists_stmt) is not None:
            skipped_matches += 1
            continue

        home_goals = _maybe_int(row.get("FTHG"))
        away_goals = _maybe_int(row.get("FTAG"))
        result = row.get("FTR") or _result_from_score(home_goals, away_goals)
        if result is not None:
            result = str(result).strip().upper()

        match = Match(
            league_id=league_id,
            season=season_value,
            match_date=match_date,
            home_team_id=home_team.id,
            away_team_id=away_team.id,
            home_goals=home_goals,
            away_goals=away_goals,
            result=result,
            home_shots=_maybe_int(row.get("HS")),
            away_shots=_maybe_int(row.get("AS")),
            home_shots_on=_maybe_int(row.get("HST")),
            away_shots_on=_maybe_int(row.get("AST")),
            home_corners=_maybe_int(row.get("HC")),
            away_corners=_maybe_int(row.get("AC")),
            home_yellow=_maybe_int(row.get("HY")),
            away_yellow=_maybe_int(row.get("AY")),
            home_red=_maybe_int(row.get("HR")),
            away_red=_maybe_int(row.get("AR")),
        )
        session.add(match)
        created_matches += 1

    session.commit()
    return ImportStats(
        league_id=league_id,
        season=season or "multiple",
        created_teams=created_teams,
        created_matches=created_matches,
        skipped_matches=skipped_matches,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="导入 football-data CSV 到数据库")
    parser.add_argument("csv_path", type=str, help="CSV 文件路径")
    parser.add_argument("--league-id", type=str, default="E0", help="联赛 ID，默认 E0")
    parser.add_argument("--season", type=str, default=None, help="赛季，如 2024-25")
    args = parser.parse_args()

    with SessionLocal() as session:
        stats = import_football_data_csv(
            session,
            args.csv_path,
            league_id=args.league_id,
            season=args.season,
        )

    print(
        f"导入完成: league={stats.league_id}, season={stats.season}, "
        f"新增球队={stats.created_teams}, 新增比赛={stats.created_matches}, 跳过比赛={stats.skipped_matches}"
    )


if __name__ == "__main__":
    main()
