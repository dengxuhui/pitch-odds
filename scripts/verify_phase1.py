from __future__ import annotations

import argparse
import sys

from sqlalchemy import func, inspect, select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from data.storage.db import SessionLocal, engine
from data.storage.models import League, Match, Team


REQUIRED_TABLES = {
    "leagues",
    "teams",
    "matches",
    "odds_opening",
    "odds_snapshots",
    "odds_anomalies",
    "team_status",
    "player_injuries",
    "model_predictions",
    "parlay_plans",
    "bet_results",
}


def verify_schema() -> tuple[bool, list[str]]:
    inspector = inspect(engine)
    existing = set(inspector.get_table_names())
    missing = sorted(REQUIRED_TABLES - existing)
    if missing:
        return False, [f"缺少数据表: {', '.join(missing)}"]
    return True, ["数据库表结构检查通过"]


def verify_data(session: Session, league_id: str, strict: bool) -> tuple[bool, list[str]]:
    messages: list[str] = []
    ok = True

    league = session.get(League, league_id)
    if league is None:
        messages.append(f"联赛不存在: {league_id}")
        return (False, messages) if strict else (True, messages)

    team_count = session.scalar(select(func.count()).select_from(Team).where(Team.league_id == league_id)) or 0
    match_count = session.scalar(select(func.count()).select_from(Match).where(Match.league_id == league_id)) or 0

    messages.append(f"联赛: {league_id}")
    messages.append(f"球队数: {team_count}")
    messages.append(f"比赛数: {match_count}")

    if strict and (team_count <= 0 or match_count <= 0):
        ok = False
        messages.append("严格模式失败: 球队数或比赛数为 0")

    return ok, messages


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 1 验收检查")
    parser.add_argument("--league-id", default="E0", help="联赛 ID，默认 E0")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="严格模式：要求联赛存在且比赛/球队数据都大于 0",
    )
    args = parser.parse_args()

    try:
        schema_ok, schema_msgs = verify_schema()
        with SessionLocal() as session:
            data_ok, data_msgs = verify_data(session, args.league_id, args.strict)
    except SQLAlchemyError as exc:
        print(f"数据库连接或查询失败: {exc}")
        sys.exit(1)

    for msg in schema_msgs + data_msgs:
        print(msg)

    if schema_ok and data_ok:
        print("Phase 1 验收通过")
        sys.exit(0)

    print("Phase 1 验收未通过")
    sys.exit(1)


if __name__ == "__main__":
    main()
