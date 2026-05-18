"""fixtures_fetcher.py — 从 football-data.co.uk 自动拉取即将进行的赛程

football-data.co.uk 每周更新两次赛程文件（周五下午 + 周二下午），
CSV 直接从公开 URL 下载，无需登录或 API key。

支持的联赛 ID（与项目其他模块一致）：
    E0  → 英超 (Premier League)
    SP1 → 西甲 (La Liga)
    D1  → 德甲 (Bundesliga)
    I1  → 意甲 (Serie A)
    F1  → 法甲 (Ligue 1)

用法：
    python3 -m data.collectors.fixtures_fetcher --leagues E0 SP1 D1 I1 F1
"""
from __future__ import annotations

import argparse
import io
import logging
import sys
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import requests

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from data.storage.db import SessionLocal
from data.storage.models import League, Match, OddsOpening, Team

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# 常量
# ──────────────────────────────────────────────

# football-data.co.uk 即将比赛赛程 CSV（包含所有主要联赛，每周更新）
FIXTURES_URL = "https://www.football-data.co.uk/fixtures.csv"

# 联赛 ID → (CSV 中的 Div 字段值, 联赛中文名)
LEAGUE_DIV_MAP: dict[str, tuple[str, str]] = {
    "E0":  ("E0",  "英超 Premier League"),
    "SP1": ("SP1", "西甲 La Liga"),
    "D1":  ("D1",  "德甲 Bundesliga"),
    "I1":  ("I1",  "意甲 Serie A"),
    "F1":  ("F1",  "法甲 Ligue 1"),
}

# 赔率列候选（与 odds.py 保持一致）
ODDS_COLUMN_CANDIDATES: dict[str, list[str]] = {
    "home": ["B365H", "PSH", "WHH", "VCH", "AvgH"],
    "draw": ["B365D", "PSD", "WHD", "VCD", "AvgD"],
    "away": ["B365A", "PSA", "WHA", "VCA", "AvgA"],
}

REQUEST_TIMEOUT = 30  # 秒


# ──────────────────────────────────────────────
# 数据结构
# ──────────────────────────────────────────────

@dataclass(frozen=True)
class FetchResult:
    league_id: str
    fetched_matches: int      # 从 CSV 解析到的场次
    new_matches: int          # 写入数据库的新场次
    skipped_matches: int      # 已存在或数据不完整跳过的场次
    new_odds: int             # 写入的开盘赔率条数


# ──────────────────────────────────────────────
# 工具函数
# ──────────────────────────────────────────────

def _parse_date(raw: Any) -> date | None:
    try:
        return pd.to_datetime(raw, dayfirst=True, utc=False).date()
    except Exception:
        return None


def _maybe_float(value: Any) -> float | None:
    try:
        v = float(value)
        return v if v > 1.0 else None
    except (TypeError, ValueError):
        return None


def _normalize_season(match_date: date) -> str:
    year = match_date.year
    if match_date.month >= 7:
        return f"{year}-{str((year + 1) % 100).zfill(2)}"
    return f"{year - 1}-{str(year % 100).zfill(2)}"


def _pick_odds_columns(columns: set[str]) -> dict[str, str] | None:
    selected: dict[str, str] = {}
    for outcome, candidates in ODDS_COLUMN_CANDIDATES.items():
        col = next((c for c in candidates if c in columns), None)
        if col is None:
            return None
        selected[outcome] = col
    return selected


# ──────────────────────────────────────────────
# 下载
# ──────────────────────────────────────────────

def _download_fixtures_csv() -> pd.DataFrame:
    """从 football-data.co.uk 下载赛程 CSV，返回 DataFrame。"""
    logger.info(f"正在下载赛程数据：{FIXTURES_URL}")
    resp = requests.get(FIXTURES_URL, timeout=REQUEST_TIMEOUT)
    resp.raise_for_status()
    # football-data.co.uk 使用 latin-1 编码
    df = pd.read_csv(io.StringIO(resp.content.decode("latin-1")))
    logger.info(f"下载完成，共 {len(df)} 行")
    return df


# ──────────────────────────────────────────────
# 导入逻辑
# ──────────────────────────────────────────────

def fetch_and_import(
    league_ids: list[str] | None = None,
    *,
    bookmaker: str = "bet365",
) -> list[FetchResult]:
    """下载赛程 CSV 并将指定联赛的即将比赛写入数据库。

    Args:
        league_ids: 要导入的联赛 ID 列表，默认全部五大联赛。
        bookmaker:  赔率来源标签，写入 OddsOpening 表时使用。

    Returns:
        每个联赛的 FetchResult。
    """
    if league_ids is None:
        league_ids = list(LEAGUE_DIV_MAP.keys())

    df_all = _download_fixtures_csv()
    results: list[FetchResult] = []

    with SessionLocal() as session:
        # 确保联赛记录存在
        _ensure_leagues(session)
        session.flush()

        for league_id in league_ids:
            if league_id not in LEAGUE_DIV_MAP:
                logger.warning(f"未知联赛 ID：{league_id}，跳过")
                continue

            div_code, league_name = LEAGUE_DIV_MAP[league_id]
            logger.info(f"处理联赛：{league_name} ({league_id})")

            # 过滤当前联赛的行
            if "Div" not in df_all.columns:
                logger.error("CSV 中缺少 'Div' 列，无法区分联赛")
                continue

            df = df_all[df_all["Div"].astype(str).str.strip() == div_code].copy()
            if df.empty:
                logger.info(f"  {league_id}：CSV 中暂无赛程数据")
                results.append(FetchResult(league_id, 0, 0, 0, 0))
                continue

            result = _import_league_fixtures(session, df, league_id, bookmaker)
            results.append(result)
            logger.info(
                f"  {league_id}：解析 {result.fetched_matches} 场，"
                f"新增 {result.new_matches} 场，跳过 {result.skipped_matches} 场，"
                f"赔率 {result.new_odds} 条"
            )

        session.commit()

    return results


def _ensure_leagues(session: Any) -> None:
    """确保五大联赛记录存在于 leagues 表。"""
    from sqlalchemy import select

    existing = {row.id for row in session.scalars(select(League)).all()}
    defaults = {
        "E0":  ("English Premier League", "England"),
        "SP1": ("LaLiga",                 "Spain"),
        "D1":  ("Bundesliga",             "Germany"),
        "I1":  ("Serie A",                "Italy"),
        "F1":  ("Ligue 1",                "France"),
    }
    for lid, (name, country) in defaults.items():
        if lid not in existing:
            session.add(League(id=lid, name=name, country=country))


def _import_league_fixtures(
    session: Any,
    df: pd.DataFrame,
    league_id: str,
    bookmaker: str,
) -> FetchResult:
    from sqlalchemy import select

    cols = set(df.columns)
    odds_cols = _pick_odds_columns(cols)

    # 建立球队名 → ID 的查找表
    team_rows = session.scalars(
        select(Team).where(Team.league_id == league_id)
    ).all()
    team_id_by_name: dict[str, int] = {t.name: t.id for t in team_rows}

    fetched = new_matches = skipped = new_odds = 0

    for _, row in df.iterrows():
        fetched += 1

        match_date = _parse_date(row.get("Date"))
        if match_date is None or match_date < date.today():
            skipped += 1
            continue

        home_name = str(row.get("HomeTeam", "")).strip()
        away_name = str(row.get("AwayTeam", "")).strip()
        if not home_name or not away_name:
            skipped += 1
            continue

        # 球队不存在时自动创建（未来新赛季可能有新球队）
        home_id = _get_or_create_team(session, team_id_by_name, league_id, home_name)
        away_id = _get_or_create_team(session, team_id_by_name, league_id, away_name)

        season = _normalize_season(match_date)

        # 查找或创建 Match 记录
        existing_match = session.scalar(
            select(Match).where(
                Match.league_id == league_id,
                Match.season == season,
                Match.match_date == match_date,
                Match.home_team_id == home_id,
                Match.away_team_id == away_id,
            )
        )

        if existing_match is None:
            match = Match(
                league_id=league_id,
                season=season,
                match_date=match_date,
                home_team_id=home_id,
                away_team_id=away_id,
            )
            session.add(match)
            session.flush()  # 获取 match.id
            new_matches += 1
        else:
            match = existing_match

        # 写入开盘赔率（若有且尚未存在）
        if odds_cols is not None:
            odds_home = _maybe_float(row.get(odds_cols["home"]))
            odds_draw = _maybe_float(row.get(odds_cols["draw"]))
            odds_away = _maybe_float(row.get(odds_cols["away"]))

            if all(v is not None for v in (odds_home, odds_draw, odds_away)):
                existing_odds = session.scalar(
                    select(OddsOpening).where(
                        OddsOpening.match_id == match.id,
                        OddsOpening.bookmaker == bookmaker,
                    )
                )
                if existing_odds is None:
                    overround = 1 / odds_home + 1 / odds_draw + 1 / odds_away  # type: ignore[operator]
                    session.add(OddsOpening(
                        match_id=match.id,
                        bookmaker=bookmaker,
                        odds_home=odds_home,
                        odds_draw=odds_draw,
                        odds_away=odds_away,
                        overround=round(overround, 4),
                        recorded_at=datetime.now(timezone.utc),
                    ))
                    new_odds += 1

    return FetchResult(
        league_id=league_id,
        fetched_matches=fetched,
        new_matches=new_matches,
        skipped_matches=skipped,
        new_odds=new_odds,
    )


def _get_or_create_team(
    session: Any,
    cache: dict[str, int],
    league_id: str,
    name: str,
) -> int:
    if name in cache:
        return cache[name]
    team = Team(league_id=league_id, name=name)
    session.add(team)
    session.flush()
    cache[name] = team.id
    return team.id


# ──────────────────────────────────────────────
# CLI 入口
# ──────────────────────────────────────────────

def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="从 football-data.co.uk 拉取即将进行的比赛赛程")
    parser.add_argument(
        "--leagues",
        nargs="+",
        default=list(LEAGUE_DIV_MAP.keys()),
        help="联赛 ID 列表（默认全部五大联赛）：E0 SP1 D1 I1 F1",
    )
    parser.add_argument(
        "--bookmaker",
        default="bet365",
        help="赔率来源标签，默认 bet365",
    )
    args = parser.parse_args()

    results = fetch_and_import(args.leagues, bookmaker=args.bookmaker)

    print("\n【赛程拉取结果】")
    total_new = 0
    for r in results:
        print(f"  {r.league_id}: 新增 {r.new_matches} 场，赔率 {r.new_odds} 条，跳过 {r.skipped_matches} 条")
        total_new += r.new_matches
    print(f"  合计新增比赛：{total_new} 场")


if __name__ == "__main__":
    main()
