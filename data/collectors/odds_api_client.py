"""odds_api_client.py — The Odds API v4 实时赔率拉取客户端

每次调用拉取五大联赛的最新赔率快照，写入 OddsSnapshot 表，
并触发赔率异常检测（odds_anomaly.py）。

密钥从环境变量 ODDS_API_KEY 读取（不写死在代码中）。
注册地址：https://the-odds-api.com/（免费额度 500 credits/月）

每次调用本模块消耗的 credits：
    1 credit × 请求的联赛数量（默认 5 大联赛 = 5 credits）

用法：
    python3 -m data.collectors.odds_api_client --leagues E0 SP1 D1 I1 F1
"""
from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config import get_odds_api_key
from data.processors.odds_anomaly import detect_odds_anomaly
from data.storage.db import SessionLocal
from data.storage.models import Match, OddsAnomaly, OddsSnapshot, Team

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# 常量
# ──────────────────────────────────────────────

BASE_URL = "https://api.the-odds-api.com/v4/sports/{sport_key}/odds/"

# 联赛 ID → The Odds API sport_key
LEAGUE_SPORT_KEY: dict[str, str] = {
    "E0":  "soccer_epl",
    "SP1": "soccer_spain_la_liga",
    "D1":  "soccer_germany_bundesliga",
    "I1":  "soccer_italy_serie_a",
    "F1":  "soccer_france_ligue_one",
}

# 拉取欧洲赔率区域，支持 bet365/Pinnacle 等欧盘庄家
REGIONS = "eu"
MARKETS = "h2h"       # 1X2 主平客
ODDS_FORMAT = "decimal"
DATE_FORMAT = "iso"

REQUEST_TIMEOUT = 20


# ──────────────────────────────────────────────
# API 调用
# ──────────────────────────────────────────────

def _fetch_odds_from_api(sport_key: str, api_key: str) -> list[dict]:
    """调用 The Odds API，返回原始 JSON 列表。"""
    url = BASE_URL.format(sport_key=sport_key)
    params = {
        "apiKey": api_key,
        "regions": REGIONS,
        "markets": MARKETS,
        "oddsFormat": ODDS_FORMAT,
        "dateFormat": DATE_FORMAT,
    }
    resp = requests.get(url, params=params, timeout=REQUEST_TIMEOUT)

    remaining = resp.headers.get("x-requests-remaining", "?")
    used = resp.headers.get("x-requests-used", "?")
    logger.info(f"  [{sport_key}] credits 剩余：{remaining}，已用：{used}")

    if resp.status_code == 401:
        raise RuntimeError("ODDS_API_KEY 无效，请检查 .env 文件中的配置")
    if resp.status_code == 429:
        raise RuntimeError("The Odds API 免费额度已耗尽（500 credits/月）")
    resp.raise_for_status()

    return resp.json()


def _extract_best_odds(bookmakers: list[dict]) -> tuple[float, float, float] | None:
    """从多个博彩公司中提取最优赔率（取均值）。"""
    home_prices, draw_prices, away_prices = [], [], []

    for bm in bookmakers:
        for market in bm.get("markets", []):
            if market.get("key") != "h2h":
                continue
            outcomes = {o["name"]: o["price"] for o in market.get("outcomes", [])}
            # h2h 格式：home_team, Draw, away_team
            values = list(outcomes.values())
            names = list(outcomes.keys())
            if len(values) == 3:
                # 找 Draw
                draw_idx = next(
                    (i for i, n in enumerate(names) if n.lower() == "draw"), None
                )
                if draw_idx is None:
                    continue
                home_away = [v for i, v in enumerate(values) if i != draw_idx]
                if len(home_away) == 2:
                    home_prices.append(home_away[0])
                    draw_prices.append(values[draw_idx])
                    away_prices.append(home_away[1])

    if not home_prices:
        return None

    avg = lambda lst: round(sum(lst) / len(lst), 3)
    return avg(home_prices), avg(draw_prices), avg(away_prices)


# ──────────────────────────────────────────────
# 匹配数据库比赛
# ──────────────────────────────────────────────

def _match_to_db_id(
    session: Any,
    league_id: str,
    home_team_api: str,
    away_team_api: str,
    commence_time: datetime,
) -> int | None:
    """将 API 返回的球队名和时间映射到数据库 Match.id。
    使用宽松模糊匹配（包含关系），适应球队名称差异。
    """
    from sqlalchemy import select

    match_date = commence_time.date()

    # 拉取当天该联赛的全部比赛
    rows = session.execute(
        select(Match, Team, Team)
        .join(Team, Match.home_team_id == Team.id, isouter=False)
        .where(
            Match.league_id == league_id,
            Match.match_date == match_date,
        )
    ).all()

    # 直接用 SQLAlchemy 分别查主客队
    matches = session.execute(
        select(Match).where(
            Match.league_id == league_id,
            Match.match_date == match_date,
        )
    ).scalars().all()

    if not matches:
        return None

    # 构建球队 id → name 映射
    teams = {
        t.id: t.name
        for t in session.execute(
            select(Team).where(Team.league_id == league_id)
        ).scalars().all()
    }

    def _fuzzy(api_name: str, db_name: str) -> bool:
        a, b = api_name.lower(), db_name.lower()
        return a == b or a in b or b in a

    for m in matches:
        home_db = teams.get(m.home_team_id, "")
        away_db = teams.get(m.away_team_id, "")
        if _fuzzy(home_team_api, home_db) and _fuzzy(away_team_api, away_db):
            return m.id

    return None


# ──────────────────────────────────────────────
# 快照写入 + 异常检测
# ──────────────────────────────────────────────

def _save_snapshot_and_detect(
    session: Any,
    match_id: int,
    odds_home: float,
    odds_draw: float,
    odds_away: float,
    snapshot_at: datetime,
    commence_time: datetime,
) -> None:
    """写入赔率快照，并对该场次历史序列运行异常检测。"""
    from sqlalchemy import select

    hours_to_kick = (commence_time - snapshot_at).total_seconds() / 3600

    # 写快照
    session.add(OddsSnapshot(
        match_id=match_id,
        bookmaker="the_odds_api_avg",
        odds_home=odds_home,
        odds_draw=odds_draw,
        odds_away=odds_away,
        overround=round(1 / odds_home + 1 / odds_draw + 1 / odds_away, 4),
        snapshot_at=snapshot_at,
        hours_to_kick=round(hours_to_kick, 2),
    ))
    session.flush()

    # 拉取该场次全部历史主场赔率序列（时间升序）做异常检测
    history = session.scalars(
        select(OddsSnapshot.odds_home)
        .where(OddsSnapshot.match_id == match_id)
        .order_by(OddsSnapshot.snapshot_at)
    ).all()

    series = [float(h) for h in history]
    result = detect_odds_anomaly(series)

    if result["alert_level"] != "NORMAL":
        # 更新或新建异常记录
        existing = session.scalar(
            select(OddsAnomaly).where(OddsAnomaly.match_id == match_id)
        )
        if existing:
            existing.alert_level = result["alert_level"]
            existing.total_drift_pct = result["total_drift_pct"]
            existing.exclude_flag = result["exclude_from_parlay"]
            existing.detected_at = snapshot_at
        else:
            session.add(OddsAnomaly(
                match_id=match_id,
                alert_level=result["alert_level"],
                anomaly_type="smart_money" if result["smart_money"] else "drift",
                total_drift_pct=result["total_drift_pct"],
                exclude_flag=result["exclude_from_parlay"],
                detected_at=snapshot_at,
            ))

        logger.warning(
            f"  ⚠ match_id={match_id}  异常级别={result['alert_level']}  "
            f"漂移={result['total_drift_pct']:.1f}%  "
            f"{'→ 已标记排除串场' if result['exclude_from_parlay'] else ''}"
        )


# ──────────────────────────────────────────────
# 主入口
# ──────────────────────────────────────────────

def pull_snapshots(league_ids: list[str] | None = None) -> dict[str, int]:
    """拉取指定联赛的实时赔率快照，写入数据库。

    Args:
        league_ids: 联赛 ID 列表，默认五大联赛。

    Returns:
        dict {league_id: 成功写入快照数}
    """
    if league_ids is None:
        league_ids = list(LEAGUE_SPORT_KEY.keys())

    api_key = get_odds_api_key()   # 从环境变量读取，不在此处硬编码
    snapshot_at = datetime.now(timezone.utc)
    saved_counts: dict[str, int] = {}

    with SessionLocal() as session:
        for league_id in league_ids:
            sport_key = LEAGUE_SPORT_KEY.get(league_id)
            if not sport_key:
                logger.warning(f"未知联赛 {league_id}，跳过")
                continue

            logger.info(f"拉取 {league_id} ({sport_key}) 赔率...")
            try:
                games = _fetch_odds_from_api(sport_key, api_key)
            except Exception as exc:
                logger.error(f"  拉取失败：{exc}")
                saved_counts[league_id] = 0
                continue

            count = 0
            for game in games:
                try:
                    commence_iso = game.get("commence_time", "")
                    commence_time = datetime.fromisoformat(
                        commence_iso.replace("Z", "+00:00")
                    )
                    # 只处理未来的比赛
                    if commence_time <= snapshot_at:
                        continue

                    home_team = game.get("home_team", "")
                    away_team = game.get("away_team", "")
                    bookmakers = game.get("bookmakers", [])

                    odds = _extract_best_odds(bookmakers)
                    if odds is None:
                        continue

                    odds_home, odds_draw, odds_away = odds

                    match_id = _match_to_db_id(
                        session, league_id, home_team, away_team, commence_time
                    )
                    if match_id is None:
                        logger.debug(
                            f"  数据库中未找到比赛：{home_team} vs {away_team} "
                            f"({commence_time.date()})，请先运行 fixtures_fetcher"
                        )
                        continue

                    _save_snapshot_and_detect(
                        session,
                        match_id,
                        odds_home,
                        odds_draw,
                        odds_away,
                        snapshot_at,
                        commence_time,
                    )
                    count += 1

                except Exception as exc:
                    logger.warning(f"  处理单场比赛出错，跳过：{exc}")
                    continue

            session.commit()
            saved_counts[league_id] = count
            logger.info(f"  {league_id}：写入 {count} 条快照")

    return saved_counts


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="拉取 The Odds API 实时赔率快照")
    parser.add_argument(
        "--leagues",
        nargs="+",
        default=list(LEAGUE_SPORT_KEY.keys()),
        help="联赛 ID 列表（默认全部）：E0 SP1 D1 I1 F1",
    )
    args = parser.parse_args()

    counts = pull_snapshots(args.leagues)

    print("\n【赔率快照结果】")
    for league_id, n in counts.items():
        print(f"  {league_id}：写入 {n} 条快照")
    total = sum(counts.values())
    print(f"  合计：{total} 条")


if __name__ == "__main__":
    main()
