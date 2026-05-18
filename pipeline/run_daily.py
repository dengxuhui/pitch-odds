"""run_daily.py — 日常完整流程编排器

完整执行顺序：
    1. 从 football-data.co.uk 拉取即将进行的赛程
    2. 启动定期轮询循环：每隔 N 小时从 The Odds API 拉取赔率快照
    3. 每次拉取后自动运行赔率异常检测
    4. 到达截止时间（首场开赛前 N 分钟）或收到 Ctrl+C 时停止轮询
    5. 输出最终预测推荐（正期望场次 + 三层串场方案）

用法示例：
    # 默认：全部五大联赛，预算 1000 元，每 6 小时拉一次赔率，开赛前 60 分钟截止
    python3 -m pipeline.run_daily

    # 指定联赛和参数
    python3 -m pipeline.run_daily \\
        --leagues E0 SP1 \\
        --budget 2000 \\
        --poll-interval 4 \\
        --cutoff-minutes 30 \\
        --output-dir reports/

停止方式：
    - 自动：当所有待预测比赛的最近开赛时间 ≤ cutoff_minutes 时自动停止
    - 手动：按 Ctrl+C（SIGINT），程序立即停止并输出当前最新预测结果
"""
from __future__ import annotations

import argparse
import logging
import signal
import sys
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from data.collectors.fixtures_fetcher import fetch_and_import
from data.collectors.odds_api_client import pull_snapshots
from pipeline._predict_runner import run_predict

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# 优雅退出处理
# ──────────────────────────────────────────────

_stop_requested = False


def _handle_sigint(signum: int, frame: object) -> None:
    global _stop_requested
    print("\n\n[手动中止] 收到 Ctrl+C，正在完成当前操作后输出结果...")
    _stop_requested = True


signal.signal(signal.SIGINT, _handle_sigint)


# ──────────────────────────────────────────────
# 截止时间判断
# ──────────────────────────────────────────────

def _get_earliest_kickoff(league_ids: list[str]) -> datetime | None:
    """查询数据库中最早的未来开赛时间（取今日赛程中最早一场）。"""
    from sqlalchemy import select

    from data.storage.db import SessionLocal
    from data.storage.models import Match

    now = datetime.now(timezone.utc)
    today = now.date()

    with SessionLocal() as session:
        rows = session.scalars(
            select(Match.match_date)
            .where(
                Match.league_id.in_(league_ids),
                Match.match_date >= today,
            )
            .order_by(Match.match_date)
            .limit(1)
        ).all()

    if not rows:
        return None

    earliest_date = rows[0]
    # football-data.co.uk 无开赛时间，保守设为当天 UTC 11:00（欧洲赛事最早时间）
    return datetime(earliest_date.year, earliest_date.month, earliest_date.day,
                    11, 0, 0, tzinfo=timezone.utc)


def _should_stop(league_ids: list[str], cutoff_minutes: int) -> bool:
    """判断是否达到截止条件（最早开赛时间距今 ≤ cutoff_minutes）。"""
    if _stop_requested:
        return True

    earliest = _get_earliest_kickoff(league_ids)
    if earliest is None:
        logger.info("数据库中无未来赛程，流程结束")
        return True

    now = datetime.now(timezone.utc)
    delta = (earliest - now).total_seconds() / 60
    logger.info(f"距最早开赛还有 {delta:.0f} 分钟（截止线：{cutoff_minutes} 分钟）")
    return delta <= cutoff_minutes


# ──────────────────────────────────────────────
# 主流程
# ──────────────────────────────────────────────

def run(
    league_ids: list[str],
    budget: float,
    poll_interval_hours: float,
    cutoff_minutes: int,
    output_dir: str | None,
    safety_margin: float,
) -> None:
    start_time = datetime.now(timezone.utc)
    logger.info("=" * 60)
    logger.info(f"pitch-odds 日常流程启动  {start_time.strftime('%Y-%m-%d %H:%M UTC')}")
    logger.info(f"联赛：{' '.join(league_ids)}  预算：{budget:.0f}元  轮询间隔：{poll_interval_hours}h  截止前：{cutoff_minutes}min")
    logger.info("=" * 60)

    # ── 步骤 1：拉取赛程 ──
    logger.info("\n[1/3] 拉取赛程（football-data.co.uk）...")
    try:
        fixture_results = fetch_and_import(league_ids)
        total_new = sum(r.new_matches for r in fixture_results)
        logger.info(f"      新增赛程 {total_new} 场")
    except Exception as exc:
        logger.error(f"      拉取赛程失败：{exc}")
        logger.error("      请检查网络连接，或手动运行：python3 -m data.collectors.fixtures_fetcher")

    # ── 步骤 2：赔率轮询循环 ──
    logger.info("\n[2/3] 开始赔率轮询（The Odds API）")
    logger.info(f"      每 {poll_interval_hours} 小时拉取一次，Ctrl+C 可随时手动停止\n")

    poll_count = 0
    while True:
        if _should_stop(league_ids, cutoff_minutes):
            break

        poll_count += 1
        logger.info(f"--- 第 {poll_count} 次轮询  {datetime.now(timezone.utc).strftime('%H:%M UTC')} ---")

        try:
            counts = pull_snapshots(league_ids)
            total_snaps = sum(counts.values())
            logger.info(f"    写入快照：{total_snaps} 条")
        except RuntimeError as exc:
            # API key 错误等不可恢复错误，直接停止
            logger.error(f"    赔率拉取失败（不可恢复）：{exc}")
            break
        except Exception as exc:
            logger.warning(f"    赔率拉取出错（将在下次重试）：{exc}")

        # 等待下一次轮询，期间每分钟检查一次截止条件和手动中止
        wait_seconds = poll_interval_hours * 3600
        elapsed = 0
        while elapsed < wait_seconds:
            if _stop_requested or _should_stop(league_ids, cutoff_minutes):
                break
            time.sleep(60)
            elapsed += 60

        if _stop_requested or _should_stop(league_ids, cutoff_minutes):
            break

    # ── 步骤 3：输出预测结果 ──
    logger.info("\n[3/3] 生成最终预测推荐...")
    try:
        for league_id in league_ids:
            run_predict(
                league_id=league_id,
                budget=budget,
                safety_margin=safety_margin,
                plan_date=date.today().isoformat(),
                output_dir=output_dir,
            )
    except Exception as exc:
        logger.error(f"    预测输出失败：{exc}")

    duration = (datetime.now(timezone.utc) - start_time).total_seconds() / 60
    logger.info(f"\n流程结束，总耗时 {duration:.0f} 分钟，共轮询 {poll_count} 次")


# ──────────────────────────────────────────────
# CLI 入口
# ──────────────────────────────────────────────

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(
        description="pitch-odds 日常流程：赛程拉取 → 赔率轮询 → 预测输出",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--leagues",
        nargs="+",
        default=["E0", "SP1", "D1", "I1", "F1"],
        help="联赛 ID（默认全部五大联赛）",
    )
    parser.add_argument(
        "--budget",
        type=float,
        default=1000.0,
        help="总投注预算（元），默认 1000",
    )
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=6.0,
        dest="poll_interval",
        help="赔率轮询间隔（小时），默认 6",
    )
    parser.add_argument(
        "--cutoff-minutes",
        type=int,
        default=60,
        dest="cutoff_minutes",
        help="最早开赛前多少分钟自动停止轮询，默认 60",
    )
    parser.add_argument(
        "--safety-margin",
        type=float,
        default=1.05,
        dest="safety_margin",
        help="正期望阈值（EV ≥ N），默认 1.05",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        dest="output_dir",
        help="预测结果 JSON 保存目录（可选）",
    )

    args = parser.parse_args()

    run(
        league_ids=args.leagues,
        budget=args.budget,
        poll_interval_hours=args.poll_interval,
        cutoff_minutes=args.cutoff_minutes,
        output_dir=args.output_dir,
        safety_margin=args.safety_margin,
    )


if __name__ == "__main__":
    main()
