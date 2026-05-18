from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from data.storage.db import SessionLocal
from backtest.engine import run_backtest
from backtest.metrics import compute_metrics
from backtest.report import write_report
from backtest.capital_sim import simulate_capital


def _parse_csv_items(raw: str) -> list[str]:
    return [part.strip() for part in raw.split(",") if part.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="回测入口（Phase 2 + Phase 4）")
    parser.add_argument("--league", default="E0", help="联赛 ID")
    parser.add_argument("--train-seasons", required=True, help="训练赛季，逗号分隔")
    parser.add_argument("--val-season", required=True, help="验证赛季（用于校准）")
    parser.add_argument("--test-season", required=True, help="测试赛季")
    parser.add_argument("--output-dir", default="reports", help="报告输出目录")
    parser.add_argument("--capital-sim", action="store_true", help="同时运行 Phase 4 资本模拟")
    parser.add_argument("--initial-capital", type=float, default=10_000.0, help="模拟初始资本")
    args = parser.parse_args()

    train_seasons = _parse_csv_items(args.train_seasons)
    with SessionLocal() as session:
        result = run_backtest(
            league_id=args.league,
            train_seasons=train_seasons,
            val_season=args.val_season,
            test_season=args.test_season,
            session=session,
        )

    metrics = compute_metrics(result)
    report_path = write_report(result, metrics, output_dir=args.output_dir)
    calib = metrics["calibration_diagnostics"]
    home_bins = len(calib["home"]["calibrated"])
    draw_bins = len(calib["draw"]["calibrated"])
    away_bins = len(calib["away"]["calibrated"])
    print(
        "回测完成: "
        f"league={args.league}, model={result.model_version}, matches={metrics['total_matches']}, "
        f"brier={metrics['brier_score']}, roi={metrics['roi']}, "
        f"calib_bins=H{home_bins}/D{draw_bins}/A{away_bins}, report={report_path}"
    )

    if args.capital_sim:
        sim = simulate_capital(result, initial_capital=args.initial_capital)
        print(
            "资本模拟: "
            f"initial={sim.initial_capital:.0f}, final={sim.final_capital:.2f}, "
            f"peak={sim.peak_capital:.2f}, roi={sim.roi:.4f}, "
            f"max_drawdown={sim.max_drawdown_pct:.4f}, "
            f"betting_days={sim.n_betting_days}, skipped={sim.n_skipped_days}, "
            f"bets={sim.total_bets}, won={sim.won_bets}, "
            f"staked={sim.total_staked:.2f}, payout={sim.total_payout:.2f}"
        )


if __name__ == "__main__":
    main()
