"""使用最优超参对五大联赛批量运行回测。

权重来自 scripts/grid_search_weights.py 在 2022-23 验证集上搜索的结果：
    E0  → form_weight=0.16, fatigue_weight=0.10
    SP1 → form_weight=0.00, fatigue_weight=0.00
    D1  → form_weight=0.00, fatigue_weight=0.00
    I1  → form_weight=0.20, fatigue_weight=0.10
    F1  → form_weight=0.20, fatigue_weight=0.10

默认使用原始概率（跳过 Platt 校准），因为测试集上发现 Platt 单赛季
过拟合会扭曲概率排名，导致 ROI 恶化。
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.storage.db import SessionLocal
from backtest.engine import run_backtest
from backtest.metrics import compute_metrics
from backtest.report import write_report

TRAIN_SEASONS = ["2018-19", "2019-20", "2020-21", "2021-22"]
VAL_SEASON    = "2022-23"
TEST_SEASON   = "2023-24"
OUTPUT_DIR    = "reports"

LEAGUE_WEIGHTS: dict[str, dict[str, float]] = {
    "E0":  {"form_weight": 0.16, "fatigue_weight": 0.10},
    "SP1": {"form_weight": 0.00, "fatigue_weight": 0.00},
    "D1":  {"form_weight": 0.00, "fatigue_weight": 0.00},
    "I1":  {"form_weight": 0.20, "fatigue_weight": 0.10},
    "F1":  {"form_weight": 0.20, "fatigue_weight": 0.10},
}


def main() -> None:
    print("五大联赛批量回测（使用验证集最优超参）")
    print(f"训练赛季: {TRAIN_SEASONS}")
    print(f"验证赛季: {VAL_SEASON} | 测试赛季: {TEST_SEASON}")
    print("-" * 70)
    print(f"{'联赛':>5} {'场数':>6} {'Brier':>8} {'命中率':>8} {'EV注数':>8} {'覆盖率':>8} {'ROI':>8} {'夏普':>8}")
    print("-" * 70)

    summary = []
    with SessionLocal() as session:
        for league_id, weights in LEAGUE_WEIGHTS.items():
            try:
                result = run_backtest(
                    league_id=league_id,
                    train_seasons=TRAIN_SEASONS,
                    val_season=VAL_SEASON,
                    test_season=TEST_SEASON,
                    session=session,
                    skip_calibration=True,
                    **weights,
                )
                metrics = compute_metrics(result)
                write_report(result, metrics, output_dir=OUTPUT_DIR)
                summary.append((league_id, metrics))
                print(
                    f"{league_id:>5} "
                    f"{metrics['total_matches']:>6} "
                    f"{metrics['brier_score']:>8.4f} "
                    f"{metrics['hit_rate']:>8.2%} "
                    f"{metrics['n_ev_bets']:>8} "
                    f"{metrics['coverage_pct']:>8.2%} "
                    f"{metrics['roi']:>8.2%} "
                    f"{metrics['sharpe_ratio']:>8.3f}"
                )
            except Exception as exc:
                print(f"{league_id:>5}  [错误] {exc}")

    print("-" * 70)
    if summary:
        avg_brier  = sum(m["brier_score"] for _, m in summary) / len(summary)
        avg_roi    = sum(m["roi"]         for _, m in summary) / len(summary)
        avg_sharpe = sum(m["sharpe_ratio"] for _, m in summary) / len(summary)
        print(f"{'均值':>5} {'':>6} {avg_brier:>8.4f} {'':>8} {'':>8} {'':>8} {avg_roi:>8.2%} {avg_sharpe:>8.3f}")
    print(f"\n报告已写入 {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
