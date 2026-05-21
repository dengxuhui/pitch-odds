"""静态训练 vs Walk-Forward 滚动训练 对比回测脚本。

对五大联赛分别运行两种模式，输出并排对比表格（Brier / ROI / Sharpe）。
超参使用 grid_search_weights.py 在验证集上搜索的最优值（与 run_all_backtests.py 一致）。
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.storage.db import SessionLocal
from backtest.engine import run_backtest
from backtest.metrics import compute_metrics

TRAIN_SEASONS = ["2018-19", "2019-20", "2020-21", "2021-22"]
VAL_SEASON    = "2022-23"
TEST_SEASON   = "2023-24"
RETRAIN_EVERY = 10

LEAGUE_WEIGHTS: dict[str, dict[str, float]] = {
    "E0":  {"form_weight": 0.16, "fatigue_weight": 0.10},
    "SP1": {"form_weight": 0.00, "fatigue_weight": 0.00},
    "D1":  {"form_weight": 0.00, "fatigue_weight": 0.00},
    "I1":  {"form_weight": 0.20, "fatigue_weight": 0.10},
    "F1":  {"form_weight": 0.20, "fatigue_weight": 0.10},
}


def _run(session, league_id: str, weights: dict, rolling: bool) -> dict | None:
    try:
        result = run_backtest(
            league_id=league_id,
            train_seasons=TRAIN_SEASONS,
            val_season=VAL_SEASON,
            test_season=TEST_SEASON,
            session=session,
            skip_calibration=True,
            rolling=rolling,
            retrain_every=RETRAIN_EVERY,
            **weights,
        )
        return compute_metrics(result)
    except Exception as exc:
        print(f"  [{league_id}] {'滚动' if rolling else '静态'} 模式出错: {exc}")
        return None


def main() -> None:
    print(f"静态训练 vs Walk-Forward 滚动训练（retrain_every={RETRAIN_EVERY}）")
    print(f"训练赛季: {TRAIN_SEASONS}  |  验证: {VAL_SEASON}  |  测试: {TEST_SEASON}")
    print()

    header = f"{'联赛':>5}  {'━━━━━━静态━━━━━━':^30}  {'━━━━━滚动━━━━━':^30}  {'ROI 变化':>9}"
    print(header)
    sub = f"{'':>5}  {'Brier':>8} {'ROI':>9} {'Sharpe':>8} {'EV注':>5}  {'Brier':>8} {'ROI':>9} {'Sharpe':>8} {'EV注':>5}  {'':>9}"
    print(sub)
    print("─" * len(header))

    summary_static:  list[tuple[str, dict]] = []
    summary_rolling: list[tuple[str, dict]] = []

    with SessionLocal() as session:
        for league_id, weights in LEAGUE_WEIGHTS.items():
            m_static  = _run(session, league_id, weights, rolling=False)
            m_rolling = _run(session, league_id, weights, rolling=True)

            def _fmt(m: dict | None) -> str:
                if m is None:
                    return f"{'错误':>8} {'':>9} {'':>8} {'':>5}"
                return (
                    f"{m['brier_score']:>8.4f} "
                    f"{m['roi']:>+9.2%} "
                    f"{m['sharpe_ratio']:>8.3f} "
                    f"{m['n_ev_bets']:>5}"
                )

            delta = ""
            if m_static and m_rolling:
                diff = m_rolling["roi"] - m_static["roi"]
                delta = f"{diff:>+9.2%}"
                summary_static.append((league_id, m_static))
                summary_rolling.append((league_id, m_rolling))

            print(f"{league_id:>5}  {_fmt(m_static)}  {_fmt(m_rolling)}  {delta}")

    print("─" * len(header))
    if summary_static and summary_rolling:
        avg_s_brier  = sum(m["brier_score"]   for _, m in summary_static)  / len(summary_static)
        avg_s_roi    = sum(m["roi"]            for _, m in summary_static)  / len(summary_static)
        avg_s_sharpe = sum(m["sharpe_ratio"]   for _, m in summary_static)  / len(summary_static)
        avg_r_brier  = sum(m["brier_score"]   for _, m in summary_rolling) / len(summary_rolling)
        avg_r_roi    = sum(m["roi"]            for _, m in summary_rolling) / len(summary_rolling)
        avg_r_sharpe = sum(m["sharpe_ratio"]   for _, m in summary_rolling) / len(summary_rolling)
        delta_avg = avg_r_roi - avg_s_roi
        print(
            f"{'均值':>5}  "
            f"{avg_s_brier:>8.4f} {avg_s_roi:>+9.2%} {avg_s_sharpe:>8.3f} {'':>5}  "
            f"{avg_r_brier:>8.4f} {avg_r_roi:>+9.2%} {avg_r_sharpe:>8.3f} {'':>5}  "
            f"{delta_avg:>+9.2%}"
        )
    print()
    print(f"滚动模式：初始窗口={TRAIN_SEASONS}，每 {RETRAIN_EVERY} 场重训（growing window）")


if __name__ == "__main__":
    main()
