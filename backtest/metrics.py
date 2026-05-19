from __future__ import annotations

from math import sqrt
from statistics import mean, pstdev
from typing import Any

from backtest.engine import BacktestResult

# 与 filter_positive_ev 保持一致
_EV_THRESHOLD = 1.05


def compute_metrics(result: BacktestResult) -> dict[str, Any]:
    """计算回测指标。

    投注策略：固定注金（Flat Stake = 1 unit），不使用 Kelly 动态定注。
    球赛定投的目标是每轮都参与、累计 ROI 为正，而非复利最大化本金，
    因此用固定注金衡量选注质量比 Kelly 更贴合实际操作。

    EV ≥ 1.05 的场次才下注，每场最多一注（取 EV 最高方向），
    与 filter_positive_ev 的"一场一注"约束保持一致。
    """
    if not result.predictions:
        raise ValueError("没有可计算指标的预测结果")

    # ── 模型质量：全量场次 ──────────────────────────────────────────
    brier_terms: list[float] = []
    hit = 0
    for item in result.predictions:
        probs = {"H": item.p_home, "D": item.p_draw, "A": item.p_away}
        one_hot = {"H": 0.0, "D": 0.0, "A": 0.0}
        one_hot[item.actual_outcome] = 1.0
        brier = (
            (probs["H"] - one_hot["H"]) ** 2
            + (probs["D"] - one_hot["D"]) ** 2
            + (probs["A"] - one_hot["A"]) ** 2
        ) / 3.0
        brier_terms.append(brier)
        if max(probs, key=probs.get) == item.actual_outcome:
            hit += 1

    # ── 投注策略模拟：固定注金 + EV 筛选 ───────────────────────────
    total_staked = 0.0
    total_payout = 0.0
    cumulative_profit = 0.0
    peak_profit = 0.0
    max_drawdown_units = 0.0   # 峰值到谷值的最大回撤（单位数）
    bet_profits: list[float] = []
    all_matchdays: set[str] = set()
    betting_matchdays: set[str] = set()

    for item in sorted(result.predictions, key=lambda x: x.match_date):
        all_matchdays.add(item.match_date)
        p_map = {"H": item.p_home, "D": item.p_draw, "A": item.p_away}
        odds_map = {"H": item.odds_home, "D": item.odds_draw, "A": item.odds_away}

        # 每场取 EV 最高的单一方向（≥ 1.05），否则跳过
        best_outcome: str | None = None
        best_ev = 0.0
        for outcome in ("H", "D", "A"):
            ev = p_map[outcome] * odds_map[outcome]
            if ev >= _EV_THRESHOLD and ev > best_ev:
                best_ev = ev
                best_outcome = outcome

        if best_outcome is None:
            continue

        stake = 1.0
        total_staked += stake
        betting_matchdays.add(item.match_date)

        won = item.actual_outcome == best_outcome
        payout = odds_map[best_outcome] if won else 0.0
        total_payout += payout
        profit = payout - stake
        bet_profits.append(profit)

        cumulative_profit += profit
        if cumulative_profit > peak_profit:
            peak_profit = cumulative_profit
        drawdown = peak_profit - cumulative_profit
        if drawdown > max_drawdown_units:
            max_drawdown_units = drawdown

    n_bets = len(bet_profits)
    roi = (total_payout - total_staked) / total_staked if total_staked > 0 else 0.0
    coverage_pct = len(betting_matchdays) / len(all_matchdays) if all_matchdays else 0.0

    if n_bets >= 2:
        avg_p = mean(bet_profits)
        std_p = pstdev(bet_profits)
        sharpe = (avg_p / std_p) * sqrt(n_bets) if std_p > 0 else 0.0
    else:
        sharpe = 0.0

    return {
        "total_matches": len(result.predictions),
        "n_ev_bets": n_bets,
        "coverage_pct": round(coverage_pct, 4),
        "brier_score": round(float(mean(brier_terms)), 6),
        "hit_rate": round(hit / len(result.predictions), 6),
        "roi": round(roi, 6),
        "max_drawdown_units": round(max_drawdown_units, 2),
        "sharpe_ratio": round(float(sharpe), 6),
        "calibration_diagnostics": _calibration_diagnostics(result),
    }


def _calibration_diagnostics(result: BacktestResult, bins: int = 10) -> dict[str, Any]:
    fields = {
        "home": ("p_home_raw", "p_home", "H"),
        "draw": ("p_draw_raw", "p_draw", "D"),
        "away": ("p_away_raw", "p_away", "A"),
    }
    diagnostics: dict[str, Any] = {}
    for label, (raw_key, calibrated_key, outcome) in fields.items():
        raw_points = [(getattr(item, raw_key), item.actual_outcome == outcome) for item in result.predictions]
        calibrated_points = [(getattr(item, calibrated_key), item.actual_outcome == outcome) for item in result.predictions]
        diagnostics[label] = {
            "raw": _bin_points(raw_points, bins=bins),
            "calibrated": _bin_points(calibrated_points, bins=bins),
        }
    return diagnostics


def _bin_points(points: list[tuple[float, bool]], *, bins: int) -> list[dict[str, Any]]:
    bucket: list[list[tuple[float, bool]]] = [[] for _ in range(bins)]
    for prob, hit in points:
        idx = min(int(prob * bins), bins - 1)
        bucket[idx].append((prob, hit))

    output: list[dict[str, Any]] = []
    for idx, items in enumerate(bucket):
        if not items:
            continue
        probs = [p for p, _ in items]
        hits = [1.0 if h else 0.0 for _, h in items]
        output.append(
            {
                "bin": idx,
                "range_start": round(idx / bins, 3),
                "range_end": round((idx + 1) / bins, 3),
                "count": len(items),
                "mean_predicted": round(mean(probs), 6),
                "actual_frequency": round(mean(hits), 6),
            }
        )
    return output
