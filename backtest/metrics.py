from __future__ import annotations

from math import sqrt
from statistics import mean, pstdev
from typing import Any

from backtest.engine import BacktestResult


def compute_metrics(result: BacktestResult) -> dict[str, Any]:
    if not result.predictions:
        raise ValueError("没有可计算指标的预测结果")

    brier_terms = []
    hit = 0
    total_stake = 0.0
    total_return = 0.0
    profits = []
    cumulative = 0.0
    peak = 0.0
    max_drawdown = 0.0

    for item in result.predictions:
        probs = {"H": item.p_home, "D": item.p_draw, "A": item.p_away}
        one_hot = {"H": 0.0, "D": 0.0, "A": 0.0}
        one_hot[item.actual_outcome] = 1.0
        brier = ((probs["H"] - one_hot["H"]) ** 2 + (probs["D"] - one_hot["D"]) ** 2 + (probs["A"] - one_hot["A"]) ** 2) / 3.0
        brier_terms.append(brier)

        pred_outcome = max(probs, key=probs.get)
        if pred_outcome == item.actual_outcome:
            hit += 1

        stake = 1.0
        odds_by_outcome = {"H": item.odds_home, "D": item.odds_draw, "A": item.odds_away}
        won = pred_outcome == item.actual_outcome
        payout = stake * odds_by_outcome[pred_outcome] if won else 0.0
        profit = payout - stake
        profits.append(profit)
        total_stake += stake
        total_return += payout

        cumulative += profit
        peak = max(peak, cumulative)
        max_drawdown = max(max_drawdown, peak - cumulative)

    roi = (total_return - total_stake) / total_stake if total_stake else 0.0
    hit_rate = hit / len(result.predictions)
    avg_profit = mean(profits)
    std_profit = pstdev(profits)
    sharpe = (avg_profit / std_profit) * sqrt(len(profits)) if std_profit > 0 else 0.0

    return {
        "total_matches": len(result.predictions),
        "brier_score": round(float(mean(brier_terms)), 6),
        "hit_rate": round(hit_rate, 6),
        "roi": round(roi, 6),
        "max_drawdown": round(max_drawdown, 6),
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
