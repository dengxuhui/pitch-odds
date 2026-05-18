from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any

from backtest.engine import BacktestPrediction, BacktestResult
from capital.allocator import allocate_capital
from capital.stop_loss import StopLossTracker
from interfaces.contracts import BetRecord, CalibratedPrediction
from optimizer.ev_filter import filter_positive_ev
from optimizer.parlay_optimizer import build_parlay_plan


# 比赛结果编码映射
_OUTCOME_MAP: dict[str, str] = {"home": "H", "draw": "D", "away": "A"}


@dataclass
class DaySimResult:
    date: str
    skipped: bool
    skip_reason: str
    records: list[BetRecord]
    daily_profit: float
    capital_after: float


@dataclass
class CapitalSimResult:
    initial_capital: float
    final_capital: float
    peak_capital: float
    capital_curve: list[float]
    daily_results: list[DaySimResult]
    n_betting_days: int
    n_skipped_days: int
    total_bets: int
    won_bets: int
    total_staked: float
    total_payout: float
    roi: float
    max_drawdown_pct: float


def simulate_capital(
    result: BacktestResult,
    *,
    initial_capital: float = 10_000.0,
    safety_margin: float = 1.05,
    daily_loss_limit_pct: float = 0.10,
    max_consecutive_loss_days: int = 3,
    max_drawdown_pct: float = 0.30,
) -> CapitalSimResult:
    """在 BacktestResult 的预测上模拟 Phase 3+4 资金曲线。

    对每个有预测的交易日：
    1. 将 BacktestPrediction 转换为 CalibratedPrediction
    2. 正期望筛选（filter_positive_ev）
    3. 若候选不足则跳过当天
    4. 生成串场方案（build_parlay_plan）
    5. 按 Half Kelly 分配注金（allocate_capital）
    6. 检查止损规则（should_bet）
    7. 模拟实际结果，更新资本

    所有计算严格按时间顺序，不引入未来信息。
    """
    if initial_capital <= 0:
        raise ValueError("initial_capital 必须大于 0")

    tracker = StopLossTracker(
        initial_capital=initial_capital,
        daily_loss_limit_pct=daily_loss_limit_pct,
        max_consecutive_loss_days=max_consecutive_loss_days,
        max_drawdown_pct=max_drawdown_pct,
    )

    # 构建 match_id → actual_outcome 查找表
    outcome_lookup: dict[int, str] = {
        p.match_id: p.actual_outcome for p in result.predictions
    }

    # 按日期分组
    by_date: dict[str, list[BacktestPrediction]] = defaultdict(list)
    for pred in result.predictions:
        by_date[pred.match_date].append(pred)

    daily_results: list[DaySimResult] = []
    capital_curve: list[float] = [initial_capital]
    total_staked = 0.0
    total_payout = 0.0
    total_bets = 0
    won_bets = 0

    for day_date in sorted(by_date):
        preds = by_date[day_date]
        calib_preds = [_to_calibrated(p) for p in preds]

        # Phase 3: 正期望筛选
        try:
            candidates = filter_positive_ev(calib_preds, safety_margin=safety_margin)
        except ValueError:
            candidates = []

        if len(candidates) < 2:
            daily_results.append(DaySimResult(
                date=day_date,
                skipped=True,
                skip_reason="候选不足",
                records=[],
                daily_profit=0.0,
                capital_after=tracker.capital,
            ))
            capital_curve.append(tracker.capital)
            continue

        # Phase 3: 生成串场方案
        try:
            plan = build_parlay_plan(candidates, day_date, tracker.capital)
        except ValueError:
            daily_results.append(DaySimResult(
                date=day_date,
                skipped=True,
                skip_reason="方案生成失败",
                records=[],
                daily_profit=0.0,
                capital_after=tracker.capital,
            ))
            capital_curve.append(tracker.capital)
            continue

        # Phase 4: 分配注金
        records = allocate_capital(plan, tracker.capital, is_simulation=True)

        total_daily_stake = sum(r["stake"] for r in records)

        # Phase 4: 止损检查
        if not tracker.should_bet(total_daily_stake):
            daily_results.append(DaySimResult(
                date=day_date,
                skipped=True,
                skip_reason="止损触发",
                records=records,
                daily_profit=0.0,
                capital_after=tracker.capital,
            ))
            capital_curve.append(tracker.capital)
            continue

        # 模拟结果：判断每注串场是否获胜
        daily_profit = 0.0
        settled: list[BetRecord] = []
        for rec in records:
            won = _parlay_won(rec["legs"], outcome_lookup)
            stake = rec["stake"]
            payout = stake * rec["total_odds"] if won else 0.0
            profit = payout - stake
            rec["won"] = won
            rec["payout"] = round(payout, 2)
            rec["profit"] = round(profit, 2)
            settled.append(rec)

            daily_profit += profit
            total_staked += stake
            total_payout += payout
            total_bets += 1
            if won:
                won_bets += 1

        tracker.record_day(daily_profit)
        capital_curve.append(tracker.capital)
        daily_results.append(DaySimResult(
            date=day_date,
            skipped=False,
            skip_reason="",
            records=settled,
            daily_profit=daily_profit,
            capital_after=tracker.capital,
        ))

    n_betting_days = sum(1 for d in daily_results if not d.skipped)
    n_skipped_days = sum(1 for d in daily_results if d.skipped)
    roi = (total_payout - total_staked) / total_staked if total_staked > 0 else 0.0

    peak = tracker.peak_capital
    drawdown_final = (peak - tracker.capital) / peak if peak > 0 else 0.0

    return CapitalSimResult(
        initial_capital=initial_capital,
        final_capital=tracker.capital,
        peak_capital=peak,
        capital_curve=capital_curve,
        daily_results=daily_results,
        n_betting_days=n_betting_days,
        n_skipped_days=n_skipped_days,
        total_bets=total_bets,
        won_bets=won_bets,
        total_staked=round(total_staked, 2),
        total_payout=round(total_payout, 2),
        roi=round(roi, 6),
        max_drawdown_pct=round(drawdown_final, 6),
    )


# ──────────────────────────────────────────────
# 内部辅助
# ──────────────────────────────────────────────

def _to_calibrated(pred: BacktestPrediction) -> CalibratedPrediction:
    """将 BacktestPrediction 转换为 CalibratedPrediction（用于 EV 筛选）。"""
    overround = (1.0 / pred.odds_home) + (1.0 / pred.odds_draw) + (1.0 / pred.odds_away)
    return {
        "match_id": pred.match_id,
        "model_version": "sim",
        "p_home": pred.p_home,
        "p_draw": pred.p_draw,
        "p_away": pred.p_away,
        "odds_home": pred.odds_home,
        "odds_draw": pred.odds_draw,
        "odds_away": pred.odds_away,
        "ev_home":  pred.p_home * pred.odds_home,
        "ev_draw":  pred.p_draw * pred.odds_draw,
        "ev_away":  pred.p_away * pred.odds_away,
        "edge_home": pred.p_home - (1.0 / pred.odds_home) / overround,
        "edge_draw": pred.p_draw - (1.0 / pred.odds_draw) / overround,
        "edge_away": pred.p_away - (1.0 / pred.odds_away) / overround,
        "smart_money_flag": False,
        "exclude_flag": False,
    }


def _parlay_won(legs: list[Any], outcome_lookup: dict[int, str]) -> bool:
    """判断串场是否全中（所有腿均命中）。"""
    for leg in legs:
        expected = _OUTCOME_MAP.get(leg["outcome"])
        actual = outcome_lookup.get(leg["match_id"])
        if expected is None or actual != expected:
            return False
    return True
