"""看板数据加载与预处理层（不依赖 Streamlit，可独立测试）。

所有函数均为纯函数或轻量工厂，接受 dict / DataFrame / dataclass，
输出 pandas DataFrame 或简单 Python 数据结构，便于 Plotly 直接消费。
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from backtest.capital_sim import CapitalSimResult, simulate_capital
from backtest.engine import BacktestPrediction, BacktestResult
from backtest.metrics import compute_metrics
from interfaces.contracts import CalibratedPrediction
from optimizer.ev_filter import filter_positive_ev
from optimizer.parlay_optimizer import build_parlay_plan
from worldcup.backtest import WCBacktestResult, compute_wc_metrics
from worldcup.data import sample_wc_matches
from worldcup.backtest import run_wc_backtest


# ──────────────────────────────────────────────
# 报告加载
# ──────────────────────────────────────────────

def load_report(path: str | Path) -> dict[str, Any]:
    """从磁盘加载回测报告 JSON（由 backtest/report.py 生成）。"""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def list_reports(report_dir: str | Path = "reports") -> list[Path]:
    """返回 reports/ 目录下所有报告文件（按修改时间降序）。"""
    d = Path(report_dir)
    if not d.exists():
        return []
    return sorted(d.glob("backtest_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)


# ──────────────────────────────────────────────
# 预测数据 → DataFrame
# ──────────────────────────────────────────────

def predictions_to_df(report_or_result: dict | BacktestResult) -> pd.DataFrame:
    """将报告 dict 或 BacktestResult 转为 DataFrame。

    列：match_id, league_id, season, match_date, actual_outcome,
        p_home_raw, p_draw_raw, p_away_raw,
        p_home, p_draw, p_away,
        odds_home, odds_draw, odds_away
    """
    if isinstance(report_or_result, BacktestResult):
        preds = [_pred_to_dict(p) for p in report_or_result.predictions]
    else:
        preds = report_or_result.get("predictions", [])

    if not preds:
        return pd.DataFrame()

    df = pd.DataFrame(preds)
    df["match_date"] = pd.to_datetime(df["match_date"])
    return df


def add_ev_columns(df: pd.DataFrame) -> pd.DataFrame:
    """向 DataFrame 追加 ev_home / ev_draw / ev_away 及最高 EV 信息。"""
    if df.empty:
        return df
    df = df.copy()
    df["ev_home"] = df["p_home"] * df["odds_home"]
    df["ev_draw"] = df["p_draw"] * df["odds_draw"]
    df["ev_away"] = df["p_away"] * df["odds_away"]
    df["best_ev"]      = df[["ev_home", "ev_draw", "ev_away"]].max(axis=1)
    df["best_outcome"] = df[["ev_home", "ev_draw", "ev_away"]].idxmax(axis=1).str.replace("ev_", "")
    return df


def filter_ev_candidates(
    df: pd.DataFrame,
    safety_margin: float = 1.05,
) -> pd.DataFrame:
    """从 DataFrame 中筛选正期望候选（每场保留最高 EV 方向）。

    Returns:
        筛选结果 DataFrame，含 match_id, outcome, odds, p_model, ev, edge 列。
    """
    if df.empty:
        return pd.DataFrame()

    preds: list[CalibratedPrediction] = []
    for _, row in df.iterrows():
        overround = (1 / row["odds_home"]) + (1 / row["odds_draw"]) + (1 / row["odds_away"])
        preds.append({
            "match_id":        int(row["match_id"]),
            "model_version":   "report",
            "p_home": float(row["p_home"]),
            "p_draw": float(row["p_draw"]),
            "p_away": float(row["p_away"]),
            "odds_home": float(row["odds_home"]),
            "odds_draw": float(row["odds_draw"]),
            "odds_away": float(row["odds_away"]),
            "ev_home":  float(row["p_home"]) * float(row["odds_home"]),
            "ev_draw":  float(row["p_draw"]) * float(row["odds_draw"]),
            "ev_away":  float(row["p_away"]) * float(row["odds_away"]),
            "edge_home": float(row["p_home"]) - (1 / float(row["odds_home"])) / overround,
            "edge_draw": float(row["p_draw"]) - (1 / float(row["odds_draw"])) / overround,
            "edge_away": float(row["p_away"]) - (1 / float(row["odds_away"])) / overround,
            "smart_money_flag": False,
            "exclude_flag":     False,
        })

    try:
        legs = filter_positive_ev(preds, safety_margin=safety_margin)
    except ValueError:
        return pd.DataFrame()

    if not legs:
        return pd.DataFrame()
    return pd.DataFrame(legs)


# ──────────────────────────────────────────────
# 校准曲线数据
# ──────────────────────────────────────────────

def calibration_df(report: dict[str, Any]) -> pd.DataFrame:
    """从报告的 calibration_diagnostics 生成校准对比 DataFrame。

    列：outcome, bin_center, mean_predicted_raw, actual_frequency,
        mean_predicted_cal, is_calibrated
    """
    calib = report.get("metrics", {}).get("calibration_diagnostics", {})
    if not calib:
        return pd.DataFrame()

    rows: list[dict] = []
    for outcome, data in calib.items():
        for stage, key in [("raw", "raw"), ("calibrated", "calibrated")]:
            for b in data.get(key, []):
                rows.append({
                    "outcome":         outcome,
                    "stage":           stage,
                    "bin_center":      (b["range_start"] + b["range_end"]) / 2,
                    "mean_predicted":  b["mean_predicted"],
                    "actual_frequency": b["actual_frequency"],
                    "count":           b["count"],
                })

    return pd.DataFrame(rows) if rows else pd.DataFrame()


# ──────────────────────────────────────────────
# 资本曲线数据
# ──────────────────────────────────────────────

def capital_curve_df(sim: CapitalSimResult) -> pd.DataFrame:
    """将 CapitalSimResult 转为可绘图 DataFrame。

    列：index(0..n), capital, date(str), is_skipped, daily_profit
    """
    dates = ["init"] + [d.date for d in sim.daily_results]
    profits = [0.0] + [d.daily_profit for d in sim.daily_results]
    skipped = [False] + [d.skipped for d in sim.daily_results]

    return pd.DataFrame({
        "date":         dates,
        "capital":      sim.capital_curve,
        "daily_profit": profits,
        "skipped":      skipped,
    })


def daily_pnl_df(sim: CapitalSimResult) -> pd.DataFrame:
    """返回按日期的净盈亏 DataFrame，仅含已投注日。"""
    rows = [
        {"date": d.date, "profit": d.daily_profit, "capital_after": d.capital_after}
        for d in sim.daily_results
        if not d.skipped
    ]
    return pd.DataFrame(rows) if rows else pd.DataFrame()


# ──────────────────────────────────────────────
# 串场推荐数据
# ──────────────────────────────────────────────

def build_parlay_summary(
    candidates_df: pd.DataFrame,
    plan_date: str,
    total_budget: float,
) -> dict[str, Any] | None:
    """将候选 DataFrame 转为串场方案（供看板展示）。

    Returns:
        ParlayPlan dict，或 None（候选不足时）。
    """
    if candidates_df.empty or len(candidates_df) < 2:
        return None

    legs = candidates_df.to_dict("records")
    try:
        return build_parlay_plan(legs, plan_date, total_budget)
    except ValueError:
        return None


# ──────────────────────────────────────────────
# 指标摘要
# ──────────────────────────────────────────────

def metrics_summary(report: dict[str, Any]) -> dict[str, Any]:
    """从报告 dict 中提取关键指标，返回展示友好的 dict。"""
    m = report.get("metrics", {})
    return {
        "比赛场次":   m.get("total_matches", "-"),
        "Brier 分":   f"{m.get('brier_score', 0):.4f}",
        "命中率":     f"{m.get('hit_rate', 0):.2%}",
        "ROI":        f"{m.get('roi', 0):.2%}",
        "最大回撤":   f"{m.get('max_drawdown', 0):.2f}",
        "夏普比率":   f"{m.get('sharpe_ratio', 0):.3f}",
    }


# ──────────────────────────────────────────────
# 演示模式（无 DB 时使用）
# ──────────────────────────────────────────────

def build_demo_wc_report() -> dict[str, Any]:
    """使用内置 WC2022 样例数据生成演示用报告 dict。"""
    matches = sample_wc_matches()
    result = run_wc_backtest(matches, train_tournaments=[], test_tournament="WC2022")
    metrics = compute_wc_metrics(result)

    preds = [
        {
            "match_id":      p.match_id,
            "league_id":     "WC",
            "season":        p.tournament,
            "match_date":    p.match_date,
            "actual_outcome": p.actual_outcome,
            "train_until":   "",
            "p_home_raw":    p.p_home_raw,
            "p_draw_raw":    p.p_draw_raw,
            "p_away_raw":    p.p_away_raw,
            "p_home":        p.p_home,
            "p_draw":        p.p_draw,
            "p_away":        p.p_away,
            "odds_home":     p.odds_home,
            "odds_draw":     p.odds_draw,
            "odds_away":     p.odds_away,
        }
        for p in result.predictions
    ]

    return {
        "league_id":     "WC",
        "model_version": result.model_version,
        "train_seasons": result.train_tournaments,
        "val_season":    result.val_tournament,
        "test_season":   result.test_tournament,
        "predictions":   preds,
        "metrics": {
            "total_matches":          metrics["total_matches"],
            "brier_score":            metrics["brier_score"],
            "hit_rate":               metrics["hit_rate"],
            "roi":                    metrics["roi"],
            "max_drawdown":           metrics["max_drawdown"],
            "sharpe_ratio":           metrics["sharpe_ratio"],
            "calibration_diagnostics": {},
        },
        "generated_at": "demo",
    }


# ──────────────────────────────────────────────
# 内部工具
# ──────────────────────────────────────────────

def _pred_to_dict(p: BacktestPrediction) -> dict[str, Any]:
    return {
        "match_id":       p.match_id,
        "league_id":      p.league_id,
        "season":         p.season,
        "match_date":     p.match_date,
        "actual_outcome": p.actual_outcome,
        "train_until":    p.train_until,
        "p_home_raw":     p.p_home_raw,
        "p_draw_raw":     p.p_draw_raw,
        "p_away_raw":     p.p_away_raw,
        "p_home":         p.p_home,
        "p_draw":         p.p_draw,
        "p_away":         p.p_away,
        "odds_home":      p.odds_home,
        "odds_draw":      p.odds_draw,
        "odds_away":      p.odds_away,
    }
