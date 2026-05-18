"""测试 dashboard/data_loader.py 中的纯数据处理函数（不依赖 Streamlit）。"""
from __future__ import annotations

import ast
import json
import tempfile
from pathlib import Path

import pandas as pd
import pytest

from backtest.engine import BacktestPrediction, BacktestResult
from backtest.capital_sim import simulate_capital
from dashboard.data_loader import (
    add_ev_columns,
    build_demo_wc_report,
    build_parlay_summary,
    calibration_df,
    capital_curve_df,
    daily_pnl_df,
    filter_ev_candidates,
    list_reports,
    load_report,
    metrics_summary,
    predictions_to_df,
)


# ──────────────────────────────────────────────
# 工厂辅助
# ──────────────────────────────────────────────

def _make_minimal_report(n: int = 10) -> dict:
    """构造最小合法报告 dict，含 n 条预测。"""
    preds = [
        {
            "match_id": i,
            "league_id": "E0",
            "season": "2023-24",
            "match_date": f"2023-08-{(i % 28) + 1:02d}",
            "actual_outcome": "H",
            "train_until": "2023-07-31",
            "p_home_raw": 0.55,
            "p_draw_raw": 0.25,
            "p_away_raw": 0.20,
            "p_home": 0.57,
            "p_draw": 0.24,
            "p_away": 0.19,
            "odds_home": 2.0,
            "odds_draw": 3.5,
            "odds_away": 5.0,
        }
        for i in range(1, n + 1)
    ]
    return {
        "league_id": "E0",
        "model_version": "dixon_coles_v1",
        "train_seasons": ["2018-19"],
        "val_season": "2022-23",
        "test_season": "2023-24",
        "predictions": preds,
        "metrics": {
            "total_matches": n,
            "brier_score": 0.20,
            "hit_rate": 0.52,
            "roi": -0.03,
            "max_drawdown": 5.0,
            "sharpe_ratio": -0.3,
            "calibration_diagnostics": {},
        },
        "generated_at": "2026-05-18T00:00:00+00:00",
    }


def _make_backtest_result(n: int = 10) -> BacktestResult:
    preds = [
        BacktestPrediction(
            match_id=i, league_id="E0", season="2023-24",
            match_date=f"2023-08-{(i % 10) + 1:02d}",
            actual_outcome="H",
            train_until="2023-07-31",
            p_home_raw=0.55, p_draw_raw=0.25, p_away_raw=0.20,
            p_home=0.60, p_draw=0.22, p_away=0.18,
            odds_home=2.0, odds_draw=3.5, odds_away=5.0,
        )
        for i in range(1, n + 1)
    ]
    return BacktestResult(
        league_id="E0", model_version="dixon_coles_v1",
        train_seasons=[], val_season="", test_season="",
        predictions=preds,
    )


# ──────────────────────────────────────────────
# load_report / list_reports
# ──────────────────────────────────────────────

def test_load_report_from_json_file(tmp_path: Path) -> None:
    report = _make_minimal_report()
    fp = tmp_path / "backtest_E0_test.json"
    fp.write_text(json.dumps(report))
    loaded = load_report(fp)
    assert loaded["league_id"] == "E0"
    assert len(loaded["predictions"]) == 10


def test_list_reports_empty_dir(tmp_path: Path) -> None:
    assert list_reports(tmp_path) == []


def test_list_reports_finds_files(tmp_path: Path) -> None:
    for name in ["backtest_E0_1.json", "backtest_E0_2.json"]:
        (tmp_path / name).write_text("{}")
    result = list_reports(tmp_path)
    assert len(result) == 2


def test_list_reports_nonexistent_dir() -> None:
    assert list_reports("/nonexistent/path/xyz") == []


# ──────────────────────────────────────────────
# predictions_to_df
# ──────────────────────────────────────────────

def test_predictions_to_df_from_report_dict() -> None:
    report = _make_minimal_report(5)
    df = predictions_to_df(report)
    assert len(df) == 5
    assert "p_home" in df.columns
    assert pd.api.types.is_datetime64_any_dtype(df["match_date"])


def test_predictions_to_df_from_backtest_result() -> None:
    result = _make_backtest_result(8)
    df = predictions_to_df(result)
    assert len(df) == 8
    assert "odds_home" in df.columns


def test_predictions_to_df_empty_returns_empty() -> None:
    df = predictions_to_df({"predictions": []})
    assert df.empty


# ──────────────────────────────────────────────
# add_ev_columns
# ──────────────────────────────────────────────

def test_add_ev_columns_adds_expected_columns() -> None:
    df = predictions_to_df(_make_minimal_report(3))
    df = add_ev_columns(df)
    for col in ("ev_home", "ev_draw", "ev_away", "best_ev", "best_outcome"):
        assert col in df.columns


def test_add_ev_columns_correct_calculation() -> None:
    df = predictions_to_df(_make_minimal_report(1))
    df = add_ev_columns(df)
    row = df.iloc[0]
    assert abs(row["ev_home"] - row["p_home"] * row["odds_home"]) < 1e-9


def test_add_ev_columns_best_ev_is_max() -> None:
    df = predictions_to_df(_make_minimal_report(5))
    df = add_ev_columns(df)
    for _, row in df.iterrows():
        assert row["best_ev"] == max(row["ev_home"], row["ev_draw"], row["ev_away"])


def test_add_ev_columns_empty_returns_empty() -> None:
    assert add_ev_columns(pd.DataFrame()).empty


# ──────────────────────────────────────────────
# filter_ev_candidates
# ──────────────────────────────────────────────

def test_filter_ev_candidates_positive_ev() -> None:
    # p_home=0.60, odds_home=2.0 → ev=1.20 > 1.05
    report = _make_minimal_report(5)
    for p in report["predictions"]:
        p["p_home"] = 0.60
        p["odds_home"] = 2.0
    df = predictions_to_df(report)
    candidates = filter_ev_candidates(df)
    assert len(candidates) > 0


def test_filter_ev_candidates_no_positive_ev() -> None:
    # p * odds < 1.05 for all outcomes
    report = _make_minimal_report(5)
    for p in report["predictions"]:
        p["p_home"] = 0.40; p["odds_home"] = 2.0   # ev=0.80
        p["p_draw"] = 0.30; p["odds_draw"] = 3.0   # ev=0.90
        p["p_away"] = 0.30; p["odds_away"] = 3.0   # ev=0.90
    df = predictions_to_df(report)
    candidates = filter_ev_candidates(df)
    assert candidates.empty


def test_filter_ev_candidates_returns_expected_columns() -> None:
    df = predictions_to_df(_make_minimal_report(5))
    candidates = filter_ev_candidates(df)
    if not candidates.empty:
        for col in ("match_id", "outcome", "odds", "p_model", "ev", "edge"):
            assert col in candidates.columns


def test_filter_ev_candidates_empty_df() -> None:
    assert filter_ev_candidates(pd.DataFrame()).empty


# ──────────────────────────────────────────────
# build_parlay_summary
# ──────────────────────────────────────────────

def test_build_parlay_summary_returns_plan_with_candidates() -> None:
    report = _make_minimal_report(10)
    for p in report["predictions"]:
        p["p_home"] = 0.60; p["odds_home"] = 2.0
        p["match_date"] = "2023-08-01"
    df = predictions_to_df(report)
    candidates = filter_ev_candidates(df)
    plan = build_parlay_summary(candidates, "2023-08-01", 1000.0)
    assert plan is not None
    assert "options" in plan


def test_build_parlay_summary_returns_none_insufficient() -> None:
    plan = build_parlay_summary(pd.DataFrame(), "2023-08-01", 1000.0)
    assert plan is None


def test_build_parlay_summary_returns_none_one_candidate() -> None:
    report = _make_minimal_report(1)
    report["predictions"][0]["p_home"] = 0.60
    report["predictions"][0]["odds_home"] = 2.0
    report["predictions"][0]["match_date"] = "2023-08-01"
    df = predictions_to_df(report)
    candidates = filter_ev_candidates(df)
    plan = build_parlay_summary(candidates, "2023-08-01", 1000.0)
    assert plan is None


# ──────────────────────────────────────────────
# calibration_df
# ──────────────────────────────────────────────

def test_calibration_df_empty_when_no_data() -> None:
    report = _make_minimal_report()
    report["metrics"]["calibration_diagnostics"] = {}
    assert calibration_df(report).empty


def test_calibration_df_with_data() -> None:
    report = _make_minimal_report()
    report["metrics"]["calibration_diagnostics"] = {
        "home": {
            "raw":        [{"bin": 0, "range_start": 0.0, "range_end": 0.1,
                            "count": 10, "mean_predicted": 0.05, "actual_frequency": 0.06}],
            "calibrated": [{"bin": 0, "range_start": 0.0, "range_end": 0.1,
                            "count": 10, "mean_predicted": 0.05, "actual_frequency": 0.05}],
        }
    }
    df = calibration_df(report)
    assert not df.empty
    assert "outcome" in df.columns
    assert set(df["stage"].unique()) == {"raw", "calibrated"}


# ──────────────────────────────────────────────
# capital_curve_df / daily_pnl_df
# ──────────────────────────────────────────────

def _make_sim() -> "CapitalSimResult":
    result = _make_backtest_result(n=30)
    return simulate_capital(result, initial_capital=10_000.0)


def test_capital_curve_df_columns() -> None:
    sim = _make_sim()
    df = capital_curve_df(sim)
    for col in ("date", "capital", "daily_profit", "skipped"):
        assert col in df.columns


def test_capital_curve_df_length() -> None:
    sim = _make_sim()
    df = capital_curve_df(sim)
    assert len(df) == len(sim.capital_curve)


def test_daily_pnl_df_only_betting_days() -> None:
    sim = _make_sim()
    df = daily_pnl_df(sim)
    # 只含实际投注日（未跳过）
    assert len(df) == sim.n_betting_days


# ──────────────────────────────────────────────
# metrics_summary
# ──────────────────────────────────────────────

def test_metrics_summary_keys() -> None:
    report = _make_minimal_report()
    summary = metrics_summary(report)
    for key in ("比赛场次", "Brier 分", "命中率", "ROI", "最大回撤", "夏普比率"):
        assert key in summary


def test_metrics_summary_formats() -> None:
    report = _make_minimal_report()
    summary = metrics_summary(report)
    assert "%" in summary["命中率"]
    assert "%" in summary["ROI"]


# ──────────────────────────────────────────────
# build_demo_wc_report
# ──────────────────────────────────────────────

def test_build_demo_wc_report_structure() -> None:
    report = build_demo_wc_report()
    assert report["league_id"] == "WC"
    assert len(report["predictions"]) > 0
    assert "metrics" in report


def test_build_demo_wc_report_predictions_valid() -> None:
    report = build_demo_wc_report()
    for p in report["predictions"]:
        total = p["p_home"] + p["p_draw"] + p["p_away"]
        assert abs(total - 1.0) < 1e-5


# ──────────────────────────────────────────────
# 语法检查：所有看板 Python 文件
# ──────────────────────────────────────────────

@pytest.mark.parametrize("py_file", list(Path("dashboard").rglob("*.py")))
def test_dashboard_files_valid_python(py_file: Path) -> None:
    source = py_file.read_text(encoding="utf-8")
    try:
        ast.parse(source)
    except SyntaxError as e:
        pytest.fail(f"{py_file} 语法错误: {e}")
