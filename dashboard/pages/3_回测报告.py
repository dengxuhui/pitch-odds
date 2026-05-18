"""回测报告页：指标摘要、资本曲线、校准诊断。"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backtest.capital_sim import simulate_capital
from backtest.engine import BacktestPrediction, BacktestResult
from dashboard.data_loader import (
    build_demo_wc_report,
    calibration_df,
    capital_curve_df,
    daily_pnl_df,
    list_reports,
    load_report,
    metrics_summary,
    predictions_to_df,
)

st.set_page_config(page_title="回测报告 | pitch-odds", page_icon="📊", layout="wide")
st.title("📊 回测报告")


# ── 数据源 ──────────────────────────────────
with st.sidebar:
    st.header("数据源")
    reports = list_reports()
    use_demo = st.checkbox("使用演示数据（WC2022）", value=not reports)

    if use_demo:
        report = build_demo_wc_report()
    else:
        report_names = [p.name for p in reports]
        selected = st.selectbox("选择报告", report_names)
        report = load_report(reports[report_names.index(selected)])

    st.divider()
    st.header("资本模拟参数")
    initial_capital = st.number_input("初始资本（元）", min_value=1000.0, value=10_000.0, step=1000.0)
    run_sim = st.button("运行资本模拟")


# ── 指标摘要 ──────────────────────────────────
st.subheader("指标摘要")
summary = metrics_summary(report)
cols = st.columns(len(summary))
for col, (k, v) in zip(cols, summary.items()):
    col.metric(k, v)

meta_cols = st.columns(3)
meta_cols[0].markdown(f"**联赛：** `{report.get('league_id', '-')}`")
meta_cols[1].markdown(f"**模型：** `{report.get('model_version', '-')}`")
meta_cols[2].markdown(f"**测试赛季：** `{report.get('test_season', '-')}`")

st.divider()

# ── 资本曲线（按需运行） ──────────────────────────────────
st.subheader("资本曲线")

if "sim_result" not in st.session_state:
    st.session_state["sim_result"] = None

if run_sim:
    with st.spinner("正在运行资本模拟…"):
        preds_raw = report.get("predictions", [])
        if preds_raw:
            bt_preds = [
                BacktestPrediction(
                    match_id=p["match_id"],
                    league_id=p.get("league_id", ""),
                    season=p.get("season", ""),
                    match_date=p["match_date"],
                    actual_outcome=p["actual_outcome"],
                    train_until=p.get("train_until", ""),
                    p_home_raw=p.get("p_home_raw", p["p_home"]),
                    p_draw_raw=p.get("p_draw_raw", p["p_draw"]),
                    p_away_raw=p.get("p_away_raw", p["p_away"]),
                    p_home=p["p_home"],
                    p_draw=p["p_draw"],
                    p_away=p["p_away"],
                    odds_home=p["odds_home"],
                    odds_draw=p["odds_draw"],
                    odds_away=p["odds_away"],
                )
                for p in preds_raw
            ]
            bt_result = BacktestResult(
                league_id=report.get("league_id", ""),
                model_version=report.get("model_version", ""),
                train_seasons=report.get("train_seasons", []),
                val_season=report.get("val_season", ""),
                test_season=report.get("test_season", ""),
                predictions=bt_preds,
            )
            sim = simulate_capital(bt_result, initial_capital=initial_capital)
            st.session_state["sim_result"] = sim
        else:
            st.warning("报告中无预测数据。")

sim = st.session_state.get("sim_result")

if sim is None:
    st.info("点击侧边栏「运行资本模拟」按钮以生成资本曲线。")
else:
    # 指标行
    capital_roi = (sim.final_capital - sim.initial_capital) / sim.initial_capital
    s_cols = st.columns(6)
    s_cols[0].metric("期末资本",       f"{sim.final_capital:,.2f}")
    s_cols[1].metric("资本回报率",     f"{capital_roi:.2%}",
                     help="(期末资本 - 初始资本) / 初始资本")
    s_cols[2].metric("注金回报率",     f"{sim.roi:.2%}",
                     help="(总回报 - 总注金) / 总注金；仅统计实际下注部分，与资本回报率口径不同")
    s_cols[3].metric("最大回撤",       f"{sim.max_drawdown_pct:.2%}")
    s_cols[4].metric("投注日",         sim.n_betting_days)
    s_cols[5].metric("跳过日",         sim.n_skipped_days)

    # 资本曲线
    curve_df = capital_curve_df(sim)
    fig_curve = go.Figure()
    fig_curve.add_trace(go.Scatter(
        x=list(range(len(sim.capital_curve))),
        y=sim.capital_curve,
        mode="lines",
        line={"color": "#4e79a7", "width": 2},
        name="资本",
        fill="tozeroy",
        fillcolor="rgba(78,121,167,0.15)",
    ))
    fig_curve.add_hline(y=initial_capital, line_dash="dash", line_color="gray",
                        annotation_text=f"初始资本 {initial_capital:,.0f}")
    fig_curve.update_layout(
        xaxis_title="交易日（序号）",
        yaxis_title="资本（元）",
        height=360,
        margin={"t": 20, "b": 40},
    )
    st.plotly_chart(fig_curve, use_container_width=True)

    # 日盈亏柱图
    pnl_df = daily_pnl_df(sim)
    if not pnl_df.empty:
        pnl_df["color"] = pnl_df["profit"].map(lambda x: "#59a14f" if x >= 0 else "#e15759")
        fig_pnl = go.Figure(go.Bar(
            x=pnl_df["date"],
            y=pnl_df["profit"],
            marker_color=pnl_df["color"],
            name="日盈亏",
        ))
        fig_pnl.add_hline(y=0, line_color="gray", line_dash="dot")
        fig_pnl.update_layout(
            xaxis_title="日期",
            yaxis_title="净盈亏（元）",
            height=280,
            margin={"t": 20, "b": 40},
        )
        st.plotly_chart(fig_pnl, use_container_width=True)

st.divider()

# ── 校准曲线 ──────────────────────────────────
st.subheader("概率校准诊断")
cal_df = calibration_df(report)

if cal_df.empty:
    st.info("演示数据中无校准诊断数据。校准数据仅在联赛完整回测报告中可用。")
else:
    sel_outcome  = st.selectbox("结果方向", ["home", "draw", "away"])
    sub = cal_df[cal_df["outcome"] == sel_outcome]

    fig_cal = go.Figure()
    for stage, color, name in [
        ("raw",        "#f28e2b", "原始（未校准）"),
        ("calibrated", "#4e79a7", "校准后"),
    ]:
        s = sub[sub["stage"] == stage]
        if s.empty:
            continue
        fig_cal.add_trace(go.Scatter(
            x=s["mean_predicted"],
            y=s["actual_frequency"],
            mode="lines+markers",
            marker={"size": 8},
            line={"color": color},
            name=name,
        ))
    # 对角线
    fig_cal.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode="lines",
        line={"dash": "dash", "color": "gray"},
        name="完美校准",
    ))
    fig_cal.update_layout(
        xaxis_title="模型预测概率",
        yaxis_title="实际发生频率",
        height=380,
        margin={"t": 20, "b": 40},
    )
    st.plotly_chart(fig_cal, use_container_width=True)
    st.caption("曲线越接近对角线，校准越准确。")
