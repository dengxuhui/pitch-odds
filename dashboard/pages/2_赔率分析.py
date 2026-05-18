"""赔率分析页：模型概率 vs 市场隐含概率、优势可视化。"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dashboard.data_loader import (
    add_ev_columns,
    build_demo_wc_report,
    filter_ev_candidates,
    list_reports,
    load_report,
    predictions_to_df,
)

st.set_page_config(page_title="赔率分析 | pitch-odds", page_icon="📈", layout="wide")
st.title("📈 赔率分析")

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

df = predictions_to_df(report)
if df.empty:
    st.warning("没有预测数据。")
    st.stop()

df = add_ev_columns(df)

# 计算市场隐含概率（去水钱）
df["overround"]    = 1 / df["odds_home"] + 1 / df["odds_draw"] + 1 / df["odds_away"]
df["imp_home"]     = (1 / df["odds_home"]) / df["overround"]
df["imp_draw"]     = (1 / df["odds_draw"]) / df["overround"]
df["imp_away"]     = (1 / df["odds_away"]) / df["overround"]
df["edge_home_v2"] = df["p_home"] - df["imp_home"]
df["edge_draw_v2"] = df["p_draw"] - df["imp_draw"]
df["edge_away_v2"] = df["p_away"] - df["imp_away"]
df["anomaly_flag"] = df["best_ev"] > 1.5  # 简化异常判断

# ── EV 分布图 ──────────────────────────────────
st.subheader("全量场次 EV 分布")

fig_ev = go.Figure()
fig_ev.add_trace(go.Histogram(
    x=df["best_ev"],
    name="最高EV",
    marker_color="#4e79a7",
    opacity=0.75,
    xbins={"size": 0.05},
))
fig_ev.add_vline(x=1.05, line_dash="dash", line_color="orange",
                 annotation_text="EV=1.05 门槛")
fig_ev.add_vline(x=1.0,  line_dash="dot",  line_color="gray",
                 annotation_text="盈亏平衡")
fig_ev.update_layout(
    xaxis_title="期望值 EV",
    yaxis_title="场次数量",
    height=350,
    margin={"t": 30, "b": 40},
)
st.plotly_chart(fig_ev, use_container_width=True)

st.divider()

# ── 模型概率 vs 市场隐含概率（散点图）──────────────────────────────────
st.subheader("模型概率 vs 市场隐含概率")

outcome_col = st.selectbox("结果方向", ["home", "draw", "away"], index=0)
outcome_map = {"home": ("p_home", "imp_home"), "draw": ("p_draw", "imp_draw"), "away": ("p_away", "imp_away")}
p_col, imp_col = outcome_map[outcome_col]

fig_scatter = go.Figure()
colors = df["anomaly_flag"].map({True: "#e15759", False: "#4e79a7"})
fig_scatter.add_trace(go.Scatter(
    x=df[imp_col],
    y=df[p_col],
    mode="markers",
    marker={"color": colors, "size": 6, "opacity": 0.7},
    text=df["match_id"].astype(str),
    name="场次",
))
# 对角线（模型概率 = 市场概率）
diag_max = 0.95
fig_scatter.add_trace(go.Scatter(
    x=[0, diag_max], y=[0, diag_max],
    mode="lines",
    line={"dash": "dash", "color": "gray"},
    name="无优势线",
    showlegend=True,
))
fig_scatter.update_layout(
    xaxis_title=f"市场隐含概率（{outcome_col}）",
    yaxis_title=f"模型概率（{outcome_col}）",
    height=420,
    margin={"t": 30, "b": 40},
)
st.plotly_chart(fig_scatter, use_container_width=True)
st.caption("🔴 高 EV 异常场次（EV > 1.5）| 🔵 正常场次 | 点在对角线上方 = 模型看好，市场低估")

st.divider()

# ── 异常赔率场次高亮 ──────────────────────────────────
anomaly_df = df[df["anomaly_flag"]].copy()
st.subheader(f"⚠️ 高 EV 异常场次（{len(anomaly_df)} 场）")
if anomaly_df.empty:
    st.success("当前报告中无高 EV 异常场次。")
else:
    cols_show = ["match_id", "match_date", "actual_outcome",
                 "odds_home", "odds_draw", "odds_away",
                 "p_home", "p_draw", "p_away", "best_ev", "best_outcome"]
    st.dataframe(
        anomaly_df[cols_show].sort_values("best_ev", ascending=False)
        .rename(columns={
            "match_id": "场次ID",
            "match_date": "比赛日期",
            "actual_outcome": "实际结果",
            "best_ev": "最高EV",
            "best_outcome": "最佳方向",
        }),
        use_container_width=True,
    )

st.divider()

# ── 单场次详细分析 ──────────────────────────────────
st.subheader("单场次详细分析")
match_ids = sorted(df["match_id"].unique())
sel_id = st.selectbox("选择场次 ID", match_ids)
row = df[df["match_id"] == sel_id].iloc[0]

c1, c2, c3 = st.columns(3)
for col, label, p_key, imp_key, odds_key in [
    (c1, "主队胜", "p_home", "imp_home", "odds_home"),
    (c2, "平局",   "p_draw", "imp_draw", "odds_draw"),
    (c3, "客队胜", "p_away", "imp_away", "odds_away"),
]:
    with col:
        st.markdown(f"**{label}**")
        st.metric("模型概率",    f"{row[p_key]:.2%}")
        st.metric("市场隐含",    f"{row[imp_key]:.2%}")
        ev = row[p_key] * row[odds_key]
        delta_color = "normal" if ev >= 1.05 else "inverse"
        st.metric("EV", f"{ev:.3f}", delta=f"{ev - 1:.3f}", delta_color=delta_color)
        st.metric("赔率",        f"{row[odds_key]:.2f}")

if row["anomaly_flag"]:
    st.error("⚠️ 该场次 EV > 1.5，赔率可能异常，建议谨慎参考。")
elif row["best_ev"] >= 1.05:
    st.success(f"✅ 存在正期望方向（{row['best_outcome']}，EV={row['best_ev']:.3f}）")
else:
    st.info("该场次无正期望投注方向。")
