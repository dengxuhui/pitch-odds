"""推荐页：当日正期望场次 + 串场方案。"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dashboard.data_loader import (
    build_demo_wc_report,
    build_parlay_summary,
    filter_ev_candidates,
    list_reports,
    load_report,
    predictions_to_df,
    add_ev_columns,
)

st.set_page_config(page_title="推荐 | pitch-odds", page_icon="📋", layout="wide")
st.title("📋 正期望推荐")


# ── 数据源选择 ──────────────────────────────────
with st.sidebar:
    st.header("数据源")
    reports = list_reports()
    use_demo = st.checkbox("使用演示数据（WC2022）", value=not reports)

    if use_demo:
        report = build_demo_wc_report()
    else:
        report_names = [p.name for p in reports]
        selected = st.selectbox("选择回测报告", report_names)
        report = load_report(reports[report_names.index(selected)])

    safety_margin = st.slider("EV 安全边际", 1.00, 1.30, 1.05, 0.01)
    total_budget  = st.number_input("总预算（元）", min_value=100.0, value=1000.0, step=100.0)


# ── 数据处理 ──────────────────────────────────
df = predictions_to_df(report)
if df.empty:
    st.warning("报告中没有预测数据。")
    st.stop()

df = add_ev_columns(df)

# 日期选择
all_dates = sorted(df["match_date"].dt.date.unique())
if not all_dates:
    st.warning("没有可用的比赛日期。")
    st.stop()

with st.sidebar:
    sel_date = st.selectbox("比赛日期", all_dates, index=len(all_dates) - 1)

day_df = df[df["match_date"].dt.date == sel_date]
candidates_df = filter_ev_candidates(day_df, safety_margin=safety_margin)


# ── 主区域 ──────────────────────────────────
col1, col2, col3 = st.columns(3)
col1.metric("当日场次", len(day_df))
col2.metric("正期望候选", len(candidates_df))
col3.metric("安全边际阈值", f"{safety_margin:.2f}")

st.divider()

# 正期望候选表
st.subheader("正期望候选场次")
if candidates_df.empty:
    st.info(f"当日无正期望场次（EV ≥ {safety_margin}）。")
else:
    display = candidates_df[["match_id", "outcome", "odds", "p_model", "ev", "edge"]].copy()
    display.columns = ["场次ID", "投注方向", "赔率", "模型概率", "期望值EV", "优势Edge"]
    display["模型概率"] = display["模型概率"].map("{:.2%}".format)
    display["期望值EV"]  = display["期望值EV"].map("{:.3f}".format)
    display["优势Edge"]  = display["优势Edge"].map("{:.3f}".format)

    # EV 高亮：EV > 1.20 标绿，EV > 1.10 标橙
    def _highlight_ev(row: pd.Series) -> list[str]:
        ev = float(row["期望值EV"])
        color = "background-color: #d4edda" if ev >= 1.20 else \
                "background-color: #fff3cd" if ev >= 1.10 else ""
        return [color] * len(row)

    st.dataframe(
        display.style.apply(_highlight_ev, axis=1),
        use_container_width=True,
    )

st.divider()

# 串场方案
st.subheader("串场方案")
plan_date_str = sel_date.isoformat()
plan = build_parlay_summary(candidates_df, plan_date_str, total_budget)

if plan is None:
    st.info("候选场次不足 2 个，无法生成串场方案。")
else:
    tier_labels = {"hedge": "🛡 保底层（40%）", "core": "⭐ 核心层（40%）", "aggressive": "🚀 冲击层（20%）"}
    tier_cols = st.columns(len(plan["options"]))
    for col, option in zip(tier_cols, plan["options"]):
        label = tier_labels.get(option["tier"], option["tier"])
        with col:
            st.markdown(f"**{label}**")
            st.metric("串场腿数", len(option["legs"]))
            st.metric("总赔率",   f"{option['total_odds']:.2f}")
            st.metric("胜率",     f"{option['win_rate']:.2%}")
            st.metric("预期EV",   f"{option['expected_ev']:.3f}")
            with st.expander("查看各腿"):
                for leg in option["legs"]:
                    st.write(f"- 场次 {leg['match_id']} | {leg['outcome']} | 赔率 {leg['odds']:.2f} | p={leg['p_model']:.2%}")
