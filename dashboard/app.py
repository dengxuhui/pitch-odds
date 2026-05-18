"""pitch-odds Streamlit 仪表板主入口。

启动命令：
    streamlit run dashboard/app.py
"""
import streamlit as st

st.set_page_config(
    page_title="pitch-odds 足球量化分析",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("⚽ pitch-odds 足球量化分析看板")
st.caption("五大联赛 + 2026 FIFA 世界杯 | Dixon-Coles × Elo 双模型")

st.markdown("""
## 导航

| 页面 | 内容 |
|---|---|
| 📋 推荐 | 当日正期望场次筛选、串场方案（保底/核心/冲击） |
| 📈 赔率分析 | 模型概率 vs 市场隐含概率、正期望识别 |
| 📊 回测报告 | 资本曲线、ROI/回撤/命中率、校准诊断 |

使用左侧导航栏切换页面。
""")

st.divider()
st.info("首次使用请先运行回测脚本生成报告：\n"
        "`python3 scripts/backtest.py --league E0 "
        "--train-seasons 2018-19,2019-20,2020-21,2021-22 "
        "--val-season 2022-23 --test-season 2023-24 --capital-sim`")
