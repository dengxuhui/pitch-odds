# pitch-odds 开发计划（基于设计文档 v0.3）

> 依据：`docs/pitch_odds_design.md`  
> 当前策略：先做英超（E0）+ 离线历史赔率 + 接口契约先行

## 1. 目标与范围

- 先完成英超（E0）端到端闭环：数据采集 → 预测 → 校准 → 优化 → 资金分配 → 回测。
- 五大联赛采用同一套工程框架，但参数独立训练，不共享模型参数。
- 世界杯模型在联赛闭环稳定后进入单独阶段开发。

## 2. 阶段计划

### Phase 1：项目基础 + 数据管道

- 建立 `interfaces/contracts.py`（TypedDict + 运行时校验）
- 初始化基础工程（`requirements.txt`、`.env.example`、`docker-compose.yml`）
- 完成数据库与迁移（SQLAlchemy + Alembic）
- 完成 EPL 历史数据导入与状态特征处理器（form/fatigue/injury/odds anomaly）

### Phase 2：Dixon-Coles 基础模型

- 建立模型基类与训练/预测接口
- 实现 Dixon-Coles 参数估计与 1X2 概率输出
- 完成按时间切割训练（2018-2022）与验证（2023）
- 建立概率校准模块（Isotonic）

### Phase 3：筛选与串场优化器

- 正期望筛选（`p_model * odds >= 1.05`）
- 单串场优化与多串场三层策略（保底/核心/冲击）
- 系统投注组合（N 取 N-1）

### Phase 4：资金分配 + 回测

- Half Kelly 注金计算与预算归一化分配
- 止损规则（单日、连亏、总回撤）
- 回测引擎与指标报告（ROI/回撤/命中率/夏普）

### Phase 5：世界杯模型

- 国际赛事数据接入与 Elo 评分
- 球员俱乐部状态映射到国家队强度
- 2014/2018/2022 历史验证

### Phase 6：可视化仪表板

- Streamlit 看板（推荐、赔率时序、回测曲线）
- 异常赔率高亮与风险提示

## 3. 里程碑进度表

| Phase | 内容 | 状态 | 完成时间 | 开发平台 | 模型 |
|---|---|---|---|---|---|
| Phase 1 | 项目脚手架、接口契约、DB 与 EPL 历史数据管道 | 已完成 | 2026-05-12 | Python 3.11 + PostgreSQL 15 + Redis 7 + Docker | 数据处理/特征工程 |
| Phase 2 | Dixon-Coles 建模、训练验证、概率校准 | 未开始 | TBD | Python 3.11 + NumPy/SciPy + scikit-learn | Dixon-Coles + Isotonic Calibration |
| Phase 3 | 正期望筛选、串场优化、系统投注 | 未开始 | TBD | Python 3.11 | EV Filter + Parlay Optimizer |
| Phase 4 | Half Kelly、止损、回测与报告 | 未开始 | TBD | Python 3.11 + PostgreSQL | Half Kelly + Backtest Engine |
| Phase 5 | 世界杯独立建模与历史验证 | 未开始 | TBD | Python 3.11 | Elo + World Cup Model |
| Phase 6 | Streamlit 仪表板与可视化 | 未开始 | TBD | Streamlit + Plotly | 展示层（消费上游模型输出） |

## 4. 验收标准（按阶段）

- Phase 1：EPL 2018-2024 数据可复现导入；接口校验可阻断非法输入。
- Phase 2：输出合法三分类概率（和为 1）；完成 2023 验证集评估与校准。
- Phase 3：`exclude_flag=True` 场次不会进入串场候选；可输出完整 `ParlayPlan`。
- Phase 4：回测严格时间顺序；止损规则在回测中可触发并记录。
- Phase 5：世界杯模型可独立训练与回测，不与联赛模型参数混用。
- Phase 6：可视化页面可查看推荐结果、风险标记与回测曲线。

## 5. 开发顺序约束

- 必须先落地 `interfaces/contracts.py`，再开发上下游模块。
- 先实现泊松/Dixon-Coles，再考虑 XGBoost 等增强模型。
- 测试集（2024/25）只用于最终评估，不用于反复调参。

## 6. Phase 1 执行命令

```bash
docker compose up -d
python3 scripts/init_db.py --mode alembic
python3 -m data.collectors.historical "/path/to/E0.csv" --league-id E0 --season 2024-25
python3 -m data.collectors.odds "/path/to/E0.csv" --league-id E0 --season 2024-25 --bookmaker bet365
python3 scripts/verify_phase1.py --league-id E0 --strict
python3 -m pytest
```
