# AGENTS.md — pitch-odds

> 面向未来 AI 会话的快速上手指引。每条都是"不看容易踩坑"的内容。

## 项目状态

**当前处于设计阶段，尚无代码实现。** 仓库根目录只有 `README.md`、`docs/` 和 `.gitignore`，所有代码结构均来自设计文档的规划，尚未落地。

权威设计文档：`docs/pitch_odds_design.md`（v0.3）——开发前必读，模块定义、接口契约、DB Schema、目录结构均以此为准。

## 技术栈

- Python 3.11+
- PostgreSQL 15（历史数据、回测记录）+ Redis 7（赔率实时缓存）
- SQLAlchemy ORM + Alembic 迁移
- 测试：pytest
- 仪表板：Streamlit + plotly
- 容器：Docker（`docker-compose.yml`，尚未创建）

## 规划目录结构（按设计文档）

```
data/collectors/      # 历史数据、赔率、球队状态采集
data/processors/      # form_score, fatigue, injury, odds_anomaly 计算
data/storage/         # SQLAlchemy models.py + schema.sql + Alembic migrations/
interfaces/contracts.py  # 所有模块间 TypedDict 接口定义（唯一真相源）
models/               # dixon_coles.py, xgboost_model.py, calibration.py, world_cup.py
optimizer/            # ev_filter.py, parlay_optimizer.py, system_bet.py
capital/              # kelly.py, allocator.py, stop_loss.py
backtest/             # engine.py, metrics.py, report.py
dashboard/app.py      # Streamlit 入口
tests/                # test_models.py, test_optimizer.py, test_capital.py
```

## 模块接口契约（关键，勿跳过）

所有模块间数据传递通过 `interfaces/contracts.py` 中的 `TypedDict` 定义，禁止跨模块直接传裸 dict。

数据流：`MatchFeatures → ModelRawOutput → CalibratedPrediction → ParlayPlan → BetRecord`

接口约定：
- 概率字段范围 `[0.0, 1.0]`，同一组三值之和 `= 1.0`（容差 `1e-6`）
- 日期统一 `'YYYY-MM-DD'`，时间统一 ISO 8601 带时区 UTC
- 空值用 `Optional[T]`，**禁止用 `0` 或 `-1` 代替空值**
- 每个模块接收数据时主动校验，拒绝非法输入，不静默修正

## 数据库要点

- 所有 `TIMESTAMP` 存 UTC，应用层做时区转换
- 概率字段 `NUMERIC(5,4)`，赔率字段 `NUMERIC(6,3)`，金额 `NUMERIC(10,2)`
- 复杂结构用 `JSONB`（如 `missing_players`、`legs`）
- 迁移工具：Alembic（`data/storage/migrations/`）
- 完整 DDL 在 `docs/pitch_odds_design.md` 附录 A

## 核心业务规则（实现时必须遵守）

- **赔率去水钱**：`overround = 1/h + 1/d + 1/a`，隐含概率 = `(1/odds) / overround`
- **正期望阈值**：`p_model × odds ≥ 1.05`（含安全边际）
- **串场三层策略**：保底 40%（2~3 串）/ 核心 40%（4~5 串）/ 冲击 20%（6~7 串）
- **Kelly 公式**：使用 Half Kelly（`fraction=0.5`），负期望时注金为 0
- **赔率异常**：`alert_level == 'HIGH'` 的场次直接移出串场候选（`exclude_flag=True`）
- **止损规则**：单日亏损 >10% 停止当日投注；连亏 3 天检查模型；总回撤 >30% 暂停

## 回测纪律（防数据泄漏）

- 严格时间顺序切割：训练 2018-2022 → 验证 2023 → 测试 2024/25
- 测试集只用一次，不得反复调参后重新测试
- 滚动验证窗口：训练 3 赛季，验证 1 赛季，每次向前滚动 1 赛季

## 联赛独立建模

五大联赛（EPL/LaLiga/Bundesliga/SerieA/Ligue1）各自独立训练，不混用数据。联赛 ID 对应 football-data.org 格式：`E0 / SP1 / D1 / I1 / F1`。

## 开发阶段顺序

Phase 1 数据管道 → Phase 2 Dixon-Coles 基础模型 → Phase 3 串场优化器 → Phase 4 资金分配+完整回测 → Phase 5 世界杯模型 → Phase 6 Streamlit 仪表板

**先实现泊松/Dixon-Coles，XGBoost 等数据积累后再引入。**

## Agent 行为约束

- **禁止启动长期运行的进程**：任务完成后或在执行任务时不要启动任何会持续占用终端的服务器进程。