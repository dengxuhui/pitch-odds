# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

足球量化分析系统，覆盖欧洲五大联赛（EPL/LaLiga/Bundesliga/SerieA/Ligue1）与 2026 FIFA 世界杯。核心目标：通过概率模型识别正期望场次，系统化生成最优多串场组合，最大化期望收益。

**当前进度**：Phase 1（数据管道）和 Phase 2（Dixon-Coles 建模）已完成，Phase 3（串场优化器）尚未开始。

## 常用命令

### 环境启动

```bash
# 启动 PostgreSQL + Redis（需要 Docker 或 Colima）
docker compose up -d

# 初始化/迁移数据库（Alembic）
python3 scripts/init_db.py --mode alembic

# 安装依赖（使用 .venv）
pip install -r requirements.txt
```

### 数据导入

```bash
# 导入历史比赛数据（CSV 来自 football-data.org）
python3 -m data.collectors.historical <csv_path> --league-id E0 --season 2024-25

# 导入开盘赔率数据
python3 -m data.collectors.odds <csv_path> --league-id E0 --bookmaker bet365 --season 2024-25
```

### 训练与回测

```bash
# 训练 Dixon-Coles 模型（Phase 2）
python3 scripts/train.py --league E0 --train-seasons 2018-19,2019-20,2020-21,2021-22 --val-season 2022-23

# 运行回测，输出报告到 reports/
python3 scripts/backtest.py --league E0 \
  --train-seasons 2018-19,2019-20,2020-21,2021-22 \
  --val-season 2022-23 --test-season 2024-25

# Phase 1 完整端到端验收
bash scripts/run_phase1_e2e.sh <csv_path> --league-id E0 --season 2024-25 --bookmaker bet365
```

### 测试

```bash
# 运行全部测试
python3 -m pytest

# 运行单个测试文件
python3 -m pytest tests/test_models.py -v

# Phase 1 严格验收
PYTHONPATH="$(pwd)" python3 scripts/verify_phase1.py --league-id E0 --strict
```

## 架构总览

六模块线性流水线，数据单向流动：

```
数据采集层 → 概率预测模型 → 概率校准层 → 串场优化器 → 资金分配层 → 回测验证层
```

模块间通过 `interfaces/contracts.py` 中的 `TypedDict` 交换数据，数据流为：

```
MatchFeatures → ModelRawOutput → CalibratedPrediction → ParlayPlan → BetRecord
```

**禁止跨模块直接传裸 dict。**

### 关键目录

| 目录 | 说明 |
|---|---|
| `interfaces/contracts.py` | 所有 TypedDict 接口定义（唯一真相源） |
| `data/collectors/` | historical.py（历史赛果）、odds.py（赔率采集） |
| `data/processors/` | form_score、fatigue、injury、odds_anomaly 四个处理器 |
| `data/storage/` | SQLAlchemy ORM（models.py）+ Alembic 迁移（migrations/） |
| `models/` | dixon_coles.py（当前主模型）、calibration.py（Isotonic 校准） |
| `backtest/` | engine.py + metrics.py + report.py |
| `scripts/` | 各阶段入口脚本，用 `PYTHONPATH=$(pwd)` 运行 |

### 数据库

- **PostgreSQL 15**：历史数据、回测记录，连接串 `postgresql+psycopg://pitch_odds:pitch_odds@localhost:5432/pitch_odds`
- **Redis 7**：赔率实时缓存（`localhost:6379`）
- 迁移：`data/storage/migrations/versions/`，使用 Alembic

字段类型约定：概率 `NUMERIC(5,4)`，赔率 `NUMERIC(6,3)`，金额 `NUMERIC(10,2)`，所有 TIMESTAMP 存 UTC。

## 核心业务规则（实现时必须遵守）

**接口约定：**
- 概率字段范围 `[0.0, 1.0]`，同一组三值之和 `= 1.0`（容差 `1e-6`，见 `PROBABILITY_TOLERANCE`）
- 日期统一 `'YYYY-MM-DD'`，时间统一 ISO 8601 带 UTC 时区
- 空值用 `Optional[T]`，**禁止用 `0` 或 `-1` 代替空值**
- 每个模块接收数据时主动校验，拒绝非法输入，不静默修正

**赔率去水钱：**
```python
overround = 1/h + 1/d + 1/a
p_implied = (1/odds) / overround
```

**正期望阈值：** `p_model × odds ≥ 1.05`

**赔率异常：** `alert_level == 'HIGH'` 的场次直接移出串场候选（`exclude_flag=True`）

**三层串场策略：** 保底 40%（2~3 串）/ 核心 40%（4~5 串）/ 冲击 20%（6~7 串）

**Kelly 公式：** Half Kelly（`fraction=0.5`），负期望时注金为 0

**止损规则：** 单日亏损 >10% 停止；连亏 3 天检查模型；总回撤 >30% 暂停

## 防数据泄漏（回测纪律）

时间切割：训练 2018-2022 → 验证 2023 → 测试 2024/25，测试集只用一次，不得反复调参后重测。五大联赛各自独立训练，联赛 ID：`E0 / SP1 / D1 / I1 / F1`。

## Agent 行为约束

**禁止启动长期运行的进程**：任务完成后不启动任何会持续占用终端的服务器进程（包括 Streamlit、数据库服务等）。Docker 服务需用 `docker compose up -d` 后台模式启动。
