# pitch-odds

足球量化分析系统，覆盖欧洲五大联赛（EPL / LaLiga / Bundesliga / Serie A / Ligue 1）与 2026 FIFA 世界杯。

核心目标：通过概率模型识别正期望场次，系统化生成最优多串场组合，最大化期望收益。

![Python](https://img.shields.io/badge/Python-3.11+-blue) ![Tests](https://img.shields.io/badge/tests-152%20passed-brightgreen) ![License](https://img.shields.io/badge/license-MIT-green)

---

## 功能概览

| 模块 | 说明 |
|---|---|
| **数据采集** | 历史赛果、开盘赔率、球队状态（伤病 / 疲劳 / 近期表现） |
| **概率模型** | Dixon-Coles v2（form/fatigue λ 修正）+ Platt Scaling 校准 |
| **世界杯模型** | Elo 评分 + 俱乐部状态辅助（2026 FIFA World Cup） |
| **串场优化器** | 正期望筛选 → 三层串场策略（保底 / 核心 / 冲击） |
| **资金分配** | Half Kelly 公式 + 单日止损 / 连亏止损 / 总回撤保护 |
| **回测引擎** | 严格时间切割，防数据泄漏，输出 JSON 报告 |
| **自动推送** | GitHub Actions 每日自动运行，结果推送到 Discord |

---

## 自动化流程（推荐方式）

系统通过 GitHub Actions 实现**全自动运行**，无需手动干预：每天定时拉取赛程 + 赔率 → 生成推荐 → 推送到 Discord。

### 架构

```
每天定时（默认北京时间 15:00）
  ↓ .github/workflows/daily.yml 自动触发
  ↓ 拉取今日赛程（football-data.co.uk）
  ↓ 拉取最新赔率快照（The Odds API）
  ↓ Dixon-Coles 预测 + 正期望筛选 + 三层串场优化
  ↓ Discord 推送推荐 Embed（概率表 + 建议注金）
     无比赛时静默，不推送
```

### 第一步：准备外部服务

| 服务 | 用途 | 获取方式 |
|---|---|---|
| [Neon.tech](https://neon.tech) 或 [Supabase](https://supabase.com) | 免费托管 PostgreSQL，持久化模型参数 | 注册免费账号，创建数据库，获取连接串 |
| [The Odds API](https://the-odds-api.com) | 实时赔率数据 | 注册免费账号（500 credits/月） |
| Discord Webhook | 推送通知 | Discord 频道设置 → 整合 → 创建 Webhook |

### 第二步：配置 GitHub Secrets

在仓库 **Settings → Secrets and variables → Actions → Secrets** 中添加：

| Secret | 说明 |
|---|---|
| `DATABASE_URL` | Neon/Supabase 连接串，如 `postgresql+psycopg://user:pass@host/db` |
| `ODDS_API_KEY` | The Odds API 密钥 |
| `DISCORD_WEBHOOK_URL` | Discord 频道 Webhook URL |

### 第三步：配置推送参数（可选）

在仓库 **Settings → Secrets and variables → Actions → Variables** 中调整：

| Variable | 默认值 | 说明 |
|---|---|---|
| `LEAGUES` | `E0 SP1 D1 I1 F1` | 关注联赛（空格分隔） |
| `BUDGET` | `1000` | 投注预算（元） |
| `SAFETY_MARGIN` | `1.05` | 正期望阈值（EV ≥ N） |

修改推送时间：编辑 `.github/workflows/daily.yml` 第 10 行的 cron 表达式（UTC 时间）：

```yaml
- cron: "0 7 * * 1-6"   # 0=北京08:00  7=北京15:00  10=北京18:00  14=北京22:00
```

### 第四步：导入历史数据并训练模型（首次）

在本地或通过 GitHub Actions 手动触发 `Train Models` 工作流之前，需先将历史数据导入数据库：

```bash
# 从 football-data.org 下载历史 CSV 后导入
python3 -m data.collectors.historical <csv_path> --league-id E0 --season 2024-25
```

然后在 GitHub Actions → **Train Models** → **Run workflow** 手动触发训练。

训练完成后 `daily.yml` 即可自动运行。

### 手动触发每日流程

除定时运行外，也可随时在 **GitHub Actions → Daily Predictions → Run workflow** 手动触发，支持临时覆盖参数（联赛、预算、日期等）。

---

## 本地开发

### 1. 环境准备

```bash
# 安装依赖
pip install -r requirements.txt

# 启动 PostgreSQL + Redis（需要 Docker）
docker compose up -d

# 初始化数据库
python3 -m alembic upgrade head
```

### 2. 下载数据

```bash
# 下载五大联赛全部历史数据（2018-19 至 2024-25，保存至 data/samples/）
bash scripts/fetch_historical_data.sh

# 只下载指定联赛
bash scripts/fetch_historical_data.sh --leagues E0 SP1

# 下载并自动导入数据库
bash scripts/fetch_historical_data.sh --import
```

### 3. 训练模型

```bash
python3 scripts/train.py --league E0 \
  --train-seasons 2018-19,2019-20,2020-21,2021-22,2022-23 \
  --val-season 2023-24
```

### 4. 超参搜索（可选）

```bash
# 在验证集上搜索最优 form_weight × fatigue_weight（约 5 分钟）
python3 scripts/grid_search_weights.py
```

### 5. 运行回测

```bash
# 单个联赛（指定最优超参）
python3 scripts/backtest.py --league E0 \
  --train-seasons 2018-19,2019-20,2020-21,2021-22 \
  --val-season 2022-23 --test-season 2023-24 \
  --form-weight 0.16 --fatigue-weight 0.10 --skip-calibration

# 五大联赛一键批量回测（使用验证集搜索最优超参）
python3 scripts/run_all_backtests.py
# 报告输出至 reports/backtest_<LEAGUE>_<TIMESTAMP>.json
```

回测指标说明（Flat Stake = 1 unit 策略）：

| 指标 | 含义 |
|---|---|
| `brier_score` | 概率误差，越低越好（基准≈0.222，目标≤0.200） |
| `hit_rate` | 最高概率结果的预测准确率 |
| `n_ev_bets` | EV≥1.05 的注押场次数 |
| `coverage_pct` | 有注场次 / 总场次比例 |
| `roi` | 固定注金策略累计回报率 |
| `max_drawdown_units` | 最大回撤（单位：注） |
| `sharpe_ratio` | 风险调整收益 |

### 6. Web 端分析

```bash
streamlit run dashboard/app.py
```

浏览器访问 `http://localhost:8501`，可查看推荐场次、赔率异常分析、回测报告等。

### 运行测试

```bash
python3 -m pytest          # 全部测试（152 tests）
python3 -m pytest -v       # 带详情
```

---

## 目录结构

```
.github/workflows/
  daily.yml         # 每日自动预测 + Discord 推送
  train.yml         # 模型训练（手动 / 每月定时）
data/
  collectors/       # 历史数据、赔率采集
  processors/       # form_score / fatigue / injury / odds_anomaly
  storage/          # SQLAlchemy models + Alembic migrations
interfaces/
  contracts.py      # 所有模块间 TypedDict 接口定义
models/
  dixon_coles.py    # Dixon-Coles 泊松模型
  calibration.py    # Platt Scaling 校准（带 L2 正则）
optimizer/          # EV 筛选 / 串场优化 / 系统投注
capital/            # Half Kelly / 资金分配 / 止损
backtest/           # 回测引擎 / 指标 / 报告
worldcup/           # Elo 模型 + 俱乐部状态辅助
pipeline/
  run_daily.py      # 本地完整轮询流程
  ci_runner.py      # CI 单次执行入口
  discord_notify.py # Discord Webhook 推送
dashboard/          # Streamlit 仪表板入口
scripts/            # 训练 / 回测 / 数据导入脚本
tests/              # pytest 测试套件
docs/               # 设计文档
```

---

## 技术栈

- **Python 3.11+**
- **PostgreSQL 15** — 历史数据、模型参数持久化
- **Redis 7** — 赔率实时缓存
- **SQLAlchemy + Alembic** — ORM 与数据库迁移
- **GitHub Actions** — 自动化调度与 Discord 推送
- **Docker Compose** — 本地开发环境

---

## 文档

- 系统设计：[`docs/pitch_odds_design.md`](docs/pitch_odds_design.md)

---

## License

MIT
