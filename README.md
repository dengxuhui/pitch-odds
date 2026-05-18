# pitch-odds

足球量化分析系统，覆盖欧洲五大联赛（EPL / LaLiga / Bundesliga / Serie A / Ligue 1）与 2026 FIFA 世界杯。

核心目标：通过概率模型识别正期望场次，系统化生成最优多串场组合，最大化期望收益。

![Python](https://img.shields.io/badge/Python-3.11+-blue) ![Tests](https://img.shields.io/badge/tests-151%20passed-brightgreen) ![License](https://img.shields.io/badge/license-MIT-green)

---

## 功能概览

| 模块 | 说明 |
|---|---|
| **数据采集** | 历史赛果、开盘赔率、球队状态（伤病/疲劳/近期表现） |
| **概率模型** | Dixon-Coles 泊松模型 + Isotonic Regression 校准 |
| **世界杯模型** | Elo 评分 + 俱乐部状态辅助（2026 FIFA World Cup） |
| **串场优化器** | 正期望筛选 → 三层串场策略（保底 / 核心 / 冲击） |
| **资金分配** | Half Kelly 公式 + 单日止损 / 连亏止损 / 总回撤保护 |
| **回测引擎** | 严格时间切割，防数据泄漏，输出 HTML/JSON 报告 |
| **仪表板** | Streamlit 可视化（推荐看板 / 赔率分析 / 回测报告） |

---

## 快速开始

### 环境准备

```bash
# 安装依赖
pip install -r requirements.txt

# 启动 PostgreSQL + Redis（需要 Docker）
docker compose up -d

# 初始化数据库
python3 scripts/init_db.py --mode alembic
```

### 导入数据

```bash
# 导入历史比赛数据（CSV 来自 football-data.org）
python3 -m data.collectors.historical <csv_path> --league-id E0 --season 2024-25

# 导入赔率数据
python3 -m data.collectors.odds <csv_path> --league-id E0 --bookmaker bet365 --season 2024-25
```

### 训练模型

```bash
# 训练 Dixon-Coles（英超示例）
python3 scripts/train.py --league E0 \
  --train-seasons 2018-19,2019-20,2020-21,2021-22 \
  --val-season 2022-23

# 训练世界杯 Elo 模型
python3 scripts/train_worldcup.py
```

### 运行回测

```bash
python3 scripts/backtest.py --league E0 \
  --train-seasons 2018-19,2019-20,2020-21,2021-22 \
  --val-season 2022-23 --test-season 2024-25
# 报告输出至 reports/backtest_<LEAGUE>_<TIMESTAMP>.json
```

### 分析回测结果

**方式一：命令行快速查看指标**

```bash
cat reports/backtest_E0_<TIMESTAMP>.json | python3 -c "
import json, sys
d = json.load(sys.stdin)
m = d['metrics']
for k, v in m.items():
    if k != 'calibration_diagnostics':
        print(f'{k}: {v}')
"
```

输出的核心指标说明：

| 指标 | 含义 |
|---|---|
| `brier_score` | 概率误差，越低越好（随机基线约 0.667） |
| `hit_rate` | 最高概率结果的预测准确率 |
| `roi` | 平注策略回报率（不含 Kelly / 串场） |
| `max_drawdown` | 最大回撤（单位：注） |
| `sharpe_ratio` | 风险调整收益 |

**方式二：Streamlit 仪表板（可视化）**

```bash
streamlit run dashboard/app.py
```

包含校准曲线、累计收益走势、赔率分析等交互图表。

**方式三：Python 脚本深度分析**

```python
import json
from pathlib import Path

report = json.loads(Path("reports/backtest_E0_<TIMESTAMP>.json").read_text())

# 校准曲线：预测概率 vs 实际频率
for outcome in ["home", "draw", "away"]:
    print(f"\n=== {outcome} ===")
    for b in report["metrics"]["calibration_diagnostics"][outcome]["calibrated"]:
        diff = b["mean_predicted"] - b["actual_frequency"]
        print(f"  [{b['range_start']:.1f}-{b['range_end']:.1f}] "
              f"预测={b['mean_predicted']:.3f} 实际={b['actual_frequency']:.3f} 偏差={diff:+.3f}")

# 逐场预测明细
predictions = report["predictions"]
high_conf = [p for p in predictions if max(p["p_home"], p["p_draw"], p["p_away"]) > 0.6]
print(f"\n高确信度场次（>60%）：{len(high_conf)}")
```

> **注意**：`roi` 基于平注策略，反映模型纯预测能力。实际投注收益需结合正期望筛选（`optimizer/`）和 Half Kelly 资金分配（`capital/`）。

---

## 购买建议完整流程

> 对即将进行的比赛获取串场推荐和建议注金。

### 第一步：训练模型（首次或新赛季后执行一次）

```bash
python3 scripts/train.py --league E0 \
  --train-seasons 2018-19,2019-20,2020-21,2021-22 \
  --val-season 2022-23
```

模型参数自动保存到 PostgreSQL（`model_params` 表）。

### 第二步：准备即将进行的比赛 CSV

创建 `upcoming.csv`（参考 `upcoming_example.csv`）：

```csv
match_id,home_team,away_team,match_date,odds_home,odds_draw,odds_away
1001,Arsenal,Chelsea,2026-05-20,1.85,3.40,4.20
1002,Manchester City,Liverpool,2026-05-20,1.70,3.60,5.00
1003,Tottenham,Manchester Utd,2026-05-21,2.10,3.30,3.50
```

> 球队名称须与数据库中一致（导入历史数据时的名称）。也可直接填 `home_team_id` / `away_team_id`（整数）。

### 第三步：运行预测脚本

```bash
python3 scripts/predict.py \
  --league E0 \
  --input upcoming.csv \
  --budget 1000 \
  --safety-margin 1.05
```

输出示例：

```
[1/4] 加载模型参数（E0）...
[2/4] 读取比赛数据：upcoming.csv（共 5 场）
[3/4] 运行模型预测与概率校准...
[4/4] 正期望筛选（EV ≥ 1.05）+ 生成串场方案...

【全部场次预测概率】
  场次                            主胜      平局      客胜    主赔    平赔    客赔
  Arsenal vs Chelsea           48.23%   26.10%   25.67%   1.85    3.40    4.20
  ...

【正期望候选场次】
  场次                           投注方向   赔率  模型概率      EV  优势Edge
  Arsenal vs Chelsea           主胜      1.85    48.23%   0.892   ...

【三层串场建议】
  ▶ 保底层（2~3串）
    组合赔率: 6.29  胜率: 23.11%  EV: 1.453
    本层预算: 400元  Half Kelly 比例: 12.30%  建议注金: 49元
    选腿：
      · Arsenal 主胜 @1.85 (p=48.23%)
      · Manchester City 主胜 @1.70 (p=...)
  ...
```

**常用参数：**

| 参数 | 说明 | 默认值 |
|---|---|---|
| `--league` | 联赛 ID（E0/SP1/D1/I1/F1） | E0 |
| `--input` | 比赛 CSV 文件路径 | 必填 |
| `--budget` | 总投注预算（元） | 1000 |
| `--safety-margin` | EV 安全边际（建议 1.05~1.15） | 1.05 |
| `--plan-date` | 方案日期 YYYY-MM-DD | 今日 |
| `--output-dir` | 将推荐结果另存为 JSON 的目录 | 不保存 |

### 第四步：可视化查看（可选）

```bash
streamlit run dashboard/app.py
```

在 📋推荐页选择对应回测报告，可交互式调整安全边际和预算。

---

### Phase 1 端到端验收

```bash
bash scripts/run_phase1_e2e.sh "/path/to/E0_2024_25.csv" \
  --league-id E0 --season 2024-25 --bookmaker bet365

PYTHONPATH="$(pwd)" python3 scripts/verify_phase1.py --league-id E0 --strict
```

---

## 测试

```bash
# 运行全部测试（151 tests）
python3 -m pytest

# 带详情输出
python3 -m pytest -v
```

---

## 目录结构

```
data/
  collectors/       # 历史数据、赔率采集
  processors/       # form_score / fatigue / injury / odds_anomaly
  storage/          # SQLAlchemy models + Alembic migrations
interfaces/
  contracts.py      # 所有模块间 TypedDict 接口定义（唯一真相源）
models/
  dixon_coles.py    # Dixon-Coles 泊松模型
  calibration.py    # Isotonic Regression 校准
optimizer/          # EV 筛选 / 串场优化 / 系统投注
capital/            # Half Kelly / 资金分配 / 止损
backtest/           # 回测引擎 / 指标 / 报告
worldcup/           # Elo 模型 + 俱乐部状态辅助
dashboard/          # Streamlit 仪表板入口
scripts/            # 各阶段入口脚本
tests/              # pytest 测试套件
docs/               # 设计文档 / 开发计划
```

---

## 技术栈

- **Python 3.11+**
- **PostgreSQL 15** — 历史数据、回测记录
- **Redis 7** — 赔率实时缓存
- **SQLAlchemy + Alembic** — ORM 与数据库迁移
- **Streamlit + Plotly** — 可视化仪表板
- **Docker Compose** — 本地开发环境

---

## 文档

- 设计文档：[`docs/pitch_odds_design.md`](docs/pitch_odds_design.md)
- 开发计划：[`docs/development_plan.md`](docs/development_plan.md)
- Phase 1 测试清单：[`docs/phase1_test_checklist.md`](docs/phase1_test_checklist.md)

---

## License

MIT
