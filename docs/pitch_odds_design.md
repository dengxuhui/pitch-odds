# pitch-odds — 足球量化分析系统设计文档 v0.3

> **项目定位**：专注足球赛事的量化分析系统，通过概率模型识别正期望场次，系统化生成最优多串场组合，最大化期望收益。
> **GitHub 仓库**：`pitch-odds`
> **覆盖范围**：欧洲五大联赛（英超、西甲、德甲、意甲、法甲）+ 2026 FIFA 世界杯
> **语言**：Python 3.11+

---

## 目录

1. [核心理念](#1-核心理念)
2. [系统架构总览](#2-系统架构总览)
3. [模块一：数据采集层](#3-模块一数据采集层)
4. [模块二：概率预测模型](#4-模块二概率预测模型)
5. [模块三：概率校准层](#5-模块三概率校准层)
6. [模块四：筛选与串场优化器](#6-模块四筛选与串场优化器)
7. [模块五：资金分配层](#7-模块五资金分配层)
8. [模块六：回测验证层](#8-模块六回测验证层)
9. [联赛模型 vs 世界杯模型](#9-联赛模型-vs-世界杯模型)
10. [技术栈](#10-技术栈)
11. [开发路线图](#11-开发路线图)
12. [待决策事项](#12-待决策事项)
13. [附录 A：数据库 Schema](#13-附录-a数据库-schema)
14. [附录 B：模块间接口契约](#14-附录-b模块间接口契约)
15. [附录 C：仓库结构](#15-附录-c仓库结构)

---

## 1. 核心理念

### 数学基础

```
期望收益 = 真实概率 × 赔率 > 1  →  正期望，值得投注

串场期望收益 = ∏P(i) × ∏O(i)
```

### 核心策略

- **不追求最高单场胜率**，而是在可接受胜率阈值下最大化串场期望收益
- **赔率作为市场信号**：赔率变化时序是额外的信息来源，而非仅作为定价依据
- **多串场组合**：用多注串场覆盖候选场次，提升整体保本概率
- **资金分配纪律**：Half Kelly + 三层分配是长期存活的关键

### 正期望的来源

```
模型算出真实概率 P_model
赔率隐含概率 P_implied = (1/赔率) / (1 + 水钱)

若 P_model > P_implied → 存在正期望缺口 → 值得纳入串场候选
```

---

## 2. 系统架构总览

```
┌─────────────────────────────────────────────────┐
│  模块一：数据采集层                                │
│  历史数据 | 实时赔率 | 球队/球员状态               │
└───────────────────┬─────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────────┐
│  模块二：概率预测模型                              │
│  泊松/Dixon-Coles | Elo | XGBoost               │
│  输出：P(主胜), P(平), P(客胜)                    │
└───────────────────┬─────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────────┐
│  模块三：概率校准层                                │
│  验证模型概率 vs 历史真实发生率，修正偏差            │
└───────────────────┬─────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────────┐
│  模块四：筛选与串场优化器                           │
│  正期望筛选 → 多串场组合生成（保底/核心/冲击）       │
└───────────────────┬─────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────────┐
│  模块五：资金分配层                                │
│  Half Kelly | 三层策略 | 止损机制                 │
└───────────────────┬─────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────────┐
│  模块六：回测验证层                                │
│  ROI曲线 | 最大回撤 | 夏普比率 | 命中率            │
└────────────────────┬────────────────────────────┘
                     │  ← 模型迭代反馈（闭环）
                     └──────────────────────────────→ 模块二
```

---

## 3. 模块一：数据采集层

### 3.1 历史比赛数据

| 字段 | 说明 |
|---|---|
| 数据源 | football-data.org（免费）、FBref、Understat、Sofascore |
| 采集方式 | 定时爬虫 + REST API，每日凌晨批量更新 |
| 存储 | PostgreSQL，按联赛+赛季分表 |

**采集字段清单：**
- 比赛结果（主胜/平/客胜）、比分
- 进球数、失球数、射门、射正、控球率
- 角球、红黄牌
- 历史交锋记录（H2H）
- 联赛完整赛季数据（2018 至今）

### 3.2 实时赔率数据

| 字段 | 说明 |
|---|---|
| 数据源 | The Odds API、OddsPortal、Betfair Exchange |
| 采集方式 | 实时轮询（赛前 24h 开始），每 15 分钟快照 |
| 存储 | Redis（实时缓存）+ PostgreSQL（历史快照） |

**采集字段清单：**
- 主胜 / 平 / 客胜欧赔
- 亚盘（让球盘）、大小球盘
- 开盘赔率 vs 即时赔率对比
- 赔率变化完整时序（每 15 分钟一个时间点）
- 计算字段：水钱、隐含概率

**水钱计算：**
```python
overround = 1/odds_home + 1/odds_draw + 1/odds_away
water_pct = overround - 1  # 通常 5%~10%

# 去水钱后隐含概率
p_implied_home = (1/odds_home) / overround
```

### 3.3 球队 / 球员状态

| 字段 | 说明 |
|---|---|
| 数据源 | Transfermarkt（伤病）、WhoScored（球员数据）、ESPN/BBC Sport |
| 采集方式 | 自动爬取 + 赛前人工核查 |
| 特殊处理 | 阵容变化、更衣室消息等软信号需人工标注 |

#### 状态量化方法

**近期战绩（指数衰减加权）：**
```python
import numpy as np

def form_score(results, days_ago):
    """
    results: list of (result, days_ago) — result: W=3, D=1, L=0
    返回标准化到 -1.0 ~ +1.0 的状态分数
    """
    lambda_decay = 0.05
    weights = [np.exp(-lambda_decay * d) for d in days_ago]
    max_possible = 3 * sum(weights)
    raw = sum(r * w for r, w in zip(results, weights))
    return (raw / max_possible) * 2 - 1  # 标准化
```

**疲劳指数：**
```python
def fatigue_index(matches_last_30d, travel_km, minutes_played_key_players):
    base = min(matches_last_30d / 10, 1.0)           # 场次密度
    travel = min(travel_km / 5000, 0.3)               # 旅行消耗
    load = min(minutes_played_key_players / 900, 0.3) # 主力负荷
    return min(base + travel + load, 1.0)             # 归一化 0~1
```

**伤病缺阵影响：**
```python
def injury_impact(missing_players):
    """
    missing_players: list of {'importance': 0~1, 'position_multiplier': float}
    position_multiplier: GK=1.5, 核心前锋=1.3, 主力后腰=0.8, 替补=0.4
    返回胜率减损（0 ~ -0.30）
    """
    raw = sum(p['importance'] * p['position_multiplier'] for p in missing_players)
    return -min(raw * 0.15, 0.30)
```

**主场优势：**
```python
# 按联赛+球队分别计算，存入数据库
# λ_home = 历史主场进球均值 / 历史客场进球均值（用于泊松模型）
home_advantage_factor = home_goals_avg / away_goals_avg
```

**心理状态系数：**
```python
def momentum_score(win_streak, loss_streak, big_loss_flag):
    score = min(win_streak * 0.025, 0.10)
    score -= min(loss_streak * 0.02, 0.08)
    if big_loss_flag:
        score -= 0.05
    return max(-0.10, min(0.10, score))
```

### 3.4 赔率时序异常检测

```python
def detect_odds_anomaly(odds_series):
    """
    odds_series: list of floats，赛前48h每15min一个赔率快照
    """
    n = len(odds_series)
    step_changes = [abs(odds_series[i] - odds_series[i-1]) / odds_series[i-1]
                    for i in range(1, n)]

    # 1. 单步突变（> 8%）
    spike = max(step_changes) > 0.08

    # 2. 累积漂移（开盘到即时 > 15%）
    total_drift = abs(odds_series[-1] - odds_series[0]) / odds_series[0]
    trend = total_drift > 0.15

    # 3. 赛前超短窗口异常（最后8h内 > 10%）
    late_window = odds_series[max(0, n-32):]  # 约8h内的快照
    late_change = abs(late_window[-1] - late_window[0]) / late_window[0] if len(late_window) > 1 else 0
    late_anomaly = late_change > 0.10

    # 市场修正检测（跳升后回落）
    correction = (max(step_changes[:n//2]) > 0.10 and
                  abs(odds_series[-1] - odds_series[0]) / odds_series[0] < 0.03)

    alert_level = 'HIGH' if (spike or late_anomaly) else 'WATCH' if trend else 'NORMAL'

    return {
        'alert_level': alert_level,
        'exclude_from_parlay': alert_level == 'HIGH',
        'is_correction': correction,
        'smart_money': spike and not correction,   # 聪明钱信号
        'total_drift_pct': round(total_drift * 100, 2)
    }
```

**四种异常模式及处理策略：**

| 模式 | 判断标准 | 模型动作 |
|---|---|---|
| 正常波动 | 单步 < 3%，无方向性 | 直接用即时赔率反推隐含概率 |
| 聪明钱流入 | 单步下跌 > 8%，同向持续 | 主胜概率上调，优先入选串场 |
| 市场修正 | 急涨后回落至原位 | 异常时段数据排除，用收盘赔率 |
| 赛前异常 | 赛前 8h 内变幅 > 15% | 直接移出串场候选，标记高风险 |

---

## 4. 模块二：概率预测模型

### 4.1 联赛模型（推荐路线）

**阶段一：泊松分布基础模型**

```python
# Dixon-Coles 改进泊松模型
# 核心思想：每支球队有独立的进攻强度 α 和防守强度 β
# 主队进球期望：λ_home = exp(α_home + β_away + γ)  # γ 为主场优势
# 客队进球期望：λ_away = exp(α_away + β_home)

# 输出：联合概率矩阵 P(home_goals=i, away_goals=j)
# 聚合得到：P(主胜), P(平), P(客胜)
```

**阶段二：XGBoost 特征模型**（数据积累后）

特征向量（每场比赛一行）：
```
近期战绩特征：
  - home_form_5, away_form_5          # 近5场加权积分
  - home_form_10, away_form_10        # 近10场加权积分

强度特征：
  - home_goals_scored_avg             # 近10场场均进球
  - home_goals_conceded_avg           # 近10场场均失球
  - away_goals_scored_avg
  - away_goals_conceded_avg
  - home_xg_avg, away_xg_avg         # 预期进球（如有数据）

状态特征：
  - home_fatigue, away_fatigue        # 疲劳指数
  - home_injury_impact                # 伤病减损
  - away_injury_impact
  - home_momentum, away_momentum     # 心理状态系数

场景特征：
  - is_home_advantage                 # 是否主场
  - league_id                         # 联赛 ID
  - match_week                        # 赛季第几轮
  - days_since_last_match_home
  - days_since_last_match_away

赔率特征：
  - odds_implied_home                 # 赔率隐含概率（去水钱后）
  - odds_drift_home                   # 赔率漂移幅度
  - smart_money_flag                  # 聪明钱信号
```

### 4.2 每个联赛独立训练

```python
# 五大联赛各自独立建模
LEAGUES = {
    'EPL':   {'id': 'E0', 'avg_goals': 2.7, 'home_adv': 1.08},
    'LaLiga':{'id': 'SP1','avg_goals': 2.6, 'home_adv': 1.10},
    'Bundesliga':{'id':'D1','avg_goals': 3.1, 'home_adv': 1.07},
    'SerieA':{'id': 'I1', 'avg_goals': 2.5, 'home_adv': 1.09},
    'Ligue1':{'id': 'F1', 'avg_goals': 2.5, 'home_adv': 1.11},
}
```

### 4.3 时间切割（防数据泄漏）

```
训练集         验证集          测试集
2018-2022  →  2023赛季  →   2024/25赛季
   ↑              ↑               ↑
 训练模型       超参调优        最终评估（只用一次）
```

---

## 5. 模块三：概率校准层

**目的：验证模型输出的概率是否与历史真实发生率吻合**

```python
from sklearn.calibration import calibration_curve

def check_calibration(y_true, y_prob, n_bins=10):
    """
    y_true: 实际结果（0/1）
    y_prob: 模型预测概率
    """
    fraction_of_positives, mean_predicted_value = calibration_curve(
        y_true, y_prob, n_bins=n_bins
    )
    # 若 mean_predicted_value ≈ fraction_of_positives → 校准良好
    # 若 mean_predicted_value > fraction_of_positives → 模型高估 → Kelly 会过度投注
    return fraction_of_positives, mean_predicted_value

# 校准方法：Platt Scaling 或 Isotonic Regression
from sklearn.calibration import CalibratedClassifierCV
calibrated_model = CalibratedClassifierCV(base_model, method='isotonic', cv='prefit')
```

---

## 6. 模块四：筛选与串场优化器

### 6.1 正期望筛选

```python
def filter_positive_ev(matches, safety_margin=1.05):
    """
    过滤出正期望场次
    safety_margin: 期望值需高于此阈值，留安全边际
    """
    candidates = []
    for match in matches:
        # 去水钱后隐含概率
        overround = sum(1/o for o in match['odds'].values())
        p_implied = {k: (1/v)/overround for k, v in match['odds'].items()}

        # 模型预测概率
        p_model = match['model_probs']

        for outcome in ['home', 'draw', 'away']:
            ev = p_model[outcome] * match['odds'][outcome]
            if ev >= safety_margin:
                candidates.append({
                    'match_id': match['id'],
                    'outcome': outcome,
                    'odds': match['odds'][outcome],
                    'p_model': p_model[outcome],
                    'p_implied': p_implied[outcome],
                    'ev': ev,
                    'edge': p_model[outcome] - p_implied[outcome]
                })

    return sorted(candidates, key=lambda x: x['ev'], reverse=True)
```

### 6.2 串场组合优化

**单一串场期望最大化：**

```python
from itertools import combinations

def find_optimal_parlay(candidates, min_win_rate=0.20, max_legs=8):
    """
    在最低胜率约束下，找期望收益最大的串场组合
    """
    best = {'ev': 0, 'combo': []}

    for n_legs in range(2, max_legs + 1):
        for combo in combinations(candidates, n_legs):
            win_rate = 1.0
            total_odds = 1.0
            for leg in combo:
                win_rate *= leg['p_model']
                total_odds *= leg['odds']

            if win_rate < min_win_rate:
                continue

            ev = win_rate * total_odds
            if ev > best['ev']:
                best = {'ev': ev, 'win_rate': win_rate,
                        'total_odds': total_odds, 'combo': combo}

    return best
```

**多串场组合策略（三层）：**

```python
def build_multi_parlay_plan(candidates, total_budget):
    """
    保底层（40%）：高胜率 2~3 串
    核心层（40%）：期望值最高 4~5 串
    冲击层（20%）：高赔率 6~7 串
    """
    # 按概率降序排列
    sorted_c = sorted(candidates, key=lambda x: x['p_model'], reverse=True)

    hedge = sorted_c[:3]       # 最高概率场次 → 保底串
    core = sorted_c[:5]        # 核心串（可与保底重叠）
    aggressive = sorted_c[:7]  # 冲击串

    return {
        'hedge':  {'legs': hedge,  'budget': total_budget * 0.4},
        'core':   {'legs': core,   'budget': total_budget * 0.4},
        'aggressive': {'legs': aggressive, 'budget': total_budget * 0.2}
    }
```

**系统投注（容错一场）：**

```python
def system_bet(candidates, n_legs, system_size):
    """
    从 n_legs 场里取 system_size 场的所有组合
    例：5串取4 → C(5,4)=5注，任意4场全中即盈利
    """
    return list(combinations(candidates, system_size))
```

---

## 7. 模块五：资金分配层

### 7.1 Half Kelly 公式

```python
def half_kelly(p, odds, fraction=0.5):
    """
    p: 真实胜率
    odds: 赔率
    fraction: Kelly 分数（0.5 = Half Kelly，推荐）
    返回：建议投入总资金的比例
    """
    b = odds - 1       # 净赔率
    kelly = (b * p - (1 - p)) / b
    return max(0, kelly * fraction)  # 负期望时不投

# 示例：
# p=0.60, odds=2.0 → kelly=(1*0.6-0.4)/1=0.20 → half_kelly=0.10 → 投入10%
```

### 7.2 三层资金分配

```python
def allocate_capital(parlay_plan, total_capital):
    """
    对各串场方案按 Kelly 计算注金，归一化到总预算内
    """
    allocations = {}
    kelly_sum = 0

    for tier, plan in parlay_plan.items():
        combo = plan['legs']
        win_rate = 1.0
        total_odds = 1.0
        for leg in combo:
            win_rate *= leg['p_model']
            total_odds *= leg['odds']

        k = half_kelly(win_rate, total_odds)
        allocations[tier] = {'kelly': k, 'win_rate': win_rate, 'total_odds': total_odds}
        kelly_sum += k

    # 若总 Kelly < 1，剩余资金保留不投（这是正确的！）
    for tier in allocations:
        allocations[tier]['amount'] = min(
            allocations[tier]['kelly'] * total_capital,
            plan['budget']
        )

    return allocations
```

### 7.3 止损机制

```python
STOP_LOSS_RULES = {
    'daily':      0.10,   # 单日亏损超过总资金 10% → 停止当日投注
    'consecutive': 3,     # 连续亏损 3 天 → 检查模型是否失效
    'drawdown':   0.30,   # 总回撤超过 30% → 暂停，重新回测
}
```

---

## 8. 模块六：回测验证层

### 8.1 核心指标

```python
def backtest_metrics(bet_history):
    """
    bet_history: list of {'stake', 'odds', 'won', 'date'}
    """
    total_staked = sum(b['stake'] for b in bet_history)
    total_returned = sum(b['stake'] * b['odds'] if b['won'] else 0
                        for b in bet_history)

    roi = (total_returned - total_staked) / total_staked

    # 最大回撤
    cumulative = 0
    peak = 0
    max_drawdown = 0
    for b in bet_history:
        profit = b['stake'] * b['odds'] - b['stake'] if b['won'] else -b['stake']
        cumulative += profit
        peak = max(peak, cumulative)
        max_drawdown = max(max_drawdown, peak - cumulative)

    # 命中率
    hit_rate = sum(1 for b in bet_history if b['won']) / len(bet_history)

    return {
        'roi': round(roi * 100, 2),
        'max_drawdown': round(max_drawdown, 2),
        'hit_rate': round(hit_rate * 100, 2),
        'total_bets': len(bet_history)
    }
```

### 8.2 严格时间切割原则

```
❌ 错误：用全量数据训练后在同一数据上评估
✓  正确：按时间顺序滚动验证

滚动窗口示例：
  训练窗口：3个赛季
  验证窗口：1个赛季
  每次向前滚动1赛季
```

---

## 9. 联赛模型 vs 世界杯模型

| 维度 | 五大联赛 | 世界杯 |
|---|---|---|
| 数据量 | 每队每赛季 38 场 | 每届最多 7 场 |
| 数据来源 | football-data.org 等 | 世预赛 + 友谊赛 + 近3届大赛 |
| 主场优势 | 固定主客场 | 中立场为主 |
| 模型独立性 | 完全独立，按联赛分模型 | 独立建模，引入球员俱乐部数据辅助 |
| 市场有效性 | 中等（存在定价缺口） | 高（全球资金涌入，难以跑赢市场） |
| 真实价值 | 主要盈利来源 | 串场组合优化 + 识别情绪溢价 |

### 世界杯特殊考虑

- **2026扩军至48队**：小组赛每组3队只踢2场，平局价值更高
- **球员俱乐部状态**：用联赛赛季末数据评估国家队球员实力
- **赛季末疲劳**：参加世界杯的球员大多刚结束漫长赛季
- **小组赛末轮**：需标记潜在"默契球"风险场次

---

## 10. 技术栈

```
后端 / 数据处理：
  Python 3.11+
  pandas, numpy, scipy, statsmodels
  scikit-learn（校准、XGBoost封装）
  xgboost / lightgbm

数据库：
  PostgreSQL 15（历史数据、回测记录）
  Redis 7（赔率实时缓存）

数据采集：
  requests / httpx（API调用）
  scrapy / playwright（网页爬取）
  APScheduler（定时任务）

可视化 / 报告：
  Streamlit（快速原型仪表板）
  plotly（交互图表）

版本控制 / 工程：
  Git + GitHub
  Docker（环境一致性）
  pytest（单元测试）
```

---

## 11. 开发路线图

### Phase 1：数据管道（2~3周）
- [ ] 搭建 PostgreSQL 数据库，设计表结构
- [ ] football-data.org 历史数据采集（5大联赛，2018至今）
- [ ] The Odds API 实时赔率接入
- [ ] 球队状态数据采集脚本
- [ ] 赔率异常检测模块

### Phase 2：基础预测模型（2~3周）
- [ ] Dixon-Coles 泊松模型实现
- [ ] 5大联赛独立参数训练
- [ ] 概率校准层（Isotonic Regression）
- [ ] 基础回测框架搭建
- [ ] 第一次端到端验证

### Phase 3：串场优化器（2周）
- [ ] 正期望筛选器
- [ ] 单串场期望最大化
- [ ] 多串场组合生成（三层策略）
- [ ] 系统投注逻辑

### Phase 4：资金分配 + 完整回测（2周）
- [ ] Half Kelly 计算器
- [ ] 止损机制
- [ ] 完整历史回测（2018~2024）
- [ ] 性能报告（ROI / 最大回撤 / 夏普比率）

### Phase 5：世界杯模型（赛前4周）
- [ ] 世预赛 + 国际赛数据采集
- [ ] Elo 国际评分系统
- [ ] 球员俱乐部状态 → 国家队实力映射
- [ ] 世界杯专用回测（2014/2018/2022）

### Phase 6：可视化仪表板（持续）
- [ ] Streamlit 每日推荐看板
- [ ] 赔率时序图
- [ ] 回测曲线
- [ ] 串场方案输出报告

---

## 12. 待决策事项

在开始编码前，以下事项需要明确：

### 必须先决策（影响数据库设计）

| # | 问题 | 选项 |
|---|---|---|
| 1 | 初始联赛范围 | 全部5大联赛同时开始 / 先做英超验证方法论 |
| 2 | 数据历史深度 | 2018至今 / 2015至今（更多数据 vs 更快启动） |
| 3 | 赔率数据策略 | 付费 API（The Odds API ~$50/月） / 先用历史数据离线测试 |

### 开始后可迭代决策

| # | 问题 | 建议 |
|---|---|---|
| 4 | 泊松 vs XGBoost 孰先 | 先泊松验证框架，后引入XGBoost提升精度 |
| 5 | 世界杯模型触发时间 | Phase 5 开始时再决策（2026年5月开赛前约4周） |
| 6 | 最低可接受串场胜率阈值 | 回测后根据历史数据确定（初始值：20%） |
| 7 | 实盘 vs 模拟 | 强烈建议至少跑完一个完整赛季模拟后再考虑实盘 |

---

## 13. 附录 A：数据库 Schema

> 实现文件：`data/storage/models.py`（SQLAlchemy ORM）
> 迁移工具：Alembic

### 核心表结构

```sql
-- ============================================================
-- 联赛/球队基础数据
-- ============================================================

CREATE TABLE leagues (
    id          VARCHAR(10)  PRIMARY KEY,          -- 'E0', 'SP1', 'D1', 'I1', 'F1'
    name        VARCHAR(100) NOT NULL,              -- 'English Premier League'
    country     VARCHAR(50)  NOT NULL,
    avg_goals   NUMERIC(4,2) DEFAULT 2.7,          -- 联赛场均进球（定期更新）
    home_adv    NUMERIC(4,3) DEFAULT 1.08          -- 主场优势系数
);

CREATE TABLE teams (
    id          SERIAL       PRIMARY KEY,
    league_id   VARCHAR(10)  REFERENCES leagues(id),
    name        VARCHAR(100) NOT NULL,
    short_name  VARCHAR(20),
    UNIQUE (league_id, name)
);

-- ============================================================
-- 历史比赛数据（核心表）
-- ============================================================

CREATE TABLE matches (
    id              SERIAL      PRIMARY KEY,
    league_id       VARCHAR(10) NOT NULL REFERENCES leagues(id),
    season          VARCHAR(10) NOT NULL,               -- '2024-25'
    match_date      DATE        NOT NULL,
    match_week      SMALLINT,                           -- 赛季第几轮
    home_team_id    INT         NOT NULL REFERENCES teams(id),
    away_team_id    INT         NOT NULL REFERENCES teams(id),

    -- 比赛结果
    home_goals      SMALLINT,
    away_goals      SMALLINT,
    result          CHAR(1),                            -- 'H' / 'D' / 'A'，NULL表示未完赛

    -- 比赛统计（可选，有则填）
    home_shots      SMALLINT,
    away_shots      SMALLINT,
    home_shots_on   SMALLINT,                           -- 射正
    away_shots_on   SMALLINT,
    home_possession NUMERIC(4,1),                       -- 控球率 %
    away_possession NUMERIC(4,1),
    home_corners    SMALLINT,
    away_corners    SMALLINT,
    home_yellow     SMALLINT,
    away_yellow     SMALLINT,
    home_red        SMALLINT,
    away_red        SMALLINT,

    created_at      TIMESTAMP   DEFAULT NOW(),
    UNIQUE (league_id, season, match_date, home_team_id, away_team_id)
);

CREATE INDEX idx_matches_date     ON matches(match_date);
CREATE INDEX idx_matches_home     ON matches(home_team_id, match_date);
CREATE INDEX idx_matches_away     ON matches(away_team_id, match_date);
CREATE INDEX idx_matches_league   ON matches(league_id, season);

-- ============================================================
-- 赔率数据
-- ============================================================

CREATE TABLE odds_opening (
    id          SERIAL  PRIMARY KEY,
    match_id    INT     NOT NULL REFERENCES matches(id),
    bookmaker   VARCHAR(50) NOT NULL,                   -- 'bet365', 'pinnacle' 等
    odds_home   NUMERIC(6,3) NOT NULL,
    odds_draw   NUMERIC(6,3) NOT NULL,
    odds_away   NUMERIC(6,3) NOT NULL,
    overround   NUMERIC(5,4),                           -- 水钱（计算字段）
    recorded_at TIMESTAMP NOT NULL,                     -- 开盘时间
    UNIQUE (match_id, bookmaker)
);

CREATE TABLE odds_snapshots (
    id              SERIAL  PRIMARY KEY,
    match_id        INT     NOT NULL REFERENCES matches(id),
    bookmaker       VARCHAR(50) NOT NULL,
    odds_home       NUMERIC(6,3) NOT NULL,
    odds_draw       NUMERIC(6,3) NOT NULL,
    odds_away       NUMERIC(6,3) NOT NULL,
    overround       NUMERIC(5,4),
    snapshot_at     TIMESTAMP NOT NULL,                 -- 快照时间（每15分钟）
    hours_to_kick   NUMERIC(5,2)                        -- 距开赛小时数（便于查询）
);

CREATE INDEX idx_odds_match    ON odds_snapshots(match_id, snapshot_at);
CREATE INDEX idx_odds_hours    ON odds_snapshots(match_id, hours_to_kick);

CREATE TABLE odds_anomalies (
    id              SERIAL  PRIMARY KEY,
    match_id        INT     NOT NULL REFERENCES matches(id),
    alert_level     VARCHAR(10) NOT NULL,               -- 'NORMAL', 'WATCH', 'HIGH'
    anomaly_type    VARCHAR(30),                        -- 'smart_money', 'correction', 'late_spike'
    max_step_change NUMERIC(5,4),                       -- 最大单步变化幅度
    total_drift_pct NUMERIC(5,2),                       -- 开盘到即时总漂移 %
    exclude_flag    BOOLEAN DEFAULT FALSE,              -- 是否排除出串场候选
    detected_at     TIMESTAMP DEFAULT NOW()
);

-- ============================================================
-- 球队 / 球员状态
-- ============================================================

CREATE TABLE team_status (
    id              SERIAL  PRIMARY KEY,
    team_id         INT     NOT NULL REFERENCES teams(id),
    as_of_date      DATE    NOT NULL,                   -- 状态截止日期（赛前）

    -- 量化指标（由 processors/ 模块计算填入）
    form_score_5    NUMERIC(4,3),                       -- 近5场加权战绩 -1.0~+1.0
    form_score_10   NUMERIC(4,3),                       -- 近10场
    fatigue_index   NUMERIC(4,3),                       -- 疲劳指数 0.0~1.0
    injury_impact   NUMERIC(4,3),                       -- 伤病减损 -0.30~0
    momentum_score  NUMERIC(4,3),                       -- 心理状态 -0.10~+0.10

    -- 原始数据
    matches_last_30d SMALLINT,
    travel_km        INT,
    missing_players  JSONB,                             -- [{"name":"..","importance":0.9,"position_mult":1.3}]

    created_at      TIMESTAMP DEFAULT NOW(),
    UNIQUE (team_id, as_of_date)
);

CREATE TABLE player_injuries (
    id          SERIAL  PRIMARY KEY,
    team_id     INT     NOT NULL REFERENCES teams(id),
    player_name VARCHAR(100) NOT NULL,
    injury_type VARCHAR(100),
    status      VARCHAR(20) NOT NULL,                   -- 'out', 'doubt', 'return'
    importance  NUMERIC(3,2),                           -- 球员重要性 0.0~1.0
    position_multiplier NUMERIC(3,2),                   -- 位置乘数
    reported_at DATE    NOT NULL,
    expected_return DATE,
    source      VARCHAR(100)                            -- 数据来源
);

-- ============================================================
-- 模型预测输出
-- ============================================================

CREATE TABLE model_predictions (
    id              SERIAL  PRIMARY KEY,
    match_id        INT     NOT NULL REFERENCES matches(id),
    model_version   VARCHAR(50) NOT NULL,               -- 'dixon_coles_v1', 'xgb_v2' 等
    predicted_at    TIMESTAMP NOT NULL,

    -- 概率输出（校准后）
    p_home          NUMERIC(5,4) NOT NULL,              -- 三者之和应 = 1.0
    p_draw          NUMERIC(5,4) NOT NULL,
    p_away          NUMERIC(5,4) NOT NULL,

    -- 期望值（对应即时赔率）
    ev_home         NUMERIC(6,4),
    ev_draw         NUMERIC(6,4),
    ev_away         NUMERIC(6,4),

    -- 赔率缺口
    edge_home       NUMERIC(5,4),                       -- p_model - p_implied
    edge_draw       NUMERIC(5,4),
    edge_away       NUMERIC(5,4),

    is_calibrated   BOOLEAN DEFAULT FALSE,
    UNIQUE (match_id, model_version, predicted_at)
);

-- ============================================================
-- 串场方案与下注记录（回测 + 实盘共用）
-- ============================================================

CREATE TABLE parlay_plans (
    id          SERIAL  PRIMARY KEY,
    plan_date   DATE    NOT NULL,
    tier        VARCHAR(20) NOT NULL,                   -- 'hedge', 'core', 'aggressive'
    legs        JSONB   NOT NULL,                       -- [{match_id, outcome, odds, p_model}]
    total_odds  NUMERIC(8,3),
    win_rate    NUMERIC(5,4),
    expected_ev NUMERIC(6,4),
    kelly_pct   NUMERIC(5,4),                          -- Kelly 建议比例
    stake       NUMERIC(10,2),                         -- 实际注金
    is_simulation BOOLEAN DEFAULT TRUE,                -- TRUE=模拟，FALSE=实盘
    created_at  TIMESTAMP DEFAULT NOW()
);

CREATE TABLE bet_results (
    id          SERIAL  PRIMARY KEY,
    plan_id     INT     NOT NULL REFERENCES parlay_plans(id),
    settled_at  TIMESTAMP,
    won         BOOLEAN,
    payout      NUMERIC(10,2),
    profit      NUMERIC(10,2)
);
```

### 字段约定

| 约定 | 说明 |
|---|---|
| 概率字段 | `NUMERIC(5,4)`，范围 0.0000~1.0000 |
| 赔率字段 | `NUMERIC(6,3)`，范围 1.000~999.999 |
| 金额字段 | `NUMERIC(10,2)`，单位为货币最小单位 |
| 时区 | 所有 TIMESTAMP 存储 UTC，应用层转换 |
| JSONB | 复杂结构用 JSONB，便于查询和索引 |

---

## 14. 附录 B：模块间接口契约

> 每个模块之间通过标准化数据结构传递数据，解耦模块实现。
> 实现方式：Python `TypedDict` + 运行时校验（pydantic 可选）

### 接口总览

```
模块一（数据采集）
    └─→ MatchFeatures        ─→ 模块二（预测模型）
                                    └─→ ModelRawOutput   ─→ 模块三（校准层）
                                                              └─→ CalibratedPrediction ─→ 模块四（优化器）
                                                                                              └─→ ParlayPlan ─→ 模块五（资金分配）
                                                                                                                    └─→ BetRecord ─→ 模块六（回测）
```

---

### 接口一：模块一 → 模块二

```python
from typing import TypedDict, Optional

class MatchFeatures(TypedDict):
    """
    模块一输出，模块二输入。
    每场比赛一个 MatchFeatures 实例。
    """
    # 基础信息
    match_id:       int
    league_id:      str             # 'E0', 'SP1' 等
    match_date:     str             # 'YYYY-MM-DD'
    match_week:     int
    home_team_id:   int
    away_team_id:   int

    # 近期战绩（指数加权，-1.0 ~ +1.0）
    home_form_5:    float
    away_form_5:    float
    home_form_10:   float
    away_form_10:   float

    # 进攻/防守强度（近10场场均）
    home_goals_scored_avg:    float
    home_goals_conceded_avg:  float
    away_goals_scored_avg:    float
    away_goals_conceded_avg:  float

    # 状态特征
    home_fatigue:         float     # 0.0 ~ 1.0
    away_fatigue:         float
    home_injury_impact:   float     # -0.30 ~ 0.0
    away_injury_impact:   float
    home_momentum:        float     # -0.10 ~ +0.10
    away_momentum:        float

    # 场景特征
    days_rest_home:   int           # 距上场天数
    days_rest_away:   int

    # 赔率特征（即时赔率，去水钱后）
    odds_home:        float
    odds_draw:        float
    odds_away:        float
    p_implied_home:   float         # 赔率隐含概率（去水钱）
    p_implied_draw:   float
    p_implied_away:   float
    odds_drift_home:  float         # 开盘到即时漂移幅度（正=赔率上升=隐含概率下降）

    # 异常标记
    smart_money_flag: bool          # True = 聪明钱信号，主胜概率应上调
    exclude_flag:     bool          # True = 赔率异常，不进入串场候选
```

---

### 接口二：模块二 → 模块三

```python
class ModelRawOutput(TypedDict):
    """
    模块二输出（未校准），模块三输入。
    """
    match_id:       int
    model_version:  str             # 'dixon_coles_v1', 'xgb_v2'
    predicted_at:   str             # ISO 时间字符串

    # 未校准概率（三者之和应 = 1.0）
    p_home_raw:     float
    p_draw_raw:     float
    p_away_raw:     float

    # Dixon-Coles 额外输出（可选）
    lambda_home:    Optional[float] # 主队进球期望值
    lambda_away:    Optional[float] # 客队进球期望值
```

---

### 接口三：模块三 → 模块四

```python
class CalibratedPrediction(TypedDict):
    """
    模块三输出（已校准），模块四输入。
    同时携带赔率信息，便于模块四直接计算期望值。
    """
    match_id:       int
    model_version:  str

    # 校准后概率（三者之和 = 1.0）
    p_home:         float
    p_draw:         float
    p_away:         float

    # 即时赔率（来自模块一透传）
    odds_home:      float
    odds_draw:      float
    odds_away:      float

    # 期望值（= p × odds）
    ev_home:        float
    ev_draw:        float
    ev_away:        float

    # 赔率缺口（= p_model - p_implied）
    edge_home:      float
    edge_draw:      float
    edge_away:      float

    # 继承自模块一的异常标记
    smart_money_flag: bool
    exclude_flag:     bool
```

---

### 接口四：模块四 → 模块五

```python
class ParlayLeg(TypedDict):
    """单个串场腿（一场比赛的一个投注方向）"""
    match_id:   int
    outcome:    str             # 'home', 'draw', 'away'
    odds:       float
    p_model:    float           # 模型预测概率
    ev:         float           # 期望值
    edge:       float           # 赔率缺口

class ParlayOption(TypedDict):
    """单条串场方案"""
    tier:       str             # 'hedge', 'core', 'aggressive'
    legs:       list[ParlayLeg]
    total_odds: float           # ∏ odds
    win_rate:   float           # ∏ p_model
    expected_ev: float          # win_rate × total_odds

class ParlayPlan(TypedDict):
    """
    模块四输出，模块五输入。
    包含当日所有串场方案。
    """
    plan_date:  str
    options:    list[ParlayOption]  # 通常 3 条：hedge/core/aggressive
    total_budget: float             # 当日总可用资金
```

---

### 接口五：模块五 → 模块六

```python
class BetRecord(TypedDict):
    """
    模块五输出，模块六输入。
    每注串场一条记录。
    """
    plan_id:        str             # 唯一标识，用于关联结果
    plan_date:      str
    tier:           str
    legs:           list[ParlayLeg]
    total_odds:     float
    win_rate:       float
    stake:          float           # 实际注金（Kelly计算后）
    kelly_pct:      float           # Kelly 建议比例
    is_simulation:  bool            # True=模拟回测，False=实盘

    # 结果字段（回测时填入，实盘时赛后更新）
    won:            Optional[bool]
    payout:         Optional[float]
    profit:         Optional[float]
    settled_at:     Optional[str]
```

---

### 接口约定

| 约定 | 说明 |
|---|---|
| 概率值 | 所有概率字段为 `float`，范围 `[0.0, 1.0]`，同一组三值之和应等于 `1.0`（容差 `1e-6`） |
| 日期格式 | 统一 `'YYYY-MM-DD'` 字符串 |
| 时间格式 | 统一 ISO 8601，带时区 `'2026-05-11T14:30:00+00:00'` |
| 空值处理 | 使用 `Optional[T]`，禁止用 `0` 或 `-1` 代替空值 |
| 验证时机 | 每个模块在接收数据时主动校验，拒绝不合法输入而非静默修正 |

---

## 15. 附录 C：仓库结构

```
pitch-odds/
├── README.md
├── requirements.txt
├── docker-compose.yml
│
├── docs/
│   └── design.md                  # 本文件
│
├── data/                          # 数据采集
│   ├── collectors/
│   │   ├── historical.py          # 历史数据采集
│   │   ├── odds.py                # 实时赔率采集
│   │   └── team_status.py         # 球队状态采集
│   ├── processors/
│   │   ├── form_score.py          # 近期战绩量化
│   │   ├── fatigue.py             # 疲劳指数
│   │   ├── injury.py              # 伤病影响
│   │   └── odds_anomaly.py        # 赔率异常检测
│   └── storage/
│       ├── models.py              # 数据库表定义（SQLAlchemy ORM）
│       ├── schema.sql             # 原始 DDL（见附录 A）
│       └── migrations/            # Alembic 迁移脚本
│
├── interfaces/
│   └── contracts.py               # 所有 TypedDict 接口定义（见附录 B）
│
├── models/                        # 预测模型
│   ├── base.py                    # 模型基类
│   ├── dixon_coles.py             # Dixon-Coles 泊松模型
│   ├── xgboost_model.py           # XGBoost 特征模型
│   ├── world_cup.py               # 世界杯专用模型
│   └── calibration.py             # 概率校准
│
├── optimizer/                     # 串场优化
│   ├── ev_filter.py               # 正期望筛选
│   ├── parlay_optimizer.py        # 串场组合优化
│   └── system_bet.py              # 系统投注逻辑
│
├── capital/                       # 资金管理
│   ├── kelly.py                   # Kelly 公式
│   ├── allocator.py               # 三层分配策略
│   └── stop_loss.py               # 止损机制
│
├── backtest/                      # 回测框架
│   ├── engine.py                  # 回测引擎（时间顺序滚动）
│   ├── metrics.py                 # ROI/回撤/夏普计算
│   └── report.py                  # 回测报告生成
│
├── dashboard/                     # 可视化
│   └── app.py                     # Streamlit 仪表板
│
└── tests/                         # 单元测试
    ├── test_models.py
    ├── test_optimizer.py
    └── test_capital.py
```

---

*文档版本：v0.3 | 项目：pitch-odds | 最后更新：2026-05 | 状态：设计阶段*
