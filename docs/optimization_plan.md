# 算法优化计划

> 依据：基于代码审计（2026-05-18）对 Phase 1~6 实现的差距分析  
> 当前状态：整体完成度 ~68%，流水线可运行，但预测质量受限于多处特征未接入

---

## 目录

1. [优先级总览](#1-优先级总览)
2. [OPT-01：联赛模型特征接入（λ 修正）](#2-opt-01联赛模型特征接入λ-修正)
3. [OPT-02：模型滚动更新机制](#3-opt-02模型滚动更新机制)
4. [OPT-03：赔率异动真实数据落地](#4-opt-03赔率异动真实数据落地)
5. [OPT-04：疲劳指数数据完善](#5-opt-04疲劳指数数据完善)
6. [OPT-05：串场腿相关性修正](#6-opt-05串场腿相关性修正)
7. [OPT-06：伤病数据接入](#7-opt-06伤病数据接入)
8. [OPT-07：模型集成（XGBoost 增强层）](#8-opt-07模型集成xgboost-增强层)
9. [参数调优指南](#9-参数调优指南)

---

## 1. 优先级总览

| 编号 | 优化项 | 优先级 | 难度 | 预期收益 | 涉及文件 |
|------|--------|--------|------|----------|----------|
| OPT-01 | form/momentum 接入联赛模型 | 🔴 高 | 低 | 预测质量显著提升 | `models/dixon_coles.py` |
| OPT-02 | 滚动/增量模型更新 | 🔴 高 | 中 | 赛季中后段准确率提升 | `scripts/train.py`、`backtest/engine.py` |
| OPT-03 | 赔率异动真实落地 | 🟠 中 | 中 | 高风险场次过滤更准确 | `data/processors/odds_anomaly.py`、采集层 |
| OPT-04 | 疲劳指数完善 | 🟠 中 | 中 | 密集赛程场次预测改善 | `data/processors/fatigue.py` |
| OPT-05 | 串场腿相关性修正 | 🟠 中 | 高 | 串场胜率估算更真实 | `optimizer/parlay_optimizer.py` |
| OPT-06 | 伤病数据接入 | 🟡 低 | 高 | 重要球员缺阵场次改善 | `data/processors/injury.py` |
| OPT-07 | XGBoost 模型集成 | 🟡 低 | 高 | 天花板提升 | `models/` 新增 |

---

## 2. OPT-01：联赛模型特征接入（λ 修正）

### 问题描述

`DixonColesModel.predict()` 当前只用 `home_team_id` / `away_team_id` 查攻防参数表，完全忽略已计算好的 `home_form_5`、`away_form_5`、`home_momentum`、`away_momentum` 等字段。这些特征在 `MatchFeatures` 中存在，但对最终概率输出零贡献。

**对比**：`WorldCupModel.predict()` 已正确实现了 momentum → Elo 微调（第94行），联赛侧应参照此方式补充。

### 根因定位

```
models/dixon_coles.py  第 121~122 行

lambda_home = exp(attack_home + defense_away + self._gamma)
lambda_away = exp(attack_away + defense_home)
# ← form / momentum / fatigue 字段从未被读取
```

### 方案 A：λ 乘法修正（推荐，改动最小）

在计算出基础 λ 后，叠加状态系数修正：

```python
# models/dixon_coles.py  predict() 修改

lambda_home = exp(attack_home + defense_away + self._gamma)
lambda_away = exp(attack_away + defense_home)

# --- 状态修正（新增）---
home_form     = float(features.get("home_form_5", 0.0))   # [-1, 1]
away_form     = float(features.get("away_form_5", 0.0))
home_momentum = float(features.get("home_momentum", 0.0)) # [-0.10, 0.10]
away_momentum = float(features.get("away_momentum", 0.0))
home_fatigue  = float(features.get("home_fatigue",  0.0)) # [0, 1]
away_fatigue  = float(features.get("away_fatigue",  0.0))

lambda_home *= max(0.5, 1.0 + home_form * self.form_weight
                            + home_momentum
                            - home_fatigue * self.fatigue_weight)
lambda_away *= max(0.5, 1.0 + away_form * self.form_weight
                            + away_momentum
                            - away_fatigue * self.fatigue_weight)
```

新增两个可调超参数到 `__init__`：

```python
def __init__(
    self,
    *,
    decay_xi: float = 0.0018,
    max_goals: int = 10,
    form_weight: float = 0.08,     # 新增：状态分对 λ 的影响系数
    fatigue_weight: float = 0.05,  # 新增：疲劳指数对 λ 的抑制系数
) -> None:
```

`max(0.5, ...)` 的下限防止极端状态把 λ 压到趋近于零。

### 方案 B：Dixon-Coles 扩展参数（更严谨，改动较大）

把 form/fatigue 作为额外项加入 λ 的指数表达式，用 MLE 同时学习系数：

```python
# 目标函数 objective() 中：
lambda_home = exp(
    attack[home_i] + defense[away_i] + gamma
    + alpha_form    * match.home_form_5
    + alpha_fatigue * (-match.home_fatigue)
)
```

需要在 `_TrainingMatch` 中存储 form/fatigue 字段，并在 `fit()` 中传入。

### 验收标准

- 有 form/momentum 数据时，同一场比赛预测概率与无特征时不同（可用单元测试断言）
- 连胜球队的 `lambda_home` 高于其历史平均
- 连败球队的 `lambda_away` 低于其历史平均
- 所有概率三元组仍满足 `sum == 1.0 ± 1e-6`
- 在验证集上 Brier Score 优于未修正版本

---

## 3. OPT-02：模型滚动更新机制

### 问题描述

当前 `train.py` 一次性训练 4 个赛季，整赛季用同一批参数预测。赛季开始时球队能力分布和赛季末期存在显著差异（伤病、主帅变动、状态波动），固定参数误差累积严重。

### 方案：Walk-Forward 滚动训练

每经过 `retrain_every_n_matches` 场比赛，用滑动窗口重新拟合模型：

```python
# scripts/train_rolling.py  新增脚本

WINDOW_SEASONS = ["2021-22", "2022-23", "2023-24"]  # 滑动窗口
RETRAIN_EVERY  = 10                                  # 每10场重训一次

def rolling_train(league_id: str, window_rows: list[dict]) -> DixonColesModel:
    cutoff = max(r["match_date"] for r in window_rows)
    model  = DixonColesModel()
    model.fit(_attach_cutoff(window_rows, cutoff), league_id)
    return model
```

在 `backtest/engine.py` 的 `run_backtest_from_rows()` 中支持 `rolling=True` 模式：

```python
# backtest/engine.py  run_backtest_from_rows() 新增参数

def run_backtest_from_rows(
    *,
    rows: list[dict],
    league_id: str,
    train_seasons: list[str],
    val_season: str,
    test_season: str,
    rolling: bool = False,         # 新增
    retrain_every: int = 10,       # 新增
) -> BacktestResult:
    ...
    if rolling:
        # 测试集按 retrain_every 分批，每批重训一次
        return _rolling_backtest(rows, ...)
    else:
        # 原有逻辑不变
        ...
```

### 验收标准

- `rolling=True` 模式下，测试集每 `retrain_every` 场比赛用一个新训练的模型预测
- 不引入未来数据（每次重训的 cutoff = 当前批次的最后一场）
- `pred.train_until < pred.match_date` 断言在滚动模式下仍然通过
- 与 `rolling=False` 对比，在同一测试集上输出两份 Brier Score 对比报告

---

## 4. OPT-03：赔率异动真实数据落地

### 问题描述

`detect_odds_anomaly()` 处理器已实现，但以下字段在整个流水线中写死为默认值：

```python
# backtest/engine.py _build_features() 和 scripts/train.py _build_features()
"odds_drift_home": 0.0,      # 始终为 0
"smart_money_flag": False,   # 始终为 False
"exclude_flag": False,       # 始终为 False
```

### 方案

**Step 1**：扩展数据库 Schema，在 `odds_opening` 或新增 `odds_movement` 表中存储多个时间点的赔率快照：

```sql
CREATE TABLE odds_movement (
    id          SERIAL PRIMARY KEY,
    match_id    INTEGER REFERENCES matches(id),
    bookmaker   VARCHAR(50),
    recorded_at TIMESTAMPTZ NOT NULL,
    odds_home   NUMERIC(6,3),
    odds_draw   NUMERIC(6,3),
    odds_away   NUMERIC(6,3)
);
```

**Step 2**：在特征构建时查询同一场比赛的赔率时序，调用 `detect_odds_anomaly()`：

```python
# backtest/engine.py _build_features() 新增

def _build_features(row: dict, odds_series: list[float] | None = None) -> MatchFeatures:
    anomaly = detect_odds_anomaly(odds_series) if odds_series else {
        "alert_level": "NORMAL", "exclude_from_parlay": False,
        "smart_money": False, "total_drift_pct": 0.0,
    }
    return {
        ...
        "odds_drift_home":  anomaly["total_drift_pct"] / 100.0,
        "smart_money_flag": anomaly["smart_money"],
        "exclude_flag":     anomaly["exclude_from_parlay"],
    }
```

**Step 3**：数据采集层补充多时点赔率爬取（开盘 / 比赛前24h / 比赛前2h）。

### 验收标准

- 赔率漂移 > 15% 的场次 `exclude_flag=True`，不进入串场候选
- 急剧单步变动 > 8% 的场次 `smart_money_flag=True`
- 在回测中统计因赔率异动被过滤的场次数量，并输出过滤前后 ROI 对比

---

## 5. OPT-04：疲劳指数数据完善

### 问题描述

`fatigue_index()` 接受三个参数，但 `travel_km` 和 `minutes_played_key_players` 始终传入 `0.0`：

```python
# backtest/engine.py 第 135 行
fat = _calc_fatigue(matches_last_30d, 0.0, 0.0)
```

当前有效的只有"近30天场次密度"分量，旅行距离和球员负荷被完全忽略。

### 方案

**阶段一（仅用现有数据）**：用赛程数据估算旅行距离（主客场城市距离）：

```python
# data/processors/fatigue.py  新增辅助函数

STADIUM_COORDS: dict[int, tuple[float, float]] = {
    # team_id: (纬度, 经度)
    # 从 football-data.org 或静态配置文件读取
}

def estimate_travel_km(team_id: int, recent_away_matches: list[dict]) -> float:
    """估算近30天客场旅行总公里数。"""
    total = 0.0
    home_coord = STADIUM_COORDS.get(team_id)
    if not home_coord:
        return 0.0
    for m in recent_away_matches:
        opp_coord = STADIUM_COORDS.get(m["opponent_id"])
        if opp_coord:
            total += _haversine(home_coord, opp_coord) * 2  # 来回
    return total
```

**阶段二（需外部数据）**：接入球员上场分钟数 API（如 football-api.com），累计主力球员近5场上场时间。

### 验收标准

- 阶段一：疲劳指数中 `travel` 分量不再为 0（至少对有城市坐标的球队）
- 客场连续作战场次的疲劳指数高于主场连续作战
- 在测试集中，高疲劳场次的预测误差是否低于修正前（Brier Score 子集对比）

---

## 6. OPT-05：串场腿相关性修正

### 问题描述

`find_optimal_parlay()` 中串场胜率计算假设各腿完全独立：

```python
# optimizer/parlay_optimizer.py 第 45~46 行
win_rate *= leg["p_model"]    # ← 独立性假设
total_odds *= leg["odds"]
```

实际上同日同联赛多场比赛存在相关性：恶劣天气、重要节点轮休、联赛排位焦灼期等因素会同时影响多场比赛，独立性假设会系统性高估串场胜率。

### 方案：简化相关性折扣

引入可配置的相关性惩罚系数，对同联赛同日多腿组合的胜率做折扣：

```python
# optimizer/parlay_optimizer.py  find_optimal_parlay() 修改

def _correlation_discount(combo: tuple[ParlayLeg, ...]) -> float:
    """
    估算串场腿间的正相关性折扣因子。
    同联赛同日腿数越多，折扣越大。
    
    简化模型：每多一条同联赛同日的腿，胜率乘以 (1 - SAME_LEAGUE_PENALTY)。
    """
    SAME_LEAGUE_PENALTY = 0.02   # 每条同联赛腿降低2%胜率
    
    league_day_counts: dict[str, int] = {}
    for leg in combo:
        key = f"{leg.get('league_id', '')}_{leg.get('match_date', '')}"
        league_day_counts[key] = league_day_counts.get(key, 0) + 1
    
    discount = 1.0
    for count in league_day_counts.values():
        if count > 1:
            discount *= (1 - SAME_LEAGUE_PENALTY) ** (count - 1)
    return discount

# 在计算 expected_ev 时应用折扣：
adjusted_win_rate = win_rate * _correlation_discount(combo)
expected_ev = adjusted_win_rate * total_odds
```

`ParlayLeg` 需要补充 `league_id` 和 `match_date` 字段（目前缺失）。

### 验收标准

- 同联赛同日 4 腿组合的实际胜率低于独立假设下的计算值（历史回测验证）
- `SAME_LEAGUE_PENALTY` 可通过验证集拟合：令历史4腿串场的平均实际胜率 ≈ 修正后预测胜率

---

## 7. OPT-06：伤病数据接入

### 问题描述

`injury_impact()` 处理器已实现完整逻辑，但 `home_injury_impact` / `away_injury_impact` 始终为 `0.0`，因为没有伤病数据源。

### 方案

**数据源选项**：
- [football-data.org](https://www.football-data.org)（免费套餐含部分球员数据）
- [API-Football](https://www.api-football.com)（含详细伤病名单，收费）
- [Transfermarkt 爬虫](https://www.transfermarkt.com)（需自行解析 HTML）

**接入框架**：

```python
# data/collectors/injury.py  新增采集器

def fetch_injury_list(match_id: int, team_id: int, api_client) -> list[MissingPlayer]:
    """
    从外部 API 拉取赛前伤病名单，转换为 MissingPlayer 格式。
    
    importance 评分参考：主力核心=1.0，轮换=0.6，替补=0.3
    position_multiplier：前锋=1.2，中场=1.0，后卫=0.9，门将=0.8
    """
    raw = api_client.get_injuries(match_id=match_id, team_id=team_id)
    return [
        {
            "importance": _estimate_importance(p["minutes_played_season"]),
            "position_multiplier": POSITION_MULTIPLIERS[p["position"]],
        }
        for p in raw["players"]
        if p["status"] in ("Injured", "Suspended")
    ]
```

在 `_build_features()` 中调用后传入 `interfaces/contracts.py` 的约束范围 `[-0.30, 0.0]`。

### 验收标准

- 有主力球员缺阵时，对应 `injury_impact` 为负值（非零）
- `injury_impact` 在 `[-0.30, 0.0]` 范围内通过 `validate_match_features()` 校验
- 缺阵场次的预测误差对比未使用伤病数据时有改善

---

## 8. OPT-07：模型集成（XGBoost 增强层）

### 问题描述

Dixon-Coles 是参数化统计模型，天然擅长捕捉攻防能力的长期均值，但对非线性特征交互（如"连胜球队遭遇疲惫客场强队"这类组合效应）拟合能力有限。

### 方案：两阶段集成

**第一层**：Dixon-Coles 输出原始概率 `(p_home_raw, p_draw_raw, p_away_raw)` + `lambda_home` + `lambda_away`

**第二层**：XGBoost 以第一层输出 + 所有 `MatchFeatures` 字段为特征，预测修正后的三分类概率：

```python
# models/xgb_ensemble.py  新增

import xgboost as xgb

class XGBEnsembleModel:
    """
    以 Dixon-Coles 输出 + MatchFeatures 为输入，
    用 XGBoost 预测修正概率（三分类 softmax）。
    
    输入特征（共约 20 维）：
        p_home_raw, p_draw_raw, p_away_raw   ← Dixon-Coles 输出
        lambda_home, lambda_away
        home_form_5, away_form_5
        home_form_10, away_form_10
        home_fatigue, away_fatigue
        home_momentum, away_momentum
        home_injury_impact, away_injury_impact
        days_rest_home, days_rest_away
        odds_drift_home, smart_money_flag
    """
    
    def fit(self, X: list[dict], y: list[str]) -> None:
        ...  # 用验证集训练
    
    def predict(self, X: dict) -> tuple[float, float, float]:
        ...  # 输出 (p_home, p_draw, p_away)
```

**训练数据切割**：第二层用与 Isotonic 校准相同的验证集训练，保持防泄漏纪律。

### 验收标准

- 集成模型在测试集上 Brier Score 优于纯 Dixon-Coles
- 集成模型与 Dixon-Coles 的预测差异分布有统计显著性（不只是噪声）
- 在验证集上的过拟合程度可接受（训练集 vs 验证集 Brier Score 差距 < 0.01）

---

## 9. 参数调优指南

### 调优原则

**只在验证集（2022-23）上调参，测试集（2024-25）只用一次做最终评估。**

### 关键参数与调优方法

| 参数 | 当前值 | 调优方法 | 预期范围 |
|------|--------|----------|----------|
| `decay_xi` | 0.0018 | 网格搜索：[0.001, 0.0015, 0.002, 0.003]，选验证集 Brier 最低 | 0.001~0.003 |
| `form_weight` | _(待新增)_ 0.08 | 网格搜索：[0.04, 0.08, 0.12]，选验证集 Brier 最低 | 0.04~0.15 |
| `fatigue_weight` | _(待新增)_ 0.05 | 同上 | 0.02~0.10 |
| `safety_margin` | 1.05 | 若回测 ROI < -5%，上调至 1.08；若正期望候选过少，下调至 1.03 | 1.03~1.10 |
| `L2正则系数` | 0.001 | 若某联赛数据量 < 200 场，上调至 0.005~0.01 | 0.001~0.01 |
| 冲击层预算 | 20% | 回测冲击层命中率 < 2% → 削减至 10% | 10%~20% |
| `SAME_LEAGUE_PENALTY` | _(待新增)_ 0.02 | 用历史串场数据拟合，令预测胜率 ≈ 实际命中率 | 0.01~0.05 |

### 推荐调优顺序

```
1. 先完成 OPT-01（特征接入），重新跑验证集基线
2. 调 decay_xi（影响最大，先定下来）
3. 调 form_weight / fatigue_weight（OPT-01 依赖）
4. 调 safety_margin（基于 ROI 观察）
5. 其余参数次之
```

### 评估指标优先级

```
Brier Score（概率校准质量）> ROI（收益）> Sharpe Ratio（风险调整收益）
```

随机猜测 Brier Score ≈ 0.333，好模型目标 ≤ 0.200。若 Brier Score 已达标但 ROI 仍为负，问题通常在 EV 阈值或串场策略，而非模型本身。
