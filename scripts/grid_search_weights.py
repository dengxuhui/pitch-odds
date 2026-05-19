"""在验证集上搜索最优 form_weight 和 fatigue_weight。

用法（在项目根目录执行）：
    python3 scripts/grid_search_weights.py

搜索范围：
    form_weight    ∈ [0.0, 0.04, 0.08, 0.12, 0.16, 0.20]
    fatigue_weight ∈ [0.0, 0.02, 0.05, 0.08, 0.10]

评估指标：验证集原始（未校准）Brier Score，越低越好。
防泄漏：model.fit() 仅使用训练赛季；验证集只用于选超参，不碰测试集。
"""
from __future__ import annotations

import sys
from itertools import product
from math import sqrt
from pathlib import Path

# 确保项目根在 sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.storage.db import SessionLocal
from backtest.engine import (
    _attach_cutoff,
    _build_features,
    _load_match_rows,
    _rows_for_seasons,
    _to_date,
    enrich_rows_with_team_features,
)
from models.dixon_coles import DixonColesModel

TRAIN_SEASONS = ["2018-19", "2019-20", "2020-21", "2021-22"]
VAL_SEASON    = "2022-23"
LEAGUES       = ["E0", "SP1", "D1", "I1", "F1"]

FORM_WEIGHTS    = [0.0, 0.04, 0.08, 0.12, 0.16, 0.20]
FATIGUE_WEIGHTS = [0.0, 0.02, 0.05, 0.08, 0.10]


def _brier_score(model: DixonColesModel, val_rows: list[dict]) -> float:
    """计算验证集上的原始（未校准）Brier Score。"""
    total = 0.0
    n = 0
    for row in val_rows:
        if row.get("result") not in {"H", "D", "A"}:
            continue
        features = _build_features(row)
        raw = model.predict(features)
        outcome = str(row["result"])
        y_h = 1.0 if outcome == "H" else 0.0
        y_d = 1.0 if outcome == "D" else 0.0
        y_a = 1.0 if outcome == "A" else 0.0
        total += (raw["p_home_raw"] - y_h) ** 2
        total += (raw["p_draw_raw"] - y_d) ** 2
        total += (raw["p_away_raw"] - y_a) ** 2
        n += 1
    return total / n if n > 0 else float("inf")


def search_league(league_id: str, session) -> dict:
    print(f"\n{'='*50}")
    print(f"  联赛: {league_id}")
    print(f"{'='*50}")

    # 加载所有赛季数据（训练 + 验证）
    all_rows = _load_match_rows(session, league_id, TRAIN_SEASONS + [VAL_SEASON])
    if not all_rows:
        print(f"  [跳过] 数据库中无 {league_id} 数据")
        return {"league_id": league_id, "best_fw": 0.08, "best_fatg": 0.05, "best_brier": None}

    # 注入 form/momentum/fatigue 特征
    enrich_rows_with_team_features(all_rows)

    train_rows = _rows_for_seasons(all_rows, TRAIN_SEASONS)
    val_rows   = _rows_for_seasons(all_rows, [VAL_SEASON])

    if not train_rows or not val_rows:
        print(f"  [跳过] 训练/验证数据不完整")
        return {"league_id": league_id, "best_fw": 0.08, "best_fatg": 0.05, "best_brier": None}

    # 训练一次（form_weight / fatigue_weight 不影响 fit）
    train_until = max(_to_date(x["match_date"]) for x in train_rows)
    model = DixonColesModel()
    model.fit(_attach_cutoff(train_rows, train_until), league_id)
    print(f"  训练完成，训练集 {len(train_rows)} 场，验证集 {len(val_rows)} 场")

    # 网格搜索
    best_brier = float("inf")
    best_fw, best_fatg = 0.0, 0.0
    print(f"  {'form_w':>8} {'fatg_w':>8} {'Brier':>10}")
    print(f"  {'-'*30}")

    for fw, fatg_w in product(FORM_WEIGHTS, FATIGUE_WEIGHTS):
        model.form_weight    = fw
        model.fatigue_weight = fatg_w
        brier = _brier_score(model, val_rows)
        marker = " ← 当前最优" if brier < best_brier else ""
        print(f"  {fw:8.2f} {fatg_w:8.2f} {brier:10.6f}{marker}")
        if brier < best_brier:
            best_brier = brier
            best_fw    = fw
            best_fatg  = fatg_w

    print(f"\n  最优: form_weight={best_fw}, fatigue_weight={best_fatg}, Brier={best_brier:.6f}")
    return {
        "league_id": league_id,
        "best_fw":    best_fw,
        "best_fatg":  best_fatg,
        "best_brier": best_brier,
    }


def main() -> None:
    print("开始验证集超参搜索（form_weight × fatigue_weight）")
    print(f"训练赛季: {TRAIN_SEASONS}")
    print(f"验证赛季: {VAL_SEASON}")
    print(f"搜索空间: {len(FORM_WEIGHTS)} × {len(FATIGUE_WEIGHTS)} = {len(FORM_WEIGHTS)*len(FATIGUE_WEIGHTS)} 组合/联赛")

    results = []
    with SessionLocal() as session:
        for league_id in LEAGUES:
            result = search_league(league_id, session)
            results.append(result)

    print(f"\n\n{'='*60}")
    print("  汇总结果（可直接用于 backtest.py 配置）")
    print(f"{'='*60}")
    print(f"  {'联赛':>6} {'form_weight':>12} {'fatigue_weight':>15} {'val_Brier':>10}")
    print(f"  {'-'*50}")
    for r in results:
        brier_str = f"{r['best_brier']:.6f}" if r["best_brier"] is not None else "   N/A"
        print(f"  {r['league_id']:>6} {r['best_fw']:>12.2f} {r['best_fatg']:>15.2f} {brier_str:>10}")

    print("\n推荐 LEAGUE_WEIGHTS 配置（复制到 scripts/backtest.py）：")
    print("LEAGUE_WEIGHTS = {")
    for r in results:
        print(f'    "{r["league_id"]}": {{"form_weight": {r["best_fw"]}, "fatigue_weight": {r["best_fatg"]}}},')
    print("}")


if __name__ == "__main__":
    main()
