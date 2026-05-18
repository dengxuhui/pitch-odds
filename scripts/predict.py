"""predict.py — 未来比赛购买建议脚本

读取用户提供的即将进行的比赛 CSV，加载数据库中已训练的模型参数，
输出正期望场次、三层串场方案和 Half Kelly 建议注金。

CSV 格式（支持球队名称或 ID 两种形式）：
    match_id,home_team,away_team,match_date,odds_home,odds_draw,odds_away
    或：
    match_id,home_team_id,away_team_id,match_date,odds_home,odds_draw,odds_away

示例：
    python3 scripts/predict.py \\
        --league E0 \\
        --input upcoming.csv \\
        --budget 1000 \\
        --safety-margin 1.05
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import date, datetime, timezone
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from sqlalchemy import desc, select

from capital.kelly import half_kelly
from data.storage.db import SessionLocal
from data.storage.models import ModelParams, Team
from models.calibration import IsotonicThreeWayCalibrator
from models.dixon_coles import DixonColesModel
from optimizer.ev_filter import filter_positive_ev
from optimizer.parlay_optimizer import build_parlay_plan


# ──────────────────────────────────────────────
# 数据加载
# ──────────────────────────────────────────────

def _load_model_params(league_id: str) -> dict:
    """从数据库加载最新的模型参数。"""
    with SessionLocal() as session:
        row = session.execute(
            select(ModelParams)
            .where(ModelParams.league_id == league_id)
            .order_by(desc(ModelParams.trained_at))
            .limit(1)
        ).scalar_one_or_none()

    if row is None:
        raise RuntimeError(
            f"数据库中没有联赛 {league_id} 的模型参数，请先运行：\n"
            f"  python3 scripts/train.py --league {league_id} "
            f"--train-seasons 2018-19,... --val-season 2022-23"
        )
    return row.params  # type: ignore[return-value]


def _resolve_team_ids(league_id: str, names: list[str]) -> dict[str, int]:
    """将球队名称解析为数据库 ID（大小写不敏感，支持模糊匹配）。"""
    if not names:
        return {}
    with SessionLocal() as session:
        teams = session.execute(
            select(Team).where(Team.league_id == league_id)
        ).scalars().all()

    name_map: dict[str, int] = {}
    for team in teams:
        name_map[team.name.lower()] = team.id
        if team.short_name:
            name_map[team.short_name.lower()] = team.id

    result: dict[str, int] = {}
    for raw_name in names:
        key = raw_name.strip().lower()
        if key in name_map:
            result[raw_name] = name_map[key]
        else:
            # 模糊匹配：包含关系
            matches = [tid for tname, tid in name_map.items() if key in tname or tname in key]
            if len(matches) == 1:
                result[raw_name] = matches[0]
            else:
                raise ValueError(
                    f"无法解析球队名称 '{raw_name}'（联赛 {league_id}）。"
                    f"\n可用球队：{sorted(set(name_map.keys()))}"
                )
    return result


def _load_csv(path: Path, league_id: str) -> list[dict]:
    """读取比赛 CSV，返回标准化的 row 列表。"""
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        raw_rows = list(reader)

    if not raw_rows:
        raise ValueError(f"CSV 文件为空：{path}")

    has_team_id = "home_team_id" in raw_rows[0]

    if not has_team_id:
        # 解析球队名 → ID
        home_names = [r["home_team"] for r in raw_rows]
        away_names = [r["away_team"] for r in raw_rows]
        all_names = list(set(home_names + away_names))
        id_map = _resolve_team_ids(league_id, all_names)

    for i, r in enumerate(raw_rows):
        match_id = int(r.get("match_id", i + 1))
        odds_home = float(r["odds_home"])
        odds_draw = float(r["odds_draw"])
        odds_away = float(r["odds_away"])
        overround = (1.0 / odds_home) + (1.0 / odds_draw) + (1.0 / odds_away)

        if has_team_id:
            home_team_id = int(r["home_team_id"])
            away_team_id = int(r["away_team_id"])
            label_home = str(home_team_id)
            label_away = str(away_team_id)
        else:
            home_name = r["home_team"]
            away_name = r["away_team"]
            home_team_id = id_map[home_name]
            away_team_id = id_map[away_name]
            label_home = home_name
            label_away = away_name

        rows.append({
            "match_id": match_id,
            "league_id": league_id,
            "match_date": r["match_date"],
            "home_team_id": home_team_id,
            "away_team_id": away_team_id,
            "label_home": label_home,
            "label_away": label_away,
            "odds_home": odds_home,
            "odds_draw": odds_draw,
            "odds_away": odds_away,
            "p_implied_home": (1.0 / odds_home) / overround,
            "p_implied_draw": (1.0 / odds_draw) / overround,
            "p_implied_away": (1.0 / odds_away) / overround,
        })
    return rows


def _build_features(row: dict) -> dict:
    return {
        "match_id": row["match_id"],
        "league_id": row["league_id"],
        "match_date": row["match_date"],
        "match_week": 0,
        "home_team_id": row["home_team_id"],
        "away_team_id": row["away_team_id"],
        "home_form_5": 0.0,
        "away_form_5": 0.0,
        "home_form_10": 0.0,
        "away_form_10": 0.0,
        "home_goals_scored_avg": 0.0,
        "home_goals_conceded_avg": 0.0,
        "away_goals_scored_avg": 0.0,
        "away_goals_conceded_avg": 0.0,
        "home_fatigue": 0.0,
        "away_fatigue": 0.0,
        "home_injury_impact": 0.0,
        "away_injury_impact": 0.0,
        "home_momentum": 0.0,
        "away_momentum": 0.0,
        "days_rest_home": 7,
        "days_rest_away": 7,
        "odds_home": row["odds_home"],
        "odds_draw": row["odds_draw"],
        "odds_away": row["odds_away"],
        "p_implied_home": row["p_implied_home"],
        "p_implied_draw": row["p_implied_draw"],
        "p_implied_away": row["p_implied_away"],
        "odds_drift_home": 0.0,
        "smart_money_flag": False,
        "exclude_flag": False,
    }


# ──────────────────────────────────────────────
# 输出格式
# ──────────────────────────────────────────────

_OUTCOME_CN = {"home": "主胜", "draw": "平局", "away": "客胜"}
_TIER_CN = {"hedge": "保底层（2~3串）", "core": "核心层（4~5串）", "aggressive": "冲击层（6~7串）"}
_TIER_BUDGETS = {"hedge": 0.40, "core": 0.40, "aggressive": 0.20}


def _print_ev_table(ev_legs: list, label_map: dict[int, tuple[str, str]]) -> None:
    print("\n【正期望候选场次】")
    print(f"  {'场次':<30} {'投注方向':<8} {'赔率':>6} {'模型概率':>8} {'EV':>7} {'优势Edge':>9}")
    print("  " + "-" * 72)
    for leg in ev_legs:
        mid = leg["match_id"]
        home_label, away_label = label_map.get(mid, (str(mid), "?"))
        match_label = f"{home_label} vs {away_label}"[:30]
        direction = _OUTCOME_CN.get(leg["outcome"], leg["outcome"])
        print(
            f"  {match_label:<30} {direction:<8} {leg['odds']:>6.2f} "
            f"{leg['p_model']:>8.2%} {leg['ev']:>7.3f} {leg['edge']:>+9.4f}"
        )


def _print_parlay_plan(plan: dict, budget: float) -> None:
    print("\n【三层串场建议】")
    for option in plan["options"]:
        tier = option["tier"]
        tier_budget = budget * _TIER_BUDGETS.get(tier, 0.0)
        kelly_f = half_kelly(option["win_rate"], option["total_odds"])
        stake = round(tier_budget * kelly_f, 2)

        print(f"\n  ▶ {_TIER_CN.get(tier, tier)}")
        print(f"    组合赔率: {option['total_odds']:.2f}  胜率: {option['win_rate']:.2%}  EV: {option['expected_ev']:.3f}")
        print(f"    本层预算: {tier_budget:.0f}元  Half Kelly 比例: {kelly_f:.2%}  建议注金: {stake:.0f}元")
        print("    选腿：")
        for leg in option["legs"]:
            direction = _OUTCOME_CN.get(leg["outcome"], leg["outcome"])
            print(f"      · {leg['match_id']} {direction} @{leg['odds']:.2f} (p={leg['p_model']:.2%})")


# ──────────────────────────────────────────────
# 主流程
# ──────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="未来比赛购买建议 — 加载已训练模型，对即将进行的比赛输出串场推荐",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--league", default="E0", help="联赛 ID（E0/SP1/D1/I1/F1），默认 E0")
    parser.add_argument("--input", required=True, help="包含即将进行比赛的 CSV 文件路径")
    parser.add_argument("--budget", type=float, default=1000.0, help="总投注预算（元），默认 1000")
    parser.add_argument("--safety-margin", type=float, default=1.05, help="EV 安全边际，默认 1.05")
    parser.add_argument("--plan-date", default=None, help="方案日期 YYYY-MM-DD，默认今日")
    parser.add_argument("--output-dir", default=None, help="可选：将推荐结果保存为 JSON 的目录")
    args = parser.parse_args()

    plan_date = args.plan_date or date.today().isoformat()
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"[错误] 找不到文件：{input_path}", file=sys.stderr)
        sys.exit(1)

    # 1. 加载模型参数
    print(f"[1/4] 加载模型参数（{args.league}）...")
    params = _load_model_params(args.league)

    model = DixonColesModel()
    model.load_params(params["model"])

    calibrator = IsotonicThreeWayCalibrator()
    calibrator.load_params(params["calibrator"])
    print(f"      模型版本: {params['model'].get('league_id', args.league)}  "
          f"训练赛季: {params.get('train_seasons', [])}  验证赛季: {params.get('val_season', '')}")

    # 2. 读取比赛数据
    print(f"[2/4] 读取比赛数据：{input_path}")
    rows = _load_csv(input_path, args.league)
    label_map: dict[int, tuple[str, str]] = {
        r["match_id"]: (r["label_home"], r["label_away"]) for r in rows
    }
    print(f"      共 {len(rows)} 场比赛")

    # 3. 预测 + 校准
    print("[3/4] 运行模型预测与概率校准...")
    calibrated = []
    for row in rows:
        features = _build_features(row)
        raw = model.predict(features)
        cal = calibrator.calibrate(raw, features)
        calibrated.append(cal)

    # 4. 正期望筛选 + 串场 + Kelly
    print(f"[4/4] 正期望筛选（EV ≥ {args.safety_margin}）+ 生成串场方案...")
    ev_legs = filter_positive_ev(calibrated, safety_margin=args.safety_margin)

    # ── 输出预测概率总览 ──
    print("\n【全部场次预测概率】")
    print(f"  {'场次':<30} {'主胜':>8} {'平局':>8} {'客胜':>8}  {'主赔':>6} {'平赔':>6} {'客赔':>6}")
    print("  " + "-" * 72)
    for cal, row in zip(calibrated, rows):
        home_label, away_label = label_map[row["match_id"]]
        match_label = f"{home_label} vs {away_label}"[:30]
        print(
            f"  {match_label:<30} {cal['p_home']:>8.2%} {cal['p_draw']:>8.2%} {cal['p_away']:>8.2%}"
            f"  {row['odds_home']:>6.2f} {row['odds_draw']:>6.2f} {row['odds_away']:>6.2f}"
        )

    if not ev_legs:
        print(f"\n[结果] 当前无正期望场次（EV ≥ {args.safety_margin}），建议不投注。")
    else:
        _print_ev_table(ev_legs, label_map)
        try:
            plan = build_parlay_plan(ev_legs, plan_date=plan_date, total_budget=args.budget)
            _print_parlay_plan(plan, args.budget)
        except ValueError as exc:
            print(f"\n[提示] 正期望场次数量不足以生成串场：{exc}")
            print("单注推荐：")
            for leg in ev_legs[:3]:
                home_label, away_label = label_map.get(leg["match_id"], (str(leg["match_id"]), "?"))
                direction = _OUTCOME_CN.get(leg["outcome"], leg["outcome"])
                kelly_f = half_kelly(leg["p_model"], leg["odds"])
                stake = round(args.budget * kelly_f, 2)
                print(f"  {home_label} vs {away_label}  {direction} @{leg['odds']:.2f}  "
                      f"建议注金: {stake:.0f}元（Half Kelly {kelly_f:.2%}）")

    # ── 可选：保存 JSON ──
    if args.output_dir:
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        out_path = out_dir / f"predict_{args.league}_{ts}.json"
        payload = {
            "league_id": args.league,
            "plan_date": plan_date,
            "total_budget": args.budget,
            "safety_margin": args.safety_margin,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "predictions": [
                {
                    "match_id": cal["match_id"],
                    "home": label_map[cal["match_id"]][0],
                    "away": label_map[cal["match_id"]][1],
                    "p_home": cal["p_home"],
                    "p_draw": cal["p_draw"],
                    "p_away": cal["p_away"],
                    "ev_home": cal["ev_home"],
                    "ev_draw": cal["ev_draw"],
                    "ev_away": cal["ev_away"],
                }
                for cal in calibrated
            ],
            "ev_candidates": list(ev_legs),
        }
        out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n[保存] 推荐结果已写入：{out_path}")


if __name__ == "__main__":
    main()
