"""_predict_runner.py — 内部辅助：将 predict.py 的核心逻辑封装为函数

供 run_daily.py 在流程结束时直接调用，避免用 subprocess 启动子进程。
"""
from __future__ import annotations

import json
import logging
from datetime import date, datetime, timezone
from pathlib import Path

from sqlalchemy import desc, select

from capital.kelly import half_kelly
from data.storage.db import SessionLocal
from data.storage.models import Match, ModelParams, OddsOpening, Team
from models.calibration import IsotonicThreeWayCalibrator
from models.dixon_coles import DixonColesModel
from optimizer.ev_filter import filter_positive_ev
from optimizer.parlay_optimizer import build_parlay_plan

logger = logging.getLogger(__name__)

_OUTCOME_CN = {"home": "主胜", "draw": "平局", "away": "客胜"}
_TIER_CN = {"hedge": "保底层（2~3串）", "core": "核心层（4~5串）", "aggressive": "冲击层（6~7串）"}
_TIER_BUDGETS = {"hedge": 0.40, "core": 0.40, "aggressive": 0.20}


def run_predict(
    league_id: str,
    budget: float,
    safety_margin: float = 1.05,
    plan_date: str | None = None,
    output_dir: str | None = None,
) -> None:
    """从数据库读取待预测比赛，输出串场推荐。

    Args:
        league_id:     联赛 ID（E0 / SP1 / D1 / I1 / F1）
        budget:        投注预算（元）
        safety_margin: EV 阈值，默认 1.05
        plan_date:     方案日期 YYYY-MM-DD，默认今日
        output_dir:    结果 JSON 保存目录（可选）
    """
    plan_date = plan_date or date.today().isoformat()
    logger.info(f"\n{'='*50}")
    logger.info(f"联赛 {league_id}  日期 {plan_date}  预算 {budget:.0f}元")

    # 1. 加载模型参数
    with SessionLocal() as session:
        row = session.execute(
            select(ModelParams)
            .where(ModelParams.league_id == league_id)
            .order_by(desc(ModelParams.trained_at))
            .limit(1)
        ).scalar_one_or_none()

    if row is None:
        logger.warning(
            f"[{league_id}] 数据库中无模型参数，跳过。\n"
            f"请先运行：python3 scripts/train.py --league {league_id} ..."
        )
        return

    params = row.params
    model = DixonColesModel()
    model.load_params(params["model"])
    calibrator = IsotonicThreeWayCalibrator()
    calibrator.load_params(params["calibrator"])

    # 2. 读取今日待预测比赛（有开盘赔率的场次）
    today = date.fromisoformat(plan_date)
    with SessionLocal() as session:
        rows = session.execute(
            select(Match, OddsOpening, Team, Team)
            .join(OddsOpening, OddsOpening.match_id == Match.id)
            .join(Team, Match.home_team_id == Team.id, isouter=False)
            .where(
                Match.league_id == league_id,
                Match.match_date == today,
            )
        ).all()

    if not rows:
        logger.info(f"[{league_id}] {plan_date} 无待预测比赛")
        return

    # 3. 预测 + 校准
    calibrated = []
    label_map: dict[int, tuple[str, str]] = {}

    for match, odds, home_team, _ in rows:
        with SessionLocal() as session:
            away_team = session.get(Team, match.away_team_id)

        label_map[match.id] = (home_team.name, away_team.name if away_team else "?")

        odds_home = float(odds.odds_home)
        odds_draw = float(odds.odds_draw)
        odds_away = float(odds.odds_away)
        overround = 1 / odds_home + 1 / odds_draw + 1 / odds_away

        features = {
            "match_id": match.id,
            "league_id": league_id,
            "match_date": str(match.match_date),
            "match_week": 0,
            "home_team_id": match.home_team_id,
            "away_team_id": match.away_team_id,
            **{k: 0.0 for k in [
                "home_form_5", "away_form_5", "home_form_10", "away_form_10",
                "home_goals_scored_avg", "home_goals_conceded_avg",
                "away_goals_scored_avg", "away_goals_conceded_avg",
                "home_fatigue", "away_fatigue", "home_injury_impact",
                "away_injury_impact", "home_momentum", "away_momentum",
                "odds_drift_home",
            ]},
            "days_rest_home": 7,
            "days_rest_away": 7,
            "odds_home": odds_home,
            "odds_draw": odds_draw,
            "odds_away": odds_away,
            "p_implied_home": (1 / odds_home) / overround,
            "p_implied_draw": (1 / odds_draw) / overround,
            "p_implied_away": (1 / odds_away) / overround,
            "smart_money_flag": False,
            "exclude_flag": False,
        }
        raw = model.predict(features)
        cal = calibrator.calibrate(raw, features)
        calibrated.append(cal)

    # 4. 输出预测概率
    print(f"\n【{league_id} 全场次预测概率】")
    print(f"  {'场次':<30} {'主胜':>8} {'平局':>8} {'客胜':>8}  {'主赔':>6} {'平赔':>6} {'客赔':>6}")
    print("  " + "-" * 72)
    for cal, (_, odds, _, _) in zip(calibrated, rows):
        home, away = label_map[cal["match_id"]]
        label = f"{home} vs {away}"[:30]
        print(
            f"  {label:<30} {cal['p_home']:>8.2%} {cal['p_draw']:>8.2%} {cal['p_away']:>8.2%}"
            f"  {float(odds.odds_home):>6.2f} {float(odds.odds_draw):>6.2f} {float(odds.odds_away):>6.2f}"
        )

    # 5. 正期望筛选 + 串场
    ev_legs = filter_positive_ev(calibrated, safety_margin=safety_margin)

    if not ev_legs:
        print(f"\n[{league_id}] 当前无正期望场次（EV ≥ {safety_margin}），建议不投注。")
    else:
        print(f"\n【{league_id} 正期望候选场次】")
        for leg in ev_legs:
            h, a = label_map.get(leg["match_id"], ("?", "?"))
            direction = _OUTCOME_CN.get(leg["outcome"], leg["outcome"])
            print(f"  {h} vs {a}  {direction} @{leg['odds']:.2f}  "
                  f"p={leg['p_model']:.2%}  EV={leg['ev']:.3f}")

        try:
            plan = build_parlay_plan(ev_legs, plan_date=plan_date, total_budget=budget)
            print(f"\n【{league_id} 三层串场建议】")
            for option in plan["options"]:
                tier = option["tier"]
                tier_budget = budget * _TIER_BUDGETS.get(tier, 0.0)
                kelly_f = half_kelly(option["win_rate"], option["total_odds"])
                stake = round(tier_budget * kelly_f, 2)
                print(f"\n  ▶ {_TIER_CN.get(tier, tier)}")
                print(f"    组合赔率:{option['total_odds']:.2f}  胜率:{option['win_rate']:.2%}  "
                      f"EV:{option['expected_ev']:.3f}")
                print(f"    本层预算:{tier_budget:.0f}元  Kelly:{kelly_f:.2%}  建议注金:{stake:.0f}元")
                for leg in option["legs"]:
                    h, a = label_map.get(leg["match_id"], ("?", "?"))
                    direction = _OUTCOME_CN.get(leg["outcome"], leg["outcome"])
                    print(f"      · {h} vs {a}  {direction} @{leg['odds']:.2f}")
        except ValueError as exc:
            print(f"\n[提示] 场次不足以生成串场：{exc}")

    # 6. 保存 JSON（可选）
    if output_dir:
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        out_path = out_dir / f"predict_{league_id}_{ts}.json"
        payload = {
            "league_id": league_id,
            "plan_date": plan_date,
            "total_budget": budget,
            "safety_margin": safety_margin,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "predictions": [
                {
                    "match_id": cal["match_id"],
                    "home": label_map[cal["match_id"]][0],
                    "away": label_map[cal["match_id"]][1],
                    "p_home": cal["p_home"],
                    "p_draw": cal["p_draw"],
                    "p_away": cal["p_away"],
                }
                for cal in calibrated
            ],
            "ev_candidates": list(ev_legs),
        }
        out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info(f"[{league_id}] 结果已保存：{out_path}")
