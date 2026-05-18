"""ci_runner.py — CI 单次执行入口（无轮询，适合 GitHub Actions）

执行顺序：
    1. 拉取赛程（football-data.co.uk）
    2. 拉取最新赔率快照（The Odds API，单次）
    3. 对每个联赛运行预测 + 串场优化
    4. 聚合结果为 JSON，写入 --output-json 路径

无比赛时输出 {"has_matches": false, "leagues": []} 并以 exit(0) 退出。

用法：
    python3 -m pipeline.ci_runner \\
        --leagues E0 SP1 D1 I1 F1 \\
        --budget 1000 \\
        --safety-margin 1.05 \\
        --output-json /tmp/predict_result.json
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import date, datetime, timezone
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from capital.kelly import half_kelly
from data.collectors.fixtures_fetcher import fetch_and_import
from data.collectors.odds_api_client import pull_snapshots
from data.storage.db import SessionLocal
from data.storage.models import Match, ModelParams, OddsOpening, Team
from models.calibration import IsotonicThreeWayCalibrator
from models.dixon_coles import DixonColesModel
from optimizer.ev_filter import filter_positive_ev
from optimizer.parlay_optimizer import build_parlay_plan
from sqlalchemy import desc, select

logger = logging.getLogger(__name__)

_OUTCOME_CN = {"home": "主胜", "draw": "平局", "away": "客胜"}
_TIER_CN = {"hedge": "保底层（2~3串）", "core": "核心层（4~5串）", "aggressive": "冲击层（6~7串）"}
_TIER_BUDGETS = {"hedge": 0.40, "core": 0.40, "aggressive": 0.20}


def _predict_league(
    league_id: str,
    budget: float,
    safety_margin: float,
    plan_date: str,
) -> dict:
    """对单个联赛运行预测，返回结构化结果 dict。"""
    result: dict = {
        "league_id": league_id,
        "plan_date": plan_date,
        "has_matches": False,
        "predictions": [],
        "ev_candidates": [],
        "parlay_plan": None,
        "error": None,
    }

    # 加载模型参数
    with SessionLocal() as session:
        row = session.execute(
            select(ModelParams)
            .where(ModelParams.league_id == league_id)
            .order_by(desc(ModelParams.trained_at))
            .limit(1)
        ).scalar_one_or_none()

    if row is None:
        result["error"] = f"数据库中无模型参数，请先运行 train.yml 训练 {league_id}"
        logger.warning(result["error"])
        return result

    params = row.params
    model = DixonColesModel()
    model.load_params(params["model"])
    calibrator = IsotonicThreeWayCalibrator()
    calibrator.load_params(params["calibrator"])

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
        return result

    result["has_matches"] = True

    # 预测 + 校准
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

    result["predictions"] = [
        {
            "match_id": cal["match_id"],
            "home": label_map[cal["match_id"]][0],
            "away": label_map[cal["match_id"]][1],
            "p_home": round(cal["p_home"], 4),
            "p_draw": round(cal["p_draw"], 4),
            "p_away": round(cal["p_away"], 4),
            "odds_home": next(
                float(o.odds_home) for m, o, ht, _ in rows if m.id == cal["match_id"]
            ),
            "odds_draw": next(
                float(o.odds_draw) for m, o, ht, _ in rows if m.id == cal["match_id"]
            ),
            "odds_away": next(
                float(o.odds_away) for m, o, ht, _ in rows if m.id == cal["match_id"]
            ),
        }
        for cal in calibrated
    ]

    ev_legs = filter_positive_ev(calibrated, safety_margin=safety_margin)
    result["ev_candidates"] = [
        {
            **leg,
            "home": label_map[leg["match_id"]][0],
            "away": label_map[leg["match_id"]][1],
            "outcome_cn": _OUTCOME_CN.get(leg["outcome"], leg["outcome"]),
        }
        for leg in ev_legs
    ]

    if ev_legs:
        try:
            plan = build_parlay_plan(ev_legs, plan_date=plan_date, total_budget=budget)
            options_out = []
            for option in plan["options"]:
                tier = option["tier"]
                tier_budget = budget * _TIER_BUDGETS.get(tier, 0.0)
                kelly_f = half_kelly(option["win_rate"], option["total_odds"])
                stake = round(tier_budget * kelly_f, 2)
                options_out.append({
                    "tier": tier,
                    "tier_cn": _TIER_CN.get(tier, tier),
                    "total_odds": option["total_odds"],
                    "win_rate": option["win_rate"],
                    "expected_ev": option["expected_ev"],
                    "tier_budget": tier_budget,
                    "kelly_fraction": round(kelly_f, 4),
                    "stake": stake,
                    "legs": [
                        {
                            **leg,
                            "home": label_map[leg["match_id"]][0],
                            "away": label_map[leg["match_id"]][1],
                            "outcome_cn": _OUTCOME_CN.get(leg["outcome"], leg["outcome"]),
                        }
                        for leg in option["legs"]
                    ],
                })
            result["parlay_plan"] = {
                "plan_date": plan_date,
                "total_budget": budget,
                "options": options_out,
            }
        except ValueError as exc:
            result["parlay_plan"] = {"error": str(exc)}

    return result


def run(
    league_ids: list[str],
    budget: float,
    safety_margin: float,
    plan_date: str,
    output_json: str,
) -> None:
    logger.info(f"CI Runner 启动  联赛={' '.join(league_ids)}  日期={plan_date}")

    # 步骤 1：拉取赛程
    logger.info("[1/3] 拉取赛程...")
    try:
        fixture_results = fetch_and_import(league_ids)
        total_new = sum(r.new_matches for r in fixture_results)
        logger.info(f"      新增赛程 {total_new} 场")
    except Exception as exc:
        logger.warning(f"      拉取赛程失败（将尝试使用已有数据）：{exc}")

    # 步骤 2：拉取赔率（单次，无轮询）
    logger.info("[2/3] 拉取最新赔率快照（单次）...")
    try:
        counts = pull_snapshots(league_ids)
        total_snaps = sum(counts.values())
        logger.info(f"      写入快照：{total_snaps} 条")
    except RuntimeError as exc:
        logger.error(f"      赔率拉取失败（不可恢复）：{exc}")
    except Exception as exc:
        logger.warning(f"      赔率拉取出错（将使用已有数据）：{exc}")

    # 步骤 3：生成预测
    logger.info("[3/3] 生成预测推荐...")
    league_results = []
    for league_id in league_ids:
        res = _predict_league(league_id, budget, safety_margin, plan_date)
        league_results.append(res)

    has_any_matches = any(r["has_matches"] for r in league_results)

    output = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "plan_date": plan_date,
        "total_budget": budget,
        "safety_margin": safety_margin,
        "has_matches": has_any_matches,
        "leagues": league_results,
    }

    out_path = Path(output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info(f"结果已写入：{out_path}")

    if not has_any_matches:
        logger.info("今日无比赛，不推送通知")


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    parser = argparse.ArgumentParser(description="pitch-odds CI 单次执行入口")
    parser.add_argument(
        "--leagues", nargs="+", default=["E0", "SP1", "D1", "I1", "F1"],
    )
    parser.add_argument("--budget", type=float, default=1000.0)
    parser.add_argument("--safety-margin", type=float, default=1.05, dest="safety_margin")
    parser.add_argument(
        "--date", default=date.today().isoformat(), dest="plan_date",
        help="预测日期 YYYY-MM-DD，默认今日",
    )
    parser.add_argument(
        "--output-json", default="/tmp/predict_result.json", dest="output_json",
    )
    args = parser.parse_args()

    run(
        league_ids=args.leagues,
        budget=args.budget,
        safety_margin=args.safety_margin,
        plan_date=args.plan_date,
        output_json=args.output_json,
    )


if __name__ == "__main__":
    main()
