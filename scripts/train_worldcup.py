"""Phase 5 世界杯模型训练与回测入口。

用法示例（离线样例数据）：
    python3 scripts/train_worldcup.py --sample

使用 CSV 文件：
    python3 scripts/train_worldcup.py \\
        --data path/to/wc_matches.csv \\
        --train WC2014,WC2018 \\
        --val WC2018 \\
        --test WC2022

CSV 格式（含表头）：
    date,tournament,stage,home_team,away_team,
    home_goals,away_goals,neutral,odds_home,odds_draw,odds_away
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from worldcup.backtest import compute_wc_metrics, run_wc_backtest
from worldcup.data import load_wc_csv, sample_wc_matches


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 5 世界杯 Elo 模型回测")
    parser.add_argument("--data", default="", help="WC 比赛 CSV 路径（空则使用内置样例）")
    parser.add_argument("--train", default="", help="训练锦标赛，逗号分隔，如 WC2014,WC2018")
    parser.add_argument("--val", default="", help="校准锦标赛（可选）")
    parser.add_argument("--test", default="WC2022", help="测试锦标赛")
    parser.add_argument("--sample", action="store_true", help="使用内置 WC2022 样例数据（自测）")
    parser.add_argument("--save-params", default="", help="模型参数保存路径（JSON）")
    args = parser.parse_args()

    if args.sample or not args.data:
        matches = sample_wc_matches()
        train_tournaments = []  # 样例只有 WC2022，直接作为测试集
        test_tournament = "WC2022"
        val_tournament = ""
    else:
        matches = load_wc_csv(args.data)
        train_tournaments = [t.strip() for t in args.train.split(",") if t.strip()]
        val_tournament = args.val.strip()
        test_tournament = args.test.strip()

    result = run_wc_backtest(
        matches,
        train_tournaments=train_tournaments,
        val_tournament=val_tournament,
        test_tournament=test_tournament,
    )

    metrics = compute_wc_metrics(result)
    print(
        f"世界杯回测完成: "
        f"train={result.train_tournaments}, test={result.test_tournament}, "
        f"matches={metrics['total_matches']}, "
        f"brier={metrics['brier_score']}, "
        f"hit_rate={metrics['hit_rate']}, "
        f"roi={metrics['roi']}"
    )

    if args.save_params:
        from worldcup.model import WorldCupModel
        tmp_model = WorldCupModel()
        tmp_model.fit(
            [{"match_id": m.match_id, "match_date": m.match_date,
              "home_team_id": m.home_team_id, "away_team_id": m.away_team_id,
              "home_goals": m.home_goals, "away_goals": m.away_goals,
              "neutral": m.neutral} for m in matches],
            league_id="WC",
        )
        path = Path(args.save_params)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(tmp_model.get_params(), indent=2))
        print(f"模型参数已保存至 {path}")


if __name__ == "__main__":
    main()
