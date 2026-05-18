"""discord_notify.py — 将预测结果格式化为 Discord Embed 并通过 Webhook 推送

从环境变量 DISCORD_WEBHOOK_URL 读取 Webhook 地址。
读取 ci_runner.py 输出的 JSON，格式化为 Discord Embed 消息。

has_matches=false 时静默退出（exit 0），不推送任何消息。

用法：
    python3 -m pipeline.discord_notify --input /tmp/predict_result.json

    # 测试：输出 payload 但不发送
    python3 -m pipeline.discord_notify --input /tmp/predict_result.json --dry-run
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import httpx

logger = logging.getLogger(__name__)

# Discord Embed 颜色
_COLOR_OK = 0x2ECC71    # 绿色：有推荐
_COLOR_INFO = 0x3498DB  # 蓝色：无推荐但有比赛
_COLOR_WARN = 0xE74C3C  # 红色：错误/警告

# 联赛显示名称
_LEAGUE_NAMES = {
    "E0":  "🏴󠁧󠁢󠁥󠁮󠁧󠁿 英超 Premier League",
    "SP1": "🇪🇸 西甲 La Liga",
    "D1":  "🇩🇪 德甲 Bundesliga",
    "I1":  "🇮🇹 意甲 Serie A",
    "F1":  "🇫🇷 法甲 Ligue 1",
}


def _fmt_pct(v: float) -> str:
    return f"{v:.1%}"


def _fmt_odds(v: float) -> str:
    return f"{v:.2f}"


def _build_league_embed(league: dict, total_budget: float) -> dict | None:
    """为单个联赛构建 Discord Embed。无比赛或有错误时返回 None。"""
    league_id = league["league_id"]
    league_name = _LEAGUE_NAMES.get(league_id, league_id)

    if league.get("error"):
        return {
            "title": f"{league_name}",
            "description": f"⚠️ {league['error']}",
            "color": _COLOR_WARN,
        }

    if not league["has_matches"]:
        return None

    fields = []

    # 概率总表
    if league["predictions"]:
        prob_lines = []
        for pred in league["predictions"]:
            home = pred["home"][:10]
            away = pred["away"][:10]
            line = (
                f"`{home} vs {away}`\n"
                f"主胜 {_fmt_pct(pred['p_home'])}(@{_fmt_odds(pred['odds_home'])})  "
                f"平 {_fmt_pct(pred['p_draw'])}(@{_fmt_odds(pred['odds_draw'])})  "
                f"客胜 {_fmt_pct(pred['p_away'])}(@{_fmt_odds(pred['odds_away'])})"
            )
            prob_lines.append(line)

        fields.append({
            "name": "📊 全场次概率",
            "value": "\n\n".join(prob_lines)[:1024],
            "inline": False,
        })

    # 正期望候选
    if league["ev_candidates"]:
        ev_lines = []
        for leg in league["ev_candidates"]:
            ev_lines.append(
                f"✅ `{leg['home']} vs {leg['away']}` "
                f"**{leg['outcome_cn']}** @{_fmt_odds(leg['odds'])}  "
                f"p={_fmt_pct(leg['p_model'])}  EV={leg['ev']:.3f}"
            )
        fields.append({
            "name": "🎯 正期望候选场次",
            "value": "\n".join(ev_lines)[:1024],
            "inline": False,
        })
    else:
        fields.append({
            "name": "🎯 正期望候选",
            "value": "当前无正期望场次，建议不投注",
            "inline": False,
        })

    # 三层串场方案
    plan = league.get("parlay_plan")
    if plan and plan.get("options"):
        for option in plan["options"]:
            legs_desc = "\n".join(
                f"  · `{leg['home']} vs {leg['away']}` {leg['outcome_cn']} @{_fmt_odds(leg['odds'])}"
                for leg in option["legs"]
            )
            value = (
                f"组合赔率: **{_fmt_odds(option['total_odds'])}**  "
                f"胜率: {_fmt_pct(option['win_rate'])}  "
                f"EV: {option['expected_ev']:.3f}\n"
                f"本层预算: {option['tier_budget']:.0f}元  "
                f"Kelly: {_fmt_pct(option['kelly_fraction'])}  "
                f"**建议注金: {option['stake']:.0f}元**\n"
                f"{legs_desc}"
            )
            fields.append({
                "name": f"🎰 {option['tier_cn']}",
                "value": value[:1024],
                "inline": False,
            })
    elif plan and plan.get("error"):
        fields.append({
            "name": "🎰 串场方案",
            "value": f"场次不足：{plan['error']}",
            "inline": False,
        })

    has_ev = bool(league["ev_candidates"])
    return {
        "title": league_name,
        "color": _COLOR_OK if has_ev else _COLOR_INFO,
        "fields": fields,
    }


def build_payload(data: dict) -> dict:
    """将 ci_runner 输出的 JSON 构建为 Discord Webhook payload。"""
    plan_date = data["plan_date"]
    generated_at = data.get("generated_at", "")
    total_budget = data.get("total_budget", 1000)

    embeds = []

    # 主 Embed：汇总标题
    total_ev = sum(
        len(lg["ev_candidates"])
        for lg in data["leagues"]
        if lg.get("ev_candidates")
    )
    summary_desc = (
        f"📅 **预测日期：{plan_date}**\n"
        f"💰 总预算：{total_budget:.0f} 元\n"
        f"🎯 正期望场次：共 {total_ev} 场\n"
        f"🕐 生成时间：{generated_at[:19].replace('T', ' ')} UTC"
    )
    embeds.append({
        "title": "⚽ pitch-odds 今日推荐",
        "description": summary_desc,
        "color": _COLOR_OK if total_ev > 0 else _COLOR_INFO,
    })

    # 各联赛 Embed
    for league in data["leagues"]:
        embed = _build_league_embed(league, total_budget)
        if embed:
            embeds.append(embed)

    # Discord 最多 10 个 Embed
    embeds = embeds[:10]

    return {"embeds": embeds}


def send_to_discord(payload: dict, webhook_url: str) -> None:
    """通过 Webhook 发送 Discord 消息。"""
    resp = httpx.post(
        webhook_url,
        json=payload,
        headers={"Content-Type": "application/json"},
        timeout=30,
    )
    if resp.status_code not in (200, 204):
        raise RuntimeError(
            f"Discord Webhook 发送失败：HTTP {resp.status_code}  {resp.text[:200]}"
        )
    logger.info("Discord 消息发送成功")


def send_error_to_discord(message: str, webhook_url: str) -> None:
    """推送错误通知到 Discord。"""
    payload = {
        "embeds": [{
            "title": "⚽ pitch-odds 运行错误",
            "description": f"❌ {message}",
            "color": _COLOR_WARN,
        }]
    }
    try:
        send_to_discord(payload, webhook_url)
    except Exception as exc:
        logger.error(f"发送错误通知也失败了：{exc}")


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    parser = argparse.ArgumentParser(description="将预测结果推送到 Discord")
    parser.add_argument("--input", required=True, help="ci_runner.py 输出的 JSON 文件路径")
    parser.add_argument(
        "--dry-run", action="store_true",
        help="只打印 payload，不实际发送",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        logger.error(f"输入文件不存在：{input_path}")
        sys.exit(1)

    data = json.loads(input_path.read_text(encoding="utf-8"))

    # 无比赛时静默退出
    if not data.get("has_matches", False):
        logger.info("今日无比赛，跳过 Discord 通知")
        sys.exit(0)

    payload = build_payload(data)

    if args.dry_run:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        logger.info("dry-run 模式：已打印 payload，未实际发送")
        return

    webhook_url = os.environ.get("DISCORD_WEBHOOK_URL", "").strip()
    if not webhook_url:
        logger.error("环境变量 DISCORD_WEBHOOK_URL 未设置")
        sys.exit(1)

    send_to_discord(payload, webhook_url)


if __name__ == "__main__":
    main()
