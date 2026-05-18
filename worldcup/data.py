from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any


# 球队名称 → 整数 ID 的全局注册表（从 1_000_001 起，避免与联赛 DB ID 冲突）
_TEAM_REGISTRY: dict[str, int] = {}
_NEXT_TEAM_ID: list[int] = [1_000_001]


def team_id_for(name: str) -> int:
    """按名称获取（或注册）国家队整数 ID。"""
    key = name.strip().lower()
    if key not in _TEAM_REGISTRY:
        _TEAM_REGISTRY[key] = _NEXT_TEAM_ID[0]
        _NEXT_TEAM_ID[0] += 1
    return _TEAM_REGISTRY[key]


def team_name_registry() -> dict[str, int]:
    """返回当前名称→ID 注册表的快照。"""
    return dict(_TEAM_REGISTRY)


@dataclass
class WorldCupMatch:
    match_id: int
    tournament: str      # 例如 "WC2022"
    stage: str           # "group" / "r16" / "qf" / "sf" / "3rd" / "final"
    match_date: str      # YYYY-MM-DD
    home_team_id: int
    away_team_id: int
    home_team_name: str
    away_team_name: str
    home_goals: int
    away_goals: int
    neutral: bool
    odds_home: float     # 若无赔率数据则填 0.0
    odds_draw: float
    odds_away: float


def load_wc_csv(path: str | Path) -> list[WorldCupMatch]:
    """从 CSV 文件加载世界杯比赛数据。

    CSV 格式（含表头）：
        date,tournament,stage,home_team,away_team,
        home_goals,away_goals,neutral,odds_home,odds_draw,odds_away

    - date:       YYYY-MM-DD
    - tournament: WC2014 / WC2018 / WC2022
    - stage:      group / r16 / qf / sf / 3rd / final
    - neutral:    true / false（世界杯一般为 true）
    - odds_*:     欧洲赔率，无数据时填 0

    Returns:
        按 match_date 升序排列的 WorldCupMatch 列表。
    """
    matches: list[WorldCupMatch] = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader):
            home_name = row["home_team"].strip()
            away_name = row["away_team"].strip()
            matches.append(WorldCupMatch(
                match_id=idx + 1,
                tournament=row["tournament"].strip(),
                stage=row["stage"].strip().lower(),
                match_date=row["date"].strip(),
                home_team_id=team_id_for(home_name),
                away_team_id=team_id_for(away_name),
                home_team_name=home_name,
                away_team_name=away_name,
                home_goals=int(row["home_goals"]),
                away_goals=int(row["away_goals"]),
                neutral=row.get("neutral", "true").strip().lower() in {"true", "1", "yes"},
                odds_home=float(row.get("odds_home") or 0),
                odds_draw=float(row.get("odds_draw") or 0),
                odds_away=float(row.get("odds_away") or 0),
            ))
    matches.sort(key=lambda m: m.match_date)
    return matches


# ──────────────────────────────────────────────
# 内置样例数据（2022 世界杯小组赛部分场次）
# 赔率来自公开博彩均值，仅用于单元测试
# ──────────────────────────────────────────────

WC2022_SAMPLE: list[dict[str, Any]] = [
    # A 组
    {"date": "2022-11-20", "tournament": "WC2022", "stage": "group",
     "home_team": "Qatar",      "away_team": "Ecuador",
     "home_goals": 0, "away_goals": 2, "neutral": True,
     "odds_home": 2.40, "odds_draw": 3.30, "odds_away": 2.95},
    # B 组
    {"date": "2022-11-21", "tournament": "WC2022", "stage": "group",
     "home_team": "England",    "away_team": "Iran",
     "home_goals": 6, "away_goals": 2, "neutral": True,
     "odds_home": 1.38, "odds_draw": 4.75, "odds_away": 9.50},
    {"date": "2022-11-21", "tournament": "WC2022", "stage": "group",
     "home_team": "USA",        "away_team": "Wales",
     "home_goals": 1, "away_goals": 1, "neutral": True,
     "odds_home": 2.20, "odds_draw": 3.20, "odds_away": 3.60},
    # C 组
    {"date": "2022-11-22", "tournament": "WC2022", "stage": "group",
     "home_team": "Argentina",  "away_team": "Saudi Arabia",
     "home_goals": 1, "away_goals": 2, "neutral": True,
     "odds_home": 1.22, "odds_draw": 6.50, "odds_away": 15.00},
    {"date": "2022-11-22", "tournament": "WC2022", "stage": "group",
     "home_team": "Mexico",     "away_team": "Poland",
     "home_goals": 0, "away_goals": 0, "neutral": True,
     "odds_home": 2.55, "odds_draw": 3.10, "odds_away": 2.85},
    # D 组
    {"date": "2022-11-22", "tournament": "WC2022", "stage": "group",
     "home_team": "France",     "away_team": "Australia",
     "home_goals": 4, "away_goals": 1, "neutral": True,
     "odds_home": 1.28, "odds_draw": 5.75, "odds_away": 12.00},
    {"date": "2022-11-22", "tournament": "WC2022", "stage": "group",
     "home_team": "Denmark",    "away_team": "Tunisia",
     "home_goals": 0, "away_goals": 0, "neutral": True,
     "odds_home": 1.72, "odds_draw": 3.70, "odds_away": 5.25},
    # E 组
    {"date": "2022-11-23", "tournament": "WC2022", "stage": "group",
     "home_team": "Spain",      "away_team": "Costa Rica",
     "home_goals": 7, "away_goals": 0, "neutral": True,
     "odds_home": 1.22, "odds_draw": 6.50, "odds_away": 15.00},
    {"date": "2022-11-23", "tournament": "WC2022", "stage": "group",
     "home_team": "Germany",    "away_team": "Japan",
     "home_goals": 1, "away_goals": 2, "neutral": True,
     "odds_home": 1.40, "odds_draw": 4.75, "odds_away": 8.00},
    # F 组
    {"date": "2022-11-23", "tournament": "WC2022", "stage": "group",
     "home_team": "Belgium",    "away_team": "Canada",
     "home_goals": 1, "away_goals": 0, "neutral": True,
     "odds_home": 1.50, "odds_draw": 4.20, "odds_away": 7.00},
    {"date": "2022-11-24", "tournament": "WC2022", "stage": "group",
     "home_team": "Morocco",    "away_team": "Croatia",
     "home_goals": 0, "away_goals": 0, "neutral": True,
     "odds_home": 3.80, "odds_draw": 3.30, "odds_away": 2.10},
    # G 组
    {"date": "2022-11-24", "tournament": "WC2022", "stage": "group",
     "home_team": "Brazil",     "away_team": "Serbia",
     "home_goals": 2, "away_goals": 0, "neutral": True,
     "odds_home": 1.42, "odds_draw": 4.60, "odds_away": 8.50},
    # H 组
    {"date": "2022-11-24", "tournament": "WC2022", "stage": "group",
     "home_team": "Portugal",   "away_team": "Ghana",
     "home_goals": 3, "away_goals": 2, "neutral": True,
     "odds_home": 1.52, "odds_draw": 4.10, "odds_away": 7.00},
    {"date": "2022-11-24", "tournament": "WC2022", "stage": "group",
     "home_team": "South Korea","away_team": "Uruguay",
     "home_goals": 0, "away_goals": 0, "neutral": True,
     "odds_home": 3.60, "odds_draw": 3.20, "odds_away": 2.15},
    # 决赛
    {"date": "2022-12-18", "tournament": "WC2022", "stage": "final",
     "home_team": "Argentina",  "away_team": "France",
     "home_goals": 3, "away_goals": 3, "neutral": True,
     "odds_home": 2.40, "odds_draw": 3.50, "odds_away": 2.85},
]


def sample_wc_matches(tournaments: list[str] | None = None) -> list[WorldCupMatch]:
    """返回内置样例比赛列表（可按锦标赛名称过滤）。"""
    src = WC2022_SAMPLE
    result = []
    for idx, row in enumerate(src):
        if tournaments and row["tournament"] not in tournaments:
            continue
        home = row["home_team"]
        away = row["away_team"]
        result.append(WorldCupMatch(
            match_id=idx + 1,
            tournament=row["tournament"],
            stage=row["stage"],
            match_date=row["date"],
            home_team_id=team_id_for(home),
            away_team_id=team_id_for(away),
            home_team_name=home,
            away_team_name=away,
            home_goals=row["home_goals"],
            away_goals=row["away_goals"],
            neutral=row["neutral"],
            odds_home=row["odds_home"],
            odds_draw=row["odds_draw"],
            odds_away=row["odds_away"],
        ))
    result.sort(key=lambda m: m.match_date)
    return result
