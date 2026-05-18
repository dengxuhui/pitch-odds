from __future__ import annotations

from data.processors.fatigue import fatigue_index
from data.processors.form_score import form_score
from data.processors.injury import injury_impact
from data.processors.momentum import momentum_score
from data.processors.odds_anomaly import detect_odds_anomaly


def test_form_score_range() -> None:
    score = form_score([3, 3, 1, 0, 3], [3, 7, 10, 14, 20])
    assert -1.0 <= score <= 1.0


def test_fatigue_index_range() -> None:
    value = fatigue_index(matches_last_30d=7, travel_km=2400, minutes_played_key_players=760)
    assert 0.0 <= value <= 1.0


def test_injury_impact_lower_bound() -> None:
    impact = injury_impact(
        [
            {"importance": 1.0, "position_multiplier": 1.5},
            {"importance": 1.0, "position_multiplier": 1.3},
            {"importance": 1.0, "position_multiplier": 1.2},
        ]
    )
    assert impact >= -0.30


def test_detect_odds_anomaly_high_alert() -> None:
    series = [2.2, 2.2, 2.18, 2.15, 1.95]
    result = detect_odds_anomaly(series)
    assert result["alert_level"] == "HIGH"
    assert result["exclude_from_parlay"] is True


def test_momentum_score_range() -> None:
    assert -0.10 <= momentum_score(0, 0, False) <= 0.10
    assert momentum_score(4, 0, False) == 0.10   # 连胜4场，上限
    assert momentum_score(0, 4, False) == -0.08  # 连败4场，下限（不含大败）
    assert momentum_score(0, 0, True) == -0.05   # 仅大败标记


def test_momentum_score_combined_cap() -> None:
    # 连败 + 大败不应低于 -0.10
    score = momentum_score(0, 10, True)
    assert score == -0.10


def test_enrich_rows_injects_form_features() -> None:
    from backtest.engine import enrich_rows_with_team_features

    rows = [
        {
            "match_id": i + 1,
            "league_id": "E0",
            "season": "2023-24",
            "match_date": f"2023-{8 + i // 10:02d}-{(i % 28) + 1:02d}",
            "home_team_id": 1,
            "away_team_id": 2,
            "home_goals": 1,
            "away_goals": 0,
            "result": "H",
            "odds_home": 2.1,
            "odds_draw": 3.3,
            "odds_away": 3.7,
        }
        for i in range(10)
    ]
    enrich_rows_with_team_features(rows)
    # 第一场没有历史，form 应为 0
    assert rows[0]["home_form_5"] == 0.0
    # 之后的场次应有非零 form（主队全胜）
    assert rows[-1]["home_form_5"] > 0.0
    # 所有行都注入了 key
    for row in rows:
        for key in ("home_form_5", "home_form_10", "home_momentum", "home_fatigue",
                    "away_form_5", "away_form_10", "away_momentum", "away_fatigue"):
            assert key in row
