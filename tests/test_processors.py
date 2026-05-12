from __future__ import annotations

from data.processors.fatigue import fatigue_index
from data.processors.form_score import form_score
from data.processors.injury import injury_impact
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
