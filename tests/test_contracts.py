from __future__ import annotations

import pytest

from interfaces.contracts import (
    validate_bet_record,
    validate_calibrated_prediction,
    validate_match_features,
    validate_model_raw_output,
    validate_parlay_plan,
)


def test_validate_match_features_passes() -> None:
    payload = {
        "match_id": 1,
        "league_id": "E0",
        "match_date": "2025-05-10",
        "p_implied_home": 0.4,
        "p_implied_draw": 0.3,
        "p_implied_away": 0.3,
        "home_fatigue": 0.2,
        "away_fatigue": 0.1,
        "home_injury_impact": -0.1,
        "away_injury_impact": -0.05,
        "home_momentum": 0.04,
        "away_momentum": -0.01,
        "odds_home": 2.1,
        "odds_draw": 3.2,
        "odds_away": 3.5,
        "exclude_flag": False,
    }
    validate_match_features(payload)


def test_validate_model_raw_output_requires_timezone() -> None:
    payload = {
        "match_id": 1,
        "model_version": "dixon_coles_v1",
        "predicted_at": "2026-05-12T12:00:00",
        "p_home_raw": 0.4,
        "p_draw_raw": 0.3,
        "p_away_raw": 0.3,
    }
    with pytest.raises(ValueError):
        validate_model_raw_output(payload)


def test_validate_calibrated_prediction_prob_sum() -> None:
    payload = {
        "match_id": 1,
        "model_version": "dixon_coles_v1",
        "p_home": 0.5,
        "p_draw": 0.3,
        "p_away": 0.3,
        "odds_home": 2.0,
        "odds_draw": 3.0,
        "odds_away": 4.0,
        "ev_home": 1.0,
        "ev_draw": 0.9,
        "ev_away": 1.2,
        "edge_home": 0.1,
        "edge_draw": -0.02,
        "edge_away": 0.05,
        "exclude_flag": False,
    }
    with pytest.raises(ValueError):
        validate_calibrated_prediction(payload)


def test_validate_parlay_plan_passes() -> None:
    payload = {
        "plan_date": "2026-05-12",
        "options": [
            {
                "tier": "core",
                "legs": [],
                "total_odds": 6.2,
                "win_rate": 0.21,
                "expected_ev": 1.3,
            }
        ],
        "total_budget": 1000,
    }
    validate_parlay_plan(payload)


def test_validate_bet_record_with_settled_time() -> None:
    payload = {
        "plan_id": "plan-001",
        "plan_date": "2026-05-12",
        "tier": "hedge",
        "legs": [],
        "total_odds": 3.2,
        "win_rate": 0.4,
        "stake": 100,
        "kelly_pct": 0.05,
        "is_simulation": True,
        "settled_at": "2026-05-12T18:00:00+00:00",
    }
    validate_bet_record(payload)
