from __future__ import annotations

from datetime import date, timedelta

from backtest.engine import run_backtest_from_rows
from backtest.metrics import compute_metrics
from models.calibration import IsotonicThreeWayCalibrator
from models.dixon_coles import DixonColesModel, _tau_correction


def _make_features(match_id: int, home_team: int, away_team: int) -> dict:
    return {
        "match_id": match_id,
        "league_id": "E0",
        "match_date": "2024-01-01",
        "match_week": 1,
        "home_team_id": home_team,
        "away_team_id": away_team,
        "home_form_5": 0.0,
        "away_form_5": 0.0,
        "home_form_10": 0.0,
        "away_form_10": 0.0,
        "home_goals_scored_avg": 1.5,
        "home_goals_conceded_avg": 1.0,
        "away_goals_scored_avg": 1.2,
        "away_goals_conceded_avg": 1.1,
        "home_fatigue": 0.0,
        "away_fatigue": 0.0,
        "home_injury_impact": 0.0,
        "away_injury_impact": 0.0,
        "home_momentum": 0.0,
        "away_momentum": 0.0,
        "days_rest_home": 7,
        "days_rest_away": 7,
        "odds_home": 2.2,
        "odds_draw": 3.2,
        "odds_away": 3.4,
        "p_implied_home": 0.43,
        "p_implied_draw": 0.30,
        "p_implied_away": 0.27,
        "odds_drift_home": 0.0,
        "smart_money_flag": False,
        "exclude_flag": False,
    }


def _synthetic_rows() -> list[dict]:
    teams = [1, 2, 3, 4]
    seasons = ["2018-19", "2019-20", "2020-21", "2021-22", "2022-23", "2023-24"]
    rows: list[dict] = []
    match_id = 1
    current_date = date(2018, 8, 1)
    for season in seasons:
        for home in teams:
            for away in teams:
                if home == away:
                    continue
                home_goals = (home + away + match_id) % 3
                away_goals = (away + match_id) % 2
                if home_goals > away_goals:
                    result = "H"
                elif home_goals < away_goals:
                    result = "A"
                else:
                    result = "D"
                rows.append(
                    {
                        "match_id": match_id,
                        "league_id": "E0",
                        "season": season,
                        "match_date": current_date.isoformat(),
                        "home_team_id": home,
                        "away_team_id": away,
                        "home_goals": home_goals,
                        "away_goals": away_goals,
                        "result": result,
                        "odds_home": 2.1,
                        "odds_draw": 3.3,
                        "odds_away": 3.7,
                    }
                )
                match_id += 1
                current_date += timedelta(days=3)
        current_date += timedelta(days=30)
    return rows


def test_dc_probability_sum() -> None:
    model = DixonColesModel()
    train_rows = _synthetic_rows()
    payload = []
    for row in train_rows:
        item = dict(row)
        item["cutoff_date"] = "2023-06-01"
        payload.append(item)
    model.fit(payload, "E0")
    raw = model.predict(_make_features(match_id=999, home_team=1, away_team=2))
    total = raw["p_home_raw"] + raw["p_draw_raw"] + raw["p_away_raw"]
    assert abs(total - 1.0) < 1e-6


def test_dc_low_score_correction() -> None:
    lambda_home = 1.2
    lambda_away = 1.0
    rho = -0.1
    corrected = _tau_correction(0, 0, lambda_home, lambda_away, rho)
    assert corrected > 1.0


def test_calibration_output_range_and_sum() -> None:
    calibrator = IsotonicThreeWayCalibrator()
    raw_outputs = [
        {
            "match_id": i,
            "model_version": "dixon_coles_v1",
            "predicted_at": "2026-05-12T00:00:00+00:00",
            "p_home_raw": 0.4,
            "p_draw_raw": 0.3,
            "p_away_raw": 0.3,
            "lambda_home": 1.2,
            "lambda_away": 1.0,
        }
        for i in range(1, 8)
    ]
    outcomes = ["H", "D", "A", "H", "H", "D", "A"]
    calibrator.fit(raw_outputs, outcomes)
    calibrated = calibrator.calibrate(raw_outputs[0], _make_features(match_id=1, home_team=1, away_team=2))
    for key in ("p_home", "p_draw", "p_away"):
        assert 0.0 <= calibrated[key] <= 1.0
    assert abs(calibrated["p_home"] + calibrated["p_draw"] + calibrated["p_away"] - 1.0) < 1e-6


def test_engine_no_leakage() -> None:
    rows = _synthetic_rows()
    result = run_backtest_from_rows(
        rows=rows,
        league_id="E0",
        train_seasons=["2018-19", "2019-20", "2020-21", "2021-22"],
        val_season="2022-23",
        test_season="2023-24",
    )
    assert result.predictions
    for pred in result.predictions:
        assert pred.train_until < pred.match_date


def test_e2e_synthetic_backtest() -> None:
    rows = _synthetic_rows()
    result = run_backtest_from_rows(
        rows=rows,
        league_id="E0",
        train_seasons=["2018-19", "2019-20", "2020-21", "2021-22"],
        val_season="2022-23",
        test_season="2023-24",
    )
    assert result.model_version == "dixon_coles_v1"
    assert len(result.predictions) > 0
    metrics = compute_metrics(result)
    assert "calibration_diagnostics" in metrics
    assert "home" in metrics["calibration_diagnostics"]
