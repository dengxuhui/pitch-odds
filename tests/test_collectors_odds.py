from __future__ import annotations

from data.collectors.odds import _calc_overround, _maybe_float, _pick_odds_columns


def test_pick_odds_columns_prefers_bet365() -> None:
    columns = {"Date", "HomeTeam", "AwayTeam", "B365H", "B365D", "B365A", "PSH"}
    selected = _pick_odds_columns(columns)
    assert selected == {"home": "B365H", "draw": "B365D", "away": "B365A"}


def test_maybe_float_filters_invalid() -> None:
    assert _maybe_float(2.15) == 2.15
    assert _maybe_float(1.0) is None
    assert _maybe_float(None) is None


def test_calc_overround() -> None:
    value = _calc_overround(2.0, 3.5, 4.0)
    assert round(value, 4) == round((1 / 2.0) + (1 / 3.5) + (1 / 4.0), 4)
