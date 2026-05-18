from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from backtest.engine import BacktestResult, serialize_backtest_result


def build_report_payload(result: BacktestResult, metrics: dict[str, Any]) -> dict[str, Any]:
    payload = serialize_backtest_result(result)
    payload["metrics"] = metrics
    payload["generated_at"] = datetime.now(timezone.utc).isoformat()
    payload["data_boundary"] = _build_data_boundary(result)
    return payload


def _build_data_boundary(result: BacktestResult) -> dict[str, Any]:
    from collections import defaultdict
    by_season: dict[str, list[str]] = defaultdict(list)
    for p in result.predictions:
        by_season[p.season].append(p.match_date)

    def _season_stats(season: str) -> dict[str, Any]:
        dates = by_season.get(season, [])
        if not dates:
            return {"matches": 0, "date_start": None, "date_end": None}
        return {
            "matches": len(dates),
            "date_start": min(dates),
            "date_end": max(dates),
        }

    return {
        "train_seasons": {s: {"matches": None} for s in result.train_seasons},
        "val_season": {result.val_season: {"matches": None}},
        "test_season": {result.test_season: _season_stats(result.test_season)},
        "note": "train/val match counts not tracked in BacktestResult; only test predictions are stored",
    }


def write_report(result: BacktestResult, metrics: dict[str, Any], output_dir: str | Path = "reports") -> Path:
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_path = directory / f"backtest_{result.league_id}_{timestamp}.json"
    payload = build_report_payload(result, metrics)
    output_path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")
    return output_path
