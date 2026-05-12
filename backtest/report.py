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
    return payload


def write_report(result: BacktestResult, metrics: dict[str, Any], output_dir: str | Path = "reports") -> Path:
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_path = directory / f"backtest_{result.league_id}_{timestamp}.json"
    payload = build_report_payload(result, metrics)
    output_path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")
    return output_path
