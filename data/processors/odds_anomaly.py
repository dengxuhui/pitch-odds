from __future__ import annotations


def detect_odds_anomaly(odds_series: list[float]) -> dict[str, object]:
    """检测赔率时序异常。"""
    if len(odds_series) < 2:
        return {
            "alert_level": "NORMAL",
            "exclude_from_parlay": False,
            "is_correction": False,
            "smart_money": False,
            "total_drift_pct": 0.0,
        }

    n = len(odds_series)
    step_changes = [abs(odds_series[i] - odds_series[i - 1]) / odds_series[i - 1] for i in range(1, n)]

    spike = max(step_changes) > 0.08
    total_drift = abs(odds_series[-1] - odds_series[0]) / odds_series[0]
    trend = total_drift > 0.15

    late_window = odds_series[max(0, n - 32) :]
    late_change = (
        abs(late_window[-1] - late_window[0]) / late_window[0]
        if len(late_window) > 1 and late_window[0] != 0
        else 0.0
    )
    late_anomaly = late_change > 0.10

    first_half = step_changes[: max(1, len(step_changes) // 2)]
    correction = max(first_half) > 0.10 and abs(odds_series[-1] - odds_series[0]) / odds_series[0] < 0.03

    alert_level = "HIGH" if (spike or late_anomaly) else "WATCH" if trend else "NORMAL"

    return {
        "alert_level": alert_level,
        "exclude_from_parlay": alert_level == "HIGH",
        "is_correction": correction,
        "smart_money": spike and not correction,
        "total_drift_pct": round(total_drift * 100.0, 2),
    }
