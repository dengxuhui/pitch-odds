from __future__ import annotations

from typing import Any

from interfaces.contracts import CalibratedPrediction, MatchFeatures, ModelRawOutput, validate_calibrated_prediction


class _StepIsotonic:
    def __init__(self) -> None:
        self._x: list[float] = []
        self._y: list[float] = []

    def fit(self, x: list[float], y: list[float]) -> None:
        if len(x) != len(y) or not x:
            raise ValueError("同位回归输入非法")
        pairs = sorted((float(px), float(py)) for px, py in zip(x, y, strict=True))
        x_vals = [p[0] for p in pairs]
        y_vals = [p[1] for p in pairs]
        n = len(y_vals)

        blocks = []
        for idx in range(n):
            blocks.append({"start": idx, "end": idx, "sum": y_vals[idx], "count": 1})
            while len(blocks) >= 2:
                left = blocks[-2]
                right = blocks[-1]
                left_mean = left["sum"] / left["count"]
                right_mean = right["sum"] / right["count"]
                if left_mean <= right_mean:
                    break
                merged = {
                    "start": left["start"],
                    "end": right["end"],
                    "sum": left["sum"] + right["sum"],
                    "count": left["count"] + right["count"],
                }
                blocks.pop()
                blocks.pop()
                blocks.append(merged)

        fitted_y = [0.0] * n
        for block in blocks:
            value = block["sum"] / block["count"]
            for idx in range(block["start"], block["end"] + 1):
                fitted_y[idx] = value

        self._x = x_vals
        self._y = fitted_y

    def predict_one(self, value: float) -> float:
        if not self._x:
            raise RuntimeError("回归器未训练")
        x = float(value)
        if x <= self._x[0]:
            return self._y[0]
        if x >= self._x[-1]:
            return self._y[-1]

        lo = 0
        hi = len(self._x) - 1
        while lo <= hi:
            mid = (lo + hi) // 2
            if self._x[mid] <= x < self._x[mid + 1]:
                return self._y[mid]
            if x < self._x[mid]:
                hi = mid - 1
            else:
                lo = mid + 1
        return self._y[-1]

    def get_params(self) -> dict[str, list[float]]:
        return {"x": self._x, "y": self._y}

    def load_params(self, params: dict[str, Any]) -> None:
        self._x = [float(v) for v in params["x"]]
        self._y = [float(v) for v in params["y"]]


class IsotonicThreeWayCalibrator:
    def __init__(self) -> None:
        self._home = _StepIsotonic()
        self._draw = _StepIsotonic()
        self._away = _StepIsotonic()
        self._fitted = False

    def fit(self, raw_outputs: list[ModelRawOutput], outcomes: list[str]) -> None:
        if len(raw_outputs) != len(outcomes):
            raise ValueError("raw_outputs 与 outcomes 长度不一致")
        if not raw_outputs:
            raise ValueError("校准训练数据为空")

        p_home = [float(x["p_home_raw"]) for x in raw_outputs]
        p_draw = [float(x["p_draw_raw"]) for x in raw_outputs]
        p_away = [float(x["p_away_raw"]) for x in raw_outputs]
        home_y = [1.0 if str(o).upper() == "H" else 0.0 for o in outcomes]
        draw_y = [1.0 if str(o).upper() == "D" else 0.0 for o in outcomes]
        away_y = [1.0 if str(o).upper() == "A" else 0.0 for o in outcomes]

        self._home.fit(p_home, home_y)
        self._draw.fit(p_draw, draw_y)
        self._away.fit(p_away, away_y)
        self._fitted = True

    def calibrate(self, raw: ModelRawOutput, features: MatchFeatures) -> CalibratedPrediction:
        if not self._fitted:
            raise RuntimeError("校准器尚未训练")

        p_home = self._home.predict_one(float(raw["p_home_raw"]))
        p_draw = self._draw.predict_one(float(raw["p_draw_raw"]))
        p_away = self._away.predict_one(float(raw["p_away_raw"]))
        p_home, p_draw, p_away = self._normalize_triplet(p_home, p_draw, p_away)

        odds_home = float(features["odds_home"])
        odds_draw = float(features["odds_draw"])
        odds_away = float(features["odds_away"])
        p_implied_home = float(features["p_implied_home"])
        p_implied_draw = float(features["p_implied_draw"])
        p_implied_away = float(features["p_implied_away"])

        output: CalibratedPrediction = {
            "match_id": int(raw["match_id"]),
            "model_version": str(raw["model_version"]),
            "p_home": p_home,
            "p_draw": p_draw,
            "p_away": p_away,
            "odds_home": odds_home,
            "odds_draw": odds_draw,
            "odds_away": odds_away,
            "ev_home": p_home * odds_home,
            "ev_draw": p_draw * odds_draw,
            "ev_away": p_away * odds_away,
            "edge_home": p_home - p_implied_home,
            "edge_draw": p_draw - p_implied_draw,
            "edge_away": p_away - p_implied_away,
            "smart_money_flag": bool(features.get("smart_money_flag", False)),
            "exclude_flag": bool(features["exclude_flag"]),
        }
        validate_calibrated_prediction(output)
        return output

    def get_params(self) -> dict[str, Any]:
        if not self._fitted:
            raise RuntimeError("校准器尚未训练")
        return {
            "home": self._home.get_params(),
            "draw": self._draw.get_params(),
            "away": self._away.get_params(),
        }

    def load_params(self, params: dict[str, Any]) -> None:
        self._home.load_params(params["home"])
        self._draw.load_params(params["draw"])
        self._away.load_params(params["away"])
        self._fitted = True

    @staticmethod
    def _normalize_triplet(home: float, draw: float, away: float) -> tuple[float, float, float]:
        home = min(max(home, 0.0), 1.0)
        draw = min(max(draw, 0.0), 1.0)
        away = min(max(away, 0.0), 1.0)
        total = home + draw + away
        if total <= 0:
            return 1 / 3, 1 / 3, 1 / 3
        return home / total, draw / total, away / total
