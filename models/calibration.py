from __future__ import annotations

from typing import Any

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

from interfaces.contracts import CalibratedPrediction, MatchFeatures, ModelRawOutput, validate_calibrated_prediction


def _safe_logit(p: np.ndarray) -> np.ndarray:
    """logit，输入约束在 (1e-6, 1-1e-6) 防止 log(0)。"""
    p_clipped = np.clip(p, 1e-6, 1.0 - 1e-6)
    return np.log(p_clipped / (1.0 - p_clipped))


class _PlattCalibrator:
    """单方向 Platt 缩放：p_cal = sigmoid(a * logit(p_raw) + b)。

    只有 2 个参数，在小样本（单赛季约 300~400 场）下不易过拟合。
    相比同位回归（Isotonic）的优势：
    - 无边界外推跳变：超出训练概率范围的值经 sigmoid 自然平滑处理
    - 参数少：不会把验证集某区间的极端偶发结果直接映射为极端概率
    """

    def __init__(self) -> None:
        self._a: float = 1.0  # 对数赔率缩放因子（逆温度）
        self._b: float = 0.0  # 偏置（系统性偏差修正）

    def fit(self, x: list[float], y: list[float]) -> None:
        if len(x) != len(y) or not x:
            raise ValueError("校准输入非法")

        x_arr = np.array(x, dtype=float)
        y_arr = np.array(y, dtype=float)
        logit_x = _safe_logit(x_arr)

        def neg_log_loss(params: np.ndarray) -> float:
            a, b = params
            p = np.clip(expit(a * logit_x + b), 1e-9, 1.0 - 1e-9)
            return -float(np.mean(y_arr * np.log(p) + (1.0 - y_arr) * np.log(1.0 - p)))

        # 对 b 加 L2 正则（λ=2.0），防止单赛季偏移过拟合：
        # b 过大会系统性抬高/压低所有预测，转移到测试集时方向不确定
        l2_lambda = 2.0

        def penalized_neg_log_loss(params: np.ndarray) -> float:
            return neg_log_loss(params) + l2_lambda * params[1] ** 2

        result = minimize(
            penalized_neg_log_loss,
            x0=np.array([1.0, 0.0]),
            method="L-BFGS-B",
            options={"maxiter": 500, "ftol": 1e-10},
        )
        self._a, self._b = float(result.x[0]), float(result.x[1])

    def predict_one(self, value: float) -> float:
        logit_v = float(_safe_logit(np.array([value]))[0])
        return float(expit(self._a * logit_v + self._b))

    def get_params(self) -> dict[str, float]:
        return {"a": self._a, "b": self._b}

    def load_params(self, params: dict[str, Any]) -> None:
        self._a = float(params["a"])
        self._b = float(params["b"])


class PlattThreeWayCalibrator:
    """三分类 Platt 缩放校准器。

    对主胜、平局、客胜三个方向各自独立拟合 Platt 缩放，
    再做归一化确保三概率之和为 1。
    """

    def __init__(self) -> None:
        self._home = _PlattCalibrator()
        self._draw = _PlattCalibrator()
        self._away = _PlattCalibrator()
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


# 向后兼容别名：engine.py / train.py / predict.py 中的旧名称无需修改
IsotonicThreeWayCalibrator = PlattThreeWayCalibrator
