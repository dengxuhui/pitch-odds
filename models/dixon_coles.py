from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from math import exp, lgamma, log
from typing import Any

import numpy as np
from scipy.optimize import Bounds, minimize

from interfaces.contracts import MatchFeatures, ModelRawOutput, validate_model_raw_output
from models.base import PredictionModel


def _tau_correction(home_goals: int, away_goals: int, lambda_home: float, lambda_away: float, rho: float) -> float:
    if home_goals == 0 and away_goals == 0:
        return max(1.0 - (lambda_home * lambda_away * rho), 1e-8)
    if home_goals == 0 and away_goals == 1:
        return max(1.0 + (lambda_home * rho), 1e-8)
    if home_goals == 1 and away_goals == 0:
        return max(1.0 + (lambda_away * rho), 1e-8)
    if home_goals == 1 and away_goals == 1:
        return max(1.0 - rho, 1e-8)
    return 1.0


@dataclass(frozen=True)
class _TrainingMatch:
    home_team: str
    away_team: str
    home_goals: int
    away_goals: int
    match_date: datetime
    cutoff_date: datetime


class DixonColesModel(PredictionModel):
    model_version = "dixon_coles_v1"

    def __init__(self, *, decay_xi: float = 0.0018, max_goals: int = 10) -> None:
        self.decay_xi = decay_xi
        self.max_goals = max_goals
        self._team_order: list[str] = []
        self._attack: dict[str, float] = {}
        self._defense: dict[str, float] = {}
        self._gamma: float = 0.0
        self._rho: float = 0.0
        self._league_id: str | None = None

    def fit(self, matches: list[dict[str, Any]], league_id: str) -> None:
        training_matches = self._prepare_training_matches(matches)
        if not training_matches:
            raise ValueError("训练数据为空")

        teams = sorted({m.home_team for m in training_matches} | {m.away_team for m in training_matches})
        if len(teams) < 2:
            raise ValueError("训练球队数量不足")
        self._team_order = teams
        n_teams = len(teams)
        team_index = {name: idx for idx, name in enumerate(teams)}

        def objective(theta: np.ndarray) -> float:
            attack = theta[:n_teams]
            defense = theta[n_teams : 2 * n_teams]
            gamma = theta[-2]
            rho = theta[-1]

            attack = attack - np.mean(attack)
            log_like = 0.0

            for match in training_matches:
                home_i = team_index[match.home_team]
                away_i = team_index[match.away_team]
                lambda_home = exp(attack[home_i] + defense[away_i] + gamma)
                lambda_away = exp(attack[away_i] + defense[home_i])
                tau = _tau_correction(match.home_goals, match.away_goals, lambda_home, lambda_away, rho)
                p_home = _poisson_pmf(match.home_goals, lambda_home)
                p_away = _poisson_pmf(match.away_goals, lambda_away)
                prob = max(tau * p_home * p_away, 1e-12)

                days_ago = (match.cutoff_date - match.match_date).days
                weight = exp(-self.decay_xi * max(days_ago, 0))
                log_like += weight * np.log(prob)

            l2_penalty = 0.001 * float(np.sum(attack**2) + np.sum(defense**2) + gamma**2 + rho**2)
            return -(log_like - l2_penalty)

        theta = np.zeros((2 * n_teams) + 2)
        lower = np.full(theta.shape, -np.inf)
        upper = np.full(theta.shape, np.inf)
        lower[-2], upper[-2] = -1.0, 1.0   # gamma 范围
        lower[-1], upper[-1] = -0.5, 0.5   # rho 范围
        result = minimize(
            objective,
            theta,
            method="L-BFGS-B",
            bounds=Bounds(lower, upper),
            options={"maxiter": 2000, "ftol": 1e-9},
        )
        theta = result.x

        attack = theta[:n_teams] - np.mean(theta[:n_teams])
        defense = theta[n_teams : 2 * n_teams]
        self._attack = {team: float(attack[idx]) for idx, team in enumerate(teams)}
        self._defense = {team: float(defense[idx]) for idx, team in enumerate(teams)}
        self._gamma = float(theta[-2])
        self._rho = float(theta[-1])
        self._league_id = league_id

    def predict(self, features: MatchFeatures) -> ModelRawOutput:
        if not self._attack or not self._defense:
            raise RuntimeError("模型尚未训练或加载参数")

        home_key = str(features["home_team_id"])
        away_key = str(features["away_team_id"])
        attack_home = self._attack.get(home_key, 0.0)
        defense_home = self._defense.get(home_key, 0.0)
        attack_away = self._attack.get(away_key, 0.0)
        defense_away = self._defense.get(away_key, 0.0)

        lambda_home = exp(attack_home + defense_away + self._gamma)
        lambda_away = exp(attack_away + defense_home)
        p_home, p_draw, p_away = self._three_way_probs(lambda_home, lambda_away, self._rho)

        output: ModelRawOutput = {
            "match_id": int(features["match_id"]),
            "model_version": self.model_version,
            "predicted_at": datetime.now(timezone.utc).isoformat(),
            "p_home_raw": p_home,
            "p_draw_raw": p_draw,
            "p_away_raw": p_away,
            "lambda_home": lambda_home,
            "lambda_away": lambda_away,
        }
        validate_model_raw_output(output)
        return output

    def get_params(self) -> dict[str, Any]:
        if not self._attack or not self._defense:
            raise RuntimeError("模型尚未训练或加载参数")
        return {
            "attack": self._attack,
            "defense": self._defense,
            "gamma": self._gamma,
            "rho": self._rho,
            "decay_xi": self.decay_xi,
            "max_goals": self.max_goals,
            "league_id": self._league_id,
        }

    def load_params(self, params: dict[str, Any]) -> None:
        self._attack = {str(k): float(v) for k, v in dict(params["attack"]).items()}
        self._defense = {str(k): float(v) for k, v in dict(params["defense"]).items()}
        self._gamma = float(params["gamma"])
        self._rho = float(params["rho"])
        self.decay_xi = float(params.get("decay_xi", self.decay_xi))
        self.max_goals = int(params.get("max_goals", self.max_goals))
        self._league_id = str(params.get("league_id")) if params.get("league_id") is not None else None
        self._team_order = sorted(set(self._attack.keys()) | set(self._defense.keys()))

    def _three_way_probs(self, lambda_home: float, lambda_away: float, rho: float) -> tuple[float, float, float]:
        home = 0.0
        draw = 0.0
        away = 0.0
        for home_goals in range(self.max_goals + 1):
            p_h = _poisson_pmf(home_goals, lambda_home)
            for away_goals in range(self.max_goals + 1):
                p_a = _poisson_pmf(away_goals, lambda_away)
                tau = _tau_correction(home_goals, away_goals, lambda_home, lambda_away, rho)
                p = max(tau * p_h * p_a, 0.0)
                if home_goals > away_goals:
                    home += p
                elif home_goals < away_goals:
                    away += p
                else:
                    draw += p
        total = home + draw + away
        if total <= 0:
            raise RuntimeError("概率矩阵异常，总和为0")
        return home / total, draw / total, away / total

    def _prepare_training_matches(self, matches: list[dict[str, Any]]) -> list[_TrainingMatch]:
        prepared: list[_TrainingMatch] = []
        for row in matches:
            home_goals = row.get("home_goals")
            away_goals = row.get("away_goals")
            if home_goals is None or away_goals is None:
                continue
            match_date = self._to_datetime(row["match_date"])
            cutoff_date = self._to_datetime(row.get("cutoff_date") or row["match_date"])
            prepared.append(
                _TrainingMatch(
                    home_team=str(row["home_team_id"]),
                    away_team=str(row["away_team_id"]),
                    home_goals=int(home_goals),
                    away_goals=int(away_goals),
                    match_date=match_date,
                    cutoff_date=cutoff_date,
                )
            )
        return prepared

    @staticmethod
    def _to_datetime(value: Any) -> datetime:
        if isinstance(value, datetime):
            return value.replace(tzinfo=None)
        return datetime.fromisoformat(str(value))


def _poisson_pmf(k: int, lam: float) -> float:
    if lam <= 0:
        return 1.0 if k == 0 else 0.0
    return exp(-lam + (k * log(lam)) - lgamma(k + 1))
