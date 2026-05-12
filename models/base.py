from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from interfaces.contracts import MatchFeatures, ModelRawOutput


class PredictionModel(ABC):
    model_version: str

    @abstractmethod
    def fit(self, matches: list[dict[str, Any]], league_id: str) -> None:
        pass

    @abstractmethod
    def predict(self, features: MatchFeatures) -> ModelRawOutput:
        pass

    @abstractmethod
    def get_params(self) -> dict[str, Any]:
        pass

    @abstractmethod
    def load_params(self, params: dict[str, Any]) -> None:
        pass
