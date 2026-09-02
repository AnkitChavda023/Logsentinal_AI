from __future__ import annotations

import pickle
from pathlib import Path
from typing import Optional

import numpy as np
from sklearn.ensemble import IsolationForest

from module2_classical_ml.config.defaults import (
    ISOLATION_FOREST_CONTAMINATION,
    ISOLATION_FOREST_MAX_SAMPLES,
    ISOLATION_FOREST_N_ESTIMATORS,
    ISOLATION_FOREST_RANDOM_STATE,
)

_CALIBRATION_LOW_PERCENTILE = 1.0
_CALIBRATION_HIGH_PERCENTILE = 99.0


class IsolationForestDetector:
    def __init__(
        self,
        n_estimators: int = ISOLATION_FOREST_N_ESTIMATORS,
        max_samples: str | int = ISOLATION_FOREST_MAX_SAMPLES,
        contamination: float = ISOLATION_FOREST_CONTAMINATION,
        random_state: int = ISOLATION_FOREST_RANDOM_STATE,
    ) -> None:
        self._model = IsolationForest(
            n_estimators=n_estimators,
            max_samples=max_samples,
            contamination=contamination,
            random_state=random_state,
        )
        self._is_trained = False
        self._calib_low: float = -0.5
        self._calib_high: float = 0.5

    def fit(self, X: np.ndarray) -> None:
       
        self._model.fit(X)
        self._is_trained = True

        raw_scores = self._model.decision_function(X)
        self._calib_low = float(np.percentile(raw_scores, _CALIBRATION_LOW_PERCENTILE))
        self._calib_high = float(np.percentile(raw_scores, _CALIBRATION_HIGH_PERCENTILE))
        if self._calib_high <= self._calib_low:
           self._calib_high = self._calib_low + 1e-6

    def _normalize(self, raw: np.ndarray) -> np.ndarray:
        span = self._calib_high - self._calib_low
        normalised = (self._calib_high - raw) / span
        return np.clip(normalised, 0.0, 1.0)

    def score(self, x: np.ndarray) -> float:
        if not self._is_trained:
            return 0.0

        x_2d = x.reshape(1, -1)
        raw = self._model.decision_function(x_2d)
        return float(self._normalize(raw)[0])

    def score_batch(self, X: np.ndarray) -> np.ndarray:
        if not self._is_trained:
            return np.zeros(len(X))
        raw = self._model.decision_function(X)
        return self._normalize(raw)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict(X)

    def save(self, path: Path) -> None:
        with open(path, "wb") as f:
            pickle.dump(
                {
                    "model": self._model,
                    "calib_low": self._calib_low,
                    "calib_high": self._calib_high,
                },
                f,
            )

    def load(self, path: Path) -> None:
        with open(path, "rb") as f:
            payload = pickle.load(f)
        if isinstance(payload, dict):
            self._model = payload["model"]
            self._calib_low = payload["calib_low"]
            self._calib_high = payload["calib_high"]
        else:
            self._model = payload
        self._is_trained = True
