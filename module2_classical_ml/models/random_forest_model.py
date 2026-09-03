
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier

from module2_classical_ml.config.defaults import (
    RANDOM_FOREST_MAX_DEPTH,
    RANDOM_FOREST_N_ESTIMATORS,
    RANDOM_FOREST_RANDOM_STATE,
)


class RandomForestAnomalyClassifier:

    def __init__(
        self,
        n_estimators: int = RANDOM_FOREST_N_ESTIMATORS,
        max_depth: int = RANDOM_FOREST_MAX_DEPTH,
        random_state: int = RANDOM_FOREST_RANDOM_STATE,
    ) -> None:
        self._model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            class_weight="balanced",
            n_jobs=-1,
        )
        self._is_trained = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self._model.fit(X, y)
        self._is_trained = True

    def score(self, x: np.ndarray) -> float:
        if not self._is_trained:
            return 0.0
        return float(self._model.predict_proba(x.reshape(1, -1))[0, 1])

    def score_batch(self, X: np.ndarray) -> np.ndarray:
        if not self._is_trained:
            return np.zeros(len(X))
        return self._model.predict_proba(X)[:, 1]

    def save(self, path: Path) -> None:
        with open(path, "wb") as f:
            pickle.dump({"model": self._model}, f)

    def load(self, path: Path) -> None:
        with open(path, "rb") as f:
            payload = pickle.load(f)
        self._model = payload["model"]
        self._is_trained = True
