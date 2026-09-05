from __future__ import annotations

import pickle
from pathlib import Path

import lightgbm as lgb
import numpy as np

from module2_classical_ml.config.defaults import (
    LIGHTGBM_LEARNING_RATE,
    LIGHTGBM_MAX_DEPTH,
    LIGHTGBM_N_ESTIMATORS,
    LIGHTGBM_RANDOM_STATE,
)


class LightGBMAnomalyClassifier:

    def __init__(
        self,
        n_estimators: int = LIGHTGBM_N_ESTIMATORS,
        learning_rate: float = LIGHTGBM_LEARNING_RATE,
        max_depth: int = LIGHTGBM_MAX_DEPTH,
        random_state: int = LIGHTGBM_RANDOM_STATE,
    ) -> None:
        self._model = lgb.LGBMClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            random_state=random_state,
            class_weight="balanced",
            n_jobs=-1,
            verbosity=-1,
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
