from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from module2_classical_ml.config.defaults import GMM_N_COMPONENTS, GMM_N_INIT, GMM_RANDOM_STATE, GMM_REG_COVAR


class GMMDetector:

    def __init__(
        self,
        n_components: int = GMM_N_COMPONENTS,
        random_state: int = GMM_RANDOM_STATE,
    ) -> None:
        self._model = GaussianMixture(
            n_components=n_components,
            random_state=random_state,
            covariance_type="diag",
            reg_covar=GMM_REG_COVAR,
            n_init=GMM_N_INIT,
            
        )
        self._scaler = StandardScaler()
        self._is_trained = False
        self._calib_low: float = -50.0
        self._calib_high: float = 0.0

    def fit(self, X_normal: np.ndarray) -> None:
       
        X_scaled = self._scaler.fit_transform(X_normal)
        self._model.fit(X_scaled)
        self._is_trained = True

        raw_scores = self._model.score_samples(X_scaled)
        self._calib_low = float(np.percentile(raw_scores, 1.0))
        self._calib_high = float(np.percentile(raw_scores, 99.0))
        if self._calib_high <= self._calib_low:
            self._calib_high = self._calib_low + 1e-6

    def _normalize(self, raw: np.ndarray) -> np.ndarray:
        span = self._calib_high - self._calib_low
        normalised = (self._calib_high - raw) / span
        return np.clip(normalised, 0.0, 1.0)

    def score(self, x: np.ndarray) -> float:
        if not self._is_trained:
            return 0.0
        x_scaled = self._scaler.transform(x.reshape(1, -1))
        raw = self._model.score_samples(x_scaled)
        return float(self._normalize(raw)[0])

    def score_batch(self, X: np.ndarray) -> np.ndarray:
        if not self._is_trained:
            return np.zeros(len(X))
        X_scaled = self._scaler.transform(X)
        raw = self._model.score_samples(X_scaled)
        return self._normalize(raw)

    def save(self, path: Path) -> None:
        with open(path, "wb") as f:
            pickle.dump(
                {
                    "model": self._model,
                    "scaler": self._scaler,
                    "calib_low": self._calib_low,
                    "calib_high": self._calib_high,
                },
                f,
            )

    def load(self, path: Path) -> None:
        with open(path, "rb") as f:
            payload = pickle.load(f)
        self._model = payload["model"]
        self._scaler = payload["scaler"]
        self._calib_low = payload["calib_low"]
        self._calib_high = payload["calib_high"]
        self._is_trained = True
