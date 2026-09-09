from __future__ import annotations

import pickle
from pathlib import Path
from typing import Optional

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from module2_classical_ml.config.defaults import (
    POLYNOMIAL_REGRESSION_ALPHA,
    POLYNOMIAL_REGRESSION_DEGREE,
    POLYNOMIAL_REGRESSION_INCLUDE_BIAS,
    POLYNOMIAL_REGRESSION_INTERACTION_ONLY,
)


class PolynomialRegressionAnomalyDetector:
    """
    Supervised High-Degree Polynomial Linear Regression detector for log telemetry.

    Fits polynomial combinations of numerical telemetry features to predict anomaly
    targets / labels (is_anomaly) using L2-regularized Ridge regression.
    Predictions are bounded and output as calibrated anomaly scores in [0.0, 1.0].
    """

    def __init__(
        self,
        degree: int = POLYNOMIAL_REGRESSION_DEGREE,
        alpha: float = POLYNOMIAL_REGRESSION_ALPHA,
        interaction_only: bool = POLYNOMIAL_REGRESSION_INTERACTION_ONLY,
        include_bias: bool = POLYNOMIAL_REGRESSION_INCLUDE_BIAS,
    ) -> None:
        self.degree = degree
        self.alpha = alpha
        self.interaction_only = interaction_only
        self.include_bias = include_bias

        self._scaler_in = StandardScaler()
        self._poly = PolynomialFeatures(
            degree=degree,
            interaction_only=interaction_only,
            include_bias=include_bias,
        )
        self._scaler_poly = StandardScaler(with_mean=False)
        self._regressor = Ridge(alpha=alpha, random_state=42)

        self._is_trained = False

    @property
    def is_trained(self) -> bool:
        return self._is_trained

    def _transform_features(self, X: np.ndarray, fit: bool = False) -> np.ndarray:
        if fit:
            X_scaled = self._scaler_in.fit_transform(X)
            X_poly = self._poly.fit_transform(X_scaled)
            X_poly_scaled = self._scaler_poly.fit_transform(X_poly)
        else:
            X_scaled = self._scaler_in.transform(X)
            X_poly = self._poly.transform(X_scaled)
            X_poly_scaled = self._scaler_poly.transform(X_poly)
        return X_poly_scaled

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        max_train_samples: Optional[int] = 100_000,
    ) -> None:
        """Fit polynomial linear regression model against anomaly target labels."""
        X = np.asarray(X, dtype=np.float32)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        y_arr = np.asarray(y, dtype=np.float32)

        if max_train_samples is not None and len(X) > max_train_samples:
            rng = np.random.default_rng(42)
            indices = rng.choice(len(X), size=max_train_samples, replace=False)
            X = X[indices]
            y_arr = y_arr[indices]

        X_poly_scaled = self._transform_features(X, fit=True)
        self._regressor.fit(X_poly_scaled, y_arr)
        self._is_trained = True

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self._is_trained:
            raise RuntimeError("Model must be trained before calling predict().")
        X = np.asarray(X, dtype=np.float32)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        X_poly_scaled = self._transform_features(X, fit=False)
        return self._regressor.predict(X_poly_scaled)

    def score(self, x: np.ndarray) -> float:
        """Compute anomaly score in [0.0, 1.0] for a single sample."""
        if not self._is_trained:
            return 0.0
        x = np.asarray(x, dtype=np.float32).reshape(1, -1)
        return float(self.score_batch(x)[0])

    def _score_chunk(self, X_chunk: np.ndarray) -> np.ndarray:
        preds = self.predict(X_chunk)
        return np.clip(preds, 0.0, 1.0).astype(np.float32)

    def score_batch(self, X: np.ndarray, batch_size: int = 50_000) -> np.ndarray:
        """Compute anomaly score in [0.0, 1.0] for a batch of samples."""
        if not self._is_trained:
            return np.zeros(len(X), dtype=np.float32)
        X = np.asarray(X, dtype=np.float32)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        if len(X) <= batch_size:
            return self._score_chunk(X)

        scores = np.empty(len(X), dtype=np.float32)
        for i in range(0, len(X), batch_size):
            end = min(i + batch_size, len(X))
            scores[i:end] = self._score_chunk(X[i:end])
        return scores

    def save(self, path: Path | str) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(
                {
                    "degree": self.degree,
                    "alpha": self.alpha,
                    "interaction_only": self.interaction_only,
                    "include_bias": self.include_bias,
                    "scaler_in": self._scaler_in,
                    "poly": self._poly,
                    "scaler_poly": self._scaler_poly,
                    "regressor": self._regressor,
                    "is_trained": self._is_trained,
                },
                f,
            )

    def load(self, path: Path | str) -> None:
        path = Path(path)
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.degree = data["degree"]
        self.alpha = data["alpha"]
        self.interaction_only = data["interaction_only"]
        self.include_bias = data["include_bias"]
        self._scaler_in = data["scaler_in"]
        self._poly = data["poly"]
        self._scaler_poly = data["scaler_poly"]
        self._regressor = data["regressor"]
        self._is_trained = data["is_trained"]
