from __future__ import annotations

from typing import Optional, Callable

import numpy as np

try:
    from lime import lime_tabular
    LIME_AVAILABLE = True
except ImportError:
    LIME_AVAILABLE = False


class LimeExplainer:

    def __init__(
        self,
        feature_names: list[str],
        training_data: Optional[np.ndarray] = None,
        predict_fn: Optional[Callable] = None,
    ) -> None:
        self._feature_names = feature_names
        self._predict_fn = predict_fn
        self._explainer = None

        if LIME_AVAILABLE and training_data is not None:
            self._explainer = lime_tabular.LimeTabularExplainer(
                training_data,
                feature_names=feature_names,
                mode="regression",
                discretize_continuous=True,
            )

    def explain(self, x: np.ndarray, num_features: int = 5) -> list[dict]:
        if not LIME_AVAILABLE or self._explainer is None or self._predict_fn is None:
            return [
                {"feature": self._feature_names[i], "value": float(x[i]), "impact": 0.0}
                for i in range(min(num_features, len(x)))
            ]

        exp = self._explainer.explain_instance(
            x,
            self._predict_fn,
            num_features=num_features,
        )
        return [
            {"feature": feat, "value": float(x[self._feature_names.index(feat)]) if feat in self._feature_names else 0.0, "impact": float(weight)}
            for feat, weight in exp.as_list()
        ]
