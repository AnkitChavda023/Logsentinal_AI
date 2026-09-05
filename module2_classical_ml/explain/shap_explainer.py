from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from module2_classical_ml.config.defaults import SHAP_KERNEL_BACKGROUND_SAMPLES

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

logger = logging.getLogger(__name__)


class ShapExplainer:

    def __init__(
        self,
        model,
        feature_names: list[str],
        background_data: Optional[np.ndarray] = None,
    ) -> None:
        self._model = model
        self._feature_names = feature_names
        self._background = background_data
        self._explainer = None

        if SHAP_AVAILABLE and model is not None:
            self._init_explainer()

    def _init_explainer(self) -> None:
        if not SHAP_AVAILABLE:
            return
        try:
            self._explainer = shap.TreeExplainer(self._model)
        except Exception:
            if self._background is None:
                logger.warning(
                    "SHAP TreeExplainer failed for this model and no "
                    "background_data was supplied for the KernelExplainer "
                    "fallback — explanations will use the simple magnitude-based fallback."
                )
                return
            self._explainer = shap.KernelExplainer(
                self._model.predict,
                shap.sample(self._background, SHAP_KERNEL_BACKGROUND_SAMPLES),
            )

    def explain(self, x: np.ndarray) -> list[dict]:
        if not SHAP_AVAILABLE or self._explainer is None:
            return self._fallback_explain(x)

        x_2d = x.reshape(1, -1)
        shap_values = self._explainer.shap_values(x_2d)

        if isinstance(shap_values, list):
            sv = shap_values[0][0]
        else:
            sv = shap_values[0]

        results = [
            {"feature": name, "value": float(x[i]), "impact": float(sv[i])}
            for i, name in enumerate(self._feature_names)
        ]
        return sorted(results, key=lambda r: abs(r["impact"]), reverse=True)

    def _fallback_explain(self, x: np.ndarray) -> list[dict]:
        return [
            {"feature": name, "value": float(x[i]), "impact": float(x[i])}
            for i, name in enumerate(self._feature_names)
        ]

    def natural_language_summary(
        self, top_features: list[dict], service: str
    ) -> str:
        if not top_features:
            return f"{service}: No dominant contributing features identified."

        parts = []
        for feat in top_features[:3]:
            fname = feat["feature"].replace("_", " ")
            fval = feat["value"]
            impact = feat["impact"]
            if abs(impact) > 0.1:
                parts.append(f"{fname} = {fval:.2f} (impact {impact:+.2f})")

        return (
            f"{service}: Anomaly driven by "
            + "; ".join(parts)
            + "."
            if parts
            else f"{service}: Unusual pattern detected."
        )
