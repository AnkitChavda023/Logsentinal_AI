from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class MethodScores:
    statistical: float = 0.0
    isolation_forest: float = 0.0
    autoencoder: float = 0.0
    rolling_window: float = 0.0


@dataclass
class EnsembleResult:
    anomaly_score: float
    classification: str  # 'ANOMALY' | 'WARNING' | 'NORMAL'
    method_scores: MethodScores


# Score thresholds
ANOMALY_THRESHOLD = 0.70
WARNING_THRESHOLD = 0.40

# Weights per method
WEIGHTS = {
    "statistical": 0.20,
    "isolation_forest": 0.35,
    "autoencoder": 0.35,
    "rolling_window": 0.10,
}


class EnsembleScorer:

    def score(
        self,
        statistical: float,
        isolation_forest: float,
        autoencoder: float,
        rolling_window: float,
    ) -> EnsembleResult:

        final_score = (
            WEIGHTS["statistical"] * statistical
            + WEIGHTS["isolation_forest"] * isolation_forest
            + WEIGHTS["autoencoder"] * autoencoder
            + WEIGHTS["rolling_window"] * rolling_window
        )
        final_score = float(max(0.0, min(1.0, final_score)))

        if final_score >= ANOMALY_THRESHOLD:
            classification = "ANOMALY"
        elif final_score >= WARNING_THRESHOLD:
            classification = "WARNING"
        else:
            classification = "NORMAL"

        return EnsembleResult(
            anomaly_score=final_score,
            classification=classification,
            method_scores=MethodScores(
                statistical=statistical,
                isolation_forest=isolation_forest,
                autoencoder=autoencoder,
                rolling_window=rolling_window,
            ),
        )
