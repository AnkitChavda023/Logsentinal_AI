from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

from module2_classical_ml.config.defaults import (
    SCORER_DEFAULT_MODEL_DIR,
    TOP_CONTRIBUTING_FEATURES_COUNT,
)
from module2_classical_ml.config.topology_loader import load_adjacency
from module2_classical_ml.explain.shap_explainer import ShapExplainer
from module2_classical_ml.features.feature_pipeline import FEATURE_NAMES, FeaturePipeline
from module2_classical_ml.models.autoencoder_model import AutoencoderDetector
from module2_classical_ml.models.ensemble_scorer import EnsembleScorer
from module2_classical_ml.models.isolation_forest_model import IsolationForestDetector
from module2_classical_ml.models.rolling_window_model import RollingWindowDetector
from module2_classical_ml.models.statistical_threshold import StatisticalThresholdDetector
from shared.schemas.anomaly_event import AnomalyEvent

logger = logging.getLogger(__name__)


class ScoringEngine:

    def __init__(
        self,
        model_dir: str | Path = SCORER_DEFAULT_MODEL_DIR,
        service_graph_config: Optional[str | Path] = None,
    ) -> None:
        adjacency = load_adjacency(service_graph_config)
        self.pipeline = FeaturePipeline(adjacency=adjacency)
        self.statistical = StatisticalThresholdDetector()
        self.rolling = RollingWindowDetector()
        self.ensemble = EnsembleScorer()

        self.isolation_forest = IsolationForestDetector()
        self.autoencoder = AutoencoderDetector()

        model_dir = Path(model_dir)
        iso_path = model_dir / "isolation_forest.pkl"
        ae_path = model_dir / "autoencoder.pt"

        if iso_path.exists():
            self.isolation_forest.load(iso_path)
            logger.info("Isolation Forest loaded from %s", iso_path)
        else:
            logger.warning(
                "No trained Isolation Forest at %s — that method will "
                "contribute a neutral 0.5 until train_isolation_forest.py has been run.",
                iso_path,
            )

        if ae_path.exists():
            self.autoencoder.load(ae_path)
            logger.info("Autoencoder loaded from %s", ae_path)
        else:
            logger.warning(
                "No trained Autoencoder at %s — that method will "
                "contribute a neutral 0.5 until train_autoencoder.py has been run.",
                ae_path,
            )

        self.shap = ShapExplainer(
            model=self.isolation_forest._model if self.isolation_forest._is_trained else None,
            feature_names=FEATURE_NAMES,
        )

        self._last_seen_ts: dict[str, datetime] = {}

    def score(
        self,
        timestamp: datetime,
        service: str,
        instance_id: str,
        trace_id: str,
        parent_span_id: Optional[str],
        log_level: str,
        latency_ms: float,
        cpu_usage: float,
        memory_mb: float,
        error_code: Optional[str],
        compute_explanation: bool = True,
    ) -> AnomalyEvent:

        feat_vec = self.pipeline.extract(
            timestamp=timestamp,
            service=service,
            instance_id=instance_id,
            trace_id=trace_id,
            parent_span_id=parent_span_id,
            log_level=log_level,
            latency_ms=latency_ms,
            cpu_usage=cpu_usage,
            memory_mb=memory_mb,
            error_code=error_code,
        )
        feat_dict = dict(zip(FEATURE_NAMES, feat_vec.tolist()))

        stat_score = self.statistical.score(service, feat_dict)
        iso_score = self.isolation_forest.score(feat_vec) if self.isolation_forest._is_trained else 0.5
        ae_score = self.autoencoder.score(feat_vec) if self.autoencoder._is_trained else 0.5
        roll_score = self._score_rolling_window(service, timestamp, stat_score)

        result = self.ensemble.score(stat_score, iso_score, ae_score, roll_score)

        self.pipeline.update_topology_health(service, result.anomaly_score)

        top_features: list[dict] = []
        explanation = ""
        if compute_explanation and self.shap:
            top_features = self.shap.explain(feat_vec)[:TOP_CONTRIBUTING_FEATURES_COUNT]
            explanation = self.shap.natural_language_summary(top_features, service)

        return AnomalyEvent(
            timestamp=timestamp,
            service=service,
            instance_id=instance_id,
            anomaly_score=result.anomaly_score,
            classification=result.classification,
            method_scores={
                "statistical": result.method_scores.statistical,
                "isolation_forest": result.method_scores.isolation_forest,
                "autoencoder": result.method_scores.autoencoder,
                "rolling_window": result.method_scores.rolling_window,
            },
            top_contributing_features=top_features,
            shap_explanation=explanation,
        )

    def _score_rolling_window(self, service: str, timestamp: datetime, stat_score: float) -> float:

        prev_ts = self._last_seen_ts.get(service)
        if prev_ts is not None and (
            prev_ts.year, prev_ts.month, prev_ts.day, prev_ts.hour, prev_ts.minute
        ) != (timestamp.year, timestamp.month, timestamp.day, timestamp.hour, timestamp.minute):
            self.rolling.commit_window(service, prev_ts)
            self.rolling.clear_current(service)

        self._last_seen_ts[service] = timestamp
        self.rolling.record(service, stat_score)
        return self.rolling.score(service, timestamp)
