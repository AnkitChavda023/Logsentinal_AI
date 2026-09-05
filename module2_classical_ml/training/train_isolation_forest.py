
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import click
import mlflow
import numpy as np
from sklearn.metrics import roc_auc_score

from module2_classical_ml.config.defaults import (
    ISOLATION_FOREST_CONTAMINATION,
    ISOLATION_FOREST_N_ESTIMATORS,
)
from module2_classical_ml.config.topology_loader import load_adjacency
from module2_classical_ml.features.feature_pipeline import build_feature_matrix
from module2_classical_ml.models.isolation_forest_model import IsolationForestDetector
from module2_classical_ml.training.data_loading import load_parquet_dataset

logger = logging.getLogger(__name__)


def train(
    parquet_dir: str | Path,
    model_output_dir: str | Path,
    service_graph_config: Optional[str | Path] = None,
    contamination: float = ISOLATION_FOREST_CONTAMINATION,
    n_estimators: int = ISOLATION_FOREST_N_ESTIMATORS,
    sample_frac: Optional[float] = None,
) -> None:
    parquet_dir = Path(parquet_dir)
    model_output_dir = Path(model_output_dir)
    model_output_dir.mkdir(parents=True, exist_ok=True)

    mlflow.set_tracking_uri(f"sqlite:///{(model_output_dir / 'mlflow.db').resolve()}")

    df = load_parquet_dataset(parquet_dir, sample_frac=sample_frac)

    adjacency = load_adjacency(service_graph_config)

    logger.info("Building engineered feature matrix (this walks every row through FeaturePipeline)...")
    X_raw, y = build_feature_matrix(df, adjacency=adjacency)

    n = len(X_raw)
    train_end = int(n * 0.80)
    val_end = int(n * 0.90)
    X_train, X_val, X_test = X_raw[:train_end], X_raw[train_end:val_end], X_raw[val_end:]
    y_train, y_val, y_test = y[:train_end], y[train_end:val_end], y[val_end:]

    X_train_normal = X_train[y_train == 0]
    logger.info("Training on %d normal samples (of %d total training rows)", len(X_train_normal), len(X_train))

    with mlflow.start_run(run_name="isolation_forest"):
        mlflow.log_params({
            "contamination": contamination,
            "n_estimators": n_estimators,
            "train_size": len(X_train),
            "normal_only_size": len(X_train_normal),
            "service_graph_config": str(service_graph_config) if service_graph_config else None,
        })

        model = IsolationForestDetector(
            n_estimators=n_estimators,
            contamination=contamination,
        )
        model.fit(X_train_normal)

        scores_test = model.score_batch(X_test)
        preds = (scores_test >= 0.5).astype(int)
        if len(np.unique(y_test)) > 1:
            auc = roc_auc_score(y_test, scores_test)
            mlflow.log_metric("roc_auc", auc)
            logger.info("ROC-AUC on test: %.4f", auc)
        else:
            logger.warning("Test split has only one class present — skipping ROC-AUC.")

        save_path = model_output_dir / "isolation_forest.pkl"
        model.save(save_path)
        mlflow.log_artifact(str(save_path))
        logger.info("Model saved to %s", save_path)


@click.command()
@click.option("--data", "parquet_dir", required=True, type=click.Path(exists=True), help="Directory of module1-generated Parquet files.")
@click.option("--model-dir", "model_output_dir", default="models/module2", show_default=True, help="Where to save the trained model.")
@click.option("--config", "service_graph_config", default=None, type=click.Path(exists=True), help="Service-graph YAML used to generate the data (enables topology features).")
@click.option("--contamination", default=ISOLATION_FOREST_CONTAMINATION, show_default=True, type=float)
@click.option("--n-estimators", default=ISOLATION_FOREST_N_ESTIMATORS, show_default=True, type=int)
@click.option("--sample-frac", default=None, type=float, help="Downsample each Parquet part to this fraction of its rows before loading, to bound peak memory on very large datasets (e.g. 0.2 for 20%).")
def main(parquet_dir, model_output_dir, service_graph_config, contamination, n_estimators, sample_frac):
    """Train the Isolation Forest anomaly detector (Module 2, Method 2)."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    train(
        parquet_dir=parquet_dir,
        model_output_dir=model_output_dir,
        service_graph_config=service_graph_config,
        contamination=contamination,
        n_estimators=n_estimators,
        sample_frac=sample_frac,
    )


if __name__ == "__main__":
    main()
