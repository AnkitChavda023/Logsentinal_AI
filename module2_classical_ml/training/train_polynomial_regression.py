from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import click
import mlflow
import numpy as np
from sklearn.metrics import roc_auc_score

from module2_classical_ml.config.defaults import (
    POLYNOMIAL_REGRESSION_ALPHA,
    POLYNOMIAL_REGRESSION_DEGREE,
    POLYNOMIAL_REGRESSION_INCLUDE_BIAS,
    POLYNOMIAL_REGRESSION_INTERACTION_ONLY,
)
from module2_classical_ml.config.topology_loader import load_adjacency
from module2_classical_ml.features.feature_pipeline import build_feature_matrix
from module2_classical_ml.models.polynomial_regression_model import (
    PolynomialRegressionAnomalyDetector,
)
from module2_classical_ml.training.data_loading import load_parquet_dataset

logger = logging.getLogger(__name__)


def train(
    parquet_dir: str | Path,
    model_output_dir: str | Path,
    service_graph_config: Optional[str | Path] = None,
    degree: int = POLYNOMIAL_REGRESSION_DEGREE,
    alpha: float = POLYNOMIAL_REGRESSION_ALPHA,
    interaction_only: bool = POLYNOMIAL_REGRESSION_INTERACTION_ONLY,
    include_bias: bool = POLYNOMIAL_REGRESSION_INCLUDE_BIAS,
    sample_frac: Optional[float] = None,
) -> None:
    parquet_dir = Path(parquet_dir)
    model_output_dir = Path(model_output_dir)
    model_output_dir.mkdir(parents=True, exist_ok=True)

    mlflow.set_tracking_uri(f"sqlite:///{(model_output_dir / 'mlflow.db').resolve()}")

    df = load_parquet_dataset(parquet_dir, sample_frac=sample_frac)
    adjacency = load_adjacency(service_graph_config)

    logger.info("Building engineered feature matrix...")
    X_raw, y = build_feature_matrix(df, adjacency=adjacency)

    from sklearn.model_selection import train_test_split
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X_raw, y, test_size=0.10, random_state=42, shuffle=True, stratify=y if len(np.unique(y)) > 1 else None
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=1/9, random_state=42, shuffle=True, stratify=y_train_full if len(np.unique(y_train_full)) > 1 else None
    )

    with mlflow.start_run(run_name=f"polynomial_regression_deg{degree}"):
        mlflow.log_params({
            "model_type": "polynomial_linear_regression",
            "degree": degree,
            "alpha": alpha,
            "interaction_only": interaction_only,
            "include_bias": include_bias,
            "train_size": len(X_train),
            "service_graph_config": str(service_graph_config) if service_graph_config else None,
        })

        model = PolynomialRegressionAnomalyDetector(
            degree=degree,
            alpha=alpha,
            interaction_only=interaction_only,
            include_bias=include_bias,
        )

        logger.info("Fitting supervised polynomial linear regression on %d samples", len(X_train))
        model.fit(X_train, y_train)

        scores_test = model.score_batch(X_test)
        if len(np.unique(y_test)) > 1:
            auc = roc_auc_score(y_test, scores_test)
            mlflow.log_metric("roc_auc", auc)
            logger.info("ROC-AUC on test: %.4f", auc)
        else:
            logger.warning("Test split has only one class present — skipping ROC-AUC.")

        save_path = model_output_dir / "polynomial_regression.pkl"
        model.save(save_path)
        mlflow.log_artifact(str(save_path))
        logger.info("Model saved to %s", save_path)


@click.command()
@click.option("--data", "parquet_dir", required=True, type=click.Path(exists=True), help="Directory of module1-generated Parquet files.")
@click.option("--model-dir", "model_output_dir", default="models/module2", show_default=True, help="Where to save the trained model.")
@click.option("--config", "service_graph_config", default=None, type=click.Path(exists=True), help="Service-graph YAML used to generate the data.")
@click.option("--degree", default=POLYNOMIAL_REGRESSION_DEGREE, show_default=True, type=int, help="Degree of polynomial features.")
@click.option("--alpha", default=POLYNOMIAL_REGRESSION_ALPHA, show_default=True, type=float, help="L2 regularization weight (Ridge alpha).")
@click.option("--interaction-only", is_flag=True, default=POLYNOMIAL_REGRESSION_INTERACTION_ONLY, help="If true, only produce interaction features.")
@click.option("--include-bias", is_flag=True, default=POLYNOMIAL_REGRESSION_INCLUDE_BIAS, help="Include intercept / bias column in polynomial features.")
@click.option("--sample-frac", default=None, type=float, help="Downsample fraction for Parquet dataset.")
def main(
    parquet_dir,
    model_output_dir,
    service_graph_config,
    degree,
    alpha,
    interaction_only,
    include_bias,
    sample_frac,
):
    """Train the High-Degree Polynomial Linear Regression model for Module 2."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    train(
        parquet_dir=parquet_dir,
        model_output_dir=model_output_dir,
        service_graph_config=service_graph_config,
        degree=degree,
        alpha=alpha,
        interaction_only=interaction_only,
        include_bias=include_bias,
        sample_frac=sample_frac,
    )



if __name__ == "__main__":
    main()
