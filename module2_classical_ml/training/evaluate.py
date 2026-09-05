from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import click
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score

from module2_classical_ml.inference.scoring_engine import ScoringEngine
from module2_classical_ml.training.data_loading import load_parquet_dataset

logger = logging.getLogger(__name__)


def evaluate_scorer(scores: np.ndarray, labels: np.ndarray, threshold: float = 0.70) -> dict:
    preds = (scores >= threshold).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="binary", zero_division=0
    )
    try:
        auc = roc_auc_score(labels, scores)
    except ValueError:
        auc = 0.0

    tn = np.sum((preds == 0) & (labels == 0))
    fp = np.sum((preds == 1) & (labels == 0))
    fpr = fp / max(tn + fp, 1)

    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "roc_auc": float(auc),
        "false_positive_rate": float(fpr),
        "true_anomaly_rate": float(labels.mean()),
        "predicted_anomaly_rate": float(preds.mean()),
    }


def top_flagged_anomalies(df: pd.DataFrame, scores: np.ndarray, top_n: int = 5) -> list[dict]:
    idx = np.argsort(scores)[::-1][:top_n]
    rows = []
    for i in idx:
        row = df.iloc[i]
        rows.append({
            "timestamp": row["timestamp"],
            "service": row["service"],
            "instance_id": row["instance_id"],
            "score": float(scores[i]),
            "true_anomaly": bool(row["is_anomaly"]) if "is_anomaly" in df.columns else None,
        })
    return rows


def evaluate_dataset(
    parquet_dir: str | Path,
    model_dir: str | Path,
    service_graph_config: Optional[str | Path] = None,
    threshold: float = 0.70,
    sample_frac: Optional[float] = None,
    top_n: int = 5,
) -> dict:
    df = load_parquet_dataset(parquet_dir, sample_frac=sample_frac)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    logger.info("Evaluating on %d rows from %s", len(df), parquet_dir)

    engine = ScoringEngine(model_dir=model_dir, service_graph_config=service_graph_config)

    ensemble_scores, method_scores, labels = [], {"statistical": [], "isolation_forest": [], "autoencoder": [], "rolling_window": []}, []
    for row in df.itertuples(index=False):
        event = engine.score(
            timestamp=row.timestamp.to_pydatetime(),
            service=row.service, instance_id=row.instance_id, trace_id=row.trace_id,
            parent_span_id=(getattr(row, "parent_span_id", None) or None),
            log_level=row.log_level, latency_ms=float(row.latency_ms),
            cpu_usage=float(row.cpu_usage), memory_mb=float(row.memory_mb),
            error_code=(getattr(row, "error_code", None) or None),
            compute_explanation=False, 
        )
        ensemble_scores.append(event.anomaly_score)
        for k in method_scores:
            method_scores[k].append(event.method_scores[k])
        labels.append(bool(row.is_anomaly))

    labels_arr = np.array(labels, dtype=int)

    ensemble_scores_arr = np.array(ensemble_scores)
    results = {"ensemble": evaluate_scorer(ensemble_scores_arr, labels_arr, threshold)}
    results["ensemble"]["top_anomalies"] = top_flagged_anomalies(df, ensemble_scores_arr, top_n)

    for method, scores in method_scores.items():
        scores_arr = np.array(scores)
        results[method] = evaluate_scorer(scores_arr, labels_arr, threshold)
        results[method]["top_anomalies"] = top_flagged_anomalies(df, scores_arr, top_n)

    return results


@click.command()
@click.option("--data", "parquet_dir", required=True, type=click.Path(exists=True), help="Directory of Parquet files to evaluate (should be a held-out test split).")
@click.option("--model-dir", default="models/module2", show_default=True, help="Directory containing trained isolation_forest.pkl / autoencoder.pt.")
@click.option("--config", "service_graph_config", default=None, type=click.Path(exists=True), help="Service-graph YAML for topology features.")
@click.option("--threshold", default=0.70, show_default=True, type=float, help="ANOMALY decision threshold (§6.3).")
@click.option("--sample-frac", default=None, type=float, help="Downsample each Parquet part to this fraction of its rows before loading, to bound peak memory on very large datasets (e.g. 0.2 for 20%).")
@click.option("--top-n", default=5, show_default=True, type=int, help="How many of each method's highest-scoring rows to print for fast identification.")
def main(parquet_dir, model_dir, service_graph_config, threshold, sample_frac, top_n):
    """Evaluate Module 2's ensemble (and each individual method) against ground truth."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    results = evaluate_dataset(parquet_dir, model_dir, service_graph_config, threshold, sample_frac, top_n)
    for name, metrics in results.items():
        click.echo(f"\n{name}:")
        top_anomalies = metrics.pop("top_anomalies", [])
        for k, v in metrics.items():
            click.echo(f"  {k:24} {v:.4f}")
        if top_anomalies:
            click.echo(f"  Top {len(top_anomalies)} flagged (possible anomalies):")
            for a in top_anomalies:
                label = "ANOMALY" if a["true_anomaly"] else ("normal" if a["true_anomaly"] is not None else "?")
                click.echo(f"    score={a['score']:.3f}  {a['service']:<20} {a['instance_id']:<15} {a['timestamp']}  [{label}]")


if __name__ == "__main__":
    main()
