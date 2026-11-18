"""
Offline Evaluation for LogSentinel

Evaluates a trained model on a held-out test set and reports metrics
against the Section 10.1 targets.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import yaml

from module3_deep_learning.config import defaults as cfg
from module3_deep_learning.models.full_model import LogSentinelModel, model_config_from_yaml
from .checkpoint import load_checkpoint
from .dataloader import LogBatchBuilder, create_dataloader
from .dataset import LogSequenceDataset
from .metrics import MetricsCalculator, MetricsResult
from .splitter import ChronologicalSplitter
from .train import build_edge_index

logger = logging.getLogger(__name__)

TARGETS = {
    "precision": cfg.TARGET_PRECISION,
    "recall": cfg.TARGET_RECALL,
    "false_positive_rate": cfg.TARGET_FALSE_POSITIVE_RATE,
    "failure_type_accuracy": cfg.TARGET_FAILURE_TYPE_ACCURACY,
    "time_to_failure_mae": cfg.TARGET_TIME_TO_FAILURE_MAE_S,
}


def evaluate(
    checkpoint_path: Path,
    config_path: Path,
    data_path: Path,
    manifest_path: Path,
    output_path: Optional[Path] = None,
    use_uncertainty: bool = False,
) -> Dict[str, Any]:
    """Evaluate a trained model on the chronological test split."""
    with open(config_path) as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    service_list = config["data"]["service_list"]
    error_code_list = config["data"]["error_code_list"]

    # Must match train.py's ModelConfig construction exactly (same
    # model_config_from_yaml call) — this rebuilds the architecture
    # load_checkpoint() then loads a state_dict into. Any mismatch (e.g.
    # a model.* field train.py read but this didn't) would silently
    # rebuild a different architecture here and fail to load, or worse,
    # load into the wrong shape.
    model_config = model_config_from_yaml(
        model_cfg_yaml=config.get("model", {}),
        num_services=len(service_list),
        num_error_codes=len(error_code_list),
        max_seq_length=config["data"].get("max_seq_length", cfg.MAX_SEQ_LENGTH),
    )
    model = LogSentinelModel(model_config).to(device)

    model, _, epoch, _ = load_checkpoint(checkpoint_path, model, device=device)
    logger.info("Loaded checkpoint from epoch %d", epoch)
    model.eval()

    window_size_minutes = config["data"].get("window_size_minutes", cfg.WINDOW_SIZE_MINUTES)
    stride_minutes = config["data"].get("stride_minutes", cfg.WINDOW_STRIDE_MINUTES)
    prediction_horizon_minutes = config["data"].get("prediction_horizon_minutes", cfg.PREDICTION_HORIZON_MINUTES)

    dataset = LogSequenceDataset(
        logs_path=data_path,
        manifest_path=manifest_path,
        service_list=service_list,
        error_code_list=error_code_list,
        window_size_minutes=window_size_minutes,
        stride_minutes=stride_minutes,
        max_seq_length=config["data"].get("max_seq_length", cfg.MAX_SEQ_LENGTH),
        prediction_horizon_minutes=prediction_horizon_minutes,
    )
    # Must match train.py's splitter construction exactly (same gap_minutes
    # reasoning and config key — see its comment) or this "test" split
    # won't be the same windows train.py actually held out.
    split_gap_minutes = config["data"].get("split_gap_minutes", window_size_minutes + prediction_horizon_minutes)
    splitter = ChronologicalSplitter(
        train_ratio=config["data"].get("train_ratio", cfg.TRAIN_RATIO),
        val_ratio=config["data"].get("val_ratio", cfg.VAL_RATIO),
        test_ratio=config["data"].get("test_ratio", cfg.TEST_RATIO),
        gap_minutes=split_gap_minutes,
    )
    _, val_subset, test_subset = splitter.split_dataset(dataset)
    if len(val_subset) == 0 or len(test_subset) == 0:
        raise ValueError(
            f"split_gap_minutes={split_gap_minutes} left an empty val ({len(val_subset)}) "
            f"or test ({len(test_subset)}) split out of {len(dataset)} total windows — "
            "either the dataset is too short for this gap, or lower data.split_gap_minutes "
            "in the config."
        )

    edge_index = build_edge_index(config["data"]["service_dependencies"], service_list)
    batch_size = config["training"].get("batch_size", cfg.DEFAULT_BATCH_SIZE)
    num_workers = config["training"].get("num_workers", 4)
    batch_builder = LogBatchBuilder(edge_index=edge_index, num_services=len(service_list), device=device)

    def _run(loader, threshold: float) -> MetricsCalculator:
        calculator = MetricsCalculator(num_failure_types=len(cfg.FAILURE_TYPES), threshold=threshold)
        with torch.no_grad():
            for batch_data in loader:
                log_batch = batch_builder.build(batch_data)
                labels = batch_builder.extract_labels(batch_data)
                output = model.predict_with_uncertainty(log_batch) if use_uncertainty else model(log_batch)
                calculator.update(
                    output.failure_probability, labels["is_failure"],
                    output.failure_type_logits, labels["failure_type"],
                    output.time_to_failure, labels["time_to_failure"],
                )
        return calculator

    # Calibrate the binary decision threshold on val, then apply it to
    # test — MetricsCalculator.find_optimal_threshold() already existed
    # (constrained to FPR <= target, maximizing F1 within that) but was
    # never called anywhere, so every run silently used the arbitrary
    # default of 0.5 regardless of what the model's probabilities actually
    # looked like. Calibrating on val rather than test matters: picking the
    # threshold from the same split you report metrics on would bias those
    # metrics optimistically.
    val_loader = create_dataloader(
        val_subset, batch_size=batch_size, service_list=service_list,
        augmentor=None, shuffle=False, num_workers=num_workers,
    )
    val_calculator = _run(val_loader, threshold=0.5)
    calibrated_threshold, _ = val_calculator.find_optimal_threshold(target_fpr=cfg.TARGET_FALSE_POSITIVE_RATE)
    logger.info(
        "Calibrated decision threshold on val split: %.3f (target FPR <= %.2f)",
        calibrated_threshold, cfg.TARGET_FALSE_POSITIVE_RATE,
    )

    test_loader = create_dataloader(
        test_subset, batch_size=batch_size, service_list=service_list,
        augmentor=None, shuffle=False, num_workers=num_workers,
    )
    metrics_calculator = _run(test_loader, threshold=calibrated_threshold)
    metrics = metrics_calculator.compute()
    target_results = _check_targets(metrics)

    results = {
        "checkpoint": str(checkpoint_path),
        "epoch": epoch,
        "calibrated_threshold": float(calibrated_threshold),
        "metrics": metrics.to_dict(),
        "targets": TARGETS,
        "target_results": target_results,
        "all_targets_met": all(target_results.values()),
    }

    _log_results(metrics, target_results)

    if output_path:
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        logger.info("Saved results to %s", output_path)

    return results


def _check_targets(metrics: MetricsResult) -> Dict[str, bool]:
    # bool(...): metrics.* fields are numpy scalars (sklearn/np return
    # types). numpy.float64 subclasses Python's float so it happens to
    # serialize fine, but a numpy.bool_ produced by comparing one does
    # NOT subclass Python's bool and json.dump() rejects it outright —
    # without this cast, --output always crashed after printing results.
    return {
        "precision": bool(metrics.precision >= TARGETS["precision"]),
        "recall": bool(metrics.recall >= TARGETS["recall"]),
        "fpr": bool(metrics.false_positive_rate <= TARGETS["false_positive_rate"]),
        "failure_type_accuracy": bool(metrics.failure_type_accuracy >= TARGETS["failure_type_accuracy"]),
        "time_mae": bool(metrics.time_to_failure_mae <= TARGETS["time_to_failure_mae"]),
    }


def _log_results(metrics: MetricsResult, target_results: Dict[str, bool]) -> None:
    def status(met: bool) -> str:
        return "PASS" if met else "FAIL"

    logger.info("=" * 60)
    logger.info("EVALUATION RESULTS")
    logger.info("=" * 60)
    logger.info("Binary classification:")
    logger.info("  Precision: %.4f (target >= %.2f) [%s]", metrics.precision, TARGETS["precision"], status(target_results["precision"]))
    logger.info("  Recall:    %.4f (target >= %.2f) [%s]", metrics.recall, TARGETS["recall"], status(target_results["recall"]))
    logger.info("  F1:        %.4f", metrics.f1)
    logger.info("  ROC-AUC:   %.4f", metrics.roc_auc)
    logger.info("  FPR:       %.4f (target <= %.2f) [%s]", metrics.false_positive_rate, TARGETS["false_positive_rate"], status(target_results["fpr"]))
    logger.info("Failure type classification:")
    logger.info("  Accuracy:  %.4f (target >= %.2f) [%s]", metrics.failure_type_accuracy, TARGETS["failure_type_accuracy"], status(target_results["failure_type_accuracy"]))
    logger.info("Time-to-failure regression:")
    logger.info("  MAE:       %.2fs (target <= %.1fs) [%s]", metrics.time_to_failure_mae, TARGETS["time_to_failure_mae"], status(target_results["time_mae"]))
    logger.info("  RMSE:      %.2fs", metrics.time_to_failure_rmse)

    all_met = all(target_results.values())
    logger.info("=" * 60)
    logger.info("ALL TARGETS MET" if all_met else "TARGETS NOT MET: %s", "" if all_met else [k for k, v in target_results.items() if not v])
    logger.info("=" * 60)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the LogSentinel Module 3 model")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--uncertainty", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    evaluate(
        checkpoint_path=args.checkpoint, config_path=args.config, data_path=args.data,
        manifest_path=args.manifest, output_path=args.output, use_uncertainty=args.uncertainty,
    )


if __name__ == "__main__":
    main()
