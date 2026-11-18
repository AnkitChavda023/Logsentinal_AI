"""
Training Entrypoint for LogSentinel Deep Learning

Thin wrapper that loads config and orchestrates training. Every path is a
CLI/config argument with a local-friendly default — this script runs
unmodified on Kaggle or locally, only --data/--model-dir/--config change
(see module3_deep_learning/KAGGLE_TRAINING.md). Device selection is
automatic: CUDA if available, CPU otherwise.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Optional

import torch
import yaml

from module3_deep_learning.config import defaults as cfg
from module3_deep_learning.models.full_model import LogSentinelModel, model_config_from_yaml
from .augmentation import LogAugmentor
from .callbacks import EarlyStopping, ModelCheckpoint
from .checkpoint import load_checkpoint
from .dataloader import LogBatchBuilder, create_dataloader
from .dataset import LogSequenceDataset
from .losses import MultiTaskLoss
from .optimizer import build_optimizer
from .scheduler import build_scheduler
from .splitter import ChronologicalSplitter
from .trainer import Trainer

logger = logging.getLogger(__name__)


def build_edge_index(dependencies: dict, service_list: list) -> torch.Tensor:
    """Build a (2, E) edge_index from a {service: [deps]} dict, 0-based indices
    matching service_list's order (the same index space dataset.py uses)."""
    service_to_idx = {s: i for i, s in enumerate(service_list)}
    edges = [
        [service_to_idx[service], service_to_idx[dep]]
        for service, deps in dependencies.items()
        if service in service_to_idx
        for dep in deps
        if dep in service_to_idx
    ]
    if not edges:
        return torch.zeros(2, 0, dtype=torch.long)
    return torch.tensor(edges, dtype=torch.long).t().contiguous()


def train(
    config_path: Path,
    data_path: Path,
    manifest_path: Path,
    output_dir: Path,
    resume_from: Optional[Path] = None,
) -> None:
    """Run training from a configuration file.

    Args:
        config_path: Path to training config YAML.
        data_path: Path to training data Parquet (local dir or
            /kaggle/input/... on Kaggle).
        manifest_path: Path to failure_manifest.csv.
        output_dir: Directory for checkpoints/outputs (local dir or
            /kaggle/working/... on Kaggle).
        resume_from: Optional checkpoint to resume from.
    """
    with open(config_path) as f:
        config = yaml.safe_load(f)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    service_list = config["data"]["service_list"]
    error_code_list = config["data"]["error_code_list"]

    model_config = model_config_from_yaml(
        model_cfg_yaml=config.get("model", {}),
        num_services=len(service_list),
        num_error_codes=len(error_code_list),
        max_seq_length=config["data"].get("max_seq_length", cfg.MAX_SEQ_LENGTH),
    )
    model = LogSentinelModel(model_config).to(device)
    logger.info("Model parameters: %s", model.get_num_parameters())

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

    # gap_minutes >= window_size + prediction_horizon is required, not just a
    # nicety: with stride < window_size, adjacent windows share raw log rows
    # by construction, and a window's label looks forward through the full
    # horizon — with the previous gap_minutes=0 default, the last train
    # window's content and label-horizon both reached into what was
    # nominally the val/test period (and val into test), so the model was
    # partially training on near-duplicates of what it was "held out" from.
    # This is the standard purge/embargo gap for time-series CV with
    # overlapping, forward-looking-labeled windows. Configurable, because a
    # short/smoke-test dataset can have too few total windows to survive
    # the full gap — see training_example.yaml's override, and the
    # ValueError below instead of silently training on an empty split.
    split_gap_minutes = config["data"].get("split_gap_minutes", window_size_minutes + prediction_horizon_minutes)
    splitter = ChronologicalSplitter(
        train_ratio=config["data"].get("train_ratio", cfg.TRAIN_RATIO),
        val_ratio=config["data"].get("val_ratio", cfg.VAL_RATIO),
        test_ratio=config["data"].get("test_ratio", cfg.TEST_RATIO),
        gap_minutes=split_gap_minutes,
    )
    train_subset, val_subset, _test_subset = splitter.split_dataset(dataset)
    if len(train_subset) == 0 or len(val_subset) == 0:
        raise ValueError(
            f"split_gap_minutes={split_gap_minutes} left an empty train ({len(train_subset)}) "
            f"or val ({len(val_subset)}) split out of {len(dataset)} total windows — either the "
            "dataset is too short for this gap, or lower data.split_gap_minutes in the config."
        )

    edge_index = build_edge_index(config["data"]["service_dependencies"], service_list)
    batch_size = config["training"].get("batch_size", cfg.DEFAULT_BATCH_SIZE)
    num_workers = config["training"].get("num_workers", 4)

    augmentor = LogAugmentor() if config["training"].get("augment", True) else None
    # shuffle=False: logs are inter-dependent across time (cascading
    # failures, gradual precursor drift, service-to-service propagation),
    # so training batches are built in the same chronological order as
    # val/test rather than the more common SGD practice of shuffling batch
    # composition each epoch — the whole pipeline (split, windowing,
    # eval) is sequential-only by design, and batch order shouldn't be
    # the one place that isn't.
    train_loader = create_dataloader(
        train_subset, batch_size=batch_size, service_list=service_list,
        augmentor=augmentor, shuffle=False, num_workers=num_workers,
    )
    val_loader = create_dataloader(
        val_subset, batch_size=batch_size, service_list=service_list,
        augmentor=None, shuffle=False, num_workers=num_workers,
    )

    optimizer = build_optimizer(
        model,
        optimizer_type=config["training"].get("optimizer", "adamw"),
        learning_rate=config["training"].get("learning_rate", cfg.DEFAULT_LEARNING_RATE),
        weight_decay=config["training"].get("weight_decay", cfg.DEFAULT_WEIGHT_DECAY),
    )
    scheduler = build_scheduler(
        optimizer,
        scheduler_type=config["training"].get("scheduler", "reduce_on_plateau"),
        num_epochs=config["training"].get("num_epochs", cfg.DEFAULT_NUM_EPOCHS),
    )

    # Computed over the train split only — using val/test windows here would
    # leak their class balance into a weight that's only supposed to counter
    # imbalance in what the model is actually trained on (see dataset.py's
    # get_class_weights/get_failure_type_class_weights docstrings for why
    # this was needed: unweighted losses were collapsing to "always predict
    # normal" / "always predict the single most common failure type").
    pos_weight = dataset.get_class_weights(indices=train_subset.indices)
    type_class_weights = dataset.get_failure_type_class_weights(indices=train_subset.indices)
    logger.info("Failure pos_weight=%.3f, failure-type class weights=%s", pos_weight, type_class_weights.tolist())

    loss_weights = config["training"].get("loss_weights", {})
    loss_fn = MultiTaskLoss(
        failure_weight=loss_weights.get("failure", cfg.LOSS_WEIGHT_FAILURE),
        type_weight=loss_weights.get("type", cfg.LOSS_WEIGHT_TYPE),
        time_weight=loss_weights.get("time", cfg.LOSS_WEIGHT_TIME),
        mlkp_weight=loss_weights.get("mlkp", cfg.LOSS_WEIGHT_MLKP),
        pos_weight=pos_weight,
        type_class_weights=type_class_weights,
    ).to(device)

    early_stopping = EarlyStopping(
        patience=config["training"].get("early_stopping_patience", cfg.DEFAULT_EARLY_STOPPING_PATIENCE)
    )
    checkpoint = ModelCheckpoint(checkpoint_dir=output_dir / "checkpoints", save_best_only=True)

    start_epoch = 0
    if resume_from is not None:
        _, _, start_epoch, _ = load_checkpoint(resume_from, model, optimizer, device)
        logger.info("Resumed from epoch %d", start_epoch)

    batch_builder = LogBatchBuilder(edge_index=edge_index, num_services=len(service_list), device=device)
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        train_loader=train_loader,
        val_loader=val_loader,
        batch_builder=batch_builder,
        device=device,
        scheduler=scheduler,
        early_stopping=early_stopping,
        checkpoint=checkpoint,
        gradient_clip_value=config["training"].get("gradient_clip", cfg.DEFAULT_GRADIENT_CLIP),
        mlflow_experiment=config.get("mlflow_experiment"),
    )

    trainer.train(num_epochs=config["training"].get("num_epochs", cfg.DEFAULT_NUM_EPOCHS), start_epoch=start_epoch)

    final_path = output_dir / "model_final.pt"
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), final_path)
    logger.info("Saved final model to %s", final_path)
    logger.info("Training complete!")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the LogSentinel Module 3 model")
    parser.add_argument("--config", type=Path, required=True, help="Training config YAML path")
    parser.add_argument("--data", type=Path, required=True, help="Log data Parquet path")
    parser.add_argument("--manifest", type=Path, required=True, help="Failure manifest CSV path")
    parser.add_argument("--output", type=Path, required=True, help="Output directory")
    parser.add_argument("--resume", type=Path, help="Resume from checkpoint")
    args = parser.parse_args()

    train(
        config_path=args.config,
        data_path=args.data,
        manifest_path=args.manifest,
        output_dir=args.output,
        resume_from=args.resume,
    )


if __name__ == "__main__":
    main()
