"""
Training Loop for LogSentinel Deep Learning

Implements the training loop with:
- Gradient clipping
- MLflow logging
- Periodic validation
- Early stopping support
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler, ReduceLROnPlateau
from torch.utils.data import DataLoader

try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

from .callbacks import EarlyStopping, ModelCheckpoint
from .dataloader import LogBatchBuilder
from .losses import MultiTaskLoss
from .metrics import MetricsCalculator

logger = logging.getLogger(__name__)


class Trainer:
    """Training orchestrator for the LogSentinel model."""

    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        loss_fn: MultiTaskLoss,
        train_loader: DataLoader,
        val_loader: DataLoader,
        batch_builder: LogBatchBuilder,
        device: torch.device,
        scheduler: Optional[_LRScheduler] = None,
        early_stopping: Optional[EarlyStopping] = None,
        checkpoint: Optional[ModelCheckpoint] = None,
        gradient_clip_value: float = 1.0,
        log_interval: int = 100,
        val_interval: int = 1,
        mlflow_experiment: Optional[str] = None,
    ):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.batch_builder = batch_builder
        self.device = device
        self.scheduler = scheduler
        self.early_stopping = early_stopping
        self.checkpoint = checkpoint
        self.gradient_clip_value = gradient_clip_value
        self.log_interval = log_interval
        self.val_interval = val_interval

        self.metrics_calculator = MetricsCalculator()

        self.current_epoch = 0
        self.global_step = 0
        self.training_history: List[Dict[str, Any]] = []

        self.mlflow_enabled = False
        if mlflow_experiment and MLFLOW_AVAILABLE:
            try:
                mlflow.set_experiment(mlflow_experiment)
                self.mlflow_enabled = True
                logger.info("MLflow logging enabled for experiment: %s", mlflow_experiment)
            except Exception as e:
                logger.warning("Failed to set up MLflow: %s", e)

    def train(self, num_epochs: int, start_epoch: int = 0) -> Dict[str, List[float]]:
        if self.mlflow_enabled:
            mlflow.start_run()
            self._log_params()

        try:
            for epoch in range(start_epoch, num_epochs):
                self.current_epoch = epoch
                train_metrics = self._train_epoch()
                val_metrics: Optional[Dict[str, float]] = None

                if (epoch + 1) % self.val_interval == 0:
                    val_metrics = self._validate()

                    if self.scheduler is not None:
                        if isinstance(self.scheduler, ReduceLROnPlateau):
                            self.scheduler.step(val_metrics["loss"])
                        else:
                            self.scheduler.step()

                    if self.early_stopping is not None:
                        self.early_stopping(val_metrics["loss"], self.model)
                        if self.early_stopping.early_stop:
                            logger.info("Early stopping triggered at epoch %d", epoch + 1)
                            self._log_epoch(epoch, train_metrics, val_metrics)
                            self.training_history.append({"epoch": epoch, "train": train_metrics, "val": val_metrics})
                            break

                    if self.checkpoint is not None:
                        self.checkpoint(val_metrics["loss"], self.model, self.optimizer, epoch)

                    self._log_epoch(epoch, train_metrics, val_metrics)

                self.training_history.append({"epoch": epoch, "train": train_metrics, "val": val_metrics})
        finally:
            if self.mlflow_enabled:
                mlflow.end_run()

        if self.early_stopping is not None and self.early_stopping.restore_best_weights:
            self.early_stopping.restore_weights(self.model)

        return self._compile_history()

    def _train_epoch(self) -> Dict[str, float]:
        self.model.train()

        epoch_loss = 0.0
        epoch_losses = {"failure_bce": 0.0, "type_ce": 0.0, "time_mse": 0.0, "mlkp_ce": 0.0}
        num_batches = 0
        start_time = time.time()

        for batch_idx, batch_data in enumerate(self.train_loader):
            self.global_step += 1

            log_batch = self.batch_builder.build(batch_data)
            labels = self.batch_builder.extract_labels(batch_data)

            self.optimizer.zero_grad()
            output = self.model(log_batch)

            loss, loss_dict = self.loss_fn(
                failure_logits=output.failure_probability,
                failure_type_logits=output.failure_type_logits,
                time_to_failure=output.time_to_failure,
                is_failure_target=labels["is_failure"],
                failure_type_target=labels["failure_type"],
                time_to_failure_target=labels["time_to_failure"],
                mlkp_logits=output.mlkp_logits,
                mlkp_targets=output.mlkp_targets,
            )

            loss.backward()
            if self.gradient_clip_value > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip_value)
            self.optimizer.step()

            epoch_loss += loss.item()
            for key in epoch_losses:
                if key in loss_dict:
                    epoch_losses[key] += loss_dict[key]
            num_batches += 1

            if (batch_idx + 1) % self.log_interval == 0:
                avg_loss = epoch_loss / num_batches
                logger.info(
                    "Epoch %d | Batch %d/%d | Loss: %.4f",
                    self.current_epoch + 1, batch_idx + 1, len(self.train_loader), avg_loss,
                )
                if self.mlflow_enabled:
                    mlflow.log_metrics(
                        {"train/loss": avg_loss, "train/learning_rate": self.optimizer.param_groups[0]["lr"]},
                        step=self.global_step,
                    )

        elapsed = time.time() - start_time
        avg_metrics = {"loss": epoch_loss / max(num_batches, 1), "elapsed_seconds": elapsed}
        for key, value in epoch_losses.items():
            avg_metrics[key] = value / max(num_batches, 1)
        return avg_metrics

    def _validate(self) -> Dict[str, float]:
        self.model.eval()
        self.metrics_calculator.reset()

        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch_data in self.val_loader:
                log_batch = self.batch_builder.build(batch_data)
                labels = self.batch_builder.extract_labels(batch_data)

                output = self.model(log_batch)
                loss, _ = self.loss_fn(
                    failure_logits=output.failure_probability,
                    failure_type_logits=output.failure_type_logits,
                    time_to_failure=output.time_to_failure,
                    is_failure_target=labels["is_failure"],
                    failure_type_target=labels["failure_type"],
                    time_to_failure_target=labels["time_to_failure"],
                    # model.eval() disables the MLKP branch (see
                    # full_model.py's forward), so these are always None
                    # here — passed through for symmetry with _train_epoch,
                    # not because validation actually uses them.
                    mlkp_logits=output.mlkp_logits,
                    mlkp_targets=output.mlkp_targets,
                )

                total_loss += loss.item()
                num_batches += 1

                self.metrics_calculator.update(
                    output.failure_probability, labels["is_failure"],
                    output.failure_type_logits, labels["failure_type"],
                    output.time_to_failure, labels["time_to_failure"],
                )

        metrics_result = self.metrics_calculator.compute()
        return {"loss": total_loss / max(num_batches, 1), **metrics_result.to_dict()}

    def _log_epoch(self, epoch: int, train_metrics: Dict[str, float], val_metrics: Dict[str, float]) -> None:
        logger.info(
            "\nEpoch %d Summary:\n  Train Loss: %.4f\n  Val Loss: %.4f\n"
            "  Val Precision: %.4f\n  Val Recall: %.4f\n  Val F1: %.4f\n"
            "  Time-to-Failure MAE: %.2fs\n",
            epoch + 1, train_metrics["loss"], val_metrics["loss"],
            val_metrics["precision"], val_metrics["recall"], val_metrics["f1"],
            val_metrics["time_to_failure_mae"],
        )
        if self.mlflow_enabled:
            mlflow.log_metrics({
                "val/loss": val_metrics["loss"],
                "val/precision": val_metrics["precision"],
                "val/recall": val_metrics["recall"],
                "val/f1": val_metrics["f1"],
                "val/roc_auc": val_metrics["roc_auc"],
                "val/time_mae": val_metrics["time_to_failure_mae"],
            }, step=self.global_step)

    def _log_params(self) -> None:
        if not self.mlflow_enabled:
            return
        params = {
            "batch_size": self.train_loader.batch_size,
            "gradient_clip": self.gradient_clip_value,
            "learning_rate": self.optimizer.param_groups[0]["lr"],
            "optimizer": type(self.optimizer).__name__,
        }
        # Distinguishes separate LSTM/GRU/BiRNN/all training runs in the
        # MLflow UI — otherwise identical-looking runs are only
        # distinguishable by their --output directory.
        model_config = getattr(self.model, "config", None)
        if model_config is not None and hasattr(model_config, "sequence_encoder"):
            params["sequence_encoder"] = model_config.sequence_encoder
        mlflow.log_params(params)

    def _compile_history(self) -> Dict[str, List[float]]:
        history: Dict[str, List[float]] = {
            "train_loss": [], "val_loss": [], "val_precision": [], "val_recall": [], "val_f1": [],
        }
        for record in self.training_history:
            history["train_loss"].append(record["train"]["loss"])
            if record["val"] is not None:
                history["val_loss"].append(record["val"]["loss"])
                history["val_precision"].append(record["val"]["precision"])
                history["val_recall"].append(record["val"]["recall"])
                history["val_f1"].append(record["val"]["f1"])
        return history
