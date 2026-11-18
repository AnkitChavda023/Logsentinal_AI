"""
Metrics Calculator for LogSentinel

Computes all metrics from Section 10.1:
- Anomaly Detection Precision > 90%
- Anomaly Detection Recall > 85%
- Root Cause Accuracy > 80%
- Failure Type Accuracy > 75%
- Time-to-Failure MAE < 30 seconds
- False Positive Rate < 5%
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import Tensor
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    precision_recall_curve,
    average_precision_score,
    confusion_matrix,
    accuracy_score
)

from module3_deep_learning.config import defaults as cfg

logger = logging.getLogger(__name__)


@dataclass
class MetricsResult:
    """Container for computed metrics."""
    
    # Binary classification metrics
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    accuracy: float = 0.0
    roc_auc: float = 0.0
    pr_auc: float = 0.0
    false_positive_rate: float = 0.0
    
    # Multi-class metrics
    failure_type_accuracy: float = 0.0
    per_class_accuracy: Dict[int, float] = field(default_factory=dict)
    
    # Regression metrics
    time_to_failure_mae: float = 0.0
    time_to_failure_rmse: float = 0.0
    
    # Confusion matrix
    confusion_matrix: Optional[np.ndarray] = None
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary for logging.

        float(...) everywhere: these fields come from sklearn/numpy ops
        and are typically numpy.float64 (which happens to subclass
        Python's float and would serialize fine) but sometimes
        numpy.float32 (torch tensors default to float32, and .numpy()
        preserves that) — which does NOT subclass float and makes
        json.dump() raise "Object of type float32 is not JSON
        serializable" the moment evaluate.py's --output actually had a
        non-default value to write.
        """
        result = {
            'precision': float(self.precision),
            'recall': float(self.recall),
            'f1': float(self.f1),
            'accuracy': float(self.accuracy),
            'roc_auc': float(self.roc_auc),
            'pr_auc': float(self.pr_auc),
            'false_positive_rate': float(self.false_positive_rate),
            'failure_type_accuracy': float(self.failure_type_accuracy),
            'time_to_failure_mae': float(self.time_to_failure_mae),
            'time_to_failure_rmse': float(self.time_to_failure_rmse)
        }

        # Add per-class accuracy
        for cls, acc in self.per_class_accuracy.items():
            result[f'class_{cls}_accuracy'] = float(acc)

        return result
    
    def meets_targets(self) -> Dict[str, bool]:
        """Check if metrics meet Section 10.1 targets."""
        # bool(...): see evaluate.py's _check_targets — numpy.bool_ from
        # comparing numpy scalars isn't JSON-serializable, unlike
        # numpy.float64 which subclasses Python's float.
        return {
            'precision_target': bool(self.precision >= cfg.TARGET_PRECISION),
            'recall_target': bool(self.recall >= cfg.TARGET_RECALL),
            'fpr_target': bool(self.false_positive_rate <= cfg.TARGET_FALSE_POSITIVE_RATE),
            'type_accuracy_target': bool(self.failure_type_accuracy >= cfg.TARGET_FAILURE_TYPE_ACCURACY),
            'time_mae_target': bool(self.time_to_failure_mae <= cfg.TARGET_TIME_TO_FAILURE_MAE_S)
        }


class MetricsCalculator:
    """
    Computes all evaluation metrics for LogSentinel.
    """
    
    def __init__(
        self,
        threshold: float = 0.5,
        num_failure_types: int = cfg.NUM_FAILURE_TYPES
    ):
        """
        Initialize the calculator.
        
        Args:
            threshold: Classification threshold for binary predictions
            num_failure_types: Number of failure type classes
        """
        self.threshold = threshold
        self.num_failure_types = num_failure_types
        
        # Accumulators for batch-wise computation
        self.reset()
    
    def reset(self) -> None:
        """Reset all accumulators."""
        self.failure_probs: List[np.ndarray] = []
        self.failure_targets: List[np.ndarray] = []
        self.type_preds: List[np.ndarray] = []
        self.type_targets: List[np.ndarray] = []
        self.time_preds: List[np.ndarray] = []
        self.time_targets: List[np.ndarray] = []
    
    def update(
        self,
        failure_prob: Tensor,
        failure_target: Tensor,
        type_logits: Tensor,
        type_target: Tensor,
        time_pred: Tensor,
        time_target: Tensor
    ) -> None:
        """
        Update accumulators with batch results.
        
        Args:
            failure_prob: [batch, num_services] failure probabilities
            failure_target: [batch, num_services] binary targets
            type_logits: [batch, num_services, num_types] type logits
            type_target: [batch, num_services] type targets (-1 for no failure)
            time_pred: [batch, num_services] predicted time
            time_target: [batch, num_services] actual time (-1 for no failure)
        """
        # Convert to numpy
        failure_prob_np = failure_prob.detach().cpu().numpy().flatten()
        failure_target_np = failure_target.detach().cpu().numpy().flatten()
        
        self.failure_probs.append(failure_prob_np)
        self.failure_targets.append(failure_target_np)
        
        # Type predictions (argmax)
        type_pred_np = type_logits.argmax(dim=-1).detach().cpu().numpy().flatten()
        type_target_np = type_target.detach().cpu().numpy().flatten()
        
        self.type_preds.append(type_pred_np)
        self.type_targets.append(type_target_np)
        
        # Time predictions
        time_pred_np = time_pred.detach().cpu().numpy().flatten()
        time_target_np = time_target.detach().cpu().numpy().flatten()
        
        self.time_preds.append(time_pred_np)
        self.time_targets.append(time_target_np)
    
    def compute(self) -> MetricsResult:
        """
        Compute all metrics from accumulated batches.
        
        Returns:
            MetricsResult with all computed metrics
        """
        # Concatenate all batches
        all_probs = np.concatenate(self.failure_probs)
        all_targets = np.concatenate(self.failure_targets)
        all_type_preds = np.concatenate(self.type_preds)
        all_type_targets = np.concatenate(self.type_targets)
        all_time_preds = np.concatenate(self.time_preds)
        all_time_targets = np.concatenate(self.time_targets)
        
        result = MetricsResult()
        
        # Binary classification metrics
        all_preds = (all_probs >= self.threshold).astype(int)
        
        result.precision = precision_score(
            all_targets, all_preds, zero_division=0
        )
        result.recall = recall_score(
            all_targets, all_preds, zero_division=0
        )
        result.f1 = f1_score(
            all_targets, all_preds, zero_division=0
        )
        result.accuracy = accuracy_score(all_targets, all_preds)
        
        # ROC-AUC and PR-AUC — undefined when only one class is present in
        # all_targets (e.g. a tiny eval split with zero positive windows).
        # Older sklearn raised ValueError for this; newer sklearn instead
        # warns and returns nan, which the `except` alone no longer catches
        # — left unchecked, that nan flows straight into the JSON report
        # (technically invalid JSON; also just not a meaningful "0 targets
        # met" result to report).
        try:
            roc_auc = roc_auc_score(all_targets, all_probs)
            pr_auc = average_precision_score(all_targets, all_probs)
            result.roc_auc = 0.0 if np.isnan(roc_auc) else roc_auc
            result.pr_auc = 0.0 if np.isnan(pr_auc) else pr_auc
        except ValueError:
            result.roc_auc = 0.0
            result.pr_auc = 0.0
        
        # False positive rate
        cm = confusion_matrix(all_targets, all_preds, labels=[0, 1])
        result.confusion_matrix = cm
        
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            result.false_positive_rate = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        
        # Failure type accuracy (only where failure actually occurred)
        valid_type_mask = all_type_targets >= 0
        if valid_type_mask.sum() > 0:
            result.failure_type_accuracy = accuracy_score(
                all_type_targets[valid_type_mask],
                all_type_preds[valid_type_mask]
            )
            
            # Per-class accuracy
            for cls in range(self.num_failure_types):
                cls_mask = all_type_targets == cls
                if cls_mask.sum() > 0:
                    result.per_class_accuracy[cls] = (
                        all_type_preds[cls_mask] == cls
                    ).mean()
        
        # Time-to-failure metrics (only where failure occurred)
        valid_time_mask = all_time_targets >= 0
        if valid_time_mask.sum() > 0:
            time_errors = np.abs(
                all_time_preds[valid_time_mask] - all_time_targets[valid_time_mask]
            )
            result.time_to_failure_mae = time_errors.mean()
            result.time_to_failure_rmse = np.sqrt((time_errors ** 2).mean())
        
        return result
    
    def compute_threshold_analysis(
        self,
        num_thresholds: int = 100
    ) -> Dict[str, np.ndarray]:
        """
        Analyze metrics at different thresholds.
        
        Args:
            num_thresholds: Number of threshold values to evaluate
            
        Returns:
            Dict with arrays of metrics at each threshold
        """
        all_probs = np.concatenate(self.failure_probs)
        all_targets = np.concatenate(self.failure_targets)
        
        thresholds = np.linspace(0, 1, num_thresholds)
        precisions = []
        recalls = []
        f1s = []
        fprs = []
        
        for thresh in thresholds:
            preds = (all_probs >= thresh).astype(int)
            
            precisions.append(precision_score(all_targets, preds, zero_division=0))
            recalls.append(recall_score(all_targets, preds, zero_division=0))
            f1s.append(f1_score(all_targets, preds, zero_division=0))
            
            cm = confusion_matrix(all_targets, preds, labels=[0, 1])
            if cm.shape == (2, 2):
                tn, fp, fn, tp = cm.ravel()
                fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
                fprs.append(fpr)
            else:
                fprs.append(0)
        
        return {
            'thresholds': thresholds,
            'precision': np.array(precisions),
            'recall': np.array(recalls),
            'f1': np.array(f1s),
            'fpr': np.array(fprs)
        }
    
    def find_optimal_threshold(
        self,
        target_fpr: float = 0.05
    ) -> Tuple[float, MetricsResult]:
        """
        Find optimal threshold for target FPR.
        
        Args:
            target_fpr: Target false positive rate (default 5% from Section 10.1)
            
        Returns:
            Tuple of (optimal_threshold, metrics_at_threshold)
        """
        analysis = self.compute_threshold_analysis()
        
        # Find threshold that achieves target FPR
        valid_mask = analysis['fpr'] <= target_fpr
        
        if not valid_mask.any():
            # Use highest threshold if can't meet target
            optimal_idx = -1
        else:
            # Find threshold with best F1 among those meeting FPR target
            valid_f1s = np.where(valid_mask, analysis['f1'], -1)
            optimal_idx = valid_f1s.argmax()
        
        optimal_threshold = analysis['thresholds'][optimal_idx]
        
        # Recompute metrics at optimal threshold
        old_threshold = self.threshold
        self.threshold = optimal_threshold
        result = self.compute()
        self.threshold = old_threshold
        
        return optimal_threshold, result
