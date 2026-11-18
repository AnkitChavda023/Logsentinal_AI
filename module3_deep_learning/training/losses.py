
"""
Multi-Task Loss Function for LogSentinel

Implements the joint loss from Section 7.6:
L_total = 1.0 × L_failure_bce + 0.5 × L_type_ce + 0.5 × L_time_mse

Time-to-failure loss is masked for samples that don't have failures.
"""

from __future__ import annotations

import logging
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from module3_deep_learning.config import defaults as cfg

logger = logging.getLogger(__name__)


class MultiTaskLoss(nn.Module):
    """
    Combined loss for multi-task failure prediction.
    
    Task 1: Binary failure prediction (BCE)
    Task 2: Failure type classification (CE)
    Task 3: Time-to-failure regression (MSE)
    
    Weights from Section 7.6:
    - L_failure_bce: 1.0
    - L_type_ce: 0.5
    - L_time_mse: 0.5
    """
    
    def __init__(
        self,
        failure_weight: float = cfg.LOSS_WEIGHT_FAILURE,
        type_weight: float = cfg.LOSS_WEIGHT_TYPE,
        time_weight: float = cfg.LOSS_WEIGHT_TIME,
        mlkp_weight: float = cfg.LOSS_WEIGHT_MLKP,
        pos_weight: Optional[float] = None,
        type_class_weights: Optional[Tensor] = None,
        label_smoothing: float = 0.0,
        time_scale: float = cfg.TIME_TO_FAILURE_MAX_SECONDS
    ):
        """
        Initialize the multi-task loss.

        Args:
            failure_weight: Weight for binary failure loss
            type_weight: Weight for failure type loss
            time_weight: Weight for time-to-failure loss
            mlkp_weight: Weight for the LogBERT-style masked log-key
                auxiliary loss (see full_model.py's _compute_mlkp) — modest
                relative to type/time weight since it's a regularizer, not
                a co-equal supervised task
            pos_weight: Positive class weight for imbalanced data (e.g.
                LogSequenceDataset.get_class_weights() on the train split)
            type_class_weights: Per-failure-type class weights, shape
                (num_failure_types,) (e.g.
                LogSequenceDataset.get_failure_type_class_weights())
            label_smoothing: Label smoothing for classification
            time_scale: Scale for normalizing time values
        """
        super().__init__()

        self.failure_weight = failure_weight
        self.type_weight = type_weight
        self.time_weight = time_weight
        self.mlkp_weight = mlkp_weight
        self.time_scale = time_scale
        self.mlkp_loss = nn.CrossEntropyLoss()

        # Binary cross-entropy with optional class weighting
        if pos_weight is not None:
            self.bce_loss = nn.BCEWithLogitsLoss(
                pos_weight=torch.tensor([pos_weight])
            )
        else:
            self.bce_loss = nn.BCEWithLogitsLoss()

        # Cross-entropy for failure type (ignore index -1 for no failure)
        self.ce_loss = nn.CrossEntropyLoss(
            ignore_index=-1,
            label_smoothing=label_smoothing,
            weight=type_class_weights,
        )
        
        # MSE for time-to-failure
        self.mse_loss = nn.MSELoss(reduction='none')
        
        logger.info(
            f"MultiTaskLoss initialized: "
            f"failure_weight={failure_weight}, type_weight={type_weight}, "
            f"time_weight={time_weight}"
        )
    
    def forward(
        self,
        failure_logits: Tensor,
        failure_type_logits: Tensor,
        time_to_failure: Tensor,
        is_failure_target: Tensor,
        failure_type_target: Tensor,
        time_to_failure_target: Tensor,
        mlkp_logits: Optional[Tensor] = None,
        mlkp_targets: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Dict[str, float]]:
        """
        Compute the combined multi-task loss.

        Args:
            failure_logits: [batch, num_services] raw logits for failure
            failure_type_logits: [batch, num_services, num_types] type logits
            time_to_failure: [batch, num_services] predicted time in seconds
            is_failure_target: [batch, num_services] binary targets
            failure_type_target: [batch, num_services] type indices (-1 for no failure)
            time_to_failure_target: [batch, num_services] time targets (-1 for no failure)
            mlkp_logits: [N_masked, num_services] LogBERT-style masked
                log-key logits (see full_model.py's _compute_mlkp), or None
                when the auxiliary task is off/not training/no positions
                were masked this batch — the term is skipped entirely then,
                same "guard when absent" shape as type_loss below.
            mlkp_targets: [N_masked] true service id for each masked position

        Returns:
            Tuple of (total_loss, loss_dict) where loss_dict contains individual losses
        """
        batch_size, num_services = failure_logits.shape
        
        # Task 1: Binary failure prediction
        # Flatten for BCE loss
        failure_loss = self.bce_loss(
            failure_logits.view(-1),
            is_failure_target.view(-1)
        )
        
        # Task 2: Failure type classification
        # Reshape: [batch * num_services, num_types] and [batch * num_services]
        # Most windows have no failure at all, so a batch's targets can be
        # entirely -1 (ignore_index) — nn.CrossEntropyLoss's default 'mean'
        # reduction divides by the count of non-ignored targets, which is 0
        # in that case, silently producing NaN and poisoning this batch's
        # total_loss (and every gradient downstream of it) instead of
        # raising. Same masked-fallback pattern as time_loss below.
        type_logits_flat = failure_type_logits.view(-1, failure_type_logits.size(-1))
        type_target_flat = failure_type_target.view(-1)
        if (type_target_flat != -1).any():
            type_loss = self.ce_loss(type_logits_flat, type_target_flat)
        else:
            type_loss = torch.tensor(0.0, device=failure_logits.device)

        # Task 3: Time-to-failure regression (masked)
        # Only compute loss where there is actually a failure
        time_mask = time_to_failure_target >= 0  # Valid time targets
        
        if time_mask.sum() > 0:
            # Normalize time values
            time_pred_normalized = time_to_failure / self.time_scale
            time_target_normalized = time_to_failure_target / self.time_scale
            
            # Compute masked MSE
            time_loss_all = self.mse_loss(time_pred_normalized, time_target_normalized)
            time_loss = (time_loss_all * time_mask.float()).sum() / time_mask.sum()
        else:
            time_loss = torch.tensor(0.0, device=failure_logits.device)

        # LogBERT-style masked log-key auxiliary loss (§3.2 Task I MLKP).
        # Skipped whenever mlkp_logits is None (auxiliary task disabled,
        # eval mode, or this batch happened to mask zero real positions) —
        # same guard shape as type_loss above, so it can never introduce a
        # NaN from an empty selection the way an unguarded CE would.
        if mlkp_logits is not None and mlkp_logits.numel() > 0:
            mlkp_loss = self.mlkp_loss(mlkp_logits, mlkp_targets)
        else:
            mlkp_loss = torch.tensor(0.0, device=failure_logits.device)

        # Combine losses
        total_loss = (
            self.failure_weight * failure_loss +
            self.type_weight * type_loss +
            self.time_weight * time_loss +
            self.mlkp_weight * mlkp_loss
        )

        # Return individual losses for logging
        loss_dict = {
            'total': total_loss.item(),
            'failure_bce': failure_loss.item(),
            'type_ce': type_loss.item(),
            'time_mse': time_loss.item(),
            'mlkp_ce': mlkp_loss.item(),
        }

        return total_loss, loss_dict


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance.
    
    FL(p_t) = -α_t (1 - p_t)^γ log(p_t)
    
    Useful for highly imbalanced failure data.
    """
    
    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        reduction: str = 'mean'
    ):
        """
        Initialize Focal Loss.
        
        Args:
            alpha: Weighting factor for positive class
            gamma: Focusing parameter (0 = CE, higher = more focus on hard examples)
            reduction: 'none', 'mean', or 'sum'
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs: Tensor, targets: Tensor) -> Tensor:
        """
        Compute focal loss.
        
        Args:
            inputs: Logits [batch_size]
            targets: Binary targets [batch_size]
            
        Returns:
            Focal loss value
        """
        p = torch.sigmoid(inputs)
        ce_loss = F.binary_cross_entropy_with_logits(
            inputs, targets, reduction='none'
        )
        
        p_t = p * targets + (1 - p) * (1 - targets)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        
        focal_weight = alpha_t * (1 - p_t) ** self.gamma
        focal_loss = focal_weight * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss


class MultiTaskFocalLoss(MultiTaskLoss):
    """
    Multi-task loss using Focal Loss for binary classification.
    
    Better handles the class imbalance in failure prediction.
    """
    
    def __init__(
        self,
        failure_weight: float = cfg.LOSS_WEIGHT_FAILURE,
        type_weight: float = cfg.LOSS_WEIGHT_TYPE,
        time_weight: float = cfg.LOSS_WEIGHT_TIME,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        **kwargs
    ):
        super().__init__(
            failure_weight=failure_weight,
            type_weight=type_weight,
            time_weight=time_weight,
            **kwargs
        )
        
        # Replace BCE with Focal Loss
        self.failure_loss_fn = FocalLoss(
            alpha=focal_alpha,
            gamma=focal_gamma
        )
    
    def forward(
        self,
        failure_logits: Tensor,
        failure_type_logits: Tensor,
        time_to_failure: Tensor,
        is_failure_target: Tensor,
        failure_type_target: Tensor,
        time_to_failure_target: Tensor,
        mlkp_logits: Optional[Tensor] = None,
        mlkp_targets: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Dict[str, float]]:
        """Compute loss with Focal Loss for failure prediction."""
        
        # Task 1: Focal loss for failure prediction
        failure_loss = self.failure_loss_fn(
            failure_logits.view(-1),
            is_failure_target.view(-1)
        )
        
        # Tasks 2 and 3 same as parent (see MultiTaskLoss.forward for why
        # type_loss needs the all-ignored guard)
        type_logits_flat = failure_type_logits.view(-1, failure_type_logits.size(-1))
        type_target_flat = failure_type_target.view(-1)
        if (type_target_flat != -1).any():
            type_loss = self.ce_loss(type_logits_flat, type_target_flat)
        else:
            type_loss = torch.tensor(0.0, device=failure_logits.device)

        time_mask = time_to_failure_target >= 0
        if time_mask.sum() > 0:
            time_pred_norm = time_to_failure / self.time_scale
            time_target_norm = time_to_failure_target / self.time_scale
            time_loss_all = self.mse_loss(time_pred_norm, time_target_norm)
            time_loss = (time_loss_all * time_mask.float()).sum() / time_mask.sum()
        else:
            time_loss = torch.tensor(0.0, device=failure_logits.device)

        # Same MLKP guard as MultiTaskLoss.forward (see there for rationale).
        if mlkp_logits is not None and mlkp_logits.numel() > 0:
            mlkp_loss = self.mlkp_loss(mlkp_logits, mlkp_targets)
        else:
            mlkp_loss = torch.tensor(0.0, device=failure_logits.device)

        total_loss = (
            self.failure_weight * failure_loss +
            self.type_weight * type_loss +
            self.time_weight * time_loss +
            self.mlkp_weight * mlkp_loss
        )

        loss_dict = {
            'total': total_loss.item(),
            'failure_focal': failure_loss.item(),
            'type_ce': type_loss.item(),
            'time_mse': time_loss.item(),
            'mlkp_ce': mlkp_loss.item(),
        }

        return total_loss, loss_dict
