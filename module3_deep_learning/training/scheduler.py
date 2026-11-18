"""
Learning Rate Scheduler Factory

Creates ReduceLROnPlateau and other schedulers.
"""

from __future__ import annotations

import logging
from typing import Optional

from torch.optim import Optimizer
from torch.optim.lr_scheduler import (
    _LRScheduler,
    ReduceLROnPlateau,
    CosineAnnealingLR,
    CosineAnnealingWarmRestarts,
    OneCycleLR,
    StepLR,
    ExponentialLR,
    LambdaLR
)

logger = logging.getLogger(__name__)


def build_scheduler(
    optimizer: Optimizer,
    scheduler_type: str = 'reduce_on_plateau',
    num_epochs: int = 100,
    steps_per_epoch: int = 100,
    **kwargs
) -> _LRScheduler:
    """
    Build learning rate scheduler.
    
    Args:
        optimizer: Optimizer to schedule
        scheduler_type: Type of scheduler
        num_epochs: Total training epochs
        steps_per_epoch: Steps per epoch (for OneCycleLR)
        **kwargs: Additional scheduler arguments
        
    Returns:
        Configured scheduler
    """
    if scheduler_type == 'reduce_on_plateau':
        # verbose= was removed from ReduceLROnPlateau in newer PyTorch
        # (superseded by get_last_lr()/optimizer.param_groups introspection,
        # which trainer.py already logs each epoch) — not passed here.
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode=kwargs.get('mode', 'min'),
            factor=kwargs.get('factor', 0.5),
            patience=kwargs.get('patience', 5),
            min_lr=kwargs.get('min_lr', 1e-7)
        )
    
    elif scheduler_type == 'cosine':
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=num_epochs,
            eta_min=kwargs.get('min_lr', 1e-7)
        )
    
    elif scheduler_type == 'cosine_warm_restarts':
        scheduler = CosineAnnealingWarmRestarts(
            optimizer,
            T_0=kwargs.get('T_0', 10),
            T_mult=kwargs.get('T_mult', 2),
            eta_min=kwargs.get('min_lr', 1e-7)
        )
    
    elif scheduler_type == 'one_cycle':
        scheduler = OneCycleLR(
            optimizer,
            max_lr=kwargs.get('max_lr', optimizer.param_groups[0]['lr'] * 10),
            epochs=num_epochs,
            steps_per_epoch=steps_per_epoch,
            pct_start=kwargs.get('pct_start', 0.3),
            anneal_strategy=kwargs.get('anneal_strategy', 'cos')
        )
    
    elif scheduler_type == 'step':
        scheduler = StepLR(
            optimizer,
            step_size=kwargs.get('step_size', 30),
            gamma=kwargs.get('gamma', 0.1)
        )
    
    elif scheduler_type == 'exponential':
        scheduler = ExponentialLR(
            optimizer,
            gamma=kwargs.get('gamma', 0.95)
        )
    
    elif scheduler_type == 'warmup_cosine':
        scheduler = WarmupCosineScheduler(
            optimizer,
            warmup_epochs=kwargs.get('warmup_epochs', 5),
            total_epochs=num_epochs,
            min_lr=kwargs.get('min_lr', 1e-7)
        )
    
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")
    
    logger.info(f"Built {scheduler_type} scheduler")
    
    return scheduler


class WarmupCosineScheduler(_LRScheduler):
    """
    Learning rate scheduler with linear warmup followed by cosine annealing.
    """
    
    def __init__(
        self,
        optimizer: Optimizer,
        warmup_epochs: int,
        total_epochs: int,
        min_lr: float = 1e-7,
        last_epoch: int = -1
    ):
        """
        Initialize warmup cosine scheduler.
        
        Args:
            optimizer: Optimizer to schedule
            warmup_epochs: Number of warmup epochs
            total_epochs: Total number of epochs
            min_lr: Minimum learning rate
            last_epoch: Last epoch number
        """
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self) -> list:
        """Compute learning rate for current epoch."""
        if self.last_epoch < self.warmup_epochs:
            # Linear warmup
            warmup_factor = (self.last_epoch + 1) / self.warmup_epochs
            return [base_lr * warmup_factor for base_lr in self.base_lrs]
        else:
            # Cosine annealing
            import math
            progress = (self.last_epoch - self.warmup_epochs) / \
                      (self.total_epochs - self.warmup_epochs)
            cosine_factor = 0.5 * (1 + math.cos(math.pi * progress))
            return [
                self.min_lr + (base_lr - self.min_lr) * cosine_factor
                for base_lr in self.base_lrs
            ]


class LinearWarmupScheduler(_LRScheduler):
    """
    Simple linear warmup scheduler.
    """
    
    def __init__(
        self,
        optimizer: Optimizer,
        warmup_steps: int,
        last_epoch: int = -1
    ):
        self.warmup_steps = warmup_steps
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self) -> list:
        if self.last_epoch < self.warmup_steps:
            warmup_factor = (self.last_epoch + 1) / self.warmup_steps
            return [base_lr * warmup_factor for base_lr in self.base_lrs]
        return self.base_lrs
