"""
Optimizer Factory for LogSentinel

Creates AdamW optimizer with configurable parameters.
"""

from __future__ import annotations

import logging
from typing import Optional, Dict, Any, Iterator

import torch
from torch import nn
from torch.optim import Optimizer, AdamW, Adam, SGD

logger = logging.getLogger(__name__)


def build_optimizer(
    model: nn.Module,
    optimizer_type: str = 'adamw',
    learning_rate: float = 1e-4,
    weight_decay: float = 0.01,
    betas: tuple = (0.9, 0.999),
    eps: float = 1e-8,
    momentum: float = 0.9,
    separate_lr_groups: bool = False,
    encoder_lr_multiplier: float = 0.1
) -> Optimizer:
    """
    Build optimizer for LogSentinel model.
    
    Args:
        model: Model to optimize
        optimizer_type: 'adamw', 'adam', or 'sgd'
        learning_rate: Base learning rate
        weight_decay: Weight decay factor
        betas: Adam betas
        eps: Adam epsilon
        momentum: SGD momentum
        separate_lr_groups: Use different LR for encoder vs decoder
        encoder_lr_multiplier: LR multiplier for encoder layers
        
    Returns:
        Configured optimizer
    """
    # Get parameter groups
    if separate_lr_groups:
        param_groups = _get_parameter_groups(
            model,
            learning_rate,
            encoder_lr_multiplier,
            weight_decay
        )
    else:
        param_groups = [{'params': model.parameters()}]
    
    # Build optimizer
    if optimizer_type.lower() == 'adamw':
        optimizer = AdamW(
            param_groups,
            lr=learning_rate,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay
        )
    elif optimizer_type.lower() == 'adam':
        optimizer = Adam(
            param_groups,
            lr=learning_rate,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay
        )
    elif optimizer_type.lower() == 'sgd':
        optimizer = SGD(
            param_groups,
            lr=learning_rate,
            momentum=momentum,
            weight_decay=weight_decay
        )
    else:
        raise ValueError(f"Unknown optimizer type: {optimizer_type}")
    
    logger.info(
        f"Built {optimizer_type} optimizer: "
        f"lr={learning_rate}, weight_decay={weight_decay}"
    )
    
    return optimizer


def _get_parameter_groups(
    model: nn.Module,
    base_lr: float,
    encoder_multiplier: float,
    weight_decay: float
) -> list:
    """
    Create parameter groups with different learning rates.
    
    Encoder layers (embedder, sequence_encoder, transformer) get lower LR.
    Decoder layers (heads) get base LR.
    """
    encoder_params = []
    decoder_params = []
    no_decay_params = []
    
    encoder_names = ['embedder', 'sequence_encoder', 'transformer_encoder', 'gnn']
    no_decay_names = ['bias', 'LayerNorm', 'layer_norm', 'norm']
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        
        # Check if no weight decay
        if any(nd in name for nd in no_decay_names):
            no_decay_params.append(param)
        # Check if encoder
        elif any(enc in name for enc in encoder_names):
            encoder_params.append(param)
        else:
            decoder_params.append(param)
    
    param_groups = [
        {
            'params': encoder_params,
            'lr': base_lr * encoder_multiplier,
            'weight_decay': weight_decay
        },
        {
            'params': decoder_params,
            'lr': base_lr,
            'weight_decay': weight_decay
        },
        {
            'params': no_decay_params,
            'lr': base_lr,
            'weight_decay': 0.0
        }
    ]
    
    # Filter out empty groups
    param_groups = [g for g in param_groups if len(g['params']) > 0]
    
    total_params = sum(len(g['params']) for g in param_groups)
    logger.info(f"Created {len(param_groups)} parameter groups with {total_params} parameters")
    
    return param_groups


class OptimizerConfig:
    """Configuration class for optimizer settings."""
    
    def __init__(
        self,
        optimizer_type: str = 'adamw',
        learning_rate: float = 1e-4,
        weight_decay: float = 0.01,
        betas: tuple = (0.9, 0.999),
        eps: float = 1e-8,
        momentum: float = 0.9,
        warmup_steps: int = 0,
        gradient_clip: float = 1.0
    ):
        self.optimizer_type = optimizer_type
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.betas = betas
        self.eps = eps
        self.momentum = momentum
        self.warmup_steps = warmup_steps
        self.gradient_clip = gradient_clip
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'optimizer_type': self.optimizer_type,
            'learning_rate': self.learning_rate,
            'weight_decay': self.weight_decay,
            'betas': self.betas,
            'eps': self.eps,
            'momentum': self.momentum,
            'warmup_steps': self.warmup_steps,
            'gradient_clip': self.gradient_clip
        }
    
    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> 'OptimizerConfig':
        return cls(**config)
