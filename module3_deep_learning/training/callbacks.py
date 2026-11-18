"""
Training Callbacks for LogSentinel

Implements:
- Early stopping (monitors validation loss)
- Model checkpointing (saves best weights)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.optim import Optimizer

logger = logging.getLogger(__name__)


class EarlyStopping:
    """
    Early stopping to terminate training when validation loss stops improving.
    
    From Section 13 concept checklist.
    """
    
    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 0.0,
        mode: str = 'min',
        restore_best_weights: bool = True
    ):
        """
        Initialize early stopping.
        
        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change to qualify as improvement
            mode: 'min' for loss, 'max' for metrics like accuracy
            restore_best_weights: Whether to restore best weights on stop
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.restore_best_weights = restore_best_weights
        
        self.counter = 0
        self.best_score: Optional[float] = None
        self.early_stop = False
        self.best_weights: Optional[dict] = None
        
        if mode == 'min':
            self.is_better = lambda current, best: current < best - min_delta
        else:
            self.is_better = lambda current, best: current > best + min_delta
    
    def __call__(
        self,
        score: float,
        model: Optional[nn.Module] = None
    ) -> bool:
        """
        Check if training should stop.
        
        Args:
            score: Current validation score
            model: Model to save weights from
            
        Returns:
            True if should stop, False otherwise
        """
        if self.best_score is None:
            self.best_score = score
            if model is not None and self.restore_best_weights:
                self.best_weights = {
                    k: v.cpu().clone() for k, v in model.state_dict().items()
                }
            return False
        
        if self.is_better(score, self.best_score):
            self.best_score = score
            self.counter = 0
            if model is not None and self.restore_best_weights:
                self.best_weights = {
                    k: v.cpu().clone() for k, v in model.state_dict().items()
                }
            logger.debug(f"Early stopping: new best score {score:.4f}")
        else:
            self.counter += 1
            logger.debug(
                f"Early stopping: no improvement for {self.counter}/{self.patience} epochs"
            )
            
            if self.counter >= self.patience:
                self.early_stop = True
                logger.info(
                    f"Early stopping triggered after {self.patience} epochs "
                    f"without improvement. Best score: {self.best_score:.4f}"
                )
        
        return self.early_stop
    
    def restore_weights(self, model: nn.Module) -> None:
        """Restore best weights to model."""
        if self.best_weights is not None:
            model.load_state_dict(self.best_weights)
            logger.info("Restored best weights from early stopping")
    
    def reset(self) -> None:
        """Reset early stopping state."""
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_weights = None


class ModelCheckpoint:
    """
    Save model checkpoints during training.
    """
    
    def __init__(
        self,
        checkpoint_dir: Path,
        filename_prefix: str = 'model',
        monitor: str = 'val_loss',
        mode: str = 'min',
        save_best_only: bool = True,
        save_weights_only: bool = False
    ):
        """
        Initialize model checkpointing.
        
        Args:
            checkpoint_dir: Directory to save checkpoints
            filename_prefix: Prefix for checkpoint filenames
            monitor: Metric to monitor
            mode: 'min' or 'max'
            save_best_only: Only save when metric improves
            save_weights_only: Save only weights, not full state
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.filename_prefix = filename_prefix
        self.monitor = monitor
        self.mode = mode
        self.save_best_only = save_best_only
        self.save_weights_only = save_weights_only
        
        self.best_score: Optional[float] = None
        
        if mode == 'min':
            self.is_better = lambda current, best: current < best
        else:
            self.is_better = lambda current, best: current > best
    
    def __call__(
        self,
        score: float,
        model: nn.Module,
        optimizer: Optimizer,
        epoch: int
    ) -> Optional[Path]:
        """
        Save checkpoint if conditions are met.
        
        Args:
            score: Current metric value
            model: Model to save
            optimizer: Optimizer to save
            epoch: Current epoch number
            
        Returns:
            Path to saved checkpoint, or None if not saved
        """
        should_save = False
        
        if self.best_score is None:
            self.best_score = score
            should_save = True
        elif self.is_better(score, self.best_score):
            self.best_score = score
            should_save = True
        elif not self.save_best_only:
            should_save = True
        
        if not should_save:
            return None
        
        # Build filename
        if self.save_best_only:
            filename = f"{self.filename_prefix}_best.pt"
        else:
            filename = f"{self.filename_prefix}_epoch{epoch:03d}.pt"
        
        filepath = self.checkpoint_dir / filename
        
        # Build checkpoint
        if self.save_weights_only:
            checkpoint = model.state_dict()
        else:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'score': score,
                'best_score': self.best_score
            }
        
        torch.save(checkpoint, filepath)
        logger.info(f"Saved checkpoint to {filepath} (score: {score:.4f})")
        
        return filepath
    
    def load_best(
        self,
        model: nn.Module,
        optimizer: Optional[Optimizer] = None
    ) -> int:
        """
        Load the best checkpoint.
        
        Args:
            model: Model to load weights into
            optimizer: Optional optimizer to load state into
            
        Returns:
            Epoch number of loaded checkpoint
        """
        filepath = self.checkpoint_dir / f"{self.filename_prefix}_best.pt"
        
        if not filepath.exists():
            raise FileNotFoundError(f"No checkpoint found at {filepath}")
        
        checkpoint = torch.load(filepath)
        
        if self.save_weights_only:
            model.load_state_dict(checkpoint)
            return 0
        else:
            model.load_state_dict(checkpoint['model_state_dict'])
            if optimizer is not None:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            logger.info(
                f"Loaded checkpoint from epoch {checkpoint['epoch']} "
                f"(score: {checkpoint['score']:.4f})"
            )
            return checkpoint['epoch']


class LearningRateLogger:
    """
    Callback to log learning rate changes.
    """
    
    def __init__(self, optimizer: Optimizer):
        self.optimizer = optimizer
        self.lr_history: list = []
    
    def __call__(self) -> float:
        """Get and log current learning rate."""
        lr = self.optimizer.param_groups[0]['lr']
        self.lr_history.append(lr)
        return lr
    
    def get_history(self) -> list:
        """Return learning rate history."""
        return self.lr_history


class GradientMonitor:
    """
    Monitor gradient statistics during training.
    """
    
    def __init__(self, model: nn.Module):
        self.model = model
        self.gradient_norms: list = []
    
    def __call__(self) -> float:
        """Compute and log gradient norm."""
        total_norm = 0.0
        for p in self.model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        self.gradient_norms.append(total_norm)
        return total_norm
    
    def get_history(self) -> list:
        """Return gradient norm history."""
        return self.gradient_norms
