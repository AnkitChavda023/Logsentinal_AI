"""
Checkpoint Management for LogSentinel

Handles saving and loading model checkpoints, including ONNX export.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Optional, Any, Tuple

import torch
import torch.nn as nn
from torch.optim import Optimizer

logger = logging.getLogger(__name__)


def save_checkpoint(
    model: nn.Module,
    optimizer: Optimizer,
    epoch: int,
    path: Path,
    metrics: Optional[Dict[str, float]] = None,
    config: Optional[Dict[str, Any]] = None
) -> None:
    """
    Save a training checkpoint.
    
    Args:
        model: Model to save
        optimizer: Optimizer to save
        epoch: Current epoch number
        path: Path to save checkpoint
        metrics: Optional metrics dict
        config: Optional config dict
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics or {},
        'config': config or {}
    }
    
    torch.save(checkpoint, path)
    logger.info(f"Saved checkpoint to {path} (epoch {epoch})")


def load_checkpoint(
    path: Path,
    model: Optional[nn.Module] = None,
    optimizer: Optional[Optimizer] = None,
    device: torch.device = torch.device('cpu')
) -> Tuple[Optional[nn.Module], Optional[Optimizer], int, Dict[str, Any]]:
    """
    Load a training checkpoint.
    
    Args:
        path: Path to checkpoint file
        model: Model to load weights into (optional)
        optimizer: Optimizer to load state into (optional)
        device: Device to load tensors to
        
    Returns:
        Tuple of (model, optimizer, epoch, metadata)
    """
    path = Path(path)
    
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    
    # weights_only=False: our checkpoints bundle optimizer state, epoch and
    # config dicts alongside the tensors, not just a plain state_dict, so
    # the newer torch default of weights_only=True (which rejects anything
    # but tensors) would break loading here.
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    
    if model is not None:
        model.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"Loaded model weights from {path}")
    
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        logger.info(f"Loaded optimizer state from {path}")
    
    epoch = checkpoint.get('epoch', 0)
    metadata = {
        'metrics': checkpoint.get('metrics', {}),
        'config': checkpoint.get('config', {})
    }
    
    return model, optimizer, epoch, metadata


def export_to_onnx(
    model: nn.Module,
    path: Path,
    input_shape: Tuple[int, ...],
    input_names: Optional[list] = None,
    output_names: Optional[list] = None,
    dynamic_axes: Optional[Dict[str, Dict[int, str]]] = None,
    opset_version: int = 14
) -> None:
    """
    Export model to ONNX format for production deployment.
    
    Args:
        model: Model to export
        path: Output ONNX file path
        input_shape: Shape of input tensor(s)
        input_names: Names for input tensors
        output_names: Names for output tensors
        dynamic_axes: Dynamic axis specifications
        opset_version: ONNX opset version
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    model.eval()
    
    # Create dummy input
    dummy_input = torch.randn(input_shape)
    
    # Default names
    if input_names is None:
        input_names = ['log_features']
    if output_names is None:
        output_names = ['failure_probability', 'failure_type', 'time_to_failure']
    
    # Export
    torch.onnx.export(
        model,
        dummy_input,
        path,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=opset_version,
        do_constant_folding=True
    )
    
    logger.info(f"Exported model to ONNX: {path}")


def export_to_torchscript(
    model: nn.Module,
    path: Path,
    method: str = 'trace',
    example_input: Optional[torch.Tensor] = None
) -> None:
    """
    Export model to TorchScript for production deployment.
    
    Args:
        model: Model to export
        path: Output file path
        method: 'trace' or 'script'
        example_input: Example input for tracing
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    model.eval()
    
    if method == 'trace':
        if example_input is None:
            raise ValueError("example_input required for tracing")
        scripted = torch.jit.trace(model, example_input)
    elif method == 'script':
        scripted = torch.jit.script(model)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    scripted.save(str(path))
    logger.info(f"Exported model to TorchScript: {path}")


class CheckpointManager:
    """
    Manages multiple checkpoints with automatic cleanup.
    """
    
    def __init__(
        self,
        checkpoint_dir: Path,
        max_checkpoints: int = 5
    ):
        """
        Initialize checkpoint manager.
        
        Args:
            checkpoint_dir: Directory for checkpoints
            max_checkpoints: Maximum number to keep
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.max_checkpoints = max_checkpoints
        self.checkpoints: list = []
    
    def save(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        epoch: int,
        score: float,
        **kwargs
    ) -> Path:
        """Save checkpoint and manage history."""
        filename = f"checkpoint_epoch{epoch:04d}_score{score:.4f}.pt"
        path = self.checkpoint_dir / filename
        
        save_checkpoint(model, optimizer, epoch, path, **kwargs)
        
        self.checkpoints.append((path, score, epoch))
        self._cleanup()
        
        return path
    
    def _cleanup(self) -> None:
        """Remove old checkpoints if exceeding max."""
        if len(self.checkpoints) <= self.max_checkpoints:
            return
        
        # Sort by score (keep best)
        self.checkpoints.sort(key=lambda x: x[1], reverse=True)
        
        # Remove excess
        while len(self.checkpoints) > self.max_checkpoints:
            path, _, _ = self.checkpoints.pop()
            if path.exists():
                path.unlink()
                logger.debug(f"Removed old checkpoint: {path}")
    
    def get_best(self) -> Optional[Path]:
        """Get path to best checkpoint."""
        if not self.checkpoints:
            return None
        
        self.checkpoints.sort(key=lambda x: x[1], reverse=True)
        return self.checkpoints[0][0]
    
    def get_latest(self) -> Optional[Path]:
        """Get path to latest checkpoint."""
        if not self.checkpoints:
            return None
        
        self.checkpoints.sort(key=lambda x: x[2], reverse=True)
        return self.checkpoints[0][0]
