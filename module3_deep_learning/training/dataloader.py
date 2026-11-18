"""
Custom DataLoader and collate function for LogSentinel

Handles variable-length sequences and builds graph batch data.
"""

from __future__ import annotations

import logging
from functools import partial
from typing import Any, Dict, List, Optional

import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)

# Raw per-log fields produced by LogSequenceDataset and consumed directly
# by LogBatch/LogEmbedder — kept as one list so collate/build stay in sync.
_RAW_FIELDS = [
    "service_ids", "log_level_onehot", "error_code_ids", "latency_ms",
    "cpu_usage", "memory_mb", "time_since_error", "hour_of_day",
    "is_anomaly", "trace_depth", "attention_mask",
]


def collate_fn(
    batch: List[Dict[str, Any]],
    service_list: List[str],
    augmentor: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Custom collate function for LogSentinel batches.

    Args:
        batch: List of samples from LogSequenceDataset.
        service_list: List of all service names (0-based index order).
        augmentor: Optional callable(features_dict) -> features_dict, applied
            per-sample before stacking. Pass one only for the training
            loader — never for validation/test.

    Returns:
        Collated batch dictionary.
    """
    batch_size = len(batch)
    num_services = len(service_list)

    if augmentor is not None:
        batch = [augmentor(sample) for sample in batch]

    collated: Dict[str, Any] = {
        field: torch.stack([sample[field] for sample in batch]) for field in _RAW_FIELDS
    }

    is_failure = torch.zeros(batch_size, num_services)
    failure_type = torch.full((batch_size, num_services), -1, dtype=torch.long)
    time_to_failure = torch.full((batch_size, num_services), -1.0)

    for b_idx, sample in enumerate(batch):
        for s_idx, service in enumerate(service_list):
            is_failure[b_idx, s_idx] = sample["is_failure"].get(service, 0.0)
            failure_type[b_idx, s_idx] = sample["failure_type"].get(service, -1)
            time_to_failure[b_idx, s_idx] = sample["time_to_failure"].get(service, -1.0)

    collated.update({
        "is_failure": is_failure,
        "failure_type": failure_type,
        "time_to_failure": time_to_failure,
        "num_services": num_services,
        "metadata": [
            {"window_start": s["window_start"], "window_end": s["window_end"]} for s in batch
        ],
    })
    return collated


def create_dataloader(
    dataset: Dataset,
    batch_size: int,
    service_list: List[str],
    augmentor: Optional[Any] = None,
    shuffle: bool = False,
    num_workers: int = 4,
    pin_memory: bool = True,
    drop_last: bool = False,
) -> DataLoader:
    """
    Create a DataLoader for LogSentinel training.

    Args:
        dataset: LogSequenceDataset (or Subset thereof).
        batch_size: Number of windows per batch.
        service_list: List of all service names.
        augmentor: Optional training-only augmentation pipeline (see
            training/augmentation.py's LogAugmentor). Leave None for
            validation/test loaders.
        shuffle: Whether to shuffle batch composition. Defaults to False —
            this project is sequential-only end to end (chronological
            splits, forward-looking labels, cascading/precursor signals
            that depend on real time order), so batch order shouldn't be
            the one place that isn't. Every current caller passes this
            explicitly regardless; the default matters for whichever
            caller doesn't.
        num_workers: Number of data loading workers.
        pin_memory: Whether to pin memory for GPU transfer.
        drop_last: Whether to drop an incomplete final batch.
    """
    # functools.partial around the module-level collate_fn, not a local
    # closure: with num_workers > 0, DataLoader pickles collate_fn to hand
    # to worker processes. On Windows (spawn-based multiprocessing, unlike
    # fork on Linux) a closure defined inside this function can't be
    # pickled at all and crashes every worker with PicklingError the
    # moment num_workers > 0 — which is also the DEFAULT here.
    collate = partial(collate_fn, service_list=service_list, augmentor=augmentor)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        collate_fn=collate,
    )


class LogBatchBuilder:
    """Converts a collated dict into a LogBatch ready for the model."""

    def __init__(self, edge_index: Tensor, num_services: int, device: Optional[torch.device] = None):
        """
        Args:
            edge_index: (2, E) static service dependency graph — the same
                graph for every window (topology doesn't change within a run).
            num_services: Number of graph nodes (== len(service_list)).
            device: Target device for tensors.
        """
        self.edge_index = edge_index
        self.num_services = num_services
        self.device = device or torch.device("cpu")

    def build(self, collated: Dict[str, Any]) -> "LogBatch":
        from module3_deep_learning.models.full_model import LogBatch

        return LogBatch(
            **{field: collated[field].to(self.device) for field in _RAW_FIELDS},
            edge_index=self.edge_index.to(self.device),
            num_services=self.num_services,
        )

    def extract_labels(self, collated: Dict[str, Any]) -> Dict[str, Tensor]:
        return {
            "is_failure": collated["is_failure"].to(self.device),
            "failure_type": collated["failure_type"].to(self.device),
            "time_to_failure": collated["time_to_failure"].to(self.device),
        }
