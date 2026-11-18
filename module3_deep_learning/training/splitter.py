"""
Chronological Train/Validation/Test Splitter

Implements time-based splitting to prevent future information leakage.
Section 9.2: 80/10/10 split, validation must come after training chronologically.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import List, Tuple, Optional

import pandas as pd
from torch.utils.data import Subset

logger = logging.getLogger(__name__)


@dataclass
class SplitInfo:
    """Information about a data split."""
    name: str
    start_time: datetime
    end_time: datetime
    num_samples: int
    indices: List[int]


class ChronologicalSplitter:
    """
    Splits data chronologically to prevent future information leakage.
    
    From Section 9.2:
    - Training: 80% (earliest data)
    - Validation: 10% (middle data)
    - Test: 10% (latest data, held out until final evaluation)
    
    IMPORTANT: Never shuffle! Validation window must come after training window.
    """
    
    def __init__(
        self,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        gap_minutes: int = 0
    ):
        """
        Initialize the splitter.
        
        Args:
            train_ratio: Fraction of data for training
            val_ratio: Fraction of data for validation
            test_ratio: Fraction of data for testing
            gap_minutes: Gap between splits to prevent leakage at boundaries
        """
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
            "Ratios must sum to 1.0"
        
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.gap_minutes = gap_minutes
    
    def split_by_time(
        self,
        timestamps: pd.Series,
        total_samples: int
    ) -> Tuple[SplitInfo, SplitInfo, SplitInfo]:
        """
        Split based on timestamps.
        
        Args:
            timestamps: Series of datetime timestamps
            total_samples: Total number of samples
            
        Returns:
            Tuple of (train_info, val_info, test_info)
        """
        min_time = timestamps.min()
        max_time = timestamps.max()
        total_duration = (max_time - min_time).total_seconds()
        
        # Calculate split times
        train_end = min_time + pd.Timedelta(seconds=total_duration * self.train_ratio)
        val_end = train_end + pd.Timedelta(seconds=total_duration * self.val_ratio)
        
        # Add gaps if specified
        gap = pd.Timedelta(minutes=self.gap_minutes)
        val_start = train_end + gap
        test_start = val_end + gap
        
        # Create masks
        train_mask = timestamps < train_end
        val_mask = (timestamps >= val_start) & (timestamps < val_end)
        test_mask = timestamps >= test_start
        
        # Get indices
        train_indices = list(train_mask[train_mask].index)
        val_indices = list(val_mask[val_mask].index)
        test_indices = list(test_mask[test_mask].index)
        
        train_info = SplitInfo(
            name="train",
            start_time=min_time,
            end_time=train_end,
            num_samples=len(train_indices),
            indices=train_indices
        )
        
        val_info = SplitInfo(
            name="validation",
            start_time=val_start,
            end_time=val_end,
            num_samples=len(val_indices),
            indices=val_indices
        )
        
        test_info = SplitInfo(
            name="test",
            start_time=test_start,
            end_time=max_time,
            num_samples=len(test_indices),
            indices=test_indices
        )
        
        logger.info(f"Split results:")
        logger.info(f"  Train: {train_info.num_samples} samples ({train_info.start_time} to {train_info.end_time})")
        logger.info(f"  Val:   {val_info.num_samples} samples ({val_info.start_time} to {val_info.end_time})")
        logger.info(f"  Test:  {test_info.num_samples} samples ({test_info.start_time} to {test_info.end_time})")
        
        return train_info, val_info, test_info
    
    def split_dataset(
        self,
        dataset: "LogSequenceDataset"
    ) -> Tuple[Subset, Subset, Subset]:
        """
        Split a LogSequenceDataset into train/val/test subsets.
        
        Args:
            dataset: The full dataset to split
            
        Returns:
            Tuple of (train_subset, val_subset, test_subset)
        """
        # Get window timestamps
        window_starts = pd.Series([w[0] for w in dataset.windows])
        
        # Get split info
        train_info, val_info, test_info = self.split_by_time(
            window_starts,
            len(dataset)
        )
        
        # Create subsets
        train_subset = Subset(dataset, train_info.indices)
        val_subset = Subset(dataset, val_info.indices)
        test_subset = Subset(dataset, test_info.indices)
        
        return train_subset, val_subset, test_subset
    
    def split_indices(
        self,
        windows: List[Tuple[datetime, datetime]]
    ) -> Tuple[List[int], List[int], List[int]]:
        """
        Get train/val/test indices from window list.
        
        Args:
            windows: List of (start_time, end_time) tuples
            
        Returns:
            Tuple of (train_indices, val_indices, test_indices)
        """
        window_starts = pd.Series([w[0] for w in windows])
        
        train_info, val_info, test_info = self.split_by_time(
            window_starts,
            len(windows)
        )
        
        return train_info.indices, val_info.indices, test_info.indices


def create_splits(
    dataset: "LogSequenceDataset",
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1
) -> Tuple[Subset, Subset, Subset]:
    """
    Convenience function to create chronological splits.
    
    Args:
        dataset: Dataset to split
        train_ratio: Training data fraction
        val_ratio: Validation data fraction
        test_ratio: Test data fraction
        
    Returns:
        Tuple of (train_subset, val_subset, test_subset)
    """
    splitter = ChronologicalSplitter(train_ratio, val_ratio, test_ratio)
    return splitter.split_dataset(dataset)
