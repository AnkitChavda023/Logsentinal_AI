"""Tests for training/splitter.py's ChronologicalSplitter."""

from datetime import datetime, timedelta

import pandas as pd

from module3_deep_learning.training.splitter import ChronologicalSplitter


def _window_starts(n, stride_minutes=1, base=None):
    base = base or datetime(2026, 1, 1)
    return pd.Series([base + timedelta(minutes=i * stride_minutes) for i in range(n)])


def test_zero_gap_leaves_train_and_val_windows_adjacent():
    """Baseline: with gap_minutes=0 (the old default), the train/val boundary
    windows are immediately adjacent in time — this is what made the
    overlapping-window leakage possible in the first place."""
    starts = _window_starts(100)
    splitter = ChronologicalSplitter(train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, gap_minutes=0)
    train_info, val_info, _ = splitter.split_by_time(starts, len(starts))
    assert val_info.start_time == train_info.end_time


def test_gap_minutes_creates_real_separation_between_splits():
    """Regression test: train.py used to construct ChronologicalSplitter
    with no gap_minutes (defaulting to 0). With window_size=5min and
    stride=1min, adjacent windows share up to 4 minutes of raw log rows by
    construction, and a window's label looks forward through the full
    prediction_horizon — so with zero gap, the last train window's content
    and label-horizon both reached into what was nominally the val/test
    period. train.py/evaluate.py now pass
    gap_minutes=window_size_minutes+prediction_horizon_minutes; this checks
    the splitter actually enforces that gap rather than just accepting the
    parameter and ignoring it."""
    starts = _window_starts(100)
    gap = 10
    splitter = ChronologicalSplitter(train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, gap_minutes=gap)
    train_info, val_info, test_info = splitter.split_by_time(starts, len(starts))

    assert (val_info.start_time - train_info.end_time).total_seconds() == gap * 60
    assert (test_info.start_time - val_info.end_time).total_seconds() == gap * 60
    # No window index should appear in more than one split.
    assert set(train_info.indices).isdisjoint(val_info.indices)
    assert set(val_info.indices).isdisjoint(test_info.indices)
    assert set(train_info.indices).isdisjoint(test_info.indices)


def test_gap_excludes_windows_within_the_embargoed_region():
    """With windows spaced 1 minute apart and a 10-minute gap, the ~10
    windows immediately before/after each split boundary must be excluded
    entirely (neither train nor val), not silently assigned to one side —
    that's the actual leakage-prevention mechanism."""
    starts = _window_starts(100)
    splitter = ChronologicalSplitter(train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, gap_minutes=10)
    train_info, val_info, test_info = splitter.split_by_time(starts, len(starts))

    total_in_splits = len(train_info.indices) + len(val_info.indices) + len(test_info.indices)
    assert total_in_splits < 100, "some windows near the boundaries must be embargoed (excluded), not assigned"
