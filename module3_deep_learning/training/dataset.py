"""
LogSequenceDataset - PyTorch Dataset for log sequences

Loads Parquet files and prepares (sequence, labels) samples for training.
Labels come from the Module 1 ground-truth manifest.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from module3_deep_learning.config import defaults as cfg

logger = logging.getLogger(__name__)

# Every log row on disk also carries free-text/high-cardinality columns
# (message, trace_id, span_id, parent_span_id, instance_id, user_id,
# http_status, failure_type) that nothing in this file reads — only
# _extract_raw_fields' inputs below are ever used. Loading them anyway is
# harmless on a small dataset but was the direct cause of an OOM on a
# real-size one (~155M rows): pulling only these columns straight out of
# parquet, instead of the full schema, is a multi-x memory cut for free.
_NEEDED_LOG_COLUMNS = [
    "timestamp", "service", "log_level", "error_code",
    "latency_ms", "cpu_usage", "memory_mb", "is_anomaly",
]


def _to_naive_utc(series: "pd.Series") -> "pd.Series":
    """Strip tz-awareness (assuming UTC, which is what module1_log_generator
    always uses) so every timestamp column in this module compares safely
    against every other, regardless of whether pyarrow/pandas happened to
    preserve tz info for that particular column."""
    if getattr(series.dt, "tz", None) is not None:
        return series.dt.tz_localize(None)
    return series


class LogSequenceDataset(Dataset):
    """
    PyTorch Dataset for log sequences.

    Transforms raw logs into training samples with:
    - Sliding windows of 5 minutes
    - Per-service labels from ground truth manifest
    - Raw per-log fields matching embeddings.LogEmbedder's input (§7.2)

    Every field this dataset produces is intentionally still "raw" (a
    service index, a latency in ms, ...) rather than pre-projected into
    the 37-dim embedding space — that projection, including the learned
    service/error-code embeddings, is LogSentinelModel's job. Baking it
    into the dataset would mean those embeddings could never actually be
    *learned* (this used to be a real bug: the old feature extractor
    wrote zeros into placeholder embedding slots that nothing ever
    filled in, so the model's service and error-code embedding layers
    received constant input and contributed nothing).
    """

    def __init__(
        self,
        logs_path: Path,
        manifest_path: Path,
        service_list: List[str],
        error_code_list: List[str],
        window_size_minutes: int = cfg.WINDOW_SIZE_MINUTES,
        stride_minutes: int = cfg.WINDOW_STRIDE_MINUTES,
        max_seq_length: int = cfg.MAX_SEQ_LENGTH,
        prediction_horizon_minutes: int = cfg.PREDICTION_HORIZON_MINUTES,
    ):
        """
        Args:
            logs_path: Path to Parquet file containing logs.
            manifest_path: Path to CSV file with ground truth failures.
            service_list: List of all service names, in the SAME order
                used everywhere else in Module 3 (graph nodes, predictor
                output). Index i in this list == service_id i == graph
                node i — one canonical 0-based index space, no separate
                "+1 for unknown" offset.
            error_code_list: List of all error codes (for consistent
                indexing). Index 0 is reserved for "no error"; real
                codes are 1-based.
            window_size_minutes: Size of each log window.
            stride_minutes: Stride between windows.
            max_seq_length: Maximum sequence length (truncate if exceeded).
            prediction_horizon_minutes: How far ahead to predict failures.
        """
        self.logs_path = Path(logs_path)
        self.manifest_path = Path(manifest_path)
        self.service_list = service_list
        self.error_code_list = error_code_list
        self.window_size_minutes = window_size_minutes
        self.stride_minutes = stride_minutes
        self.max_seq_length = max_seq_length
        self.prediction_horizon = timedelta(minutes=prediction_horizon_minutes)

        # 0-based: service_list[i] <-> service_id i, matching graph node
        # indices (see LogBatch's docstring in models/full_model.py).
        self.service_to_idx = {s: i for i, s in enumerate(service_list)}
        # 1-based: 0 reserved for "no error" (padding_idx in LogEmbedder).
        self.error_code_to_idx = {e: i + 1 for i, e in enumerate(error_code_list)}

        # Canonical failure-type order, shared with predictor_service.py and
        # the model's class index space (see config/defaults.py's docstring
        # for why this must be a fixed list and not derived per-file).
        self.failure_type_to_idx = {ft: i for i, ft in enumerate(cfg.FAILURE_TYPES)}

        self._load_data()
        self._create_windows()

        logger.info(
            "LogSequenceDataset initialized: %d windows, %d total logs, %d failure events",
            len(self.windows), len(self.logs_df), len(self.failures_df),
        )

    def _read_and_shrink(self, path: Path) -> pd.DataFrame:
        """Read one parquet part, columns pruned to _NEEDED_LOG_COLUMNS and
        the low-cardinality string columns encoded as categoricals with a
        fixed, shared category set (service_list / LOG_LEVEL_MAP / +error
        codes) — fixed so every part's categories agree and pd.concat can
        stay categorical instead of silently widening back to object dtype
        the moment two parts' observed value sets differ.

        Also parses timestamp to a compact datetime64 here rather than
        after concatenating every part (the previous approach): parquet's
        raw round-trip dtype for a tz-aware column is a ~72-byte Python
        object per row, 9x a datetime64[ns]'s 8 bytes — on a real-size
        dataset that gap alone was bigger than every other column in this
        method combined."""
        df = pd.read_parquet(path, columns=_NEEDED_LOG_COLUMNS)
        df["service"] = df["service"].astype(self._service_category_dtype)
        df["log_level"] = df["log_level"].astype(self._log_level_category_dtype)
        df["error_code"] = df["error_code"].fillna("").astype(self._error_code_category_dtype)
        df["timestamp"] = _to_naive_utc(pd.to_datetime(df["timestamp"]))
        return df

    def _load_data(self) -> None:
        """Load logs and failure manifest."""
        logger.info("Loading logs from %s", self.logs_path)
        self._service_category_dtype = pd.CategoricalDtype(categories=self.service_list)
        self._log_level_category_dtype = pd.CategoricalDtype(categories=list(cfg.LOG_LEVEL_MAP.keys()))
        self._error_code_category_dtype = pd.CategoricalDtype(categories=[""] + list(self.error_code_list))

        if self.logs_path.is_dir():
            # Module 1 writes failure_manifest.csv and service_graph.html
            # into the SAME directory as the *.parquet parts (see its
            # README's "how to run" — this is the natural, documented
            # layout). Pointing pd.read_parquet() straight at that
            # directory makes pyarrow try to parse every file in it as
            # Parquet, including the CSV, and crash. Glob explicitly
            # instead, same pattern module2_classical_ml/training/evaluate.py
            # uses for the same reason.
            parts = sorted(self.logs_path.glob("*.parquet"))
            if not parts:
                raise FileNotFoundError(f"No .parquet files found in {self.logs_path}")
            self.logs_df = pd.concat([self._read_and_shrink(p) for p in parts], ignore_index=True)
        else:
            self.logs_df = self._read_and_shrink(self.logs_path)

        self.logs_df = self.logs_df.sort_values("timestamp").reset_index(drop=True)

        logger.info("Loading manifest from %s", self.manifest_path)
        self.failures_df = pd.read_csv(self.manifest_path)
        # Parquet (pyarrow) round-trips timestamps as tz-aware UTC, while the
        # manifest CSV's ISO-8601 strings (e.g. "...+00:00") parse to
        # tz-aware too but as a DIFFERENT pandas dtype than the Parquet
        # column ends up with — comparing them directly raises
        # "Invalid comparison between dtype=datetime64[ns, UTC] and
        # Timestamp". Both sides represent the same simulation clock, so
        # normalising both to naive UTC once, here, avoids relying on the
        # two file formats happening to agree on tz-awareness.
        self.failures_df["start_timestamp"] = _to_naive_utc(pd.to_datetime(self.failures_df["start_timestamp"]))
        self.failures_df["end_timestamp"] = _to_naive_utc(pd.to_datetime(self.failures_df["end_timestamp"]))

        unknown_types = set(self.failures_df["failure_type"].unique()) - set(cfg.FAILURE_TYPES)
        if unknown_types:
            logger.warning(
                "Manifest contains failure types not in the canonical taxonomy "
                "(config/defaults.py FAILURE_TYPES), they will be dropped as labels: %s",
                unknown_types,
            )

    def _create_windows(self) -> None:
        """Create sliding windows over the log data.

        A window's label looks *forward* from window_end through
        prediction_horizon for the next failure (_get_labels). Near the
        very end of the observed data, that forward-looking horizon runs
        past max_time, into time we have no logs/manifest coverage for —
        there's no way to tell "no failure happens" from "we simply
        stopped recording". _get_labels can't distinguish the two and
        defaults to is_failure=0, so those trailing windows are silently
        force-labeled negative regardless of the true (unobserved) future.

        That's a real problem here specifically because splitting is
        chronological/sequential-only (ChronologicalSplitter — no
        shuffling, by design): the test split is always the *last* N% of
        windows by start time, i.e. exactly the right-censored tail. On
        a short-duration dataset the censored region (prediction_horizon
        worth of windows) can be the entire val/test split, guaranteeing
        near-zero true positives there independent of how well the model
        actually learned — this was confirmed empirically: on the 43-
        simulated-minute microservices23 run, the last 5 of 40 windows
        (exactly prediction_horizon_minutes / stride_minutes) had zero
        positive labels while the dataset overall was ~37% positive.

        Fix: stop generating windows once their horizon would extend
        past max_time, so every window this dataset ever produces (and
        therefore every split of it) has a fully-observed label.
        """
        self.windows: List[Tuple[datetime, datetime]] = []

        min_time = self.logs_df["timestamp"].min()
        max_time = self.logs_df["timestamp"].max()
        window_delta = timedelta(minutes=self.window_size_minutes)
        stride_delta = timedelta(minutes=self.stride_minutes)
        label_visibility_cutoff = max_time - self.prediction_horizon

        current_start = min_time
        while current_start + window_delta <= label_visibility_cutoff:
            self.windows.append((current_start, current_start + window_delta))
            current_start += stride_delta

        logger.info(
            "Created %d windows (label-visibility cutoff %s, prediction_horizon=%s)",
            len(self.windows), label_visibility_cutoff, self.prediction_horizon,
        )

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        window_start, window_end = self.windows[idx]

        mask = (self.logs_df["timestamp"] >= window_start) & (self.logs_df["timestamp"] < window_end)
        window_logs = self.logs_df[mask]

        raw = self._extract_raw_fields(window_logs)
        labels = self._get_labels(window_end)

        return {
            **raw,
            "is_failure": labels["is_failure"],
            "failure_type": labels["failure_type"],
            "time_to_failure": labels["time_to_failure"],
            "window_start": window_start.isoformat(),
            "window_end": window_end.isoformat(),
        }

    def _extract_raw_fields(self, logs: pd.DataFrame) -> Dict[str, torch.Tensor]:
        """
        Extract the raw (pre-embedding) per-log fields LogBatch/LogEmbedder
        expect (§7.2), padded/truncated to max_seq_length.
        """
        L = self.max_seq_length
        service_ids = np.zeros(L, dtype=np.int64)
        log_level_onehot = np.zeros((L, cfg.LOG_LEVEL_DIM), dtype=np.float32)
        error_code_ids = np.zeros(L, dtype=np.int64)
        latency_ms = np.zeros(L, dtype=np.float32)
        cpu_usage = np.zeros(L, dtype=np.float32)
        memory_mb = np.zeros(L, dtype=np.float32)
        time_since_error = np.full(L, cfg.MAX_TIME_SINCE_ERROR_S, dtype=np.float32)
        hour_of_day = np.zeros(L, dtype=np.float32)
        is_anomaly = np.zeros(L, dtype=np.float32)
        # Module 1 doesn't currently emit a trace-depth column on the flat
        # schema, so this stays at a constant placeholder until it does —
        # see module3_deep_learning/README.md "Known limitations".
        trace_depth = np.ones(L, dtype=np.float32)
        attention_mask = np.zeros(L, dtype=np.float32)

        n = min(len(logs), L)
        last_error_time: Dict[str, datetime] = {}

        for i, row in enumerate(logs.head(n).itertuples()):
            service = getattr(row, "service", "unknown")
            service_ids[i] = self.service_to_idx.get(service, 0)

            log_level = getattr(row, "log_level", "INFO")
            level_idx = cfg.LOG_LEVEL_MAP.get(log_level, 1)
            log_level_onehot[i, level_idx] = 1.0

            error_code = getattr(row, "error_code", "") or ""
            error_code_ids[i] = self.error_code_to_idx.get(error_code, 0)

            latency_ms[i] = float(getattr(row, "latency_ms", 0.0) or 0.0)
            cpu_usage[i] = float(getattr(row, "cpu_usage", 0.0) or 0.0)
            memory_mb[i] = float(getattr(row, "memory_mb", 0.0) or 0.0)

            timestamp = row.timestamp
            if log_level in ("ERROR", "FATAL", "CRITICAL"):
                last_error_time[service] = timestamp
            if service in last_error_time:
                time_since_error[i] = (timestamp - last_error_time[service]).total_seconds()

            hour_of_day[i] = timestamp.hour + timestamp.minute / 60.0
            is_anomaly[i] = 1.0 if bool(getattr(row, "is_anomaly", False)) else 0.0
            attention_mask[i] = 1.0

        return {
            "service_ids": torch.from_numpy(service_ids),
            "log_level_onehot": torch.from_numpy(log_level_onehot),
            "error_code_ids": torch.from_numpy(error_code_ids),
            "latency_ms": torch.from_numpy(latency_ms),
            "cpu_usage": torch.from_numpy(cpu_usage),
            "memory_mb": torch.from_numpy(memory_mb),
            "time_since_error": torch.from_numpy(time_since_error),
            "hour_of_day": torch.from_numpy(hour_of_day),
            "is_anomaly": torch.from_numpy(is_anomaly),
            "trace_depth": torch.from_numpy(trace_depth),
            "attention_mask": torch.from_numpy(attention_mask),
        }

    def _get_labels(self, window_end: datetime) -> Dict[str, Any]:
        """Per-service labels: does a failure start within the prediction horizon?"""
        horizon_end = window_end + self.prediction_horizon
        mask = (self.failures_df["start_timestamp"] >= window_end) & (
            self.failures_df["start_timestamp"] <= horizon_end
        )
        future_failures = self.failures_df[mask]

        is_failure: Dict[str, float] = {s: 0.0 for s in self.service_list}
        failure_type: Dict[str, int] = {s: -1 for s in self.service_list}
        time_to_failure: Dict[str, float] = {s: -1.0 for s in self.service_list}

        for _, failure in future_failures.iterrows():
            service = failure["affected_service"]
            failure_type_name = failure["failure_type"]
            if service in is_failure and failure_type_name in self.failure_type_to_idx:
                is_failure[service] = 1.0
                failure_type[service] = self.failure_type_to_idx[failure_type_name]
                ttf = (failure["start_timestamp"] - window_end).total_seconds()
                time_to_failure[service] = max(0.0, ttf)

        return {"is_failure": is_failure, "failure_type": failure_type, "time_to_failure": time_to_failure}

    def get_class_weights(self, indices: Optional[Sequence[int]] = None) -> float:
        """`pos_weight` for `nn.BCEWithLogitsLoss` (negative_count / positive_count),
        computed only over `indices` (pass the train split's indices — computing
        this over the whole dataset, including val/test, would leak their class
        balance into a weight that's only supposed to describe what the model is
        trained on). Real failures are a minority class by construction (§ note
        in training_microservices23.yaml); without this, BCEWithLogitsLoss
        default-weights the plentiful "normal" windows equally to the sparse
        failure ones, and the optimizer's easiest way to cut loss is to mostly
        predict "no failure" — exactly the high-precision/low-recall collapse
        seen in practice (LSTM on the 8h run: precision 0.785, recall 0.147)."""
        window_indices = range(len(self)) if indices is None else indices
        total_positive = 0
        total_negative = 0
        for idx in window_indices:
            labels = self._get_labels(self.windows[idx][1])
            for service in self.service_list:
                if labels["is_failure"][service] > 0.5:
                    total_positive += 1
                else:
                    total_negative += 1

        if total_positive == 0:
            return 1.0
        return total_negative / total_positive

    def get_failure_type_class_weights(self, indices: Optional[Sequence[int]] = None) -> torch.Tensor:
        """Balanced per-class weights for the failure-type CrossEntropyLoss,
        shape (num_failure_types,), computed only over `indices` (train split)
        and only over windows that actually have a failure (failure_type=-1
        windows are excluded from this loss via ignore_index already). Failure
        types are not remotely uniform (module1's failure_injection_rate picks
        a type uniformly, but real-world traffic/topology makes some far more
        exploitable than others), so an unweighted CrossEntropyLoss collapses
        to whichever type is most common — confirmed in practice: LSTM's
        per-class failure-type accuracy on the 8h run was 0.97 for one class
        and 0.0 for ten of the other eleven."""
        window_indices = range(len(self)) if indices is None else indices
        num_types = len(cfg.FAILURE_TYPES)
        counts = torch.zeros(num_types, dtype=torch.float32)
        for idx in window_indices:
            labels = self._get_labels(self.windows[idx][1])
            for service in self.service_list:
                type_idx = labels["failure_type"][service]
                if type_idx >= 0:
                    counts[type_idx] += 1

        total = counts.sum().item()
        if total == 0:
            return torch.ones(num_types, dtype=torch.float32)
        # Balanced weight = total / (num_classes * count). Classes never seen
        # in these `indices` are clamped to count=1 rather than given weight
        # 0: these weights get reused, unchanged, to score *validation*
        # batches with this same loss_fn (see train.py) — a type absent
        # from the chronologically-earlier train split can still show up
        # later in val/test, and a weight of exactly 0 there makes
        # CrossEntropyLoss's weighted-mean reduction divide by zero the
        # moment a batch's only non-ignored target is that class (NaN,
        # caught via a real local training dry run: "Val Loss: nan" every
        # epoch despite Train Loss being finite). Treating an unseen class
        # as "as rare as something seen exactly once" keeps its weight
        # high (correctly reflecting real scarcity) while staying finite.
        return total / (num_types * counts.clamp(min=1))
