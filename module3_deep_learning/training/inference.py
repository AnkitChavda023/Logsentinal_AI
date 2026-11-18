"""
Batch Inference for LogSentinel

Run inference on arbitrary log windows outside the live predictor service.
Useful for evaluating historical incidents or ad-hoc analysis of a
Parquet file.
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch

from module3_deep_learning.config import defaults as cfg
from module3_deep_learning.models.full_model import LogBatch, LogSentinelModel, ModelOutput
from module3_deep_learning.training.dataset import _to_naive_utc

logger = logging.getLogger(__name__)


class BatchInferenceEngine:
    """Runs batch inference on log data for a single analysis window."""

    def __init__(
        self,
        model: LogSentinelModel,
        service_list: List[str],
        error_code_list: List[str],
        edge_index: torch.Tensor,
        device: Optional[torch.device] = None,
        max_seq_length: int = cfg.MAX_SEQ_LENGTH,
    ):
        self.model = model
        self.service_list = service_list
        self.service_to_idx = {s: i for i, s in enumerate(service_list)}
        self.error_code_to_idx = {e: i + 1 for i, e in enumerate(error_code_list)}
        self.edge_index = edge_index
        self.device = device or torch.device("cpu")
        self.max_seq_length = max_seq_length

        self.model.to(self.device)
        self.model.eval()

    def predict(self, logs_df: pd.DataFrame, with_uncertainty: bool = False) -> Dict[str, Any]:
        batch = self._build_batch(logs_df)
        with torch.no_grad():
            output = self.model.predict_with_uncertainty(batch) if with_uncertainty else self.model(batch)
        return self._format_output(output, with_uncertainty)

    def predict_from_parquet(
        self,
        parquet_path: Path,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        with_uncertainty: bool = False,
    ) -> Dict[str, Any]:
        df = pd.read_parquet(parquet_path)
        # Normalise to naive UTC regardless of whether pyarrow preserved
        # tz-awareness on round-trip (see dataset.py's _to_naive_utc — same
        # "Invalid comparison between dtype=datetime64[ns, UTC] and
        # Timestamp" failure mode applies here against caller-supplied
        # start_time/end_time, which are typically naive datetimes).
        df["timestamp"] = _to_naive_utc(pd.to_datetime(df["timestamp"]))
        if start_time is not None:
            df = df[df["timestamp"] >= start_time.replace(tzinfo=None)]
        if end_time is not None:
            df = df[df["timestamp"] < end_time.replace(tzinfo=None)]

        if len(df) == 0:
            logger.warning("No logs found in specified time range")
            return {"predictions": [], "num_logs": 0}

        logger.info("Processing %d logs from %s", len(df), parquet_path)
        return self.predict(df, with_uncertainty)

    def _build_batch(self, logs_df: pd.DataFrame) -> LogBatch:
        """Build a (1, T, ...) LogBatch — one window — from a logs DataFrame."""
        L = self.max_seq_length
        n = min(len(logs_df), L)

        service_ids = np.zeros(L, dtype=np.int64)
        log_level_onehot = np.zeros((L, cfg.LOG_LEVEL_DIM), dtype=np.float32)
        error_code_ids = np.zeros(L, dtype=np.int64)
        latency_ms = np.zeros(L, dtype=np.float32)
        cpu_usage = np.zeros(L, dtype=np.float32)
        memory_mb = np.zeros(L, dtype=np.float32)
        time_since_error = np.full(L, cfg.MAX_TIME_SINCE_ERROR_S, dtype=np.float32)
        hour_of_day = np.zeros(L, dtype=np.float32)
        is_anomaly = np.zeros(L, dtype=np.float32)
        trace_depth = np.ones(L, dtype=np.float32)
        attention_mask = np.zeros(L, dtype=np.float32)

        last_error_time: Dict[str, datetime] = {}
        for i, row in enumerate(logs_df.head(n).itertuples()):
            service = getattr(row, "service", "unknown")
            service_ids[i] = self.service_to_idx.get(service, 0)

            log_level = getattr(row, "log_level", "INFO")
            log_level_onehot[i, cfg.LOG_LEVEL_MAP.get(log_level, 1)] = 1.0

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

        def _t(arr: np.ndarray) -> torch.Tensor:
            return torch.from_numpy(arr).unsqueeze(0).to(self.device)

        return LogBatch(
            service_ids=_t(service_ids),
            log_level_onehot=_t(log_level_onehot),
            error_code_ids=_t(error_code_ids),
            latency_ms=_t(latency_ms),
            cpu_usage=_t(cpu_usage),
            memory_mb=_t(memory_mb),
            time_since_error=_t(time_since_error),
            hour_of_day=_t(hour_of_day),
            is_anomaly=_t(is_anomaly),
            trace_depth=_t(trace_depth),
            attention_mask=_t(attention_mask),
            edge_index=self.edge_index.to(self.device),
            num_services=len(self.service_list),
        )

    def _format_output(self, output: ModelOutput, with_uncertainty: bool) -> Dict[str, Any]:
        """output tensors are (1, num_services, ...) — one window's worth."""
        failure_prob = output.failure_probability[0].cpu().numpy()
        failure_type_probs = torch.softmax(output.failure_type_logits[0], dim=-1).cpu().numpy()
        time_to_failure = output.time_to_failure[0].cpu().numpy()

        predictions = []
        for i, service in enumerate(self.service_list):
            pred = {
                "service": service,
                "failure_probability": float(failure_prob[i]),
                "predicted_failure_type": cfg.FAILURE_TYPES[int(failure_type_probs[i].argmax())],
                "failure_type_confidence": float(failure_type_probs[i].max()),
                "time_to_failure_seconds": float(time_to_failure[i]),
            }
            if with_uncertainty and output.failure_prob_std is not None:
                pred["failure_prob_std"] = float(output.failure_prob_std[0, i].cpu())
                pred["time_to_failure_std"] = float(output.time_to_failure_std[0, i].cpu())
            predictions.append(pred)

        predictions.sort(key=lambda x: -x["failure_probability"])
        return {
            "predictions": predictions,
            "num_services": len(self.service_list),
            "timestamp": datetime.utcnow().isoformat(),
        }
