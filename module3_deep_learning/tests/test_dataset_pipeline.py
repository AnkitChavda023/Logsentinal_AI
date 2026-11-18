"""
Tests for the dataset -> dataloader -> LogBatch -> model pipeline, using
small synthetic Parquet/CSV files (no dependency on a real module1 run).
Exercises the full path a real training run takes, catching wiring bugs
unit tests on individual pieces would miss.
"""

from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest
import torch

from module3_deep_learning.config import defaults as cfg
from module3_deep_learning.models.full_model import LogSentinelModel, ModelConfig
from module3_deep_learning.training.dataloader import LogBatchBuilder, create_dataloader
from module3_deep_learning.training.dataset import LogSequenceDataset
from module3_deep_learning.training.train import build_edge_index

SERVICES = ["api-gateway", "payment-service", "database"]
ERROR_CODES = ["ERR_TIMEOUT", "ERR_5XX"]


def _write_synthetic_data(tmp_path):
    # 1200 rows @ 1/sec = 20 simulated minutes. Needs enough headroom past
    # window_size_minutes + prediction_horizon_minutes (up to 2+10=12min in
    # these tests) for _create_windows' label-visibility cutoff to still
    # leave windows standing — see dataset.py's _create_windows docstring.
    start = datetime(2026, 1, 1, 0, 0, 0)
    rows = []
    for i in range(1200):
        ts = start + timedelta(seconds=i)
        service = SERVICES[i % len(SERVICES)]
        rows.append({
            "timestamp": ts, "service": service, "instance_id": f"{service}-pod-1",
            "trace_id": f"trace-{i}", "span_id": f"span-{i}", "log_level": "ERROR" if i % 37 == 0 else "INFO",
            "message": "x", "latency_ms": 50.0 + (i % 20), "cpu_usage": 0.3, "memory_mb": 512.0,
            "http_status": 500 if i % 37 == 0 else 200,
            "error_code": "ERR_TIMEOUT" if i % 37 == 0 else "",
            "user_id": "user_1", "is_anomaly": False, "failure_type": "",
        })
    logs_df = pd.DataFrame(rows)
    logs_path = tmp_path / "logs.parquet"
    logs_df.to_parquet(logs_path)

    manifest_df = pd.DataFrame([
        {
            "schedule_id": "SCHED-0001", "service_name": "payment-service", "affected_service": "payment-service",
            "failure_type": "LATENCY_SPIKE", "start_timestamp": start + timedelta(seconds=200),
            "end_timestamp": start + timedelta(seconds=260), "params": "{}",
        },
    ])
    manifest_path = tmp_path / "failure_manifest.csv"
    manifest_df.to_csv(manifest_path, index=False)

    return logs_path, manifest_path


def _write_tz_aware_synthetic_data(tmp_path):
    """
    Mirrors real module1_log_generator output: log timestamps carry
    tzinfo=UTC (module1's simulation clock is always UTC-aware), and the
    manifest CSV's timestamps are ISO-8601 strings with an explicit
    "+00:00" offset. Parquet round-tripping and pd.read_csv's ISO
    parsing don't necessarily agree on tz-awareness dtype even though
    both represent the same UTC instants — this fixture exists
    specifically to catch that mismatch.
    """
    # See _write_synthetic_data's comment: needs >= window_size + horizon
    # of headroom past the fixed failure event for windows to survive the
    # label-visibility cutoff.
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    rows = []
    for i in range(1200):
        ts = start + timedelta(seconds=i)
        service = SERVICES[i % len(SERVICES)]
        rows.append({
            "timestamp": ts, "service": service, "instance_id": f"{service}-pod-1",
            "trace_id": f"trace-{i}", "span_id": f"span-{i}", "log_level": "INFO",
            "message": "x", "latency_ms": 50.0, "cpu_usage": 0.3, "memory_mb": 512.0,
            "http_status": 200, "error_code": "", "user_id": "user_1",
            "is_anomaly": False, "failure_type": "",
        })
    logs_df = pd.DataFrame(rows)
    logs_path = tmp_path / "logs.parquet"
    logs_df.to_parquet(logs_path)

    manifest_df = pd.DataFrame([{
        "failure_id": "F0001", "start_timestamp": (start + timedelta(seconds=200)).isoformat(),
        "end_timestamp": (start + timedelta(seconds=260)).isoformat(), "affected_service": "payment-service",
        "failure_type": "LATENCY_SPIKE", "root_cause_service": "payment-service", "severity": "MEDIUM",
        "phase_transitions": "[]", "resolution_time": (start + timedelta(seconds=260)).isoformat(),
    }])
    manifest_path = tmp_path / "failure_manifest.csv"
    manifest_df.to_csv(manifest_path, index=False)

    return logs_path, manifest_path


def test_dataset_handles_tz_aware_logs_and_manifest(tmp_path):
    """
    Regression test: comparing logs_df's tz-aware Parquet-round-tripped
    timestamp column against the manifest CSV's independently-parsed
    tz-aware column used to raise
    "Invalid comparison between dtype=datetime64[ns, UTC] and Timestamp"
    the first time a real (non-zero) label lookup ran — LogSequenceDataset
    must normalise both to the same tz representation once, up front.
    """
    logs_path, manifest_path = _write_tz_aware_synthetic_data(tmp_path)
    dataset = LogSequenceDataset(
        logs_path=logs_path, manifest_path=manifest_path, service_list=SERVICES,
        error_code_list=ERROR_CODES, window_size_minutes=2, stride_minutes=1, max_seq_length=50,
        prediction_horizon_minutes=10,
    )

    found_positive = False
    for i in range(len(dataset)):
        sample = dataset[i]
        if sample["is_failure"]["payment-service"] > 0.5:
            found_positive = True
            break
    assert found_positive


def test_dataset_loads_from_directory_containing_manifest_alongside_parquet(tmp_path):
    """
    Regression test: module1_log_generator writes *.parquet parts AND
    failure_manifest.csv into the SAME output directory (its documented,
    natural layout — and what this module's own README's How-To-Run
    tells users to pass as --data). Pointing pd.read_parquet() straight
    at that directory used to make pyarrow try to parse every file in
    it as Parquet, including the manifest CSV, and crash with
    "Parquet magic bytes not found". LogSequenceDataset must glob for
    *.parquet explicitly instead of reading the whole directory blindly.
    """
    logs_path, manifest_path = _write_synthetic_data(tmp_path)
    assert logs_path.parent == manifest_path.parent == tmp_path

    dataset = LogSequenceDataset(
        logs_path=tmp_path,  # the directory, not the specific .parquet file
        manifest_path=manifest_path, service_list=SERVICES,
        error_code_list=ERROR_CODES, window_size_minutes=2, max_seq_length=50,
    )
    assert len(dataset) > 0
    assert len(dataset.logs_df) > 0


def test_dataset_produces_expected_raw_field_shapes(tmp_path):
    logs_path, manifest_path = _write_synthetic_data(tmp_path)
    dataset = LogSequenceDataset(
        logs_path=logs_path, manifest_path=manifest_path, service_list=SERVICES,
        error_code_list=ERROR_CODES, window_size_minutes=2, stride_minutes=1, max_seq_length=50,
    )
    assert len(dataset) > 0

    sample = dataset[0]
    assert sample["service_ids"].shape == (50,)
    assert sample["log_level_onehot"].shape == (50, cfg.LOG_LEVEL_DIM)
    assert sample["attention_mask"].sum() > 0
    assert set(sample["is_failure"].keys()) == set(SERVICES)


def test_windows_respect_label_visibility_cutoff(tmp_path):
    """
    Regression test: a window's label looks forward through
    prediction_horizon past window_end (_get_labels). Any window whose
    horizon would run past the last observed timestamp can't be honestly
    labeled — we have no data past that point to say whether a failure
    happens there, yet _get_labels' default (is_failure=0) can't tell
    "confirmed no failure" apart from "we simply stopped recording", so
    those trailing windows used to get force-labeled negative. Because
    ChronologicalSplitter's test split is always the chronologically last
    N% of windows, that right-censored tail always landed in val/test,
    guaranteeing near-zero true positives there regardless of model
    quality (found on the real microservices23 dataset: last 5/40 windows
    had zero positive labels while the dataset overall was ~37% positive).
    _create_windows must never generate a window past that cutoff.
    """
    logs_path, manifest_path = _write_synthetic_data(tmp_path)
    dataset = LogSequenceDataset(
        logs_path=logs_path, manifest_path=manifest_path, service_list=SERVICES,
        error_code_list=ERROR_CODES, window_size_minutes=2, stride_minutes=1, max_seq_length=50,
        prediction_horizon_minutes=5,
    )
    assert len(dataset) > 0
    max_time = dataset.logs_df["timestamp"].max()
    for _, window_end in dataset.windows:
        assert window_end + dataset.prediction_horizon <= max_time


def test_dataloader_survives_multiprocess_workers(tmp_path):
    """
    Regression test: create_dataloader() used to build its collate_fn as a
    local closure. With num_workers > 0 (the default), DataLoader pickles
    collate_fn to send to worker processes — on Windows (spawn-based
    multiprocessing, unlike fork on Linux) a closure can't be pickled at
    all, so this crashed with PicklingError on the very platform this
    project runs on, the instant a real (non-zero-worker) DataLoader was
    ever iterated. num_workers=2 here to actually exercise process spawn,
    not just construction.
    """
    logs_path, manifest_path = _write_synthetic_data(tmp_path)
    dataset = LogSequenceDataset(
        logs_path=logs_path, manifest_path=manifest_path, service_list=SERVICES,
        error_code_list=ERROR_CODES, window_size_minutes=2, stride_minutes=1, max_seq_length=50,
    )
    from module3_deep_learning.training.augmentation import LogAugmentor

    loader = create_dataloader(
        dataset, batch_size=2, service_list=SERVICES, augmentor=LogAugmentor(),
        shuffle=False, num_workers=2,
    )
    batch = next(iter(loader))
    assert batch["service_ids"].shape[0] == 2


def test_create_dataloader_defaults_to_sequential_order(tmp_path):
    """
    This project is sequential-only end to end by explicit design (no
    random splits, no random sampling — see LogSequenceDataset/
    ChronologicalSplitter) because logs are inter-dependent across time
    (cascading failures, precursor drift, service-to-service propagation).
    train.py's train_loader used to be the one exception, shuffling batch
    composition each epoch (ordinary SGD practice, but inconsistent with
    every other stage of this pipeline). Both the call site and
    create_dataloader's own default now default to sequential batch
    order; this checks two full passes over the same DataLoader yield
    windows in the identical order — not just "unshuffled once by luck."
    """
    logs_path, manifest_path = _write_synthetic_data(tmp_path)
    dataset = LogSequenceDataset(
        logs_path=logs_path, manifest_path=manifest_path, service_list=SERVICES,
        error_code_list=ERROR_CODES, window_size_minutes=2, stride_minutes=1, max_seq_length=50,
    )
    loader = create_dataloader(dataset, batch_size=2, service_list=SERVICES, num_workers=0)

    def _service_id_sequence():
        return [batch["service_ids"].clone() for batch in loader]

    epoch_1 = _service_id_sequence()
    epoch_2 = _service_id_sequence()
    assert len(epoch_1) == len(epoch_2) > 1
    for b1, b2 in zip(epoch_1, epoch_2):
        assert torch.equal(b1, b2)


def test_service_index_matches_edge_index_convention(tmp_path):
    """service_to_idx must be 0-based and match build_edge_index's convention
    (this was a real bug: the dataset used to reserve index 0 for "unknown"
    while the graph builder used 0-based indices directly)."""
    logs_path, manifest_path = _write_synthetic_data(tmp_path)
    dataset = LogSequenceDataset(
        logs_path=logs_path, manifest_path=manifest_path, service_list=SERVICES,
        error_code_list=ERROR_CODES, window_size_minutes=2, max_seq_length=50,
    )
    edge_index = build_edge_index({"api-gateway": ["payment-service"], "payment-service": ["database"]}, SERVICES)

    assert dataset.service_to_idx == {s: i for i, s in enumerate(SERVICES)}
    assert edge_index.max().item() < len(SERVICES)


def test_failure_type_labels_use_canonical_taxonomy(tmp_path):
    logs_path, manifest_path = _write_synthetic_data(tmp_path)
    dataset = LogSequenceDataset(
        logs_path=logs_path, manifest_path=manifest_path, service_list=SERVICES,
        error_code_list=ERROR_CODES, window_size_minutes=2, stride_minutes=1, max_seq_length=50,
        prediction_horizon_minutes=10,
    )
    assert dataset.failure_type_to_idx == {ft: i for i, ft in enumerate(cfg.FAILURE_TYPES)}

    # Find a window whose horizon covers the scheduled LATENCY_SPIKE at t=200s.
    found_positive = False
    for i in range(len(dataset)):
        sample = dataset[i]
        if sample["is_failure"]["payment-service"] > 0.5:
            found_positive = True
            assert sample["failure_type"]["payment-service"] == cfg.FAILURE_TYPES.index("LATENCY_SPIKE")
            break
    assert found_positive, "expected at least one window to have a positive label from the manifest"


def _write_class_imbalance_manifest(tmp_path):
    """Two LATENCY_SPIKE failures (t=400s, t=430s — close together so their
    horizons overlap, inflating how many windows see LATENCY_SPIKE) and one
    TIMEOUT failure (t=850s, isolated). MEMORY_LEAK never appears at all."""
    start = datetime(2026, 1, 1, 0, 0, 0)
    manifest_df = pd.DataFrame([
        {"failure_id": "F0001", "start_timestamp": start + timedelta(seconds=400),
         "end_timestamp": start + timedelta(seconds=430), "affected_service": "payment-service",
         "failure_type": "LATENCY_SPIKE", "root_cause_service": "payment-service", "severity": "MEDIUM",
         "phase_transitions": "[]", "resolution_time": (start + timedelta(seconds=430)).isoformat()},
        {"failure_id": "F0002", "start_timestamp": start + timedelta(seconds=430),
         "end_timestamp": start + timedelta(seconds=460), "affected_service": "payment-service",
         "failure_type": "LATENCY_SPIKE", "root_cause_service": "payment-service", "severity": "MEDIUM",
         "phase_transitions": "[]", "resolution_time": (start + timedelta(seconds=460)).isoformat()},
        {"failure_id": "F0003", "start_timestamp": start + timedelta(seconds=850),
         "end_timestamp": start + timedelta(seconds=880), "affected_service": "payment-service",
         "failure_type": "TIMEOUT", "root_cause_service": "payment-service", "severity": "MEDIUM",
         "phase_transitions": "[]", "resolution_time": (start + timedelta(seconds=880)).isoformat()},
    ])
    manifest_path = tmp_path / "class_imbalance_manifest.csv"
    manifest_df.to_csv(manifest_path, index=False)
    return manifest_path


def test_get_class_weights_returns_negative_over_positive_ratio(tmp_path):
    """Regression test for MultiTaskLoss's pos_weight never being wired up:
    get_class_weights() must return neg_count/pos_count (the ratio
    nn.BCEWithLogitsLoss's pos_weight actually expects), not the "balanced"
    per-class weight pair a naive inverse-frequency implementation would
    return — passing the wrong shape/scale here silently mis-weights the
    loss instead of erroring."""
    logs_path, manifest_path = _write_synthetic_data(tmp_path)
    class_imbalance_manifest = _write_class_imbalance_manifest(tmp_path)
    dataset = LogSequenceDataset(
        logs_path=logs_path, manifest_path=class_imbalance_manifest, service_list=SERVICES,
        error_code_list=ERROR_CODES, window_size_minutes=2, stride_minutes=1, max_seq_length=50,
    )
    pos_weight = dataset.get_class_weights()
    assert isinstance(pos_weight, float)
    assert pos_weight > 1.0, "failures are the minority class here, so neg/pos must be > 1"

    # Manually recompute neg/pos from the same window/label logic to cross-check the formula.
    total_pos = total_neg = 0
    for window_start, window_end in dataset.windows:
        labels = dataset._get_labels(window_end)
        for service in SERVICES:
            if labels["is_failure"][service] > 0.5:
                total_pos += 1
            else:
                total_neg += 1
    assert pos_weight == pytest.approx(total_neg / total_pos)


def test_get_class_weights_respects_indices_subset(tmp_path):
    """get_class_weights/get_failure_type_class_weights must be computable
    over an arbitrary index subset (train.py passes the train split's
    indices specifically to avoid leaking val/test class balance into a
    weight that only describes the training distribution) — restricting to
    a window range with a different positive rate must change the result."""
    logs_path, manifest_path = _write_synthetic_data(tmp_path)
    class_imbalance_manifest = _write_class_imbalance_manifest(tmp_path)
    dataset = LogSequenceDataset(
        logs_path=logs_path, manifest_path=class_imbalance_manifest, service_list=SERVICES,
        error_code_list=ERROR_CODES, window_size_minutes=2, stride_minutes=1, max_seq_length=50,
    )
    all_indices = list(range(len(dataset)))
    half = len(all_indices) // 2
    first_half_weight = dataset.get_class_weights(indices=all_indices[:half])
    second_half_weight = dataset.get_class_weights(indices=all_indices[half:])
    full_weight = dataset.get_class_weights()
    assert first_half_weight != second_half_weight or first_half_weight != full_weight, (
        "restricting to a subset with a different failure distribution should change the weight"
    )


def test_get_failure_type_class_weights_favors_rare_types(tmp_path):
    """The rarer TIMEOUT (1 event, isolated) must get a higher weight than
    the more common LATENCY_SPIKE (2 overlapping events). A type that never
    appears in `indices` (MEMORY_LEAK) must still get a nonzero weight, not
    0 — 0 caused nn.CrossEntropyLoss's weighted-mean reduction to divide by
    zero (NaN) the moment that class showed up as the only valid target in
    a *different* index range this same weight tensor gets reused to score
    (train.py reuses train-split weights to score val/test batches too, and
    a type absent from train can still legitimately appear later). Regression
    test for exactly that: a real local training dry run logged "Val Loss:
    nan" every epoch until this weight was floored above zero."""
    logs_path, manifest_path = _write_synthetic_data(tmp_path)
    class_imbalance_manifest = _write_class_imbalance_manifest(tmp_path)
    dataset = LogSequenceDataset(
        logs_path=logs_path, manifest_path=class_imbalance_manifest, service_list=SERVICES,
        error_code_list=ERROR_CODES, window_size_minutes=2, stride_minutes=1, max_seq_length=50,
    )
    weights = dataset.get_failure_type_class_weights()
    assert weights.shape == (len(cfg.FAILURE_TYPES),)

    latency_spike_idx = cfg.FAILURE_TYPES.index("LATENCY_SPIKE")
    timeout_idx = cfg.FAILURE_TYPES.index("TIMEOUT")
    memory_leak_idx = cfg.FAILURE_TYPES.index("MEMORY_LEAK")

    assert weights[memory_leak_idx].item() > 0.0
    # Unseen (MEMORY_LEAK) is clamped to count=1 — the smallest possible
    # denominator — so nothing observed can outweigh it.
    assert weights[memory_leak_idx].item() >= weights[timeout_idx].item()
    assert weights[timeout_idx].item() > weights[latency_spike_idx].item() > 0.0


def test_full_pipeline_dataset_to_model_forward(tmp_path):
    logs_path, manifest_path = _write_synthetic_data(tmp_path)
    dataset = LogSequenceDataset(
        logs_path=logs_path, manifest_path=manifest_path, service_list=SERVICES,
        error_code_list=ERROR_CODES, window_size_minutes=2, stride_minutes=1, max_seq_length=50,
    )
    loader = create_dataloader(dataset, batch_size=4, service_list=SERVICES, shuffle=False, num_workers=0)
    edge_index = build_edge_index({"api-gateway": ["payment-service"], "payment-service": ["database"]}, SERVICES)
    batch_builder = LogBatchBuilder(edge_index=edge_index, num_services=len(SERVICES), device=torch.device("cpu"))

    model = LogSentinelModel(ModelConfig(num_services=len(SERVICES), num_error_codes=len(ERROR_CODES)))

    collated = next(iter(loader))
    log_batch = batch_builder.build(collated)
    labels = batch_builder.extract_labels(collated)

    out = model(log_batch)

    assert out.failure_probability.shape == labels["is_failure"].shape
    assert out.failure_type_logits.shape[:2] == labels["failure_type"].shape
