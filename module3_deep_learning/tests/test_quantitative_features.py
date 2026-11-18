"""
Tests for quantitative/count_features.py (LogAnomaly §3.3's quantitative
count-vector branch, adapted to per-service statistics — see the module
docstring). Uses a hand-built batch with known exact values rather than
random data, so every expected number can be computed by hand and checked
exactly — the strongest kind of test for a purely deterministic function.
"""

import torch

from module3_deep_learning.config import defaults as cfg
from module3_deep_learning.quantitative import QUANT_FEATURE_DIM, compute_quantitative_features


def _one_hot(idx: int) -> list:
    row = [0.0] * 5
    row[idx] = 1.0
    return row


def test_known_values_for_two_services():
    # service 0 = positions 0,1 ; service 1 = positions 2,3
    service_ids = torch.tensor([[0, 0, 1, 1]])
    log_level_onehot = torch.tensor([[
        _one_hot(1),  # INFO
        _one_hot(3),  # ERROR
        _one_hot(0),  # DEBUG
        _one_hot(2),  # WARN
    ]])
    error_code_ids = torch.tensor([[0, 5, 0, 0]])
    latency_ms = torch.tensor([[100.0, 300.0, 50.0, 150.0]])
    cpu_usage = torch.tensor([[0.1, 0.5, 0.2, 0.3]])
    memory_mb = torch.tensor([[1000.0, 2000.0, 500.0, 1500.0]])
    is_anomaly = torch.tensor([[0.0, 1.0, 0.0, 0.0]])
    attention_mask = torch.ones(1, 4)

    features = compute_quantitative_features(
        service_ids=service_ids, log_level_onehot=log_level_onehot,
        error_code_ids=error_code_ids, latency_ms=latency_ms, cpu_usage=cpu_usage,
        memory_mb=memory_mb, is_anomaly=is_anomaly, attention_mask=attention_mask,
        num_services=2,
    )

    assert features.shape == (1, 2, QUANT_FEATURE_DIM)

    expected_service_0 = torch.tensor([
        0.5,                              # volume: 2/4
        0.0, 0.5, 0.0, 0.5, 0.0,          # level fracs: 1x INFO, 1x ERROR out of 2
        0.5,                              # error_rate: 1/2 lines have a nonzero error code
        0.5,                              # anomaly_rate: 1/2 lines anomalous
        200.0 / cfg.MAX_LATENCY_MS,       # mean latency
        300.0 / cfg.MAX_LATENCY_MS,       # max latency
        0.3,                               # mean cpu
        0.5,                               # max cpu
        1500.0 / cfg.MAX_MEMORY_MB,       # mean memory
        2000.0 / cfg.MAX_MEMORY_MB,       # max memory
    ])
    expected_service_1 = torch.tensor([
        0.5,
        0.5, 0.0, 0.5, 0.0, 0.0,
        0.0,
        0.0,
        100.0 / cfg.MAX_LATENCY_MS,
        150.0 / cfg.MAX_LATENCY_MS,
        0.25,
        0.3,
        1000.0 / cfg.MAX_MEMORY_MB,
        1500.0 / cfg.MAX_MEMORY_MB,
    ])

    assert torch.allclose(features[0, 0], expected_service_0, atol=1e-6)
    assert torch.allclose(features[0, 1], expected_service_1, atol=1e-6)


def test_service_absent_from_window_is_all_zero_not_nan():
    service_ids = torch.tensor([[0, 0]])
    log_level_onehot = torch.tensor([[_one_hot(1), _one_hot(1)]])
    error_code_ids = torch.zeros(1, 2, dtype=torch.long)
    latency_ms = torch.full((1, 2), 100.0)
    cpu_usage = torch.full((1, 2), 0.2)
    memory_mb = torch.full((1, 2), 500.0)
    is_anomaly = torch.zeros(1, 2)
    attention_mask = torch.ones(1, 2)

    features = compute_quantitative_features(
        service_ids=service_ids, log_level_onehot=log_level_onehot,
        error_code_ids=error_code_ids, latency_ms=latency_ms, cpu_usage=cpu_usage,
        memory_mb=memory_mb, is_anomaly=is_anomaly, attention_mask=attention_mask,
        num_services=3,   # service 2 never appears
    )

    assert torch.isfinite(features).all()
    assert torch.allclose(features[0, 2], torch.zeros(QUANT_FEATURE_DIM))


def test_padding_positions_are_excluded():
    service_ids = torch.tensor([[0, 0, 0]])
    log_level_onehot = torch.tensor([[_one_hot(1), _one_hot(1), _one_hot(4)]])
    error_code_ids = torch.zeros(1, 3, dtype=torch.long)
    latency_ms = torch.tensor([[100.0, 100.0, 99999.0]])  # last is padding, must be ignored
    cpu_usage = torch.tensor([[0.2, 0.2, 1.0]])
    memory_mb = torch.tensor([[500.0, 500.0, 99999.0]])
    is_anomaly = torch.zeros(1, 3)
    attention_mask = torch.tensor([[1.0, 1.0, 0.0]])  # position 2 is padding

    features = compute_quantitative_features(
        service_ids=service_ids, log_level_onehot=log_level_onehot,
        error_code_ids=error_code_ids, latency_ms=latency_ms, cpu_usage=cpu_usage,
        memory_mb=memory_mb, is_anomaly=is_anomaly, attention_mask=attention_mask,
        num_services=1,
    )

    # volume should count only the 2 real positions, not the padding one.
    assert torch.isclose(features[0, 0, 0], torch.tensor(2.0 / 3.0), atol=1e-6)
    assert torch.isclose(features[0, 0, 8], torch.tensor(100.0 / cfg.MAX_LATENCY_MS), atol=1e-6)
