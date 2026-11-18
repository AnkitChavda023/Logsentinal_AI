"""
Tests for attention/log_attention_pooling.py (LogRobust §3.3 Eq. 2-3,
adapted to per-service pooling — see the module docstring).
"""

import torch

from module3_deep_learning.attention import AttentionPooling


def _zero_out_scores(pooling: AttentionPooling) -> None:
    """Forces every raw attention score to 0 regardless of input, so the
    resulting softmax is uniform over each service's real positions —
    i.e. this makes AttentionPooling behave exactly like a plain masked
    mean, the strict-generalization property its docstring claims."""
    with torch.no_grad():
        for layer in pooling.score:
            if isinstance(layer, torch.nn.Linear):
                layer.weight.zero_()
                layer.bias.zero_()


def test_output_shapes():
    pooling = AttentionPooling(hidden_dim=4)
    x = torch.randn(2, 6, 4)
    service_ids = torch.randint(0, 3, (2, 6))
    mask = torch.ones(2, 6, dtype=torch.bool)

    pooled, weights = pooling(x, service_ids, num_services=3, mask=mask)

    assert pooled.shape == (2, 3, 4)
    assert weights.shape == (2, 3, 6)
    assert torch.isfinite(pooled).all()
    assert torch.isfinite(weights).all()


def test_uniform_weights_reduce_to_mean_pool():
    """With scores forced to 0 (see _zero_out_scores), this must reproduce
    the old _pool_by_service's plain masked mean exactly — proving
    attention pooling is a strict generalization, not a behavior change,
    when the model hasn't yet learned any non-uniform weighting."""
    pooling = AttentionPooling(hidden_dim=4)
    _zero_out_scores(pooling)

    x = torch.arange(5 * 4, dtype=torch.float32).reshape(1, 5, 4)
    service_ids = torch.tensor([[0, 0, 1, 1, 2]])
    mask = torch.ones(1, 5, dtype=torch.bool)

    pooled, weights = pooling(x, service_ids, num_services=3, mask=mask)

    assert torch.allclose(pooled[0, 0], x[0, [0, 1]].mean(dim=0), atol=1e-6)
    assert torch.allclose(pooled[0, 1], x[0, [2, 3]].mean(dim=0), atol=1e-6)
    assert torch.allclose(pooled[0, 2], x[0, 4], atol=1e-6)
    # Uniform over each service's own positions: 0.5/0.5 for the 2-line
    # services, 1.0 for the 1-line service.
    assert torch.allclose(weights[0, 0, [0, 1]], torch.tensor([0.5, 0.5]), atol=1e-6)
    assert torch.allclose(weights[0, 2, 4], torch.tensor(1.0), atol=1e-6)


def test_masked_positions_are_excluded():
    pooling = AttentionPooling(hidden_dim=4)
    _zero_out_scores(pooling)

    x = torch.arange(6 * 4, dtype=torch.float32).reshape(1, 6, 4)
    service_ids = torch.tensor([[0, 0, 0, 1, 1, 1]])
    # Position 2 (service 0) is padding — must not contribute to pooling.
    mask = torch.tensor([[True, True, False, True, True, True]])

    pooled, weights = pooling(x, service_ids, num_services=2, mask=mask)

    assert weights[0, 0, 2] == 0.0
    assert torch.allclose(pooled[0, 0], x[0, [0, 1]].mean(dim=0), atol=1e-6)


def test_service_absent_from_window_gets_zero_vector_not_nan():
    """A service with zero real log lines in this window (softmax over an
    all -inf row) must come back as a zero vector, never NaN."""
    pooling = AttentionPooling(hidden_dim=4)
    _zero_out_scores(pooling)

    x = torch.randn(1, 4, 4)
    service_ids = torch.tensor([[0, 0, 1, 1]])   # service 2 never appears
    mask = torch.ones(1, 4, dtype=torch.bool)

    pooled, weights = pooling(x, service_ids, num_services=3, mask=mask)

    assert torch.isfinite(pooled).all()
    assert torch.isfinite(weights).all()
    assert torch.allclose(pooled[0, 2], torch.zeros(4))
    assert torch.allclose(weights[0, 2], torch.zeros(4))


def test_gradients_flow_through_pooling():
    pooling = AttentionPooling(hidden_dim=4)
    x = torch.randn(2, 5, 4, requires_grad=True)
    service_ids = torch.randint(0, 2, (2, 5))
    mask = torch.ones(2, 5, dtype=torch.bool)

    pooled, _ = pooling(x, service_ids, num_services=2, mask=mask)
    pooled.sum().backward()

    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
    assert any(p.grad is not None and torch.isfinite(p.grad).all() for p in pooling.parameters())
