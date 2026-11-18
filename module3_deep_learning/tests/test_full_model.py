"""
Tests for models/full_model.py — the composed LSTM+GRU+BiRNN+Transformer+GNN
model. These specifically guard against the bug this module was rewritten
to fix: per-service pooling and batched-graph GNN propagation must keep
every sample in a batch isolated from every other sample, and must produce
exactly one prediction per (window, service) pair, not per window.
"""

import torch

from module3_deep_learning.config import defaults as cfg
from module3_deep_learning.models.full_model import LogBatch, LogSentinelModel, ModelConfig, model_config_from_yaml

SERVICES = ["api-gateway", "auth-service", "payment-service", "database", "cache"]
NUM_SERVICES = len(SERVICES)
NUM_ERROR_CODES = 4


def _make_batch(batch_size: int, seq_len: int, edge_index: torch.Tensor) -> LogBatch:
    torch.manual_seed(0)
    return LogBatch(
        service_ids=torch.randint(0, NUM_SERVICES, (batch_size, seq_len)),
        log_level_onehot=torch.nn.functional.one_hot(torch.randint(0, 5, (batch_size, seq_len)), 5).float(),
        error_code_ids=torch.randint(0, NUM_ERROR_CODES + 1, (batch_size, seq_len)),
        latency_ms=torch.rand(batch_size, seq_len) * 500,
        cpu_usage=torch.rand(batch_size, seq_len),
        memory_mb=torch.rand(batch_size, seq_len) * 4096,
        time_since_error=torch.rand(batch_size, seq_len) * 3600,
        hour_of_day=torch.rand(batch_size, seq_len) * 24,
        is_anomaly=(torch.rand(batch_size, seq_len) > 0.9).float(),
        trace_depth=torch.randint(1, 5, (batch_size, seq_len)).float(),
        attention_mask=torch.ones(batch_size, seq_len),
        edge_index=edge_index,
        num_services=NUM_SERVICES,
    )


def _edge_index() -> torch.Tensor:
    edges = [[0, 1], [0, 2], [2, 3], [1, 3], [0, 4]]
    return torch.tensor(edges, dtype=torch.long).t().contiguous()


def test_output_shapes_are_per_service_not_per_window():
    config = ModelConfig(num_services=NUM_SERVICES, num_error_codes=NUM_ERROR_CODES)
    model = LogSentinelModel(config)
    batch = _make_batch(batch_size=4, seq_len=30, edge_index=_edge_index())

    out = model(batch)

    assert out.failure_probability.shape == (4, NUM_SERVICES)
    assert out.failure_type_logits.shape == (4, NUM_SERVICES, cfg.NUM_FAILURE_TYPES)
    assert out.time_to_failure.shape == (4, NUM_SERVICES)
    assert torch.isfinite(out.failure_probability).all()
    assert (out.failure_probability >= 0).all() and (out.failure_probability <= 1).all()


def test_gcn_ablation_path_produces_same_shapes():
    config = ModelConfig(num_services=NUM_SERVICES, num_error_codes=NUM_ERROR_CODES, use_gat=False)
    model = LogSentinelModel(config)
    batch = _make_batch(batch_size=2, seq_len=20, edge_index=_edge_index())

    out = model(batch)

    assert out.failure_probability.shape == (2, NUM_SERVICES)
    assert out.failure_type_logits.shape == (2, NUM_SERVICES, cfg.NUM_FAILURE_TYPES)


def test_batched_gnn_does_not_leak_across_samples():
    """
    The batched edge_index trick offsets node indices per sample so one
    static graph can serve a whole batch. If the offset arithmetic were
    wrong, one sample's node features could attend to another sample's
    nodes. Changing sample 0's input must not change sample 1's output.
    """
    config = ModelConfig(num_services=NUM_SERVICES, num_error_codes=NUM_ERROR_CODES)
    model = LogSentinelModel(config)
    model.eval()

    edge_index = _edge_index()
    batch_a = _make_batch(batch_size=3, seq_len=25, edge_index=edge_index)
    batch_b = _make_batch(batch_size=3, seq_len=25, edge_index=edge_index)
    # Make sample 0 differ between the two batches, keep samples 1 and 2 identical.
    for field in (
        "service_ids", "log_level_onehot", "error_code_ids", "latency_ms", "cpu_usage",
        "memory_mb", "time_since_error", "hour_of_day", "is_anomaly", "trace_depth", "attention_mask",
    ):
        getattr(batch_b, field)[1:] = getattr(batch_a, field)[1:]

    with torch.no_grad():
        out_a = model(batch_a)
        out_b = model(batch_b)

    assert torch.allclose(out_a.failure_probability[1:], out_b.failure_probability[1:], atol=1e-5)
    assert torch.allclose(out_a.time_to_failure[1:], out_b.time_to_failure[1:], atol=1e-4)


def test_predict_with_uncertainty_returns_std_and_ci():
    config = ModelConfig(num_services=NUM_SERVICES, num_error_codes=NUM_ERROR_CODES)
    model = LogSentinelModel(config)
    batch = _make_batch(batch_size=2, seq_len=15, edge_index=_edge_index())

    out = model.predict_with_uncertainty(batch, num_samples=5)

    assert out.failure_prob_std is not None
    assert out.failure_prob_std.shape == (2, NUM_SERVICES)
    assert (out.failure_prob_std >= 0).all()
    assert out.failure_prob_ci["lower"].shape == (2, NUM_SERVICES)
    assert (out.failure_prob_ci["lower"] <= out.failure_prob_ci["upper"]).all()


def test_single_sequence_encoder_builds_only_that_one():
    """sequence_encoder='lstm' etc. must not construct the other two —
    they should stay None, not just go unused."""
    config = ModelConfig(num_services=NUM_SERVICES, num_error_codes=NUM_ERROR_CODES, sequence_encoder="lstm")
    model = LogSentinelModel(config)

    assert model.lstm is not None
    assert model.gru is None
    assert model.birnn is None

    batch = _make_batch(batch_size=2, seq_len=15, edge_index=_edge_index())
    out = model(batch)
    assert out.failure_probability.shape == (2, NUM_SERVICES)

    params = model.get_num_parameters()
    assert "lstm" in params and "gru" not in params and "birnn" not in params


def test_each_single_sequence_encoder_runs_and_produces_finite_output():
    for choice in ("lstm", "gru", "birnn"):
        config = ModelConfig(num_services=NUM_SERVICES, num_error_codes=NUM_ERROR_CODES, sequence_encoder=choice)
        model = LogSentinelModel(config)
        batch = _make_batch(batch_size=2, seq_len=12, edge_index=_edge_index())

        out = model(batch)

        assert out.failure_probability.shape == (2, NUM_SERVICES)
        assert torch.isfinite(out.failure_probability).all(), f"non-finite output for sequence_encoder={choice!r}"


def test_invalid_sequence_encoder_raises():
    config = ModelConfig(num_services=NUM_SERVICES, num_error_codes=NUM_ERROR_CODES, sequence_encoder="not-a-real-choice")
    try:
        LogSentinelModel(config)
        assert False, "expected ValueError for an invalid sequence_encoder"
    except ValueError:
        pass


def test_model_config_from_yaml_wires_every_field():
    """
    Regression test: train.py/evaluate.py/predictor_service.py each used to
    build ModelConfig by hand, reading only 6 of its ~20 fields from YAML
    (embed_dim, seq_hidden_dim, sequence_encoder, transformer_num_layers,
    gnn_num_layers, use_gat). gnn_hidden_dim, transformer_num_heads,
    seq_num_layers, seq_dropout, transformer_dropout, gnn_dropout, and
    others were silently never read — a config trying to tune them had no
    effect, with no error to say so. This checks every overridable field
    round-trips through model_config_from_yaml, not just the original six.
    """
    overrides = {
        "service_embed_dim": 4, "error_embed_dim": 4, "embed_dim": 16,
        "seq_hidden_dim": 32, "seq_num_layers": 1, "seq_dropout": 0.11,
        "sequence_encoder": "gru",
        "transformer_num_layers": 1, "transformer_num_heads": 2,
        "transformer_ff_dim": 64, "transformer_dropout": 0.22,
        "gnn_hidden_dim": 32, "gnn_num_layers": 1, "gnn_dropout": 0.33,
        "gat_dropout": 0.44, "gat_num_heads": 2, "use_gat": False,
        "mc_dropout_samples": 7,
        "use_attention_pooling": False, "use_quantitative_features": False,
        "use_mlkp_auxiliary": False, "mlkp_mask_ratio": 0.3,
    }
    config = model_config_from_yaml(
        model_cfg_yaml=overrides, num_services=NUM_SERVICES,
        num_error_codes=NUM_ERROR_CODES, max_seq_length=42,
    )
    for key, value in overrides.items():
        assert getattr(config, key) == value, f"{key} did not round-trip through model_config_from_yaml"
    assert config.max_seq_length == 42
    assert config.num_services == NUM_SERVICES
    assert config.num_error_codes == NUM_ERROR_CODES

    # And it must actually build without error with these overrides applied.
    model = LogSentinelModel(config)
    assert model.config.gnn_hidden_dim == 32


def test_model_config_from_yaml_defaults_match_config_defaults():
    """An empty model_cfg_yaml must fall back to config/defaults.py exactly
    (same values ModelConfig's own dataclass defaults use) — no silent
    drift between the two default sources."""
    config = model_config_from_yaml(
        model_cfg_yaml={}, num_services=NUM_SERVICES,
        num_error_codes=NUM_ERROR_CODES, max_seq_length=cfg.MAX_SEQ_LENGTH,
    )
    default_config = ModelConfig(num_services=NUM_SERVICES, num_error_codes=NUM_ERROR_CODES)
    assert config == default_config


def test_non_default_seq_hidden_dim_does_not_crash():
    """
    Regression test: TransformerEncoder's output projection used to be
    hardcoded to nn.Linear(d_model, 256), coupled to
    config/defaults.py's SEQ_HIDDEN_DIM only by coincidence (both
    happened to be 256). full_model.py concatenates the transformer
    branch's output with the LSTM/GRU/BiRNN branch's (seq_hidden_dim-wide)
    output and feeds that into a Linear(seq_hidden_dim*2, seq_hidden_dim)
    — so any seq_hidden_dim other than 256 raised a matrix-shape error at
    the very first forward pass. This exercises a config that would have
    hit that exactly (a smaller model — the actual motivation for this
    fix: shrinking model capacity to match a small dataset).
    """
    config = ModelConfig(
        num_services=NUM_SERVICES, num_error_codes=NUM_ERROR_CODES,
        embed_dim=32, seq_hidden_dim=64, transformer_num_heads=4,
        transformer_num_layers=1, gnn_hidden_dim=64, gnn_num_layers=1,
    )
    model = LogSentinelModel(config)
    edge_index = _edge_index()
    batch = _make_batch(batch_size=2, seq_len=10, edge_index=edge_index)
    out = model(batch)
    assert out.failure_probability.shape == (2, NUM_SERVICES)


def test_forward_handles_empty_edge_index():
    """A single-service topology (no dependencies) must not crash the GNN step."""
    config = ModelConfig(num_services=NUM_SERVICES, num_error_codes=NUM_ERROR_CODES)
    model = LogSentinelModel(config)
    batch = _make_batch(batch_size=2, seq_len=10, edge_index=torch.zeros(2, 0, dtype=torch.long))

    out = model(batch)

    assert out.failure_probability.shape == (2, NUM_SERVICES)


# --- Paper-grounded upgrades: LogRobust attention pooling, LogAnomaly
# quantitative branch, LogBERT masked log-key auxiliary task (all default
# on — see config/defaults.py) ---------------------------------------------


def test_all_paper_upgrades_enabled_train_forward_and_backward():
    """All three upgrades are on by default (ModelConfig() with no
    overrides). This is the real, end-to-end path: finite total loss with
    a finite (and, given seq_len=30 and the default 15% mask ratio,
    virtually certain to be populated) MLKP term, and gradients reaching
    every new module."""
    torch.manual_seed(0)
    config = ModelConfig(num_services=NUM_SERVICES, num_error_codes=NUM_ERROR_CODES)
    model = LogSentinelModel(config)
    model.train()
    assert model.pooler is not None
    assert model.quant_proj is not None
    assert model.mlkp_head is not None

    batch = _make_batch(batch_size=4, seq_len=30, edge_index=_edge_index())
    out = model(batch)

    assert out.service_attention_weights is not None
    assert out.service_attention_weights.shape == (4, NUM_SERVICES, 30)
    assert out.mlkp_logits is not None, "expected >=1 masked position at seq_len=30, mask_ratio=0.15"
    assert out.mlkp_logits.shape[1] == NUM_SERVICES
    assert out.mlkp_targets.shape[0] == out.mlkp_logits.shape[0]
    assert torch.isfinite(out.mlkp_logits).all()

    loss = out.failure_probability.sum() + out.mlkp_logits.sum()
    loss.backward()
    assert model.pooler.score[0].weight.grad is not None
    assert model.mlkp_head.weight.grad is not None
    assert model.mask_embedding.grad is not None

    params = model.get_num_parameters()
    assert "attention_pooling" in params
    assert "quantitative_branch" in params
    assert "mlkp_auxiliary" in params


def test_mlkp_disabled_in_eval_mode():
    """The MLKP branch is train-time only — eval mode must never populate
    mlkp_logits/targets, both to save the extra Transformer pass at
    inference and so it can never leak into a served prediction."""
    config = ModelConfig(num_services=NUM_SERVICES, num_error_codes=NUM_ERROR_CODES)
    model = LogSentinelModel(config)
    model.eval()
    batch = _make_batch(batch_size=2, seq_len=20, edge_index=_edge_index())

    with torch.no_grad():
        out = model(batch)

    assert out.mlkp_logits is None
    assert out.mlkp_targets is None


def test_paper_upgrades_can_be_disabled_and_fall_back():
    """use_*=False must fully disable each branch (module is None) and fall
    back to the pre-upgrade behavior (_pool_by_service, no quant fusion, no
    MLKP pass) rather than partially wiring in."""
    config = ModelConfig(
        num_services=NUM_SERVICES, num_error_codes=NUM_ERROR_CODES,
        use_attention_pooling=False, use_quantitative_features=False, use_mlkp_auxiliary=False,
    )
    model = LogSentinelModel(config)
    assert model.pooler is None
    assert model.quant_proj is None and model.quant_fusion is None
    assert model.mask_embedding is None and model.mlkp_head is None

    batch = _make_batch(batch_size=2, seq_len=15, edge_index=_edge_index())
    model.train()
    out = model(batch)

    assert out.service_attention_weights is None
    assert out.mlkp_logits is None
    assert out.failure_probability.shape == (2, NUM_SERVICES)

    params = model.get_num_parameters()
    assert "attention_pooling" not in params
    assert "quantitative_branch" not in params
    assert "mlkp_auxiliary" not in params
