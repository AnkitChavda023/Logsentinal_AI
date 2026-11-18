"""
Full LogSentinel Deep Learning Model (Section 7.2-7.7).

Composes the already-implemented building blocks:
  embeddings.LogEmbedder        -> 37-dim log embedding      (§7.2)
  sequence_models.{LSTM,GRU,BiRNN,RNN,BiLSTM}Encoder -> per-log sequence context (§7.3)
  transformer.TransformerEncoder -> cross-service attention   (§7.4)
  gnn.{GATStack,GCNStack}        -> graph-propagated service state (§7.5)
  heads.{FailureClassifier,FailureTypeClassifier,TimeToFailureRegressor} (§7.6)
  uncertainty.mc_predict_all     -> MC-Dropout uncertainty     (§7.7)

This file used to hand-roll a second, independent copy of every one of
those components (its own LogEmbedder, its own attention math, ...)
instead of importing them. Two problems came from that: the duplicated
code never actually used the service/error-code embeddings it created
(dead `nn.Embedding` layers whose output was computed and then discarded),
and — the more serious bug — nothing in the forward pass ever pooled log
representations *per service*. A 5-minute window mixes log lines from
every service in the topology, but the old code treated the whole batch
of "one window" as if it were already "one row per service", so the GNN
step (which needs one node per service) received one node per *window*
instead. That's only dimensionally valid by accident when
batch_size == num_services; for any real topology + batch size it either
silently computed nonsense or raised an index error inside the graph
conv (edge_index references service nodes that don't exist in a
batch-of-windows-shaped tensor). Composing the existing modules and
pooling explicitly by service (`_pool_by_service` below) fixes this at
the root instead of patching each symptom separately.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from torch import Tensor

from module3_deep_learning.attention import AttentionPooling
from module3_deep_learning.config import defaults as cfg
from module3_deep_learning.embeddings import LogEmbedder
from module3_deep_learning.gnn import GATStack, GCNStack
from module3_deep_learning.heads import (
    FailureClassifier,
    FailureTypeClassifier,
    TimeToFailureRegressor,
)
from module3_deep_learning.quantitative import QUANT_FEATURE_DIM, compute_quantitative_features
from module3_deep_learning.sequence_models import (
    BiLSTMEncoder,
    BiRNNEncoder,
    GRUEncoder,
    LSTMEncoder,
    RNNEncoder,
)
from module3_deep_learning.transformer import TransformerEncoder
from module3_deep_learning.uncertainty import mc_predict_all

logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Configuration for the LogSentinel model. Every field has a default
    sourced from config/defaults.py — nothing here is a bare literal."""

    num_services: int = 10
    num_error_codes: int = 32

    service_embed_dim: int = cfg.SERVICE_EMBED_DIM
    error_embed_dim: int = cfg.ERROR_EMBED_DIM
    embed_dim: int = cfg.EMBED_DIM
    max_latency_ms: float = cfg.MAX_LATENCY_MS
    max_memory_mb: float = cfg.MAX_MEMORY_MB
    max_trace_depth: int = cfg.MAX_TRACE_DEPTH
    max_time_since_error_s: float = cfg.MAX_TIME_SINCE_ERROR_S

    seq_hidden_dim: int = cfg.SEQ_HIDDEN_DIM
    seq_num_layers: int = cfg.SEQ_NUM_LAYERS
    seq_dropout: float = cfg.SEQ_DROPOUT
    sequence_encoder: str = cfg.SEQUENCE_ENCODER_DEFAULT
    """One of cfg.VALID_SEQUENCE_ENCODERS. "all" = original 3-way ensemble
    (LSTM+GRU+BiRNN averaged); a single name builds and runs only that
    encoder, so it can be trained/compared as its own independent model."""

    transformer_num_layers: int = cfg.TRANSFORMER_NUM_LAYERS
    transformer_num_heads: int = cfg.TRANSFORMER_NUM_HEADS
    transformer_ff_dim: int = cfg.TRANSFORMER_FF_DIM
    transformer_dropout: float = cfg.TRANSFORMER_DROPOUT
    max_seq_length: int = cfg.MAX_SEQ_LENGTH

    gnn_hidden_dim: int = cfg.GNN_HIDDEN_DIM
    gnn_num_layers: int = cfg.GNN_NUM_LAYERS
    gnn_dropout: float = cfg.GNN_DROPOUT
    gat_dropout: float = cfg.GAT_DROPOUT
    gat_num_heads: int = cfg.GAT_NUM_HEADS
    use_gat: bool = cfg.USE_GAT_BY_DEFAULT

    num_failure_types: int = cfg.NUM_FAILURE_TYPES
    time_to_failure_max_seconds: float = cfg.TIME_TO_FAILURE_MAX_SECONDS

    mc_dropout_samples: int = cfg.MC_DROPOUT_SAMPLES

    # Paper-grounded upgrades (see config/defaults.py's docstring for the
    # exact papers/sections each one adapts, and models/full_model.py's
    # LogSentinelModel for the adaptation to this project's dependent
    # multi-service windows).
    use_attention_pooling: bool = cfg.USE_ATTENTION_POOLING_BY_DEFAULT
    use_quantitative_features: bool = cfg.USE_QUANTITATIVE_FEATURES_BY_DEFAULT
    use_mlkp_auxiliary: bool = cfg.USE_MLKP_AUXILIARY_BY_DEFAULT
    mlkp_mask_ratio: float = cfg.MLKP_MASK_RATIO


def model_config_from_yaml(
    model_cfg_yaml: Dict[str, Any],
    num_services: int,
    num_error_codes: int,
    max_seq_length: int,
) -> ModelConfig:
    """Builds a ModelConfig from a training config's `model:` section,
    falling back to config/defaults.py for anything unset.

    Centralizes what used to be duplicated, incomplete field-by-field
    construction across train.py/evaluate.py/predictor_service.py: each
    one only read embed_dim/seq_hidden_dim/sequence_encoder/
    transformer_num_layers/gnn_num_layers/use_gat, so gnn_hidden_dim,
    transformer_num_heads, seq_num_layers, and every dropout rate were
    silently never read from YAML at all — a config trying to tune them
    had no effect, and the three sites could each drift independently
    since none of them shared this logic.

    max_seq_length is a required, explicit argument rather than another
    model_cfg_yaml lookup: it's fundamentally a data-shape parameter (the
    positional encoding size must match what LogSequenceDataset actually
    produces per window), so it belongs to config["data"], not
    config["model"] — pass config["data"].get("max_seq_length", ...).
    """
    g = model_cfg_yaml.get
    return ModelConfig(
        num_services=num_services,
        num_error_codes=num_error_codes,
        service_embed_dim=g("service_embed_dim", cfg.SERVICE_EMBED_DIM),
        error_embed_dim=g("error_embed_dim", cfg.ERROR_EMBED_DIM),
        embed_dim=g("embed_dim", cfg.EMBED_DIM),
        max_latency_ms=g("max_latency_ms", cfg.MAX_LATENCY_MS),
        max_memory_mb=g("max_memory_mb", cfg.MAX_MEMORY_MB),
        max_trace_depth=g("max_trace_depth", cfg.MAX_TRACE_DEPTH),
        max_time_since_error_s=g("max_time_since_error_s", cfg.MAX_TIME_SINCE_ERROR_S),
        seq_hidden_dim=g("seq_hidden_dim", cfg.SEQ_HIDDEN_DIM),
        seq_num_layers=g("seq_num_layers", cfg.SEQ_NUM_LAYERS),
        seq_dropout=g("seq_dropout", cfg.SEQ_DROPOUT),
        sequence_encoder=g("sequence_encoder", cfg.SEQUENCE_ENCODER_DEFAULT),
        transformer_num_layers=g("transformer_num_layers", cfg.TRANSFORMER_NUM_LAYERS),
        transformer_num_heads=g("transformer_num_heads", cfg.TRANSFORMER_NUM_HEADS),
        transformer_ff_dim=g("transformer_ff_dim", cfg.TRANSFORMER_FF_DIM),
        transformer_dropout=g("transformer_dropout", cfg.TRANSFORMER_DROPOUT),
        max_seq_length=max_seq_length,
        gnn_hidden_dim=g("gnn_hidden_dim", cfg.GNN_HIDDEN_DIM),
        gnn_num_layers=g("gnn_num_layers", cfg.GNN_NUM_LAYERS),
        gnn_dropout=g("gnn_dropout", cfg.GNN_DROPOUT),
        gat_dropout=g("gat_dropout", cfg.GAT_DROPOUT),
        gat_num_heads=g("gat_num_heads", cfg.GAT_NUM_HEADS),
        use_gat=g("use_gat", cfg.USE_GAT_BY_DEFAULT),
        num_failure_types=g("num_failure_types", cfg.NUM_FAILURE_TYPES),
        time_to_failure_max_seconds=g("time_to_failure_max_seconds", cfg.TIME_TO_FAILURE_MAX_SECONDS),
        mc_dropout_samples=g("mc_dropout_samples", cfg.MC_DROPOUT_SAMPLES),
        use_attention_pooling=g("use_attention_pooling", cfg.USE_ATTENTION_POOLING_BY_DEFAULT),
        use_quantitative_features=g("use_quantitative_features", cfg.USE_QUANTITATIVE_FEATURES_BY_DEFAULT),
        use_mlkp_auxiliary=g("use_mlkp_auxiliary", cfg.USE_MLKP_AUXILIARY_BY_DEFAULT),
        mlkp_mask_ratio=g("mlkp_mask_ratio", cfg.MLKP_MASK_RATIO),
    )


@dataclass
class LogBatch:
    """
    One batch of log windows, in the raw (pre-embedding) shape that
    embeddings.LogEmbedder expects directly — see its docstring for the
    exact 10-field breakdown. Building the 37-dim vector is the
    embedder's job, not the data pipeline's, so nothing upstream ever
    has to know the internal embedding layout.

    service_ids use the SAME 0-based index as `edge_index` and as
    position in `service_list` everywhere else in Module 3 (dataset
    labels, graph nodes, predictor output ordering) — there is no
    separate "+1 for unknown" offset. Padding positions may safely
    carry service_id 0 (a real service) because attention_mask always
    excludes them from pooling.
    """

    service_ids: Tensor            # (B, T) long
    log_level_onehot: Tensor       # (B, T, 5) float
    error_code_ids: Tensor         # (B, T) long, 0 = no error
    latency_ms: Tensor             # (B, T) float
    cpu_usage: Tensor              # (B, T) float, already 0-1
    memory_mb: Tensor              # (B, T) float
    time_since_error: Tensor       # (B, T) float, seconds
    hour_of_day: Tensor            # (B, T) float, 0-24
    is_anomaly: Tensor             # (B, T) float/bool
    trace_depth: Tensor            # (B, T) float
    attention_mask: Tensor         # (B, T) 1 = real log line, 0 = padding
    edge_index: Tensor             # (2, E) static service dependency graph
    num_services: int              # number of graph nodes (== len(service_list))


@dataclass
class ModelOutput:
    """Per-service predictions for one batch of windows."""

    failure_probability: Tensor        # (B, S)
    failure_type_logits: Tensor        # (B, S, num_failure_types)
    time_to_failure: Tensor            # (B, S)

    failure_prob_std: Optional[Tensor] = None
    time_to_failure_std: Optional[Tensor] = None
    failure_prob_ci: Optional[Dict[str, Tensor]] = None
    time_to_failure_ci: Optional[Dict[str, Tensor]] = None

    service_embeddings: Optional[Tensor] = None    # (B, S, 256) pre-GNN
    gnn_node_embeddings: Optional[Tensor] = None    # (B, S, 256) post-GNN

    # LogRobust-style interpretability: per-service attention weight over
    # every log position in the window (0 outside that service's own
    # positions). None when use_attention_pooling=False.
    service_attention_weights: Optional[Tensor] = None   # (B, S, T)

    # LogBERT-style masked log-key auxiliary task (see MultiTaskLoss). Both
    # None outside training (use_mlkp_auxiliary=False, eval mode, or the
    # rare window where random masking selected zero real positions).
    mlkp_logits: Optional[Tensor] = None    # (N_masked, num_services)
    mlkp_targets: Optional[Tensor] = None   # (N_masked,)

    def to_dict(self) -> Dict[str, Any]:
        result = {
            "failure_probability": self.failure_probability.detach().cpu().numpy().tolist(),
            "failure_type_logits": self.failure_type_logits.detach().cpu().numpy().tolist(),
            "time_to_failure": self.time_to_failure.detach().cpu().numpy().tolist(),
        }
        if self.failure_prob_std is not None:
            result["failure_prob_std"] = self.failure_prob_std.detach().cpu().numpy().tolist()
        if self.time_to_failure_std is not None:
            result["time_to_failure_std"] = self.time_to_failure_std.detach().cpu().numpy().tolist()
        return result


def _pool_by_service(x: Tensor, service_ids: Tensor, num_services: int, mask: Tensor) -> Tensor:
    """
    Mean-pool per-log representations into one vector per service.

    A 5-minute window contains logs from every service, interleaved in
    time. Everything downstream (the GNN, the prediction heads) needs
    exactly one representation per service, so this is the seam between
    "per-log" and "per-service" — every branch that produces per-log
    output (sequence encoders, transformer) is pooled the same way here.

    Args:
        x:            (B, T, D) per-log representations.
        service_ids:  (B, T) long, 0-based index into the service list.
        num_services: number of distinct services (graph nodes).
        mask:         (B, T) bool, True = real log line.

    Returns:
        (B, num_services, D)
    """
    B, T, D = x.shape
    pooled = x.new_zeros(B, num_services, D)
    counts = x.new_zeros(B, num_services, 1)
    for s in range(num_services):
        svc_mask = (service_ids == s) & mask          # (B, T)
        svc_mask_f = svc_mask.unsqueeze(-1).float()    # (B, T, 1)
        pooled[:, s, :] = (x * svc_mask_f).sum(dim=1)
        counts[:, s, :] = svc_mask_f.sum(dim=1)
    return pooled / counts.clamp(min=1)


def _batch_edge_index(edge_index: Tensor, num_services: int, batch_size: int) -> Tensor:
    """
    Tile a single static service-dependency graph into a block-diagonal
    batch graph, the standard PyTorch Geometric batching trick: sample i's
    nodes are offset by i * num_services, so node indices never collide
    across samples even though every sample shares the same topology.
    """
    if edge_index.numel() == 0 or batch_size == 1:
        return edge_index
    device = edge_index.device
    offsets = torch.arange(batch_size, device=device) * num_services
    offsets = offsets.repeat_interleave(edge_index.size(1))     # (E*B,)
    tiled = edge_index.repeat(1, batch_size)                    # (2, E*B)
    return tiled + offsets.unsqueeze(0)


class LogSentinelModel(nn.Module):
    """
    Full multi-task deep learning model (§7): predicts, for every service
    in a 5-minute log window, (1) probability of failure in the next 5
    minutes, (2) which of 12 failure types, (3) seconds until failure.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        if config.sequence_encoder not in cfg.VALID_SEQUENCE_ENCODERS:
            raise ValueError(
                f"sequence_encoder={config.sequence_encoder!r} is not one of {cfg.VALID_SEQUENCE_ENCODERS}"
            )
        self.config = config

        self.embedder = LogEmbedder(
            num_services=config.num_services,
            num_error_codes=config.num_error_codes,
            embedding_dim=config.embed_dim,
            service_embed_dim=config.service_embed_dim,
            error_embed_dim=config.error_embed_dim,
            max_latency_ms=config.max_latency_ms,
            max_memory_mb=config.max_memory_mb,
            max_trace_depth=config.max_trace_depth,
            max_time_since_error=config.max_time_since_error_s,
        )

        seq_kwargs = dict(
            input_dim=config.embed_dim,
            hidden_dim=config.seq_hidden_dim,
            num_layers=config.seq_num_layers,
            dropout=config.seq_dropout,
        )
        # Only build the encoder(s) config.sequence_encoder actually asks
        # for — a single-encoder model shouldn't pay the parameter count
        # (or training time) of the other two it will never call.
        which = config.sequence_encoder
        self.lstm = LSTMEncoder(**seq_kwargs) if which in ("lstm", "all") else None
        self.gru = GRUEncoder(**seq_kwargs) if which in ("gru", "all") else None
        self.birnn = BiRNNEncoder(**seq_kwargs) if which in ("birnn", "all") else None
        # "rnn"/"bilstm" are independent single-encoder options like
        # lstm/gru/birnn, NOT folded into "all" — "all" stays the original
        # 3-way LSTM+GRU+BiRNN ensemble so existing results stay comparable.
        self.rnn = RNNEncoder(**seq_kwargs) if which == "rnn" else None
        self.bilstm = BiLSTMEncoder(**seq_kwargs) if which == "bilstm" else None

        self.transformer = TransformerEncoder(
            d_model=config.embed_dim,
            nhead=config.transformer_num_heads,
            num_layers=config.transformer_num_layers,
            dim_feedforward=config.transformer_ff_dim,
            dropout=config.transformer_dropout,
            output_dim=config.seq_hidden_dim,
        )

        # Sequence branch (avg of LSTM/GRU/BiRNN) + Transformer branch -> one
        # per-log representation.
        self.combine = nn.Sequential(
            nn.Linear(config.seq_hidden_dim * 2, config.seq_hidden_dim),
            nn.LayerNorm(config.seq_hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.transformer_dropout),
        )

        # LogRobust §3.3: attention-weighted per-service pooling instead of
        # a uniform mean (_pool_by_service, kept below as the ablation
        # fallback when this is off — see AttentionPooling's docstring for
        # why it's a strict generalization, never a regression).
        self.pooler = AttentionPooling(config.seq_hidden_dim) if config.use_attention_pooling else None

        # LogAnomaly §3.3: quantitative per-service count/statistics branch,
        # fused with the pooled sequence representation before the GNN.
        if config.use_quantitative_features:
            self.quant_proj = nn.Sequential(
                nn.Linear(QUANT_FEATURE_DIM, config.seq_hidden_dim),
                nn.LayerNorm(config.seq_hidden_dim),
                nn.ReLU(),
                nn.Dropout(config.seq_dropout),
            )
            self.quant_fusion = nn.Linear(config.seq_hidden_dim * 2, config.seq_hidden_dim)
        else:
            self.quant_proj = None
            self.quant_fusion = None

        # LogBERT §3.2 Task I (MLKP): train-time-only auxiliary task, see
        # _compute_mlkp. mask_embedding replaces a masked log line's
        # embedding before a second, additional Transformer-only forward
        # pass; mlkp_head predicts which service produced the masked line
        # from the surrounding cross-service context.
        if config.use_mlkp_auxiliary:
            self.mask_embedding = nn.Parameter(torch.zeros(config.embed_dim))
            nn.init.normal_(self.mask_embedding, std=0.02)
            self.mlkp_head = nn.Linear(config.seq_hidden_dim, config.num_services)
        else:
            self.mask_embedding = None
            self.mlkp_head = None

        if config.use_gat:
            self.gnn = GATStack(
                in_channels=config.seq_hidden_dim,
                hidden_channels=config.gnn_hidden_dim,
                out_channels=config.seq_hidden_dim,
                heads=config.gat_num_heads,
                dropout=config.gat_dropout,
            )
        else:
            self.gnn = GCNStack(
                in_channels=config.seq_hidden_dim,
                hidden_channels=config.gnn_hidden_dim,
                out_channels=config.seq_hidden_dim,
                num_layers=config.gnn_num_layers,
            )

        self.failure_head = FailureClassifier(input_dim=config.seq_hidden_dim)
        self.type_head = FailureTypeClassifier(
            input_dim=config.seq_hidden_dim, num_classes=config.num_failure_types
        )
        self.time_head = TimeToFailureRegressor(
            input_dim=config.seq_hidden_dim, max_seconds=config.time_to_failure_max_seconds
        )

        logger.info("LogSentinelModel initialized: %s", config)

    def forward(self, batch: LogBatch) -> ModelOutput:
        B, T = batch.service_ids.shape
        S = batch.num_services
        mask_bool = batch.attention_mask.bool()
        lengths = batch.attention_mask.sum(dim=1).clamp(min=1)

        embedded = self.embedder(
            service_ids=batch.service_ids,
            log_level_onehot=batch.log_level_onehot,
            error_code_ids=batch.error_code_ids,
            latency_ms=batch.latency_ms,
            cpu_usage=batch.cpu_usage,
            memory_mb=batch.memory_mb,
            time_since_error=batch.time_since_error,
            hour_of_day=batch.hour_of_day,
            is_anomaly=batch.is_anomaly,
            trace_depth=batch.trace_depth,
        )  # (B, T, embed_dim)

        # Sequence branch. "all" (the original design) ensembles all three
        # architectures by averaging rather than picking one (§7.3); a
        # single named sequence_encoder runs only that one, so it can be
        # trained/compared as an independent model instead of always
        # being folded into the 3-way average.
        if self.config.sequence_encoder == "all":
            seq_out = (
                self.lstm(embedded, lengths)
                + self.gru(embedded, lengths)
                + self.birnn(embedded, lengths)
            ) / 3.0  # (B, T, seq_hidden_dim)
        elif self.config.sequence_encoder == "lstm":
            seq_out = self.lstm(embedded, lengths)
        elif self.config.sequence_encoder == "gru":
            seq_out = self.gru(embedded, lengths)
        elif self.config.sequence_encoder == "birnn":
            seq_out = self.birnn(embedded, lengths)
        elif self.config.sequence_encoder == "rnn":
            seq_out = self.rnn(embedded, lengths)
        else:
            seq_out = self.bilstm(embedded, lengths)

        # Transformer branch: full self-attention over the whole window,
        # for cross-service correlations a purely-sequential model can't see.
        trans_out = self.transformer(embedded, src_key_padding_mask=~mask_bool)  # (B, T, seq_hidden_dim)

        combined = self.combine(torch.cat([seq_out, trans_out], dim=-1))  # (B, T, seq_hidden_dim)

        # From here on, everything is per-service, not per-log.
        service_attn_weights: Optional[Tensor] = None
        if self.pooler is not None:
            service_repr, service_attn_weights = self.pooler(combined, batch.service_ids, S, mask_bool)
        else:
            service_repr = _pool_by_service(combined, batch.service_ids, S, mask_bool)  # (B, S, H)

        # LogAnomaly §3.3: fuse a quantitative count/statistics branch in
        # alongside the pooled sequence representation.
        if self.quant_proj is not None:
            quant_feats = compute_quantitative_features(
                service_ids=batch.service_ids,
                log_level_onehot=batch.log_level_onehot,
                error_code_ids=batch.error_code_ids,
                latency_ms=batch.latency_ms,
                cpu_usage=batch.cpu_usage,
                memory_mb=batch.memory_mb,
                is_anomaly=batch.is_anomaly,
                attention_mask=batch.attention_mask,
                num_services=S,
            )  # (B, S, QUANT_FEATURE_DIM)
            quant_embedded = self.quant_proj(quant_feats)  # (B, S, H)
            service_repr = self.quant_fusion(torch.cat([service_repr, quant_embedded], dim=-1))

        # LogBERT §3.2 Task I (MLKP): train-time-only auxiliary loss signal,
        # computed from the *unmasked* `embedded` used above — never feeds
        # back into service_repr, so it can only regularize, not corrupt,
        # the supervised prediction path.
        mlkp_logits: Optional[Tensor] = None
        mlkp_targets: Optional[Tensor] = None
        if self.mlkp_head is not None and self.training:
            mlkp_logits, mlkp_targets = self._compute_mlkp(embedded, batch.service_ids, mask_bool)

        node_features = service_repr.reshape(B * S, -1)
        if batch.edge_index.numel() > 0:
            batched_edges = _batch_edge_index(batch.edge_index, S, B)
            gnn_out = self.gnn(node_features, batched_edges)
        else:
            gnn_out = node_features
        gnn_out = gnn_out.reshape(B, S, -1)  # (B, S, H)

        failure_prob = self.failure_head(gnn_out).squeeze(-1)   # (B, S)
        failure_type_logits = self.type_head(gnn_out)            # (B, S, num_failure_types)
        time_to_failure = self.time_head(gnn_out).squeeze(-1)    # (B, S)

        return ModelOutput(
            failure_probability=failure_prob,
            failure_type_logits=failure_type_logits,
            time_to_failure=time_to_failure,
            service_embeddings=service_repr,
            gnn_node_embeddings=gnn_out,
            service_attention_weights=service_attn_weights,
            mlkp_logits=mlkp_logits,
            mlkp_targets=mlkp_targets,
        )

    def _compute_mlkp(
        self, embedded: Tensor, service_ids: Tensor, mask_bool: Tensor
    ) -> "tuple[Optional[Tensor], Optional[Tensor]]":
        """LogBERT §3.2 Task I, adapted to a dependent multi-service window:
        mask ~mlkp_mask_ratio of real log positions and predict which
        *service* produced each masked line from the surrounding pattern of
        every other service's activity in the window — forcing the model to
        learn cross-service co-occurrence rather than single-entity
        self-consistency (LogBERT itself recovers a masked line's own
        template from its own sequence's context; there is no second
        "entity" to be dependent on in that single-sequence setting).

        Runs a second, additional Transformer-only forward pass on a masked
        *copy* of `embedded` — never touches the `embedded` tensor used by
        the main supervised path above.
        """
        rand_mask = (torch.rand(mask_bool.shape, device=mask_bool.device) < self.config.mlkp_mask_ratio) & mask_bool
        if not bool(rand_mask.any()):
            return None, None

        masked_embedded = embedded.clone()
        masked_embedded[rand_mask] = self.mask_embedding.to(masked_embedded.dtype)

        trans_masked = self.transformer(masked_embedded, src_key_padding_mask=~mask_bool)  # (B, T, seq_hidden_dim)
        logits_all = self.mlkp_head(trans_masked)  # (B, T, num_services)

        mlkp_logits = logits_all[rand_mask]    # (N_masked, num_services)
        mlkp_targets = service_ids[rand_mask]  # (N_masked,)
        return mlkp_logits, mlkp_targets

    def predict_with_uncertainty(self, batch: LogBatch, num_samples: Optional[int] = None) -> ModelOutput:
        """Monte Carlo Dropout uncertainty estimation (§7.7)."""
        T = num_samples or self.config.mc_dropout_samples
        stats = mc_predict_all(self, batch, T=T)

        # A clean, dropout-disabled forward pass for the type logits and
        # the interpretability embeddings (MC dropout only needs to
        # perturb failure_probability/time_to_failure for their CIs).
        self.eval()
        with torch.no_grad():
            point_estimate = self.forward(batch)

        return ModelOutput(
            failure_probability=stats["failure"]["mean"],
            failure_type_logits=point_estimate.failure_type_logits,
            time_to_failure=stats["time"]["mean"],
            failure_prob_std=stats["failure"]["std"],
            time_to_failure_std=stats["time"]["std"],
            failure_prob_ci={"lower": stats["failure"]["lower_ci"], "upper": stats["failure"]["upper_ci"]},
            time_to_failure_ci={"lower": stats["time"]["lower_ci"], "upper": stats["time"]["upper_ci"]},
            service_embeddings=point_estimate.service_embeddings,
            gnn_node_embeddings=point_estimate.gnn_node_embeddings,
            service_attention_weights=point_estimate.service_attention_weights,
        )

    def get_num_parameters(self) -> Dict[str, int]:
        """Parameter counts per component, for logging/debugging. Only
        components sequence_encoder actually built are included — no
        lstm/gru/birnn key for one that's None."""
        def count(module: Optional[nn.Module]) -> int:
            return sum(p.numel() for p in module.parameters())

        counts = {"embedder": count(self.embedder)}
        for name, module in (
            ("lstm", self.lstm), ("gru", self.gru), ("birnn", self.birnn),
            ("rnn", self.rnn), ("bilstm", self.bilstm),
        ):
            if module is not None:
                counts[name] = count(module)
        counts.update({
            "transformer": count(self.transformer),
            "gnn": count(self.gnn),
            "failure_head": count(self.failure_head),
            "type_head": count(self.type_head),
            "time_head": count(self.time_head),
        })
        if self.pooler is not None:
            counts["attention_pooling"] = count(self.pooler)
        if self.quant_proj is not None:
            counts["quantitative_branch"] = count(self.quant_proj) + count(self.quant_fusion)
        if self.mlkp_head is not None:
            counts["mlkp_auxiliary"] = count(self.mlkp_head) + self.mask_embedding.numel()
        counts["total"] = count(self)
        return counts
