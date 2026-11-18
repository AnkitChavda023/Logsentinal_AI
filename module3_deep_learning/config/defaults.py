"""
Centralised defaults for Module 3 (Section 7).

Every tunable that isn't an exact spec-mandated constant lives here instead
of being a literal scattered across the codebase — same pattern as
module2_classical_ml/config/defaults.py. Values that ARE fixed by the spec
(e.g. multi-task loss weights 1.0/0.5/0.5, the 12-class failure taxonomy)
are kept here too since they're shared across many files and must stay in
lock-step.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Canonical failure-type taxonomy (§5.3 / §7.6 Task 2)
#
# This exact order is the model's class index space for the 12-way failure
# type classifier — index 0 is always LATENCY_SPIKE, index 11 is always
# DEPENDENCY_API_BREAK, etc. It MUST be derived the same way everywhere
# (training labels, inference output, predictor_service response) or a
# checkpoint trained with one ordering silently produces wrong labels when
# read back with another. Matches module1_log_generator's FailureType enum
# order exactly (kept as a literal, not an import, to preserve the
# module boundary — module3 must not import module1 directly).
# ---------------------------------------------------------------------------
FAILURE_TYPES: list[str] = [
    "LATENCY_SPIKE",
    "TIMEOUT",
    "CONNECTION_POOL_EXHAUSTION",
    "MEMORY_LEAK",
    "CASCADING_FAILURE",
    "THUNDERING_HERD",
    "NETWORK_PARTITION",
    "RATE_LIMITING",
    "DEADLOCK",
    "ZOMBIE_INSTANCE",
    "CONFIGURATION_DRIFT",
    "DEPENDENCY_API_BREAK",
]
NUM_FAILURE_TYPES = len(FAILURE_TYPES)

# ---------------------------------------------------------------------------
# Log embedding (§7.2)
# ---------------------------------------------------------------------------
EMBED_DIM = 128
SERVICE_EMBED_DIM = 16
ERROR_EMBED_DIM = 8
LOG_LEVEL_DIM = 5
MAX_LATENCY_MS = 10_000.0
MAX_MEMORY_MB = 8_192.0
MAX_TRACE_DEPTH = 20
MAX_TIME_SINCE_ERROR_S = 3_600.0

LOG_LEVEL_MAP = {
    "DEBUG": 0, "INFO": 1, "WARN": 2, "WARNING": 2, "ERROR": 3, "FATAL": 4, "CRITICAL": 4,
}

# ---------------------------------------------------------------------------
# Sequence models — LSTM / GRU / BiRNN (§7.3)
# ---------------------------------------------------------------------------
SEQ_HIDDEN_DIM = 256
SEQ_NUM_LAYERS = 2
SEQ_DROPOUT = 0.3

VALID_SEQUENCE_ENCODERS = ("lstm", "gru", "birnn", "rnn", "bilstm", "all")
SEQUENCE_ENCODER_DEFAULT = "all"
"""Which sequence encoder full_model.py builds and runs.
"all" is the original §7.3 design (LSTM+GRU+BiRNN averaged into one
branch — unchanged by the later addition of "rnn"/"bilstm" below, so
existing "all"-ensemble results stay comparable). A single named choice
builds and runs only that one encoder, for training/comparing it as an
independent model — e.g. to answer "does the 3-way ensemble actually beat
its best single member," which "all" alone can never tell you. "rnn"
(vanilla, ungated) and "bilstm" (bidirectional LSTM, a distinct sibling of
"birnn" — see sequence_models/bilstm_encoder.py's docstring) are two more
such independent single-encoder options, added the same way lstm/gru/birnn
already work."""

# ---------------------------------------------------------------------------
# Transformer (§7.4)
# ---------------------------------------------------------------------------
TRANSFORMER_NUM_LAYERS = 4
TRANSFORMER_NUM_HEADS = 8
TRANSFORMER_FF_DIM = 512
TRANSFORMER_DROPOUT = 0.1
MAX_SEQ_LENGTH = 300  # max log lines per analysis window (also PositionalEncoding max_len)

# ---------------------------------------------------------------------------
# GNN (§7.5)
# ---------------------------------------------------------------------------
GNN_HIDDEN_DIM = 256
GNN_NUM_LAYERS = 3
GNN_DROPOUT = 0.2  # GCN path
GAT_DROPOUT = 0.3  # GAT path (attention dropout)
GAT_NUM_HEADS = 4
USE_GAT_BY_DEFAULT = True  # GAT is the primary model; GCN is the ablation baseline

# ---------------------------------------------------------------------------
# Prediction heads (§7.6)
# ---------------------------------------------------------------------------
HEAD_HIDDEN_DIM = 64
TIME_TO_FAILURE_MAX_SECONDS = 300.0

# Multi-task loss weights — exact spec values (§7.6), kept alongside the
# taxonomy rather than as free-floating literals in losses.py.
LOSS_WEIGHT_FAILURE = 1.0
LOSS_WEIGHT_TYPE = 0.5
LOSS_WEIGHT_TIME = 0.5

# ---------------------------------------------------------------------------
# Uncertainty (§7.7)
# ---------------------------------------------------------------------------
MC_DROPOUT_SAMPLES = 50
CONFIDENCE_Z_SCORE = 1.96  # 95% CI

# ---------------------------------------------------------------------------
# Windowing / prediction horizon (§7 batch job cadence)
# ---------------------------------------------------------------------------
WINDOW_SIZE_MINUTES = 5
WINDOW_STRIDE_MINUTES = 1
PREDICTION_HORIZON_MINUTES = 5
PREDICTOR_INTERVAL_SECONDS = 30

# ---------------------------------------------------------------------------
# Root cause tracing (§7.8) — shared with Module 2's ensemble thresholds
# (§6.3: WARNING >= 0.40, ANOMALY >= 0.70) so "anomalous" means the same
# thing in both modules.
# ---------------------------------------------------------------------------
ANOMALY_THRESHOLD = 0.40
CRITICAL_ANOMALY_THRESHOLD = 0.70
ROOT_CAUSE_MAX_DEPTH = 10
ROOT_CAUSE_MIN_CONFIDENCE = 0.5

# ---------------------------------------------------------------------------
# Evaluation targets (§10.1)
# ---------------------------------------------------------------------------
TARGET_PRECISION = 0.90
TARGET_RECALL = 0.85
TARGET_FALSE_POSITIVE_RATE = 0.05
TARGET_FAILURE_TYPE_ACCURACY = 0.75
TARGET_TIME_TO_FAILURE_MAE_S = 30.0

# ---------------------------------------------------------------------------
# Training defaults
# ---------------------------------------------------------------------------
DEFAULT_BATCH_SIZE = 16
DEFAULT_LEARNING_RATE = 1e-4
DEFAULT_WEIGHT_DECAY = 0.01
DEFAULT_GRADIENT_CLIP = 1.0
DEFAULT_NUM_EPOCHS = 50
DEFAULT_EARLY_STOPPING_PATIENCE = 10
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1

# ---------------------------------------------------------------------------
# Env vars (mirrors module2_classical_ml/config/defaults.py's convention)
# ---------------------------------------------------------------------------
PREDICTOR_MODEL_DIR_ENV_VAR = "LOGSENTINEL_MODULE3_MODEL_DIR"
PREDICTOR_CONFIG_ENV_VAR = "LOGSENTINEL_MODULE3_CONFIG"
SCORER_DEFAULT_MODEL_DIR = "models/module3"

# ---------------------------------------------------------------------------
# Paper-grounded architecture upgrades, adapted from the independent-entity
# setting each paper assumes (one HDFS block / one BGL node / one Service-X
# instance) to this project's dependent multi-service windows:
#   - LogRobust (Zhang et al., ESEC/FSE'19): attention-weighted pooling
#     instead of raw occurrence counts (§3.3) -> attention/log_attention_pooling.py
#   - LogAnomaly (Meng et al., IJCAI'19): quantitative count-vector branch
#     alongside the sequential branch (§3.3) -> quantitative/count_features.py
#   - LogBERT (Guo et al., 2021): masked log-key prediction from bidirectional
#     context (§3.2 Task I MLKP) -> models/full_model.py's mask_embedding/mlkp_head
# ---------------------------------------------------------------------------
USE_ATTENTION_POOLING_BY_DEFAULT = True
USE_QUANTITATIVE_FEATURES_BY_DEFAULT = True
USE_MLKP_AUXILIARY_BY_DEFAULT = True
MLKP_MASK_RATIO = 0.15  # LogBERT Fig 4b's sweet spot (0.1-0.5 range, F1 peaks ~0.15-0.3)
LOSS_WEIGHT_MLKP = 0.3  # modest vs. LOSS_WEIGHT_TYPE/TIME=0.5 — regularizer, not co-equal task
