# Module 3 — Deep Learning Failure Prediction (§7)

Predicts, for every service in the topology, whether it will fail in the
next 5 minutes, what type of failure (12-way classification), and how many
seconds until it happens — plus a causal root-cause chain through the
service dependency graph. Trains on Module 1's labeled synthetic logs;
serves predictions over a small FastAPI service Module 4 can call.

This README covers what was audited/fixed in this module and how to run
it. For the *why* behind Module 1/2's design choices see their own
READMEs; this one assumes you're already familiar with the project.

---

## 1. What was wrong, and what changed

The scaffolding already had two things: a set of small, individually
correct building blocks (`embeddings/`, `sequence_models/`,
`transformer/`, `gnn/`, `heads/`, `uncertainty/` — one file per §7.2-§7.7
component), and a separate `models/full_model.py` that was supposed to
assemble them into the actual model. It didn't — `full_model.py` was a
second, independent, hand-rolled reimplementation of every one of those
components instead of importing them, and nothing anywhere in the
codebase actually imported the good versions. That duplication is where
every real bug in this module traced back to:

1. **The model never pooled log representations per service — the
   central bug.** A 5-minute analysis window contains log lines from
   *every* service, interleaved in time. Everything past the sequence
   encoders needs exactly one vector *per service*, not per window. The
   old `full_model.py` skipped that step entirely: it treated "one
   window" as if it were already "one row per service" and fed that
   straight into the GNN. That's only dimensionally valid by accident
   when `batch_size == num_services`; for any real config it either
   silently computed nonsense or crashed with an index error inside the
   graph convolution (`edge_index` referencing service nodes that don't
   exist in a batch-of-windows-shaped tensor). Confirmed by tracing it
   through: `predictor_service.py` indexed `output.failure_probability[i]`
   for every service in the topology — on a tensor that was actually
   shaped `[batch_size]`, i.e. it would `IndexError` on the very first
   real request for any topology with more than one service.
   **Fixed**: added explicit per-service mean-pooling
   (`_pool_by_service` in `models/full_model.py`) after the sequence and
   transformer branches, and batched-graph propagation
   (`_batch_edge_index`) that tiles the static service graph once per
   sample in the batch with node-index offsets — the standard PyTorch
   Geometric batching trick. A regression test
   (`tests/test_full_model.py::test_batched_gnn_does_not_leak_across_samples`)
   asserts changing one sample's input never changes another sample's
   output.

2. **The duplicated embedder computed the service/error-code embeddings
   and then threw them away.** `full_model.py`'s own `LogEmbedder`
   created `nn.Embedding` layers for service and error code identity,
   called them, and never used the result — `service_emb` was computed
   and discarded every forward pass. Its actual input was a pre-built
   "37-dim" vector from `dataset.py` that reserved 24 zero-columns as
   placeholder space for those embeddings and then never filled them in
   (the comment literally said "filled by model" — nothing did). The
   model could never learn *which service or error code* produced a log
   line. **Fixed**: standardized on the already-correct
   `embeddings.LogEmbedder`, which takes raw typed fields directly
   (`service_ids`, `error_code_ids`, ...) and performs the embedding
   lookups inline, and rewrote `dataset.py` to emit those raw fields
   instead of a padded, partially-dead blob.

3. **Two incompatible service-index conventions.** `dataset.py` reserved
   index 0 for "unknown service" and numbered real services 1..N; the
   graph-building code (`_build_edge_index`, used by `train.py`,
   `evaluate.py`, and the old `predictor_service.py`) numbered them
   0..N-1 with no reservation. Both were used to mean "this service's
   position," so service index *i* meant a different service depending
   on which file touched it. **Fixed**: one canonical 0-based index
   space everywhere — `service_list[i]` is service_id `i` is graph node
   `i`, full stop (see the docstring on `LogBatch` in
   `models/full_model.py`). Covered by
   `tests/test_dataset_pipeline.py::test_service_index_matches_edge_index_convention`.

4. **Failure-type class indices were derived per-dataset.**
   `failure_type_to_idx` used to come from
   `failures_df['failure_type'].unique()` — so which integer meant
   "TIMEOUT" depended on which failure types happened to appear in
   whatever Parquet file was loaded that run. A checkpoint trained on one
   data file would silently mean something different when evaluated
   against another. **Fixed**: a single canonical 12-type ordering lives
   in `config/defaults.py::FAILURE_TYPES` (matching
   `module1_log_generator`'s `FailureType` enum order), imported by
   `dataset.py`, `predictor_service.py`, and the metrics/loss code alike.

5. **MC-Dropout uncertainty ran the model 3x more than necessary.**
   `full_model.py` had its own hand-rolled `predict_with_uncertainty`
   that was never connected to the also-present, already-correct
   `uncertainty/mc_dropout.py`. **Fixed**: extended `mc_dropout.py` with
   `mc_predict_all()`, which runs the T stochastic forward passes *once*
   and aggregates failure/type/time from the same samples, and wired
   `LogSentinelModel.predict_with_uncertainty` to call it.

6. **Augmentation and 6 of the module's own subpackages were dead code.**
   `training/augmentation.py` (`LogAugmentor`) existed but nothing in
   `train.py` ever constructed or called it. `embeddings/`,
   `sequence_models/`, `transformer/`, `gnn/`, `heads/`, and
   `uncertainty/` had no `__init__.py` and were never imported by
   anything except `full_model.py`'s duplicate code path (confirmed by
   grep — nothing outside `transformer/` itself referenced them).
   **Fixed**: gave every subpackage a proper `__init__.py`, rewired
   `full_model.py` to actually import and compose them, and wired
   `LogAugmentor` into `train.py`'s training DataLoader only (never
   validation/test) — rewritten to operate on the new named-field
   sample dict instead of hardcoded column offsets into a tensor layout
   that no longer exists.

7. **Smaller correctness fixes**, each confirmed by reading the actual
   call sites: `training/checkpoint.py`'s `torch.load` needed
   `weights_only=False` (newer PyTorch defaults to `True`, which rejects
   checkpoints that bundle optimizer state and metadata, not just
   tensors); `training/evaluate.py` referenced `Optional` without
   importing it; the multi-task loss's `.unsqueeze(0)` calls in
   `trainer.py`/`evaluate.py` were band-aids over the shape bug in (1)
   and are gone now that the model produces the correct `(batch,
   num_services, ...)` shape directly; `root_cause_tracer.py`'s
   anomaly/critical thresholds (0.40 / 0.70) were literals duplicated in
   four places — now sourced once from `config/defaults.py`, matching
   Module 2's ensemble thresholds so "anomalous" means the same thing in
   both modules.

Everything above was found by actually reading every call site the model
output flows through (not just the model in isolation), and every fix is
covered by a passing test — `pytest module3_deep_learning/tests/`.

### A design choice worth calling out: LSTM + GRU + BiRNN are *ensembled*, not selected

§7.3 asks for all three sequence architectures. The model runs all
three over the same embedded window and averages their per-timestep
outputs before the Transformer/GNN stages, rather than picking one at
training time. `use_gat` in `ModelConfig` is the one true "pick A or B"
ablation switch (GAT is primary per §7.5; GCN is the ablation baseline)
— the LSTM/GRU/BiRNN ensemble isn't meant to be an ablation knob the same
way.

### Known limitation: `trace_depth` is currently a constant

The §7.2 feature list includes trace depth (how deep in a call chain a
log line sits). `module1_log_generator`'s flat log schema doesn't
currently emit that as a column — `trace_simulator.py` computes a `depth`
during its DFS trace walk but never writes it onto the `LogEntry`. Rather
than change Module 1's already-finalized 16-column schema as a side
effect of a Module 3 audit, `dataset.py` and `training/inference.py` set
this field to a constant placeholder (`1.0`) with a comment explaining
why. If this signal turns out to matter for real training results, the
fix is a small, additive one in `module1_log_generator/models/log_entry.py`
+ `core/trace_simulator.py` + the CSV/OTel writers — not in Module 3.

---

## 2. Architecture (§7.2-§7.7)

```
raw log window (5 min, up to 300 lines, mixes every service)
  │
  ▼
embeddings.LogEmbedder            37-dim per-log embedding (§7.2)
  │
  ├─► sequence_models: LSTM + GRU + BiRNN, averaged   (§7.3, per-log output)
  └─► transformer.TransformerEncoder                   (§7.4, per-log output, full self-attention)
  │
  ▼  combine (concat + Linear + LayerNorm + ReLU)
per-log 256-dim representation
  │
  ▼  _pool_by_service (mean over each service's log positions)
per-service 256-dim representation  ──►  (batch * num_services, 256)
  │
  ▼  gnn.GATStack (primary) / gnn.GCNStack (ablation), batched via
     _batch_edge_index over the static service dependency graph  (§7.5)
per-service, graph-aware 256-dim representation
  │
  ├─► heads.FailureClassifier        → failure_probability   (§7.6 Task 1)
  ├─► heads.FailureTypeClassifier    → failure_type_logits    (§7.6 Task 2)
  └─► heads.TimeToFailureRegressor   → time_to_failure        (§7.6 Task 3)
  │
  ▼ (optional) uncertainty.mc_predict_all — T stochastic passes  (§7.7)
  │
  ▼ (optional) causal_chain.RootCauseTracer — walks the dependency
     graph upstream from the highest-risk service                (§7.8)
```

| Package | Role |
|---|---|
| `config/defaults.py` | Every tunable + the canonical 12-type failure taxonomy — no bare literals elsewhere |
| `embeddings/` | §7.2 — 37-dim log embedding |
| `sequence_models/` | §7.3 — LSTM / GRU / BiRNN encoders |
| `transformer/` | §7.4 — cross-service attention |
| `gnn/` | §7.5 — GAT (primary) / GCN (ablation) |
| `heads/` | §7.6 — the three prediction heads |
| `uncertainty/` | §7.7 — MC-Dropout |
| `causal_chain/` | §7.8 — root cause tracing through the dependency graph |
| `models/full_model.py` | Composes all of the above into `LogSentinelModel` |
| `training/` | Dataset, DataLoader, splitter, losses, trainer, callbacks, CLI entrypoints |
| `inference/predictor_service.py` | FastAPI service — the deployed artifact |

---

## 3. How to run

### 3.1 Setup

```bash
pip install -r requirements.txt   # from the repo root
```

Uses the same `venv`/`requirements.txt` as Modules 1 and 2 — `torch` and
`torch-geometric` are already listed there.

### 3.2 Generate training data (Module 1, no GPU needed)

```bash
python -m module1_log_generator.cli \
  --config module1_log_generator/config/services_ecommerce.yaml \
  --scale medium --format parquet --hours 168 --seed 42 \
  --output-dir data/generated
```

This also writes `data/generated/failure_manifest.csv` — Module 3's
training labels.

### 3.3 Configure

Copy `module3_deep_learning/config/training_example.yaml` and edit
`data.service_list` / `data.service_dependencies` / `data.error_code_list`
to match whichever `module1_log_generator/config/*.yaml` you generated
data with — the topology in both files must describe the same system, or
service indices in the trained model won't line up with the graph.

### 3.4 Train

```bash
python -m module3_deep_learning.training.train \
  --config module3_deep_learning/config/training_example.yaml \
  --data data/generated \
  --manifest data/generated/failure_manifest.csv \
  --output models/module3
```

Device selection is automatic (`cuda` if available, else `cpu`) — this
script is the one piece of the project worth running on a GPU; see
[`KAGGLE_TRAINING.md`](KAGGLE_TRAINING.md) for the local-generate →
Kaggle-train → local-serve workflow. Every path is a CLI flag, so the
exact same command runs on Kaggle with `/kaggle/input/...` /
`/kaggle/working/...` paths instead — no code changes needed either way.

Checkpoints land in `models/module3/checkpoints/model_best.pt`; the final
weights in `models/module3/model_final.pt`.

### 3.5 Evaluate against §10.1 targets

```bash
python -m module3_deep_learning.training.evaluate \
  --checkpoint models/module3/checkpoints/model_best.pt \
  --config module3_deep_learning/config/training_example.yaml \
  --data data/generated \
  --manifest data/generated/failure_manifest.csv \
  --output models/module3/eval_results.json
```

Reports precision/recall/F1/ROC-AUC/FPR, failure-type accuracy, and
time-to-failure MAE/RMSE against the chronological test split, with a
pass/fail check against each §10.1 target. Add `--uncertainty` to run
predictions through MC-Dropout instead of a single forward pass.

### 3.6 Serve predictions

```bash
python -c "
import uvicorn
from module3_deep_learning.inference.predictor_service import create_predictor_app
app = create_predictor_app('models/module3/model_final.pt', 'module3_deep_learning/config/training_example.yaml')
uvicorn.run(app, host='0.0.0.0', port=8003)
"
```

Or via Docker (`docker/Dockerfile.dl-predictor`, already wired to
`python -m module3_deep_learning.inference.predictor_service`).

Endpoints:
- `POST /predict` — send a `PredictionRequest` (a list of `LogEntry`,
  matching Module 1's flat schema) and get back a `PredictionResponse`
  with one `ServicePrediction` per service, sorted by failure
  probability, plus a causal chain for the top service if its
  probability is above the critical threshold.
- `POST /ingest` + the service's own `run_periodic_prediction()` — a
  buffering path for streaming logs in continuously and predicting on a
  fixed cadence (`config/defaults.py::PREDICTOR_INTERVAL_SECONDS`,
  30s) instead of one-shot request/response.
- `GET /health`

### 3.7 Run the tests

```bash
python -m pytest module3_deep_learning/tests/ -v
```

9 tests: shape correctness for both the GAT and GCN paths, the
batched-GNN no-cross-sample-leakage regression guard, MC-Dropout output
shapes, empty-graph handling, and the full
dataset → dataloader → LogBatch → model wiring using synthetic
Parquet/CSV fixtures (no real generated data required to run the suite).

### 3.8 Quick sanity check without any data

```python
import torch
from module3_deep_learning.models.full_model import LogSentinelModel, ModelConfig, LogBatch

config = ModelConfig(num_services=5, num_error_codes=10)
model = LogSentinelModel(config)
print(model.get_num_parameters())
```
