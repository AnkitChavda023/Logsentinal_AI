# Module 2 Results — `microservices23-4h` (precursor-fixed dataset)

Real run, on Kaggle, of Module 2's full classical-ML ensemble against
`data/generated_microservices23_4h` (Kaggle dataset
`chadaankit/logsentinel-microservices23-4h`) — the precursor-signal-fixed,
sequential-only 4-simulated-hour dataset generated this session for Module 3
(23-service topology, `module1_log_generator/config/microservices_23_full.yaml`).

This is a new, third run — distinct from the two already documented in
`PRESENTATION.md` §6, which used an earlier `microservices_23_full.yaml`
generation, before the precursor-drift fix
(`module1_log_generator/failures/failure_injector.py`'s `_apply_precursor`)
existed. Same evaluation code path as those two runs
(`training/evaluate.py`'s `evaluate_scorer()` / `top_flagged_anomalies()`),
so the numbers are directly comparable in shape.

Kernel: `chadaankit/logsentinel-module-2-microservices23-4h` (notebook source:
`kaggle_module2_microservices23_4h/train_microservices23_4h.ipynb`).

## Run setup

- **Data loaded**: 3,106,313 rows — a 4% sequential-per-file sample of the
  full ~77.7M-row dataset (`load_parquet_dataset(..., sample_frac=0.04)`;
  a chronological prefix of each part, not a random draw — see
  `training/data_loading.py`), landing close to the ~2.9M-row size the
  already-documented Run 2 used.
- **Feature matrix**: (3,106,313, 26) — the same 26-dim engineered feature
  vector described in `PRESENTATION.md` §2 (Groups A-G), anomaly rate 14.4%
  overall.
- **Split**: chronological 80/20 (never random) — 2,485,050 train rows
  (2,144,520 normal), 621,263 test rows. True anomaly rate in the test
  split: 16.98%.
- **Decision threshold**: 0.70 (spec §6.3).
- **Models evaluated**: Isolation Forest, Autoencoder, GMM, Random Forest,
  LightGBM — LOF and PCA reconstruction stay removed (see `PRESENTATION.md`
  §3/§7 for why).
- **Total Kaggle runtime**: ~76 minutes (CPU only, `enable_gpu: false`).

## Results

| Rank | Model | Type | ROC-AUC | Precision | Recall | F1 | FPR | Predicted anomaly rate |
|---|---|---|---|---|---|---|---|---|
| 1 | LightGBM | Supervised | 0.9987 | 0.9936 | 0.9444 | 0.9684 | 0.0013 | 16.14% |
| 2 | Random Forest | Supervised | 0.9979 | 0.9893 | 0.7541 | 0.8558 | 0.0017 | 12.94% |
| 3 | GMM | Unsupervised | 0.9735 | 0.7466 | 0.9454 | 0.8343 | 0.0656 | 21.51% |
| 4 | Isolation Forest | Unsupervised | 0.9707 | 0.6800 | 0.9248 | 0.7837 | 0.0890 | 23.10% |
| 5 | Autoencoder | Unsupervised | 0.9272 | 0.9844 | 0.8538 | 0.9144 | 0.0028 | 14.73% |

(True anomaly rate: 16.98% for every row above — same held-out test split.)

### High-Degree Polynomial Linear Regression (Dedicated Kaggle Run)

Evaluated independently on Kaggle (`chadaankit/logsentinel-module-2-microservices23-4h`) running **only** the High-Degree Polynomial Linear Regression model (`PolynomialRegressionAnomalyDetector`, degree=3, L2 Ridge alpha=1.0):

| Model | Type | ROC-AUC | Precision | Recall | F1 | FPR | Predicted anomaly rate |
|---|---|---|---|---|---|---|---|
| Polynomial Regression | Supervised | 0.9918 | 0.9991 | 0.8470 | 0.9168 | 0.0001 | 14.40% |

*(Evaluated at threshold 0.70 on the same held-out 621,263 test rows)*

## Findings

1. **GMM is the best unsupervised model on all three real runs now, not just two.** Run 1 (ecommerce): #2 unsupervised at 0.9957. Run 2 (earlier microservices23): #3 overall, best unsupervised at 0.9922. This run: best unsupervised at 0.9735. That's a consistent result across three different datasets and two different topologies — the strongest cross-dataset finding this module has produced.
2. **Random Forest is *not* suspiciously perfect here (0.9979, not 1.0000)**, continuing the pattern from Run 2 rather than Run 1's likely-memorized 1.0000 on a much smaller failure count. With 16.98% of this test split anomalous (far more failure examples to generalize from than Run 1's 16 total events), a high-but-not-perfect score is the more trustworthy result.
3. **Autoencoder has the lowest ROC-AUC (0.9272) of the five but by far the best precision (0.9844) and lowest false-positive rate (0.28%).** It's conservative: it misses more true anomalies (recall 0.8538) than GMM or Isolation Forest, but almost never cries wolf. Isolation Forest is the opposite extreme — second-highest ROC-AUC among unsupervised models (0.9707) but the *worst* precision (0.68) and FPR (8.9%) of all five, over-flagging by a wide margin. For a real alerting pipeline, that precision/recall shape matters more than the ROC-AUC ranking alone (see `PRESENTATION.md` §5 on why).
4. **This dataset's test-split anomaly rate (16.98%) is notably higher than either prior run** (Run 1: 5.8%, Run 2: 9.8%). This reflects real, higher failure density in this particular 4h/883-failure generation and where the chronological test slice happens to fall — not the precursor-drift fix itself, which only perturbs latency/cpu/memory *before* a failure's official onset and deliberately leaves `is_anomaly` untouched until then (`failure_injector.py`'s `_apply_precursor` docstring).
5. **High-Degree Polynomial Regression achieves exceptional performance on 4h**: Supervised mode reaches **0.9918 ROC-AUC** with **99.91% precision** and virtually zero false alarms (**0.01% FPR**), competitive with tree ensembles while providing a continuous polynomial manifold.



## Reproducing this run

```
kaggle_module2_microservices23_4h/
├── build_notebook.py       # generates train_microservices23_4h.ipynb
├── kernel-metadata.json    # id: chadaankit/logsentinel-module-2-microservices23-4h
└── train_microservices23_4h.ipynb
```

```bash
cd kaggle_module2_microservices23_4h
python build_notebook.py
kaggle kernels push -p .
```

Requires `logsentinel-module2-code` and `logsentinel-microservices23-4h` as
attached Kaggle dataset inputs (wired via `dataset_sources` in
`kernel-metadata.json`).
