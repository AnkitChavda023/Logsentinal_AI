# Module 2 Results — `microservices23-2h` (precursor-fixed dataset)

Real run, on Kaggle, of Module 2's full classical-ML ensemble against
`data/generated_microservices23_2h` (Kaggle dataset
`chadaankit/logsentinel-microservices23-2h`) — the same precursor-signal-fixed,
sequential-only generation config as the 4h dataset
(`module1_log_generator/config/microservices_23_full.yaml`, 23 services),
just a shorter 2-simulated-hour run.

This is the fourth real Module 2 run overall — see
[`RESULTS_MICROSERVICES23_4H.md`](RESULTS_MICROSERVICES23_4H.md) for the
third (the 4h version of this same topology/config) and `PRESENTATION.md`
§6 for the first two (an earlier microservices23 generation, before the
precursor-drift fix existed). Same evaluation code path across all four
(`training/evaluate.py`'s `evaluate_scorer()` / `top_flagged_anomalies()`).

Kernel: `chadaankit/logsentinel-module-2-microservices23-2h` (notebook
source: `kaggle_module2_microservices23_2h/train_microservices23_2h.ipynb`).

## Run setup

- **Data loaded**: 3,105,197 rows — an 8% sequential-per-file sample of the
  full ~38.8M-row dataset (`SAMPLE_FRAC=0.08`, a chronological prefix of
  each part, not a random draw), deliberately chosen so the sampled row
  count (~3.1M) matches the 4h run almost exactly — any difference between
  the two runs' numbers below reflects the underlying dataset (2h vs. 4h
  of simulated failures), not sample size.
- **Feature matrix**: (3,105,197, 26), anomaly rate 14.3% overall.
- **Split**: chronological 80/20 — 2,484,157 train rows (2,180,532 normal),
  621,040 test rows. True anomaly rate in the test split: **22.37%**
  (notably higher than the 4h run's 16.98%, and higher again than either
  originally-documented run in `PRESENTATION.md` — see Findings below).
- **Decision threshold**: 0.70 (spec §6.3).
- **Models evaluated**: Isolation Forest, Autoencoder, GMM, Random Forest,
  LightGBM.
- **Total Kaggle runtime**: ~58 minutes (CPU only) — noticeably faster than
  the 4h run's ~76 minutes despite an almost-identical row count; most
  likely session-to-session variance in Kaggle's shared CPU allocation
  rather than anything dataset-dependent.

## Results

| Rank | Model | Type | ROC-AUC | Precision | Recall | F1 | FPR | Predicted anomaly rate |
|---|---|---|---|---|---|---|---|---|
| 1 | LightGBM | Supervised | 0.9997 | 0.9978 | 0.9747 | 0.9861 | 0.0006 | 21.86% |
| 2 | Random Forest | Supervised | 0.9979 | 0.9897 | 0.9655 | 0.9775 | 0.0029 | 21.83% |
| 3 | GMM | Unsupervised | 0.9827 | 0.8525 | 0.9509 | 0.8990 | 0.0474 | 24.96% |
| 4 | Isolation Forest | Unsupervised | 0.9801 | 0.6981 | 0.9915 | 0.8193 | 0.1236 | 31.78% |
| 5 | Autoencoder | Unsupervised | 0.9491 | 0.9870 | 0.7809 | 0.8720 | 0.0030 | 17.70% |

(True anomaly rate: 22.37% for every row above — same held-out test split.)

### High-Degree Polynomial Linear Regression 

Evaluated independently on Kaggle (`chadaankit/logsentinel-module-2-microservices23-2h`) running **only** the High-Degree Polynomial Linear Regression model (`PolynomialRegressionAnomalyDetector`, degree=3, L2 Ridge alpha=1.0):

| Model | Type | ROC-AUC | Precision | Recall | F1 | FPR | Predicted anomaly rate |
|---|---|---|---|---|---|---|---|
| Polynomial Regression | Supervised | 0.9501 | 0.9986 | 0.8540 | 0.9207 | 0.0003 | 19.13% |

*(Evaluated at threshold 0.70 on the same held-out 621,040 test rows)*

## Findings

1. **GMM is now the best unsupervised model in all four real runs, not just three.** Run 1: 0.9957. Run 2: 0.9922. 4h run: 0.9735. This run: 0.9827. Four different datasets, two different topologies, always the top unsupervised model — this is now a well-established result for this project, not a one-off.
2. **Random Forest's recall jumped sharply versus the 4h run** (0.9655 here vs. 0.7541 on 4h), closing most of the gap with LightGBM. The most likely driver: this test split has more positive (anomalous) examples to learn from (22.37% vs. 16.98%), giving Random Forest a richer signal to generalize the anomaly boundary from — consistent with the pattern already noted in `PRESENTATION.md` §6 finding 1 (more failure examples → more genuinely capable, less suspiciously-perfect scores).
3. **Isolation Forest's recall is now near-perfect (0.9915) but at real cost: the worst precision (0.698) and highest false-positive rate (12.36%) of all five models**, both markedly worse than its own numbers on the 4h run (0.68 precision/8.9% FPR there too, but this pushes the trade-off further). It is flagging nearly a third of all test rows (31.78% predicted vs. 22.37% true) — the least trustworthy model here for a low-noise alerting pipeline, same conclusion as the 4h run, just more pronounced.
4. **Autoencoder stays the conservative outlier across both new runs**: lowest ROC-AUC of the five (0.9491) but the best-or-near-best precision (0.987) and lowest FPR (0.30%) alongside Random Forest. It misses more real anomalies (recall 0.78) than any other model, but essentially never false-alarms — the same shape as the 4h run's Autoencoder result.
5. **Test-split anomaly rate keeps climbing across runs**: 5.8% (Run 1) → 9.8% (Run 2) → 16.98% (4h) → 22.37% (2h). This is a property of each dataset's chronological tail (where the held-out 20% test split happens to fall relative to the failure schedule), not of the precursor-drift fix itself — the precursor mechanism only perturbs latency/cpu/memory before a failure's official onset and deliberately leaves `is_anomaly` untouched until then.
6. **High-Degree Polynomial Regression (Supervised) delivers stellar precision (0.9986) with an FPR of only 0.03% and F1 of 0.9207.** By expanding into degree-3 polynomial combinations with regularized Ridge regression, it captures complex interaction surfaces without exploding false alarms.


## Reproducing this run

```
kaggle_module2_microservices23_2h/
├── build_notebook.py       # generates train_microservices23_2h.ipynb
├── kernel-metadata.json    # id: chadaankit/logsentinel-module-2-microservices23-2h
└── train_microservices23_2h.ipynb
```

```bash
cd kaggle_module2_microservices23_2h
python build_notebook.py
kaggle kernels push -p .
```

Requires `logsentinel-module2-code` and `logsentinel-microservices23-2h` as
attached Kaggle dataset inputs (wired via `dataset_sources` in
`kernel-metadata.json`).
