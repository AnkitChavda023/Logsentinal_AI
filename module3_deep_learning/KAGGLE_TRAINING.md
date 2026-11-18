# Running Module 3 (Deep Learning) on Kaggle — Setup Guide

Modules 1, 2, and 4 run entirely locally (no GPU needed — confirmed this session: Isolation Forest and the small Autoencoder both train in seconds on CPU). **Module 3** (LSTM + GRU + BiRNN + Transformer + GNN, per §7 of the spec — ~5.3M parameters) is a real deep model that will train much faster on a GPU than on your local CPU-only machine — that's the one piece worth using Kaggle for.

This guide covers the split workflow: **generate data locally → train on Kaggle (GPU) → bring the checkpoint back and serve/evaluate locally.** Module 3's training code runs identically in both places (see §6 below) — nothing here requires two different code paths, and every command below was verified against the actual, working CLI this session (not a planned/future interface).

---

## 1. One-time Kaggle account setup

1. Create a free account at [kaggle.com](https://www.kaggle.com) if you don't have one.
2. **Verify your phone number** on your account (Settings → Phone Verification) — this is required to unlock GPU access; without it, notebooks are CPU-only.
3. Get an API token: Kaggle → **Settings** → **API** → **Create New Token**. This downloads `kaggle.json` containing your username + key.
4. Put it where the `kaggle` CLI expects it:
   ```bash
   mkdir -p ~/.kaggle
   mv ~/Downloads/kaggle.json ~/.kaggle/kaggle.json
   chmod 600 ~/.kaggle/kaggle.json      # on Windows/git-bash this step is a no-op, safe to skip
   ```
5. Install the CLI locally (used to upload datasets and, optionally, pull results back down):
   ```bash
   pip install kaggle
   kaggle --version   # confirms auth works
   ```

**GPU quota to expect:** Kaggle gives ~30 GPU-hours/week (P100 or T4×2, your choice in notebook settings), and each individual session is capped around 9–12 hours before it auto-stops. Plenty for training Module 3 in chunks; not enough to run it as an always-on service — that part still happens locally.

---

## 2. Getting your data onto Kaggle

Two ways to get there — generate locally then upload the result, or generate straight on Kaggle's compute. Pick whichever suits your workflow; both end with the same thing, a Kaggle Dataset containing the Parquet parts + `failure_manifest.csv`.

### Path A — generate locally, then upload

Generate your training dataset locally first (this doesn't need a GPU):

```bash
python -m module1_log_generator.cli \
  --config module1_log_generator/config/services_ecommerce.yaml \
  --scale medium --format parquet --hours 168 --seed 42 \
  --output-dir data/generated
```

Then upload `data/generated/` as a **Kaggle Dataset**:

**Via website (simplest for a one-off):**
Kaggle → **Create** → **New Dataset** → drag in the `data/generated/` folder → give it a title (e.g. `logsentinel-training-data`) → Create.

**Via CLI (better if you'll re-upload updated data repeatedly):**
```bash
cd data/generated
kaggle datasets init -p .                      # creates a dataset-metadata.json — edit the "title"/"id" fields
kaggle datasets create -p .                     # first upload
# later, after regenerating data:
kaggle datasets version -p . -m "regenerated with 7 days"
```

### Path B — generate directly on Kaggle (no local generation step)

Useful if you'd rather not generate and move a large Parquet dataset over your own connection, or want the whole pipeline — generate → save dataset → train — to live inside Kaggle. This needs a **separate notebook from the training one** (or a separate first section in the same notebook), because it needs the whole project's code, not just `module3_deep_learning/`.

1. Upload the **whole project** (`module1_log_generator/`, `module3_deep_learning/`, `shared/`, `requirements.txt` — not `data/`, `venv/`, or `.git/`) as one Kaggle Dataset, same `kaggle datasets init`/`create` pattern as §4 below, e.g. titled `logsentinel-project-code`.
2. New Notebook → **Settings** → Internet **On** (GPU not needed for this step — generation is CPU-only, feel free to leave the accelerator off to save your GPU-hour quota) → **Add input**: the project-code dataset.
3. Copy the code in and install what Module 1 needs (Kaggle's base image already has `numpy`/`pandas`/`pyarrow`/`networkx`; these four usually aren't preinstalled):
   ```python
   import shutil, sys
   shutil.copytree("/kaggle/input/logsentinel-project-code", "/kaggle/working/project", dirs_exist_ok=True)
   sys.path.insert(0, "/kaggle/working/project")
   !pip install -q pydantic click pyyaml faker
   ```
4. Run the generator straight into `/kaggle/working/`:
   ```python
   !cd /kaggle/working/project && python -m module1_log_generator.cli \
       --config module1_log_generator/config/services_ecommerce.yaml \
       --scale medium --format parquet --hours 168 --seed 42 \
       --output-dir /kaggle/working/generated_data
   ```
   This also writes `/kaggle/working/generated_data/failure_manifest.csv` and `service_graph.html` — same output shape as a local run.
5. Also drop the training config next to it so it rides along in the same dataset:
   ```python
   shutil.copy(
       "/kaggle/working/project/module3_deep_learning/config/training_example.yaml",
       "/kaggle/working/generated_data/training_example.yaml",
   )
   ```
6. **Save it as a new Kaggle Dataset** — two ways:
   - **Notebook UI (simplest):** Save Version (so the run actually completes and `/kaggle/working/` is finalized) → open the notebook's **Output/Data** panel → the `generated_data/` folder appears there → click **"New Dataset"** on it → gives you e.g. `<you>/generated-data` you can attach as input to the training notebook in §5.
   - **`kaggle` CLI, scripted, from inside the notebook** (needs your Kaggle API credentials available as notebook **Secrets** — Add-ons → Secrets → add `KAGGLE_USERNAME`/`KAGGLE_KEY`):
     ```python
     import os
     from kaggle_secrets import UserSecretsClient
     secrets = UserSecretsClient()
     os.environ["KAGGLE_USERNAME"] = secrets.get_secret("KAGGLE_USERNAME")
     os.environ["KAGGLE_KEY"] = secrets.get_secret("KAGGLE_KEY")

     !cd /kaggle/working/generated_data && kaggle datasets init -p .
     # edit dataset-metadata.json's "title"/"id" (or write it directly with Python before init)
     !cd /kaggle/working/generated_data && kaggle datasets create -p .
     ```
   Either way you end up with a `logsentinel-training-data`-equivalent Kaggle Dataset, ready to attach to the training notebook exactly like Path A's output — continue at §5 (Creating the training notebook) using this dataset as the data input instead.

Whichever path you used, the training config (`training_example.yaml`) should end up **inside the same dataset** as the Parquet + manifest — that's what lets §5's `--config`/`--data`/`--manifest` paths all point at one attached input.

---

## 3. Getting Module 3's code onto Kaggle

**If you used Path B above**, you already uploaded the whole project (including `module3_deep_learning/`) as one code dataset — reuse that same dataset as the training notebook's code input and skip straight to §5. This section is for Path A, or if you'd rather keep a code-only dataset separate from the full project.

Since git push isn't a clean option here (see note above the guide), upload code the same way as data — as a Kaggle Dataset, kept separate from the data dataset so you can update code without re-uploading gigabytes of Parquet:

```bash
cd module3_deep_learning
kaggle datasets init -p .
# edit dataset-metadata.json: title "logsentinel-module3-code", id "<your-username>/logsentinel-module3-code"
kaggle datasets create -p .
```

Also include `shared/` (schemas Module 3 imports) and `requirements.txt` in that same upload — either copy them alongside before zipping, or upload as one more small dataset. Whenever you change the code locally, re-upload with `kaggle datasets version -p . -m "..."` and re-run the notebook cell that unzips/copies it in — no need to recreate the notebook.

---

## 4. Creating the training notebook

1. Kaggle → **Create** → **New Notebook**.
2. Right panel → **Settings**:
   - **Accelerator**: GPU T4 ×2 (or P100 if offered) — anything but "None".
   - **Internet**: **On** (needed to `pip install` anything not preinstalled, e.g. `torch-geometric`).
   - **Add input**: attach both datasets you created above (code + data).
3. First cell — install anything Kaggle's base image doesn't already have (Kaggle preinstalls `torch`; `torch-geometric` usually still needs an explicit, CUDA-version-matched install):
   ```python
   import torch
   print(torch.__version__, torch.version.cuda, torch.cuda.is_available())
   !pip install -q torch-geometric
   ```
4. Copy the code dataset into the working directory (Kaggle inputs are read-only, and Python needs the package importable):
   ```python
   import shutil
   shutil.copytree("/kaggle/input/logsentinel-module3-code", "/kaggle/working/module3_deep_learning", dirs_exist_ok=True)
   shutil.copytree("/kaggle/input/logsentinel-module3-code-shared", "/kaggle/working/shared", dirs_exist_ok=True)  # if uploaded separately
   import sys; sys.path.insert(0, "/kaggle/working")
   ```
5. Run training, pointing at the Kaggle input path for data and Kaggle's writable output path for checkpoints (flags match `training/train.py`'s actual `argparse` CLI — see `module3_deep_learning/README.md` §3.4):
   ```python
   !python -m module3_deep_learning.training.train \
       --config /kaggle/input/logsentinel-training-data/training_example.yaml \
       --data /kaggle/input/logsentinel-training-data \
       --manifest /kaggle/input/logsentinel-training-data/failure_manifest.csv \
       --output /kaggle/working/models/module3
   ```
   `num_epochs`, `batch_size`, learning rate etc. are read from the config YAML, not CLI flags — edit `training_example.yaml` (or a copy of it) before uploading it as part of the data dataset. Add `--resume /kaggle/working/models/module3/checkpoints/model_best.pt` to continue a run across Kaggle's ~9-12hr session cap.
6. Everything written under `/kaggle/working/` is downloadable from the notebook's **Output** tab after the run finishes (or via `kaggle kernels output <your-username>/<notebook-slug> -p ./downloaded` from your local machine).

---

## 5. Bringing the trained model back locally

Download the checkpoint(s) from the notebook's Output tab (or the CLI command above) into your local `models/module3/` directory, mirroring how Module 2's `models/module2/isolation_forest.pkl`/`autoencoder.pt` already work. Module 3's inference code then runs **locally**, CPU is fine for serving — GPU only mattered for training throughput, not for scoring one request at a time.

---

## 6. The design rule that makes this split painless

Module 3's training/inference code follows the exact same two rules Module 2's Autoencoder does — meaning **no Kaggle-specific code branch is ever needed**:

1. **Device-agnostic**: `torch.device("cuda" if torch.cuda.is_available() else "cpu")`, used in `training/train.py`, `training/evaluate.py`, and `inference/predictor_service.py`. On Kaggle this picks up the T4/P100 automatically; locally it falls back to CPU automatically. Same line of code, same file, both places.
2. **Nothing hardcoded — every path is a CLI argument with a sensible local default** (`--data`, `--manifest`, `--config`, `--output`, plain `argparse.Path` options — matching the style `training/train.py` and `training/evaluate.py` already used). Running on Kaggle is just passing `/kaggle/input/...`/`/kaggle/working/...` instead of `data/generated`/`models/module3` — the script itself doesn't know or care which environment it's in.

So: build once, run the identical script either place, only the `--data`/`--manifest`/`--output` flags change.
