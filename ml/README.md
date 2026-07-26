# SeeCal — ML Pipeline

Fine-tunes Qwen3.5-4B (natively multimodal) on the Nutrition5k dataset to
estimate calories, macros, and ingredients from a single food photo. All
commands in this document are run from `ml/` (paths inside the generated
JSONL are relative to this directory, not to any subdirectory).

## Setup

Three scripts get a fresh checkout to a working state. Each is idempotent
(safe to re-run) and self-documenting (`--help`).

### 1. `./setup.sh` — create the Python environment

```bash
./setup.sh
```

Creates `ml/.venv` with the pinned training stack (mlx-vlm 0.6.7, mlx 0.32.0,
mlx-lm 0.31.3, transformers 5.14.1, datasets 5.0.0, torch 2.13.0,
torchvision 0.28.0, pytest) if it doesn't already exist. If `ml/.venv`
already exists, it verifies the installed versions against these pins and
reports status without touching the venv. Requires python3.12+ on PATH.
Ends with an import smoke check (`mlx_vlm`, `torch`, `torchvision`, `PIL`,
`numpy`).

torch/torchvision are installed as an explicit, separate step — mlx-vlm's
own extras do not pull them in, and training fails (late and confusingly)
without torchvision, since the Qwen3VL processor needs it at import time.

### 2. `./download_dataset.sh` — fetch Nutrition5k

```bash
./download_dataset.sh          # subset this pipeline needs (~3-4GB)
./download_dataset.sh --full   # entire public bucket (~181GB)
```

Downloads into `ml/Nutrition5K/` from the dataset's public GCS bucket
(`gs://nutrition5k_dataset/nutrition5k_dataset/`, see
[google-research-datasets/Nutrition5k](https://github.com/google-research-datasets/Nutrition5k)).
Requires `gsutil`. Released under CC BY 4.0 — run `./download_dataset.sh
--help` for the full license/citation notice. Dataset files are never
committed to git (see `.gitignore`).

### 3. `./download_model.sh` — fetch the base model

```bash
./download_model.sh                      # into ~/models/Qwen3.5-4B-MLX-4bit
./download_model.sh --dir /some/path      # custom MODELS_DIR
```

Downloads `mlx-community/Qwen3.5-4B-MLX-4bit` from Hugging Face via `hf
download` (falls back to installing `huggingface_hub[cli]` into
`ml/.venv` if no system `hf`/`huggingface-cli` is found). Prints the
resolved model path and the `MODEL_PATH` environment variable to set —
`train.sh` reads `MODEL_PATH` (default `$HOME/models/Qwen3.5-4B-MLX-4bit`)
so this is the only thing you need to wire up after downloading.

## Prep, train, eval, convert

<!-- TODO (R4): document prep.sh — select_images.py -> prepare_finetune.py ->
     smoke_test.py, including the validation-ladder gates (32-dish overfit,
     500-iter probe vs. base model) before any multi-hour training run. -->

<!-- TODO (R4): document train.sh — MODEL_PATH param, co-tenancy warning
     (don't run other Metal workloads during training), --grad-checkpoint
     tradeoff, resume-from-checkpoint. -->

<!-- TODO (R4): document eval.sh — REQUIRED --limit param (no silent-20
     default), how to read the MAE table. -->

<!-- TODO (R4): document convert.sh — adapter -> Swift LoRA format for iOS,
     adapter_config.json version stamping. -->

<!-- TODO (R4): end-to-end walkthrough with hardware requirements, timings
     (~3.5h train on M3 Ultra), the full validation ladder, and the current
     expected-metrics table (v5: calories MAE 54.4 / median 36.2 on 50
     dishes vs. base model 83.4). -->
