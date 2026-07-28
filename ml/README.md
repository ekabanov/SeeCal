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

The official metadata (`dish_metadata_cafe1.csv` / `dish_metadata_cafe2.csv`)
is a raw, variable-width format — not what `prepare_finetune.py` reads.
After fetching, this script **automatically runs `convert_metadata.py`**
to derive the tidy, fixed-schema CSVs the pipeline actually reads
(`dish_nutrition_values.csv`, `dish_ingredients.csv`,
`ingredients_metadata.csv`) whenever they're missing. By default this
reproduces the dataset's historical, published numbers **exactly**:
cafe1 only (4768 dishes / 27225 ingredient rows / 555 ingredients) — cafe2's
238 dishes were, for reasons lost to history (likely an oversight, not a
deliberate filter), never folded into the tidy files this repo's metrics
are based on. Pass `--include-cafe2` (or run `convert_metadata.py --cafes
all` yourself) to recover cafe2's dishes too (~228 of which have
`realsense_overhead` imagery, so are usable for training) — this yields a
different, larger dataset (5006 dishes) that will not match the numbers in
this document. See `convert_metadata.py`'s module docstring for the full
verification story (it byte-for-byte reproduces this repo's tidy CSVs from
the real raw files; see `tests/test_convert_metadata.py`'s integration
check).

### 3. `./download_model.sh` — fetch the base model

```bash
./download_model.sh                      # into ~/models/mlx-community/Qwen3.5-4B-MLX-4bit
./download_model.sh --dir /some/path      # custom MODELS_DIR
```

Downloads `mlx-community/Qwen3.5-4B-MLX-4bit` from Hugging Face via `hf
download` (falls back to installing `huggingface_hub[cli]` into
`ml/.venv` if no system `hf`/`huggingface-cli` is found). Prints the
resolved model path and the `MODEL_PATH` environment variable to set —
`train.sh` reads `MODEL_PATH` (default `$HOME/models/Qwen3.5-4B-MLX-4bit`)
so this is the only thing you need to wire up after downloading.

## End-to-end walkthrough

```bash
./setup.sh                                   # 1. python env
./download_dataset.sh                        # 2. Nutrition5k subset (+ auto metadata convert)
./download_model.sh                          # 3. base model
export MODEL_PATH=~/models/mlx-community/Qwen3.5-4B-MLX-4bit   # printed by download_model.sh
./prep.sh                                    # 4. select images -> JSONL -> smoke test
./train.sh                                   # 5. LoRA fine-tune (~3.5h on M3 Ultra)
./eval.sh adapters_v5 --limit 325            # 6. full test-set evaluation
./convert.sh adapters_v5                     # 7. convert for iOS
```

### 4. `./prep.sh` — select images, build fine-tune JSONL, smoke test

```bash
./prep.sh                                    # overhead-only, no depth (matches v5)
./prep.sh --depth-mode text                  # depth-augmented variant B
./prep.sh --skip-smoke                       # data only, skip the smoke test
```

Wraps `select_images.py --max-images 1` (overhead photo only — matches the
deployed iPhone scenario) → `prepare_finetune.py --only-overhead` →
`smoke_test.py` on the freshly generated `train.jsonl`. This is rung 1 of
the validation ladder (below); `--depth-mode {none,text,image}` is
forwarded to `prepare_finetune.py`, auto-enabling `select_images.py
--with-depth` when it isn't `none` (depth modes need `depth_raw.png`
present in `dataset_clean/`). `--out-dir` overrides the output directory
(default depends on `--depth-mode`: `finetune_data_v2` /
`finetune_data_v2d_txt` / `finetune_data_v2d_img`). Run `./prep.sh --help`
for the full flag list.

### 5. `./train.sh` — LoRA fine-tune

```bash
./train.sh                                             # fresh v5-style run
OUTPUT_PATH=adapters_v7 ./train.sh                      # new versioned run
ADAPTER_PATH=adapters_v7 OUTPUT_PATH=adapters_v7 ./train.sh   # resume v7
```

Configured entirely through environment variables (no positional args) —
`MODEL_PATH` (base model), `OUTPUT_PATH` (adapter output dir, **pick an
explicit versioned name** like `adapters_v7` for any run that isn't a
deliberate resume of the existing default), `DATASET`
(default `finetune_data_v2`), and `ADAPTER_PATH` (resume from an existing
checkpoint). The script warns — but does not refuse — if `OUTPUT_PATH`
already has a checkpoint and `ADAPTER_PATH` is unset, since that combination
means you're about to overwrite it. Run `./train.sh --help` for the full
list and the resume example.

Do **not** run anything else GPU-heavy (LM Studio, `eval.sh`, another
training run) at the same time — Metal memory doesn't swap, so co-tenancy
silently OOM-kills the run instead of failing cleanly. See Hardware
requirements below.

### 6. `./eval.sh` — evaluate an adapter

```bash
./eval.sh adapters_v5 --limit 325                       # full test set
./eval.sh adapters_v5 --limit 20 --out-json runs/quick.json   # quick check
```

Wraps `infer.py --test-set` for evaluating a LoRA adapter against a JSONL
test set (MAE + median error for calories/protein/fat/carbs).
`ADAPTER_PATH` is a **required** first positional argument, and `--limit`
is a **required** flag with no default: `infer.py`'s own `--limit` defaults
to 20, and that default once silently truncated a full overnight evaluation
run down to 20 samples with no warning. `eval.sh` refuses to guess for you.
`--test-set` defaults to `finetune_data_v2/test.jsonl`; `--out-json`
auto-names its output under `runs/` (`runs/eval_<adapter>_<timestamp>.json`)
unless you override it. Run `./eval.sh --help` for the full flag list.

### 7. `./convert.sh` — convert an adapter for iOS

```bash
./convert.sh adapters_v5                                # -> adapters_v5_swift/
./convert.sh adapters_v6 --version v6                    # explicit version stamp
```

Wraps `convert_adapter_for_swift.py`, converting an mlx-vlm LoRA adapter
into the format `mlx-swift-lm`'s `LoRAContainer` loads (key renaming
`.A`/`.B` → `.lora_a`/`.lora_b`, scale-convention normalization between the
legacy and current mlx-vlm stacks — see the script's docstring). It also
stamps a `"seecal_adapter_version"` key into the output
`adapter_config.json` — the exact key iOS's `ModelInfoResolver` looks for
first (see
`ios/SeeCal/Sources/SeeCalInference/ModelInfoResolver.swift`), before
falling back to parsing a trailing `_v<N>` off the directory name. The
version defaults to that same `_v<N>` parse of `--adapter-path`'s directory
name (e.g. `adapters_v5` → `v5`); override with `--version`. Output
defaults to `<adapter-path>_swift`; override with `--output-path`.

## The validation ladder

Never start a multi-hour training run without climbing this ladder in
order — skipping straight to a long run is exactly how a past training run
(the "v4" adapter) died mid-run and went unnoticed dead for most of a day:

1. **`./prep.sh`** (rung 1, automated): builds the JSONL and runs
   `smoke_test.py` — ~1 minute, no model weights beyond the
   processor/config. Catches every silent failure mode that has previously
   cost multi-day training runs: a missing top-level `images` column, absent
   `pixel_values` (vision encoder never running), collapsed/truncated
   sequences, image-token-count mismatches, and a completion mask that's
   empty or leaks onto image tokens.
2. **A 32-dish overfit run** (manual): confirms the model can memorize a
   tiny set — i.e. gradients and masking are wired correctly through actual
   weight updates, not just batch construction.
3. **A short probe run** (manual, ~500 iterations): evaluate with
   `./eval.sh` against the un-adapted base model as a baseline before
   committing to a full run.

Only after all three are green should `./train.sh` run unattended for
hours.

## Expected metrics

MAE / median absolute error on a 50-dish held-out sample (`finetune_data_v2/test.jsonl`,
`infer.py`/`eval.sh`). The full 325-dish test-set evaluation is tracked as a
follow-up (see the open-source reorg plan, task R7) — treat these 50-dish
numbers as directionally reliable, not final.

| Model                              | Calories MAE | Calories median | Parse failures (/50) |
|-------------------------------------|-------------:|----------------:|----------------------:|
| Base model (no adapter)             |         83.4 |            63.9 |                      0 |
| `adapters_v5` (shipping)            |         54.4 |            36.2 |                      1 |
| `adapters_v6` (depth-augmented, text)  |     59.2 |            44.7 |                      0 |

`adapters_v6` adds a depth-sensor volume line to the prompt (the depth
track's variant B, see `docs/design/2026-07-26-depth-design-brief.md`).
Its 1000-iteration probe beat the matched v5 probe by 33% calories MAE, but
the full 2-epoch run came back a statistical tie with v5 (50 dishes paired:
+5.2 kcal, t=0.56) — not a real improvement. **The depth track is stopped;
`adapters_v5` remains the shipping adapter**, and `adapters_v6` is kept only
for reference. See `runs/eval_v5/`, `runs/eval_v6/`, and
`runs/eval_v4_baseline/` for the raw per-sample data behind this table, and
AGENTS.md for the complete run-by-run history (including the four broken
training runs that preceded v5).

## Hardware requirements

- **Apple Silicon required** — mlx / mlx-vlm are Metal-only; there is no
  CUDA/CPU fallback path in this repo.
- **Training**: ~3.5 hours for the full 2-epoch v5-style run on an M3
  Ultra, peaking **47-58GB of Metal memory** (without `--grad-checkpoint`;
  add it back for ~13GB peak at ~15% slower throughput — see `train.sh
  --help`).
- **Co-tenancy warning**: Metal memory does not swap. Running anything else
  GPU-heavy during training (LM Studio, `eval.sh`, a second training run)
  competes for the same memory and silently OOM-kills the training process
  instead of failing with a clear error. Don't share the machine with
  another Metal workload while `./train.sh` is running.
- **Inference/eval**: comfortably fits alongside normal desktop use; the
  4-bit base model + LoRA adapter is far smaller than the training peak.
- **Disk**: ~3-4GB for the dataset subset (`./download_dataset.sh`
  default), ~181GB for `--full`; a few GB for the base model
  (`./download_model.sh`); LoRA adapters are small (tens of MB).

## Troubleshooting

- **`ModuleNotFoundError` / processor init fails mentioning `torchvision`**:
  the Qwen3VL processor imports `torchvision` at init time, but mlx-vlm's
  own `[train]` extra does not install it. `./setup.sh` installs
  torch/torchvision as an explicit separate step for exactly this reason —
  re-run it, or `ml/.venv/bin/pip install torch torchvision` directly. The
  failure mode reads as "hanging" or a confusing late traceback rather than
  a clean missing-dependency error, because it happens well after argument
  parsing and dataset loading look fine.
- **`ArrowInvalid: Column(/messages/[]/content) changed from string to
  array in row 0`**: pyarrow (via HuggingFace `datasets`, used inside
  mlx-vlm) requires every row's `messages[].content` to have the *same*
  type across the whole JSONL. `prepare_finetune.py` already produces
  array-typed content for every role to avoid this — if you hand-edit a
  JSONL file, keep every `content` field an array, never a bare string.
- **Training run seems to hang or die overnight with no error, machine
  felt sluggish beforehand**: almost certainly Metal-memory co-tenancy (see
  Hardware requirements above) — another GPU-heavy process (commonly LM
  Studio) competed for memory and the training process was OOM-killed
  without a clean error. Check `<OUTPUT_PATH>_train.log`'s last timestamp
  against when the machine was otherwise in use, and close every other
  Metal workload before restarting.
- **`eval.sh` refuses to run with "`--limit` is REQUIRED"**: this is
  deliberate — see the `eval.sh` section above. Pass an explicit `--limit`
  (325 for the full committed test set).
- **`smoke_test.py` / `prep.sh` fails to load a processor**: `--model-dir`
  (or `prep.sh --model-dir`) must point at a real, fully-downloaded model
  directory with a `config.json` and tokenizer files — run
  `./download_model.sh` first, or point `MODEL_PATH` at an existing local
  copy.
