# SeeCal — ML Pipeline

Trains SeeCal's compact visual specialist and a conditioned Qwen3.5-4B LoRA
to estimate calories, macros, and ingredients from a single food photo. All
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

## Factored-pipeline track (not shipping yet)

The approved migration in
`../docs/specs/2026-07-29-factored-pipeline-design.md` separates IDENTIFY,
SCALE, RESOLVE, and ASSEMBLE. Its Stage-0/1 commands are:

```bash
./download_fdc.sh
.venv/bin/python download_foodseg103.py
./download_nutritionverse_real.sh
.venv/bin/python make_nutritionverse_eval.py
.venv/bin/python make_alias_table.py \
  --database datasets/fdc/seecal-nutrition.sqlite \
  --reviewed-aliases factored_pipeline/training_aliases_v1.tsv \
  --misses datasets/fdc/training-alias-misses-v1.tsv
.venv/bin/python make_alias_table.py \
  --database datasets/fdc/seecal-nutrition.sqlite \
  --reviewed-aliases factored_pipeline/nutritionverse_train_aliases_v1.tsv \
  --reviewed-source reviewed_nutritionverse_official_train_v1 \
  --misses datasets/fdc/nutritionverse-train-alias-misses-v1.tsv
.venv/bin/python make_identify_data.py \
  --foodseg-root datasets/foodseg103 \
  --frb-teacher-manifest \
    datasets/food_recognition_2022/derived/pilot-5000/pilot-5000-enriched.jsonl \
  --frb-require-fully-resolved \
  --resolver-database datasets/fdc/seecal-nutrition.sqlite \
  --max-image-edge 1024 \
  --output-dir finetune_data_id_v2_frb_resolved
.venv/bin/python check_prompt_parity.py \
  --data finetune_data_id_v2_frb_resolved/train.jsonl --records 8
.venv/bin/python smoke_test.py \
  --data finetune_data_id_v2_frb_resolved/train.jsonl --records 8 --batch-size 4
.venv/bin/python make_scale_v2_data.py --output-dir datasets/scale_v2
.venv/bin/python -m visual_specialist.scale train \
  --manifest-dir datasets/scale_v2 \
  --output-dir runs/factored/scale-v2-probe \
  --sampling source-group-balanced --ordered-quantiles
```

`score_harness.py {audit,resolve,mass}` runs the paired E0/E1/E2 contract.
`identify_infer.py` is the new-schema evaluation path once an IDENTIFY
adapter exists. Generated data, USDA sources/database, adapters, and run
artifacts are intentionally gitignored. The current v8 adapter and specialist
remain the build/release selection until every gate in the design passes.
`download_foodseg103.py` pins the Apache-2.0 Hugging Face mirror at revision
`34e1208e14bc3595d544fc8c3f3c6673253fd9ef`, records the four source-object
SHA-256 values, and materializes the 4,983/2,135 image-mask splits locally.
The upstream project archive returned HTTP 502 during execution; the pinned
mirror contains the same 7,118-record dataset and exact 104-label mask schema.
The FNDDS release uses legacy nutrient numbers (203/204/205) in its
`nutrient_id` column. `make_fdc_db.py` intentionally accepts both those and
the modern 1003/1004/1005 IDs; the regression test prevents silently dropping
all FNDDS composites again. The resulting database has 13,589 profiles and
196 category medians before reviewed aliases.
`download_nutritionverse_real.sh` pins the public Kaggle dataset to v2 and
`make_nutritionverse_eval.py` derives IDENTIFY, monolith-baseline, and SCALE
manifests for all 889 OOD images while keeping the 225 dish IDs explicit for
paired aggregation.

SCALE-v2 keeps licensing boundaries explicit. The default manifest uses only
permissive Nutrition5K data. For the free, non-commercial SeeCal track,
official NutritionVerse Train and Food Portion Benchmark can be included while
their held-out data stays untouched:

```bash
.venv/bin/hf download issai/Food_Portion_Benchmark \
  --repo-type dataset \
  --revision 53fcacf4b9dbe24c1c6ffa5a2cdb9d8c502e482f \
  --include 'FPB_Dataset/RGB/train/**' \
  --include 'FPB_Dataset/RGB/val/**' \
  --local-dir datasets/food-portion-benchmark
.venv/bin/python make_fpb_scale_data.py \
  --dataset-root <pinned-snapshot-or-local-dir> \
  --skip-incomplete-weights \
  --exclude-test-capture-overlap
.venv/bin/python make_scale_v2_data.py \
  --nutritionverse-manifest datasets/nutritionverse-real/scale-eval-v2-1024.jsonl \
  --include-noncommercial-training \
  --fpb-manifest-dir datasets/food-portion-benchmark/scale \
  --output-dir datasets/scale_v2_nc_fpb_1024
```

Before training on FPB, keep its official test split frozen and score the
current checkpoint zero-shot. The converter reports and excludes records with
explicit unknown (`-1`) object weights instead of summing partial targets:

```bash
.venv/bin/python make_fpb_scale_data.py \
  --dataset-root <pinned-snapshot-or-local-dir> \
  --output-dir datasets/food-portion-benchmark/scale-zero-shot \
  --splits test --skip-incomplete-weights --grouping food-size
.venv/bin/python -m visual_specialist.scale evaluate \
  --checkpoint runs/factored/scale-v2-probe-b-nv-1024/best.pt \
  --manifest datasets/food-portion-benchmark/scale-zero-shot/test.jsonl \
  --output runs/factored/scale-v2-probe-b-nv-1024/eval-fpb-test-zero-shot.json \
  --include-all-views
```

The frozen result is 2,123 usable images / 164 groups, 165.7 g equal-group
MAE, 55.3% MAPE, and 20.9% interval coverage. FPB train/validation therefore
belongs in the next representation-training probe; simple point calibration
is not sufficient.

Training manifests use original-capture groups rather than the legacy
food-size benchmark groups. They contain 8,929 clean train records / 8,404
groups and 2,170 clean validation records / 2,170 groups. Two transformed train
images derived from captures also present in official test are excluded. FPB
official validation stays in SCALE validation; it is never folded into
training.

The zero-training follow-up audits are reproducible:

```bash
.venv/bin/python scale_occupancy_audit.py
.venv/bin/python portion_prior_audit.py
```

The occupancy audit uses depth food footprint for Nutrition5K, COCO masks for
NutritionVerse, and YOLO box union for FPB. It reports raw, current-center-crop,
and simulated-letterbox correlations plus group-level mass support. The portion
audit uses oracle names/shares and verifies that raw USDA typical servings are
not safe as a production fallback.

Probe C uses two matched arms after FPB train/validation are available. Both
use `--checkpoint-selection minimax-source-mape --pareto-tolerance 0.10`;
C1 uses `--geometry center-crop`, and C2 changes only that flag to
`--geometry letterbox`. Checkpoints record their input geometry, and evaluation
uses it automatically unless explicitly overridden.

Probe C is complete. C1 is selected: it reaches 40.1 g Nutrition5K overhead
MAE, 42.0 g Nutrition5K side equal-group MAE, 99.0 g NutritionVerse
official-Val equal-scene MAE, and 73.3 g FPB equal-group MAE. C2 improves the
two Nutrition5K values to 37.5/41.0 g but is worse on NutritionVerse (100.3 g)
and FPB (80.0 g), so it is rejected by worst-source MAPE. C1 repairs FPB
ordering from 23/40 to 37/40 complete size triads; reproduce with:

```bash
.venv/bin/python scale_ordering_audit.py \
  runs/factored/scale-v2-probe-b-nv-1024/eval-fpb-test-zero-shot.json \
  runs/factored/scale-v2-probe-c1-fpb-center/eval-fpb-test-calibrated.json \
  runs/factored/scale-v2-probe-c2-fpb-letterbox/eval-fpb-test-calibrated.json
```

NutritionVerse official Val is always a frozen test split. Related views are
grouped before splitting, and source/group-balanced sampling prevents the
39k-view Nutrition5K rig from drowning out the smaller phone-photo sources.
These datasets and their derived manifests are gitignored and are never
bundled with the app.

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
ADAPTER_PATH=adapters_v7 OUTPUT_PATH=adapters_v7-cont ITERS=1000 \
  ./resume_train.sh                                    # guarded warm-start
```

Configured entirely through environment variables (no positional args) —
`MODEL_PATH` (base model), `OUTPUT_PATH` (adapter output dir, **pick an
explicit versioned name** like `adapters_v7` for any run that isn't a
deliberate re-run of the existing default), and `DATASET`
(default `finetune_data_v2`). `train.sh` intentionally rejects
`ADAPTER_PATH`: mlx-vlm 0.6.7's built-in resume path failed to freeze the base
model and exposed 366.9M trainable parameters instead of the 32.464896M LoRA
parameters.

For a saved adapter, use `resume_train.sh` with a **different**
`OUTPUT_PATH` and an explicit number of continuation `ITERS`. It freezes the
base model before loading the LoRA weights and still runs through the
trainable-parameter guard. This is a warm-start, not an exact resume:
mlx-vlm's adapter checkpoints do not contain Adam optimizer state or the
random data-shuffle position.

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

## Current baselines and device pilot

The default real-world device-test build now selects the factored pipeline:
E3 FoodSeg-trained Qwen IDENTIFY, SCALE C1, the local USDA resolver, and
deterministic assembly. Its frozen 354-image IDENTIFY evaluation has zero parse,
schema, repair, or post-repair rejection failures, 29/29 correct non-food
refusals, and zero false refusals. On the shared clean 72-dish Nutrition5K
slice, the assembled candidate is 41.0 kcal MAE versus its 38.5 kcal
current-SCALE oracle; this is an experimental field pilot, not evidence of
real-phone accuracy.

The protected `v8-conditioned` rollback runs the S1 MobileNetV3 specialist
before Qwen3.5-4B. On all 325 untouched test dishes it scores 56.8 kcal MAE /
29.7 median with zero parse failures; it also refuses 29/29 held-out non-food
images. Select it with `SEECAL_MODEL_STACK=v8 scripts/build.sh --device`. See
`../docs/plans/2026-07-28-visual-specialist-conditioned-qwen-plan.md` and
`visual_specialist/README.md` for the complete pipeline and artifact contract.

## Historical 50-dish metrics

MAE / median absolute error on a 50-dish held-out sample (`finetune_data_v2/test.jsonl`,
`infer.py`/`eval.sh`). The full 325-dish test-set evaluation is tracked as a
follow-up (see the open-source reorg plan, task R7) — treat these 50-dish
numbers as directionally reliable, not final.

| Model                              | Calories MAE | Calories median | Parse failures (/50) |
|-------------------------------------|-------------:|----------------:|----------------------:|
| Base model (no adapter)             |         83.4 |            63.9 |                      0 |
| `adapters_v5` (historical food baseline) |    54.4 |            36.2 |                      1 |
| `adapters_v6` (depth-augmented, text)  |     59.2 |            44.7 |                      0 |

`adapters_v6` adds a depth-sensor volume line to the prompt (the depth
track's variant B, see `docs/design/2026-07-26-depth-design-brief.md`).
Its 1000-iteration probe beat the matched v5 probe by 33% calories MAE, but
the full 2-epoch run came back a statistical tie with v5 (50 dishes paired:
+5.2 kcal, t=0.56) — not a real improvement. **The depth track is stopped.**
`adapters_v6` is kept only for reference. See `runs/eval_v5/`, `runs/eval_v6/`, and
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
