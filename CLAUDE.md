# SeeCal — Calorie Detection from Food Images

## Goal
Fine-tune Qwen3.5-4B (natively multimodal) on the Nutrition5K dataset to estimate calories, macros, and ingredients from a single food photo.

## Model
- **Qwen3.5-4B-MLX-bf16** — from `mlx-community/Qwen3.5-4B-MLX-bf16` on HuggingFace
- Downloaded to `~/models/Qwen3.5-4B-MLX-bf16`
- Qwen3.5 is natively multimodal (no separate VL models) — vision is built in
- Fine-tuning uses **mlx-vlm** (NOT mlx-lm, which doesn't support Qwen3.5 architecture)
- mlx-vlm installed from git main: `pip install git+https://github.com/Blaizzy/mlx-vlm.git`
- Requires Python >= 3.10 and `torchvision` (needed by the Qwen3VL processor)

## Dataset: Nutrition5K
- Source: `Nutrition5K/` directory
- ~4,793 dishes with multiple camera angles per dish
- `dish_nutrition_values.csv` — calories, mass, fat, carb, protein per dish (4,768 rows)
- `dish_ingredients.csv` — per-ingredient breakdown with name, grams, calories, macros (27,225 rows, ~5.7 ingredients/dish)
- `ingredients_metadata.csv` — ingredient lookup table
- Imagery: `imagery/realsense_overhead/` (rgb.png, depth) + `imagery/side_angles/` (4 cameras × 10 frames)

## Pipeline Scripts

### `01_select_images.py`
Selects up to 3 images per dish and copies to `dataset_clean/`:
- `overhead.jpg` — always included (top-down, best real-world match)
- `side_a.jpg` — camera A, frame 5 (if --max-images >= 2)
- `side_c.jpg` — camera C, frame 5 (if --max-images >= 3)
- Depth images excluded (not useful for deployment)
- Use `--max-images 1` for overhead-only (matches deployment scenario)

Stats from full run: 3,472 dishes processed, 9,445 images total (~2.7 avg/dish), 18 skipped.

### `02_prepare_finetune.py`
Generates train/valid/test JSONL for mlx-vlm LoRA fine-tuning:
- Loads both nutrition CSV and ingredients CSV
- One record per image (not per dish) — 3 images = 3 training examples
- Split by dish (80/10/10), then expanded to per-image records
- Image paths stored **relative to --out-dir** (portable) or absolute via `--image-root`
- Output: `finetune_data/train.jsonl`, `valid.jsonl`, `test.jsonl`

Stats: 3,262 matched dishes → ~7,091 train / 885 valid / 887 test records.

#### JSONL format (mlx-vlm compatible)
ALL content fields are arrays (system, user, assistant) — this is required because pyarrow (used
by HuggingFace datasets inside mlx-vlm) demands a consistent schema across all rows. Mixed types
(string for system/assistant, array for user) trigger `ArrowInvalid: Column(/messages/[]/content)
changed from string to array in row 0`.

**No system message** — Qwen3.5's Jinja2 chat template raises "System message cannot contain
images." for ANY list/array content in system role, even a text-only array. Since pyarrow forces
uniform types and the Jinja2 template forbids arrays in system role, we drop the system message
entirely and fold the system instruction into the user text (same approach mlx-vlm's Qwen
auto-formatter uses). Inference must match — no system block, instruction in user prompt.

```json
{
  "messages": [
    {"role": "user",      "content": [
      {"type": "image", "image": "../dataset_clean/dish_xxx/overhead.jpg"},
      {"type": "text",  "text": "You are a nutrition expert...\n\nLook at this meal and identify..."}
    ]},
    {"role": "assistant", "content": [{"type": "text",  "text": "{\"calories\": 246.0, ...}"}]}
  ]
}
```

The `{"type": "image"}` entry causes Qwen3.5's processor to insert real vision tokens
(`<|vision_start|><|image_pad|><|vision_end|>`) during training — matching inference exactly.

The assistant response includes both nutrition totals and ingredient list (sorted by grams descending):
```json
{"calories": 246.0, "protein_g": 33.1, "fat_g": 5.6, "carbs_g": 16.3, "mass_g": 332.0, "ingredients": [{"name": "broccoli", "grams": 129.0}, {"name": "chicken breast", "grams": 89.5}, ...]}
```

## Training Command
```bash
python -m mlx_vlm.lora \
  --model-path ~/models/Qwen3.5-4B-MLX-bf16 \
  --dataset finetune_data \
  --train-mode sft \
  --train-on-completions \
  --lora-rank 16 \
  --lora-alpha 32 \
  --batch-size 4 \
  --iters 1000 \
  --output-path adapters
```

Key flags:
- `--model-path` (not --model) — mlx-vlm syntax
- `--dataset` (not --data) — points to directory with train.jsonl + valid.jsonl
- `--train-mode sft` — supervised fine-tuning (LoRA is always on; sft is the objective)
- `--train-on-completions` — loss only on assistant response tokens, not prompt
- No `--train-vision` — keeps vision encoder frozen, only tunes language layers
- `--lora-rank 16` — replaces mlx-lm's --num-layers concept

## Training Results
### Run 2 (BROKEN — training without images, do not use `adapters_v2/`)
- LoRA training completed (1000 iterations) with mlx-vlm
- Adapters saved to `adapters_v2/`
- **Bug**: `use_embedded_images = True` for qwen in mlx-vlm's `datasets.py` caused `images=None`
  to always be passed to `prepare_inputs`. The vision encoder never ran during training.
  No `pixel_values` in any training batch → LoRA adapters learned without image features.
  Inference output: `<think>\n\n</think>\n\n<|vision_start|><|image_pad|><|vision_end|>` (parse error).
- **Fix**: Patched `.venv/lib/python3.14/site-packages/mlx_vlm/trainer/datasets.py`:
  1. Added `_extract_images_from_content()` helper that loads PIL Images from
     `{"type": "image", "image": path}` entries in message content arrays.
  2. In `VisionDataset.process()`: when no top-level images, call the helper on `conversations`.
  3. Removed `qwen` from `use_embedded_images` so loaded PIL images are passed to `prepare_inputs`.
  **Retrain required with `adapters_v3/`.**

### Run 1 (BROKEN — wrong format, do not use `adapters/`)
- LoRA training completed (1000 iterations) with mlx-vlm
- Adapters saved to `adapters/` directory (adapters.safetensors + checkpoints every 100 iters)
- 496 adapter weight keys across all 32 model layers
- **Bug**: training JSONL used `<image>` placeholder text + top-level `images` field; the `<image>`
  token was tokenised as literal text rather than converted to vision tokens. LoRA adapters learned
  a text→JSON mapping without any visual grounding. Inference with the adapter produced garbled
  repetition of the system prompt. Confirmed: base model (no adapter) gives valid JSON output.
- **Fix**: `02_prepare_finetune.py` updated and `finetune_data/` regenerated with correct format.
  Retrain needed.

### `03_infer.py`
Inference script using mlx-vlm with LoRA adapter (no fusing needed):
- Single image: `python 03_infer.py --image food.jpg`
- Test set evaluation: `python 03_infer.py --test-set finetune_data/test.jsonl --limit 20`
- Computes MAE for calories, protein, fat, carbs

### `fix_adapter_for_fuse.py`
Patches mlx-vlm adapter output for mlx-lm fuse compatibility:
- Adds missing `num_layers` field to `adapter_config.json`
- Strips `language_model.` prefix from weight keys (mlx-lm expects `model.layers.N...`)
- Outputs to `adapters_fixed/` directory

Fuse command after fix:
```bash
python fix_adapter_for_fuse.py --adapter-path adapters --output-path adapters_fixed
python -m mlx_lm.fuse \
  --model ~/models/Qwen3.5-4B-MLX-bf16 \
  --adapter-path adapters_fixed \
  --save-path ./fused-model
```

## Issues Resolved
1. **mlx-lm doesn't support qwen3_5** — use mlx-vlm instead
2. **Python 3.9 too old** — mlx-vlm requires >= 3.10, recreate venv
3. **Missing torchvision** — Qwen3VLVideoProcessor needs PyTorch+torchvision for processor init
4. **pyarrow schema error** — `ArrowInvalid: Column(/messages/[]/content) changed from string to
   array`. pyarrow requires all messages[].content to be the same type. Fix: make ALL content
   fields arrays (user and assistant). This also matches mlx-vlm's own Qwen auto-formatter.
5. **System message cannot contain images** — Qwen3.5's Jinja2 chat template raises this error
   for ANY list/array content in system role (even text-only arrays). Combined with constraint #4
   (pyarrow needs uniform types), system must be a string but user must be an array — impossible.
   Fix: drop system message entirely, fold system instruction into user message text. This is what
   mlx-vlm's own Qwen format does (lora.py lines 94-104). Inference updated to match.
6. **Ambiguous --train flag** — mlx-vlm has --train-vision, --train-on-completions, --train-mode;
   use full flag names
7. **--train-mode lora invalid** — valid choices are `sft` or `orpo`; LoRA is always enabled
8. **mlx-lm fuse AttributeError num_layers** — mlx-vlm adapter_config.json only has
   rank/alpha/dropout; mlx-lm fuse expects num_layers + weight keys without `language_model.`
   prefix; fixed by `fix_adapter_for_fuse.py`
9. **Vision tower missing from fused model** — `mlx_lm.fuse` only handles language weights; vision
   tower weights are dropped. `patch_fused_vision.py` can copy them back, but naming mismatches
   between mlx-lm and mlx-vlm conventions left 96 params missing. Practical solution: use base
   model + adapter via mlx-vlm (no fusing needed for inference).
10. **Training format bug (root cause of garbled output)** — `<image>` in JSONL was tokenised as
    literal text (not vision tokens); LoRA adapters learned text→JSON without visual grounding.
    Fixed by using `[{"type": "image", "image": path}, ...]` array content in user message.
    Dataset regenerated. **Retrain required.**
11. **GenerationResult object** — mlx-vlm generate() returns GenerationResult, not a string; access
    `.text` attribute.
12. **prompt_utils vs processor.apply_chat_template** — use `mlx_vlm.prompt_utils.apply_chat_template`
    for inference (inserts real vision tokens); `processor.apply_chat_template(tokenize=False)` keeps
    `<image>` as literal text.
13. **Qwen3.5 thinking mode** — chat template appends `<think>\n` to assistant turn; strip it for
    direct JSON output in inference.
14. **mlx-vlm `use_embedded_images` bug for Qwen** — `datasets.py` sets `use_embedded_images=True`
    for any `model_type.startswith("qwen")`, which causes `images=None` to always be passed to
    `prepare_inputs`. The vision encoder never runs, `pixel_values` is absent from every training
    batch, and LoRA adapters learn a degenerate text-only mapping. Inference after such training
    outputs raw vision tokens (`<|vision_start|><|image_pad|><|vision_end|>`) with no JSON.
    Fix: patch `datasets.py` — add `_extract_images_from_content()` helper that loads PIL Images
    from `{"type": "image", "image": path}` entries, call it in `process()` when no top-level
    images field exists, and remove `qwen` from the `use_embedded_images` condition.
15. **PyArrow None-fill causes double `<|image_pad|>` → `image_grid_thw` IndexError** —
    HuggingFace datasets uses PyArrow to load JSONL. Because our content arrays have mixed-key
    dicts (image entries have `"image"` key; text entries have `"text"` key), PyArrow normalises
    to a unified schema by filling `None` for every missing key. After loading, a text entry
    looks like `{"type": "text", "image": None, "text": "..."}`. The Qwen3VL Jinja2 template
    checks `'image' in item` (key presence, not value), so this text entry is also treated as
    an image → two `<|image_pad|>` tokens in the prompt. `prepare_inputs` only has one PIL image
    → `image_grid_thw[1]` IndexError: `index 1 is out of bounds for dimension 0 with size 1`.
    Fix: patch `datasets.py` — add `_normalize_conversation()` helper that strips None-valued
    keys from content dicts, and call it at the start of `get_prompt()` before
    `apply_chat_template`. This restores the original semantics (text entries have no `"image"`
    key) so the template emits exactly one `<|image_pad|>` per actual image.
16. **`mx.stack` fails for Qwen3VL dynamic tiling** — Qwen3VL tiles images based on resolution;
    different images produce different numbers of patches → different `pixel_values` shapes →
    `[stack] All arrays must have the same shape` in `sft_trainer.py:204`.
    Fix: patch `sft_trainer.py` — in `iterate_batches`, try `mx.stack` first; if shapes differ
    fall back to `mx.concatenate(squeezed_pv, axis=0)`. Qwen3VL models are designed to receive
    concatenated patches across the batch; `image_grid_thw` (batched as `[batch, 3]`) tells the
    model how many patches belong to each image. Also applied same try/fallback to `extra_keys`
    loop so `image_grid_thw` concatenates cleanly if needed.
17. **`len(input_ids)` returns 1 instead of seq_len — batch truncated to 33 tokens** —
    `prepare_inputs` returns `input_ids` with shape `(1, seq_len)` (batch dim = 1). In
    `iterate_batches`, `lengths = [min(len(x["input_ids"]), max_seq_length) for x in items]`
    calls `len()` on this 2-D array, which returns the first dimension = 1, not the true
    sequence length. As a result `max_len=1`, `padded_len=33`, and every sequence is sliced
    to 33 tokens. Image tokens (300–2040 per image) start around position 5 and almost all
    are silently discarded. Only the first ~28 `<|image_pad|>` tokens survive per sequence
    (4 × 28 = 112 in the error message) while the vision tower correctly processes all
    patches (4 × 1170+ = 4680 / 6420 features) → mismatch.
    Confirmed by debug prints: per-item `input_ids shape: (1, 2346)` with 2040 image tokens,
    but batched `input_ids: (4, 33)` with only 116 image tokens total.
    Fix: replace `len(x["input_ids"])` with `np.array(x["input_ids"]).reshape(-1).shape[0]`
    so the true flat sequence length is used regardless of leading batch dimension.
18. **`--assistant-id` default wrong for Qwen3.5 → `--train-on-completions` silently disabled** —
    mlx-vlm's default `assistant_id=77091` decodes to `'[hash'` in Qwen3.5's tokenizer. Because
    this token never appears in any training sequence, `np.where(row == 77091)` always returns
    empty, `assistant_response_index` stays at -1, and `range_matrix <= -1` is always False →
    `weight_mask` stays all-ones → loss is computed over the full 600-token sequence (300 image
    tokens + repeated question prompt + JSON response) instead of just the ~250-token JSON
    completion. The model learned to predict image tokens and its own question, collapsing to the
    most frequent ingredient pattern (olive oil appears in 48% of Nutrition5K dishes). All three
    previous training runs (adapters/, adapters_v2/, adapters_v3/) were affected by this bug.
    Confirmed: `processor.tokenizer.encode('assistant') = [74455]`.
    Fix: pass `--assistant-id 74455` to all mlx_vlm.lora training commands for Qwen3.5.
19. **`lora.py` crashes when `--iters` > dataset size** — `dataset.select(range(iters))` raises
    `IndexError` when `iters > len(dataset)`. The `iterate_batches` function already has a
    `while True:` loop that cycles through the dataset for multi-epoch training, so the select
    only needs to cover up to `len(dataset)` examples.
    Fix: change to `dataset.select(range(min(iters, len(dataset))))`.

## Next Steps
1. **Retrain** with `adapters_v4/` using the corrected command below (all 5 patches + correct assistant-id).
2. **Evaluate** on test split (325 records, overhead-only) using `03_infer.py --test-set finetune_data/test.jsonl`
3. **iPhone deployment**: fused model → CoreML conversion (or MLX Swift on-device)
4. Consider `--train-vision` if accuracy plateaus (unfreezes vision encoder, more expensive)

### Retrain Command (use this — all patches applied including --assistant-id fix)
```bash
python 02_prepare_finetune.py --only-overhead  # regenerate dataset_clean/ first if needed

python -m mlx_vlm.lora \
  --model-path /Users/jevgenikabanov/.lmstudio/models/mlx-community/Qwen3.5-4B-MLX-4bit \
  --dataset finetune_data \
  --train-mode sft \
  --train-on-completions \
  --assistant-id 74455 \
  --lora-rank 16 \
  --lora-alpha 32 \
  --batch-size 1 \
  --max-seq-length 1024 \
  --learning-rate 1e-5 \
  --grad-checkpoint \
  --iters 13000 \
  --output-path adapters_v4
```

**Why adapters_v3 failed**: `--train-on-completions` was silently disabled in all previous runs
because `--assistant-id` defaulted to 77091 (`'[hash'` token), which never appears in Qwen3.5
sequences. Loss was computed over all 600 tokens (300 image tokens + prompt + JSON), causing
mode collapse to the most frequent ingredient pattern (olive oil, 48% of dishes).
