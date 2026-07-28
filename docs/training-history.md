# Training history: from broken adapters to conditioned v8

This is the forensic record of every training run and bug that predates the
current, working pipeline (`ml/setup.sh` → `ml/prep.sh` → `ml/train.sh` →
`ml/eval.sh` → `ml/convert.sh`, documented in `ml/README.md`). It exists so
the next person who hits a familiar-looking failure can find out whether it's
new or a rediscovery. Nothing here is required reading to use the pipeline —
start with `ml/README.md` instead.

## Adapter runs, in order

### Run 1 — `adapters/` (BROKEN: no visual grounding, wrong JSONL format)
Completed 1000 iterations, 496 adapter weight keys across all 32 layers. The
training JSONL used a literal `<image>` placeholder plus a top-level `images`
field; the `<image>` token was tokenized as ordinary text rather than expanded
into vision tokens, so the adapter learned a text→JSON mapping with zero
visual grounding. Inference produced garbled repetition of the system prompt.
Confirmed by contrast: the un-adapted base model gave valid JSON on the same
inputs. **Fix**: rewrite the JSONL generator to emit
`[{"type": "image", "image": path}, ...]` array content in the user message
(fix #10 below). Do not use `adapters/`.

### Run 2 — `adapters_v2/` (BROKEN: trained with no images at all)
Completed 1000 iterations against the corrected JSONL format above — but still
failed. mlx-vlm's `datasets.py` set `use_embedded_images = True` for any Qwen
model, which meant `images=None` was always passed to `prepare_inputs`
regardless of the JSONL's content. The vision encoder never ran; no
`pixel_values` ever appeared in a training batch. Inference output was
`<think>\n\n</think>\n\n<|vision_start|><|image_pad|><|vision_end|>` — a parse
failure with visible vision-token placeholders and no JSON. **Fix**: patched
`.venv/.../mlx_vlm/trainer/datasets.py` to extract PIL images from the
content-embedded `{"type": "image", ...}` entries and stopped special-casing
Qwen out of receiving them (fix #14 below). Do not use `adapters_v2/`.

### Run 3 — `adapters_v3/` (BROKEN: completion masking silently disabled)
Retrained after fixes #10 and #14 landed, so images finally reached the vision
encoder — but `--train-on-completions` had no effect, because mlx-vlm's
default `--assistant-id` (77091) decodes to a token that never appears in a
Qwen3.5 sequence. The loss mask was therefore all-ones: the model was trained
to predict the entire 600-token sequence (image tokens + repeated prompt +
JSON), not just the ~250-token completion. It collapsed to the single most
frequent ingredient pattern in the dataset (olive oil, present in 48% of
Nutrition5k dishes) regardless of the input photo. **Fix**: pass
`--assistant-id 74455` (the real `'assistant'` token id for Qwen3.5's
tokenizer) explicitly (fix #18 below). Do not use `adapters_v3/`.

### Run 4 — `adapters_v4/` (INCOMPLETE: died mid-run, unverified until forensics)
Launched via `train.sh` with `--batch-size 8 --iters 1000 --output-path
adapters` (a bare, non-versioned name — checkpoints landed in the repo root
and were later moved to `runs/v4_root_output/`). No training log survives;
file timestamps show it died somewhere between iteration 800 and 900, roughly
consistent with an unattended overnight run that crashed or was killed without
anyone noticing. `adapters_v4/adapters.safetensors` is a hand-recovered
iteration-800 checkpoint, not the output of a completed run. It also inherited
a second bug: `--iters 1000` combined with a `dataset.select(range(iters))`
clamp (fix #19) meant training only ever iterated the first 1000 of 2594
available examples — under one epoch, never reaching the back 60% of the
data — independent of dying early. **Verdict** (file forensics + eval,
2026-07-26): 50/50 parse failures on a 50-dish held-out set via `infer.py`
— the adapter is dead. The base model (no adapter) on the same 50 dishes:
calories MAE 83.4 / median 63.9, protein MAE 6.7, fat MAE 4.9, carbs MAE 12.4,
0 parse failures — this became the number v5 needed to beat. Do not use
`adapters_v4/` for anything beyond historical reference.

### The toolchain rewrite, and `adapters_v5/` (historical food baseline)
Runs 1–4 all trained against a hand-patched, ancient mlx-vlm checkout on a
Python 3.14 venv. On 2026-07-26 a second, unpatched toolchain
(`mlx-vlm==0.6.7`) was stood up and the legacy one retired outright (see
"Toolchain rewrite" below) — every legacy patch (fixes #14–17) turned out to
already be fixed upstream. `adapters_v5` trained clean on the new stack, 2
epochs, LR 1e-4: calories MAE 54.4 / median 36.2 on the same 50-dish set,
1/50 parse failures. It became the first working baseline and later the food
control for v7/v8 — see `ml/README.md` for metrics and reproduction.

### The depth track, and `adapters_v6/` (reference only, not shipped)
A parallel effort added a depth-sensor volume estimate to the prompt
(`ml/depth_features.py`, glass-platform-corrected geometry). The image-variant
(feeding a depth map as a second image) was blocked entirely — mlx-vlm's
trainer collation is broken for multi-image records on both 0.6.7 and git
main (reported upstream). The text-variant (`adapters_v6`, a depth volume
line appended to the prompt) looked promising at the 1000-iteration probe
(33% better calories MAE than the matched v5 probe) but came back a
statistical tie with v5 on the full 2-epoch run (50 dishes paired: +5.2 kcal,
t=0.56; v6 59.2 vs v5 54.4 MAE) — not a real improvement. **The depth track is
stopped.** `adapters_v6` is kept
for reference only. See `docs/design/2026-07-26-depth-design-brief.md` for
the full design rationale.

### The conditioned specialist track, and `adapters_v8_numeric_4b/` (shipping)

The v8 track separates numeric visual estimation from open-vocabulary food
reasoning. A 9.5 MB Core ML MobileNetV3 specialist predicts calibrated
p10/p50/p90 intervals for total mass, calories, protein, fat, and carbs. A
fresh Qwen3.5-4B LoRA sees the original image plus that fallible measurement
block and retains responsibility for food refusal, ingredient names, and
per-item allocation.

The full untouched 325-dish result is 56.8 kcal MAE / 29.7 median with zero
parse failures; macros are 5.1/4.4/6.0 MAE and held-out not-food refusal is
29/29. On the 322 dishes shared with v5, v8 is 57.2 versus 59.0 kcal MAE
(paired −1.8 kcal, bootstrap 95% CI −8.3 to +4.7): directionally better but
not statistically decisive. The user selected the system for its complete
schema reliability, best median, and extensible calibrated specialist path.
The exact research/evaluation record is
`docs/plans/2026-07-28-visual-specialist-conditioned-qwen-plan.md`.

## Toolchain rewrite (2026-07-26)

The legacy stack (`.venv/`, Python 3.14, five hand-applied patches to
mlx-vlm's trainer internals) was retired the same day it was replaced —
`.venv/` deleted, `patches/` removed from the working tree (still recoverable
from git history) — once all four legacy adapters (v1–v4) were confirmed
dead. `ml/.venv` (mlx-vlm 0.6.7, unpatched, straight from PyPI) is now the
only training environment; see `ml/setup.sh` and `ml/README.md` for current
setup.

Two migration traps found while porting to 0.6.7, both silent (no error,
just wrong results) rather than loud:

1. **0.6.7 loads images only from a top-level `images` JSONL field.**
   Content-embedded `{"type": "image", "image": path}` entries (fix #10's
   format) are parsed for prompt *text* but silently ignored for *pixel*
   loading — training runs to completion with no visual grounding at all,
   exactly Run 2's failure mode, but without Run 2's visible garbled output.
   Fixed by regenerating the dataset with a top-level `images` array added to
   every record (now `finetune_data_v2/` and later variants).
2. **LoRA scale convention changed.** It's now `alpha / rank` (so
   `--lora-rank 16 --lora-alpha 32` gives effective scale 2), whereas the
   legacy stack used a bare `alpha` (32) as the scale directly. Any adapter
   trained on the legacy stack must have its `alpha` reinterpreted, not
   copy-pasted, when converted for another consumer — see
   `ml/convert_adapter_for_swift.py`'s docstring for exactly how the iOS
   converter handles this.

`--train-on-completions` alone is sufficient on 0.6.7 — completion masking is
prefix-based now, there is no `--assistant-id` flag, and fix #18 (below) no
longer applies.

## Bug forensics (fixes #1–19)

Numbered in discovery order. Fixes #1–13 predate the toolchain rewrite and
were fixed by changes to this repo's own scripts; fixes #14–19 were fixed by
patches to the legacy mlx-vlm checkout and are **superseded upstream** by
0.6.7 (kept here only so a bug that resurfaces in some future mlx-vlm version
is recognizable).

1. **mlx-lm doesn't support `qwen3_5`** — use mlx-vlm instead; mlx-lm has no
   architecture support for Qwen3.5's multimodal layers.
2. **Python 3.9 too old** — mlx-vlm requires >= 3.10; recreate the venv.
3. **Missing torchvision** — `Qwen3VLVideoProcessor` needs torch+torchvision
   at processor-init time even though mlx-vlm's own `[train]` extra doesn't
   install them; `ml/setup.sh` installs both as an explicit separate step.
4. **pyarrow schema error** — `ArrowInvalid: Column(/messages/[]/content)
   changed from string to array in row 0`. HuggingFace `datasets` (via
   pyarrow) requires every row's `messages[].content` to share one type
   across the whole JSONL. Fix: make every role's `content` an array, never a
   bare string — this also matches mlx-vlm's own Qwen auto-formatter.
5. **"System message cannot contain images"** — Qwen3.5's Jinja2 chat
   template rejects any list/array content in the system role, even
   text-only arrays — which collides with fix #4 (pyarrow needs uniform
   array-typed content everywhere). Fix: drop the system message entirely,
   fold the system instruction into the user message's text. Inference must
   build the identical prompt shape (see `check_prompt_parity.py`).
6. **Ambiguous `--train` flag** — mlx-vlm actually has `--train-vision`,
   `--train-on-completions`, and `--train-mode`; always use the full name.
7. **`--train-mode lora` is invalid** — valid values are `sft` or `orpo`;
   LoRA itself is always enabled, `--train-mode` only selects the objective.
8. **`mlx_lm.fuse` `AttributeError: num_layers`** — mlx-vlm's
   `adapter_config.json` only has rank/alpha/dropout; `mlx_lm.fuse` expects a
   `num_layers` field and weight keys without the `language_model.` prefix.
   Fixed (for the fuse experiment in `attic/fix_adapter_for_fuse.py`) by
   adding the field and stripping the prefix.
9. **Vision tower missing after fuse** — `mlx_lm.fuse` only merges language
   weights; the vision tower is dropped entirely. A patch to copy it back
   left ~96 params still mismatched between mlx-lm and mlx-vlm naming
   conventions. Practical resolution: never fuse for serving — run the base
   model + adapter directly through mlx-vlm (Python) or `LoRAContainer`
   (Swift). `attic/patch_fused_vision.py` and `attic/fix_adapter_for_fuse.py`
   are kept for reference only.
10. **Root cause of Run 1's garbled output** — `<image>` as literal JSONL text
    is tokenized as text, not expanded to vision tokens; the adapter learns
    text→JSON with no visual grounding. Fix: content-embedded
    `[{"type": "image", "image": path}, ...]` array entries, which cause the
    processor to insert real vision tokens during both training and
    inference.
11. **`GenerationResult` object** — mlx-vlm's `generate()` returns a
    `GenerationResult`, not a plain string; read its `.text` attribute.
12. **`prompt_utils` vs. `processor.apply_chat_template`** — use
    `mlx_vlm.prompt_utils.apply_chat_template` for inference (it inserts real
    vision tokens); `processor.apply_chat_template(tokenize=False)` leaves
    `<image>` as literal text instead.
13. **Qwen3.5 thinking mode** — the chat template appends `<think>\n` to the
    assistant turn; strip it before generation for direct JSON output.
14. **`use_embedded_images` bug for Qwen (root cause of Run 2)** — mlx-vlm's
    `datasets.py` set `use_embedded_images=True` for any `model_type`
    starting with `qwen`, forcing `images=None` into `prepare_inputs`
    regardless of what the JSONL contained — the vision encoder never ran.
    Fix: patch `datasets.py` to extract PIL images from content-embedded
    entries and stop special-casing Qwen out of receiving them.
15. **PyArrow None-fill causes a phantom second `<|image_pad|>`** — because
    content dicts mix keys (image entries have `"image"`, text entries have
    `"text"`), pyarrow normalizes to a union schema and fills the *other*
    key with `None` on every row. A text entry then looks like
    `{"type": "text", "image": None, "text": "..."}`. Qwen3VL's Jinja2
    template checks `'image' in item` (key presence, not truthiness), so the
    text entry is misidentified as a second image — two `<|image_pad|>`
    tokens are emitted for one real image, and
    `prepare_inputs`/`image_grid_thw` mismatch with an `IndexError`. Fix:
    strip `None`-valued keys from content dicts before templating.
16. **`mx.stack` fails for Qwen3VL's dynamic tiling** — different input
    images produce different patch counts, hence different `pixel_values`
    shapes, and `mx.stack` requires uniform shapes across a batch. Fix: try
    `mx.stack` first, fall back to `mx.concatenate(..., axis=0)` — Qwen3VL is
    designed to receive concatenated patches across a batch, with
    `image_grid_thw` telling the model how many patches belong to each image.
17. **`len(input_ids)` silently returns 1, batch truncated to 33 tokens** —
    `prepare_inputs` returns `input_ids` shaped `(1, seq_len)`; calling
    Python's `len()` on it returns the leading batch dimension (1), not the
    sequence length. The resulting `max_len=1` truncated every sequence to
    ~33 tokens, discarding almost all image tokens (300–2040 per image)
    while the vision tower still processed every patch — a silent shape
    mismatch, not a crash. Fix: use
    `np.array(x["input_ids"]).reshape(-1).shape[0]` instead of `len(...)`.
18. **`--assistant-id` default wrong for Qwen3.5 (root cause of Run 3)** —
    mlx-vlm's default `assistant_id=77091` decodes to `'[hash'` in Qwen3.5's
    tokenizer, a token that never appears in any training sequence, so the
    completion mask never activates and loss is computed over the entire
    sequence. Confirmed: `processor.tokenizer.encode('assistant') = [74455]`.
    Fix: pass `--assistant-id 74455` explicitly. No longer applicable on
    mlx-vlm >= 0.6.7, where completion masking is prefix-based and there is
    no `--assistant-id` flag at all.
19. **`lora.py` crashes when `--iters` exceeds dataset size** —
    `dataset.select(range(iters))` raises `IndexError` once `iters` is
    larger than the dataset. The batch-iteration loop already cycles the
    dataset indefinitely for multi-epoch training, so the `select` only
    needs to cover `min(iters, len(dataset))`.

## See also

- `ml/README.md` — the current, working pipeline: setup, data prep, training,
  evaluation, and the expected-metrics table.
- `docs/design/2026-07-26-depth-design-brief.md` — full design rationale for
  the depth track (variants A/B, why A was blocked, why B was stopped).
- `docs/architecture.md` — the ml/iOS prompt-parity contract these bugs kept
  breaking, and how it's now enforced (`check_prompt_parity.py`,
  `smoke_test.py`).
