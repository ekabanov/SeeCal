# Visual specialist

This package trains the compact, multi-task image model described in
`docs/plans/2026-07-28-visual-specialist-conditioned-qwen-plan.md`.

Numeric targets come only from measured Nutrition5K records. FRB and Gemini
records have numeric loss masks disabled; they can improve the shared visual
representation and semantic heads but cannot teach calories, macros, or mass.

## Runtime contract

The deployment model accepts one RGB image and no text, depth, metadata, or
Qwen output. The caller resizes the short side to 232 pixels, center-crops to
224 by 224, converts to a float tensor with shape `[1, 3, 224, 224]`, and
applies ImageNet normalization.

The Core ML graph returns raw tensors in this fixed order:

| Name | Shape | Postprocessing |
|---|---:|---|
| `numeric_log1p` | `[1, 5, 3]` | clamp, `expm1`; fields are mass, calories, protein, fat, carbs and quantiles are p10, p50, p90 |
| `food_logit` | `[1]` | sigmoid |
| `frb_logits` | `[1, 498]` | auxiliary experiment only |
| `container_logits` | `[1, 6]` | auxiliary experiment only |
| `cooking_logits` | `[1, 8]` | auxiliary experiment only |
| `mixed_logit` | `[1]` | auxiliary experiment only |
| `occlusion_logits` | `[1, 3]` | auxiliary experiment only |

The 1,280-dimensional backbone embedding is available during training but is
deliberately omitted from Core ML. The leading `s1-sideviews` checkpoint only
trains the numeric and food heads, so its auxiliary logits must not be consumed
at runtime. Only uncertainty fields that pass held-out calibration are eligible
for the later Qwen prompt.

Run commands from `ml/`:

```bash
.venv/bin/python -m visual_specialist.manifest \
  --frb-manifest datasets/food_recognition_2022/derived/pilot-5000/pilot-5000-enriched.jsonl \
  --out-dir datasets/visual_specialist/pilot-v1

.venv/bin/python -m visual_specialist.train \
  --experiment s0-overhead \
  --manifest-dir datasets/visual_specialist/pilot-v1 \
  --output-dir runs/visual-specialist/s0-overhead
```

The causal sequence is:

- `s0-overhead`: measured overhead images plus negatives;
- `s1-sideviews`: adds measured side views;
- `s2-frb-raw`: adds the raw 498-class FRB labels;
- `s3-frb-teacher`: additionally enables filtered Gemini attributes.

`export_coreml.py --trace-only` validates the TorchScript graph in the training
environment. Core ML conversion targets an iOS 17 float16 ML Program. The
currently pinned Core ML 9 export environment uses Python 3.13, the newest
Python version that release officially supports.

The selected S1 deployment artifact is produced from `ml/` with:

```bash
.venv-specialist-export/bin/python -m visual_specialist.export_coreml \
  --checkpoint runs/visual-specialist/s1-sideviews-v2/best.pt \
  --output runs/visual-specialist/deployment/source/SeeCalVisualSpecialist.mlpackage
xcrun coremlcompiler compile \
  runs/visual-specialist/deployment/source/SeeCalVisualSpecialist.mlpackage \
  runs/visual-specialist/deployment
```

`VisualSpecialistTests/testCoreMLSpecialistParityScaffold` compares the compiled
model plus Swift resize/crop/normalization against recorded Python predictions.
The selected Lanczos pipeline measured 10.0% mean relative drift across all
five estimates and 10.2% for calories on 50 images, inside the 20% corruption
range used to make the conditioned Qwen training data robust.

Each run writes:

- `config.json`: immutable run inputs and toolchain;
- `metrics.jsonl`: append-only per-epoch metrics;
- `status.json`: atomic current state, percentage, elapsed time, ETA, and best
  calorie MAE;
- `best.pt`: the best validation checkpoint.

Inspect all runs without parsing logs:

```bash
.venv/bin/python -m visual_specialist.progress
```
