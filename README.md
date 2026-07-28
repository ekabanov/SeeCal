# SeeCal

Point your phone at a plate of food. Get calories, macros, and a per-ingredient
breakdown back — entirely on-device, no server, no network call.

SeeCal combines a compact Core ML visual specialist with a conditioned
**Qwen3.5-4B** LoRA trained on **Nutrition5k**, then runs both models directly
on iPhone with no server.

## How it works

```mermaid
flowchart LR
    A[Meal photo] --> B[Core ML specialist]
    A --> D[Qwen3.5-4B + conditioned LoRA]
    B -->|calibrated mass/calorie/macro intervals| D
    D -->|JSON, on-device| F[Calories / macros / ingredients]
```

The two halves — the training pipeline and the iOS app — never share code.
They agree on a narrow contract: converted model artifacts and a prompt string,
including the specialist measurement block, that must be byte-identical on
both sides (see `docs/architecture.md`).

## Results

MAE / median absolute calorie error on the **full 325-dish** held-out set
(`ml/finetune_data_v2/test.jsonl`, via `ml/infer.py` / `ml/eval.sh`):

| Model | Calories MAE | Calories median | Parse failures | Refuses non-food |
|---|---:|---:|---:|:--:|
| Base model (no adapter) | 83.4¹ | 63.9¹ | 0 | no |
| `adapters_v5` | 59.0 (n=322) | 31.4 | 3 | no |
| `adapters_v7b` | 63.4 (n=324) | 30.5 | 1 | yes |
| **Conditioned 4B + specialist (shipping)** | **56.8 (n=325)** | **29.7** | **0** | **yes** |

¹ 50-dish sample; the base model was never run over the full set.

**Read the MAE column carefully.** Parse failures are excluded per-adapter, so
v5's 59.0 (n=322) and v7b's 63.4 (n=324) are computed over *different dish
sets* and are not directly comparable. Paired on the 324 dishes both answered,
v5 is 62.5 and v7b 63.4 — a difference of +0.87 kcal (t = 0.23, not
significant), with v7b's median better and its parse failures fewer. v7b adds
not-food refusal at no measurable food-accuracy cost.

The conditioned system answers all 325 dishes and retains 29/29 held-out
not-food refusal. On the 322 dishes shared with v5 its calorie MAE is 57.2
versus 59.0 (paired mean delta −1.8 kcal; 95% bootstrap CI −8.3 to +4.7), so
the historical-set gain is directional rather than statistically decisive.
Its advantages are the best median, zero parse failures, calibrated auxiliary
uncertainty, and a clean on-device path for future specialist improvements.

See `ml/README.md` for the complete metrics table (including the
depth-augmented `adapters_v6`, a measured tie that was not shipped, and
`adapters_v7`, whose heavier negative dose caused a real regression),
`docs/plans/2026-07-27-v7-notfood-plan.md` for the refusal experiment, and
`docs/training-history.md` for the four earlier, broken training runs.

## Quickstart

**Training pipeline** (Apple Silicon required — mlx/mlx-vlm are Metal-only):

```bash
cd ml
./setup.sh                                       # python env (mlx-vlm 0.6.7 stack)
./download_dataset.sh                            # Nutrition5k subset (CC BY 4.0, not committed)
./download_model.sh                              # Qwen3.5-4B-MLX-4bit base model
export MODEL_PATH=~/models/mlx-community/Qwen3.5-4B-MLX-4bit
./prep.sh && ./train.sh && ./eval.sh adapters_v5 --limit 325 && ./convert.sh adapters_v5
```

The exact conditioned-system reproduction and artifact paths are documented in
`docs/plans/2026-07-28-visual-specialist-conditioned-qwen-plan.md`.

Full walkthrough, hardware requirements, and the validation ladder (never
skip straight to a multi-hour training run) in **`ml/README.md`**.

**iOS app** (Xcode 26+, iOS 17+, physical iPhone 15 Pro+ recommended — the
simulator builds fine but can't run real MLX inference):

```bash
SEECAL_ALLOW_NO_WEIGHTS=1 scripts/build.sh       # fast UI-only simulator build
MODELS_DIR=/path/to/models scripts/build.sh --device   # signed device build, weights bundled
scripts/test.sh                                  # ml pytest + swift test + iOS build check
```

Full walkthrough, weights-bundling layout, and release flow (TestFlight /
App Store) in **`ios/README.md`**.

## Repo map

```
ml/       training pipeline: dataset prep, LoRA fine-tuning, evaluation,
          Swift-format adapter conversion. ml/README.md.
ios/      SeeCal SwiftPM package (all app code) + thin XcodeGen app wrapper.
          ios/README.md, ios/SeeCal/README.md.
scripts/  build.sh, test.sh, release-{setup,testflight,appstore}.sh.
docs/     architecture.md (the ml <-> iOS contract), training-history.md
          (every training bug, in forensic detail), third-party.md (license
          audit), specs/ + design/ (the binding app spec and visual
          prototype), plans/ (dated planning docs).
attic/    retired one-off scripts, kept for reference only.
```

`AGENTS.md` is a short agent-facing guide to the same material — read the
docs above for the full version.

## License

The code in this repository is **MIT-licensed** — see `LICENSE`.

That covers the code only:

- **Nutrition5k** (the training dataset) is Google Research's, released
  under **CC BY 4.0**. It is never committed to this repo — `ml/download_dataset.sh`
  downloads your own copy directly from the dataset's public bucket, under
  the dataset's own terms.
- **Qwen3.5** (the base model) is Alibaba's, released under the
  **Apache License 2.0**. It is downloaded from Hugging Face
  (`ml/download_model.sh`), not redistributed here.
- **Fine-tuned LoRA adapter weights** produced by this pipeline are not
  committed to this repository either.

See `docs/third-party.md` for the full dependency license audit (mlx,
mlx-vlm, mlx-swift, mlx-swift-lm, transformers, XcodeGen, etc. — nothing
copyleft anywhere in the stack).
