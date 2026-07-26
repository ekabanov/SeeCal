# SeeCal — agent guide

Photo → calories/macros/ingredients, on-device, via a Qwen3.5-4B LoRA
fine-tune. Full story: top-level `README.md`. Architecture + the ml/iOS
contract: `docs/architecture.md`. This file is the short, working-memory
version for agents — see `docs/` for everything it points to instead of
repeating.

## Repo map

```
ml/       training pipeline (Python + mlx-vlm). ALL commands run from ml/ —
          see ml/README.md for setup -> prep -> train -> eval -> convert.
ios/      SeeCal SwiftPM package (ios/SeeCal) + thin XcodeGen app wrapper
          (ios/App). See ios/README.md and ios/SeeCal/README.md.
scripts/  build.sh, test.sh, release-{setup,testflight,appstore}.sh —
          the only supported way to build/test/release the app.
docs/     architecture.md, training-history.md, third-party.md,
          specs/ (binding app spec), design/ (prototype = binding visual
          spec), plans/ (dated planning docs, historical).
attic/    retired one-off debugging/patch scripts, kept for reference only.
```

## Conventions

- **Run every `ml/` command from `ml/`.** JSONL image paths are relative to
  that directory, not the repo root — running from elsewhere silently breaks
  image loading (fails loud in `smoke_test.py`, not silently, if you do it
  right).
- **Spec wins over code for the app.** `docs/specs/2026-07-26-app-spec.md`
  and `docs/design/prototype/seecal-prototype.html` are binding — the
  prototype HTML is the visual spec (colors, spacing, motion), not a
  reference sketch. If the code and the spec/prototype disagree, the
  spec/prototype is right until a human says otherwise.
- Never commit datasets, model weights, or adapters (`.gitignore` already
  covers `ml/Nutrition5K/`, `ml/adapters*/`, `*.safetensors`, `ml/runs/`).
  Never commit release credentials (`.secrets/`).

## Build & test

```bash
cd ml && .venv/bin/python -m pytest tests/        # ml pipeline unit tests
cd ios/SeeCal && swift test -Xcxx -DFMT_CONSTEVAL= # Swift package tests
scripts/test.sh                                    # both of the above + iOS build check
scripts/test.sh --skip-build                       # skip the slow xcodebuild step
```

`-DFMT_CONSTEVAL=` works around Xcode 26's clang being stricter about
`consteval` than the `fmt` vendored inside `mlx-swift`. If you open
`ios/SeeCal` directly in Xcode instead of the CLI, the auto-generated scheme
is named `SeeCal-Package` — add the same flag to its build settings or
you'll hit the same error there. The app wrapper's own xcodebuild scheme
(via `scripts/build.sh`, `ios/App/project.yml`) is `SeeCal`, already wired
with the flag.

## Key gotchas

- **`eval.sh` requires `--limit`.** `infer.py`'s own `--limit` silently
  defaults to 20 and once truncated a full overnight eval without warning.
  `eval.sh` refuses to guess — pass it explicitly (325 for the full
  committed test set).
- **Metal memory doesn't swap.** Don't run `train.sh` alongside anything
  else GPU-heavy (LM Studio, `eval.sh`, a second training run) — co-tenancy
  silently OOM-kills training instead of failing cleanly.
- **torchvision must be installed explicitly.** mlx-vlm's own `[train]`
  extra doesn't pull it in, but the Qwen3VL processor imports it at init
  time. `ml/setup.sh` installs torch/torchvision as a separate step for
  exactly this reason.
- **Adapter↔prompt byte-parity is the single most load-bearing invariant in
  this project.** Training and inference must build the *exact* same prompt
  text (no system message, same vision-token placement) or the LoRA adapter
  learns a mapping the app never actually sends it. Enforced by
  `ml/check_prompt_parity.py` and `ml/smoke_test.py`. See
  `docs/architecture.md` for the full contract and
  `docs/training-history.md` for every way this broke before it was fixed.

## Current state (see `docs/` for detail, not this file)

- Shipping adapter: `adapters_v5` (calories MAE 54.4 / median 36.2 on 50
  held-out dishes vs. base model 83.4 / 63.9 — see `ml/README.md` for the
  full table). `adapters_v6` (depth-augmented) was a statistical tie with v5
  and is not shipped. `adapters_v1`–`v4` are all confirmed dead — see
  `docs/training-history.md`.
- 325-dish evaluation of v5 vs. v6 is queued, not yet run.
- iOS app: full product build-out complete against the prototype spec;
  `scripts/build.sh`/`scripts/test.sh` are green.
