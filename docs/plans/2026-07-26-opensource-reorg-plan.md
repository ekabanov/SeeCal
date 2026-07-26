# Open-Source Monorepo Reorg Plan (R1–R7)

> Fresh subagent per task; Fable reviews after each; serialized (shared paths).
> Tiers: Fable plan/review; Opus complex (R5); Sonnet the rest.

**Goal (user):** clean monorepo; docs/specs/READMEs/CLAUDE.md cleanup; build/test/release
scripts (TestFlight + App Store); scripts to download the dataset, prep it, train, eval,
convert — all well documented. Prepare for open-sourcing (nothing is published without
explicit user confirmation).

**Branch:** `opensource-reorg` off main.

## Target layout
```
/            README.md · LICENSE (MIT) · CLAUDE.md (slim agent guide) · .gitignore
├── ml/      ← ENTIRE training pipeline; all commands run from ml/
│   ├── README.md          end-to-end walkthrough (download → prep → smoke → train → eval → convert)
│   ├── setup.sh           creates ml/.venv (mlx-vlm 0.6.7 pinned stack; torch+torchvision separately)
│   ├── download_dataset.sh  Nutrition5K from its public GCS bucket (data NEVER in git)
│   ├── download_model.sh    base model from HF (mlx-community/Qwen3.5-4B-MLX-4bit)
│   ├── prep.sh · train.sh · eval.sh · convert.sh   thin, documented entrypoints
│   ├── select_images.py (01_) · prepare_finetune.py (02_) · infer.py (03_) ·
│   │   smoke_test.py (00_) · depth_features.py · convert_adapter_for_swift.py ·
│   │   check_prompt_parity.py · tests/
│   └── Nutrition5K/ dataset_clean/ finetune_data*/ adapters*/ runs/   (untracked; moved as-is)
├── ios/
│   ├── SeeCal/   SwiftPM package (unchanged layout)
│   ├── App/      Xcode app wrapper via XcodeGen (R5) — needed for TestFlight/App Store
│   └── README.md
├── docs/     specs/ plans/ design/ + architecture.md + training-history.md
├── scripts/  build.sh · test.sh · release-testflight.sh · release-appstore.sh
└── attic/    (unchanged)
```

## Invariants
- **JSONL image paths are relative to the pipeline root.** Data dirs move under ml/
  KEEPING their names (ml/dataset_clean/…), and every pipeline command runs from ml/ —
  existing generated JSONLs stay valid, nothing regenerates.
- Weights/datasets/runs stay untracked; .gitignore paths updated to ml/-prefixed.
- The legacy stack is gone; ml/setup.sh recreates ONLY the 0.6.7 stack (pin versions
  from the current .venv-vlm067: mlx-vlm 0.6.7, mlx 0.32.0, mlx-lm 0.31.3,
  transformers 5.14.1, datasets 5.0.0, torch 2.13.0, torchvision 0.28.0, pytest).
- Nothing is pushed/published anywhere in this track; publishing is a separate
  user-confirmed step.
- After every task: ml pytest suite green (run from ml/) AND ios suite green.

## Tasks
### R2 (Sonnet): monorepo restructure
git mv tracked pipeline files into ml/ with the renames above (drop number prefixes);
plain mv for untracked data dirs; fix ALL internal references (infer.py's importlib load
of prepare_finetune, tests' imports, train.sh paths, check_prompt_parity paths,
convert_adapter_for_swift defaults, .gitignore, docs/plans+specs path mentions where
load-bearing). Check ios ModelAssetResolver for repo-relative simulator fallback paths
(adapters_v5_swift) and update. Verify: pytest from ml/ green; check_prompt_parity runs;
ios suite green; `git status` clean of accidental data adds.

### R3 (Sonnet): acquisition + setup scripts
ml/setup.sh (idempotent venv + pinned installs + torchvision note + smoke import check),
ml/download_dataset.sh (Nutrition5K public GCS bucket via gsutil/curl, resumable,
size warning ~180GB full / document the subset actually needed: realsense_overhead +
metadata CSVs; verify against the two known-corrupt depth files note),
ml/download_model.sh (huggingface-cli or curl resolve URLs into ~/models or ml/models,
path consumed by train.sh/infer.py via a single MODEL_PATH convention). Each script
--help'd and documented in ml/README.md skeleton.

### R4 (Sonnet): pipeline entrypoints + ml/README.md
prep.sh (select_images → prepare_finetune → smoke_test, with the validation-ladder
gates), train.sh (moved + MODEL_PATH param + co-tenancy warning), eval.sh (REQUIRED
--limit param, defaults to full test set — fixes the silent-20 default trap),
convert.sh (adapter → Swift format; stamp adapter version into adapter_config.json —
closes the P7 follow-up). ml/README.md: full end-to-end walkthrough incl. hardware
requirements, timings (~3.5h train on M3 Ultra), validation ladder, expected metrics
table (v5: cal MAE 54.4/median 36.2 on 50 dishes vs base 83.4).

### R5 (Opus): iOS app wrapper + release scripts
ios/App via XcodeGen project.yml (app target embedding the SeeCal package; weights
bundling documented as a build-phase copy from a configurable MODELS_DIR — weights not
in git; increased-memory entitlement; iOS 17+, iPhone 15 Pro+ note), scripts/build.sh,
scripts/test.sh (both suites: ml pytest + swift test + iOS-destination build with the
FMT_CONSTEVAL flag), scripts/release-testflight.sh + release-appstore.sh (xcodebuild
archive → exportArchive → App Store Connect API upload; --dry-run mode that stops
before upload; clear error if signing not configured). **Secrets handling (user
directive)**: `scripts/release-setup.sh` interactively collects the App Store Connect
credentials once (key ID, issuer ID, and the .p8 key file — prompted with no echo,
never printed, never logged) and stores them in repo-local `.secrets/` (chmod 700,
key file chmod 600). The release scripts source `.secrets/` automatically — no manual
env setup ever. `.secrets/` goes into `.gitignore` FIRST, and release-setup.sh must
verify gitignore coverage (`git check-ignore`) before writing anything, refusing to
proceed otherwise. ios/README.md updated. Scripts must be shellcheck-clean and
runnable to the point of graceful failure without certs.

### R6 (Sonnet): docs sweep + open-source hygiene
Top-level README.md (project story, architecture diagram (mermaid), results table,
quickstart for both halves, license + dataset attribution); CLAUDE.md REWRITTEN slim
(repo map, conventions, build/test commands, key gotchas) with the historical war
stories condensed into docs/training-history.md (preserve the bug-forensics value);
docs/architecture.md (pipeline + app + how they meet at the adapter/prompt contract);
LICENSE (MIT — code only; note that Nutrition5K data and model weights are NOT covered
and not redistributed); license/attribution audit (Nutrition5K terms, Qwen Apache 2.0,
mlx/mlx-vlm/mlx-swift licenses); secrets scan of tracked files AND git history (report
findings, fix nothing destructive without user approval).

### R7 (Fable): final review + relaunch evals
Whole-branch review; verify both test suites + a real end-to-end dry run of the
entrypoint scripts; merge to main; THEN relaunch the 325-dish evals from ml/
(cd ml && .venv/bin/python -u infer.py --test-set finetune_data_v2/test.jsonl
--limit 325 --adapter-path adapters_v5 …, then v6 with finetune_data_v2d_txt) —
adapter decision + model-card accuracy copy follow.

## Order
R2 → R3 → R4 → R5 → R6 → R7 (strictly serial).
