# SeeCal execution plan — 2026-07-26

Source: full-project review + recheck + research (session d2b53ffc). Goal: offline
photo→calories on iPhone. Executor: Fable main loop; Sonnet subagents for standard
tasks; Fable subagent for complex iOS wiring.

## Global constraints
- Prompt text (SYSTEM + "\n\n" + USER) must remain byte-identical across
  02_prepare_finetune.py, 03_infer.py, and iOS QwenPromptBuilder. No system role.
- Output schema: {"total_calories", "protein_g", "fat_g", "carbs_g",
  "items":[{"name","estimated_grams","calories","protein_g","fat_g","carbs_g"}]}
  (+ optional confidence/uncertainty_flags on the app side).
- mlx-vlm 0.6.7 requires a TOP-LEVEL "images" field per JSONL record; content-embedded
  image entries are ignored for pixel loading.
- mlx-vlm ≥0.5.0 LoRA scale = alpha/rank (standard convention).
- Old venv (.venv, patched 0.4.0) must remain untouched until the new stack passes
  the validation ladder — it is the only known-working trainer.
- Never `git add` model weights/datasets: .gitignore is the guard; verify staged
  sizes before every commit.

## Tasks
- T1 (Fable, ops): Git protection. .gitignore; patches/ backup of the 3 patched
  .venv files + diffs vs pristine 718f69e9; neutralize nested repo
  (ios/SeeCalApp/.git → .git.disabled, record its commit hash); move root
  checkpoint litter to runs/v4_root_output/; baseline commit on main; branch
  `execute-plan`.
- T2 (Fable, code): Harden 03_infer.py eval: per-sample try/except on metric
  extraction (count schema failures separately), thread --max-tokens through
  evaluate_test_set, default max tokens 1536. Then launch background baseline
  eval: adapters_v4 vs base, 50 test dishes each, logged to runs/eval_v4_baseline/.
- T3 (Sonnet): New venv .venv-vlm067 with mlx-vlm[train]==0.6.7; verify import,
  CLI flags (--epochs, --train-on-completions, no assistant-id needed), torch dep.
- T4 (Sonnet): 02_prepare_finetune.py v2: add top-level "images" field (paths
  relative to repo root, as today); drop ingredient items with grams < 0.05 from
  targets (recompute nothing else); keep prompt byte-identical; output to
  finetune_data_v2/ with same seed/split logic; report counts + 2-record spot check.
- T5 (Fable authors, gate): 00_smoke_test.py in repo, dual-stack (0.4.0 assistant-id
  masking / 0.6.7 completion_mask); run on .venv-vlm067 + finetune_data_v2. GATE.
- T6 (gate, background): Overfit micro-run — 32 dishes, ~200 iters, batch 1,
  new stack, adapters output runs/overfit_smoke/. Pass = loss collapses AND
  generation on 2 training images approximately reproduces targets. GATE.
- T7 (gate, background): Probe — 500 iters full finetune_data_v2, eval 50 test
  dishes, MAE must beat base-model baseline from T2. GATE → then launch full run
  (2-3 epochs, max-seq-length 2048, adapters_v5/).
- T8 (Fable subagent): iOS adapter loading — convert_adapter_for_swift.py
  (.A/.B → .lora_a/.lora_b + mlx-lm-style config, scale=32.0 for v4-era adapters);
  wire LoRAContainer.from(directory:)+load into MLXQwen35RunnerBuilder honoring
  config.adapterPath; parity test scaffold vs 03_infer.py output.
- T9 (Sonnet): iOS app fixes — file-backed persistence (Application Support JSON),
  scan reentrancy guard + picker disable, per-runtime error surfacing, today-filter
  for consumedToday, move meal images out of tmp/.
- T10 (Sonnet): Docs sync — CLAUDE.md: correct schema example, image-path
  convention (repo-root relative), real counts (2594/325/325), v4 run post-mortem
  (died iter 800, actual flags), new-toolchain section (0.6.7, no patches, traps);
  iOS README: adapters_v3 → current, timeout drift.
- Final: whole-branch review (most capable model), fix wave, merge decision.

## Task order
T1 → {T2, T3} → T4 → T5 → T6 → T7 (training track)
T1 → {T8, T9} → T10 (iOS/docs track, parallel with training track)
