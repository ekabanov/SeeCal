# Depth-Input + Post-v5 Execution Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development.
> Fresh subagent per task, task review after each, gates decide branches.

**Goal:** Improve calorie accuracy by adding portion-size signal from depth (training:
Nutrition5K RealSense; inference: iPhone LiDAR), plus close the remaining backlog, on
top of the completed v5 pipeline.

**Architecture:** Two candidate depth representations trained as probes — (A) depth as
a second image, (B) depth as derived text features (volume/height via plane fit) — the
winner graduates to a full run (adapters_v6) and only then gets an iOS capture path.
All training on the 0.6.7 stack with the speed flags validated by the speed experiments.

**Tech stack:** Python (.venv-vlm067 for training/eval; numpy/PIL for depth), mlx-vlm
0.6.7, Swift/ARKit for capture (phase D4 only).

## Global Constraints
- Base prompt (SYSTEM + "\n\n" + USER) stays byte-identical to v5 in every variant;
  variant B APPENDS one exactly-specified line (below) — training and inference must
  append the identical line, and iOS QwenPromptBuilder must replicate it verbatim.
- Output JSON schema unchanged: total_calories/protein_g/fat_g/carbs_g/items[...].
- Every new dataset variant must pass 00_smoke_test.py before any training.
- Never evaluate an adapter under a venv that doesn't match its training stack.
- GPU serialization: never run two training/eval jobs concurrently.
- Weights/datasets stay untracked (existing .gitignore); JSONL text is tracked.
- Depth-raw units are UNVERIFIED (uint16, observed range ~0-501, rig height 35.9cm per
  Nutrition5k paper) — Task D1 must determine them empirically before anything builds
  on them; check Nutrition5K/ dir for docs/readme first.

## Phase 0 (already queued, runs first): v5 endgame
v5 eval (50 dishes vs base 83.4 cal MAE) → speed experiment matrix (batch 1/4/8 ×
grad-checkpoint, 32-dish set, pick config: real tokens/sec gain, peak mem < ~70GB)
→ update train.sh flags → convert v5 for Swift (exercises converter new-format branch)
→ CLAUDE.md numbers → merge execute-plan → main. Depth work begins only after this.

---

### Task D1: depth_features.py — units, plane fit, volume (Sonnet)
**Files:** Create: `depth_features.py`, `tests/test_depth_features.py` (pytest, runs with .venv)
**Produces:** `load_depth(path) -> np.ndarray (meters, HxW, NaN=invalid)`;
`plane_fit(depth) -> (normal, d, inlier_mask)` (RANSAC on border+plate pixels);
`food_stats(depth, plane) -> {"volume_ml": float, "max_height_mm": float, "coverage_cm2": float}`;
`depth_to_height_image(depth, plane) -> PIL.Image` (8-bit grayscale, 0=plate plane,
255=60mm+, for variant A).
Steps: (1) determine raw units empirically — check any docs in Nutrition5K/, then
validate: median border depth should ≈ rig height 0.359m under the right scale
(candidate scales: /1000 mm, /10000 [10⁻⁴ m]); assert chosen scale in a test with 5
dishes. (2) Volume: per-pixel footprint from pinhole model — D435 FOV 640px ≈ 65°
horizontal → focal ≈ 616px; footprint = (z/f)²; integrate max(0, plane_z - z) over
food pixels (height > 3mm noise floor, inside plate region). (3) Sanity harness: run on
20 dishes joined to dish_nutrition_values.csv mass; report volume vs mass correlation
(expect r > 0.5; food density ~0.5-1.5 g/ml). Test asserts correlation > 0.3 as a
regression floor, prints the table. Commit.

### Task D2: dataset variants (Sonnet, after D1)
**Files:** Modify: `01_select_images.py` (add `--with-depth`: copy depth_raw.png to
dataset_clean/<dish>/depth_raw.png for overhead dishes; idempotent, skip existing);
Modify: `02_prepare_finetune.py` (add `--depth-mode {none,image,text}`, default none —
existing behavior byte-identical when none).
- mode=image: records get `"images": [rgb, height_image_path]` (height image rendered
  via depth_to_height_image into dataset_clean/<dish>/height.png) + second content
  image entry. Out-dir finetune_data_v2d_img/.
- mode=text: user text = BASE_PROMPT + "\n\nEstimated food volume from depth sensor: ~{volume_ml:.0f} ml (max height {max_height_mm:.0f} mm)." Out-dir finetune_data_v2d_txt/. Round to
  whole numbers; line ABSENT when depth missing for a dish (graceful degradation is in-distribution).
- Same seed/split; dishes lacking depth keep plain records in both variants.
Run both generations; report counts, depth coverage %, 2-record spot checks. Commit.

### Task D3: inference parity for variants (Fable — prompt-parity is load-bearing)
**Files:** Modify: `03_infer.py` (add `--depth-mode`, `--depth-image`, accept per-record
second image from test JSONL / compute text line via depth_features when evaluating
variant sets); Modify: `00_smoke_test.py` only if variant records fail it as-is (two-image
records: assert image-token count = sum over BOTH grids).
GATE: smoke test passes on BOTH variant train.jsonl files; a manual prompt-parity check
(train prefix == infer prompt) passes for both variants, same method as T6.

### Task D4: probes + decision (me/Fable, GPU-serialized)
Probe each variant: 1000 iters, speed-validated flags, runs/probe_v6_{img,txt}/;
eval 50 test dishes with matched depth-mode. Compare calorie MAE/median + parse rate vs
v5-full and base. DECISION GATE: winner = better calorie MAE than v5 by >5% with parse
failures <=2/50; if neither clears, STOP depth track, report, await user direction.
Winner → full run (adapters_v6, 2 epochs) → full eval → convert_adapter_for_swift.

### Task D5: iOS depth capture (Fable agent, only after D4 picks a winner)
**Files:** Create: `ios/SeeCal/Sources/SeeCalInference/DepthCapture.swift` (AVDepthData/
ARKit LiDAR capture behind `LiDARCapability.isAvailable`), plane-fit+volume port
(Accelerate; same algorithm/constants as depth_features.py, cross-checked in a parity
test against Python outputs on 3 exported RealSense maps); Modify: QwenPromptBuilder —
appends the EXACT variant-B line when volume present (verbatim format string from D2),
or attaches second image for variant A; absent depth = plain v5 prompt. swift tests.
Non-LiDAR devices: feature off (variant B degrades in-distribution).

### Task D6: backlog closeout (Sonnet)
- FileBackedStoreIO decode-failure: log + rename unreadable file to `<name>.bak-<ts>`
  before writing fresh state; test with corrupt JSON fixture.
- Swift↔Python parity test run with the shipped adapter (v5 or v6): same image both
  stacks, assert total_calories within 5% — document result in CLAUDE.md.
- errors_cal asymmetry in 03_infer.py summary: include all four error lists.
- CONFIRM WITH USER before: deleting dataset_clean_sample/ (1.9GB dup),
  finetune_data_sample/, moving fix_adapter_for_fuse.py / patch_fused_vision.py /
  debug_shapes.py to attic/.

### Task D7: docs + merge (Sonnet)
CLAUDE.md: depth pipeline section (units finding, feature definitions, variant results
table, decision), update Next Steps. Ledger. Final review of depth branch (Fable),
fix wave, merge.

## Task order
Phase 0 → D1 → D2 → D3(gate) → D4(gate, GPU) → {D5 if winner, D6 parallel-safe} → D7
D6 items 1+3 may run any time after Phase 0 (no GPU, disjoint files).
