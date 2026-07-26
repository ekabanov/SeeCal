# Depth pipeline design brief (Fable research pass, 2026-07-26)

Governs tasks D1-fix..D5 of docs/plans/2026-07-26-depth-plan.md. Empirics measured on
local Nutrition5K data (200-dish samples); web claims cited at bottom.

## (a) Unit scale — SETTLED, plus critical rig discovery
- **1 unit = 1e-4 m** (official README: "1 meter = 10,000 units"; paper: "depth units
  of 1e-4"). Plate-surface mode measured 0.341–0.362 m ≈ 35.9 cm rig height. The
  plan's "range 0-501" premise was false (values ~2,600–4,200 valid, speckle to 65,535).
- **GLASS PLATFORM TRAP**: the dish sits on glass; IR depth sees THROUGH it to the
  table ~4.8 cm below (border median 0.407 m). The capture plane (0.359 m) is invisible
  in depth. Border-pixel plane fits reference the WRONG surface → all heights inflated
  ~48 mm → 1.6–3.3 L volumes (measured failure, matches D1 draft's output).
- Invalid = `raw == 0 OR raw > 4200` → NaN (98.5% of dishes carry >4,200 speckle);
  no inpainting (holes hug tall-food edges = stereo shadow; median invalid 15%, p95 27%).
- Intrinsics: none published. Paper's per-pixel area 5.957e-3 cm² @ 0.359 m implies
  **f = 465.1 px** (D435 color @ 640x480; the spec'd 616 was the D415 value → pure
  ×1.75 volume bias). c = (320, 240). Depth is RGB-aligned, 8-s temporal average.
- Depth coverage: 3,473/3,490 dataset_clean dishes (99.5%); 1/2,594 train records lacks it.
  Two corrupt files (already handled in D1): dish_1556572657 (~10x-off), dish_1564159636
  (zero-byte). GCS dataset has depth_train_ids.txt/depth_test_ids.txt (curated RGB-D
  subset) — not in local copy, worth downloading as cross-check.

## (b) Algorithm (replaces D1 draft's border RANSAC)
1. z = raw*1e-4; invalid → NaN.
2. **Reference = support-surface mode**: 1 mm-bin histogram of central-disk (r<240px)
   depths restricted to z ∈ [0.30, 0.398] m (upper cut excludes below-glass table —
   without it the mode snaps to 0.407 and volumes explode). Median mode 0.358 m.
3. plane_fit = robust fit to pixels within ±4 mm of the mode (constant-z suffices on
   N5k — table std 3.3 mm; the fit path exists for iPhone handheld tilt).
4. height = ref − z; food mask = central ∧ height ∈ (4 mm, 150 mm);
   volume = Σ height·(z/465.1)²; **max_height_mm = p99** (not max — speckle);
   coverage = Σ(z/f)² over food pixels.
5. Validation on n=199: geometric r=0.32 (Spearman 0.34), implied median density
   0.42 g/ml; with white-plate RGB mask (min(R,G,B)>130 ∧ sat<0.22): r=0.48/0.54,
   density 0.54 (IQR 0.36–0.96). Volume p5/med/p95 = 38/329/841 ml.
6. **Decisive test: implied median density m/V ∈ [0.3, 1.2] g/ml** — the r-gate alone
   passes on the buggy border-fit version (r=0.57 with density 0.06!); the density
   assertion is what discriminates correct from correlated-but-biased.
7. White-plate mask is a N5k-specific crutch: keep it in the sanity harness only;
   feed the model the geometric feature (VLM sees the plate in RGB; consistent shell
   bias is learnable).
8. Bowls: do NOT special-case in D1 (support-mode reference = consistent overcount;
   RGB disambiguates container type; top-view bowl interior is ill-posed anyway).
   Occlusion is absorbed by the same consistent bias.

## (c) Training variants
- **Variant B (volume text) — favored**: N5k's own ablation had scalar volume BEAT
  RGB-D fusion (16.5% vs 18.8% cal MAE); DPF ceiling over RGB is modest. Whole-ml /
  whole-mm rounding as planned. **Add feature dropout: omit the depth line for
  10–15% of depth-having train records** (coverage is 99.5%, so the no-depth iPhone
  fallback is otherwise OOD). Optionally add coverage_cm2 (strongest correlate after
  volume).
- **Variant A (second image) — the novel/risky arm**: depth_to_height_image with
  **255 = 120 mm** (not 60 — p99 heights hit 113 mm; 60 saturates 10–15% of dishes),
  0 = plane, invalid → 0, render ~320x240 (full-res second image ≈ doubles sequence
  length for no info gain).

## (d) iPhone (D5)
- ARKit sceneDepth 256x192 metric meters + ARConfidenceLevel; use smoothedSceneDepth
  (analogue of the 8-s average). AVFoundation LiDAR AVDepthData also metric. Native
  LiDAR ~576 pts ML-upsampled: volume integral mostly cancels smoothing; expect mild
  underestimate on spiky food. Literature: ±1 cm on small objects; 14% weight MAE on
  48 real meals (preclinical study); SnapCalorie = existence proof.
- **Plane-fit quality, not LiDAR noise, is the bottleneck**: ±2–3 mm plane bias over
  ~450 cm² ≈ ±100–130 ml. RANSAC on the table annulus around the plate (no glass trap
  on iPhone); require ARKit gravity vector agrees with fitted normal within ~10°;
  use ARCamera.intrinsics scaled to depth-map resolution (never hardcode 465).
- Capture guidance: 35–45 cm top-down (LiDAR min range ~20–25 cm; brackets the 35.9 cm
  training distance — matters for variant A; variant B features are distance-invariant).
  Reject frames with >30% low-confidence depth in the plate region.
- Parity: on iPhone the plate sits on the table → "height above support surface,
  plate shell included" = same feature definition as the N5k glass reference.

## (e) D1-fix checklist (each item corrects the draft implementation)
1. Unit test: plate-mode depth ∈ [0.33, 0.37] m over ≥5 dishes (not border ≈ 0.359).
2. f: 616 → 465.1.
3. Reference: border RANSAC → central-disk mode in [0.30, 0.398] m + robust fit ±4 mm.
4. Invalid: add raw > 4200 cut.
5. Noise floor: 4 mm.
6. Add density assertion: median mass/volume over ≥20 dishes ∈ [0.3, 1.2] g/ml.
7. max_height_mm: max → p99.
8. Height image: 255 = 120 mm, ~320x240 output.
9. (D2) depth-line dropout 10–15%.

## (f) D2 pinned spec (Fable design pass, 2026-07-26 — implementation-only for Sonnet)
1. `01_select_images.py --with-depth`: copy `imagery/realsense_overhead/depth_raw.png`
   → `dataset_clean/<dish>/depth_raw.png` for dishes already in dataset_clean;
   idempotent (skip existing). No other behavior changes.
2. `02_prepare_finetune.py --depth-mode {none,image,text}`, default `none` =
   byte-identical to current output (verify: regenerate to a temp dir, diff against
   finetune_data_v2/ — must be empty).
3. Depth stats cache: `dataset_clean/depth_stats.csv` (dish_id, status, volume_ml,
   max_height_mm, coverage_cm2). status ∈ {ok, missing, corrupt, fit_failed}. Computed
   via depth_features.{load_depth,plane_fit,food_stats} on first --depth-mode≠none run;
   reused if present; `--recompute-depth-stats` forces. Any exception or NaN → non-ok
   status → the dish gets a plain (v5-format) record in both variants.
4. Split identity: dish→{train,valid,test} assignment MUST equal finetune_data_v2's
   (same seed/logic untouched). Assert by comparing dish-id sets per split against the
   existing v2 JSONLs.
5. **Feature dropout (both variants, train split only)**: drop depth for 12% of
   depth-ok train dishes, selected deterministically from dish_id
   (`md5(dish_id) % 100 < 12` — stable, variant-independent, so both variants drop the
   same dishes). Dropped/valid/test handling: valid+test records always keep depth when
   status=ok (no dropout outside train).
6. Variant B (`text`, out-dir `finetune_data_v2d_txt/`): user text =
   BASE + `"\n\nEstimated food volume from depth sensor: ~{volume_ml:.0f} ml (max height {max_height_mm:.0f} mm)."`
   No coverage_cm2 in the line (keep iOS parity surface minimal). Line absent for
   non-ok or dropped dishes.
7. Variant A (`image`, out-dir `finetune_data_v2d_img/`): render
   `depth_to_height_image` → `dataset_clean/<dish>/height.png` (grayscale L, 320x240).
   Record: top-level `"images": [rgb, height]`; user content =
   [image rgb, image height, text]. Non-ok/dropped → single-image v5-format record.
8. Report: counts per split/variant, depth-ok coverage %, dropout count, 2 spot-check
   records per variant, and run `00_smoke_test.py` on both variant train.jsonl files —
   report pass/fail but do NOT modify the smoke test (two-image accounting is D3/Fable).

Sources: Nutrition5k README + paper (arXiv 2103.03375) · DPF-Nutrition (arXiv
2310.11702 / PMC10706621) · Sci Rep 2021 iPhone LiDAR accuracy · preclinical
depth-food-volumetry (PMC7142738) · ARKit sceneDepth docs · dining-bowl modeling
(PMC11435675) · MUSEFood (arXiv 1903.07437).
