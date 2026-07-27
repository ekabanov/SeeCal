# SeeCal backlog

Deferred work items. Not scheduled; picked up when prioritized. Dated when added.

## On-device depth (v6) — real-life depth test (added 2026-07-27)

**Goal:** ship both the v5 (plain) and v6 (depth-augmented) adapters and let the
user switch between them in Settings, so depth can be tested on real handheld
LiDAR captures — the one thing the held-out eval could not answer (v6 ≈ v5 on
rig-captured depth, but real-world LiDAR is a different distribution).

**Why it's a feature, not a toggle:** v6's prompt is v5's plus one measured line —
`Estimated food volume from depth sensor: ~<V> ml (max height <H> mm).` Feeding v6
a v5-style prompt (no line) breaks adapter↔prompt parity → garbage. So v6 needs
that line computed on-device, and computed *on the same scale it was trained on*.

**The hard part — calibration.** `ml/depth_features.py` is hard-calibrated to the
Nutrition5K RealSense rig: `FOCAL_PX = 465.1` (D435 color sensor @ 640×480), fixed
0.359 m overhead distance, and plane-fit thresholds tuned to that rig's glass
platform ("glass-platform trap"). iPhone LiDAR is a different sensor (256×192,
different intrinsics), handheld at variable distance, no glass platform. A naive
port produces volumes on a different scale → v6 gets out-of-distribution input and
the "test" is meaningless (cf. the documented 1.75× bias from a wrong focal length).

**Scope (in effort order):**
1. LiDAR depth capture on device (ARKit / AVFoundation `AVDepthData`) + camera intrinsics.
2. Swift port of `plane_fit` + `food_stats` (volume_ml, max_height_mm).
3. **Calibration** so phone volume estimates land on the training-time scale — the load-bearing part.
4. Settings toggle {v5 (no depth line) | v6 (with depth line)}; both adapters fused/bundled (`ml/fuse.sh adapters_v6 --out-path fused_v6` is trivial).

**Suggested approach:** Fable-planned track (capture → Swift features → calibration
strategy → toggle), then implement. A quick uncalibrated version (rough phone volume,
eyeballed) is possible first but is not a rigorous test — label it as such if shipped.
