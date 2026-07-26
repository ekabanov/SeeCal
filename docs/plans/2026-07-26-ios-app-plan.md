# iOS App Execution Plan (from prototype spec)

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development.
> Fresh subagent per task; Fable reviews after each; serialize tasks (shared package).

**Authority:** docs/specs/2026-07-26-app-spec.md (spec wins over existing code).
**Branch:** `ios-app` (off depth-track). **Package:** ios/SeeCal (SwiftPM).
**Build/test:** `swift test -Xcxx -DFMT_CONSTEVAL=` from ios/SeeCal (Xcode 26 workaround).
**Model tiers (user directive):** Fable = plan/review/gates; Opus = complex tasks;
Sonnet = well-specified tasks.

## Global rules
- **Design fidelity (user directive): the committed prototype
  `docs/design/prototype/seecal-prototype.html` is the binding visual spec.** Every
  UI task (P3–P7) must read the prototype's HTML/CSS for its screen and match it —
  layout, spacing, copy, colors, component appearance — noting any platform-forced
  deviations in the task report.
- Evolve existing types (ProfileModels, TrackingModels, MealEditDraft, AppViewModel,
  RootView…); no parallel-world files. GoalEditDraft is RETIRED by P1 (goal is computed).
- Every task: run the FULL package test suite; all green before commit; one commit per
  task, message references the task id.
- No GPU involved anywhere here (training runs concurrently on the Mac).
- UI must support dark mode + Reduce Motion from the start (spec §9).
- Never commit model weights/datasets.

## Tasks
### P1 (Sonnet): Goal engine + profile model
Evolve UserProfile per spec §2 (sex, dob, heightCm, weightKg, activity, weeklyRateKg).
Pure `GoalCalculator` (BMR/TDEE/goal/macro targets/band check) with the spec's
reference vector as a test (BMR 1743, TDEE 2701, goal 2150). Retire GoalEditDraft and
its tests; AppViewModel exposes computed goal. Migrate persisted profile if shape
changed (synthesize defaults; keep .bak behavior).

### P2 (Sonnet): Meal items + migration
MealItem with base-values + linear scaling per spec §2; entry totals derived. Extend
persistence models + migration test (legacy entry → single synthetic item). Evolve
MealEditDraft to per-item gram stepping (±5 g, min 5) serving both new-scan and edit
modes; update its tests.

### P3 (Sonnet): Design system + tab shell
Color/token layer (spec §9, light+dark), card/section-label/chip/stepper/slider
components, 5-slot tab bar with center Scan FAB (overhang + halo), tab-bar-hidden
mode for camera. RootView restructured to Today/History/Scan/Profile/Settings
scaffolds (existing content temporarily parked, compiles + tests green).

### P4 (Sonnet): Today screen
Spec §4: ring card + macro bars, meal rows (photo thumb, tap → edit sheet hook),
privacy chip, day filter. ViewModel unit tests for totals/targets.

### P5 (Opus): Onboarding + Profile
Spec §3 + §7: 6-step wizard (evolve OnboardingDraft), shared components (chips,
steppers, weekly-rate slider with recommended band + warnings), Profile screen with
live goal recompute + transparent math string (exact format from spec §3 step 6).
Tests: draft state machine, math-string formatting, band warnings, skip path.

### P6 (Opus): Scan → Analyzing → Result flow
Spec §5. CaptureService protocol + mock (simulator/test) + AVFoundation impl;
camera UI with coaching overlays (gravity level; distance chip only when depth
available); Analyzing screen driven by real InferenceRuntime state (staged checklist
presentational); background continuation: navigation does not cancel, completion
banner, new-scan clears banner; result/edit sheet per spec (depth meta hidden when
absent); log/discard/save semantics + toasts; error retry path. Tests: flow state
machine incl. navigate-away-then-banner, no-persist-without-confirm, draft reuse.

### P7 (Sonnet): History + Settings
Spec §6 + §8: aggregation (week/month/6-months buckets, weekly averages) in
ProgressModels with unit tests incl. color-class thresholds; chart view with axis
zone + goal chip; stats row; Settings screen (Sync SOON rows, Capture toggles, model
card reading real bundled config versions).

### P8 (Fable): Final review + fix wave + merge decision
Whole-branch review against spec; fix wave; decide merge order vs depth-track (D5
will build on P6's capture service).

## Order
P1 → P2 → P3 → P4 → P5 → P6 → P7 → P8 (strictly serial; shared files).
