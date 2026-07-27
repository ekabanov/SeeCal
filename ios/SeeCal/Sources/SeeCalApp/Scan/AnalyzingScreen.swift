import SwiftUI
import SeeCalDomain

/// `#scr-analyzing` — the captured photo card with a sweeping scan line, over a
/// staged checklist driven by REAL inference state (spec §5). The prototype's
/// first stage ("Measuring portion depth") is omitted entirely — no depth until
/// D5 — and its `.simnote` ("compressed to 4 s in this prototype") does not
/// exist here because the app runs real inference.
///
/// Stage semantics: "Identifying foods" pulses for the whole model call (a
/// single inference produces everything); on completion BOTH stages check off,
/// with the parse's real item names in the first done-subtitle. Errors render
/// in the same layout with the runtime's surfaced message + Retry (same photo).
struct AnalyzingScreen: View {
    @ObservedObject var controller: ScanFlowController

    @Environment(\.accessibilityReduceMotion) private var reduceMotion

    /// When the current analysis began, captured so the two visible stages can
    /// advance over time. There is a single inference call with no observable
    /// sub-stages, so — like the prototype, which fully simulates its stages —
    /// "Identifying foods" runs during the vision-prefill window and hands off to
    /// "Estimating nutrition" for the (longer) token-generation tail. Without this
    /// the second stage was coded to only ever be pending/done and never lit up.
    @State private var analyzeStart: Date?

    /// Rough vision-prefill → token-generation handoff. Prefill is a few seconds;
    /// generation dominates the ~15–20 s wait, so the pivot is early.
    private let stagePivot: TimeInterval = 4

    private enum StageState {
        case pending
        case running
        case done
    }

    var body: some View {
        VStack(spacing: 0) {
            shotCard
                .padding(.top, 22)
                .padding(.horizontal, 34)

            if isNotFood {
                notFoodBlock
                    .padding(.top, 26)
                    .padding(.horizontal, 34)
            } else {
                steps
                    .padding(.top, 26)
                    .padding(.horizontal, 34)

                if case let .error(message, _) = controller.phase {
                    errorBlock(message: message)
                        .padding(.top, 26)
                        .padding(.horizontal, 34)
                }
            }

            Spacer(minLength: 0)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .top)
        .background(Theme.appBg.ignoresSafeArea())
        .onAppear {
            if isRunning, analyzeStart == nil { analyzeStart = Date() }
        }
        .onChange(of: isRunning) { _, running in
            analyzeStart = running ? Date() : nil
        }
    }

    // MARK: - Derived stage state

    private var photoPath: String? {
        switch controller.phase {
        case let .analyzing(photoPath), let .error(_, photoPath), let .notFood(photoPath):
            return photoPath
        case let .presentingResult(draft), let .ready(draft):
            return draft.imagePath
        case .idle, .capturing:
            return nil
        }
    }

    private var isNotFood: Bool {
        if case .notFood = controller.phase { return true }
        return false
    }

    private var isRunning: Bool {
        if case .analyzing = controller.phase { return true }
        return false
    }

    private var completedDraft: MealEditDraft? {
        switch controller.phase {
        case let .presentingResult(draft), let .ready(draft):
            return draft
        default:
            return nil
        }
    }

    // MARK: - Shot card (`.shot` + `.scanline`)

    private var shotCard: some View {
        Group {
            if let photoPath, let image = PlatformImageLoader.image(atPath: photoPath) {
                image
                    .resizable()
                    .aspectRatio(contentMode: .fill)
            } else {
                Color(hex: 0x262420) // .shot background
            }
        }
        .frame(maxWidth: .infinity)
        .aspectRatio(240.0 / 190.0, contentMode: .fit) // prototype shot viewBox
        // Scanline is an OVERLAY, not a ZStack sibling: ScanlineView wraps a
        // greedy GeometryReader, and as a sibling it would expand the container to
        // full height (blowing the photo up and pushing the step rows off-screen).
        // As an overlay it is bounded by the aspect-constrained photo box.
        .overlay {
            if isRunning {
                ScanlineView(reduceMotion: reduceMotion)
            }
        }
        .clipShape(RoundedRectangle(cornerRadius: 22, style: .continuous))
        .cardElevation()
        .accessibilityLabel("Captured meal photo")
    }

    // MARK: - Steps (`.steps` / `.step`)

    private var steps: some View {
        // Periodic ticks (not per-frame — this only needs to cross the pivot)
        // recompute elapsed so the running stage advances over time.
        TimelineView(.periodic(from: .now, by: 0.25)) { context in
            let elapsed = analyzeStart.map { max(0, context.date.timeIntervalSince($0)) } ?? 0
            VStack(alignment: .leading, spacing: 16) {
                stageRow(
                    title: "Identifying foods",
                    subtitle: identifyingSubtitle,
                    state: identifyingState(elapsed: elapsed)
                )
                stageRow(
                    title: "Estimating nutrition",
                    subtitle: estimatingSubtitle,
                    state: estimatingState(elapsed: elapsed)
                )
            }
            .frame(maxWidth: .infinity, alignment: .leading)
        }
    }

    private func identifyingState(elapsed: TimeInterval) -> StageState {
        if completedDraft != nil { return .done }
        guard isRunning else { return .pending }
        // Runs during the prefill window, then checks off as generation begins.
        return elapsed < stagePivot ? .running : .done
    }

    private func estimatingState(elapsed: TimeInterval) -> StageState {
        if completedDraft != nil { return .done }
        guard isRunning else { return .pending }
        // Lights up once the handoff point passes and stays running until done.
        return elapsed < stagePivot ? .pending : .running
    }

    private var identifyingSubtitle: String {
        if let draft = completedDraft {
            return ScanFlowController.foodsFoundSubtitle(items: draft.items)
        }
        return "Vision model reading the photo"
    }

    private var estimatingSubtitle: String {
        completedDraft != nil ? "Done" : "Per-ingredient calories & macros"
    }

    private func stageRow(title: String, subtitle: String, state: StageState) -> some View {
        HStack(alignment: .top, spacing: 12) {
            StageIcon(state: state, reduceMotion: reduceMotion)
                .padding(.top, 1)

            VStack(alignment: .leading, spacing: 1) {
                Text(title)
                    .font(.system(size: 15, weight: .semibold))
                    .foregroundStyle(Theme.appInk)
                Text(subtitle)
                    .font(.system(size: 12.5))
                    .foregroundStyle(Theme.appInk2)
            }
        }
        .opacity(state == .pending ? 0.38 : 1) // .step{opacity:.38} vs .on/.done
        .accessibilityElement(children: .combine)
    }

    /// 24pt circle: `app-line` border when pending; basil border + pulsing 10pt
    /// basil dot while running; filled basil + white check when done.
    private struct StageIcon: View {
        let state: StageState
        let reduceMotion: Bool

        // Full pulse cycle (0.5 s each way in the prototype's autoreversing
        // easeInOut).
        private let pulsePeriod: Double = 1.0

        var body: some View {
            ZStack {
                Circle()
                    .fill(state == .done ? Theme.basil : Color.clear)
                Circle()
                    .stroke(state == .pending ? Theme.appLine : Theme.basil, lineWidth: 2)

                switch state {
                case .pending:
                    EmptyView()
                case .running:
                    // Time-based pulse (see ScanlineView): a `repeatForever`
                    // animation would be stopped by a tap anywhere on the screen.
                    Group {
                        if reduceMotion {
                            pulseDot(scale: 1, opacity: 1)
                        } else {
                            TimelineView(.animation) { context in
                                let p = pulsePhase(at: context.date) // 0…1
                                pulseDot(scale: 0.7 + 0.3 * p, opacity: 0.6 + 0.4 * p)
                            }
                        }
                    }
                case .done:
                    Image(systemName: "checkmark")
                        .font(.system(size: 11, weight: .bold))
                        .foregroundStyle(.white)
                }
            }
            .frame(width: 24, height: 24)
        }

        private func pulseDot(scale: Double, opacity: Double) -> some View {
            Circle()
                .fill(Theme.basil)
                .frame(width: 10, height: 10)
                .scaleEffect(scale)
                .opacity(opacity)
        }

        /// Smooth 0→1→0 pulse over `pulsePeriod`, derived from the timeline clock.
        private func pulsePhase(at date: Date) -> Double {
            let t = date.timeIntervalSinceReferenceDate.truncatingRemainder(dividingBy: pulsePeriod)
            return (1 - cos(2 * .pi * t / pulsePeriod)) / 2
        }
    }

    // MARK: - Not food (v7 refusal): friendly terminal state, no red error

    /// Shown when the model refused (`{"not_food": true}`): the photo isn't a
    /// meal. Reuses the error block's button layout but with neutral copy and a
    /// basil-forward "New scan" primary — this is a normal outcome, not a fault.
    private var notFoodBlock: some View {
        VStack(alignment: .leading, spacing: 14) {
            VStack(alignment: .leading, spacing: 4) {
                Text("No food detected")
                    .font(.system(size: 17, weight: .bold))
                    .foregroundStyle(Theme.appInk)
                Text("This photo doesn’t look like a meal. Point the camera at your food and try again.")
                    .font(.system(size: 13))
                    .foregroundStyle(Theme.appInk2)
                    .fixedSize(horizontal: false, vertical: true)
            }

            Button {
                controller.newScanAfterError()
            } label: {
                Text("New scan")
                    .font(.system(size: 16, weight: .bold))
                    .foregroundStyle(.white)
                    .frame(maxWidth: .infinity)
                    .padding(.vertical, 14)
                    .background(Theme.basil)
                    .clipShape(RoundedRectangle(cornerRadius: 14, style: .continuous))
            }
            .buttonStyle(.plain)
            .accessibilityHint("Discards this photo and opens the camera")

            // Same-photo retry, in case the model misjudged a borderline shot.
            Button {
                controller.retry()
            } label: {
                Text("Try again")
                    .font(.system(size: 16, weight: .bold))
                    .foregroundStyle(Theme.appInk)
                    .frame(maxWidth: .infinity)
                    .padding(.vertical, 14)
                    .background(Theme.appCard)
                    .clipShape(RoundedRectangle(cornerRadius: 14, style: .continuous))
            }
            .buttonStyle(.plain)
            .accessibilityHint("Runs the analysis again on the same photo")
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .accessibilityElement(children: .contain)
    }

    // MARK: - Error + Retry (spec §5: same staged UI, runtime's message)

    private func errorBlock(message: String) -> some View {
        VStack(alignment: .leading, spacing: 14) {
            Text(message)
                .font(.system(size: 13))
                .foregroundStyle(Theme.danger)
                .fixedSize(horizontal: false, vertical: true)

            Button {
                controller.retry()
            } label: {
                Text("Retry")
                    .font(.system(size: 16, weight: .bold))
                    .foregroundStyle(.white)
                    .frame(maxWidth: .infinity)
                    .padding(.vertical, 14)
                    .background(Theme.basil)
                    .clipShape(RoundedRectangle(cornerRadius: 14, style: .continuous))
            }
            .buttonStyle(.plain)
            .accessibilityHint("Runs the analysis again on the same photo")

            // Secondary escape hatch: a failed analysis must never dead-end —
            // abandon this photo and go back to the camera for a fresh capture.
            Button {
                controller.newScanAfterError()
            } label: {
                Text("New scan")
                    .font(.system(size: 16, weight: .bold))
                    .foregroundStyle(Theme.appInk)
                    .frame(maxWidth: .infinity)
                    .padding(.vertical, 14)
                    .background(Theme.appCard)
                    .clipShape(RoundedRectangle(cornerRadius: 14, style: .continuous))
            }
            .buttonStyle(.plain)
            .accessibilityHint("Discards this photo and opens the camera")
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }
}

// MARK: - Scan line (`.scanline` + `@keyframes sweep`)

/// A 64pt gradient band with a bright bottom edge sweeping top→bottom on a
/// 1.6 s loop. Reduce Motion pins it statically at 40% height, exactly like the
/// prototype's `@media (prefers-reduced-motion)` override.
///
/// The sweep is driven from wall-clock time via `TimelineView(.animation)`, NOT
/// a `withAnimation(...repeatForever)` started in `.onAppear`. A `repeatForever`
/// animation is interrupted by any unrelated transaction — e.g. a tap anywhere
/// on the analyzing screen combined with the parent's `.animation(value:)`
/// scopes — which visibly stopped the sweep. A time-based redraw can't be
/// cancelled by taps or re-renders.
private struct ScanlineView: View {
    let reduceMotion: Bool

    private let bandHeight: CGFloat = 64
    private let period: Double = 1.6
    private let glow = Color(hex: 0x4FD394)

    var body: some View {
        GeometryReader { geometry in
            let height = geometry.size.height
            Group {
                if reduceMotion {
                    band
                        .frame(height: bandHeight)
                        .offset(y: height * 0.4) // static under Reduce Motion
                } else {
                    TimelineView(.animation) { context in
                        band
                            .frame(height: bandHeight)
                            .offset(y: offsetY(at: context.date, in: height))
                    }
                }
            }
        }
        .clipped()
        .allowsHitTesting(false)
        .accessibilityHidden(true)
    }

    private var band: some View {
        VStack(spacing: 0) {
            LinearGradient(
                colors: [.clear, glow.opacity(0.28), .clear],
                startPoint: .top,
                endPoint: .bottom
            )
            Rectangle()
                .fill(glow.opacity(0.7))
                .frame(height: 1.5)
        }
    }

    /// Sweeps linearly from `-bandHeight` (above the photo) to `height * 1.05`
    /// (just past the bottom) every `period` seconds, derived purely from the
    /// timeline's clock so it never pauses on interaction.
    private func offsetY(at date: Date, in height: CGFloat) -> CGFloat {
        let phase = date.timeIntervalSinceReferenceDate
            .truncatingRemainder(dividingBy: period) / period // 0…1
        let travel = height * 1.05 + bandHeight
        return -bandHeight + CGFloat(phase) * travel
    }
}
