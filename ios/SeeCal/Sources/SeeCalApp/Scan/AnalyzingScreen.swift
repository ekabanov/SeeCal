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

            steps
                .padding(.top, 26)
                .padding(.horizontal, 34)

            if case let .error(message, _) = controller.phase {
                errorBlock(message: message)
                    .padding(.top, 26)
                    .padding(.horizontal, 34)
            }

            Spacer(minLength: 0)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .top)
        .background(Theme.appBg.ignoresSafeArea())
    }

    // MARK: - Derived stage state

    private var photoPath: String? {
        switch controller.phase {
        case let .analyzing(photoPath), let .error(_, photoPath):
            return photoPath
        case let .presentingResult(draft), let .ready(draft):
            return draft.imagePath
        case .idle, .capturing:
            return nil
        }
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
        ZStack(alignment: .top) {
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
        VStack(alignment: .leading, spacing: 16) {
            stageRow(
                title: "Identifying foods",
                subtitle: identifyingSubtitle,
                state: identifyingState
            )
            stageRow(
                title: "Estimating nutrition",
                subtitle: estimatingSubtitle,
                state: estimatingState
            )
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }

    private var identifyingState: StageState {
        if completedDraft != nil { return .done }
        return isRunning ? .running : .pending
    }

    private var estimatingState: StageState {
        completedDraft != nil ? .done : .pending
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

        @State private var pulsing = false

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
                    Circle()
                        .fill(Theme.basil)
                        .frame(width: 10, height: 10)
                        .scaleEffect(reduceMotion ? 1 : (pulsing ? 1 : 0.7))
                        .opacity(reduceMotion ? 1 : (pulsing ? 1 : 0.6))
                        .onAppear {
                            guard !reduceMotion else { return }
                            withAnimation(.easeInOut(duration: 0.5).repeatForever(autoreverses: true)) {
                                pulsing = true
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
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }
}

// MARK: - Scan line (`.scanline` + `@keyframes sweep`)

/// A 64pt gradient band with a bright bottom edge sweeping top→bottom on a
/// 1.6 s loop. Reduce Motion pins it statically at 40% height, exactly like the
/// prototype's `@media (prefers-reduced-motion)` override.
private struct ScanlineView: View {
    let reduceMotion: Bool

    @State private var sweeping = false

    private let bandHeight: CGFloat = 64
    private let glow = Color(hex: 0x4FD394)

    var body: some View {
        GeometryReader { geometry in
            band
                .frame(height: bandHeight)
                .offset(y: offsetY(in: geometry.size.height))
                .onAppear {
                    guard !reduceMotion else { return }
                    withAnimation(.easeInOut(duration: 1.6).repeatForever(autoreverses: false)) {
                        sweeping = true
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

    private func offsetY(in height: CGFloat) -> CGFloat {
        if reduceMotion {
            return height * 0.4 // static position under Reduce Motion
        }
        return sweeping ? height * 1.05 : -bandHeight
    }
}
