import SwiftUI

#if canImport(UIKit)
import UIKit
#endif

/// `#scr-camera` — the full-screen viewfinder (tab bar hidden, spec §5):
/// live preview, ✕ close (top-left), gravity-driven level indicator centered
/// on the plate, hint text + shutter at the bottom over a scrim gradient.
///
/// The prototype's `.vf-top` distance chip and "LiDAR depth · capturing portion
/// size" pill are depth affordances: they render only when the capture service
/// reports `supportsDepthCapture`, which is false until depth-plan D5 lands —
/// so today they simply don't appear (spec §5: "hide chip on non-LiDAR devices").
struct CameraScreen: View {
    @ObservedObject var controller: ScanFlowController

    @State private var authorization: CameraAuthorization = .notDetermined
    // Starts visibly off-level so the hint begins at "Hold flat over the plate"
    // until real gravity arrives (prototype's bubble also starts adrift).
    @State private var gravity = GravityReading(x: 0.6, y: 0.5)
    @Environment(\.accessibilityReduceMotion) private var reduceMotion

    /// Camera-overlay palette: these sit on top of the live image, so they are
    /// fixed dark-scene colors from the prototype CSS, not light/dark tokens.
    private enum Overlay {
        static let background = Color(hex: 0x0D0F0D)     // .scr.camera
        static let levelGreen = Color(hex: 0x4FD394)      // .level .bubble
        static let levelAmber = Color(hex: 0xF28C3B)      // .level.off .bubble
        static let hintInk = Color(hex: 0xE6E9E5)         // .hint
        static let closeInk = Color(hex: 0xE8EBE7)        // .vf-close
    }

    var body: some View {
        ZStack {
            Overlay.background.ignoresSafeArea()

            if authorization == .denied {
                deniedView
            } else {
                controller.captureService.makePreviewView()

                if controller.capturePreferences.captureCoachingEnabled {
                    levelIndicator
                }

                if controller.capturePreferences.useLiDARDepth && controller.captureService.supportsDepthCapture {
                    depthAffordances
                }

                bottomControls
            }

            closeButton
        }
        .task {
            authorization = await controller.captureService.requestAccess()
            guard authorization == .authorized else { return }
            controller.captureService.startSession()
            for await reading in controller.captureService.gravityUpdates() {
                if reduceMotion {
                    gravity = reading
                } else {
                    withAnimation(.easeOut(duration: 0.4)) {
                        gravity = reading
                    }
                }
            }
        }
        .onDisappear {
            controller.captureService.stopSession()
        }
        .alert(
            "Capture failed",
            isPresented: Binding(
                get: { controller.captureError != nil },
                set: { if !$0 { controller.captureError = nil } }
            )
        ) {
            Button("OK", role: .cancel) { controller.captureError = nil }
        } message: {
            Text(controller.captureError ?? "Unknown error")
        }
    }

    // MARK: - Level indicator (`.level` + `.bubble`)

    /// 88pt outer ring + 34pt inner target, centered where the plate sits
    /// (prototype: `top:44%`); the 22pt bubble drifts with device gravity and
    /// snaps to center when flat. Ring/bubble turn amber while off-level.
    private var levelIndicator: some View {
        GeometryReader { geometry in
            ZStack {
                Circle()
                    .stroke(Color.white.opacity(0.35), lineWidth: 1.5)
                    .frame(width: 88, height: 88)
                Circle()
                    .stroke(Color.white.opacity(0.55), lineWidth: 1.5)
                    .frame(width: 34, height: 34)
                Circle()
                    .fill(gravity.isLevel ? Overlay.levelGreen : Overlay.levelAmber)
                    .frame(width: 22, height: 22)
                    .offset(bubbleOffset)
            }
            .position(x: geometry.size.width / 2, y: geometry.size.height * 0.44)
        }
        .allowsHitTesting(false)
        .accessibilityHidden(true)
    }

    /// Gravity x/y (≈0 when flat, toward ±1 tilted) mapped onto the prototype's
    /// bubble travel (±26px at full drift), clamped to stay inside the ring.
    private var bubbleOffset: CGSize {
        let scale: Double = 52
        let limit: Double = 26
        return CGSize(
            width: min(limit, max(-limit, gravity.x * scale)),
            height: min(limit, max(-limit, gravity.y * scale))
        )
    }

    // MARK: - Depth affordances (`.vf-top`, dormant until D5)

    /// Distance chip + LiDAR pill. Unreachable today (`supportsDepthCapture`
    /// is false everywhere); layout kept to the prototype so D5 only has to
    /// feed it data.
    private var depthAffordances: some View {
        VStack(spacing: 8) {
            HStack(spacing: 8) {
                Circle().fill(Overlay.levelAmber).frame(width: 8, height: 8)
                Text("— cm")
                    .font(.system(size: 13, weight: .bold))
                    .monospacedDigit()
            }
            .foregroundStyle(Overlay.levelAmber)
            .padding(.vertical, 8)
            .padding(.horizontal, 14)
            .background(Color(hex: 0x0A0C0A).opacity(0.72))
            .clipShape(Capsule())

            Text("LiDAR depth · capturing portion size")
                .font(.system(size: 11, weight: .semibold))
                .tracking(0.44)
                .foregroundStyle(Color(hex: 0xC8CFC9))
                .padding(.vertical, 5)
                .padding(.horizontal, 11)
                .background(Color(hex: 0x0A0C0A).opacity(0.6))
                .clipShape(Capsule())
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .top)
        .padding(.top, 12)
    }

    // MARK: - Bottom controls (`.vf-bottom`: hint + shutter)

    private var bottomControls: some View {
        VStack(spacing: 14) {
            // Hint text is a coaching overlay (spec §5); the shutter directly
            // below it is NEVER gated by this toggle — capture always works.
            if controller.capturePreferences.captureCoachingEnabled {
                Text(gravity.isLevel ? "Framed — capture when ready" : "Hold flat over the plate")
                    .font(.system(size: 13, weight: .semibold))
                    .foregroundStyle(Overlay.hintInk)
                    .shadow(color: .black.opacity(0.5), radius: 6, y: 1)
            }

            ShutterButton(reduceMotion: reduceMotion) {
                Task { await controller.shutterTapped() }
            }
        }
        .padding(.bottom, 34)
        .padding(.top, 40)
        .frame(maxWidth: .infinity)
        .background(
            LinearGradient(
                colors: [.clear, .black.opacity(0.55)],
                startPoint: .top,
                endPoint: .bottom
            )
        )
        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .bottom)
    }

    // MARK: - Close (`.vf-close`)

    private var closeButton: some View {
        Button {
            controller.closeCamera()
        } label: {
            Image(systemName: "xmark")
                .font(.system(size: 15, weight: .semibold))
                .foregroundStyle(Overlay.closeInk)
                .frame(width: 34, height: 34)
                .background(Color(hex: 0x0A0C0A).opacity(0.6))
                .clipShape(Circle())
        }
        .buttonStyle(.plain)
        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
        .padding(.top, 10)
        .padding(.leading, 20)
        .accessibilityLabel("Close camera")
    }

    // MARK: - Permission denied

    /// Sensible denied-state UI (no prototype counterpart — the prototype has
    /// no permission model): explain, offer Settings on iOS, keep ✕ available.
    private var deniedView: some View {
        VStack(spacing: 14) {
            Image(systemName: "camera.fill")
                .font(.system(size: 40, weight: .light))
                .foregroundStyle(.white.opacity(0.7))
            Text("Camera access needed")
                .font(.system(size: 17, weight: .bold))
                .foregroundStyle(.white)
            Text("SeeCal uses the camera to analyze your meals — photos never leave this iPhone. Allow camera access in Settings to scan.")
                .font(.system(size: 13.5))
                .foregroundStyle(.white.opacity(0.75))
                .multilineTextAlignment(.center)
                .padding(.horizontal, 44)
            #if os(iOS)
            Button {
                if let url = URL(string: UIApplication.openSettingsURLString) {
                    UIApplication.shared.open(url)
                }
            } label: {
                Text("Open Settings")
                    .font(.system(size: 16, weight: .bold))
                    .foregroundStyle(.white)
                    .padding(.vertical, 12)
                    .padding(.horizontal, 28)
                    .background(Theme.basil)
                    .clipShape(RoundedRectangle(cornerRadius: 14, style: .continuous))
            }
            .buttonStyle(.plain)
            .padding(.top, 6)
            #endif
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }
}

// MARK: - Shutter (`.shutter`)

/// 74pt ring (4pt white border) around a 58pt white fill; the fill scales to
/// 0.88 while pressed (`.shutter:active i{transform:scale(.88)}`), unless
/// Reduce Motion is on.
private struct ShutterButton: View {
    let reduceMotion: Bool
    let action: () -> Void

    var body: some View {
        Button(action: action) {
            Circle()
                .stroke(Color.white, lineWidth: 4)
                .frame(width: 74, height: 74)
                .contentShape(Circle())
        }
        .buttonStyle(ShutterButtonStyle(reduceMotion: reduceMotion))
        .accessibilityLabel("Capture photo")
    }

    private struct ShutterButtonStyle: ButtonStyle {
        let reduceMotion: Bool

        func makeBody(configuration: Configuration) -> some View {
            configuration.label
                .overlay {
                    Circle()
                        .fill(Color.white)
                        .frame(width: 58, height: 58)
                        .scaleEffect(!reduceMotion && configuration.isPressed ? 0.88 : 1)
                        .animation(reduceMotion ? nil : .easeOut(duration: 0.1), value: configuration.isPressed)
                }
        }
    }
}
