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
    @State private var barcodeAmount = ""
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

                if controller.captureService.supportsDepthCapture {
                    depthAffordances
                }

                if case .found = controller.barcodeState {
                    barcodeTarget
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
        .task(id: authorization) {
            guard authorization == .authorized else { return }
            for await barcode in controller.captureService.barcodeUpdates() {
                controller.barcodeDetected(barcode)
            }
        }
        .onChange(of: controller.barcodeState) { _, state in
            if case let .found(product) = state {
                barcodeAmount = String(Int((product.defaultAmount ?? 100).rounded()))
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

    private var barcodeTarget: some View {
        RoundedRectangle(cornerRadius: 13, style: .continuous)
            .stroke(Overlay.levelGreen, lineWidth: 2)
            .frame(width: 190, height: 104)
            .overlay(alignment: .top) {
                Text("Barcode detected")
                    .font(.system(size: 10.5, weight: .heavy))
                    .tracking(0.3)
                    .foregroundStyle(Color(hex: 0x8CE5B7))
                    .padding(.vertical, 5)
                    .padding(.horizontal, 9)
                    .background(Color(hex: 0x183C2A))
                    .clipShape(Capsule())
                    .offset(y: -29)
            }
            .shadow(color: Overlay.levelGreen.opacity(0.38), radius: 9)
            .allowsHitTesting(false)
    }

    // MARK: - Bottom controls

    private var bottomControls: some View {
        VStack(spacing: 12) {
            barcodeCard

            // Hint text is a coaching overlay (spec §5); the shutter directly
            // below it is NEVER gated by this toggle — capture always works.
            if controller.capturePreferences.captureCoachingEnabled,
               controller.barcodeState == .idle {
                Text(gravity.isLevel ? "Framed — capture when ready" : "Hold flat over the plate")
                    .font(.system(size: 13, weight: .semibold))
                    .foregroundStyle(Overlay.hintInk)
                    .shadow(color: .black.opacity(0.5), radius: 6, y: 1)
            }

            HStack {
                cameraSideAction(
                    title: "Manual",
                    systemImage: "square.and.pencil"
                ) {
                    controller.beginManualMeal()
                }

                Spacer()

                ShutterButton(reduceMotion: reduceMotion) {
                    Task { await controller.shutterTapped() }
                }

                Spacer()

                VStack(spacing: 5) {
                    Image(systemName: "barcode.viewfinder")
                        .font(.system(size: 20, weight: .medium))
                        .foregroundStyle(Color(hex: 0x8CE5B7))
                        .frame(width: 42, height: 42)
                        .background(Color(hex: 0x0A0C0A).opacity(0.72))
                        .clipShape(Circle())
                        .overlay(Circle().stroke(.white.opacity(0.18), lineWidth: 1))
                    Text("Barcode auto")
                        .font(.system(size: 10.5, weight: .bold))
                        .foregroundStyle(Color(hex: 0xD9DED9))
                }
                .frame(width: 86)
                .accessibilityElement(children: .combine)
                .accessibilityLabel("Barcode detection is automatic")
            }
            .padding(.horizontal, 24)
        }
        .padding(.bottom, 30)
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

    @ViewBuilder
    private var barcodeCard: some View {
        switch controller.barcodeState {
        case .idle:
            EmptyView()
        case let .lookingUp(code):
            HStack(spacing: 10) {
                ProgressView().tint(Theme.basil)
                VStack(alignment: .leading, spacing: 2) {
                    Text("Looking up barcode")
                        .font(.system(size: 14, weight: .bold))
                    Text(code)
                        .font(.system(size: 11))
                        .foregroundStyle(Theme.appInk2)
                        .monospacedDigit()
                }
                Spacer()
            }
            .barcodeCameraCard()
        case let .found(product):
            if let nutrition = product.nutritionPer100 {
                VStack(alignment: .leading, spacing: 10) {
                    barcodeProductHeader(product: product) {
                        controller.dismissBarcode()
                    }
                    Text("\(nutrition.kcal.formattedOneDecimal) kcal · \(nutrition.protein.formattedOneDecimal)p · \(nutrition.fat.formattedOneDecimal)f · \(nutrition.carbs.formattedOneDecimal)c per 100 \(product.amountUnit.symbol)")
                        .font(.system(size: 11))
                        .foregroundStyle(Theme.appInk2)
                        .lineLimit(1)

                    HStack(spacing: 8) {
                        Text("Consumed")
                            .font(.system(size: 11.5, weight: .bold))
                            .foregroundStyle(Theme.appInk2)
                        TextField("100", text: $barcodeAmount)
                            .font(.system(size: 14, weight: .bold))
                            .monospacedDigit()
                            .multilineTextAlignment(.trailing)
                            .barcodeNumericKeyboard()
                            .padding(.vertical, 8)
                            .padding(.horizontal, 9)
                            .frame(width: 72)
                            .background(Theme.appBg)
                            .clipShape(RoundedRectangle(cornerRadius: 10, style: .continuous))
                        Text(product.amountUnit.symbol)
                            .font(.system(size: 11, weight: .bold))
                            .foregroundStyle(Theme.appInk2)
                        Spacer()
                        Button("Review") {
                            guard let amount = parsedBarcodeAmount else { return }
                            controller.reviewBarcodeProduct(product, amount: amount)
                        }
                        .font(.system(size: 12, weight: .heavy))
                        .foregroundStyle(.white)
                        .padding(.vertical, 9)
                        .padding(.horizontal, 13)
                        .background(Theme.basil)
                        .clipShape(RoundedRectangle(cornerRadius: 10, style: .continuous))
                        .disabled(parsedBarcodeAmount == nil)
                        .opacity(parsedBarcodeAmount == nil ? 0.5 : 1)
                    }
                }
                .barcodeCameraCard()
            } else {
                VStack(alignment: .leading, spacing: 10) {
                    barcodeProductHeader(product: product) {
                        controller.dismissBarcode()
                    }
                    Text("This record is missing calories or macros. Finish it manually so missing values aren’t treated as zero.")
                        .font(.system(size: 11.5))
                        .foregroundStyle(Theme.appInk2)
                    barcodeManualButton(product)
                }
                .barcodeCameraCard()
            }
        case let .failed(_, message):
            VStack(alignment: .leading, spacing: 10) {
                HStack {
                    Text("Barcode lookup unavailable")
                        .font(.system(size: 14, weight: .bold))
                    Spacer()
                    Button {
                        controller.dismissBarcode()
                    } label: {
                        Image(systemName: "xmark")
                    }
                    .foregroundStyle(Theme.appInk2)
                }
                Text(message)
                    .font(.system(size: 11.5))
                    .foregroundStyle(Theme.appInk2)
                barcodeManualButton(nil)
            }
            .barcodeCameraCard()
        }
    }

    private func barcodeProductHeader(product: BarcodeProduct, dismiss: @escaping () -> Void) -> some View {
        HStack(alignment: .top, spacing: 10) {
            Image(systemName: "barcode")
                .font(.system(size: 18, weight: .medium))
                .foregroundStyle(Theme.basil)
                .frame(width: 34, height: 34)
                .background(Theme.basilSoft)
                .clipShape(RoundedRectangle(cornerRadius: 10, style: .continuous))
            VStack(alignment: .leading, spacing: 2) {
                Text(product.name)
                    .font(.system(size: 14, weight: .bold))
                    .lineLimit(1)
                Text("\(product.barcode) · \(product.provider)")
                    .font(.system(size: 9.5))
                    .foregroundStyle(Theme.appInk2)
                    .lineLimit(1)
            }
            Spacer()
            Button(action: dismiss) {
                Image(systemName: "xmark")
                    .font(.system(size: 11, weight: .bold))
            }
            .foregroundStyle(Theme.appInk2)
        }
    }

    private func barcodeManualButton(_ product: BarcodeProduct?) -> some View {
        Button("Enter manually") {
            controller.enterBarcodeProductManually(product)
        }
        .font(.system(size: 12, weight: .heavy))
        .foregroundStyle(.white)
        .frame(maxWidth: .infinity)
        .padding(.vertical, 10)
        .background(Theme.basil)
        .clipShape(RoundedRectangle(cornerRadius: 10, style: .continuous))
    }

    private var parsedBarcodeAmount: Double? {
        let normalized = barcodeAmount.replacingOccurrences(of: ",", with: ".")
        guard let value = Double(normalized), value > 0 else { return nil }
        return value
    }

    private func cameraSideAction(
        title: String,
        systemImage: String,
        action: @escaping () -> Void
    ) -> some View {
        Button(action: action) {
            VStack(spacing: 5) {
                Image(systemName: systemImage)
                    .font(.system(size: 18, weight: .medium))
                    .foregroundStyle(Color(hex: 0xF4F6F3))
                    .frame(width: 42, height: 42)
                    .background(Color(hex: 0x0A0C0A).opacity(0.72))
                    .clipShape(Circle())
                    .overlay(Circle().stroke(.white.opacity(0.18), lineWidth: 1))
                Text(title)
                    .font(.system(size: 10.5, weight: .bold))
                    .foregroundStyle(Color(hex: 0xF4F6F3))
            }
            .frame(width: 86)
        }
        .buttonStyle(.plain)
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
            Button("Enter meal manually") {
                controller.beginManualMeal()
            }
            .font(.system(size: 14, weight: .bold))
            .foregroundStyle(.white)
            .padding(.top, 4)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }
}

private extension View {
    func barcodeCameraCard() -> some View {
        self
            .padding(12)
            .background(Theme.appCard)
            .foregroundStyle(Theme.appInk)
            .clipShape(RoundedRectangle(cornerRadius: 16, style: .continuous))
            .shadow(color: .black.opacity(0.35), radius: 15, y: 8)
            .padding(.horizontal, 16)
    }

    @ViewBuilder
    func barcodeNumericKeyboard() -> some View {
        #if os(iOS)
        self.keyboardType(.decimalPad)
        #else
        self
        #endif
    }
}

private extension Double {
    var formattedOneDecimal: String {
        String(format: "%.1f", self)
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
