import SwiftUI

/// Placeholder for the full-screen camera flow (spec §5, real implementation in
/// P6's `CaptureService`). For P3 this only needs to demonstrate the tab-bar-hidden
/// mechanism: a full-screen black scene with a close button that returns to Today.
public struct ScanStubScreen: View {
    private let onClose: () -> Void

    public init(onClose: @escaping () -> Void) {
        self.onClose = onClose
    }

    public var body: some View {
        ZStack(alignment: .topLeading) {
            Color.black.ignoresSafeArea()

            VStack(spacing: 12) {
                Image(systemName: "camera.fill")
                    .font(.system(size: 40, weight: .light))
                    .foregroundStyle(.white.opacity(0.7))
                Text("Camera capture lands in P6")
                    .font(.system(size: 15, weight: .semibold))
                    .foregroundStyle(.white.opacity(0.7))
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity)

            Button(action: onClose) {
                Image(systemName: "xmark")
                    .font(.system(size: 15, weight: .semibold))
                    .foregroundStyle(.white)
                    .frame(width: 34, height: 34)
                    .background(Color.black.opacity(0.6))
                    .clipShape(Circle())
            }
            .buttonStyle(.plain)
            .padding(.top, 66)
            .padding(.leading, 20)
            .accessibilityLabel("Close camera")
        }
    }
}
