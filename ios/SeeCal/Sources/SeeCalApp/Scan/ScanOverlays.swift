import SwiftUI

/// `.toast` — dark (ink-colored) pill, centered near the bottom (~160pt up in
/// the prototype), auto-dismissed by `ScanFlowController.showToast` after
/// ~1.9 s. This view is purely presentational; lifetime lives in the controller.
struct ToastView: View {
    let text: String

    var body: some View {
        Text(text)
            .font(.system(size: 13.5, weight: .semibold))
            .foregroundStyle(Theme.appBg)
            .padding(.vertical, 11)
            .padding(.horizontal, 18)
            .background(Theme.appInk)
            .clipShape(Capsule())
            .lineLimit(1)
    }
}

/// `.ready-banner` — the persistent basil pill above the tab bar shown when an
/// analysis finished while the user was elsewhere: "✓ Meal analyzed — tap to
/// review". Tapping opens the result sheet from anywhere.
struct ReadyBannerView: View {
    let action: () -> Void

    var body: some View {
        Button(action: action) {
            HStack(spacing: 8) {
                Image(systemName: "checkmark")
                    .font(.system(size: 13, weight: .bold))
                Text("Meal analyzed — tap to review")
                    .font(.system(size: 13.5, weight: .bold))
            }
            .foregroundStyle(.white)
            .padding(.vertical, 12)
            .padding(.horizontal, 20)
            .background(Theme.basil)
            .clipShape(Capsule())
            .shadow(color: Theme.basil.opacity(0.45), radius: 12, x: 0, y: 8)
        }
        .buttonStyle(.plain)
        .accessibilityLabel("Analysis ready, review meal")
    }
}
