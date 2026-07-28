import SwiftUI

/// Branded handoff between iOS's launch screen and the app becoming usable.
///
/// Loading the bundled 4B model can take long enough to feel like a real app
/// state, so this uses the app-icon artwork and the product design tokens
/// instead of exposing an implementation detail ("Loading MLX model").
struct StartupLoadingView: View {
    var body: some View {
        ZStack {
            Theme.appBg
                .ignoresSafeArea()

            RadialGradient(
                colors: [
                    Theme.basilSoft.opacity(0.9),
                    Theme.appBg.opacity(0)
                ],
                center: .top,
                startRadius: 20,
                endRadius: 430
            )
            .ignoresSafeArea()

            VStack(spacing: 0) {
                Spacer(minLength: 40)

                startupIcon(size: 128, cornerRadius: 29)

                Text("SeeCal")
                    .font(.system(size: 32, weight: .bold))
                    .tracking(-0.7)
                    .foregroundStyle(Theme.appInk)
                    .padding(.top, 24)

                Text("Preparing your on-device model…")
                    .font(.system(size: 15, weight: .semibold))
                    .foregroundStyle(Theme.appInk2)
                    .multilineTextAlignment(.center)
                    .padding(.top, 8)

                ProgressView()
                    .controlSize(.regular)
                    .tint(Theme.basil)
                    .padding(.top, 22)
                    .accessibilityLabel("Preparing your on-device model")

                Text("This can take a moment.")
                    .font(.system(size: 12.5, weight: .medium))
                    .foregroundStyle(Theme.appInk2)
                    .padding(.top, 10)

                Spacer(minLength: 32)

                Label("Photos stay on this iPhone", systemImage: "lock.fill")
                    .font(.system(size: 12.5, weight: .semibold))
                    .foregroundStyle(Theme.basil)
                    .padding(.vertical, 11)
                    .padding(.horizontal, 15)
                    .background(Theme.basilSoft)
                    .clipShape(Capsule())
                    .padding(.bottom, 28)
            }
            .padding(.horizontal, 24)
        }
    }
}

/// Keeps startup failures in the same visual context as loading while leaving
/// the underlying error visible for device debugging.
struct StartupFailureView: View {
    let message: String
    let retry: () -> Void

    var body: some View {
        ZStack {
            Theme.appBg
                .ignoresSafeArea()

            VStack(spacing: 0) {
                startupIcon(size: 88, cornerRadius: 20)

                Text("Model couldn’t start")
                    .font(.system(size: 24, weight: .bold))
                    .foregroundStyle(Theme.appInk)
                    .padding(.top, 22)

                Text("SeeCal couldn’t prepare the on-device model.")
                    .font(.system(size: 15, weight: .medium))
                    .foregroundStyle(Theme.appInk2)
                    .multilineTextAlignment(.center)
                    .padding(.top, 8)

                Text(message)
                    .font(.system(size: 12, weight: .regular, design: .monospaced))
                    .foregroundStyle(Theme.appInk2)
                    .multilineTextAlignment(.center)
                    .lineLimit(5)
                    .padding(.top, 16)
                    .textSelection(.enabled)

                Button(action: retry) {
                    Text("Try again")
                        .font(.system(size: 15, weight: .bold))
                        .foregroundStyle(.white)
                        .frame(maxWidth: .infinity)
                        .padding(.vertical, 14)
                        .background(Theme.basil)
                        .clipShape(RoundedRectangle(cornerRadius: 14, style: .continuous))
                }
                .buttonStyle(.plain)
                .padding(.top, 24)
            }
            .frame(maxWidth: 360)
            .padding(.horizontal, 28)
        }
    }
}

private func startupIcon(size: CGFloat, cornerRadius: CGFloat) -> some View {
    Image("SeeCalStartupIcon")
        .resizable()
        .scaledToFit()
        .frame(width: size, height: size)
        .clipShape(RoundedRectangle(cornerRadius: cornerRadius, style: .continuous))
        .overlay {
            RoundedRectangle(cornerRadius: cornerRadius, style: .continuous)
                .stroke(.white.opacity(0.12), lineWidth: 1)
        }
        .shadow(color: Theme.shadowColor.opacity(0.18), radius: 20, x: 0, y: 12)
        .accessibilityHidden(true)
}
