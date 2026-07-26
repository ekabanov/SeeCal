import SwiftUI
import SeeCalDomain

/// Spec §8 / `#scr-settings`: Capture toggles (LiDAR depth, capture coaching —
/// both persisted), a Sync card of disabled "SOON" roadmap rows (no
/// functional toggles — copy must not claim sync exists), and the on-device
/// model card (version/quantization read from the bundled config at runtime,
/// never hardcoded). Matches the prototype's markup + CSS (`.setrow`,
/// `.switch`, `.soonbadge`, `.modelcard`) — see
/// `docs/design/prototype/seecal-prototype.html` lines 828-854.
public struct SettingsScreen: View {
    @ObservedObject private var viewModel: AppViewModel

    public init(viewModel: AppViewModel) {
        self.viewModel = viewModel
    }

    public var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 12) {
                PageHeader(subtitle: "SeeCal", title: "Settings")
                    .padding(.top, 8)

                SectionLabel("Capture")
                Card {
                    VStack(spacing: 0) {
                        toggleRow(
                            title: "Use LiDAR depth",
                            subtitle: "Improves portion-size accuracy",
                            isOn: Binding(
                                get: { viewModel.capturePreferences.useLiDARDepth },
                                set: { newValue in
                                    var preferences = viewModel.capturePreferences
                                    preferences.useLiDARDepth = newValue
                                    Task { await viewModel.updateCapturePreferences(preferences) }
                                }
                            )
                        )
                        Divider().overlay(Theme.appLine)
                        toggleRow(
                            title: "Capture coaching",
                            subtitle: "Distance & level guides",
                            isOn: Binding(
                                get: { viewModel.capturePreferences.captureCoachingEnabled },
                                set: { newValue in
                                    var preferences = viewModel.capturePreferences
                                    preferences.captureCoachingEnabled = newValue
                                    Task { await viewModel.updateCapturePreferences(preferences) }
                                }
                            )
                        )
                    }
                }

                SectionLabel("Sync")
                Card {
                    VStack(spacing: 0) {
                        soonRow(
                            title: "iCloud sync",
                            subtitle: "Meals, weight and goal on all your devices"
                        )
                        Divider().overlay(Theme.appLine)
                        soonRow(
                            title: "Apple Health",
                            subtitle: "Write meals, read weight & workouts"
                        )
                    }
                }

                SectionLabel("On-device model")
                modelCard
            }
            .padding(.horizontal, 18)
            .padding(.bottom, Theme.screenBottomInset)
        }
        .background(Theme.appBg)
    }

    // MARK: - Capture toggle row (`.setrow` + `.switch`)

    private func toggleRow(title: String, subtitle: String, isOn: Binding<Bool>) -> some View {
        HStack(spacing: 12) {
            rowLabel(title: title, subtitle: subtitle)
            Spacer(minLength: 8)
            SettingsSwitch(isOn: isOn)
        }
        .padding(.vertical, 13)
    }

    // MARK: - Sync roadmap row (`.setrow` + `.soonbadge`)

    private func soonRow(title: String, subtitle: String) -> some View {
        HStack(spacing: 12) {
            rowLabel(title: title, subtitle: subtitle)
            Spacer(minLength: 8)
            Text("SOON")
                .font(.system(size: 10, weight: .heavy))
                .tracking(0.6)
                .foregroundStyle(Theme.appInk2)
                .padding(.horizontal, 8)
                .padding(.vertical, 3)
                .background(Theme.appLine)
                .clipShape(RoundedRectangle(cornerRadius: 6, style: .continuous))
        }
        .padding(.vertical, 13)
    }

    private func rowLabel(title: String, subtitle: String) -> some View {
        VStack(alignment: .leading, spacing: 1) {
            Text(title)
                .font(.system(size: 15, weight: .semibold))
                .foregroundStyle(Theme.appInk)
            Text(subtitle)
                .font(.system(size: 12, weight: .medium))
                .foregroundStyle(Theme.appInk2)
        }
    }

    // MARK: - On-device model card (`.modelcard`)

    private var modelCard: some View {
        Card {
            VStack(alignment: .leading, spacing: 6) {
                HStack(alignment: .top, spacing: 8) {
                    Text(AppViewModel.modelCardTitle(info: viewModel.modelInfo))
                        .font(.system(size: 15, weight: .bold))
                        .foregroundStyle(Theme.appInk)
                        .fixedSize(horizontal: false, vertical: true)
                    Spacer(minLength: 4)
                    Text("ON DEVICE")
                        .font(.system(size: 10, weight: .heavy))
                        .tracking(0.6)
                        .foregroundStyle(Theme.basil)
                        .padding(.horizontal, 7)
                        .padding(.vertical, 3)
                        .background(Theme.basilSoft)
                        .clipShape(RoundedRectangle(cornerRadius: 6, style: .continuous))
                }
                Text(AppViewModel.modelCardSubtitle(info: viewModel.modelInfo))
                    .font(.system(size: 12.5))
                    .foregroundStyle(Theme.appInk2)
                    .fixedSize(horizontal: false, vertical: true)
            }
        }
    }
}

// MARK: - Switch (`.switch`)

/// `.switch{width:48px;height:29px;border-radius:15px;background:var(--basil)}`
/// + a 24pt white knob that slides to the trailing edge when on. The
/// prototype's demo markup hardcodes both switches to the "on" (basil)
/// appearance since it has no interactive state; this real toggle also
/// renders an "off" (neutral `--app-line`) appearance, a platform-conventional
/// addition the static prototype had no need to specify.
private struct SettingsSwitch: View {
    @Binding var isOn: Bool

    var body: some View {
        Button {
            isOn.toggle()
        } label: {
            ZStack(alignment: isOn ? .trailing : .leading) {
                Capsule()
                    .fill(isOn ? Theme.basil : Theme.appLine)
                    .frame(width: 48, height: 29)
                Circle()
                    .fill(Color.white)
                    .frame(width: 24, height: 24)
                    .shadow(color: .black.opacity(0.3), radius: 3, y: 1)
                    .padding(2.5)
            }
        }
        .buttonStyle(.plain)
        .accessibilityLabel(isOn ? "On" : "Off")
        .accessibilityAddTraits(.isButton)
    }
}
