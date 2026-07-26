import SwiftUI
import SeeCalDomain

/// The app's five-slot scaffold (spec §1): Today / History / [Scan] / Profile /
/// Settings, plus the full-screen onboarding flow and the current-scan-error alert,
/// both of which apply app-wide rather than to any single tab.
///
/// Screen content is split into `TodayScreen`/`HistoryScreen`/`ProfileScreen`/
/// `SettingsScreen` (each parking the pre-P3 functionality closest to its spec
/// section); this container only owns tab selection, the custom `SeeCalTabBar`, and
/// the two app-wide presentations (onboarding sheet, camera full-screen cover).
public struct RootView: View {
    @StateObject private var viewModel: AppViewModel
    @State private var selectedTab: AppTab = .today
    @State private var isShowingCamera = false
    @State private var isShowingOnboarding = false
    @State private var onboardingDraft = OnboardingDraft()

    public init(viewModel: AppViewModel) {
        _viewModel = StateObject(wrappedValue: viewModel)
    }

    public var body: some View {
        ZStack(alignment: .bottom) {
            Theme.appBg.ignoresSafeArea()

            Group {
                switch selectedTab {
                case .today:
                    TodayScreen(viewModel: viewModel)
                case .history:
                    HistoryScreen(viewModel: viewModel)
                case .profile:
                    ProfileScreen(viewModel: viewModel, onEditProfile: presentOnboarding)
                case .settings:
                    SettingsScreen()
                }
            }

            // `.tabbar.hidden{display:none}` — the bar is hidden entirely (not just
            // dimmed/disabled) whenever the camera is on screen.
            if !isShowingCamera {
                SeeCalTabBar(selection: $selectedTab) {
                    isShowingCamera = true
                }
            }
        }
        .task {
            await viewModel.loadEntries()
            if viewModel.requiresOnboarding {
                presentOnboarding()
            }
        }
        .alert("Error", isPresented: Binding(
            get: { viewModel.lastError != nil },
            set: { newValue in
                if !newValue { viewModel.lastError = nil }
            }
        )) {
            Button("OK", role: .cancel) { viewModel.lastError = nil }
        } message: {
            Text(viewModel.lastError ?? "Unknown error")
        }
        .sheet(isPresented: $isShowingOnboarding) {
            OnboardingSheet(
                draft: onboardingDraft,
                isFirstRun: viewModel.requiresOnboarding,
                onCancel: {
                    if !viewModel.requiresOnboarding {
                        isShowingOnboarding = false
                    }
                },
                onSave: { savedDraft in
                    Task {
                        do {
                            let profile = try savedDraft.toUserProfile()
                            await viewModel.completeOnboarding(with: profile)
                            onboardingDraft = savedDraft
                            isShowingOnboarding = false
                        } catch {
                            viewModel.lastError = error.localizedDescription
                        }
                    }
                }
            )
            .interactiveDismissDisabled(viewModel.requiresOnboarding)
        }
        .cameraFullScreenCover(isPresented: $isShowingCamera) {
            ScanStubScreen(onClose: { isShowingCamera = false })
        }
    }

    /// Shared by the first-run gate (`requiresOnboarding`, no profile yet — starts
    /// from spec §2 defaults) and Profile's "Edit Profile" button (starts from the
    /// existing profile) — same sheet serves both, as it did pre-P3.
    private func presentOnboarding() {
        onboardingDraft = viewModel.userProfile.map(OnboardingDraft.init(profile:)) ?? OnboardingDraft()
        isShowingOnboarding = true
    }
}

/// Full-screen 6-step wizard is P5's job (spec §3); this remains the placeholder
/// single-form editor from pre-P3, still wired to `OnboardingDraft`/`UserProfile` so
/// onboarding and profile-editing both keep working end to end.
private struct OnboardingSheet: View {
    @State var draft: OnboardingDraft
    let isFirstRun: Bool
    let onCancel: () -> Void
    let onSave: (OnboardingDraft) -> Void

    var body: some View {
        NavigationStack {
            Form {
                Section("About You") {
                    Picker("Sex", selection: $draft.sex) {
                        Text("Male").tag(BiologicalSex.male)
                        Text("Female").tag(BiologicalSex.female)
                    }
                    DatePicker("Birthday", selection: $draft.dateOfBirth, displayedComponents: .date)
                    TextField("Height (cm)", text: $draft.heightCmText)
                    TextField("Weight (kg)", text: $draft.weightKgText)
                }

                Section("Lifestyle") {
                    Picker("Activity", selection: $draft.activity) {
                        Text("Sedentary").tag(ActivityLevel.sedentary)
                        Text("Light").tag(ActivityLevel.light)
                        Text("Moderate").tag(ActivityLevel.moderate)
                        Text("Active").tag(ActivityLevel.active)
                    }
                    Stepper(
                        String(format: "Weekly target: %.1f kg/week", draft.weeklyRateKg),
                        value: $draft.weeklyRateKg,
                        in: UserProfile.weeklyRateRange,
                        step: 0.1
                    )
                }
            }
            .navigationTitle("Setup")
            .toolbar {
                ToolbarItem(placement: .cancellationAction) {
                    if !isFirstRun {
                        Button("Cancel", action: onCancel)
                    }
                }
                ToolbarItem(placement: .confirmationAction) {
                    Button("Save") { onSave(draft) }
                }
            }
        }
    }
}
