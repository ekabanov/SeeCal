import SwiftUI
import SeeCalDomain

/// The app's five-slot scaffold (spec §1): Today / History / [Scan] / Profile /
/// Settings, plus the full-screen onboarding wizard and the current-scan-error
/// alert, both of which apply app-wide rather than to any single tab.
///
/// Onboarding (spec §3) shows once, when no persisted profile exists — as a
/// full-screen overlay covering the tabs, exactly like the prototype's
/// `.onboard{position:absolute; inset:0}` layer. Finishing (or skipping)
/// persists a profile and lands on Today. Profile editing no longer opens a
/// modal: the Profile tab itself is the inline editor (P5).
public struct RootView: View {
    @StateObject private var viewModel: AppViewModel
    @State private var selectedTab: AppTab = .today
    @State private var isShowingCamera = false
    @State private var isShowingOnboarding = false
    @Environment(\.accessibilityReduceMotion) private var reduceMotion

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
                    ProfileScreen(viewModel: viewModel)
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
        .overlay {
            if isShowingOnboarding {
                OnboardingView { profile in
                    Task {
                        await viewModel.completeOnboarding(with: profile)
                        selectedTab = .today
                        withAnimation(reduceMotion ? nil : .easeOut(duration: 0.25)) {
                            isShowingOnboarding = false
                        }
                    }
                }
                .transition(reduceMotion ? .identity : .opacity)
                .zIndex(10)
            }
        }
        .task {
            await viewModel.loadEntries()
            if viewModel.requiresOnboarding {
                isShowingOnboarding = true
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
        .cameraFullScreenCover(isPresented: $isShowingCamera) {
            ScanStubScreen(onClose: { isShowingCamera = false })
        }
    }
}
