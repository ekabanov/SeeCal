import SwiftUI
import SeeCalDomain
import SeeCalPersistence

/// The app's five-slot scaffold (spec §1): Today / History / [Scan] / Profile /
/// Settings, plus the full-screen onboarding wizard, the scan → analyzing →
/// result flow (spec §5), and the current-scan-error alert.
///
/// Scan-flow architecture mirrors the prototype's navigation exactly:
/// - The scan surface (camera or analyzing screen) renders IN PLACE of the
///   selected tab while `ScanFlowController.isScanSurfaceVisible`. The tab bar
///   hides only for the camera (`.tabbar.hidden` when screen == "camera");
///   the analyzing screen keeps it, so any tab tap navigates away WITHOUT
///   cancelling inference.
/// - Completion auto-presents the result sheet when the analyzing screen is
///   frontmost, else parks it behind the `.ready-banner` above the tab bar.
/// - One `MealResultSheet` serves fresh scans and meal-row edits alike, fed by
///   `ScanFlowController.activeSheetDraft`.
///
/// Onboarding (spec §3) shows once, when no persisted profile exists — as a
/// full-screen overlay covering the tabs, exactly like the prototype's
/// `.onboard{position:absolute; inset:0}` layer.
public struct RootView: View {
    @StateObject private var viewModel: AppViewModel
    @StateObject private var scanController: ScanFlowController
    @State private var selectedTab: AppTab = .today
    @State private var isShowingOnboarding = false
    @Environment(\.accessibilityReduceMotion) private var reduceMotion

    /// - Parameter captureService: override for previews/tests; `nil` picks the
    ///   platform default (real camera on an iOS device, mock elsewhere).
    public init(viewModel: AppViewModel, captureService: CaptureService? = nil) {
        _viewModel = StateObject(wrappedValue: viewModel)
        _scanController = StateObject(wrappedValue: ScanFlowController(
            viewModel: viewModel,
            inference: viewModel.scanInferenceRunner,
            captureService: captureService ?? makeDefaultCaptureService()
        ))
    }

    public var body: some View {
        ZStack(alignment: .bottom) {
            Theme.appBg.ignoresSafeArea()

            content

            if !scanController.isTabBarHidden {
                SeeCalTabBar(
                    selection: $selectedTab,
                    onScanTapped: { scanController.scanTapped() },
                    onTabTapped: { _ in scanController.leaveScanSurface() }
                )
            }

            if scanController.isBannerVisible {
                ReadyBannerView {
                    scanController.bannerTapped()
                }
                .padding(.bottom, Theme.tabBarHeight + 70) // `.ready-banner{bottom:158px}`
                .transition(reduceMotion ? .identity : .opacity.combined(with: .move(edge: .bottom)))
            }

            if let toast = scanController.toastMessage {
                ToastView(text: toast)
                    .padding(.bottom, Theme.tabBarHeight + 72) // `.toast{bottom:160px}`
                    .transition(reduceMotion ? .identity : .opacity.combined(with: .move(edge: .bottom)))
            }
        }
        .animation(reduceMotion ? nil : .easeOut(duration: 0.3), value: scanController.isBannerVisible)
        .animation(reduceMotion ? nil : .easeOut(duration: 0.3), value: scanController.toastMessage)
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
            scanController.onRequestTabSwitch = { tab in
                selectedTab = tab
            }
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
        .sheet(isPresented: resultSheetPresented) {
            if let draft = scanController.activeSheetDraft {
                MealResultSheet(
                    draft: draft,
                    onPrimary: { finalDraft in
                        Task {
                            if finalDraft.isEditingExisting {
                                await scanController.saveChanges(finalDraft)
                            } else {
                                await scanController.logMeal(finalDraft)
                            }
                        }
                    },
                    onSecondary: {
                        if draft.isEditingExisting {
                            scanController.cancelEdit()
                        } else {
                            scanController.discardResult()
                        }
                    }
                )
                .resultSheetStyling()
            }
        }
    }

    /// The scan surface replaces the selected tab's screen while visible
    /// (prototype: camera/analyzing are screens at the same level as the tabs).
    @ViewBuilder
    private var content: some View {
        if scanController.isScanSurfaceVisible {
            switch scanController.phase {
            case .capturing:
                CameraScreen(controller: scanController)
            default:
                // .analyzing, .error, and the completed states while the sheet
                // (or banner) sits above the analyzing screen.
                AnalyzingScreen(controller: scanController)
            }
        } else {
            switch selectedTab {
            case .today:
                TodayScreen(
                    viewModel: viewModel,
                    scrollToTopToken: scanController.todayScrollToTopToken,
                    onEditMeal: { entry in scanController.beginEdit(entry: entry) }
                )
            case .history:
                HistoryScreen(viewModel: viewModel)
            case .profile:
                ProfileScreen(viewModel: viewModel)
            case .settings:
                SettingsScreen()
            }
        }
    }

    /// Presentation binding for the shared result/edit sheet. A `false` write
    /// while state still says "presenting" is an interactive dismissal
    /// (swipe-down): edit mode cancels; a fresh result parks behind the ready
    /// banner instead of being lost. Programmatic closes (log/discard/save)
    /// change state first, so those writes no-op here.
    private var resultSheetPresented: Binding<Bool> {
        Binding(
            get: { scanController.activeSheetDraft != nil },
            set: { isPresented in
                guard !isPresented else { return }
                if scanController.editDraft != nil {
                    scanController.cancelEdit()
                } else {
                    scanController.resultSheetDismissed()
                }
            }
        )
    }
}

private extension View {
    /// Prototype `.sheet`: 26pt top corner radius, max-height 86%, custom grab
    /// handle (the sheet draws its own, so the system indicator is hidden).
    /// These presentation modifiers are iOS-only; the macOS test host uses the
    /// default sheet chrome.
    @ViewBuilder
    func resultSheetStyling() -> some View {
        #if os(iOS)
        self
            .presentationDetents([.fraction(0.86)])
            .presentationDragIndicator(.hidden)
            .presentationCornerRadius(26)
        #else
        self
        #endif
    }
}
