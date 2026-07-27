import SwiftUI

/// The four ordinary tab-bar destinations. "Scan" is deliberately not a case here —
/// in the prototype it's a momentary action (opens the full-screen camera), not a
/// persisted selection state; see `SeeCalTabBar.onScanTapped`.
public enum AppTab: String, CaseIterable, Identifiable {
    case today
    case history
    case profile
    case settings

    public var id: String { rawValue }

    var label: String {
        switch self {
        case .today: return "Today"
        case .history: return "History"
        case .profile: return "Profile"
        case .settings: return "Settings"
        }
    }

    /// SF Symbols standing in for the prototype's hand-drawn glyphs (platform
    /// deviation noted in the P3 task report): calendar-with-dot -> `calendar`,
    /// bars -> `chart.bar.fill`, person -> `person.fill`, gear -> `gearshape.fill`.
    var systemImage: String {
        switch self {
        case .today: return "calendar"
        case .history: return "chart.bar.fill"
        case .profile: return "person.fill"
        case .settings: return "gearshape.fill"
        }
    }
}

/// `.tabbar` — five-slot bottom bar (`grid-template-columns:1fr 1fr 88px 1fr 1fr`):
/// Today, History, a center Scan FAB, Profile, Settings. Translucent card-colored
/// background with a top hairline; the FAB overhangs the bar with a background-
/// colored halo ring and a brand-colored shadow. Callers hide this view entirely
/// (rather than disabling it) while the camera is active, matching
/// `.tabbar.hidden{display:none}`.
public struct SeeCalTabBar: View {
    @Binding private var selection: AppTab
    private let onScanTapped: () -> Void
    private let onTabTapped: ((AppTab) -> Void)?
    @Environment(\.accessibilityReduceMotion) private var reduceMotion

    /// - Parameter onTabTapped: fires on EVERY ordinary-tab tap, including
    ///   re-taps of the already-selected tab (which don't change `selection`).
    ///   The scan flow uses this to leave the camera/analyzing surface on any
    ///   tab tap, exactly like the prototype's tab handler (P6).
    public init(
        selection: Binding<AppTab>,
        onScanTapped: @escaping () -> Void,
        onTabTapped: ((AppTab) -> Void)? = nil
    ) {
        self._selection = selection
        self.onScanTapped = onScanTapped
        self.onTabTapped = onTabTapped
    }

    public var body: some View {
        HStack(spacing: 0) {
            tabButton(.today)
            tabButton(.history)
            scanButton
            tabButton(.profile)
            tabButton(.settings)
        }
        .padding(.top, 10)
        .frame(height: Theme.tabBarHeight, alignment: .top)
        .frame(maxWidth: .infinity)
        .background {
            // `.tabbar{background:color-mix(in srgb, var(--app-card) 88%, transparent);
            // backdrop-filter:blur(14px)}` — a translucent card-colored blur. SwiftUI
            // material is the platform-idiomatic stand-in (explicitly allowed by the
            // P3 task brief); tinted with `appCard` so it still reads as the card
            // surface color rather than a neutral system material.
            //
            // The top hairline lives HERE, in the background layer (behind the bar's
            // content), not as an .overlay — otherwise it renders in front of the
            // Scan FAB and its `appBg` halo can't mask it, so the line visibly runs
            // across the raised button. Behind the content, the FAB halo punches
            // through the line exactly as the prototype's `box-shadow 0 0 0 6px
            // var(--app-bg)` does.
            ZStack(alignment: .top) {
                Rectangle()
                    .fill(.ultraThinMaterial)
                    .overlay(Theme.appCard.opacity(0.35))
                Rectangle().fill(Theme.appLine).frame(height: 1)
            }
            // Extend the bar's background DOWN through the bottom safe area
            // (home-indicator strip) so scrolled content can't show beneath it,
            // like a standard iOS bottom bar. Only the background bleeds down —
            // the icon row stays above the home indicator.
            .ignoresSafeArea(edges: .bottom)
        }
        .accessibilityElement(children: .contain)
    }

    private func tabButton(_ tab: AppTab) -> some View {
        Button {
            select(tab)
        } label: {
            VStack(spacing: 3) {
                Image(systemName: tab.systemImage)
                    .font(.system(size: 20, weight: .regular))
                Text(tab.label)
                    .font(.system(size: 10.5, weight: .semibold))
            }
            .foregroundStyle(selection == tab ? Theme.basil : Theme.appInk2)
            .frame(maxWidth: .infinity)
            .padding(.top, 2)
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .accessibilityAddTraits(selection == tab ? [.isSelected] : [])
    }

    /// `.tab.scan .cam{width:64px;height:64px;border-radius:50%;background:basil;
    /// margin-top:-36px;box-shadow:0 8px 20px rgba(31,122,82,.45),0 0 0 6px var(--app-bg)}`
    /// — the `0 0 0 6px var(--app-bg)` term is the halo ring: a solid app-background-
    /// colored spread that visually punches the raised circle through the bar's
    /// translucency. Reproduced here as a same-colored circle sized 12pt larger
    /// (6pt of spread on every side) sitting directly behind the basil circle.
    private var scanButton: some View {
        Button(action: onScanTapped) {
            VStack(spacing: 6) {
                ZStack {
                    Circle()
                        .fill(Theme.appBg)
                        .frame(
                            width: Theme.scanFABDiameter + 12,
                            height: Theme.scanFABDiameter + 12
                        )
                    Circle()
                        .fill(Theme.basil)
                        .frame(width: Theme.scanFABDiameter, height: Theme.scanFABDiameter)
                        .shadow(color: Theme.basil.opacity(0.45), radius: 10, x: 0, y: 8)
                    Image(systemName: "camera.fill")
                        .font(.system(size: 22, weight: .semibold))
                        .foregroundStyle(.white)
                }
                .offset(y: -Theme.scanFABOverhang)

                Text("Scan")
                    .font(.system(size: 10.5, weight: .bold))
                    .foregroundStyle(Theme.basil)
                    .offset(y: -Theme.scanFABOverhang)
            }
            .frame(width: 88)
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .accessibilityLabel("Scan a meal")
    }

    private func select(_ tab: AppTab) {
        onTabTapped?(tab)
        guard tab != selection else { return }
        if reduceMotion {
            selection = tab
        } else {
            withAnimation(.easeOut(duration: 0.15)) {
                selection = tab
            }
        }
    }
}
