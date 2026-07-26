import SwiftUI

/// Spec §8 placeholder. Sync (SOON badges), Capture toggles, and the on-device
/// model card (reading real bundled config versions) are P7's job. Nothing existed
/// under "Settings" before this task, so this is a fresh scaffold rather than
/// parked functionality — it exists to prove the tab is reachable and styled with
/// the shared design-system components.
public struct SettingsScreen: View {
    public init() {}

    public var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 12) {
                PageHeader(subtitle: "SeeCal", title: "Settings")
                    .padding(.top, 8)

                SectionLabel("On-device model")
                Card {
                    Text("Sync, Capture, and model-info settings land in P7 (spec §8).")
                        .font(.system(size: 13.5))
                        .foregroundStyle(Theme.appInk2)
                }
            }
            .padding(.horizontal, 18)
            .padding(.bottom, Theme.screenBottomInset)
        }
        .background(Theme.appBg)
    }
}
