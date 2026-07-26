import SwiftUI
import SeeCalDomain

/// Spec §7 placeholder. The full header card / body card / lifestyle card / goal
/// card layout (with live-recompute goal math) is P5's job; for P3 this parks the
/// existing profile view + edit trigger, and the weight log (add/list/trend), which
/// is the closest existing functionality to "Profile".
public struct ProfileScreen: View {
    @ObservedObject private var viewModel: AppViewModel
    private let onEditProfile: () -> Void

    public init(viewModel: AppViewModel, onEditProfile: @escaping () -> Void) {
        self.viewModel = viewModel
        self.onEditProfile = onEditProfile
    }

    public var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 12) {
                PageHeader(subtitle: "You", title: "Profile")
                    .padding(.top, 8)

                if let profile = viewModel.userProfile {
                    SectionLabel("Profile")
                    Card {
                        VStack(alignment: .leading, spacing: 6) {
                            Text("Sex: \(profile.sex.rawValue.capitalized)")
                            Text("Age: \(GoalCalculator.age(dateOfBirth: profile.dateOfBirth))")
                            Text("Height: \(profile.heightCm) cm")
                            Text(String(format: "Weight: %.1f kg", profile.weightKg))
                            Text("Activity: \(activityTitle(profile.activity))")
                            Text(String(format: "Weekly target: %.1f kg/week", profile.weeklyRateKg))
                            Button("Edit Profile", action: onEditProfile)
                                .font(.system(size: 13.5, weight: .semibold))
                                .foregroundStyle(Theme.basil)
                                .padding(.top, 4)
                        }
                        .font(.system(size: 13.5))
                        .foregroundStyle(Theme.appInk)
                    }
                } else {
                    Card {
                        Text("No profile yet.")
                            .foregroundStyle(Theme.appInk2)
                    }
                }

                SectionLabel("Weight")
                Card {
                    VStack(alignment: .leading, spacing: 10) {
                        Button("Add Sample Weight (78.4 kg)") {
                            Task { await viewModel.addWeightEntry(kg: 78.4) }
                        }
                        .font(.system(size: 13.5, weight: .semibold))
                        .foregroundStyle(Theme.basil)

                        if !viewModel.weightEntries.isEmpty {
                            Divider().overlay(Theme.appLine)
                            ForEach(viewModel.weightEntries) { entry in
                                HStack {
                                    Text(entry.date, style: .date)
                                    Spacer()
                                    Text(String(format: "%.1f kg", entry.weightKg))
                                        .foregroundStyle(Theme.appInk2)
                                }
                                .font(.system(size: 13.5))
                            }
                        }
                    }
                }

                SectionLabel("8-week weight trend")
                Card {
                    VStack(spacing: 8) {
                        ForEach(viewModel.weeklyWeightTrend, id: \.weekStart) { point in
                            HStack {
                                Text(point.weekStart, style: .date)
                                Spacer()
                                Text(point.averageWeightKg > 0 ? String(format: "%.1f kg", point.averageWeightKg) : "—")
                                    .foregroundStyle(Theme.appInk2)
                            }
                            .font(.system(size: 13.5))
                        }
                    }
                }
            }
            .padding(.horizontal, 18)
            .padding(.bottom, Theme.screenBottomInset)
        }
        .background(Theme.appBg)
    }

    private func activityTitle(_ level: ActivityLevel) -> String {
        switch level {
        case .sedentary: return "Sedentary"
        case .light: return "Light"
        case .moderate: return "Moderate"
        case .active: return "Active"
        }
    }
}
