import SwiftUI
import SeeCalDomain

/// Spec §7 / prototype `#scr-profile`: header card (avatar + "Logging since"),
/// Body card (birthday / sex / height / weight with trend note), Lifestyle card
/// (activity chips + weekly-target slider), Daily goal card (big kcal +
/// transparent math + macro split row).
///
/// This screen IS the profile editor — every control writes through to the
/// persisted profile via `AppViewModel.updateProfile`, which recomputes the goal
/// immediately (Today's ring reflects it on the next render). The pre-P5 modal
/// edit-profile sheet is retired.
public struct ProfileScreen: View {
    @ObservedObject private var viewModel: AppViewModel

    public init(viewModel: AppViewModel) {
        self.viewModel = viewModel
    }

    public var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 12) {
                PageHeader(subtitle: "Your data drives the goal", title: "Profile")
                    .padding(.top, 8)

                if let profile = viewModel.userProfile {
                    headerCard

                    SectionLabel("Body")
                    bodyCard(profile)

                    SectionLabel("Lifestyle")
                    lifestyleCard(profile)

                    SectionLabel("Daily goal")
                    goalCard(profile)
                } else {
                    Card {
                        Text("Complete onboarding to set up your profile.")
                            .font(.system(size: 13.5))
                            .foregroundStyle(Theme.appInk2)
                    }
                }
            }
            .padding(.horizontal, 18)
            .padding(.bottom, Theme.screenBottomInset)
        }
        .background(Theme.appBg)
    }

    // MARK: - Write-through editing

    /// Every edit funnels through here: mutate a copy of the persisted profile,
    /// then hand it to the view model, which persists it and recomputes the goal
    /// in the same step (spec §7).
    private func update(_ mutate: @escaping (inout UserProfile) -> Void) {
        guard var profile = viewModel.userProfile else { return }
        mutate(&profile)
        let updated = profile
        Task { await viewModel.updateProfile(updated) }
    }

    // MARK: - Header card (`.profhead`)

    /// No name is collected anywhere in the product (the spec §3 wizard has no
    /// name step; skip defaults are "reference profile minus name"), so the
    /// header identity is a generic "You" with a person glyph in the brand
    /// circle where the prototype showed the demo user's initials.
    private var headerCard: some View {
        Card {
            HStack(spacing: 14) {
                ZStack {
                    Circle().fill(Theme.basil)
                    Image(systemName: "person.fill")
                        .font(.system(size: 19, weight: .heavy))
                        .foregroundStyle(Color.white)
                }
                .frame(width: 56, height: 56)

                VStack(alignment: .leading, spacing: 2) {
                    Text("You")
                        .font(.system(size: 18, weight: .bold))
                        .foregroundStyle(Theme.appInk)
                    Text(loggingSinceText)
                        .font(.system(size: 12.5))
                        .foregroundStyle(Theme.appInk2)
                }
            }
            .frame(maxWidth: .infinity, alignment: .leading)
        }
    }

    private var loggingSinceText: String {
        let since = viewModel.loggingSinceDate ?? Date()
        return "Logging since \(AppViewModel.loggingSinceString(for: since))"
    }

    // MARK: - Body card

    private func bodyCard(_ profile: UserProfile) -> some View {
        Card {
            VStack(spacing: 0) {
                settingRow(title: "Birthday", note: "\(GoalCalculator.age(dateOfBirth: profile.dateOfBirth)) years") {
                    DatePicker(
                        "Birthday",
                        selection: Binding(
                            get: { profile.dateOfBirth },
                            set: { newValue in update { $0.dateOfBirth = newValue } }
                        ),
                        in: ...Date(),
                        displayedComponents: .date
                    )
                    .labelsHidden()
                    .datePickerStyle(.compact)
                    .tint(Theme.basil)
                }

                Divider().overlay(Theme.appLine)

                settingRow(title: "Sex") {
                    HStack(spacing: 6) {
                        ChipButton("Male", isSelected: profile.sex == .male) {
                            update { $0.sex = .male }
                        }
                        ChipButton("Female", isSelected: profile.sex == .female) {
                            update { $0.sex = .female }
                        }
                    }
                }

                Divider().overlay(Theme.appLine)

                settingRow(title: "Height") {
                    GramStepper(
                        valueText: "\(profile.heightCm)",
                        unitText: "cm",
                        decrementLabel: "Shorter",
                        incrementLabel: "Taller",
                        onDecrement: { update { $0.heightCm = UserProfile.clampHeight($0.heightCm - 1) } },
                        onIncrement: { update { $0.heightCm = UserProfile.clampHeight($0.heightCm + 1) } }
                    )
                }

                Divider().overlay(Theme.appLine)

                settingRow(
                    title: "Weight",
                    note: AppViewModel.monthTrendString(deltaKg: viewModel.monthWeightChangeKg)
                ) {
                    GramStepper(
                        valueText: String(format: "%.1f", profile.weightKg),
                        unitText: "kg",
                        decrementLabel: "Less",
                        incrementLabel: "More",
                        onDecrement: { update { $0.weightKg = UserProfile.clampWeight($0.weightKg - 0.5) } },
                        onIncrement: { update { $0.weightKg = UserProfile.clampWeight($0.weightKg + 0.5) } }
                    )
                }
            }
        }
    }

    // MARK: - Lifestyle card

    private func lifestyleCard(_ profile: UserProfile) -> some View {
        Card {
            VStack(spacing: 0) {
                settingRow(title: "Activity") {
                    ChipsRow(ActivityLevel.allCases) { level in
                        ChipButton(activityTitle(level), isSelected: profile.activity == level) {
                            update { $0.activity = level }
                        }
                    }
                }

                Divider().overlay(Theme.appLine)

                // Weekly target — block row (`.setrow{display:block}`) hosting the
                // same slider component as onboarding step 5.
                VStack(alignment: .leading, spacing: 4) {
                    Text("Weekly target")
                        .font(.system(size: 15, weight: .semibold))
                        .foregroundStyle(Theme.appInk)
                    WeeklyRateSlider(value: Binding(
                        get: { profile.weeklyRateKg },
                        set: { newValue in update { $0.weeklyRateKg = UserProfile.clampWeeklyRate(newValue) } }
                    ))
                }
                .padding(.vertical, 13)
            }
        }
    }

    private func activityTitle(_ level: ActivityLevel) -> String {
        switch level {
        case .sedentary: return "Sedentary"
        case .light: return "Light"
        case .moderate: return "Moderate"
        case .active: return "Active"
        }
    }

    // MARK: - Daily goal card (`.goalcard`)

    private func goalCard(_ profile: UserProfile) -> some View {
        let goal = GoalCalculator.goalCalories(for: profile)
        return Card {
            VStack(alignment: .leading, spacing: 0) {
                // `.kcal-hero`
                HStack(alignment: .firstTextBaseline, spacing: 8) {
                    Text(AppViewModel.formattedKcal(Double(goal)))
                        .font(.system(size: 46, weight: .heavy))
                        .tracking(-1.38)
                        .monospacedDigit()
                        .foregroundStyle(Theme.appInk)
                    Text("kcal / day")
                        .font(.system(size: 15, weight: .bold))
                        .foregroundStyle(Theme.appInk2)
                }
                .padding(.top, 2)
                .padding(.bottom, 8)

                // `.goalmath` (top border + 6px gap above the text).
                Divider().overlay(Theme.appLine)
                Text(AppViewModel.goalMathString(for: profile))
                    .font(.system(size: 12))
                    .monospacedDigit()
                    .foregroundStyle(Theme.appInk2)
                    .lineSpacing(4)
                    .fixedSize(horizontal: false, vertical: true)
                    .padding(.top, 6)
                    .padding(.bottom, 8)

                // Macro split row (`.setrow` with top border).
                Divider().overlay(Theme.appLine)
                HStack(alignment: .center, spacing: 12) {
                    VStack(alignment: .leading, spacing: 1) {
                        Text("Macro split")
                            .font(.system(size: 15, weight: .semibold))
                            .foregroundStyle(Theme.appInk)
                        Text("Protein / fat / carbs")
                            .font(.system(size: 12, weight: .medium))
                            .foregroundStyle(Theme.appInk2)
                    }
                    Spacer(minLength: 8)
                    Text("30 · 25 · 45 %")
                        .font(.system(size: 14, weight: .semibold))
                        .monospacedDigit()
                        .foregroundStyle(Theme.appInk2)
                }
                .padding(.top, 13)
            }
        }
    }

    // MARK: - Shared row scaffolding (`.setrow`)

    private func settingRow<Control: View>(
        title: String,
        note: String? = nil,
        @ViewBuilder control: () -> Control
    ) -> some View {
        HStack(alignment: .center, spacing: 12) {
            VStack(alignment: .leading, spacing: 1) {
                Text(title)
                    .font(.system(size: 15, weight: .semibold))
                    .foregroundStyle(Theme.appInk)
                if let note {
                    Text(note)
                        .font(.system(size: 12, weight: .medium))
                        .foregroundStyle(Theme.appInk2)
                }
            }
            Spacer(minLength: 8)
            control()
        }
        .padding(.vertical, 13)
    }
}
