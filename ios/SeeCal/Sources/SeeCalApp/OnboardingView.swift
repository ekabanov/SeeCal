import SwiftUI
import SeeCalDomain

/// Spec §3 / prototype `.onboard`: the full-screen, six-step first-run wizard.
/// Shown when no persisted profile exists (RootView gates it); finishing (or
/// "Skip for now" on step 1) persists a profile via the injected callback and
/// lands on Today.
///
/// Copy, step order, dots, button titles, option-card descriptions, and the
/// weekly-rate slider all match the prototype markup verbatim. Platform
/// deviation (spec §9 allows it): the birthday field uses the native SwiftUI
/// `DatePicker` instead of the HTML `<input type="date">`.
public struct OnboardingView: View {
    @State private var draft = OnboardingDraft()
    @Environment(\.accessibilityReduceMotion) private var reduceMotion

    private let now: () -> Date
    private let onComplete: (UserProfile) -> Void

    public init(
        now: @escaping () -> Date = Date.init,
        onComplete: @escaping (UserProfile) -> Void
    ) {
        self.now = now
        self.onComplete = onComplete
    }

    public var body: some View {
        VStack(spacing: 0) {
            dots
                .padding(.bottom, 28)

            stepContent
                .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)

            actions
        }
        .padding(.horizontal, 26)
        .padding(.top, 40)
        .padding(.bottom, 30)
        .background(Theme.appBg.ignoresSafeArea())
        .accessibilityAddTraits(.isModal)
    }

    // MARK: - Progress dots (`.ob-dots`)

    private var dots: some View {
        HStack(spacing: 6) {
            ForEach(OnboardingDraft.Step.allCases, id: \.rawValue) { step in
                Circle()
                    .fill(step == draft.step ? Theme.basil : Theme.appLine)
                    .frame(width: 7, height: 7)
            }
        }
        .frame(maxWidth: .infinity)
        .accessibilityHidden(true)
    }

    // MARK: - Steps (`.ob-step`)

    @ViewBuilder
    private var stepContent: some View {
        Group {
            switch draft.step {
            case .welcome: welcomeStep
            case .aboutYou: aboutYouStep
            case .body: bodyStep
            case .activity: activityStep
            case .weeklyTarget: weeklyTargetStep
            case .goal: goalStep
            }
        }
        .id(draft.step)
        .transition(reduceMotion ? .identity : .opacity)
    }

    /// Step 1 — Welcome. The privacy chip is pinned to the bottom of the step
    /// area (`.ob-priv{margin-top:auto}`).
    private var welcomeStep: some View {
        VStack(alignment: .leading, spacing: 0) {
            // `.ob-art` — camera glyph, centered.
            Image(systemName: "camera")
                .font(.system(size: 56, weight: .light))
                .foregroundStyle(Theme.basil)
                .frame(maxWidth: .infinity)
                .padding(.top, 18)
                .padding(.bottom, 22)

            stepTitle("Welcome to SeeCal")
            stepParagraph("Point your camera at a meal — SeeCal estimates calories, macros and portions, entirely on this iPhone.")
            stepParagraph("A few details set your daily calorie goal. You can change everything later in Profile.")

            Spacer(minLength: 12)

            // `.ob-priv`
            Text("Meals are analyzed on your iPhone — nothing is uploaded.")
                .font(.system(size: 12.5, weight: .semibold))
                .foregroundStyle(Theme.basil)
                .fixedSize(horizontal: false, vertical: true)
                .padding(.vertical, 11)
                .padding(.horizontal, 14)
                .frame(maxWidth: .infinity, alignment: .leading)
                .background(Theme.basilSoft)
                .clipShape(RoundedRectangle(cornerRadius: 12, style: .continuous))
        }
    }

    /// Step 2 — About you: sex chips + birthday with a live "N years old" note.
    private var aboutYouStep: some View {
        scrollingStep {
            stepTitle("About you")
            stepParagraph("Age and sex set your resting metabolism.")

            settingRow(title: "Sex") {
                HStack(spacing: 6) {
                    ChipButton("Male", isSelected: draft.sex == .male) { draft.sex = .male }
                    ChipButton("Female", isSelected: draft.sex == .female) { draft.sex = .female }
                }
            }
            Divider().overlay(Theme.appLine)
            settingRow(title: "Birthday", note: "\(ageYears) years old") {
                DatePicker(
                    "Birthday",
                    selection: $draft.dateOfBirth,
                    in: ...now(),
                    displayedComponents: .date
                )
                .labelsHidden()
                .datePickerStyle(.compact)
                .tint(Theme.basil)
            }
        }
    }

    /// Step 3 — Your body: height/weight steppers.
    private var bodyStep: some View {
        scrollingStep {
            stepTitle("Your body")
            stepParagraph("Height and weight scale the estimate to you.")

            settingRow(title: "Height") {
                GramStepper(
                    valueText: "\(draft.heightCm)",
                    unitText: "cm",
                    decrementLabel: "Shorter",
                    incrementLabel: "Taller",
                    onDecrement: { draft.stepHeight(by: -1) },
                    onIncrement: { draft.stepHeight(by: 1) }
                )
            }
            Divider().overlay(Theme.appLine)
            settingRow(title: "Weight") {
                GramStepper(
                    valueText: String(format: "%.1f", draft.weightKg),
                    unitText: "kg",
                    decrementLabel: "Less",
                    incrementLabel: "More",
                    onDecrement: { draft.stepWeight(by: -0.5) },
                    onIncrement: { draft.stepWeight(by: 0.5) }
                )
            }
        }
    }

    /// Step 4 — Activity: full-width option cards with small descriptions
    /// (`.ob-list .chipbtn`).
    private var activityStep: some View {
        scrollingStep {
            stepTitle("How active are you?")
            stepParagraph("Pick what an ordinary week looks like.")

            VStack(spacing: 8) {
                activityOption(.sedentary, title: "Sedentary", description: "Desk job, little exercise")
                activityOption(.light, title: "Light", description: "1–2 workouts a week")
                activityOption(.moderate, title: "Moderate", description: "3–5 workouts a week")
                activityOption(.active, title: "Active", description: "Daily training or a physical job")
            }
            .padding(.top, 6)
        }
    }

    private func activityOption(_ level: ActivityLevel, title: String, description: String) -> some View {
        let isSelected = draft.activity == level
        return Button {
            draft.activity = level
        } label: {
            VStack(alignment: .leading, spacing: 2) {
                Text(title)
                    .font(.system(size: 15, weight: isSelected ? .bold : .semibold))
                    .foregroundStyle(isSelected ? Theme.basil : Theme.appInk2)
                Text(description)
                    .font(.system(size: 12, weight: .medium))
                    .foregroundStyle(isSelected ? Theme.basil : Theme.appInk2)
            }
            .frame(maxWidth: .infinity, alignment: .leading)
            .padding(.vertical, 12)
            .padding(.horizontal, 14)
            .background(isSelected ? Theme.basilSoft : Theme.appBg)
            .clipShape(RoundedRectangle(cornerRadius: 14, style: .continuous))
            .overlay(
                RoundedRectangle(cornerRadius: 14, style: .continuous)
                    .stroke(isSelected ? Color.clear : Theme.appLine, lineWidth: 1)
            )
            .contentShape(RoundedRectangle(cornerRadius: 14, style: .continuous))
        }
        .buttonStyle(.plain)
        .accessibilityLabel("\(title). \(description)")
        .accessibilityAddTraits(isSelected ? .isSelected : [])
    }

    /// Step 5 — Weekly target: the shared slider component.
    private var weeklyTargetStep: some View {
        scrollingStep {
            stepTitle("Your weekly target")
            stepParagraph("How fast do you want your weight to change? This sets how far the goal sits below or above what you burn.")

            WeeklyRateSlider(value: Binding(
                get: { draft.weeklyRateKg },
                set: { draft.setWeeklyRate($0) }
            ))
        }
    }

    /// Step 6 — Goal reveal: big kcal + transparent math + closing copy.
    private var goalStep: some View {
        let profile = draft.userProfile()
        let goal = GoalCalculator.goalCalories(for: profile, now: now())
        return scrollingStep {
            stepTitle("Your daily goal")

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
            .padding(.top, 10)
            .padding(.bottom, 2)

            // `.goalmath` (no top border on this instance, per the prototype).
            Text(AppViewModel.goalMathString(for: profile, now: now()))
                .font(.system(size: 12))
                .monospacedDigit()
                .foregroundStyle(Theme.appInk2)
                .lineSpacing(4)
                .fixedSize(horizontal: false, vertical: true)
                .padding(.top, 6)

            stepParagraph("SeeCal tracks every logged meal against this number. Adjust anything later in Profile.")
                .padding(.top, 14)
        }
    }

    // MARK: - Actions (`.ob-actions` + `.ob-skip`)

    private var actions: some View {
        VStack(spacing: 0) {
            HStack(spacing: 10) {
                // Hidden (but space-preserving) on step 1, like `visibility:hidden`.
                actionButton("Back", isPrimary: false) {
                    withAnimation(stepAnimation) { draft.goBack() }
                }
                .opacity(draft.showsBack ? 1 : 0)
                .disabled(!draft.showsBack)
                .accessibilityHidden(!draft.showsBack)

                actionButton(draft.primaryButtonTitle, isPrimary: true) {
                    if draft.isLastStep {
                        onComplete(draft.userProfile())
                    } else {
                        withAnimation(stepAnimation) { draft.advance() }
                    }
                }
            }
            .padding(.top, 16)

            if draft.showsSkip {
                Button("Skip for now") {
                    onComplete(OnboardingDraft.skipDefaults())
                }
                .buttonStyle(.plain)
                .font(.system(size: 13, weight: .semibold))
                .foregroundStyle(Theme.appInk2)
                .padding(.top, 12)
                .padding(.bottom, 2)
                .frame(maxWidth: .infinity)
            }
        }
    }

    /// `.btn` / `.btn.primary` / `.btn.ghost`.
    private func actionButton(_ title: String, isPrimary: Bool, action: @escaping () -> Void) -> some View {
        Button(action: action) {
            Text(title)
                .font(.system(size: 16, weight: .bold))
                .foregroundStyle(isPrimary ? Color.white : Theme.appInk)
                .frame(maxWidth: .infinity)
                .padding(.vertical, 14)
                .background(isPrimary ? Theme.basil : Theme.appBg)
                .clipShape(RoundedRectangle(cornerRadius: 14, style: .continuous))
                .contentShape(RoundedRectangle(cornerRadius: 14, style: .continuous))
        }
        .buttonStyle(.plain)
    }

    // MARK: - Shared step scaffolding

    private var stepAnimation: Animation? {
        reduceMotion ? nil : .easeInOut(duration: 0.22)
    }

    private var ageYears: Int {
        GoalCalculator.age(dateOfBirth: draft.dateOfBirth, now: now())
    }

    /// `.ob-step{overflow-y:auto}` — steps 2–6 scroll if they overflow.
    private func scrollingStep<Content: View>(@ViewBuilder content: () -> Content) -> some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 0) {
                content()
            }
            .frame(maxWidth: .infinity, alignment: .leading)
        }
    }

    /// `.ob-step h3`.
    private func stepTitle(_ text: String) -> some View {
        Text(text)
            .font(.system(size: 24, weight: .bold))
            .tracking(-0.48)
            .foregroundStyle(Theme.appInk)
            .padding(.bottom, 6)
    }

    /// `.ob-step p`.
    private func stepParagraph(_ text: String) -> some View {
        Text(text)
            .font(.system(size: 14))
            .foregroundStyle(Theme.appInk2)
            .lineSpacing(4)
            .fixedSize(horizontal: false, vertical: true)
            .padding(.bottom, 12)
    }

    /// `.ob-step .setrow` — title (with an optional small note beneath) + trailing control.
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
        .padding(.vertical, 14)
    }
}
