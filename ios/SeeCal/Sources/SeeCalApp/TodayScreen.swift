import SwiftUI
import SeeCalDomain
import SeeCalPersistence

/// Spec §4 / `#scr-today`: ring card (consumed/goal kcal + three macro bars),
/// MEALS section, meal-row list, privacy chip. Matches the prototype's markup +
/// CSS (`.pagehead`, `.card`, `.ringrow`, `.ring`, `.macros`/`.macro`/`.bar`,
/// `.sectionlabel`, `.meal`, `.mealcard`, `.privchip`) — see
/// `docs/design/prototype/seecal-prototype.html` lines 582-616 and `renderToday()`
/// (~line 1139) for the exact text formats this file reproduces.
///
/// Meal rows open the shared result/edit sheet (spec §5) via `onEditMeal` —
/// presentation is owned by `RootView`/`ScanFlowController`, so editing uses
/// the exact same `MealResultSheet` component as a fresh scan.
public struct TodayScreen: View {
    @ObservedObject private var viewModel: AppViewModel
    private let onEditMeal: (MealLogEntry) -> Void
    /// Bumped by the scan flow after "Log meal" so the list lands scrolled to
    /// top ("switch Today scrolled to top", spec §5).
    private let scrollToTopToken: UUID

    public init(
        viewModel: AppViewModel,
        scrollToTopToken: UUID = UUID(),
        onEditMeal: @escaping (MealLogEntry) -> Void = { _ in }
    ) {
        self.viewModel = viewModel
        self.scrollToTopToken = scrollToTopToken
        self.onEditMeal = onEditMeal
    }

    public var body: some View {
        ScrollViewReader { proxy in
            ScrollView {
                VStack(alignment: .leading, spacing: 12) {
                    PageHeader(subtitle: AppViewModel.dateSubtitle(for: Date()), title: "Today")
                        .padding(.top, 8)
                        .id(Self.topAnchorID)

                    Card {
                        ringRow
                    }

                    SectionLabel("Meals")

                    mealsSection

                    PrivacyChip("Analyzed entirely on this iPhone.")
                }
                .padding(.horizontal, 18)
                .padding(.bottom, Theme.screenBottomInset)
            }
            .background(Theme.appBg)
            .onChange(of: scrollToTopToken) {
                proxy.scrollTo(Self.topAnchorID, anchor: .top)
            }
        }
    }

    private static let topAnchorID = "today-top"

    // MARK: - Ring row (`.ringrow`)

    @ViewBuilder
    private var ringRow: some View {
        let consumed = viewModel.consumedToday
        let target = viewModel.dailyTarget
        HStack(alignment: .center, spacing: 18) {
            CalorieRing(consumedCalories: consumed.calories, goalCalories: target.calories)
            VStack(spacing: 12) {
                MacroBarRow(
                    title: "Protein",
                    color: Theme.protein,
                    consumed: consumed.proteinGrams,
                    target: target.proteinGrams
                )
                MacroBarRow(
                    title: "Fat",
                    color: Theme.fat,
                    consumed: consumed.fatGrams,
                    target: target.fatGrams
                )
                MacroBarRow(
                    title: "Carbs",
                    color: Theme.carbs,
                    consumed: consumed.carbsGrams,
                    target: target.carbsGrams
                )
            }
        }
    }

    // MARK: - Meals (`.sectionlabel` + `.card.mealcard` of `.meal` rows)

    @ViewBuilder
    private var mealsSection: some View {
        let todaysEntries = viewModel.mealEntries.filter { Calendar.current.isDateInToday($0.createdAt) }

        if todaysEntries.isEmpty {
            Card {
                Text("No meals logged yet")
                    .font(.system(size: 13.5))
                    .foregroundStyle(Theme.appInk2)
            }
        } else {
            Card(padding: 2) {
                VStack(spacing: 0) {
                    ForEach(Array(todaysEntries.enumerated()), id: \.element.id) { index, entry in
                        if index > 0 {
                            Divider().overlay(Theme.appLine)
                        }
                        MealRowView(
                            entry: entry,
                            onTap: { onEditMeal(entry) },
                            onDelete: { Task { await viewModel.deleteMeal(id: entry.id) } }
                        )
                    }
                }
            }
        }
    }
}

// MARK: - Calorie ring (`.ring`)

/// `.ring{width:132px;height:132px}` + `#ringArc{stroke-width:12;stroke-linecap:round}`,
/// `.ring svg{transform:rotate(-90deg)}` (starts at 12 o'clock). Center shows the
/// consumed kcal (large, tabular) + "of N kcal" caption
/// (`renderToday()`: `"of "+fmt(state.goal)+" kcal"`).
private struct CalorieRing: View {
    let consumedCalories: Double
    let goalCalories: Double

    @Environment(\.accessibilityReduceMotion) private var reduceMotion

    private var fraction: Double {
        AppViewModel.progressFraction(consumed: consumedCalories, target: goalCalories)
    }

    var body: some View {
        ZStack {
            Circle()
                .stroke(Theme.appLine, lineWidth: 12)
            Circle()
                .trim(from: 0, to: fraction)
                .stroke(Theme.basil, style: StrokeStyle(lineWidth: 12, lineCap: .round))
                .rotationEffect(.degrees(-90))
                .animation(reduceMotion ? nil : .easeOut(duration: 0.7), value: fraction)

            VStack(spacing: 2) {
                Text(AppViewModel.formattedKcal(consumedCalories))
                    .font(.system(size: 28, weight: .bold))
                    .tracking(-0.56)
                    .monospacedDigit()
                    .foregroundStyle(Theme.appInk)
                Text("of \(AppViewModel.formattedKcal(goalCalories)) kcal")
                    .font(.system(size: 11, weight: .semibold))
                    .foregroundStyle(Theme.appInk2)
            }
        }
        .padding(6) // half the 12pt stroke width, so the arc doesn't clip past the 132pt tile.
        .frame(width: 132, height: 132)
        .accessibilityElement(children: .combine)
    }
}

// MARK: - Macro bar (`.macro`/`.bar`)

/// One `.macro` row: label line ("Protein" bold in the macro color, "consumed /
/// target g" in ink-2 at the trailing edge) over a 6pt rounded `.bar`.
private struct MacroBarRow: View {
    let title: String
    let color: Color
    let consumed: Double
    let target: Double

    @Environment(\.accessibilityReduceMotion) private var reduceMotion

    private var fraction: Double {
        AppViewModel.progressFraction(consumed: consumed, target: target)
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack {
                Text(title)
                    .font(.system(size: 12.5, weight: .bold))
                    .foregroundStyle(color)
                Spacer()
                Text("\(AppViewModel.roundedGrams(consumed)) / \(AppViewModel.roundedGrams(target)) g")
                    .font(.system(size: 12.5, weight: .medium))
                    .foregroundStyle(Theme.appInk2)
                    .monospacedDigit()
            }
            GeometryReader { geometry in
                ZStack(alignment: .leading) {
                    Capsule().fill(Theme.appLine)
                    Capsule()
                        .fill(color)
                        .frame(width: geometry.size.width * fraction)
                        .animation(reduceMotion ? nil : .easeOut(duration: 0.5), value: fraction)
                }
            }
            .frame(height: 6)
        }
        .accessibilityElement(children: .combine)
    }
}
