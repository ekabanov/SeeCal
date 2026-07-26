import SwiftUI
import SeeCalDomain
import SeeCalPersistence

/// Spec §6 / `#scr-history`: segmented Week·Month·6 months control, bar chart
/// with a dashed goal line + axis-zone labels + legend, a stats row, and a
/// recent-meals card below — matches the prototype's markup + CSS (`.seg`,
/// `.chart`/`.chart.dense`, `.cbar`, `.goal-line`, `.legend`, `.statrow`/
/// `.stat`, `.mealcard`) and its `renderHistory()` JS. See
/// `docs/design/prototype/seecal-prototype.html` lines 701-741 and ~1376-1420.
public struct HistoryScreen: View {
    @ObservedObject private var viewModel: AppViewModel
    private let onEditMeal: (MealLogEntry) -> Void
    @State private var range: HistoryRange = .week

    public init(viewModel: AppViewModel, onEditMeal: @escaping (MealLogEntry) -> Void = { _ in }) {
        self.viewModel = viewModel
        self.onEditMeal = onEditMeal
    }

    private var chartData: HistoryChartData {
        viewModel.historyChartData(for: range)
    }

    public var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 12) {
                PageHeader(subtitle: chartData.subtitle, title: "History")
                    .padding(.top, 8)

                Card {
                    VStack(spacing: 10) {
                        segmentedControl
                        HistoryChartView(data: chartData)
                        legend
                    }
                }

                statsRow

                recentMealsSection
            }
            .padding(.horizontal, 18)
            .padding(.bottom, Theme.screenBottomInset)
        }
        .background(Theme.appBg)
    }

    // MARK: - Segmented control (`.seg`)

    private var segmentedControl: some View {
        HStack(spacing: 2) {
            segmentButton(.week, title: "Week")
            segmentButton(.month, title: "Month")
            segmentButton(.halfYear, title: "6 months")
        }
        .padding(2)
        .background(Theme.appBg)
        .clipShape(RoundedRectangle(cornerRadius: 10, style: .continuous))
    }

    private func segmentButton(_ candidate: HistoryRange, title: String) -> some View {
        let isSelected = range == candidate
        return Button {
            range = candidate
        } label: {
            Text(title)
                .font(.system(size: 13, weight: .semibold))
                .foregroundStyle(isSelected ? Theme.appInk : Theme.appInk2)
                .frame(maxWidth: .infinity)
                .padding(.vertical, 6)
                .background(isSelected ? Theme.appCard : Color.clear)
                .clipShape(RoundedRectangle(cornerRadius: 8, style: .continuous))
                .cardElevation(enabled: isSelected)
        }
        .buttonStyle(.plain)
        .accessibilityAddTraits(isSelected ? [.isSelected] : [])
    }

    // MARK: - Legend (`.legend`)

    private var legend: some View {
        HStack(spacing: 14) {
            legendItem(color: Theme.basil, text: "on target")
            legendItem(color: Theme.fat, text: ">5% over")
            legendItem(color: Theme.danger, text: ">25% over")
        }
        .padding(.top, 10)
        .overlay(alignment: .top) {
            Rectangle().fill(Theme.appLine).frame(height: 1)
        }
    }

    private func legendItem(color: Color, text: String) -> some View {
        HStack(spacing: 5) {
            RoundedRectangle(cornerRadius: 3, style: .continuous)
                .fill(color)
                .frame(width: 9, height: 9)
            Text(text)
                .font(.system(size: 11, weight: .semibold))
                .foregroundStyle(Theme.appInk2)
        }
    }

    // MARK: - Stats row (`.statrow`/`.stat`)

    private var statsRow: some View {
        HStack(spacing: 10) {
            statTile(
                value: chartData.averageCaloriesPerLoggedDay.map { AppViewModel.formattedKcal($0) } ?? "—",
                label: "avg kcal / day"
            )
            statTile(
                value: "\(chartData.daysLogged) of \(chartData.daysInRange)",
                label: "days logged"
            )
            statTile(
                value: chartData.averageProteinPerLoggedDay.map { "\(AppViewModel.roundedGrams($0)) g" } ?? "—",
                label: "avg protein"
            )
        }
    }

    private func statTile(value: String, label: String) -> some View {
        VStack(alignment: .leading, spacing: 2) {
            Text(value)
                .font(.system(size: 18, weight: .heavy))
                .tracking(-0.18)
                .monospacedDigit()
                .foregroundStyle(Theme.appInk)
            Text(label.uppercased())
                .font(.system(size: 10.5, weight: .bold))
                .tracking(0.53)
                .foregroundStyle(Theme.appInk2)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(.vertical, 10)
        .padding(.horizontal, 12)
        .background(Theme.appBg)
        .clipShape(RoundedRectangle(cornerRadius: 14, style: .continuous))
    }

    // MARK: - Recent meals (`.mealcard`, spec §6: "same row component as Today")

    @ViewBuilder
    private var recentMealsSection: some View {
        let entries = viewModel.recentMealEntries
        if !entries.isEmpty {
            SectionLabel("Recent meals")
            Card(padding: 2) {
                VStack(spacing: 0) {
                    ForEach(Array(entries.enumerated()), id: \.element.id) { index, entry in
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

private extension View {
    /// Only the selected `.seg` pill gets `--app-elev` (`.seg button.on{box-
    /// shadow:var(--app-elev)}`) — the unselected ones stay flat.
    @ViewBuilder
    func cardElevation(enabled: Bool) -> some View {
        if enabled {
            self.cardElevation()
        } else {
            self
        }
    }
}

// MARK: - Chart (`.chart`/`.chart.dense`, `.cbar`, `.goal-line`)

/// Renders one `HistoryChartData`: a fixed-height bar area (with the dashed
/// goal line drawn over it) followed by a SEPARATE axis-zone row of labels —
/// never overlaid on the bars, matching the prototype's reserved
/// `margin-bottom:26px` label zone, just via layout instead of absolute
/// positioning.
struct HistoryChartView: View {
    let data: HistoryChartData

    /// `.chart{gap:10px}` (week) vs `.chart.dense{gap:3px}` (month/6-months).
    private var barSpacing: CGFloat {
        data.range == .week ? 10 : 3
    }

    private var isDense: Bool {
        data.range != .week
    }

    var body: some View {
        VStack(spacing: 0) {
            ZStack(alignment: .top) {
                HStack(alignment: .bottom, spacing: barSpacing) {
                    ForEach(Array(data.bars.enumerated()), id: \.offset) { _, bar in
                        BarColumn(bar: bar, axisMax: data.axisMax, isDense: isDense)
                    }
                }
                GoalLineOverlay(goalCalories: data.goalCalories, axisMax: data.axisMax)
            }
            .frame(height: 150)
            .padding(.top, 14)

            // Axis-zone labels: a row of the same width/spacing as the bars,
            // living entirely below the 150pt bar area.
            HStack(alignment: .top, spacing: barSpacing) {
                ForEach(Array(data.bars.enumerated()), id: \.offset) { _, bar in
                    Text(bar.label ?? "")
                        .font(.system(size: 10.5, weight: .semibold))
                        .foregroundStyle(Theme.appInk2)
                        .lineLimit(1)
                        .minimumScaleFactor(0.8)
                        .frame(maxWidth: .infinity)
                }
            }
            .padding(.top, 7)
        }
    }
}

/// One bar: `.cbar i` (colored fill, rounded top, min-height so a zero/
/// unlogged value still shows a sliver) with the `.cbar.today` 2pt ink
/// outline when this is the rightmost/most-recent bar.
private struct BarColumn: View {
    let bar: HistoryBar
    let axisMax: Double
    let isDense: Bool

    var body: some View {
        GeometryReader { geometry in
            RoundedRectangle(cornerRadius: isDense ? 2 : 6, style: .continuous)
                .fill(fillColor)
                .frame(width: isDense ? nil : min(26, geometry.size.width), height: barHeight(in: geometry.size.height))
                .frame(maxWidth: .infinity, alignment: .bottom)
                .overlay(alignment: .bottom) {
                    if bar.isToday {
                        RoundedRectangle(cornerRadius: isDense ? 2 : 6, style: .continuous)
                            .stroke(Theme.appInk, lineWidth: 2)
                            .frame(width: isDense ? nil : min(26, geometry.size.width), height: barHeight(in: geometry.size.height))
                    }
                }
        }
    }

    private func barHeight(in totalHeight: CGFloat) -> CGFloat {
        guard axisMax > 0 else { return 3 }
        guard bar.value > 0 else { return 3 }
        return max(4, CGFloat(bar.value / axisMax) * totalHeight)
    }

    private var fillColor: Color {
        switch bar.colorClass {
        case .ok: return Theme.basil
        case .warn: return Theme.fat
        case .over: return Theme.danger
        case .none: return Theme.appLine
        }
    }
}

/// `.goal-line` — a dashed horizontal rule at the goal's height, with a
/// card-background chip label ("goal 2,200") that the bars never draw over
/// (it sits in its own overlay above them, at the far edge like the
/// prototype's `right:0`).
private struct GoalLineOverlay: View {
    let goalCalories: Double
    let axisMax: Double

    var body: some View {
        GeometryReader { geometry in
            let fraction = axisMax > 0 ? min(1, max(0, goalCalories / axisMax)) : 0
            let y = geometry.size.height * (1 - fraction)
            ZStack(alignment: .topTrailing) {
                Path { path in
                    path.move(to: CGPoint(x: 0, y: y))
                    path.addLine(to: CGPoint(x: geometry.size.width, y: y))
                }
                .stroke(style: StrokeStyle(lineWidth: 1.5, dash: [4, 3]))
                .foregroundStyle(Theme.appInk2.opacity(0.55))

                Text("goal \(AppViewModel.formattedKcal(goalCalories))")
                    .font(.system(size: 10, weight: .bold))
                    .monospacedDigit()
                    .foregroundStyle(Theme.appInk2)
                    .padding(.horizontal, 6)
                    .padding(.vertical, 1)
                    .background(Theme.appCard)
                    .clipShape(RoundedRectangle(cornerRadius: 6, style: .continuous))
                    .offset(y: max(0, y - 9))
            }
        }
        .allowsHitTesting(false)
    }
}
