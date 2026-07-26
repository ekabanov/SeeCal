import SwiftUI

/// Spec §6 placeholder. The real week/month/6-months bar chart + stats row +
/// recent-meals list is P7's job; for P3 this just needs to be a reachable scaffold
/// screen that keeps the existing 7-day progress data visible (parked here since it
/// is the closest existing functionality to "History").
public struct HistoryScreen: View {
    @ObservedObject private var viewModel: AppViewModel

    public init(viewModel: AppViewModel) {
        self.viewModel = viewModel
    }

    public var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 12) {
                PageHeader(subtitle: "Last 7 days", title: "History")
                    .padding(.top, 8)

                SectionLabel("7-day progress")

                Card {
                    VStack(spacing: 10) {
                        if viewModel.weeklyProgress.isEmpty {
                            Text("No meals logged yet")
                                .font(.system(size: 13.5))
                                .foregroundStyle(Theme.appInk2)
                        } else {
                            ForEach(viewModel.weeklyProgress, id: \.dayStart) { point in
                                HStack {
                                    Text(point.dayStart, style: .date)
                                        .font(.system(size: 13.5, weight: .semibold))
                                        .foregroundStyle(Theme.appInk)
                                    Spacer()
                                    Text("\(Int(point.calories)) kcal")
                                        .font(.system(size: 13.5))
                                        .monospacedDigit()
                                        .foregroundStyle(Theme.appInk2)
                                }
                                if point.dayStart != viewModel.weeklyProgress.last?.dayStart {
                                    Divider().overlay(Theme.appLine)
                                }
                            }
                        }
                    }
                }

                Text("Week/month/6-month bar chart lands in P7 (spec §6).")
                    .font(.system(size: 12, weight: .medium))
                    .foregroundStyle(Theme.appInk2)
            }
            .padding(.horizontal, 18)
            .padding(.bottom, Theme.screenBottomInset)
        }
        .background(Theme.appBg)
    }
}
