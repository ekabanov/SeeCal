import SwiftUI
import SeeCalDomain
import SeeCalPersistence

#if canImport(UIKit)
import UIKit
#elseif canImport(AppKit)
import AppKit
#endif

/// Spec §4 / `#scr-today`: ring card (consumed/goal kcal + three macro bars),
/// MEALS section, meal-row list, privacy chip. Matches the prototype's markup +
/// CSS (`.pagehead`, `.card`, `.ringrow`, `.ring`, `.macros`/`.macro`/`.bar`,
/// `.sectionlabel`, `.meal`, `.mealcard`, `.privchip`) — see
/// `docs/design/prototype/seecal-prototype.html` lines 582-616 and `renderToday()`
/// (~line 1139) for the exact text formats this file reproduces.
///
/// Deviation from the prototype: the pre-P4 "Add meal" card (meal-type picker +
/// photo library picker, temporarily parked here since P3) is removed — it has no
/// counterpart in `#scr-today`. Manually adding a meal photo is unavailable via UI
/// until P6 wires the real camera → analyzing → result flow onto the tab bar's
/// Scan FAB (`ScanStubScreen` is still just a stub); `AppViewModel.addMealPhoto`
/// itself is untouched and already covered by `AppViewModelTrackingTests`.
public struct TodayScreen: View {
    @ObservedObject private var viewModel: AppViewModel

    @State private var editingEntry: MealLogEntry?
    @State private var draft: MealEditDraft?

    public init(viewModel: AppViewModel) {
        self.viewModel = viewModel
    }

    public var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 12) {
                PageHeader(subtitle: AppViewModel.dateSubtitle(for: Date()), title: "Today")
                    .padding(.top, 8)

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
        .sheet(item: $editingEntry) { entry in
            if let draft {
                MealEditSheet(
                    draft: draft,
                    onCancel: {
                        editingEntry = nil
                        self.draft = nil
                    },
                    onSave: { savedDraft in
                        Task {
                            do {
                                let updatedEntry = try savedDraft.committedEntry()
                                await viewModel.updateMeal(updatedEntry)
                            } catch {
                                viewModel.lastError = error.localizedDescription
                            }
                            editingEntry = nil
                            self.draft = nil
                        }
                    }
                )
            }
        }
    }

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
                        mealRow(entry)
                    }
                }
            }
        }
    }

    /// One `.meal` row: 52pt rounded thumb, name, "HH:mm · N g protein", kcal
    /// (bold + small "kcal" unit), chevron. Tapping opens the existing edit-sheet
    /// wiring (P6 will replace `MealEditSheet` with the real result/edit sheet).
    private func mealRow(_ entry: MealLogEntry) -> some View {
        Button {
            editingEntry = entry
            draft = MealEditDraft(entry: entry)
        } label: {
            HStack(spacing: 12) {
                MealThumbnail(imagePath: entry.imagePath)

                VStack(alignment: .leading, spacing: 1) {
                    Text(entry.name)
                        .font(.system(size: 15, weight: .semibold))
                        .foregroundStyle(Theme.appInk)
                        .lineLimit(1)
                    Text("\(Self.timeString(entry.createdAt)) · \(AppViewModel.roundedGrams(entry.totals.proteinGrams)) g protein")
                        .font(.system(size: 12.5))
                        .foregroundStyle(Theme.appInk2)
                        .lineLimit(1)
                        .truncationMode(.tail)
                }

                Spacer(minLength: 8)

                (
                    Text(AppViewModel.formattedKcal(entry.totals.calories))
                        .font(.system(size: 15, weight: .bold))
                        .foregroundColor(Theme.appInk)
                    + Text(" kcal")
                        .font(.system(size: 11, weight: .semibold))
                        .foregroundColor(Theme.appInk2)
                )
                .monospacedDigit()

                Image(systemName: "chevron.right")
                    .font(.system(size: 12, weight: .semibold))
                    .foregroundStyle(Theme.appInk2.opacity(0.7))
            }
            .padding(.vertical, 12)
            .padding(.horizontal, 14)
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .accessibilityLabel("Edit \(entry.name)")
        .contextMenu {
            // `.swipeActions` only has an effect inside a `List`; this VStack-based
            // layout (matching the prototype's card-based meal rows, not a system
            // List) uses a context menu instead so deleting an entry stays reachable.
            Button("Delete", role: .destructive) {
                Task { await viewModel.deleteMeal(id: entry.id) }
            }
        }
    }

    /// "HH:mm", zero-padded 24-hour clock (matches the prototype's
    /// `String(now.getHours()).padStart(2,"0")+":"+...`).
    private static func timeString(_ date: Date) -> String {
        let formatter = DateFormatter()
        formatter.locale = Locale(identifier: "en_US_POSIX")
        formatter.dateFormat = "HH:mm"
        return formatter.string(from: date)
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

// MARK: - Meal thumbnail (`.meal .thumb`)

/// `.meal .thumb{width:52px;height:52px;border-radius:12px;background:var(--app-line)}`
/// — loads the entry's photo from disk (`imagePath`); falls back to a neutral
/// (`app-line`-colored) placeholder glyph when the file is missing or unreadable.
private struct MealThumbnail: View {
    let imagePath: String

    var body: some View {
        ZStack {
            Theme.appLine
            if let image = Self.loadImage(imagePath) {
                image
                    .resizable()
                    .aspectRatio(contentMode: .fill)
            } else {
                Image(systemName: "photo")
                    .font(.system(size: 18, weight: .regular))
                    .foregroundStyle(Theme.appInk2)
            }
        }
        .frame(width: 52, height: 52)
        .clipShape(RoundedRectangle(cornerRadius: 12, style: .continuous))
    }

    private static func loadImage(_ path: String) -> Image? {
        guard !path.isEmpty, FileManager.default.fileExists(atPath: path) else { return nil }
        #if canImport(UIKit)
        guard let uiImage = UIImage(contentsOfFile: path) else { return nil }
        return Image(uiImage: uiImage)
        #elseif canImport(AppKit)
        guard let nsImage = NSImage(contentsOfFile: path) else { return nil }
        return Image(nsImage: nsImage)
        #else
        return nil
        #endif
    }
}

/// Placeholder layout only — the real result/edit sheet (photo, hero total, macro
/// chips, per-item gram steppers per spec §5) lands in P6. This just exercises
/// `MealEditDraft`'s per-item API well enough to keep the Today tab compiling and
/// interactive.
private struct MealEditSheet: View {
    @State var draft: MealEditDraft
    let onCancel: () -> Void
    let onSave: (MealEditDraft) -> Void

    var body: some View {
        NavigationStack {
            Form {
                Section("Detected Items") {
                    ForEach(draft.items) { item in
                        VStack(alignment: .leading, spacing: 4) {
                            Text(item.name)
                            GramStepper(
                                valueText: String(Int(item.grams)),
                                unitText: "g",
                                decrementLabel: "Fewer grams",
                                incrementLabel: "More grams",
                                onDecrement: { draft.stepGrams(itemID: item.id, by: -MealItem.gramStep) },
                                onIncrement: { draft.stepGrams(itemID: item.id, by: MealItem.gramStep) }
                            )
                            Text("\(Int(item.kcal)) kcal")
                                .font(.caption)
                                .foregroundStyle(Theme.appInk2)
                        }
                    }
                }
                Section("Totals") {
                    let totals = draft.totals
                    Text("\(Int(totals.calories)) kcal")
                        .fontWeight(.semibold)
                    Text("P \(Int(totals.proteinGrams))g • F \(Int(totals.fatGrams))g • C \(Int(totals.carbsGrams))g")
                        .font(.caption)
                        .foregroundStyle(Theme.appInk2)
                }
            }
            .navigationTitle(draft.name)
            .toolbar {
                ToolbarItem(placement: .cancellationAction) {
                    Button("Cancel", action: onCancel)
                }
                ToolbarItem(placement: .confirmationAction) {
                    Button("Save") { onSave(draft) }
                }
            }
        }
    }
}
