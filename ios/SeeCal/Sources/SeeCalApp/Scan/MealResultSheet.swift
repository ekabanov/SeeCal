import SwiftUI
import SeeCalDomain

/// `.sheet` — the bottom result sheet, serving BOTH modes off one
/// `MealEditDraft` (spec §5): a fresh scan's review-before-log, and editing an
/// already-logged meal (tapped from a meal row). Layout, copy, and hierarchy
/// follow the prototype exactly: grab handle, `.result-head` (photo thumb,
/// name, depth meta — only when volume metadata exists), `.kcal-hero`,
/// `.macrochips`, "DETECTED ITEMS — ADJUST IF NEEDED" rows with ±5 g
/// steppers (min 5 g, live linear rescale),
/// `.edited-note` footnote, and the `.sheet-actions` pair — [Discard]/[Log
/// meal] for a new scan, [Cancel]/[Save changes] in edit mode.
struct MealResultSheet: View {
    @State private var draft: MealEditDraft
    /// Flips on the moment the primary action fires, so a second rapid tap
    /// can't commit the same draft twice while the first is persisting.
    @State private var isCommitting = false

    /// Log meal / Save changes with the (possibly gram-adjusted) final draft.
    private let onPrimary: (MealEditDraft) -> Void
    /// Discard (new scan) / Cancel (edit).
    private let onSecondary: () -> Void
    /// Fires on every draft edit (gram steppers), so the controller's
    /// presented draft tracks the sheet's local copy — an interactive
    /// dismissal then parks the ADJUSTED draft, not the original.
    private let onDraftChanged: (MealEditDraft) -> Void

    init(
        draft: MealEditDraft,
        onPrimary: @escaping (MealEditDraft) -> Void,
        onSecondary: @escaping () -> Void,
        onDraftChanged: @escaping (MealEditDraft) -> Void = { _ in }
    ) {
        _draft = State(initialValue: draft)
        self.onPrimary = onPrimary
        self.onSecondary = onSecondary
        self.onDraftChanged = onDraftChanged
    }

    var body: some View {
        VStack(spacing: 0) {
            grabHandle

            ScrollView {
                VStack(alignment: .leading, spacing: 0) {
                    resultHead
                    kcalHero
                    macroChips
                    itemsLabel
                    itemRows
                    footnote
                }
                .padding(.horizontal, 20)
                .padding(.bottom, 18)
            }

            actions
        }
        .background(Theme.appCard.ignoresSafeArea())
        .onChange(of: draft) { newValue in
            onDraftChanged(newValue)
        }
    }

    // MARK: - Grab handle (`.grab`)

    private var grabHandle: some View {
        RoundedRectangle(cornerRadius: 3)
            .fill(Theme.appLine)
            .frame(width: 38, height: 5)
            .padding(.top, 10)
            .padding(.bottom, 8)
    }

    // MARK: - Header (`.result-head`)

    private var resultHead: some View {
        HStack(spacing: 14) {
            thumbnail

            VStack(alignment: .leading, spacing: 3) {
                Text(draft.name)
                    .font(.system(size: 19, weight: .bold))
                    .tracking(-0.19)
                    .foregroundStyle(Theme.appInk)
                    .lineLimit(2)

                // Depth meta row: rendered ONLY when volume metadata exists —
                // data-driven and dormant until D5 populates volumeMl.
                if let volumeMl = draft.volumeMl {
                    HStack(spacing: 6) {
                        HStack(spacing: 4) {
                            Image(systemName: "star.fill")
                                .font(.system(size: 9, weight: .bold))
                            Text("depth-assisted")
                                .font(.system(size: 11, weight: .bold))
                        }
                        .foregroundStyle(Theme.basil)
                        .padding(.vertical, 3)
                        .padding(.horizontal, 8)
                        .background(Theme.basilSoft)
                        .clipShape(Capsule())

                        if let maxHeightMm = draft.maxHeightMm {
                            Text("~\(Int(volumeMl.rounded())) ml · max height \(Int(maxHeightMm.rounded())) mm")
                                .font(.system(size: 12))
                                .foregroundStyle(Theme.appInk2)
                                .monospacedDigit()
                        }
                    }
                }
            }

            Spacer(minLength: 0)
        }
        .padding(.top, 8)
        .padding(.bottom, 4)
    }

    private var thumbnail: some View {
        ZStack {
            Color(hex: 0x262420) // .result-head .thumb background
            if let path = draft.imagePath, let image = PlatformImageLoader.image(atPath: path) {
                image
                    .resizable()
                    .aspectRatio(contentMode: .fill)
            } else {
                Image(systemName: "photo")
                    .font(.system(size: 20, weight: .regular))
                    .foregroundStyle(.white.opacity(0.5))
            }
        }
        .frame(width: 64, height: 64)
        .clipShape(RoundedRectangle(cornerRadius: 16, style: .continuous))
    }

    // MARK: - Kcal hero (`.kcal-hero`)

    private var kcalHero: some View {
        let totals = draft.totals
        return HStack(alignment: .firstTextBaseline, spacing: 8) {
            Text(AppViewModel.formattedKcal(totals.calories))
                .font(.system(size: 46, weight: .heavy))
                .tracking(-1.38)
                .monospacedDigit()
                .foregroundStyle(Theme.appInk)
            Text("kcal")
                .font(.system(size: 15, weight: .bold))
                .foregroundStyle(Theme.appInk2)
            Spacer(minLength: 0)
        }
        .padding(.top, 14)
        .padding(.bottom, 2)
        .accessibilityElement(children: .combine)
    }

    // MARK: - Macro chips (`.macrochips` / `.mc`)

    private var macroChips: some View {
        let totals = draft.totals
        return HStack(spacing: 8) {
            macroChip(value: totals.proteinGrams, label: "Protein", color: Theme.protein)
            macroChip(value: totals.fatGrams, label: "Fat", color: Theme.fat)
            macroChip(value: totals.carbsGrams, label: "Carbs", color: Theme.carbs)
        }
        .padding(.top, 6)
        .padding(.bottom, 14)
    }

    private func macroChip(value: Double, label: String, color: Color) -> some View {
        VStack(alignment: .leading, spacing: 1) {
            // `tot.p.toFixed(1)+" g"` — one decimal.
            Text(String(format: "%.1f g", value))
                .font(.system(size: 17, weight: .bold))
                .monospacedDigit()
                .foregroundStyle(Theme.appInk)
            Text(label.uppercased())
                .font(.system(size: 11, weight: .bold))
                .tracking(0.55)
                .foregroundStyle(color)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(.vertical, 10)
        .padding(.horizontal, 12)
        .background(Theme.appBg)
        .clipShape(RoundedRectangle(cornerRadius: 14, style: .continuous))
        .accessibilityElement(children: .combine)
    }

    // MARK: - Items (`.items-label` + `.item` rows)

    private var itemsLabel: some View {
        Text("Detected items — adjust if needed".uppercased())
            .font(.system(size: 12.5, weight: .bold))
            .tracking(0.75)
            .foregroundStyle(Theme.appInk2)
            .padding(.vertical, 6)
    }

    private var itemRows: some View {
        VStack(spacing: 0) {
            ForEach(Array(draft.items.enumerated()), id: \.element.id) { index, item in
                if index > 0 {
                    Divider().overlay(Theme.appLine)
                }
                itemRow(item)
            }
        }
    }

    private func itemRow(_ item: MealItem) -> some View {
        HStack(spacing: 10) {
            VStack(alignment: .leading, spacing: 1) {
                Text(item.name.capitalized) // .item .nm .t{text-transform:capitalize}
                    .font(.system(size: 15, weight: .semibold))
                    .foregroundStyle(Theme.appInk)
                    .lineLimit(1)
                // "44p · 5f · 57c"
                Text("\(Int(item.protein.rounded()))p · \(Int(item.fat.rounded()))f · \(Int(item.carbs.rounded()))c")
                    .font(.system(size: 12))
                    .foregroundStyle(Theme.appInk2)
                    .monospacedDigit()
            }
            .frame(maxWidth: .infinity, alignment: .leading)

            GramStepper(
                valueText: String(Int(item.grams.rounded())),
                unitText: "g",
                decrementLabel: "Less \(item.name)",
                incrementLabel: "More \(item.name)",
                onDecrement: { draft.stepGrams(itemID: item.id, by: -MealItem.gramStep) },
                onIncrement: { draft.stepGrams(itemID: item.id, by: MealItem.gramStep) }
            )

            // `.ik` — item kcal, small unit stacked beneath.
            VStack(alignment: .trailing, spacing: 0) {
                Text("\(Int(item.kcal.rounded()))")
                    .font(.system(size: 14, weight: .bold))
                    .monospacedDigit()
                    .foregroundStyle(Theme.appInk)
                Text("kcal")
                    .font(.system(size: 10.5, weight: .semibold))
                    .foregroundStyle(Theme.appInk2)
            }
            .frame(width: 62, alignment: .trailing)
        }
        .padding(.vertical, 11)
    }

    // MARK: - Footnote (`.edited-note`, copy verbatim)

    private var footnote: some View {
        Text("Estimates come from the on-device model. Adjusting grams rescales an item's nutrition proportionally.")
            .font(.system(size: 11.5))
            .lineSpacing(3)
            .foregroundStyle(Theme.appInk2)
            .padding(.top, 8)
            .fixedSize(horizontal: false, vertical: true)
    }

    // MARK: - Actions (`.sheet-actions`)

    private var actions: some View {
        HStack(spacing: 10) {
            sheetButton(
                draft.isEditingExisting ? "Cancel" : "Discard",
                background: Theme.appBg,
                foreground: Theme.appInk,
                disabled: isCommitting
            ) {
                onSecondary()
            }
            sheetButton(
                draft.isEditingExisting ? "Save changes" : "Log meal",
                background: Theme.basil,
                foreground: .white,
                disabled: isCommitting
            ) {
                guard !isCommitting else { return }
                isCommitting = true
                onPrimary(draft)
            }
        }
        .padding(.top, 14)
        .padding(.horizontal, 20)
        .padding(.bottom, 26)
        .overlay(alignment: .top) {
            Rectangle().fill(Theme.appLine).frame(height: 1)
        }
    }

    private func sheetButton(
        _ title: String,
        background: Color,
        foreground: Color,
        disabled: Bool = false,
        action: @escaping () -> Void
    ) -> some View {
        Button(action: action) {
            Text(title)
                .font(.system(size: 16, weight: .bold))
                .foregroundStyle(foreground)
                .frame(maxWidth: .infinity)
                .padding(.vertical, 14)
                .background(background)
                .clipShape(RoundedRectangle(cornerRadius: 14, style: .continuous))
        }
        .buttonStyle(.plain)
        .disabled(disabled)
        .opacity(disabled ? 0.55 : 1)
    }
}
