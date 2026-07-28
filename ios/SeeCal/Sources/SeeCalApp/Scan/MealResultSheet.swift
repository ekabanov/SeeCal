import SwiftUI
import SeeCalDomain

/// Shared fresh-result and logged-meal editor. The summary and focused
/// ingredient editor are two panels inside this one sheet.
struct MealResultSheet: View {
    @State private var draft: MealEditDraft
    @State private var editor: IngredientEditorDraft?
    @State private var deletedItem: DeletedItem?
    @State private var undoTask: Task<Void, Never>?
    @State private var isCommitting = false
    @State private var confirmsMealDeletion = false

    private let onPrimary: (MealEditDraft) -> Void
    private let onSecondary: () -> Void
    private let onDeleteMeal: (MealEditDraft) -> Void
    private let onDraftChanged: (MealEditDraft) -> Void

    init(
        draft: MealEditDraft,
        onPrimary: @escaping (MealEditDraft) -> Void,
        onSecondary: @escaping () -> Void,
        onDeleteMeal: @escaping (MealEditDraft) -> Void = { _ in },
        onDraftChanged: @escaping (MealEditDraft) -> Void = { _ in }
    ) {
        _draft = State(initialValue: draft)
        _editor = State(initialValue: draft.origin == .manual && draft.items.isEmpty ? IngredientEditorDraft() : nil)
        self.onPrimary = onPrimary
        self.onSecondary = onSecondary
        self.onDeleteMeal = onDeleteMeal
        self.onDraftChanged = onDraftChanged
    }

    var body: some View {
        VStack(spacing: 0) {
            grabHandle

            if editor != nil {
                ingredientEditor
            } else {
                summary
            }
        }
        .background(Theme.appCard.ignoresSafeArea())
        .onChange(of: draft) { _, newValue in
            onDraftChanged(newValue)
        }
        .onDisappear {
            undoTask?.cancel()
        }
        .alert("Delete this meal?", isPresented: $confirmsMealDeletion) {
            Button("Cancel", role: .cancel) {}
            Button("Delete", role: .destructive) {
                onDeleteMeal(draft)
            }
        } message: {
            Text(draft.imagePath == nil
                ? "This removes the logged meal. This action can’t be undone."
                : "This removes the logged meal and its photo. This action can’t be undone.")
        }
    }

    private var grabHandle: some View {
        RoundedRectangle(cornerRadius: 3)
            .fill(Theme.appLine)
            .frame(width: 38, height: 5)
            .padding(.top, 10)
            .padding(.bottom, 8)
    }

    // MARK: - Summary

    private var summary: some View {
        VStack(spacing: 0) {
            ScrollView {
                VStack(alignment: .leading, spacing: 0) {
                    resultHead
                    kcalHero
                    macroChips
                    itemsLabel
                    itemRows
                    addIngredientButton
                    footnote

                    if draft.isEditingExisting {
                        deleteMealButton
                    }
                }
                .padding(.horizontal, 20)
                .padding(.bottom, 18)
            }

            if let deletedItem {
                undoBar(deletedItem)
            }
            summaryActions
        }
    }

    private var resultHead: some View {
        HStack(spacing: 14) {
            thumbnail

            VStack(alignment: .leading, spacing: 4) {
                HStack(spacing: 6) {
                    TextField("Meal name", text: $draft.name, axis: .vertical)
                        .font(.system(size: 19, weight: .bold))
                        .foregroundStyle(Theme.appInk)
                        .lineLimit(1...2)
                        .accessibilityLabel("Meal name")

                    Image(systemName: "pencil")
                        .font(.system(size: 12, weight: .semibold))
                        .foregroundStyle(Theme.appInk2)
                        .accessibilityHidden(true)
                }

                if let source = draft.barcodeSource {
                    HStack(spacing: 6) {
                        HStack(spacing: 4) {
                            Image(systemName: "barcode")
                                .font(.system(size: 10, weight: .bold))
                            Text("package label")
                                .font(.system(size: 11, weight: .bold))
                        }
                        .foregroundStyle(Theme.basil)
                        .padding(.vertical, 3)
                        .padding(.horizontal, 8)
                        .background(Theme.basilSoft)
                        .clipShape(Capsule())

                        Text("\(source.provider) · \(source.barcode)")
                            .font(.system(size: 12))
                            .foregroundStyle(Theme.appInk2)
                            .lineLimit(1)
                    }
                } else if let volumeMl = draft.volumeMl {
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
                                .lineLimit(1)
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
            draft.origin == .photo ? Color(hex: 0x262420) : Theme.basilSoft
            if let path = draft.imagePath, let image = PlatformImageLoader.image(atPath: path) {
                image
                    .resizable()
                    .aspectRatio(contentMode: .fill)
            } else {
                Image(systemName: draft.origin == .barcode ? "barcode" : "fork.knife")
                    .font(.system(size: 20))
                    .foregroundStyle(draft.origin == .photo ? .white.opacity(0.5) : Theme.basil)
            }
        }
        .frame(width: 64, height: 64)
        .clipShape(RoundedRectangle(cornerRadius: 16, style: .continuous))
    }

    private var kcalHero: some View {
        HStack(alignment: .firstTextBaseline, spacing: 8) {
            Text(AppViewModel.formattedKcal(draft.totals.calories))
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

    private var macroChips: some View {
        HStack(spacing: 8) {
            macroChip(value: draft.totals.proteinGrams, label: "Protein", color: Theme.protein)
            macroChip(value: draft.totals.fatGrams, label: "Fat", color: Theme.fat)
            macroChip(value: draft.totals.carbsGrams, label: "Carbs", color: Theme.carbs)
        }
        .padding(.top, 6)
        .padding(.bottom, 14)
    }

    private func macroChip(value: Double, label: String, color: Color) -> some View {
        VStack(alignment: .leading, spacing: 1) {
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

    private var itemsLabel: some View {
        Text((draft.isEditingExisting || draft.origin != .photo
            ? "Ingredients — tap to edit"
            : "Detected items — tap to edit").uppercased())
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
                Button {
                    editor = IngredientEditorDraft(item: item)
                } label: {
                    itemRow(item)
                }
                .buttonStyle(.plain)
                .accessibilityLabel("Edit \(item.name)")
            }
        }
    }

    private func itemRow(_ item: MealItem) -> some View {
        HStack(spacing: 10) {
            VStack(alignment: .leading, spacing: 2) {
                HStack(spacing: 6) {
                    Text(item.name.capitalized)
                        .font(.system(size: 15, weight: .semibold))
                        .foregroundStyle(Theme.appInk)
                        .lineLimit(1)
                    if item.sourceReference?.source == .barcode {
                        statusChip("Label")
                    } else if item.isManual {
                        statusChip("Added")
                    } else if item.isEdited {
                        statusChip("Edited")
                    }
                }
                Text("\(Int(item.grams.rounded())) \(item.amountUnit.symbol) · \(item.protein.formattedOneDecimal)p · \(item.fat.formattedOneDecimal)f · \(item.carbs.formattedOneDecimal)c")
                    .font(.system(size: 12))
                    .foregroundStyle(Theme.appInk2)
                    .monospacedDigit()
                    .lineLimit(1)
            }
            .frame(maxWidth: .infinity, alignment: .leading)

            VStack(alignment: .trailing, spacing: 0) {
                Text("\(Int(item.kcal.rounded()))")
                    .font(.system(size: 14, weight: .bold))
                    .monospacedDigit()
                    .foregroundStyle(Theme.appInk)
                Text("kcal")
                    .font(.system(size: 10.5, weight: .semibold))
                    .foregroundStyle(Theme.appInk2)
            }

            Image(systemName: "chevron.right")
                .font(.system(size: 12, weight: .semibold))
                .foregroundStyle(Theme.appInk2.opacity(0.7))
        }
        .padding(.vertical, 11)
        .contentShape(Rectangle())
    }

    private func statusChip(_ title: String) -> some View {
        Text(title.uppercased())
            .font(.system(size: 9, weight: .heavy))
            .tracking(0.45)
            .foregroundStyle(Theme.basil)
            .padding(.horizontal, 6)
            .padding(.vertical, 2)
            .background(Theme.basilSoft)
            .clipShape(Capsule())
    }

    private var addIngredientButton: some View {
        Button {
            editor = IngredientEditorDraft()
        } label: {
            Label("Add ingredient", systemImage: "plus")
                .font(.system(size: 14, weight: .bold))
                .foregroundStyle(Theme.basil)
                .frame(maxWidth: .infinity)
                .padding(.vertical, 12)
                .overlay {
                    RoundedRectangle(cornerRadius: 12, style: .continuous)
                        .stroke(Theme.appInk2.opacity(0.55), style: StrokeStyle(lineWidth: 1, dash: [4]))
                }
        }
        .buttonStyle(.plain)
        .padding(.top, 8)
    }

    private var footnote: some View {
        Text("Corrections update an item’s nutrition density, so later amount changes scale them. Meal totals always equal the sum of its ingredients.")
            .font(.system(size: 11.5))
            .lineSpacing(3)
            .foregroundStyle(Theme.appInk2)
            .padding(.top, 10)
            .fixedSize(horizontal: false, vertical: true)
    }

    private var deleteMealButton: some View {
        Button(role: .destructive) {
            confirmsMealDeletion = true
        } label: {
            Text("Delete meal")
                .font(.system(size: 15, weight: .bold))
                .foregroundStyle(Theme.danger)
                .frame(maxWidth: .infinity)
                .padding(.vertical, 13)
                .overlay {
                    RoundedRectangle(cornerRadius: 13, style: .continuous)
                        .stroke(Theme.danger, lineWidth: 1.5)
                }
        }
        .buttonStyle(.plain)
        .disabled(isCommitting)
        .padding(.top, 16)
    }

    private func undoBar(_ deleted: DeletedItem) -> some View {
        HStack(spacing: 12) {
            Text("\(deleted.item.name) deleted")
                .font(.system(size: 13, weight: .semibold))
                .lineLimit(1)
            Spacer(minLength: 4)
            Button("Undo") {
                undoTask?.cancel()
                draft.restoreItem(deleted.item, at: deleted.index)
                deletedItem = nil
            }
            .font(.system(size: 13, weight: .bold))
            .foregroundStyle(Color(light: 0x1F7A52, dark: 0x7BE3AD))
        }
        .foregroundStyle(.white)
        .padding(.vertical, 11)
        .padding(.horizontal, 16)
        .background(Color(hex: 0x262A27))
        .clipShape(RoundedRectangle(cornerRadius: 12, style: .continuous))
        .padding(.horizontal, 20)
        .padding(.bottom, 8)
    }

    private var summaryActions: some View {
        HStack(spacing: 10) {
            sheetButton(
                draft.isEditingExisting ? "Cancel" : "Discard",
                background: Theme.appBg,
                foreground: Theme.appInk,
                disabled: isCommitting,
                action: onSecondary
            )
            sheetButton(
                draft.isEditingExisting ? "Save changes" : "Log meal",
                background: Theme.basil,
                foreground: .white,
                disabled: isCommitting || !draft.isValid
            ) {
                guard !isCommitting, draft.isValid else { return }
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

    // MARK: - Focused ingredient editor

    private var ingredientEditor: some View {
        let binding = Binding<IngredientEditorDraft>(
            get: { editor ?? IngredientEditorDraft() },
            set: { editor = $0 }
        )

        return VStack(spacing: 0) {
            IngredientEditorPanel(
                editor: binding,
                onCancel: { editor = nil },
                onDone: saveIngredient,
                onDelete: deleteIngredient
            )
        }
    }

    private func saveIngredient() {
        guard let editor, let item = editor.makeItem() else { return }
        if editor.originalItem == nil {
            draft.addItem(item)
        } else {
            draft.replaceItem(item)
        }
        self.editor = nil
    }

    private func deleteIngredient() {
        guard let itemID = editor?.originalItem?.id,
              let removed = draft.removeItem(id: itemID)
        else { return }

        editor = nil
        undoTask?.cancel()
        deletedItem = DeletedItem(item: removed.item, index: removed.index)
        undoTask = Task {
            try? await Task.sleep(for: .seconds(5))
            guard !Task.isCancelled else { return }
            deletedItem = nil
        }
    }

    private func sheetButton(
        _ title: String,
        background: Color,
        foreground: Color,
        disabled: Bool,
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

private struct DeletedItem {
    let item: MealItem
    let index: Int
}

// MARK: - Ingredient editor panel

private struct IngredientEditorPanel: View {
    @Binding var editor: IngredientEditorDraft
    let onCancel: () -> Void
    let onDone: () -> Void
    let onDelete: () -> Void

    var body: some View {
        VStack(spacing: 0) {
            ScrollView {
                VStack(alignment: .leading, spacing: 16) {
                    header
                    nameField
                    numericField("Amount", unit: editor.amountUnit.symbol, text: gramsBinding, reset: .grams)
                    numericField("Calories", unit: "kcal", text: $editor.kcalText, reset: .kcal)
                    numericField("Protein", unit: "g", text: $editor.proteinText, reset: .protein)
                    numericField("Fat", unit: "g", text: $editor.fatText, reset: .fat)
                    numericField("Carbs", unit: "g", text: $editor.carbsText, reset: .carbs)

                    Text("Changing the amount rescales calories and macros from their current values. Calories and macros may be corrected independently.")
                        .font(.system(size: 11.5))
                        .lineSpacing(3)
                        .foregroundStyle(Theme.appInk2)
                        .fixedSize(horizontal: false, vertical: true)

                    if editor.hasSourceReference {
                        Button(editor.resetItemTitle) {
                            editor.resetWholeItem()
                        }
                        .font(.system(size: 14, weight: .bold))
                        .foregroundStyle(Theme.basil)
                        .frame(maxWidth: .infinity)
                        .padding(.vertical, 12)
                        .background(Theme.basilSoft)
                        .clipShape(RoundedRectangle(cornerRadius: 12, style: .continuous))
                    }

                    if editor.originalItem != nil {
                        Button("Delete ingredient", role: .destructive, action: onDelete)
                            .font(.system(size: 14, weight: .bold))
                            .foregroundStyle(Theme.danger)
                            .frame(maxWidth: .infinity)
                            .padding(.vertical, 12)
                            .overlay {
                                RoundedRectangle(cornerRadius: 12, style: .continuous)
                                    .stroke(Theme.danger, lineWidth: 1.25)
                            }
                    }
                }
                .padding(.horizontal, 20)
                .padding(.bottom, 18)
            }

            HStack(spacing: 10) {
                actionButton("Cancel", background: Theme.appBg, foreground: Theme.appInk, action: onCancel)
                actionButton(
                    "Done",
                    background: Theme.basil,
                    foreground: .white,
                    disabled: !editor.isValid,
                    action: onDone
                )
            }
            .padding(.top, 14)
            .padding(.horizontal, 20)
            .padding(.bottom, 26)
            .overlay(alignment: .top) {
                Rectangle().fill(Theme.appLine).frame(height: 1)
            }
        }
    }

    private var header: some View {
        HStack(alignment: .center, spacing: 10) {
            Button(action: onCancel) {
                Image(systemName: "chevron.left")
                    .font(.system(size: 15, weight: .bold))
                    .foregroundStyle(Theme.appInk)
                    .frame(width: 34, height: 34)
                    .background(Theme.appBg)
                    .clipShape(Circle())
            }
            .buttonStyle(.plain)
            .accessibilityLabel("Back to meal")

            VStack(alignment: .leading, spacing: 1) {
                Text(editor.originalItem == nil ? "Add ingredient" : "Edit ingredient")
                    .font(.system(size: 19, weight: .bold))
                    .foregroundStyle(Theme.appInk)
                Text(editor.sourceLabel)
                    .font(.system(size: 11, weight: .bold))
                    .foregroundStyle(Theme.basil)
                    .textCase(.uppercase)
                    .tracking(0.55)
            }
            Spacer()
        }
        .padding(.top, 6)
    }

    private var nameField: some View {
        VStack(alignment: .leading, spacing: 6) {
            fieldLabel("Name", reset: .name)
            TextField("Ingredient name", text: $editor.name)
                .font(.system(size: 16, weight: .semibold))
                .foregroundStyle(Theme.appInk)
                .padding(.horizontal, 13)
                .padding(.vertical, 12)
                .background(Theme.appBg)
                .clipShape(RoundedRectangle(cornerRadius: 12, style: .continuous))
        }
    }

    private var gramsBinding: Binding<String> {
        Binding(
            get: { editor.gramsText },
            set: { editor.updateGramsText($0) }
        )
    }

    private func numericField(
        _ title: String,
        unit: String,
        text: Binding<String>,
        reset: IngredientResetField
    ) -> some View {
        VStack(alignment: .leading, spacing: 6) {
            fieldLabel(title, reset: reset)
            HStack(spacing: 8) {
                TextField("0", text: text)
                    .font(.system(size: 17, weight: .bold))
                    .monospacedDigit()
                    .foregroundStyle(Theme.appInk)
                    .numericKeyboard()
                Text(unit)
                    .font(.system(size: 13, weight: .semibold))
                    .foregroundStyle(Theme.appInk2)
            }
            .padding(.horizontal, 13)
            .padding(.vertical, 12)
            .background(Theme.appBg)
            .clipShape(RoundedRectangle(cornerRadius: 12, style: .continuous))
        }
    }

    private func fieldLabel(_ title: String, reset: IngredientResetField) -> some View {
        HStack {
            Text(title.uppercased())
                .font(.system(size: 11.5, weight: .bold))
                .tracking(0.65)
                .foregroundStyle(Theme.appInk2)
            Spacer()
            if editor.hasSourceReference {
                Button("Reset") {
                    editor.reset(reset)
                }
                .font(.system(size: 11.5, weight: .bold))
                .foregroundStyle(Theme.basil)
            }
        }
    }

    private func actionButton(
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

private enum IngredientResetField {
    case name, grams, kcal, protein, fat, carbs
}

private struct IngredientEditorDraft {
    let originalItem: MealItem?
    var name: String
    var gramsText: String
    var kcalText: String
    var proteinText: String
    var fatText: String
    var carbsText: String
    private var lastValidGrams: Double?

    var amountUnit: MealAmountUnit {
        originalItem?.amountUnit ?? .grams
    }

    init(item: MealItem? = nil) {
        originalItem = item
        name = item?.name ?? ""
        gramsText = item.map { String(Int($0.grams.rounded())) } ?? ""
        kcalText = item.map { String(Int($0.kcal.rounded())) } ?? ""
        proteinText = item.map { $0.protein.formattedOneDecimal } ?? ""
        fatText = item.map { $0.fat.formattedOneDecimal } ?? ""
        carbsText = item.map { $0.carbs.formattedOneDecimal } ?? ""
        lastValidGrams = item?.grams
    }

    var hasSourceReference: Bool {
        originalItem?.sourceReference != nil
    }

    var sourceLabel: String {
        switch originalItem?.sourceReference?.source {
        case .barcode:
            return "Package label"
        case .model:
            return "Model estimate"
        case nil:
            return "Added manually"
        }
    }

    var resetItemTitle: String {
        originalItem?.sourceReference?.source == .barcode
            ? "Reset item to package label"
            : "Reset item to estimate"
    }

    var isValid: Bool {
        guard !name.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty,
              let grams = Self.number(gramsText),
              grams.rounded() >= 1
        else { return false }

        return [kcalText, proteinText, fatText, carbsText].allSatisfy {
            $0.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
                || (Self.number($0) ?? -1) >= 0
        }
    }

    mutating func updateGramsText(_ newText: String) {
        gramsText = newText
        guard let newGrams = Self.number(newText), newGrams > 0 else { return }
        if let oldGrams = lastValidGrams, oldGrams > 0, oldGrams != newGrams {
            let ratio = newGrams / oldGrams
            kcalText = Self.scaledText(kcalText, by: ratio, decimals: 0)
            proteinText = Self.scaledText(proteinText, by: ratio, decimals: 1)
            fatText = Self.scaledText(fatText, by: ratio, decimals: 1)
            carbsText = Self.scaledText(carbsText, by: ratio, decimals: 1)
        }
        lastValidGrams = newGrams
    }

    mutating func reset(_ field: IngredientResetField) {
        guard let estimate = originalItem?.sourceReference else { return }
        switch field {
        case .name:
            name = estimate.name
        case .grams:
            updateGramsText(String(Int(estimate.base.grams.rounded())))
        case .kcal:
            kcalText = modelValueText(estimate.base.kcal, baseGrams: estimate.base.grams, decimals: 0)
        case .protein:
            proteinText = modelValueText(estimate.base.protein, baseGrams: estimate.base.grams, decimals: 1)
        case .fat:
            fatText = modelValueText(estimate.base.fat, baseGrams: estimate.base.grams, decimals: 1)
        case .carbs:
            carbsText = modelValueText(estimate.base.carbs, baseGrams: estimate.base.grams, decimals: 1)
        }
    }

    mutating func resetWholeItem() {
        guard let estimate = originalItem?.sourceReference else { return }
        name = estimate.name
        gramsText = String(Int(estimate.base.grams.rounded()))
        kcalText = String(Int(estimate.base.kcal.rounded()))
        proteinText = estimate.base.protein.formattedOneDecimal
        fatText = estimate.base.fat.formattedOneDecimal
        carbsText = estimate.base.carbs.formattedOneDecimal
        lastValidGrams = estimate.base.grams
    }

    func makeItem() -> MealItem? {
        guard isValid, let parsedGrams = Self.number(gramsText) else { return nil }
        let grams = max(1, parsedGrams.rounded())
        let kcal = max(0, Self.number(kcalText) ?? 0).rounded()
        let protein = Self.roundOneDecimal(max(0, Self.number(proteinText) ?? 0))
        let fat = Self.roundOneDecimal(max(0, Self.number(fatText) ?? 0))
        let carbs = Self.roundOneDecimal(max(0, Self.number(carbsText) ?? 0))

        return MealItem(
            id: originalItem?.id ?? UUID(),
            name: name.trimmingCharacters(in: .whitespacesAndNewlines),
            grams: grams,
            amountUnit: amountUnit,
            base: MealItemBase(
                grams: grams,
                kcal: kcal,
                protein: protein,
                fat: fat,
                carbs: carbs
            ),
            sourceReference: originalItem?.sourceReference
        )
    }

    private func modelValueText(_ value: Double, baseGrams: Double, decimals: Int) -> String {
        let grams = Self.number(gramsText) ?? baseGrams
        let current = baseGrams == 0 ? 0 : value / baseGrams * grams
        return Self.text(current, decimals: decimals)
    }

    private static func scaledText(_ text: String, by ratio: Double, decimals: Int) -> String {
        guard let value = number(text) else { return text }
        return Self.text(value * ratio, decimals: decimals)
    }

    private static func text(_ value: Double, decimals: Int) -> String {
        String(format: "%.\(decimals)f", value)
    }

    private static func number(_ text: String) -> Double? {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return nil }

        let formatter = NumberFormatter()
        formatter.locale = .current
        formatter.numberStyle = .decimal
        if let number = formatter.number(from: trimmed) {
            return number.doubleValue
        }
        return Double(trimmed.replacingOccurrences(of: ",", with: "."))
    }

    private static func roundOneDecimal(_ value: Double) -> Double {
        (value * 10).rounded() / 10
    }
}

private extension Double {
    var formattedOneDecimal: String {
        String(format: "%.1f", self)
    }
}

private extension View {
    @ViewBuilder
    func numericKeyboard() -> some View {
        #if os(iOS)
        self.keyboardType(.decimalPad)
        #else
        self
        #endif
    }
}
